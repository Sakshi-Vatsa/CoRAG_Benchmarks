# from src.agent import RagPath
import threading
from copy import deepcopy
from typing import List, Dict, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from agent_utils import RagPath
from generate_prompts import get_generate_subquery_prompt, \
    get_generate_final_answer_prompt, get_generate_intermediate_answer_prompt
from load_corpus import format_input_context
from search_utils import _normalize_subquery, search_by_http
from utils import batch_truncate
from vllm_client import VllmClient


class CoRagAgent:
    def __init__(self, vllm_client: VllmClient, corpus: Dataset):
        self.vllm_client = vllm_client
        self.corpus = corpus
        self.tokenizer: PreTrainedTokenizerFast = vllm_client.tokenizer
        self.lock = threading.Lock()

    def sample_path(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            **kwargs
    ) -> RagPath:
        past_subqueries: List[str] = kwargs.pop('past_subqueries', [])
        past_subanswers: List[str] = kwargs.pop('past_subanswers', [])
        past_doc_ids: List[List[str]] = kwargs.pop('past_doc_ids', [])
        assert len(past_subqueries) == len(past_subanswers) == len(past_doc_ids)

        subquery_temp: float = temperature
        num_llm_calls: int = 0
        max_num_llm_calls: int = 4 * (max_path_length - len(past_subqueries))
        while len(past_subqueries) < max_path_length and num_llm_calls < max_num_llm_calls:
            num_llm_calls += 1
            messages: List[Dict] = get_generate_subquery_prompt(
                query=query,
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                task_desc=task_desc,
            )
            self._truncate_long_messages(messages, max_length=max_message_length)

            subquery: str = self.vllm_client.call_chat(messages=messages, temperature=subquery_temp, **kwargs)
            subquery = _normalize_subquery(subquery)

            if subquery in past_subqueries:
                subquery_temp = max(subquery_temp, 0.7)
                continue

            subquery_temp = temperature
            subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                subquery=subquery, max_message_length=max_message_length
            )

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)
            past_doc_ids.append(doc_ids)

        return RagPath(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            past_doc_ids=past_doc_ids,
        )

    def generate_final_answer(
            self, corag_sample: RagPath, task_desc: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        return self.vllm_client.call_chat(messages=messages, **kwargs)

    def _truncate_long_messages(self, messages: List[Dict], max_length: int):
        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self.lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]

    def sample_subqueries(self, query: str, task_desc: str, n: int = 10, max_message_length: int = 4096, **kwargs) -> \
    List[str]:
        messages: List[Dict] = get_generate_subquery_prompt(
            query=query,
            past_subqueries=kwargs.pop('past_subqueries', []),
            past_subanswers=kwargs.pop('past_subanswers', []),
            task_desc=task_desc,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        # completion: ChatCompletion = self.vllm_client.call_chat(messages=messages, return_str=False, **kwargs)
        # subqueries: List[str] = [_normalize_subquery(c.message.content) for c in completion.choices]
        # subqueries = list(set(subqueries))[:n]

        completions: List[str] = self.vllm_client.call_chat(messages=messages, return_str=False, **kwargs)
        subqueries: List[str] = [_normalize_subquery(text) for text in completions]
        subqueries = list(set(subqueries))[:n]

        return subqueries

    def _get_subanswer_and_doc_ids(
            self, subquery: str, max_message_length: int = 4096
    ) -> Tuple[str, List]:
        retriever_results: List[Dict] = search_by_http(query=subquery)
        doc_ids: List[str] = [res['doc_id'] for res in retriever_results]
        documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]

        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subanswer: str = self.vllm_client.call_chat(messages=messages, temperature=0.7, max_new_tokens=128)
        return subanswer, doc_ids

    def tree_search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        return self._search(
            query=query, task_desc=task_desc,
            max_path_length=max_path_length,
            max_message_length=max_message_length,
            temperature=temperature,
            expand_size=expand_size, num_rollouts=num_rollouts, beam_size=beam_size,
            **kwargs
        )

    def best_of_n(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            **kwargs
    ) -> RagPath:
        sampled_paths: List[RagPath] = []

        for idx in range(n):
            is_greedy = idx == 0
            temp = 1e-5 if is_greedy else temperature  # very low but >0 for greedy
            do_sample = not is_greedy

            path: RagPath = self.sample_path(
                query=query, task_desc=task_desc,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=temp,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))]

    def _search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        candidates: List[RagPath] = [RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])]
        for step in range(max_path_length):
            new_candidates: List[RagPath] = []
            for candidate in candidates:
                new_subqueries: List[str] = self.sample_subqueries(
                    query=query, task_desc=task_desc,
                    past_subqueries=deepcopy(candidate.past_subqueries),
                    past_subanswers=deepcopy(candidate.past_subanswers),
                    n=expand_size, temperature=temperature, max_message_length=max_message_length
                )
                for subquery in new_subqueries:
                    new_candidate: RagPath = deepcopy(candidate)
                    new_candidate.past_subqueries.append(subquery)
                    subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                        subquery=subquery, max_message_length=max_message_length
                    )
                    new_candidate.past_subanswers.append(subanswer)
                    new_candidate.past_doc_ids.append(doc_ids)
                    new_candidates.append(new_candidate)

            if len(new_candidates) > beam_size:
                scores: List[float] = []
                for path in new_candidates:
                    score = self._eval_state_without_answer(
                        path, num_rollouts=num_rollouts,
                        task_desc=task_desc,
                        max_path_length=max_path_length,
                        temperature=temperature,
                        max_message_length=max_message_length
                    )
                    scores.append(score)

                # lower scores are better
                new_candidates = [c for c, s in sorted(zip(new_candidates, scores), key=lambda x: x[1])][:beam_size]

            candidates = new_candidates

        return candidates[0]

    def _eval_single_path(self, current_path: RagPath, max_message_length: int = 4096) -> float:
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=current_path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
        )
        messages.append({'role': 'assistant', 'content': 'No relevant information found'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        completions: List[str] = self.vllm_client.call_chat(
            messages=messages,
            return_str=True,  # Return raw string text
            max_new_tokens=16
        )

        response_text = completions[0].strip().lower()
        return 1.0 if response_text != "no relevant information found" else 0.0

        # response: ChatCompletion = self.vllm_client.call_chat(
        #     messages=messages,
        #     return_str=False,
        #     max_tokens=1,
        #     extra_body={
        #         "prompt_logprobs": 1
        #     }
        # )
        # answer_logprobs: List[float] = parse_answer_logprobs(response)

        # return sum(answer_logprobs) / len(answer_logprobs)

    def _eval_state_without_answer(
            self, path: RagPath, num_rollouts: int, task_desc: str,
            max_path_length: int = 3,
            temperature: float = 0.7,
            max_message_length: int = 4096
    ) -> float:
        assert len(path.past_subqueries) > 0
        if num_rollouts <= 0:
            return self._eval_single_path(path)

        rollout_paths: List[RagPath] = []
        for _ in range(num_rollouts):
            rollout_path: RagPath = self.sample_path(
                query=path.query, task_desc=task_desc,
                max_path_length=min(max_path_length, len(path.past_subqueries) + 2),  # rollout at most 2 steps
                temperature=temperature, max_message_length=max_message_length,
                past_subqueries=deepcopy(path.past_subqueries),
                past_subanswers=deepcopy(path.past_subanswers),
                past_doc_ids=deepcopy(path.past_doc_ids),
            )
            rollout_paths.append(rollout_path)

        scores: List[float] = [self._eval_single_path(p) for p in rollout_paths]
        # TODO: should we use the min score instead?
        return sum(scores) / len(scores)