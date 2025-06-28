import threading
import copy
import argparse
import os
import torch
from typing import Dict, List
from functools import partial

from transformers import PreTrainedTokenizerFast, AutoTokenizer

from agent_utils import RagPath
from corag_agent import CoRagAgent
from load_corpus import format_documents_for_final_answer, corpus
from load_dataset import total_cnt, ds
from logger_config import logger
from utils import AtomicCounter
from vllm_client import VllmClient


def _generate_single_example(ex: Dict, decode_strategy: str) -> Dict:
    processed_cnt: AtomicCounter = AtomicCounter()
    tokenizer_lock: threading.Lock = threading.Lock()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    vllm_client = VllmClient(model_name)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    corag_agent: CoRagAgent = CoRagAgent(vllm_client=vllm_client, corpus=corpus)
    max_path_length = 3

    if decode_strategy == 'greedy' or max_path_length < 1:
        path = corag_agent.sample_path(
            query=ex['query'], task_desc=ex['task_desc'],
            max_path_length=max_path_length,
            temperature=0.7, max_new_tokens=64
        )
        print(path)

    elif decode_strategy == 'tree_search':
        path: RagPath = corag_agent.tree_search(
            query=ex['query'], task_desc=ex['task_desc'],
            max_path_length=max_path_length,
            temperature=0.7, max_new_tokens=64
        )
        print(path)

    elif decode_strategy == 'best_of_n':
        path: RagPath = corag_agent.best_of_n(
            query=ex['query'], task_desc=ex['task_desc'],
            max_path_length=max_path_length,
            temperature=0.7, n=4, max_new_tokens=64
        )
        print(path)

    else:
        raise ValueError(f"Unsupported decode_strategy: {decode_strategy}")

    documents: List[str] = format_documents_for_final_answer(
        context_doc_ids=ex['context_doc_ids'],
        tokenizer=tokenizer, corpus=corpus,
        lock=tokenizer_lock
    )

    prediction: str = corag_agent.generate_final_answer(
        corag_sample=path,
        task_desc=ex['task_desc'],
        documents=documents,
        max_message_length=3072,
        temperature=0.7, max_new_tokens=128
    )

    ex_with_path = copy.deepcopy(ex)
    ex_with_path['subqueries'] = path.past_subqueries
    ex_with_path['subanswers'] = path.past_subanswers
    ex_with_path['path_doc_ids'] = path.past_doc_ids

    if 'title' in corpus.column_names:
        ex_with_path['path_doc_titles'] = [
            [corpus[int(doc_id)]['title'] for doc_id in doc_ids] for doc_ids in path.past_doc_ids
        ]

    ex_with_path['prediction'] = prediction

    processed_cnt.increment()
    if processed_cnt.value % 10 == 0:
        logger.info(
            f'Processed {processed_cnt.value} / {total_cnt} examples, '
            f'average token consumed: {vllm_client.token_consumed.value / processed_cnt.value:.2f}'
        )

    return ex_with_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode_strategy", type=str, default="greedy",
                        choices=["greedy", "tree_search", "best_of_n"])
    parser.add_argument("--num_instances", type=int, default=-1,
                        help="Number of examples to process. -1 means all.")
    parser.add_argument("--output_file", type=str, default="results.txt")

    args = parser.parse_args()

    # Truncate dataset if needed
    selected_ds = ds.select(range(5))

    # Wrap the generator
    generator = partial(_generate_single_example, decode_strategy=args.decode_strategy)
    results: List[Dict] = list(map(generator, selected_ds))

    print('Results - ', results)

    with open(args.output_file, 'w') as f:
        for line in results:
            f.write("%s\n" % line)


if __name__ == '__main__':
    main()
