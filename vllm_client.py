from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import AtomicCounter


class VllmClient:
    def __init__(self, model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,  # Use float32 for MPS (M1-M4)
            trust_remote_code=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = device
        self.token_consumed = AtomicCounter()

    def _format_messages(self, messages: List[Dict]) -> str:
        # Assuming OpenAI-style messages: [{'role': 'user', 'content': '...'}, ...]
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"

    def call_chat(self, messages: List[Dict], return_str: bool = True, max_new_tokens: int = 256, **kwargs) -> Union[
        str, Dict]:
        prompt = self._format_messages(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        temperature = kwargs.pop("temperature", 0.7)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            **kwargs
        )

        self.token_consumed.increment(num=output.shape[-1])
        response = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        if return_str:
            return response.strip()
        else:
            # Simulate OpenAI-style `ChatCompletion`
            return {
                "choices": [{"message": {"role": "assistant", "content": response.strip()}}],
                "usage": {"total_tokens": output.shape[-1]}
            }

    def batch_call_chat(self, messages: List[List[Dict]], return_str: bool = True, **kwargs) -> List[Union[str, Dict]]:
        return [self.call_chat(m, return_str=return_str, **kwargs) for m in messages]