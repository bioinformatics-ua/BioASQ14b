import gc
from collections.abc import Sequence

import torch
from vllm import LLM, SamplingParams

from bioasq.phase_b.backends.base import BaseModelBackend


class VLLMBackend(BaseModelBackend):
    """Local model backend using vLLM.

    Parameters
    ----------
    model_path:
        Path to model weights on disk.
    max_new_tokens:
        Maximum tokens to generate per prompt.
    temperature:
        Sampling temperature.
    tensor_parallel_size:
        Number of GPUs to shard across.
    gpu_memory_utilization:
        Fraction of GPU VRAM vLLM may use (0.0-1.0).
    max_model_len:
        Maximum context length in tokens.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.5,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ) -> None:
        self.model_path: str = model_path
        self.sampling_params: SamplingParams = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        self.tensor_parallel_size: int = tensor_parallel_size
        self.gpu_memory_utilization: float = gpu_memory_utilization
        self.max_model_len: int = max_model_len
        self._llm: LLM | None = None

    def load(self) -> None:
        """Load model weights onto GPU via vLLM."""

        print(f"Loading model from {self.model_path}...")
        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )
        print("Model loaded.")

    def generate(self, prompt: str) -> str:
        """Generate a single completion."""
        return self.generate_batch([prompt])[0]

    # def generate_chat(self, messages: Sequence[dict[str, str]]) -> str:
    #     """Generate using the model's chat template."""
    #     if self._llm is None:
    #         msg: str = "Model is not loaded. Call load() first."
    #         raise RuntimeError(msg)
    #     outputs = self._llm.generate(
    #         messages=[list(messages)],
    #         sampling_params=self.sampling_params,
    #         use_tqdm=False,
    #     )
    #     return outputs[0].outputs[0].text

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to fit within context window."""
        if self._llm is None:
            return prompt
        tokenizer = self._llm.get_tokenizer()
        max_new_tokens: int = self.sampling_params.max_tokens or 16
        max_input: int = self.max_model_len - max_new_tokens
        ids: list[int] = tokenizer.encode(prompt)
        if len(ids) <= max_input:
            return prompt
        ids = ids[:max_input]
        s = tokenizer.decode(ids)
        return s if isinstance(s, str) else s[0]

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        """Run inference on a batch of prompts."""
        if self._llm is None:
            msg: str = "Model is not loaded. Call load() first."
            raise RuntimeError(msg)
        truncated: list[str] = [self._truncate_prompt(p) for p in prompts]
        outputs = self._llm.generate(truncated, self.sampling_params)  # type: ignore[union-attr]
        return [o.outputs[0].text for o in outputs]

    def unload(self) -> None:
        """Release GPU memory."""
        del self._llm
        self._llm = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded.")
