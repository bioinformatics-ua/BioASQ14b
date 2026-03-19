from vllm import LLM, SamplingParams
from loaders.base import BaseModelBackend


class VLLMBackend(BaseModelBackend):
    """
    Local model backend using vLLM.

    Replaces the lmdeploy-based inference in the old initial_generation.py
    and the Unsloth-based inference in infrence_custom.py — both are now
    handled here through a single vLLM interface.

    The model path is passed in at instantiation and should be set in the
    Slurm script as an environment variable or argument, not hardcoded here.

    Example:
        backend = VLLMBackend(model_path="/data/models/my-model")
        backend.load()
        answer = backend.generate("Question: ... Context: ...")
        backend.unload()
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.5,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ):
        # Path to the model weights on disk — set this in the Slurm script
        self.model_path = model_path

        # Sampling parameters used for every generation call
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        # How many GPUs to shard the model across.
        # Set to the number of GPUs allocated in the Slurm script.
        self.tensor_parallel_size = tensor_parallel_size

        # Fraction of GPU VRAM vLLM may use (0.0–1.0).
        # Lower this when sharing a GPU with other processes.
        self.gpu_memory_utilization = gpu_memory_utilization

        # Maximum context length in tokens.
        self.max_model_len = max_model_len

        # Holds the vLLM engine after load() is called
        self._llm: LLM | None = None

    def load(self) -> None:
        """Load model weights onto GPU via vLLM."""
        print(f"Loading model from {self.model_path}...")
        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=True,  # skip torch.compile — avoids needing Python dev headers
        )
        print("Model loaded.")

    def generate(self, prompt: str) -> str:
        """Run inference on a single prompt (used by cloud backends)."""
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Run inference on a batch of prompts at once.

        vLLM is most efficient when given all prompts together — it can
        schedule and batch them optimally across the GPU rather than running
        each sequentially.
        """
        if self._llm is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        outputs = self._llm.generate(prompts, self.sampling_params)

        # Each output has a list of completions — we request only one per prompt
        return [o.outputs[0].text for o in outputs]

    def unload(self) -> None:
        """
        Release GPU memory.

        vLLM does not expose an explicit unload method, so we delete the
        engine object and let Python's garbage collector free the memory.
        """
        import gc
        import torch

        del self._llm
        self._llm = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded.")
