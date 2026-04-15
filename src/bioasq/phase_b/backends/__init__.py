from bioasq.phase_b.backends.base import BaseModelBackend


def get_backend(
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
    request_delay: float = 0.0,
    local_max_tokens: int | None = None,
) -> BaseModelBackend:
    """Instantiate and load one backend per model.

    Model strings use ``"backend|model_name"`` format, e.g.
    ``"openrouter|google/gemini-2.5-flash"`` or ``"local|google/medgemma-27b-text-it"``, or
    ``"external|192.168.1.2:8080|google/medgemma-27b-text-it"`` for custom OpenAI-based endpoints.
    If no prefix is given, defaults to ``"openrouter"``.
    """
    from bioasq.phase_b.backends.cloud import OpenRouterBackend

    if "|" in model:
        backend_type, model_name = model.split("|", 1)
    else:
        backend_type, model_name = "openrouter", model

    if backend_type == "local":
        from bioasq.phase_b.backends.local import VLLMBackend

        backend = VLLMBackend(
            model_path=model_name,
            max_new_tokens=local_max_tokens if local_max_tokens is not None else max_tokens,
            temperature=temperature,
            tensor_parallel_size=2,
        )
    else:
        base_url, model_name = (
            model_name.split("|", 1) if backend_type == "external" else (None, model_name)
        )
        backend = OpenRouterBackend(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            request_delay=request_delay,
            base_url=base_url,
        )
    backend.load()
    return backend
