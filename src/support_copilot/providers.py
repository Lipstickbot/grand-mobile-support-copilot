"""Explicit local backend selection: never silently changes models or runtime on failure."""


def create_generator(settings):
    kwargs = dict(model_path=settings.model_path,
                  adapter_path=settings.adapter_path if settings.use_adapter else None,
                  runtime_dir=settings.runtime_dir, max_new_tokens=settings.max_new_tokens,
                  timeout_seconds=settings.generation_timeout_seconds,
                  device=settings.device, dtype=settings.model_dtype)
    if settings.inference_backend == "stream":
        from .generation import LocalGenerator
        return LocalGenerator(**kwargs)
    if settings.inference_backend == "resident":
        from .resident import ResidentGenerator
        return ResidentGenerator(**kwargs)
    if settings.inference_backend == "llamacpp":
        from .llamacpp import LlamaCppGenerator
        return LlamaCppGenerator(**kwargs, gguf_path=settings.gguf_path,
                                 gguf_adapter_path=settings.gguf_adapter_path,
                                 binary_path=settings.llama_binary_path, port=settings.llama_port)
    raise ValueError("Unknown inference backend")
