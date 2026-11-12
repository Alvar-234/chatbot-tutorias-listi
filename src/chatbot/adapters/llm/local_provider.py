from .openai_compat import OpenAICompatProvider


def LocalProvider(model: str, base_url: str = "http://localhost:11434/v1",
                  timeout_s: float = 60.0) -> OpenAICompatProvider:
    """Ollama / llama.cpp / vLLM expuestos con API compatible con OpenAI (contingencia local)."""
    return OpenAICompatProvider(name="local", base_url=base_url, model=model,
                                timeout_s=timeout_s)
