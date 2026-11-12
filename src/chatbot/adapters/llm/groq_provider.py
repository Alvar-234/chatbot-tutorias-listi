import os

from .openai_compat import OpenAICompatProvider


def GroqProvider(model: str, timeout_s: float = 30.0) -> OpenAICompatProvider:
    return OpenAICompatProvider(name="groq", base_url="https://api.groq.com/openai/v1",
                                model=model, api_key=os.environ.get("GROQ_API_KEY", ""),
                                timeout_s=timeout_s)
