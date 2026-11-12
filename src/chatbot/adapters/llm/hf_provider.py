import os

from .openai_compat import OpenAICompatProvider


def HuggingFaceProvider(model: str, timeout_s: float = 30.0) -> OpenAICompatProvider:
    return OpenAICompatProvider(name="huggingface", base_url="https://router.huggingface.co/v1",
                                model=model, api_key=os.environ.get("HF_TOKEN", ""),
                                timeout_s=timeout_s)
