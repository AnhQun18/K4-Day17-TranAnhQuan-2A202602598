from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "gpt": "openai",
    "open_ai": "openai",
    "open-ai": "openai",
    "google": "gemini",
    "google-genai": "gemini",
    "open_router": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map common aliases to canonical provider names."""
    normalized = value.strip().lower()
    return _ALIASES.get(normalized, normalized)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Live initialization fails explicitly; callers select offline mode themselves.
    """
    provider = normalize_provider(config.provider)

    if provider not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {provider!r}")
    if provider == "custom" and not config.base_url:
        raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
    if provider not in {"custom", "ollama"} and not config.api_key:
        raise ValueError(f"Missing API key for {provider}. Fill in the root .env file.")

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "custom":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or "sk-placeholder",
                base_url=config.base_url,
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434",
            )

        if provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    except ImportError as exc:
        raise RuntimeError(f"Missing provider library for {provider}; install requirements.txt") from exc

    raise ValueError(f"Unsupported provider: {provider!r}")


BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Với câu hỏi về người dùng, chỉ dùng facts đã cung cấp; "
    "nếu chưa biết thì nói rõ. Ưu tiên thông tin đính chính mới nhất. "
    "Trả lời ngắn gọn và đúng câu hỏi. "
)


def invoke_chat(model, messages: list[dict[str, str]]) -> tuple[str, int, int]:
    """Normalize LangChain responses and count provider usage when available.

    If a provider omits usage metadata, use the same estimator as offline mode.
    Network/authentication errors propagate instead of silently faking a reply.
    """
    from memory_store import estimate_tokens

    result = model.invoke(messages)
    content = result.content
    if isinstance(content, list):
        content = "".join(block if isinstance(block, str) else block.get("text", "") for block in content)
    usage = getattr(result, "usage_metadata", None) or {}
    return (
        str(content),
        usage.get("input_tokens", sum(estimate_tokens(item["content"]) for item in messages)),
        usage.get("output_tokens", estimate_tokens(str(content))),
    )
