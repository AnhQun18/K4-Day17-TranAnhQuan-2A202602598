from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, memory settings, and model configurations for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load the root .env without overriding existing environment variables."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    load_dotenv(root / ".env")

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    defaults = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-3.6-flash",
        "anthropic": "claude-sonnet-4-5",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-4o-mini",
    }
    if provider not in defaults:
        raise ValueError(f"Unsupported provider: {provider!r}")

    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    key_name = api_key_env.get(provider)
    base_url = None
    if provider == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    elif provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"

    model = ProviderConfig(
        provider=provider,
        model_name=os.getenv("LLM_MODEL") or defaults[provider],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=os.getenv(key_name) if key_name else None,
        base_url=base_url,
    )
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=800,
        compact_keep_messages=4,
        model=model,
        judge_model=replace(model),
    )
