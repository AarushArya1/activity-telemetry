"""Provider interface. Core logic depends only on this, never on a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Any failure to get a usable structured response."""


class LLMRefusal(LLMError):
    """The model declined the request."""


class LLMConfigError(RuntimeError):
    """Misconfiguration (e.g. missing credentials). Aborts the run instead of skipping a chunk."""


@dataclass
class LLMResponse:
    data: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False


class LLMProvider(Protocol):
    name: str
    model: str

    def settings(self) -> dict[str, Any]:
        """Everything that affects the output (used in the cache key and run params)."""
        ...

    def generate_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> LLMResponse:
        ...


def provider_settings(llm_cfg: dict[str, Any]) -> dict[str, Any]:
    """The `llm.<provider>` sub-section of the config."""
    name = llm_cfg.get("provider", "gemini")
    if name not in llm_cfg:
        raise ValueError(f"config has no llm.{name} section")
    return llm_cfg[name]


def get_provider(llm_cfg: dict[str, Any]) -> LLMProvider:
    name = llm_cfg.get("provider", "gemini")
    p = provider_settings(llm_cfg)
    if name == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model=p["model"], temperature=p.get("temperature"), seed=p.get("seed"),
                              retry_attempts=int(p.get("retry_attempts", 5)))
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=p["model"], effort=p.get("effort"), temperature=p.get("temperature"),
                                 server_fallbacks=bool(p.get("server_fallbacks", True)))
    raise ValueError(f"unknown LLM provider {name!r} (add one in workflow_recon/llm/)")
