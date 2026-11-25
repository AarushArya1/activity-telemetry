"""Anthropic (Claude) provider using structured outputs (JSON schema).

Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login`
profile) — never from config files.
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLMConfigError, LLMError, LLMRefusal, LLMResponse

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, effort: str | None = "high", temperature: float | None = None,
                 server_fallbacks: bool = True, client: Any = None):
        self.model = model
        self.effort = effort
        self.temperature = temperature
        self.server_fallbacks = server_fallbacks
        self._client = client

    def settings(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "effort": self.effort,
                "temperature": self.temperature, "server_fallbacks": self.server_fallbacks}

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def generate_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> LLMResponse:
        import anthropic

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": user}],
                                      output_config=output_config)
        if self.temperature is not None:  # current Claude models reject sampling params
            kwargs["temperature"] = self.temperature
        try:
            if self.server_fallbacks:
                with self.client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default") as stream:
                    msg = stream.get_final_message()
            else:
                with self.client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
        except TypeError as e:  # raised by the SDK before any request when no credentials resolve
            if "authentication" in str(e).lower():
                raise LLMConfigError("no Anthropic credentials: set ANTHROPIC_API_KEY (or run `ant auth login`)") from e
            raise
        except anthropic.AuthenticationError as e:
            raise LLMConfigError("Anthropic credentials were rejected (401)") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"bad request: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("rate limited (after SDK retries)") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("connection error") from e

        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise LLMRefusal(f"model declined (category={cat})")
        if msg.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("response was not valid JSON") from e
        u = msg.usage
        usage = {k: int(getattr(u, k, 0) or 0) for k in
                 ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        return LLMResponse(data=data, model=getattr(msg, "model", self.model), usage=usage)
