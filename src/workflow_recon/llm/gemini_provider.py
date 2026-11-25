"""Google Gemini provider (google-genai SDK) using JSON-schema structured output.

The API key comes from the environment only: GEMINI_API_KEY (or GOOGLE_API_KEY),
which may be set in your shell or in the repo's gitignored .env file.

Note: on Google's free (unpaid) tier, inputs and outputs may be used to improve
Google's products and read by human reviewers (Gemini API terms). Only redacted
text is ever sent, but titles and URLs still describe your activity.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMConfigError, LLMError, LLMRefusal, LLMResponse

_BLOCKED = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "MODEL_ARMOR"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, temperature: float | None = None, seed: int | None = None,
                 retry_attempts: int = 5, client: Any = None):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.retry_attempts = retry_attempts
        self._client = client

    def settings(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "temperature": self.temperature, "seed": self.seed}

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMConfigError("no Gemini API key: set GEMINI_API_KEY in .env or your shell")
            # Free-tier quotas are tight: retry rate-limit / transient errors with backoff.
            retry = types.HttpRetryOptions(attempts=self.retry_attempts, initial_delay=2.0, max_delay=60.0,
                                           http_status_codes=[429, 500, 502, 503, 504])
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=retry))
        return self._client

    def generate_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> LLMResponse:
        from google.genai import errors, types

        cfg: dict[str, Any] = dict(system_instruction=system, response_mime_type="application/json",
                                   response_json_schema=schema, max_output_tokens=max_tokens)
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self.seed is not None:
            cfg["seed"] = self.seed
        try:
            resp = self.client.models.generate_content(model=self.model, contents=user,
                                                       config=types.GenerateContentConfig(**cfg))
        except errors.ClientError as e:
            msg = str(getattr(e, "message", "") or e)
            if e.code in (401, 403) or "api key" in msg.lower():
                raise LLMConfigError(f"Gemini rejected the API key ({e.code})") from e
            if e.code == 404:
                raise LLMConfigError(f"unknown Gemini model {self.model!r} (check llm.gemini.model)") from e
            if e.code == 429:
                raise LLMError("Gemini quota exhausted (429) after retries; free-tier daily limits may apply") from e
            raise LLMError(f"Gemini request error {e.code}: {msg[:200]}") from e
        except errors.ServerError as e:
            raise LLMError(f"Gemini server error {e.code}") from e

        fb = getattr(resp, "prompt_feedback", None)
        if fb is not None and getattr(fb, "block_reason", None):
            raise LLMRefusal(f"prompt blocked ({_name(fb.block_reason)})")
        cands = resp.candidates or []
        reason = _name(getattr(cands[0], "finish_reason", None)) if cands else None
        if reason in _BLOCKED:
            raise LLMRefusal(f"response blocked ({reason})")
        if reason == "MAX_TOKENS":
            raise LLMError("response truncated at max_output_tokens")
        text = resp.text
        if not text:
            raise LLMError(f"empty response (finish_reason={reason})")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("response was not valid JSON") from e
        u = resp.usage_metadata
        usage = {"input_tokens": int(getattr(u, "prompt_token_count", 0) or 0),
                 "output_tokens": int(getattr(u, "candidates_token_count", 0) or 0)
                 + int(getattr(u, "thoughts_token_count", 0) or 0),
                 "cache_read_input_tokens": int(getattr(u, "cached_content_token_count", 0) or 0)}
        return LLMResponse(data=data, model=getattr(resp, "model_version", None) or self.model, usage=usage)


def _name(v: Any) -> str | None:
    return None if v is None else getattr(v, "name", str(v))
