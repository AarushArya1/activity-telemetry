"""On-disk response cache keyed by (prompt version, provider settings, full prompt, schema).

Lives under .cache/ (gitignored): responses are derived from redacted input only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import LLMProvider, LLMResponse


def cache_key(prompt_version: str, settings: dict, system: str, user: str, schema: dict) -> str:
    blob = json.dumps({"v": prompt_version, "settings": settings, "system": system, "user": user,
                       "schema": schema}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedProvider:
    """Wraps a provider; identical requests are served from disk."""

    def __init__(self, inner: LLMProvider, cache_dir: str | Path, prompt_version: str):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.prompt_version = prompt_version
        self.hits = 0
        self.misses = 0

    def settings(self) -> dict[str, Any]:
        return self.inner.settings()

    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int) -> LLMResponse:
        key = cache_key(self.prompt_version, self.settings(), system, user, schema)
        path = self.dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            d = json.loads(path.read_text())
            return LLMResponse(data=d["data"], model=d["model"], usage=d.get("usage", {}), cached=True)
        self.misses += 1
        resp = self.inner.generate_json(system, user, schema, max_tokens)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"data": resp.data, "model": resp.model, "usage": resp.usage}))
        return resp
