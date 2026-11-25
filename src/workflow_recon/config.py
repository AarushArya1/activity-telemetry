"""Load config.yaml (with a deep-merge over overrides)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> list[str]:
    """Load KEY=VALUE lines from the gitignored .env into os.environ.

    Variables already set in the shell win. Returns the names loaded (never values).
    """
    path = Path(path)
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.removeprefix("export ").partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    load_dotenv()
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a `paths.<key>` entry relative to the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p
