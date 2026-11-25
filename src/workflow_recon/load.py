"""Stage 1 — load the ActivityWatch export and the task log into tidy DataFrames."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

TZ_DEFAULT = "America/Los_Angeles"

# Bucket-id prefix -> source name. Matched by prefix because the suffix is the
# (changing) network hostname.
BUCKET_PREFIXES: list[tuple[str, str]] = [
    ("aw-watcher-window", "window"),
    ("aw-watcher-web", "web"),
    ("aw-watcher-afk", "afk"),
    ("aw-watcher-jetbrains", "editor"),
    ("aw-watcher-vscode", "editor"),
]
EDITOR_TYPES = {"app.editor.activity"}

EVENT_COLUMNS = [
    "event_id", "source", "bucket_id", "start_utc", "end_utc", "duration_s",
    "app", "title", "url", "domain", "status", "incognito", "audible",
    "project", "file", "language", "start_local", "end_local",
]


def source_for_bucket(bucket_id: str, bucket_type: str | None = None) -> str | None:
    for prefix, source in BUCKET_PREFIXES:
        if bucket_id.startswith(prefix):
            return source
    if bucket_type in EDITOR_TYPES:
        return "editor"
    return None


def read_export(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def inspect_export(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-bucket summary: id, source, event count, data keys, time range. Prints no titles/urls."""
    rows = []
    for bid, b in raw.get("buckets", {}).items():
        events = b.get("events", [])
        keys: Counter = Counter()
        for e in events:
            keys.update((e.get("data") or {}).keys())
        ts = [e["timestamp"] for e in events if "timestamp" in e]
        rows.append({
            "bucket_id": bid,
            "source": source_for_bucket(bid, b.get("type")),
            "type": b.get("type"),
            "n_events": len(events),
            "data_keys": sorted(keys),
            "first": min(ts) if ts else None,
            "last": max(ts) if ts else None,
        })
    return rows


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return None
    host = host.split("@")[-1]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _event_hash(bucket_prefix: str, e: dict) -> str:
    payload = json.dumps([bucket_prefix, e.get("timestamp"), e.get("duration"), e.get("data")],
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def flatten_export(raw: dict[str, Any], tz: str = TZ_DEFAULT) -> pd.DataFrame:
    """Flatten all known buckets into one event table, sorted by start time.

    Buckets of the same watcher under different hostnames are concatenated.
    `event_id` is a content hash, stable across re-exports of the cumulative file.
    """
    records: list[dict[str, Any]] = []
    seen: Counter = Counter()
    for bid, b in raw.get("buckets", {}).items():
        source = source_for_bucket(bid, b.get("type"))
        if source is None:
            continue
        prefix = bid.split("_", 1)[0]
        for e in b.get("events", []):
            d = e.get("data") or {}
            h = _event_hash(prefix, e)
            seen[h] += 1
            eid = f"{source}:{h}" + (f"-{seen[h]}" if seen[h] > 1 else "")
            url = d.get("url")
            records.append({
                "event_id": eid,
                "source": source,
                "bucket_id": bid,
                "timestamp": e.get("timestamp"),
                "duration_s": float(e.get("duration") or 0.0),
                "app": d.get("app"),
                "title": d.get("title"),
                "url": url,
                "domain": domain_of(url),
                "status": d.get("status"),
                "incognito": bool(d.get("incognito", False)),
                "audible": d.get("audible"),
                "project": d.get("project"),
                "file": d.get("file"),
                "language": d.get("language"),
            })
    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame.from_records(records)
    df["start_utc"] = pd.to_datetime(df.pop("timestamp"), utc=True, format="ISO8601")
    df["duration_s"] = df["duration_s"].clip(lower=0.0)
    df["end_utc"] = df["start_utc"] + pd.to_timedelta(df["duration_s"], unit="s")
    df["start_local"] = df["start_utc"].dt.tz_convert(tz)
    df["end_local"] = df["end_utc"].dt.tz_convert(tz)
    df["audible"] = df["audible"].astype("boolean")
    df = df.sort_values(["start_utc", "source", "event_id"], kind="stable").reset_index(drop=True)
    return df[EVENT_COLUMNS]


def _cache_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


def load_events(path: str | Path, tz: str = TZ_DEFAULT, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load + flatten the export, caching the result as Parquet (keyed by file size/mtime)."""
    path = Path(path)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"events_{_cache_key(path)}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df["start_local"] = df["start_utc"].dt.tz_convert(tz)
            df["end_local"] = df["end_utc"].dt.tz_convert(tz)
            return df
    df = flatten_export(read_export(path), tz=tz)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
    return df


# ---------------------------------------------------------------- task log

_TASK_LOG_ALIASES = {
    "task id": "task_id", "id": "task_id", "date": "date", "label": "label",
    "start": "start", "end": "end", "minutes": "minutes", "steps": "steps",
    "apps / sites": "apps_sites", "apps/sites": "apps_sites",
    "repeat of (task id)": "repeat_of", "repeat of": "repeat_of",
    "interrupted?": "interrupted", "interrupted": "interrupted",
    "notes": "notes", "later correction": "later_correction",
}


def _parse_local_date(s: str, default_year: int) -> pd.Timestamp:
    s = re.sub(r"^[A-Za-z]{3,9},?\s+", "", str(s).strip())  # drop weekday, e.g. "Wed 11/18"
    if re.fullmatch(r"\d{1,2}/\d{1,2}", s):
        s = f"{s}/{default_year}"
    return pd.Timestamp(pd.to_datetime(s)).normalize()


def load_task_log(path: str | Path, tz: str = TZ_DEFAULT, default_year: int = 2026) -> pd.DataFrame:
    """Ground truth. For display and scoring ONLY — never pass to a method."""
    raw = pd.read_csv(path, dtype=str).fillna("")
    raw.columns = [_TASK_LOG_ALIASES.get(c.strip().lower(), c.strip().lower().replace(" ", "_")) for c in raw.columns]
    rows = []
    for _, r in raw.iterrows():
        if not r.get("task_id"):
            continue
        day = _parse_local_date(r["date"], default_year)
        start = (day + pd.to_timedelta(pd.to_datetime(r["start"]).strftime("%H:%M:%S"))).tz_localize(tz)
        end = (day + pd.to_timedelta(pd.to_datetime(r["end"]).strftime("%H:%M:%S"))).tz_localize(tz)
        if end <= start:  # crosses midnight
            end += pd.Timedelta(days=1)
        rec = dict(r)
        rec.update(start_local=start, end_local=end,
                   start_utc=start.tz_convert("UTC"), end_utc=end.tz_convert("UTC"))
        rows.append(rec)
    return pd.DataFrame(rows)
