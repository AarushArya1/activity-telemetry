"""Stage 3 — clean raw events into a segment table.

Order (PROJECT_CONTEXT.md §6, stage 3):
  1. AFK masking (+ idle apps such as loginwindow)
  2. Chrome dedup: browser window events take url/title from overlapping web events
  3. Drop private: incognito time, and optional private intervals
  4. Normalize URLs and titles
  5. Merge into activity segments, absorb micro-segments
  6. Derive a compact token per segment
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import pandas as pd

SEGMENT_COLUMNS = [
    "segment_id", "start", "end", "duration_s", "app", "domain", "path",
    "title_norm", "url_norm", "token", "detail", "source_event_ids",
]

Interval = tuple[float, float]  # epoch seconds


@dataclass
class CleanResult:
    segments: pd.DataFrame
    idle: list[tuple[pd.Timestamp, pd.Timestamp]]  # masked (AFK / idle / private) periods, local tz
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------ normalization

_BROWSER_SUFFIX = re.compile(r"\s+[-–—|]\s+(Google Chrome|Chrome|Safari|Firefox|Brave|Arc|Microsoft Edge)$", re.I)
# unread counts like "Inbox (16,221)" — but never an area code like "(510) 555-0142"
_UNREAD = re.compile(r"\s*\(\d[\d,]*\+?\)(?![\s.-]*\d{3}[\s.-]\d{4})")
_LEAD_MARK = re.compile(r"^[●•*]\s*")
_WS = re.compile(r"\s+")
_SITE_SUFFIX = re.compile(r"^(?P<body>.+?)\s+[-–—|•·]\s+(?P<site>[^-–—|•·]{2,40})$")


def normalize_title(title: str | None, domain: str | None = None) -> str:
    if not title:
        return ""
    t = _BROWSER_SUFFIX.sub("", title.strip())
    t = _UNREAD.sub("", t)
    t = _LEAD_MARK.sub("", t)
    if domain:
        # Strip a trailing " - <Site>" when the site name is the domain (" - YouTube" on youtube.com).
        m = _SITE_SUFFIX.match(t)
        if m:
            site = re.sub(r"[^a-z0-9]", "", m.group("site").lower())
            if site and site in domain.replace(".", ""):
                t = m.group("body")
    return _WS.sub(" ", t).strip()


def normalize_url(url: str | None, allowlist: Iterable[str] = ()) -> tuple[str | None, str | None, str | None]:
    """Return (domain, path, url_norm) with tracking params dropped.

    url_norm is scheme-less: 'linkedin.com/jobs/search?keywords=intern'.
    """
    if not url:
        return None, None, None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None, None, None
    host = parts.netloc.lower().split("@")[-1]
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None, None, None
    path = parts.path or "/"
    if parts.fragment.startswith("/"):  # hash-routed SPA, e.g. localhost:5600/#/timeline
        path = path.rstrip("/") + "/#" + parts.fragment
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")
    allow = {a.lower() for a in allowlist}
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if k.lower() in allow]
    q = ("?" + urlencode(kept)) if kept else ""
    return host, path, f"{host}{'' if path == '/' else path}{q}"


_PATH_WORD = re.compile(r"^[a-z][a-z-]{1,19}$")


def make_token(app: str | None, domain: str | None, path: str | None, depth: int = 1) -> str:
    if domain:
        tok = domain
        if depth >= 1 and path:
            first = path.strip("/").split("/")[0].lower()
            if _PATH_WORD.match(first):
                tok = f"{domain}/{first}"
        return tok
    return app or "unknown"


# ------------------------------------------------------------ interval utils

def _merge_intervals(ivs: Iterable[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for s, e in sorted(ivs):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _subtract(s: float, e: float, masks: list[Interval], mask_starts: np.ndarray) -> list[Interval]:
    """Subtract sorted disjoint masks from [s, e)."""
    if not masks:
        return [(s, e)]
    i = max(int(np.searchsorted(mask_starts, s, side="right")) - 1, 0)
    out, cur = [], s
    while i < len(masks) and masks[i][0] < e:
        ms, me = masks[i]
        if me > cur:
            if ms > cur:
                out.append((cur, min(ms, e)))
            cur = max(cur, me)
        i += 1
    if cur < e:
        out.append((cur, e))
    return [(a, b) for a, b in out if b - a > 1e-6]


def _to_local(epoch: float, tz: str) -> pd.Timestamp:
    """Epoch seconds -> tz-aware Timestamp at microsecond precision (what datetime holds)."""
    return pd.Timestamp(round(epoch * 1e6), unit="us", tz="UTC").tz_convert(tz)


def _epoch(series: pd.Series) -> np.ndarray:
    return (series.dt.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


# ------------------------------------------------------------ main

def build_pieces(events: pd.DataFrame, cfg: dict,
                 private_intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]] = ()) -> tuple[list[dict], list[Interval], dict]:
    """Foreground pieces (dicts with epoch s/e) after dedup and masking."""
    c = cfg["clean"]
    idle_apps = set(c["idle_apps"])
    browsers = set(c["browser_apps"])
    stats: dict = {"events_in": events["source"].value_counts().to_dict()}

    win = events[events["source"] == "window"].copy()
    web = events[events["source"] == "web"].copy()
    afk = events[events["source"] == "afk"]
    editor = events[events["source"] == "editor"]

    # Masks: AFK, idle apps, incognito web time, private intervals.
    masks: list[Interval] = []
    if c.get("afk_mask", True) and not afk.empty:
        a = afk[afk["status"] == "afk"]
        masks += list(zip(_epoch(a["start_utc"]), _epoch(a["end_utc"])))
    idle_win = win[win["app"].isin(idle_apps)]
    masks += list(zip(_epoch(idle_win["start_utc"]), _epoch(idle_win["end_utc"])))
    if c.get("drop_incognito", True) and not web.empty:
        inc = web[web["incognito"]]
        masks += list(zip(_epoch(inc["start_utc"]), _epoch(inc["end_utc"])))
        stats["incognito_dropped"] = int(len(inc))
        web = web[~web["incognito"]]
    for ps, pe in private_intervals:
        masks.append((pd.Timestamp(ps).timestamp(), pd.Timestamp(pe).timestamp()))
    masks = _merge_intervals(masks)
    mask_starts = np.array([m[0] for m in masks])

    win = win[~win["app"].isin(idle_apps)]
    w_s, w_e = _epoch(win["start_utc"]), _epoch(win["end_utc"])
    web = web.sort_values("start_utc")
    b_s, b_e = _epoch(web["start_utc"]), _epoch(web["end_utc"])
    max_web = float((b_e - b_s).max()) if len(b_s) else 0.0
    web_rows = web[["event_id", "url", "title"]].to_dict("records")

    ed_s, ed_e = _epoch(editor["start_utc"]), _epoch(editor["end_utc"])
    ed_rows = editor[["project", "language"]].to_dict("records")

    raw_pieces: list[dict] = []
    for (s, e, row) in zip(w_s, w_e, win[["event_id", "app", "title"]].to_dict("records")):
        if e <= s:
            continue
        if row["app"] in browsers and len(b_s):
            lo = int(np.searchsorted(b_s, s - max_web, side="left"))
            hi = int(np.searchsorted(b_s, e, side="left"))
            covered: list[Interval] = []
            for j in range(lo, hi):
                os_, oe = max(s, b_s[j]), min(e, b_e[j])
                if oe > os_:
                    wr = web_rows[j]
                    raw_pieces.append({"s": os_, "e": oe, "app": row["app"], "title": wr["title"],
                                       "url": wr["url"], "ids": [row["event_id"], wr["event_id"]]})
                    covered.append((os_, oe))
            for gs, ge in _subtract(s, e, _merge_intervals(covered),
                                    np.array([x[0] for x in _merge_intervals(covered)])):
                raw_pieces.append({"s": gs, "e": ge, "app": row["app"], "title": row["title"],
                                   "url": None, "ids": [row["event_id"]]})
        else:
            raw_pieces.append({"s": s, "e": e, "app": row["app"], "title": row["title"],
                               "url": None, "ids": [row["event_id"]]})

    # Editor enrichment: attach project/language to overlapping IDE pieces.
    if len(ed_s):
        order = np.argsort(ed_s)
        ed_s, ed_e = ed_s[order], ed_e[order]
        ed_rows = [ed_rows[i] for i in order]
        for p in raw_pieces:
            j = int(np.searchsorted(ed_s, p["e"], side="left")) - 1
            if j >= 0 and ed_e[j] > p["s"]:
                r = ed_rows[j]
                p["detail"] = " / ".join(str(x) for x in (r.get("project"), r.get("language")) if x)

    pieces: list[dict] = []
    for p in raw_pieces:
        for s, e in _subtract(p["s"], p["e"], masks, mask_starts):
            pieces.append({**p, "s": s, "e": e})
    pieces.sort(key=lambda p: (p["s"], p["e"]))
    stats["pieces"] = len(pieces)
    stats["masked_s"] = round(float(sum(e - s for s, e in masks)), 1)
    return pieces, masks, stats


def _merge_segments(rows: list[dict], gap_s: float, key) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if out and key(out[-1]) == key(r) and r["s"] - out[-1]["e"] <= gap_s:
            out[-1]["e"] = max(out[-1]["e"], r["e"])
            out[-1]["ids"].extend(r["ids"])
        else:
            out.append({**r, "ids": list(r["ids"])})
    return out


def clean_events(events: pd.DataFrame, cfg: dict,
                 private_intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]] = ()) -> CleanResult:
    tz = cfg["timezone"]
    c = cfg["clean"]
    aliases = c.get("app_aliases", {})
    pieces, masks, stats = build_pieces(events, cfg, private_intervals)

    rows = []
    for p in pieces:
        app = aliases.get(p["app"], p["app"])
        domain, path, url_norm = normalize_url(p["url"], c["url_param_allowlist"])
        rows.append({"s": p["s"], "e": p["e"], "app": app, "domain": domain, "path": path,
                     "url_norm": url_norm, "title_norm": normalize_title(p["title"], domain),
                     "detail": p.get("detail"), "ids": p["ids"]})

    exact = lambda r: (r["app"], r["domain"], r["title_norm"])  # noqa: E731
    segs = _merge_segments(rows, c["merge_gap_s"], exact)

    # Absorb micro-segments into a same-source predecessor (e.g. clicking through
    # LinkedIn listings), then drop whatever is still too short.
    min_s = c["min_segment_s"]
    source = lambda r: (r["app"], r["domain"])  # noqa: E731
    absorbed: list[dict] = []
    for r in segs:
        if (r["e"] - r["s"] < min_s and absorbed and source(absorbed[-1]) == source(r)
                and r["s"] - absorbed[-1]["e"] <= max(c["merge_gap_s"], min_s)):
            absorbed[-1]["e"] = max(absorbed[-1]["e"], r["e"])
            absorbed[-1]["ids"].extend(r["ids"])
        else:
            absorbed.append(r)
    kept = [r for r in absorbed if r["e"] - r["s"] >= min_s]
    stats["micro_dropped"] = len(absorbed) - len(kept)
    segs = _merge_segments(kept, c["merge_gap_s"], exact)

    depth = c.get("token_path_depth", 1)
    out = []
    for i, r in enumerate(segs, 1):
        out.append({
            "segment_id": f"S{i:04d}",
            "start": _to_local(r["s"], tz),
            "end": _to_local(r["e"], tz),
            "duration_s": round(r["e"] - r["s"], 3),
            "app": r["app"], "domain": r["domain"], "path": r["path"],
            "title_norm": r["title_norm"], "url_norm": r["url_norm"],
            "token": make_token(r["app"], r["domain"], r["path"], depth),
            "detail": r["detail"],
            "source_event_ids": list(dict.fromkeys(r["ids"])),
        })
    df = pd.DataFrame(out, columns=SEGMENT_COLUMNS)
    for col in ("app", "domain", "path", "title_norm", "url_norm", "detail"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    df.attrs["redacted"] = False
    stats["segments"] = len(df)
    stats["tokens"] = int(df["token"].nunique()) if len(df) else 0
    idle = [(_to_local(s, tz), _to_local(e, tz)) for s, e in masks]
    return CleanResult(df, idle, stats)
