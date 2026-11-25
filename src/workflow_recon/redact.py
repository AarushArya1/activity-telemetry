"""Stage 4 — redact personal data from the segment table.

Pure, testable functions. Returns the redacted table plus a report of how many
entities of each type were replaced (counts only, never values).

Entity placeholders:
  <EMAIL> <PHONE> <ID> <PERSON> <CHAT>

Layers (each independently testable):
  1. Chat contexts (Messages app, Instagram/LinkedIn DMs...): title -> <CHAT>
  2. Social profile URLs (linkedin.com/in/<x>, instagram.com/<x>...): handle -> <PERSON>
  3. Regex: emails, phone numbers, long ID-like tokens (URLs and titles)
  4. User term list (data/redact_terms.txt, not committed): -> <PERSON>
  5. Presidio PERSON detection, if installed (use_presidio: auto|true|false)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .clean import make_token

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
PHONE_RE = re.compile(
    r"(?<![\w/])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?!\w)"  # (510) 555-0142, 510.555.0142
    r"|(?<![\w/])\+\d{10,14}(?!\w)"                                     # +15105550142
    r"|(?<![\w/.-])\d{3}-\d{4}(?![\w-])"                                # 555-0142 (local)
)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
LONG_DIGITS_RE = re.compile(r"(?<![\d<])\d{6,}(?![\d>])")

_PROFILE_RULES: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"^(?P<pre>(?:[a-z]+\.)?linkedin\.com/(?:in|pub))/(?P<h>[^/?#]+)"), set()),
    (re.compile(r"^(?P<pre>instagram\.com)/(?P<h>[^/?#]+)"),
     {"direct", "explore", "reels", "reel", "p", "stories", "accounts", "about", "developer", "<id>"}),
    (re.compile(r"^(?P<pre>(?:x|twitter)\.com)/(?P<h>[^/?#]+)"),
     {"home", "explore", "search", "notifications", "messages", "settings", "i", "compose"}),
    (re.compile(r"^(?P<pre>tiktok\.com)/(?P<h>@[^/?#]+)"), set()),
    (re.compile(r"^(?P<pre>facebook\.com)/(?P<h>[^/?#]+)"),
     {"messages", "groups", "marketplace", "watch", "events", "home.php", "<id>"}),
]

PLACEHOLDERS = ("<EMAIL>", "<PHONE>", "<ID>", "<PERSON>", "<CHAT>")


@dataclass
class RedactionResult:
    segments: pd.DataFrame
    report: dict = field(default_factory=dict)


class Redactor:
    def __init__(self, cfg: dict, terms: Iterable[str] = (), person_detector: Callable[[str], list[tuple[int, int]]] | None = None):
        r = cfg["redact"]
        self.chat_apps = set(r.get("chat_apps", []))
        self.chat_prefixes = tuple(r.get("chat_url_prefixes", []))
        self.id_min_len = int(r.get("id_min_len", 12))
        self.token_depth = cfg["clean"].get("token_path_depth", 1)
        terms = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
        self.terms_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I) if terms else None
        self.person_detector = person_detector
        self.counts: Counter = Counter()

    # ---- primitives (each returns the new string and updates counts)

    def _sub(self, pattern: re.Pattern, repl: str, text: str, kind: str) -> str:
        new, n = pattern.subn(repl, text)
        self.counts[kind] += n
        return new

    def _id_token(self, tok: str) -> bool:
        if UUID_RE.fullmatch(tok):
            return True
        if tok.isdigit() and len(tok) >= 5:
            return True
        return (len(tok) >= self.id_min_len and re.fullmatch(r"[A-Za-z0-9_\-=.~%]+", tok) is not None
                and re.search(r"\d", tok) is not None)

    def redact_url(self, url: str | None) -> str | None:
        if not isinstance(url, str) or not url:
            return None
        url = self._sub(EMAIL_RE, "<EMAIL>", url, "EMAIL")
        for pat, reserved in _PROFILE_RULES:
            m = pat.match(url)
            if m and m.group("h").lower() not in reserved and not m.group("h").startswith("<"):
                url = f"{m.group('pre')}/<PERSON>" + url[m.end():]
                self.counts["PERSON"] += 1
                break
        base, _, query = url.partition("?")
        parts = base.split("/")
        for i, p in enumerate(parts[1:], 1):
            if p and not p.startswith("<") and self._id_token(p):
                parts[i] = "<ID>"
                self.counts["ID"] += 1
        base = "/".join(parts)
        if query:
            kv = []
            for pair in query.split("&"):
                k, _, v = pair.partition("=")
                if v and self._id_token(v):
                    v = "<ID>"
                    self.counts["ID"] += 1
                kv.append(f"{k}={v}" if _ else k)
            base += "?" + "&".join(kv)
        return base

    def redact_text(self, text: str | None) -> str | None:
        if not isinstance(text, str) or not text:
            return text if isinstance(text, str) else None
        text = self._sub(EMAIL_RE, "<EMAIL>", text, "EMAIL")
        text = self._sub(PHONE_RE, "<PHONE>", text, "PHONE")
        text = self._sub(UUID_RE, "<ID>", text, "ID")
        text = self._sub(LONG_DIGITS_RE, "<ID>", text, "ID")
        if self.terms_re is not None:
            text = self._sub(self.terms_re, "<PERSON>", text, "PERSON")
        if self.person_detector is not None:
            spans = sorted(self.person_detector(text), reverse=True)
            for s, e in spans:
                if "<" in text[s:e] or ">" in text[s:e]:
                    continue
                text = text[:s] + "<PERSON>" + text[e:]
                self.counts["PERSON"] += 1
        return text

    def is_chat(self, app: str | None, url: str | None) -> bool:
        return (app in self.chat_apps) or (isinstance(url, str) and url.startswith(self.chat_prefixes))

    def is_profile(self, url: str | None) -> bool:
        return bool(url and "/<PERSON>" in url)

    # ---- table

    def redact_segments(self, segments: pd.DataFrame) -> RedactionResult:
        self.counts = Counter()
        df = segments.copy()
        changed = 0
        titles, urls, paths, details, tokens = [], [], [], [], []
        for r in df.itertuples(index=False):
            url = self.redact_url(r.url_norm)
            if self.is_chat(r.app, r.url_norm):
                title = "<CHAT>" if r.title_norm else r.title_norm
                if r.title_norm and r.title_norm != "<CHAT>":
                    self.counts["CHAT"] += 1
            elif self.is_profile(url):
                title = "<PERSON> profile"
                self.counts["PERSON"] += 1
            else:
                title = self.redact_text(r.title_norm)
            path = None
            if url and r.domain:
                path = "/" + url.split("?")[0].split("/", 1)[1] if "/" in url.split("?")[0] else "/"
            detail = self.redact_text(r.detail)
            changed += sum(int(new != (old if isinstance(old, str) else None))
                           for new, old in ((title, r.title_norm), (url, r.url_norm), (detail, r.detail)))
            titles.append(title); urls.append(url); paths.append(path if r.domain else r.path)
            details.append(detail)
            tokens.append(make_token(r.app, r.domain, path if r.domain else r.path, self.token_depth))
        for col, vals in (("title_norm", titles), ("url_norm", urls), ("path", paths), ("detail", details)):
            df[col] = pd.Series(vals, index=df.index, dtype=object)
        df["token"] = tokens
        df.attrs["redacted"] = True
        report = {"entities": dict(self.counts), "strings_changed": changed, "segments": len(df),
                  "presidio": self.person_detector is not None, "user_terms": self.terms_re is not None}
        return RedactionResult(df, report)


# ------------------------------------------------------------ helpers

def presidio_detector() -> Callable[[str], list[tuple[int, int]]] | None:
    """PERSON span detector backed by Presidio, or None if not installed."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
    except Exception:
        return None
    engine = AnalyzerEngine()

    def detect(text: str) -> list[tuple[int, int]]:
        return [(r.start, r.end) for r in engine.analyze(text=text, entities=["PERSON"], language="en")
                if r.score >= 0.6]
    return detect


def load_terms(path: str | Path | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def make_redactor(cfg: dict, terms_path: str | Path | None = None) -> Redactor:
    mode = cfg["redact"].get("use_presidio", "auto")
    detector = None
    if mode in (True, "auto", "true"):
        detector = presidio_detector()
        if detector is None and mode is True:
            raise RuntimeError("use_presidio: true but presidio-analyzer is not installed")
    return Redactor(cfg, load_terms(terms_path), detector)


def redact_segments(segments: pd.DataFrame, cfg: dict, terms_path: str | Path | None = None) -> RedactionResult:
    return make_redactor(cfg, terms_path).redact_segments(segments)


def find_leaks(text: str) -> dict[str, int]:
    """Last-line check before text leaves the machine: residual emails / phones."""
    return {k: n for k, n in (("EMAIL", len(EMAIL_RE.findall(text))), ("PHONE", len(PHONE_RE.findall(text)))) if n}
