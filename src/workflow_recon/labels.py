"""Label taxonomy (PROJECT_CONTEXT.md §4.3) and transparent rule-based label hints.

The taxonomy is allowed as method input. The rules below are derived from the
taxonomy's definitions and typical apps only — not from the task log — and are
kept generic (domains, app names, and the label names' own keywords).

`label_hint(app, domain, path, title)` returns (label, strength):
  strength = "strong"  -> unambiguous evidence for the label
  strength = "weak"    -> ambiguous (YouTube, Claude, VS Code, unknown sites...):
                          label is only a fallback when surrounding context is missing
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABELS: list[dict[str, str]] = [
    {"name": "CS 61B Homework/Projects",
     "covers": "CS 61B homework, labs, projects, including debugging and submitting",
     "typical": "IntelliJ, Gradescope, Ed, CS 61B course site, terminal / Git"},
    {"name": "CS 61A Notes/Homework",
     "covers": "CS 61A notes, homework, labs, projects",
     "typical": "VS Code, blank.page, Gradescope, cs61a.org, YouTube, Ed, Claude"},
    {"name": "Quantum Computing @ Berkeley",
     "covers": "Club project work on the VQE / LABS project, or general club work",
     "typical": "Slack, Google Colab, Google Slides, arXiv, Claude"},
    {"name": "Dart (Startup) Work",
     "covers": "Planning (Google Docs, Instagram, Claude) or development (VS Code)",
     "typical": "VS Code, terminal, GitHub, Google Docs, Instagram, Claude, dartme.org, Supabase"},
    {"name": "Orcapo tracking work",
     "covers": "Coding or research for this telemetry project",
     "typical": "VS Code, terminal, GitHub, Claude, ActivityWatch"},
    {"name": "Internship application",
     "covers": "Finding postings, tailoring resumes and cover letters, submitting",
     "typical": "LinkedIn, Handshake, company career pages, Docs"},
    {"name": "Email / Slack / Social Media",
     "covers": "Reading and replying to messages (quantum-team Slack counts as Quantum)",
     "typical": "Gmail, Slack, LinkedIn messages, Instagram messages"},
    {"name": "Admin / planning",
     "covers": "Calendar, course enrollment, to-do lists",
     "typical": "Google Calendar, CalCentral, notes app"},
    {"name": "Entertainment",
     "covers": "Videos, social media, games",
     "typical": "YouTube, Reddit, Instagram, krillion.io, LinkedIn feed"},
    {"name": "Private", "covers": "Anything not to be analyzed", "typical": "Any"},
    {"name": "Other", "covers": "Anything else", "typical": "Any"},
]

LABEL_NAMES: list[str] = [l["name"] for l in LABELS]
# Methods never predict "Private" (those events are dropped before analysis).
PREDICTABLE_LABELS: list[str] = [n for n in LABEL_NAMES if n != "Private"]

L61B, L61A, LQC, LDART, LORC, LINT, LMSG, LADMIN, LENT, _LPRIV, LOTHER = LABEL_NAMES

STRONG, WEAK = "strong", "weak"


def taxonomy_text() -> str:
    """Render the taxonomy for a prompt."""
    return "\n".join(f"- {l['name']}: {l['covers']}. Typical: {l['typical']}." for l in LABELS if l["name"] != "Private")


@dataclass(frozen=True)
class Rule:
    label: str
    strength: str
    domain: str | None = None        # regex on domain
    path: str | None = None          # regex on url path
    app: str | None = None           # regex on app name
    title: str | None = None         # regex on title


def _r(label: str, strength: str, **kw) -> Rule:
    return Rule(label, strength, **kw)


# Title keywords taken from the label names / definitions themselves.
KEYWORD_RULES: list[Rule] = [
    _r(L61A, STRONG, title=r"\b(cs\s?)?61\s?a\b"),
    _r(L61B, STRONG, title=r"\b(cs\s?)?61\s?b\b"),
    _r(LQC, STRONG, title=r"\b(quantum|vqe|qiskit)\b"),
    _r(LDART, STRONG, title=r"\bdart(me)?\b"),
    _r(LORC, STRONG, title=r"\b(orcapo|activitywatch|telemetry)\b"),
]

# First match wins. Order: specific (domain+path) before generic (domain).
SOURCE_RULES: list[Rule] = [
    # --- Course sites
    _r(L61A, STRONG, domain=r"(^|\.)cs61a\.org$"),
    _r(L61B, STRONG, domain=r"(^|\.)(datastructur\.es|cs61b\.org)$"),
    _r(L61B, STRONG, app=r"^IntelliJ IDEA"),
    _r(L61A, WEAK, domain=r"(^|\.)blank\.page$"),           # notes tool used for 61A
    # --- Messaging (path-specific before site-wide)
    _r(LMSG, STRONG, domain=r"(^|\.)linkedin\.com$", path=r"^/messaging"),
    _r(LMSG, STRONG, domain=r"(^|\.)instagram\.com$", path=r"^/direct"),
    _r(LMSG, STRONG, domain=r"^mail\.google\.com$"),
    _r(LMSG, STRONG, domain=r"(^|\.)(outlook\.(live|office)\.com|outlook\.com)$"),
    _r(LMSG, STRONG, app=r"^(Mail|Microsoft Outlook)$"),
    _r(LMSG, WEAK, app=r"^(Messages|WhatsApp|Telegram|Signal|Discord|Messenger)$"),
    _r(LMSG, WEAK, app=r"^Slack$"),
    _r(LMSG, WEAK, domain=r"(^|\.)slack\.com$"),
    # --- Internship
    _r(LINT, STRONG, domain=r"(^|\.)linkedin\.com$", path=r"^/jobs"),
    _r(LINT, STRONG, domain=r"(^|\.)joinhandshake\.com$"),
    _r(LINT, STRONG, domain=r"(^|\.)(greenhouse\.io|lever\.co|ashbyhq\.com|myworkdayjobs\.com|"
                            r"smartrecruiters\.com|icims\.com|workable\.com|jobvite\.com|wellfound\.com)$"),
    _r(LINT, STRONG, domain=r"^(careers?|jobs)\."),
    _r(LINT, STRONG, path=r"^/(careers?|jobs)(/|$)"),
    # --- Entertainment
    _r(LENT, STRONG, domain=r"(^|\.)linkedin\.com$", path=r"^/feed"),
    _r(LENT, STRONG, domain=r"(^|\.)(reddit\.com|krillion\.io|netflix\.com|twitch\.tv|tiktok\.com|x\.com|twitter\.com)$"),
    _r(LENT, WEAK, domain=r"(^|\.)instagram\.com$"),
    _r(LENT, WEAK, domain=r"(^|\.)youtube\.com$"),
    _r(LENT, WEAK, app=r"^(Spotify|Music|TV)$"),
    # --- Quantum
    _r(LQC, WEAK, domain=r"^colab\.research\.google\.com$"),
    _r(LQC, WEAK, domain=r"(^|\.)arxiv\.org$"),
    _r(LQC, WEAK, domain=r"^docs\.google\.com$", path=r"^/presentation"),
    # --- Dart
    _r(LDART, STRONG, domain=r"(^|\.)dartme\.org$"),
    _r(LDART, WEAK, domain=r"(^|\.)supabase\.(com|co)$"),
    # --- Orcapo
    _r(LORC, STRONG, app=r"^(ActivityWatch|aw-qt)$"),
    _r(LORC, STRONG, domain=r"^(localhost|127\.0\.0\.1)(:5600)?$", path=r"^/(#/)?(activity|timeline|buckets|query)"),
    # --- Admin
    _r(LADMIN, STRONG, domain=r"^calendar\.google\.com$"),
    _r(LADMIN, STRONG, domain=r"(^|\.)calcentral\.berkeley\.edu$"),
    _r(LADMIN, WEAK, app=r"^(Calendar|Notes|Reminders|Todoist)$"),
    # --- Ambiguous course tools (61A vs 61B decided by title keywords / context)
    _r(LOTHER, WEAK, domain=r"(^|\.)(gradescope\.com|edstem\.org)$"),
    # --- Ambiguous general tools
    _r(LOTHER, WEAK, domain=r"(^|\.)(claude\.ai|chatgpt\.com|gemini\.google\.com|perplexity\.ai)$"),
    _r(LOTHER, WEAK, domain=r"(^|\.)(google\.com|bing\.com|duckduckgo\.com)$"),
    _r(LOTHER, WEAK, domain=r"(^|\.)github\.com$"),
    _r(LOTHER, WEAK, domain=r"^docs\.google\.com$"),
    _r(LOTHER, WEAK, app=r"^(VS Code|Terminal|iTerm2|Warp|Finder|Preview|Claude|ChatGPT)$"),
]

_COMPILED: dict[int, dict[str, re.Pattern]] = {}


def _compiled(rule: Rule) -> dict[str, re.Pattern]:
    key = id(rule)
    if key not in _COMPILED:
        _COMPILED[key] = {f: re.compile(getattr(rule, f), re.I) for f in ("domain", "path", "app", "title")
                          if getattr(rule, f)}
    return _COMPILED[key]


def _matches(rule: Rule, fields: dict[str, str]) -> bool:
    pats = _compiled(rule)
    return all(p.search(fields.get(f) or "") for f, p in pats.items())


def label_hint(app: str | None, domain: str | None, path: str | None, title: str | None) -> tuple[str, str]:
    """Return (label, strength) for one segment. Unknown sources -> ("Other", "weak")."""
    fields = {k: v if isinstance(v, str) else "" for k, v in
              (("app", app), ("domain", domain), ("path", path), ("title", title))}
    for rule in KEYWORD_RULES:
        if _matches(rule, fields):
            return rule.label, rule.strength
    for rule in SOURCE_RULES:
        if _matches(rule, fields):
            # Quantum-team Slack counts as Quantum (taxonomy note).
            if rule.label == LMSG and re.search(r"slack", fields["app"] + fields["domain"], re.I) \
                    and re.search(r"\b(quantum|qcb|vqe|labs)\b", fields["title"], re.I):
                return LQC, STRONG
            return rule.label, rule.strength
    return LOTHER, WEAK
