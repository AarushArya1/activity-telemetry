"""Generate a synthetic ActivityWatch export + task log for testing (no real personal data).

Writes:
  tests/fixtures/fake_export.json     (ActivityWatch "export all buckets" shape)
  tests/fixtures/fake_task_log.csv    (same columns as the real task log)

The export reproduces the quirks listed in PROJECT_CONTEXT.md §4.1:
  - buckets matched by prefix, with the hostname changing between days
  - events stored newest-first; UTC timestamps; durations in seconds
  - Chrome logged twice (window watcher + web watcher), tracking params in URLs
  - sub-second events (clicking through job listings)
  - the web watcher (and window watcher) keep logging during AFK
  - loginwindow (lock screen), incognito tabs
  - fake PII: emails, a phone number, chat names, order ids, a social profile
  - a JetBrains editor bucket

Usage:  python scripts/make_fake_data.py [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

TRACK = "utm_source=newsletter&trackingId=Zx81kQ0pLm2Vb7Yt&refId=a1b2c3d4e5f6g7h8"


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.t: datetime | None = None
        self.window: list[dict] = []
        self.web: list[dict] = []
        self.editor: list[dict] = []
        self.spans: list[tuple[datetime, datetime, str]] = []  # (start, end, afk|not-afk)
        self.host = "host-a.example.net"
        self.events_by_host: dict[str, dict[str, list]] = {}

    # ---- cursor
    def at(self, day: str, hhmm: str) -> "Builder":
        self.t = datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=TZ)
        return self

    def _emit(self, kind: str, start: datetime, dur: float, data: dict) -> None:
        bucket = self.events_by_host.setdefault(self.host, {"window": [], "web": [], "afk": [], "editor": []})
        bucket[kind].append({"timestamp": iso(start), "duration": round(dur, 3), "data": data})

    def _active(self, start: datetime, end: datetime) -> None:
        self.spans.append((start, end, "not-afk"))

    # ---- activities
    def act(self, app: str, title: str, dur: float, url: str | None = None, incognito: bool = False,
            audible: bool = False, editor: dict | None = None) -> "Builder":
        start = self.t
        self._emit("window", start, dur, {"app": app, "title": title})
        if url is not None:
            # web watcher starts a fraction later and is split into a couple of heartbeats
            off = self.rng.uniform(0.05, 0.4)
            self._emit("web", start + timedelta(seconds=off), max(dur - off, 0.0),
                       {"url": url, "title": title, "audible": audible, "incognito": incognito,
                        "tabCount": self.rng.randint(5, 25)})
        if editor:
            self._emit("editor", start, dur, editor)
        self._active(start, start + timedelta(seconds=dur))
        self.t = start + timedelta(seconds=dur + self.rng.uniform(0.0, 1.0))
        return self

    def chrome(self, title: str, url: str, dur: float, **kw) -> "Builder":
        return self.act("Google Chrome", title, dur, url=url, **kw)

    def alternate(self, items: list[tuple], total: float, dwell: tuple[float, float]) -> "Builder":
        """Rapid back-and-forth between items for `total` seconds."""
        end = self.t + timedelta(seconds=total)
        i = 0
        while self.t < end:
            d = min(self.rng.uniform(*dwell), (end - self.t).total_seconds())
            if d < 0.5:
                break
            item = items[i % len(items)]
            if item[0] == "chrome":
                self.chrome(item[1], item[2], d)
            else:
                self.act(item[0], item[1], d, editor=item[2] if len(item) > 2 else None)
            nxt = i + 1 if self.rng.random() < 0.8 else self.rng.randrange(len(items))
            while len(items) > 1 and items[nxt % len(items)] == item:
                nxt += 1
            i = nxt
        return self

    def clicks(self, title_fmt: str, url_fmt: str, n: int) -> "Builder":
        """Sub-second events, e.g. clicking through LinkedIn job listings."""
        for k in range(n):
            jid = 4000000000 + self.rng.randint(0, 99999999)
            self.chrome(title_fmt.format(k=k), url_fmt.format(jid=jid), self.rng.uniform(0.2, 1.6))
        return self

    def away(self, minutes: float, keep_logging: tuple | None = None) -> "Builder":
        """AFK. Optionally the watchers keep reporting the open tab (the quirk to mask)."""
        start = self.t
        dur = minutes * 60
        self.spans.append((start, start + timedelta(seconds=dur), "afk"))
        if keep_logging:
            title, url = keep_logging
            self._emit("window", start, dur, {"app": "Google Chrome", "title": title})
            self._emit("web", start, dur, {"url": url, "title": title, "audible": False,
                                           "incognito": False, "tabCount": 12})
        self.t = start + timedelta(seconds=dur)
        return self

    def lock(self, minutes: float) -> "Builder":
        start = self.t
        dur = minutes * 60
        self._emit("window", start, dur, {"app": "loginwindow", "title": ""})
        self.spans.append((start, start + timedelta(seconds=dur), "afk"))
        self.t = start + timedelta(seconds=dur)
        return self

    # ---- output
    def flush_afk(self) -> None:
        """Turn spans into afk-bucket events for the current host (merging adjacent states)."""
        spans = sorted(self.spans)
        merged: list[list] = []
        for s, e, st in spans:
            if merged and merged[-1][2] == st and (s - merged[-1][1]).total_seconds() < 60:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, st])
        for s, e, st in merged:
            self._emit("afk", s, (e - s).total_seconds(), {"status": st})
        self.spans = []

    def export(self) -> dict:
        buckets = {}
        kinds = {"window": ("aw-watcher-window", "currentwindow", "aw-watcher-window"),
                 "web": ("aw-watcher-web-chrome", "web.tab.current", "aw-client-web"),
                 "afk": ("aw-watcher-afk", "afkstatus", "aw-watcher-afk"),
                 "editor": ("aw-watcher-jetbrains", "app.editor.activity", "aw-watcher-jetbrains")}
        for host, by_kind in self.events_by_host.items():
            for kind, events in by_kind.items():
                if not events:
                    continue
                prefix, btype, client = kinds[kind]
                bid = f"{prefix}_{host}"
                buckets[bid] = {"id": bid, "created": "2026-11-30T08:00:00+00:00", "type": btype,
                                "client": client, "hostname": host, "data": {},
                                # ActivityWatch stores newest first
                                "events": sorted(events, key=lambda e: e["timestamp"], reverse=True)}
        return {"buckets": buckets}


# ------------------------------------------------------------ scenario

SLIDES = ("chrome", "38-Aggregation.key", "https://cs61a.org/assets/slides/38-Aggregation.pdf")
NOTES = ("chrome", "61A Lecture Notes", "https://blank.page/?" + TRACK)
CLAUDE = ("chrome", "GROUP BY with HAVING - Claude", "https://claude.ai/chat/3f2c9a1e-77b0-4d5e-9c1a-0b9e8d7f6a5c")
VID1 = ("chrome", "Aggregation Functions - YouTube", "https://www.youtube.com/watch?v=Qw3rTy7uIoP&list=PLx0sYbCqOb8T")
VID2 = ("chrome", "Grouping Rows - YouTube", "https://www.youtube.com/watch?v=Zx9cVb2nMaS&list=PLx0sYbCqOb8T")
HW = ("chrome", "Homework 9 | CS 61A Fall 2026", "https://cs61a.org/hw/hw09/")
VSC = ("Code", "hw09.py — hw09")
ED = ("chrome", "Question about Q3 – Ed Discussion", "https://edstem.org/us/courses/81234/discussion/7733221")


def lecture_review(b: Builder, minutes_slides: float, minutes_videos: float, interrupt: bool) -> None:
    b.alternate([SLIDES, NOTES, SLIDES, CLAUDE, NOTES], minutes_slides * 60, (8, 70))
    if interrupt:
        b.act("Messages", "Jordan Rivera", 45)
        b.chrome("Order Details | DoorDash", "https://www.doordash.com/orders/9f8e7d6c5b4a39281706/?" + TRACK, 35)
    b.alternate([VID1, NOTES, VID2, NOTES], minutes_videos * 60, (30, 240))


def internship(b: Builder) -> None:
    b.alternate([("chrome", "Job Search | Handshake",
                  "https://app.joinhandshake.com/stu/postings?page=1&per_page=25&sort_direction=desc&query=software%20intern"),
                 ("chrome", "Software Engineering Intern - Summer 2027 | Handshake",
                  "https://app.joinhandshake.com/jobs/98765432?searchId=abcd1234efgh5678")], 8 * 60, (20, 90))
    b.chrome("summer 2027 internship Jobs | LinkedIn",
             "https://www.linkedin.com/jobs/search/?keywords=summer%202027%20internship&" + TRACK, 40)
    b.clicks("Software Engineering Intern {k} - Example Co | LinkedIn",
             "https://www.linkedin.com/jobs/search/?currentJobId={jid}&keywords=summer%202027%20internship&eBP=CwEAAAGXyZ", 25)
    b.alternate([("chrome", "Software Engineering Intern | LinkedIn",
                  "https://www.linkedin.com/jobs/view/4123456789/?refId=Q1w2E3r4T5y6U7i8"),
                 ("chrome", "summer 2027 internship Jobs | LinkedIn",
                  "https://www.linkedin.com/jobs/search/?keywords=summer%202027%20internship")], 5 * 60, (20, 80))
    b.alternate([("chrome", "Apply - Software Intern - Example Corp", "https://careers.examplecorp.com/apply/req-20931?src=linkedin"),
                 ("chrome", "Job Application for Software Engineer Intern at Sample Labs",
                  "https://boards.greenhouse.io/samplelabs/jobs/5512345678?gh_src=xyz123")], 10 * 60, (40, 180))


def messaging(b: Builder) -> None:
    b.chrome("Inbox (1,234) - alex.example@berkeley.edu - UC Berkeley Mail", "https://mail.google.com/mail/u/0/#inbox", 70)
    b.chrome("Re: lunch? call me at (510) 555-0142 - alex.example@berkeley.edu - UC Berkeley Mail",
             "https://mail.google.com/mail/u/0/#inbox/FMfcgzQXJWDsKmbLqTvRsPwNz", 40)
    b.chrome("Instagram • Chats", "https://www.instagram.com/direct/t/17845123456789012/", 60)
    b.act("Slack", "general (Channel) - Example Workspace - Slack", 55)
    b.chrome("Sam Example (@samexample) • Instagram photos and videos", "https://www.instagram.com/samexample/", 20)


def homework(b: Builder, minutes: float, submit: bool) -> None:
    b.chrome(HW[1], HW[2] + "?" + TRACK, 150)
    b.chrome("hw09.zip", "https://cs61a.org/hw/hw09/hw09.zip", 3)
    b.alternate([VSC, VSC, HW, VSC, ED, VSC], minutes * 60, (20, 200))
    if submit:
        b.chrome("Homework 9 | Gradescope", "https://www.gradescope.com/courses/912345/assignments/5432109/submissions", 90)
        b.chrome("Submission Results | Gradescope", "https://www.gradescope.com/courses/912345/assignments/5432109/submissions/301234567", 45)


def project_61b(b: Builder, setup: bool) -> None:
    ij = "IntelliJ IDEA"
    ed = {"project": "proj5", "language": "Java", "file": "WorldGenerator.java"}
    if setup:
        b.chrome("Project 5B: World Generation | CS 61B Fall 2026", "https://fa26.datastructur.es/projects/proj5b/", 6 * 60)
        b.chrome("Proj 5B Design Doc - Google Docs", "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit", 4 * 60)
        b.act(ij, "proj5 – Main.java", 5 * 60, editor={**ed, "file": "Main.java"})
    b.alternate([(ij, "proj5 – WorldGenerator.java", ed),
                 ("chrome", "Project 5B: World Generation | CS 61B Fall 2026", "https://fa26.datastructur.es/projects/proj5b/"),
                 (ij, "proj5 – WorldGenerator.java", ed),
                 ("chrome", "Proj 5B Design Doc - Google Docs", "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit")],
                25 * 60, (40, 300))
    b.act("Terminal", "proj5 — git push — 80×24", 60)
    b.chrome("Commits · group-42/proj5", "https://github.com/group-42/proj5/commits/main", 40)


def build(seed: int) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    b = Builder(rng)
    log: list[dict] = []

    def logrow(tid, day, label, start, end, steps="", repeat="", interrupted=False, notes=""):
        log.append({"Task ID": tid, "Date": day, "Label": label, "Start": start, "End": end,
                    "Minutes": "", "Steps": steps, "Apps / sites": "", "Repeat of (Task ID)": repeat,
                    "Interrupted?": "TRUE" if interrupted else "FALSE", "Notes": notes, "Later correction": ""})

    hm = lambda: b.t.strftime("%H:%M")  # noqa: E731

    # ---------------- Day 1 (host A)
    d1 = "2026-12-01"
    b.host = "host-a.example.net"
    b.at(d1, "08:55").chrome("Google Calendar - Week of November 30, 2026", "https://calendar.google.com/calendar/u/0/r/week", 90)
    b.at(d1, "09:00")
    s0 = hm(); lecture_review(b, 22, 16, interrupt=True)
    logrow("F01", d1, "CS 61A Notes/Homework", s0, hm(), interrupted=True)
    b.away(25, keep_logging=VID2[1:])                     # watchers keep logging while away
    b.at(d1, "10:05")
    s0 = hm(); internship(b)
    logrow("F02", d1, "Internship application", s0, hm())
    s0 = hm(); messaging(b)
    logrow("F03", d1, "Email / Slack / Social Media", s0, hm())
    b.lock(30)
    b.at(d1, "13:00")
    s0 = hm(); homework(b, 36, submit=False)
    logrow("F04", d1, "CS 61A Notes/Homework", s0, hm())
    s0 = hm(); b.chrome("krillion.io", "https://krillion.io/play?room=Ab12Cd34Ef56", 7 * 60, audible=True)
    logrow("F05", d1, "Entertainment", s0, hm())
    s0 = hm()
    b.alternate([VSC, HW, VSC], 13 * 60, (30, 200))
    b.chrome("Homework 9 | Gradescope", "https://www.gradescope.com/courses/912345/assignments/5432109/submissions", 90)
    logrow("F06", d1, "CS 61A Notes/Homework", s0, hm(), notes="Continuation of F04")
    b.chrome("secret stuff", "https://example.org/private", 180, incognito=True)
    b.lock(10)
    b.flush_afk()

    # ---------------- Day 2 (host B — hostname changed)
    d2 = "2026-12-02"
    b.host = "host-b.example.net"
    b.at(d2, "10:00")
    s0 = hm(); internship(b)
    logrow("F07", d2, "Internship application", s0, hm(), repeat="F02")
    b.away(40)
    b.at(d2, "15:00")
    s0 = hm(); project_61b(b, setup=True)
    logrow("F08", d2, "CS 61B Homework/Projects", s0, hm())
    b.away(30)
    b.at(d2, "20:00")
    s0 = hm(); lecture_review(b, 10, 10, interrupt=False)
    logrow("F09", d2, "CS 61A Notes/Homework", s0, hm(), repeat="F01")
    b.at(d2, "21:00")
    s0 = hm()
    b.alternate([("chrome", "r/programming", "https://www.reddit.com/r/programming/"),
                 ("chrome", "Funny Cat Compilation - YouTube", "https://www.youtube.com/watch?v=CaT123dOg45")], 12 * 60, (40, 200))
    logrow("F10", d2, "Entertainment", s0, hm())
    b.flush_afk()
    return b.export(), log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    export, log = build(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fake_export.json").write_text(json.dumps(export, indent=1))
    with open(OUT / "fake_task_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    n = sum(len(b["events"]) for b in export["buckets"].values())
    print(f"wrote {n} events in {len(export['buckets'])} buckets, {len(log)} logged tasks -> {OUT}")


if __name__ == "__main__":
    main()
