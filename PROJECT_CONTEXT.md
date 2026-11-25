# Workflow Reconstruction from Activity Telemetry — Project Context

This file is the brief for any AI coding agent helping on this project. Read it fully before writing code. If something here conflicts with what you see in the actual data, trust the data, tell Aarush, and update this file.

---

## 1. TL;DR

- **What:** Take raw ActivityWatch telemetry (which apps, windows and websites a person used, with timestamps) and automatically reconstruct their **workflows**: what task they were doing, when it started and ended, and the ordered steps they took.
- **Two methods to build and compare:**
  1. **Sequence / process mining** (classical, deterministic, e.g. PM4Py, PrefixSpan)
  2. **LLM-based workflow inference** (structured outputs, every step must cite the source events it came from)
- **This week:** data preprocessing pipeline → both methods → Streamlit demo that shows both methods' output side by side for a chosen time range.
- **Not this week:** formal benchmarking. But design every output so benchmarking next week is trivial (see §7, common output schema).
- **Who:** Aarush Arya (UC Berkeley, ML & SWE intern at Orcapo), with weekly check-ins with Abhiraam (Orcapo).

---

## 2. Why this project exists

**Orcapo** (Orca21 Labs, Berkeley SkyDeck Batch 22) helps companies adopt AI by first understanding how their employees actually work: which tools they use, what steps make up their tasks, and which tasks repeat often enough to automate or assist with AI. Today that mapping is done with surveys and interviews, which is slow, expensive and relies on people's memory.

This project tests whether that mapping can be done **passively** from activity telemetry instead. Aarush is the test subject: ActivityWatch has been tracking his laptop, and he has hand-logged his real tasks as ground truth.

**Long-term goal (not this week):** an API Orcapo can plug into its product that takes a client employee's raw activity data and returns workflow summaries automatically. So write the pipeline as clean, importable modules with clear inputs and outputs, not as one-off notebook code.

---

## 3. Scope and phases

| Phase | When | What |
|---|---|---|
| Data collection | Done (ongoing lightly) | ActivityWatch running 24/7; tasks hand-logged in a spreadsheet |
| **Pipeline + methods + demo** | **This week** | Load → filter → clean → redact → Method A / Method B → common output → Streamlit |
| Benchmarking | Next week | Compare methods against ground truth on a held-out test set |

### Dev set vs. test set (important)

- The **17 tasks logged so far (T01–T17)** are the **development set**. Build, debug and tune on these only.
- Tasks logged **after** the methods are frozen will be the **test set** for next week's benchmark.
- Do not tune prompts, thresholds or rules on anything beyond T17. If new tasks appear in the task log, do not look at them for tuning unless Aarush says so.

---

## 4. The data

There are two inputs: the ActivityWatch export (raw telemetry) and the task log (ground truth).

### 4.1 ActivityWatch export (`aw-buckets-export.json`)

A single large JSON file from ActivityWatch's "Export all buckets as JSON". It is **cumulative**: it contains every event since ActivityWatch was installed (Nov 17, 2026), across all buckets.

**Structure** (verify against the real file before relying on it):

```json
{
  "buckets": {
    "<bucket_id>": {
      "id": "<bucket_id>",
      "created": "...",
      "type": "...",
      "client": "...",
      "hostname": "wifi-10-45-132-90.wifi.berkeley.edu",
      "data": {},
      "events": [
        {"timestamp": "2026-11-19T05:09:44.808000+00:00", "duration": 12598.363, "data": { ... }}
      ]
    }
  }
}
```

**Buckets and their `data` fields:**

| Bucket id prefix | What it records | `data` fields |
|---|---|---|
| `aw-watcher-window_<host>` | Foreground app and window title (macOS) | `app` (e.g. "Google Chrome", "Messages", "IntelliJ IDEA"), `title` |
| `aw-watcher-web-chrome_<host>` | Active Chrome tab (browser extension) | `url`, `title`, `audible`, `incognito`, `tabCount` |
| `aw-watcher-afk_<host>` | Whether the user is at the keyboard | `status`: `"afk"` or `"not-afk"` |
| JetBrains / editor bucket (may or may not exist) | IntelliJ project and language (aw-watcher-jetbrains plugin) | Unknown; inspect. Likely project and language, maybe file |

**Things you must handle:**

1. **Match buckets by prefix, not exact id.** The `<host>` suffix is the laptop's network hostname (currently `wifi-10-45-132-90.wifi.berkeley.edu`), which can change between Wi-Fi networks. The same watcher may appear under several bucket ids; concatenate them.
2. **Events are stored newest-first.** Sort by timestamp.
3. **Timestamps are UTC** (`+00:00`). The task log is in **Pacific time** (`America/Los_Angeles`, which is UTC−8 for the whole collection period since DST ended Nov 1). Convert with a proper tz library, never by hand. Example: a task logged 22:55–23:30 on Nov 18 Pacific is 06:55–07:30 UTC on Nov 19.
4. **`duration` is in seconds** (float). Event end = timestamp + duration.
5. **Lots of 0-second and sub-second events** from quickly switching tabs or clicking through lists (e.g. LinkedIn job listings change the URL on every click).
6. **Chrome is recorded twice:** once by the window watcher (`app: "Google Chrome"`, `title: "<page title>"`) and once by the web watcher (`url`, `title`). When the foreground app is Chrome, the web event is the richer one; merge or deduplicate.
7. **The web watcher keeps logging while the user is away.** An open tab will show as "active" during AFK periods. Use the AFK bucket to mask these out.
8. **Gaps** where all buckets are empty mean the laptop was asleep or ActivityWatch was not running.
9. **`loginwindow`** as the app = the macOS lock screen. Treat as idle.
10. **Messy URLs.** Many URLs carry tracking parameters (`trackingId`, `refId`, `eBP`, long tokens). Strip them. Keep meaningful parts (domain, path, and a small allowlist of query params such as `q`, `keywords`).

**Example events (real shapes, values shortened):**

```json
{"timestamp": "2026-11-19T07:26:24.289000+00:00", "duration": 1.374,
 "data": {"url": "https://www.linkedin.com/jobs/search-results/?currentJobId=...&keywords=summer%202027%20internship&...",
          "title": "Software Engineering Intern, AIML Platform - HP IQ | HP | LinkedIn",
          "audible": false, "incognito": false, "tabCount": 18}}

{"timestamp": "2026-11-18T21:12:09+00:00", "duration": 22.0,
 "data": {"app": "Google Chrome", "title": "SQL self-join query - Claude"}}

{"timestamp": "2026-11-19T03:39:13.502000+00:00", "duration": 5431.305,
 "data": {"status": "afk"}}
```

Typical window titles seen: `"61A Gleam"` (blank.page notes), `"35-Tables.key"` (cs61a.org slides), `"Joining Tables - YouTube"`, `"Inbox (817) - <email> - UC Berkeley Mail"`, `"Berkeley Roommates"` (Messages app chat).

### 4.2 Task log (ground truth)

Hand-logged by Aarush in a spreadsheet; export it to CSV (e.g. `data/task_log.csv`). One row per task.

| Column | Meaning |
|---|---|
| Task ID | `T01`, `T02`, ... |
| Date | Local date (Pacific) |
| Label | One of the labels in §4.3 |
| Start, End | Local clock times (Pacific), minute precision |
| Minutes | End − Start |
| Steps | Ordered, numbered steps (multi-line text). Present for most tasks, blank for some |
| Apps / sites | What he remembers using (free text) |
| Repeat of (Task ID) | If this task is another instance of an earlier workflow, the ID of the **first** instance |
| Interrupted? | TRUE if there were brief interruptions inside the task window |
| Notes | Free text: variants, continuations, context |
| Later correction | Corrections made after the fact (keep originals intact) |

**Conventions in the log:**

- Steps that repeat across instances of a workflow are written with **identical wording** so they can be matched exactly (e.g. "Build the random world/tile generation in IntelliJ, referring to the project spec and design doc"). Details that differ go in parentheses.
- **Continuations:** one sitting split by a break is logged as separate rows. Example: T15 (61A project) → T16 (7-min Entertainment break) → T17 (continuation of T15). Notes say this explicitly. Only the first part carries "Repeat of".
- **Short interruptions** (under ~1–2 min: Messages, DoorDash, a quick email) are not logged as rows; they are inside the task window and `Interrupted?` is TRUE.
- Untracked time: ActivityWatch logs 24/7 but only some time is hand-logged. **Unlogged time is "unlabeled", not "nothing happened".**

A full summary of T01–T17 is in the Appendix.

### 4.3 Label taxonomy

The label set (with definitions) may be given to the methods. That is the taxonomy, not ground truth.

| Label | What counts | Typical apps / sites |
|---|---|---|
| CS 61B Homework/Projects | CS 61B homework, labs, projects, including debugging and submitting | IntelliJ, Gradescope, Ed, CS 61B course site, terminal / Git |
| CS 61A Notes/Homework | CS 61A notes, homework, labs, projects | VS Code, blank.page, Gradescope, cs61a.org, YouTube, Ed, Claude |
| Quantum Computing @ Berkeley | Club project work on the VQE / LABS project, or general club work | Slack, Google Colab, Google Slides, arXiv, Claude |
| Dart (Startup) Work | Planning (Google Docs, Instagram, Claude) or development (VS Code) | VS Code, terminal, GitHub, Google Docs, Instagram, Claude, dartme.org, Supabase |
| Orcapo tracking work | Coding or research for this telemetry project | VS Code, terminal, GitHub, Claude, ActivityWatch |
| Internship application | Finding postings, tailoring resumes and cover letters, submitting | LinkedIn, Handshake, company career pages, Docs |
| Email / Slack / Social Media | Reading and replying to messages (quantum-team Slack counts as Quantum) | Gmail, Slack, LinkedIn messages, Instagram messages |
| Admin / planning | Calendar, course enrollment, to-do lists | Google Calendar, CalCentral, notes app |
| Entertainment | Videos, social media, games | YouTube, Reddit, Instagram, krillion.io, LinkedIn feed |
| Private | Anything not to be analyzed | Any |
| Other | Anything else | Any |

Note the deliberate ambiguity: the same app can map to different labels (YouTube = 61A lecture or Entertainment; Slack = Quantum or Email/Slack; Claude appears in almost every label). Context, titles and neighboring events must decide.

---

## 5. Privacy and data handling (non-negotiable)

The raw data is personal. It contains contact names and group-chat names (Messages), email addresses (Gmail tab titles), order URLs (DoorDash), job-search history and more.

- **Never commit raw data** to git. Put all data under `data/` and add it to `.gitignore`. Same for LLM response caches and any intermediate files containing titles or URLs.
- **Redact before anything leaves the machine.** Every string sent to an LLM API must go through the redaction step first (§6, stage 4).
- Don't print raw titles or URLs in logs beyond what's needed for local debugging.
- The `Private` label and any `incognito: true` events: drop them entirely before analysis.
- Redaction itself will be evaluated later (leak rate), so keep the redaction step a separate, testable function, and log what kinds of entities were replaced (counts, not values).

---

## 6. Pipeline architecture

```
aw-buckets-export.json ─┐
                        ├─► 1. Load & flatten ─► 2. Select time window ─► 3. Clean & merge ─► 4. Redact
task_log.csv ───────────┘                                                                      │
                                                                     ┌─────────────────────────┤
                                                                     ▼                         ▼
                                                         5A. Sequence mining          5B. LLM inference
                                                                     │                         │
                                                                     └──────► 6. Common output schema ◄──┘
                                                                                      │
                                                                                      ▼
                                                                             7. Streamlit demo
```

### Stage 1 — Load and flatten
- Parse the export; collect buckets by prefix; concatenate across hostnames.
- Produce one tidy DataFrame of events: `event_id` (stable, e.g. `source:index` or hash), `source` (`window` / `web` / `afk` / `editor`), `start_utc`, `end_utc`, `duration_s`, `app`, `title`, `url`, `domain`, `status`, plus `start_local` / `end_local` in America/Los_Angeles.
- Sort by start time. Cache to Parquet so the big JSON is parsed once.
- Load the task log into a DataFrame with `start_local` / `end_local` as tz-aware datetimes, and also UTC.

### Stage 2 — Select the time window to analyze
- **Do not feed the methods windows cut exactly at the logged task boundaries.** That leaks the ground truth: the method would be told where tasks start and end, making segmentation look perfect.
- Instead, select a larger context window: a whole day, a "session" (continuous activity between long AFK/sleep gaps), or a task window padded by a configurable margin (e.g. ±15–30 min, merging overlapping windows). Make this configurable.
- Keep the ground-truth intervals separately for display and (next week) scoring. Scoring will only count predictions that overlap logged windows; predictions in unlogged time are ignored, not penalized.

### Stage 3 — Clean and merge
Configurable, in this order:
1. **AFK masking:** clip or drop window/web events during `afk` periods. Treat `loginwindow` as idle.
2. **Chrome dedup:** where a window event is `Google Chrome` and a web event overlaps it, keep the web event's `url` + `title` (and the window `app`).
3. **Drop private:** incognito events, and anything inside a task labeled `Private`.
4. **Normalize:**
   - URLs: lowercase domain, drop tracking query params, keep path + allowlisted params.
   - Titles: strip suffixes like ` - Google Chrome`, ` - YouTube`, unread counts like `Inbox (16,221)`; collapse whitespace.
5. **Merge into activity segments:** consecutive events with the same `(app or domain, normalized title)` separated by small gaps (e.g. < 5 s) become one segment. Drop or absorb segments shorter than a threshold (e.g. < 2 s) into neighbors.
6. **Token for mining:** derive a compact activity token per segment, e.g. `domain` for web (`cs61a.org`, `youtube.com`, `claude.ai`) or `app` for native (`VS Code`, `IntelliJ IDEA`, `Messages`), optionally refined by a resource hint (`cs61a.org/slides`, `youtube.com/watch`).

Output: a cleaned segment table with `segment_id`, `start`, `end`, `duration_s`, `app`, `domain`, `title_norm`, `url_norm`, `token`, `source_event_ids` (list).

### Stage 4 — Redact
- Replace emails, phone numbers, person names (e.g. Microsoft Presidio `PERSON`), and long ID-like tokens in URLs with placeholders (`<EMAIL>`, `<PERSON>`, `<ID>`).
- Messages chat names, Instagram DM names, Gmail addresses are the main known risks.
- Return the redacted segment table plus a redaction report (counts per entity type).
- Required before Stage 5B. Optional but recommended before display in Streamlit.

### Stage 5A — Method A: sequence / process mining
Deterministic, no LLM. Suggested approach (adjust as needed):
1. **Segmentation into tasks:** split the token sequence into task segments using context-switch signals: long idle gaps, plus change points where the set of active tokens in a sliding window changes sharply (e.g. Jaccard similarity of token sets between adjacent windows below a threshold).
2. **Step extraction:** within each task segment, collapse rapid back-and-forth into higher-level steps. Example: alternating `cs61a.org ↔ blank.page ↔ claude.ai` every few seconds is one step ("read slides while taking notes"), not 50 steps. Use run-length grouping or a dwell-time threshold.
3. **Pattern mining:** build a directly-follows graph per segment with PM4Py (case = task segment, activity = token, timestamp = segment start). Mine frequent sequential patterns across segments (PrefixSpan or PM4Py variants) to detect **repeated workflows**.
4. **Labeling:** a simple, transparent mapping from tokens to labels (keyword/domain rules derived from the label taxonomy in §4.3, plus title keywords like "61A" / "61B" / "Gradescope"). Ambiguous tokens (YouTube, Claude, Slack) are resolved by the majority of non-ambiguous tokens in the segment.
5. **Repeat detection:** compare segments' step sequences (e.g. normalized edit distance or pattern overlap) and link similar ones.

### Stage 5B — Method B: LLM workflow inference
1. **Input:** the redacted, cleaned segment table for a window, rendered as compact text lines, one per segment, each with its `segment_id`:
   ```
   [S0412] 21:03:10–21:06:55 | VS Code | <file name> 
   [S0413] 21:06:55–21:07:20 | cs61a.org | Project description
   [S0414] 21:07:20–21:09:02 | edstem.org | <post title>
   ```
   Plus the label taxonomy with definitions.
2. **Output:** structured JSON (use the provider's structured-output / JSON-schema mode, validated with Pydantic) matching the common schema in §7.
3. **Grounding rule:** every task and every step must cite `evidence_segment_ids` that exist in the input and fall inside that task's time range. After the call, **validate**: drop or flag any step citing nonexistent or out-of-range IDs. Count these as hallucinated steps.
4. **Long windows:** chunk by session or time with a small overlap, then merge tasks that span chunk boundaries.
5. **Determinism and cost:** temperature 0; cache responses on disk keyed by a hash of (prompt template version, model, input text). Keep the model and provider configurable (Aarush has access to LLM APIs; don't hard-code one vendor in the core logic). Read API keys from environment variables only.
6. Keep prompt templates in versioned files (`prompts/v1_*.txt`), not inline strings, so changes are trackable.

### (Optional) Baseline
A trivial baseline is useful later: label each segment by its dominant app/domain via the same rules as 5A, with no segmentation or steps. Cheap to add now, since benchmarking will want it.

---

## 7. Common output schema (both methods must produce this)

```python
class Step(BaseModel):
    order: int
    description: str                    # e.g. "Read lecture slides on cs61a.org while taking notes in blank.page"
    start: datetime                     # tz-aware
    end: datetime
    evidence_segment_ids: list[str]     # must exist in input

class PredictedTask(BaseModel):
    task_id: str                        # method-local, e.g. "A-003" / "B-003"
    label: str                          # one of the taxonomy labels
    start: datetime
    end: datetime
    summary: str                        # one sentence
    steps: list[Step]
    repeat_of: str | None               # task_id of an earlier similar task, if detected
    confidence: float | None
    evidence_segment_ids: list[str]

class MethodRun(BaseModel):
    method: str                         # "sequence_mining" | "llm" | "baseline"
    method_version: str
    window_start: datetime
    window_end: datetime
    params: dict                        # thresholds, model name, prompt version, etc.
    tasks: list[PredictedTask]
    warnings: list[str]                 # e.g. hallucinated citations dropped
```

Save each run as JSON under `outputs/runs/<method>/<window_id>.json`. Next week's benchmark will read these files and compare them against the task log (label match, time-chunk overlap, steps recovered, repeat detection, hallucinated-step rate, redaction leak rate). Don't build the benchmark now, but don't make it hard.

---

## 8. Streamlit demo

**Start with a proof of concept** (requested by Abhiraam): paste a small chunk of raw data as text/markdown, run both methods, show the outputs. Then grow it into:

1. **Input:** upload `aw-buckets-export.json` (or use a cached Parquet) and optionally `task_log.csv`. Pick a date, a session, or a logged task (which selects a padded context window, per §6 stage 2).
2. **Raw timeline:** interactive timeline (e.g. Plotly) of cleaned segments colored by app/domain, with AFK periods shaded.
3. **Side-by-side results:** Method A | Method B columns, each showing predicted tasks (label, time range, summary) and their ordered steps. Clicking a step highlights its evidence segments on the timeline.
4. **Ground-truth overlay (toggle):** show the logged tasks as a separate lane for visual comparison. Display only; never passed to the methods.
5. **Diagnostics:** number of events in → segments after cleaning → tokens; redaction counts; LLM warnings (dropped citations); runtime and token cost for the LLM.

Keep heavy computation in the pipeline modules; Streamlit should only call them, with `st.cache_data` for loading and parsing.

---

## 9. Suggested repo layout

```
workflow-recon/
├── PROJECT_CONTEXT.md        # this file
├── README.md
├── pyproject.toml / requirements.txt
├── config.yaml               # thresholds, padding, model name, paths
├── .gitignore                # data/, outputs/, .cache/, .env
├── data/                     # NOT committed: aw-buckets-export.json, task_log.csv, *.parquet
├── prompts/
│   └── v1_workflow_inference.txt
├── src/workflow_recon/
│   ├── load.py               # stage 1
│   ├── windows.py            # stage 2
│   ├── clean.py              # stage 3
│   ├── redact.py             # stage 4
│   ├── schema.py             # §7 Pydantic models
│   ├── methods/
│   │   ├── sequence_mining.py
│   │   ├── llm_inference.py
│   │   └── baseline.py
│   └── io.py                 # save/load MethodRun JSON
├── app/
│   └── streamlit_app.py
├── tests/
│   └── fixtures/             # tiny synthetic export + task log (no real personal data)
└── outputs/                  # NOT committed
```

**Stack:** Python 3.11+, pandas, pyarrow, pydantic, pm4py, prefixspan (or equivalent), presidio-analyzer + presidio-anonymizer (or a regex-first redactor if Presidio is too heavy), plotly, streamlit, and an LLM SDK behind a thin wrapper.

---

## 10. Conventions for the agent

- **Inspect before assuming.** On first run, print the bucket ids, event counts per bucket, the `data` keys per bucket, and time range. The export structure above is from screenshots; confirm it.
- **Small, testable functions** with type hints. Pure functions for cleaning and redaction; tests use the synthetic fixture, never real data.
- **Everything configurable** (thresholds, padding, model, prompt version) via `config.yaml`. Record the config inside every `MethodRun`.
- **Timezones:** keep tz-aware datetimes everywhere. Never compare naive and aware times. Never hand-subtract 8 hours.
- **Never pass ground truth into a method.** The task log's labels, start/end times, steps, apps, repeat links and notes are for display and future scoring only. The label taxonomy (§4.3) is allowed as method input.
- **Don't over-tune to T01–T17.** Prefer simple, explainable rules and generic prompts over special cases for specific tasks.
- Ask Aarush before making product decisions (e.g. changing the output schema, dropping a method, adding a paid service).

---

## 11. Known open questions

- Does a JetBrains/editor bucket exist in the export, and what fields does it have?
- Are there multiple hostname variants (bucket ids) in the export?
- Exact padding / session definition for context windows (start with: sessions split by ≥ 20 min AFK/no-data gaps; padded task windows ±20 min).
- Step granularity: how coarse should a "step" be? Target the granularity of the task log's steps (3–5 steps per task, each a few minutes or more).
- How Method A should name steps in human-readable form (template from tokens + titles is fine for now).

---

## 12. This week's milestones

1. Loader + flattening + Parquet cache; sanity printout of buckets and ranges.
2. Window selection + cleaning + segment table; eyeball it for T01 (61A lecture review) and T15–T17 (61A project with a break).
3. Redaction function + report.
4. Common schema + Method A v1.
5. Method B v1 with citation validation and caching.
6. Streamlit proof of concept (pasted chunk) → full demo (upload, timeline, side by side, GT overlay toggle).

Good first sanity checks:
- **T01 (Nov 18, 12:30–13:32):** expect rapid alternation `cs61a.org` (35-Tables slides) ↔ `blank.page` ("61A Gleam") ↔ `claude.ai` ("SQL self-join query"), then YouTube ("Joining Tables", "Cross Joins") from ~13:14; short DoorDash/Gmail/Messages interruptions around 13:13–13:18.
- **T15 → T16 → T17 (Nov 23, 21:03–21:56):** VS Code + cs61a.org + Ed, then a 7-minute krillion.io break, then VS Code + Gradescope. A good method finds the break and links T15 and T17 as one session.

---

## Appendix — Development set (T01–T17)

All times Pacific (America/Los_Angeles). "Rep." = Repeat of. "Int." = Interrupted.

| ID | Date | Label | Start–End | Min | Rep. | Int. | Notes |
|---|---|---|---|---|---|---|---|
| T01 | Wed 11/18 | CS 61A Notes/Homework | 12:30–13:32 | 62 | | ✓ | Lecture review: slides + notes + Claude + videos. Interruptions (DoorDash, Messages) |
| T02 | Wed 11/18 | Orcapo tracking work | 13:33–13:45 | 12 | | ✓ | ActivityWatch, Claude. No steps |
| T03 | Wed 11/18 | Internship application | 22:55–23:30 | 35 | | | Handshake → LinkedIn → company sites |
| T04 | Thu 11/19 | Quantum Computing @ Berkeley | 16:05–16:40 | 35 | | ✓ | Slides + verify against code + run in Colab |
| T05 | Thu 11/19 | Internship application | 19:48–20:07 | 19 | T03 | | Same steps as T03 |
| T06 | Thu 11/19 | Dart (Startup) Work | 21:39–23:04 | 85 | | ✓ | Google Doc → VS Code/Supabase → test dartme.org → Instagram chat loop |
| T07 | Fri 11/20 | CS 61A Notes/Homework | 11:37–11:52 | 15 | T01 | | Variant: videos + notes, no slides |
| T08 | Sat 11/21 | Internship application | 20:52–21:08 | 16 | T03 | | Same steps as T03 |
| T09 | Sat 11/21 | CS 61A Notes/Homework | 21:18–22:24 | 66 | | | Homework: cs61a.org → VS Code → Gradescope (different workflow from T01) |
| T10 | Sun 11/22 | Email / Slack / Social Media | 14:47–14:51 | 4 | | ✓ | Gmail, LinkedIn, Instagram, Slack. Very short |
| T11 | Sun 11/22 | CS 61B Homework/Projects | 15:00–16:00 | 60 | | | Project 5B: spec, design doc, IntelliJ setup, starter code, build |
| T12 | Sun 11/22 | CS 61B Homework/Projects | 20:35–21:36 | 61 | T11 | ✓ | Second session of 5B (continuation-style repeat), no setup; commit + push |
| T13 | Sun 11/22 | Other | 21:53–22:09 | 16 | | | Finding and booking a haircut (Google, Gemini, Booksy) |
| T14 | Mon 11/23 | CS 61A Notes/Homework | 20:28–20:52 | 24 | T01 | | Lecture review: "37-Databases" slides + notes + videos |
| T15 | Mon 11/23 | CS 61A Notes/Homework | 21:03–21:33 | 30 | T09 | ✓ | 61A project (AI-allowed), VS Code + cs61a.org + Ed. Split by break T16 |
| T16 | Mon 11/23 | Entertainment | 21:34–21:41 | 7 | | | Break: krillion.io |
| T17 | Mon 11/23 | CS 61A Notes/Homework | 21:43–21:56 | 13 | | | Continuation of T15; submit to Gradescope |

**Repeated workflows in the dev set:**
- Internship application: T03, T05, T08 (identical steps)
- 61A lecture review: T01, T07 (variant), T14
- 61A homework/project: T09, T15 (+ T17 continuation)
- 61B Project 5B sessions: T11, T12

**Ground-truth steps** (for display and future scoring only, never method input):

- **T01:** 1. Read the lecture slides on cs61a.org ("35-Tables") 2. Take notes in blank.page (while reading the slides) 3. Ask Claude about SQL self-joins 4. Watch the lecture videos on YouTube ("Tables v2" playlist)
- **T03 / T05 / T08:** 1. Search Handshake for postings and apply 2. Search LinkedIn for postings 3. For each posting that interests me, fill in the external application on the company's website
- **T04:** 1. Rewrite the slide on our group's implementation in Google Slides 2. Verify the slide against the code 3. Run the code in Google Colab to make sure it works
- **T06:** 1. Review the proposed changes in the Google Doc 2. Build the changes in VS Code (including any Supabase database changes) 3. Test on dartme.org 4. Check the Instagram group chat for new changes, update the Google Doc, and repeat if needed
- **T07:** 1. Watch the lecture videos on YouTube 2. Take notes in blank.page (alternating with the videos)
- **T09:** 1. Read the homework description on cs61a.org and download the zip file 2. Work on the homework in VS Code, referring back to the problems on cs61a.org 3. Submit on Gradescope
- **T10:** 1. Look through emails 2. Look through LinkedIn messages 3. Look through Instagram messages 4. Look through Slack (replying where needed)
- **T11:** 1. Read the Project 5B spec on the CS 61B course website 2. Read our group's design doc 3. Open the project in IntelliJ / terminal 4. Read through the starter code to understand how it works 5. Build the random world/tile generation in IntelliJ, referring to the project spec and design doc
- **T12:** 1. Build the random world/tile generation in IntelliJ, referring to the project spec and design doc 2. Commit and push changes to GitHub
- **T13:** 1. Compare local haircut places (social media, reviews) 2. Find haircuts that would work best 3. Check availability on booking sites 4. Book an appointment
- **T14:** 1. Read the lecture slides on cs61a.org ("37-Databases") 2. Take notes in blank.page (while reading the slides) 3. Watch the lecture videos on YouTube ("Databases" playlist), taking notes in blank.page
- **T15:** 1. Read the project description on cs61a.org 2. Work on the project in VS Code, referring back to cs61a.org and Ed posts
- **T17:** 1. Continue working on the project in VS Code, referring back to cs61a.org 2. Submit to Gradescope
