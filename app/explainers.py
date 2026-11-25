"""Text for the two explainer popups in the Streamlit app."""

PROJECT_OVERVIEW = """
### The idea
Companies want to know **how their people actually work**: which tools they use, what steps a task
takes, and which tasks repeat often enough to automate. Today that's found out with surveys and
interviews, which are slow and rely on memory.

This project tests whether we can figure it out **passively** instead, from activity telemetry.
[ActivityWatch](https://activitywatch.net) runs on the laptop and records which app or website was in
front, the window/page title, and whether the person was at the keyboard. From that raw stream we try to
reconstruct the **workflow**: what tasks happened, when each started and ended, and the ordered steps.

### What you do here
1. Upload an ActivityWatch export (about a day of data).
2. Pick a method.
3. See the workflow it reconstructs: tasks, labels, time ranges and steps.

### What the methods can see
- The **cleaned, redacted activity** for the day: one line per "segment" with a start/end time,
  the app or website, the page title and the cleaned URL. Away-from-keyboard time is removed.
- The **list of task labels** (e.g. "CS 61A Notes/Homework", "Internship application") and what each covers.
- Their **own settings** (thresholds, prompt, model) and their **own earlier results** (to spot repeats).

### What they never see
- Anything from the hand-written **task log** (the ground truth): no start/end times, no labels,
  no steps, no "repeat of" links. The methods have to find task boundaries on their own.
- **Unredacted text.** Emails, phone numbers, chat names, profile handles and long IDs are replaced with
  placeholders like `<EMAIL>` or `<CHAT>` before anything is analyzed, and before anything goes to an LLM.

### The goal
Given a whole day, each method should:
- **split** it into tasks and give each a **label**,
- describe each task as a few **ordered steps** (the way a person would write a to-do list),
- **cite evidence**: every step points to the activity segments it came from,
- notice **repeats**: the same workflow done again, or the same task resumed after a break.

Later, both methods get scored against the hand-logged tasks, to see which reconstructs workflows better.
"""

CORE_CONCEPTS = """
### The pipeline

```
ActivityWatch JSON → 1 Load → 2 Pick window → 3 Clean & merge → 4 Redact → Method A or B → workflow
```

| Stage | What happens | Code |
|---|---|---|
| 1 Load | Flatten all buckets (window, web, AFK, editor) into one table of events, UTC → Pacific | `load.py` |
| 2 Window | Select one local day (default) or a session | `windows.py` |
| 3 Clean | Mask AFK/lock-screen time; merge Chrome's two recordings; drop incognito; strip tracking params and title noise; merge events into **segments**; give each a short **token** like `cs61a.org` or `VS Code` | `clean.py` |
| 4 Redact | Replace emails, phones, chat names, profile handles, long IDs with placeholders; count what was replaced | `redact.py` |

---

### Method A: sequence / process mining (no AI)
Deterministic: the same input always gives the same output, and every decision is a readable rule.

1. **Label hints.** Simple rules map sites/apps to labels (`cs61a.org` → 61A, `joinhandshake.com` → Internship).
   Some are *ambiguous* (YouTube, Claude, VS Code): they take the label of the work around them.
2. **Interruptions.** Label runs under ~2.5 min (a quick text, a DoorDash check) are folded into the task they interrupt.
3. **Task boundaries.** Idle gaps end tasks. Inside a same-label stretch, a **change point** splits it when
   there's a pause *and* the mix of sites before vs. after is very different
   (**Jaccard similarity**: overlap ÷ union of the two site mixes).
4. **Steps.** Time is cut into 1-minute bins. Neighbouring bins with a similar site mix become one step, so
   rapid back-and-forth (slides ↔ notes ↔ Claude) is *one* step, not fifty. Steps are capped at 5.
5. **Process mining.** A **directly-follows graph** (which site follows which, how often; built with
   **PM4Py**) and **PrefixSpan** (finds step sequences that recur across tasks).
6. **Repeats.** Each task's step sequence is compared with earlier ones using **edit distance**
   (how many insert/delete/replace operations turn one sequence into the other) plus site-mix similarity.

Code: `methods/sequence_mining.py`, `methods/common.py`, `labels.py`

---

### Method B: LLM inference
1. **Render** the redacted segments as compact text, one line each, with an ID:
   `[S0412] 21:03:10–21:06:55 (3m45s) | VS Code | hw09.py`
2. **Prompt** (`prompts/v1_workflow_inference.txt`) gives the model the label list and the rules: find tasks,
   label them, write 2–5 steps, and **cite segment IDs** for every step.
3. **Structured output.** The model must answer in JSON matching a fixed **JSON schema**, so the output
   can be parsed reliably instead of being free text.
4. **Grounding check.** After the call, every cited ID is verified: it must exist and lie inside its task.
   Invented or out-of-range citations are dropped and counted as **hallucinations**.
5. **Long days** are split into chunks (preferably at long breaks) and tasks that straddle a chunk
   boundary are merged back together.
6. **Cache.** Responses are saved to disk keyed by a hash of the exact prompt, so re-running costs nothing
   and gives identical output.
7. **Privacy guard.** The method refuses unredacted input and re-scans the final prompt for emails/phones.

The model provider is swappable (Gemini by default, Claude optional) behind one small interface.
Code: `methods/llm_inference.py`, `llm/`

---

### Both methods produce the same output
A `MethodRun` → list of `PredictedTask` (label, start, end, summary, steps, repeat_of, confidence,
evidence IDs) → each `Step` (description, start, end, evidence IDs). Defined with **Pydantic** in
`schema.py`, so next week's benchmark can compare methods directly.

---

### Tools & technologies
| Tool | Used for |
|---|---|
| Python, **pandas**, NumPy | Event tables, interval maths |
| **PyArrow / Parquet** | Caching the parsed export |
| **Pydantic** | Output schema and validation of LLM JSON |
| **PM4Py** | Directly-follows graphs (process mining) |
| PrefixSpan (own implementation) | Frequent sequential patterns |
| **Google Gemini** (`google-genai`) / Anthropic Claude | LLM method, via structured JSON output |
| Regex + optional **Presidio** | Redaction |
| **Streamlit** + **Altair** | This demo app and its timeline chart |
| **pytest** | Tests on synthetic data (`scripts/make_fake_data.py`) |
"""
