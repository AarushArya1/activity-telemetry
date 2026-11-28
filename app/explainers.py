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
4. Optionally open **Patterns across days**: all saved days are pooled per label to show how many
   different ways the same kind of task gets done, and which way is most common.

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

Across many days, the **Patterns** tab turns those results into a picture of your routine workflows:
for each label, the distinct step orders ("variants") and how often each one happens.

Later, both methods get scored against the hand-logged tasks, to see which reconstructs workflows better.
"""

CORE_CONCEPTS = """
### The pipeline

```
ActivityWatch JSON → 1 Load → 2 Pick window → 3 Clean & merge → 4 Redact → Method A or B → workflow
                                                     saved workflows from many days → Patterns tab
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

**Across days (the "Patterns" tab).** All saved tasks with the same label are pooled into one
**event log** (case = task, activity = a step's main sites, ordered by time). PM4Py then lists the
**variants**, the distinct orders in which those tasks were done and how often each occurred, and draws a
directly-follows graph for the label. Few variants that cover most tasks = a routine, automatable workflow.
Code: `patterns.py`

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
| **PM4Py** | Directly-follows graphs and variants (process mining, per day and across days) |
| PrefixSpan (own implementation) | Frequent sequential patterns |
| **Google Gemini** (`google-genai`) / Anthropic Claude | LLM method, via structured JSON output |
| Regex + optional **Presidio** | Redaction |
| **Streamlit** + **Altair** | This demo app and its timeline chart |
| **pytest** | Tests on synthetic data (`scripts/make_fake_data.py`) |
"""

BENCHMARK_PLAN = """
*Summary of `BENCHMARK_PLAN.md`. The plan is still a draft and may change; nothing here is built yet.*

### What the benchmark is for
Compare the **sequence-mining** method, the **LLM** method and a simple **baseline** (label each
stretch of activity by its main app or website: no task splitting, no steps) against the hand-written
task log, and answer the questions Orcapo would ask. Every result is read as
**"how much better than the baseline?"**

### Ground rules
- **Only score what can be checked.** The task log doesn't cover every minute, so predictions in
  unlogged time are ignored, except inside **fully logged blocks** where every minute has a label.
- **Meaning over wording.** "Code in IntelliJ" can match a logged step written differently.
- **Same input for everyone.** Each method gets the same cleaned, redacted day plus the label list.
  Nobody sees the answers.

### The data
| Set | Tasks | Used for |
|---|---|---|
| Development | T01–T17 (Nov 18–23) | Building and tuning. Scores shown for reference only (they flatter the methods) |
| **Test** | Tasks logged *after* the methods are frozen | **The headline results** |

- **Freeze first:** the code, prompt and settings of each method are committed (and the commit
  hash recorded) *before* anyone looks at the test data.
- **Still to log for the test set:** normal tasks with steps; **2–3 fully logged blocks** of 2–3 hours
  where *every* minute gets a row (including breaks and games), marked `FULL BLOCK START` / `END` in
  Notes; and at least one more instance of each repeated workflow (internship application,
  61A lecture review, 61A homework, 61B project).

---

### The five benchmarks, most important first

**1 · B4 Step recovery: "Does it reconstruct *how* I did the task?"**
Steps are the actual product: they show which parts of a workflow an AI tool could take over.
For each task that was found, predicted steps are matched to logged steps by meaning, using an **LLM
judge** (fixed prompt, checked by hand on ~10 tasks; if it disagrees more than ~1 in 5 times, the
prompt gets fixed first).
- *Step recall*: share of the real steps that some predicted step covers.
- *Step precision*: share of predicted steps that match a real step.
- Example: 1 predicted step covering 1 of 5 real steps → recall 0.20, precision 1.00.

**2 · B5 Repeat detection: "Does it notice when I do the same workflow again?"**
Only workflows that recur are worth automating. Scored on **pairs** of tasks: if T03, T05 and T08 are
the same workflow, the true pairs are T03–T05, T03–T08 and T05–T08. The method's "repeat of" links
produce its own pairs. *Pair precision* and *pair recall*. Resuming a task after a break
(a continuation) does **not** count as a repeat.

**3 · B3 Invented tasks: "Does it make things up?"**
A made-up workflow in a client report damages trust, so this matters more than catching everything.
Checked only inside the fully logged blocks. A predicted task is **correct** if at least half of it
overlaps a logged task with the same label; otherwise it's **invented**.
- *Task precision* and *invented tasks per hour*.
- Also reported with low-confidence tasks removed (one threshold, chosen on the dev set and then
  fixed), e.g. Minecraft time that was labeled as 61B work at confidence 0.03.

**4 · B1 Label accuracy: "Does it know what I was doing?"**
Logged time is cut into **1-minute slots**; each slot is right if the predicted label matches.
- *Accuracy*: share of logged minutes labeled correctly.
- *Macro-F1*: the same idea averaged per label, so short tasks count as much as long ones.
- Plus a **confusion matrix** (which labels get mistaken for which).
- The cleanest single number, and it doesn't punish a method for splitting time into many tasks.

**5 · B2 Task detection: "Does it find each task?"**
Each logged task is matched to the same-label prediction that overlaps it most. It counts as found if the
overlap is at least half of the combined time span (**IoU ≥ 0.5**).
- *Task recall*: share of logged tasks found. B4 and B5 depend on these matches.
- *Boundary error* (minutes off at start and end) goes in the appendix, since exact minutes don't change
  any recommendation.

---

### Pass/fail guardrails
| Check | Applies to | Pass if |
|---|---|---|
| **Invalid citations**: steps pointing to segments that don't exist or lie outside the task | LLM | 0 after validation (report how many were dropped) |
| **Redaction leaks**: a hand-made list of sensitive strings (contact names, emails, order IDs) searched for in everything sent to the LLM and every output | Both | **0 leaks**. A hard requirement: no company will run this otherwise |
| **Cost**: runtime per day, LLM tokens and dollars | Both | Reported, no threshold |

---

### What the final write-up shows
1. **Workflow inventory (first):** one table per method with one row per real repeated workflow:
   how many real instances, whether the method found it as a repeated workflow (✓ if at least 2
   instances are found *and* linked), how many instances it linked, average step recall,
   **number of variants** (how many different step orders the workflow was done in, true vs. predicted,
   as in the *Patterns across days* tab), and total time (true vs. predicted). Under it: any **invented** recurring workflows.
   This is the view Orcapo would hand a client: *"these are your recurring workflows, how you do them,
   and how long they take."*
2. **One metrics table** (test set only, columns in priority order): baseline, sequence mining, and the
   LLM, each with and without the confidence filter. Dev-set numbers go in an appendix.
3. A confusion matrix per method, and **2–3 worked examples** in this demo: a clear success, an invented
   task, and a step-by-step comparison.

### How scoring will work (later)
A scoring script reads **only** `task_log.csv` and the saved runs in `outputs/runs/`. It converts all
times to UTC and fails loudly on any label not in the taxonomy. It checks that fully logged blocks have
no unlogged gaps over 2 minutes. It writes `results/<date>_summary.json` plus a per-task CSV so every
number can be traced back. Each method is run once per config, with LLM answers cached so results are
reproducible.
"""
