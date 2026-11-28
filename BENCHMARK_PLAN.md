# Benchmark Plan — Workflow Reconstruction

How the two methods (sequence mining and LLM inference), plus a simple baseline, will be compared next week. This is deliberately small: **five benchmarks**, each answering one question Orcapo would ask, plus three pass/fail guardrails.

Read alongside `PROJECT_CONTEXT.md` (data formats, output schema, no-leakage rules).

---

## 1. Principles

1. **Score only what we can verify.** The task log covers only part of the tracked time. Predictions in unlogged time are ignored, except inside **fully logged blocks** (§2), where every minute has a label.
2. **Don't punish different wording or granularity.** The methods name steps differently than Aarush does ("Code in IntelliJ (World.java)" vs. five detailed steps). Steps are matched by meaning, not text.
3. **Same input for every method.** Each method runs on whole sessions or days of cleaned, redacted activity plus the label taxonomy. No ground truth is ever passed in.
4. **Always include the baseline.** Baseline = label each segment by its dominant app or domain, with no segmentation or steps. Every result is read as "how much better than the baseline".

---

## 1b. What matters most to Orcapo

Orcapo's goal is to replace surveys and interviews with an automatic map of how people work, then point clients to the workflows most worth automating or assisting with AI. The benchmarks are ranked by how directly they measure that:

| Rank | Benchmark | Why it matters to Orcapo |
|---|---|---|
| 1 | **B4 Step recovery** | Steps are the actual product: they show which parts of a workflow an AI tool could take over. A label alone doesn't |
| 2 | **B5 Repeat detection** | Only recurring workflows are worth automating. This is how opportunities get found and ranked |
| 3 | **B3 Invented tasks** | Output goes to clients. A made-up workflow in a client report damages trust fast, so precision matters more than catching everything |
| 4 | **B1 Label accuracy** | Answers "how much time goes where", needed to size the payoff. Less transferable, since every client will have its own label set. Still the cleanest single number to present |
| 5 | **B2 Task boundaries** | Exact start/end minutes don't change any recommendation. Task recall stays because B4 and B5 rely on its matching; boundary error goes in the appendix |

**Redaction leaks** (§4) is a hard requirement, not a footnote: no company will run this on employees' laptops unless personal data provably stays protected. Present it prominently.

---

## 2. Test data

| Set | Tasks | Use |
|---|---|---|
| Development | T01–T17 (Nov 18–23) | Building and tuning only. Report scores here for reference, but they are optimistic |
| **Test** | Tasks logged after the methods are frozen | **Headline results** |

**Test-set logging requirements (next weekend / early next week):**

- Log tasks as usual, with steps for repeated workflows, using the identical wording already used for shared steps.
- **2–3 fully logged blocks of 2–3 hours each.** Every minute gets a row, including gaming, browsing and breaks. Mark them in Notes as `FULL BLOCK START` / `FULL BLOCK END`. These are required for Benchmark 3.
- At least one more instance of each repeated workflow: internship application, 61A lecture review, 61A homework, 61B project.

**Freeze rule:** commit each method's code, prompt version and config before looking at any test-set data. Record the commit hash in the results.

---

## 3. The five benchmarks

### B1. Label accuracy (time-based) — *"Does it know what I was doing?"*

- Split all logged time into **1-minute slots**. Each slot's true label is the task-log label. Each slot's predicted label is the label of the predicted task covering most of that minute, or `none`.
- **Report:**
  - **Accuracy**: share of logged minutes with the correct label
  - **Macro-F1** across labels, so short labels count as much as long ones
- **Also show** one confusion matrix (true label × predicted label) in the write-up, not as a score.
- Why this is the headline: it doesn't depend on how many tasks a method creates, so a method that splits time into 29 tasks isn't penalized for that alone.

### B2. Task detection and boundaries — *"Does it find each task and where it starts and ends?"*

- Match each logged task to the predicted task **with the same label** that overlaps it most. It counts as found if temporal IoU ≥ 0.5 (overlap ÷ union of the two time spans).
- **Report:**
  - **Task recall**: share of logged tasks found
  - **Median boundary error**: minutes between true and predicted start, and end, over found tasks
- **Continuations** (e.g. T15 + T17 split by a break): score with them kept separate. Also note in the write-up whether the break itself was found as its own task.

### B3. Invented tasks — *"Does it make things up?"*

- Measured **only inside fully logged blocks**, where every minute has ground truth.
- A predicted task is **correct** if it overlaps a logged task with the same label for at least half of its own duration. Otherwise it is **invented**.
- **Report:**
  - **Task precision**: share of predicted tasks that are correct
  - **Invented tasks per hour**
- **Confidence filter:** also report task precision and B2 task recall with tasks below a confidence threshold removed (pick one threshold on the dev set, e.g. 0.3, and keep it fixed). Example from the dev set: the Minecraft tasks labeled CS 61A / CS 61B had confidence 0.22 and 0.03.

### B4. Step recovery — *"Does it reconstruct how I did the task?"*

- For each task found in B2 whose logged row has steps, match predicted steps to true steps **by meaning**:
  - **Primary:** an LLM judge with a fixed prompt. It gets the two step lists and returns which true steps each predicted step covers. One predicted step may cover several true steps. Temperature 0; prompt version recorded.
  - **Check:** hand-review the judge on ~10 tasks; if it disagrees with Aarush on more than ~1 in 5 matches, fix the judge prompt before reporting.
- **Report:**
  - **Step recall**: share of true steps covered by at least one predicted step
  - **Step precision**: share of predicted steps that cover at least one true step
- Example: T11 with a single predicted step "Code in IntelliJ (World.java)" → recall 1/5 = 0.20, precision 1/1 = 1.00.

### B5. Repeat detection — *"Does it notice when I do the same workflow again?"*

- Score over **pairs of tasks**. True pairs come from the task log's "Repeat of" column: every task linked to the same first instance forms a pair with it and with each other, e.g. T03–T05, T03–T08, T05–T08.
- Predicted pairs come from the methods' `repeat_of` links, mapped onto logged tasks through the B2 matches. Apply the same grouping rule.
- **Report:** **pair precision** and **pair recall**.
- Continuation links (T15 → T17) are **not** counted as true repeats. Note them separately if a method makes them.

---

## 4. Guardrails (pass/fail, reported in one line each)

| Guardrail | Applies to | Pass condition |
|---|---|---|
| **Invalid citations**: steps citing segment IDs that don't exist or fall outside the task | LLM method | 0% after validation; report how many were dropped |
| **Redaction leaks**: a hand-made list of known sensitive strings (contact and chat names, email addresses, order IDs) searched for in everything sent to the LLM and in all outputs | Both | 0 leaks |
| **Cost**: runtime per day of data, plus LLM tokens and dollars | Both | Reported, no threshold |

---

## 5. Results (what goes in the write-up)

### 5a. Workflow inventory (lead with this)

The view Orcapo cares about, built from B3–B5 and the method's predicted times. No new metric, just the results arranged the way Orcapo's product would report to a client: "here are this person's recurring workflows, how they do them, and how long they take."

One table per method, one row per **real repeated workflow** in the test set (e.g. internship applications, 61A lecture review, 61A homework, 61B project):

| Workflow | Instances (true) | Found as a repeated workflow? | Instances linked (B5) | Step recall (B4, avg.) | # Variants: true vs. predicted | Total time: true vs. predicted (min) |
|---|---|---|---|---|---|---|
| Internship application | 3 | ✓ / ✗ | 2 / 3 | 0.67 | 1 vs. 2 | 70 vs. 64 |
| ... | | | | | | |

Under each table, one line: **workflows invented**, i.e. predicted repeated workflows that match no real one (from B3/B5), e.g. "Minecraft sessions reported as recurring CS 61B work".

**How to fill it:**
- **Found as a repeated workflow:** ✓ if at least 2 of its true instances are matched (B2) and linked to each other by the method (B5).
- **# Variants:** how many distinct ways the workflow was done, i.e. distinct step orders, as in the demo's *Patterns across days* tab. **True:** distinct ordered step lists among its logged instances (exact text, since shared steps use identical wording). **Predicted:** distinct step-level traces (each step named by its main site(s)) among the predicted tasks matched to those instances (B2). Fewer variants = a more routine, more automatable workflow; a predicted count far above the true one means the method's steps are unstable across repeats.
- **Total time, predicted:** sum of the durations of the predicted tasks matched to that workflow's instances. Compare with the summed durations from the task log.

### 5b. Metrics table

One table, test set only. Dev-set numbers go in an appendix. Columns are ordered by priority (§1b).

| Method | B4 Step recall | B4 Step precision | B5 Pair P / R | B3 Precision | B3 Invented / hr | B1 Acc. | B1 Macro-F1 | B2 Task recall | Redaction leaks | Cost / day |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | n/a | n/a | n/a | | | | | | | |
| Sequence mining | | | | | | | | | | |
| Sequence mining + conf. filter | | | | | | | | | | |
| LLM | | | | | | | | | | |
| LLM + conf. filter | | | | | | | | | | |

B2 boundary error goes in the appendix.

Plus:
- One confusion matrix per method (B1)
- 2–3 worked examples shown in the Streamlit demo: one clear success, one invented task (e.g. Minecraft → CS 61B), one step-recovery comparison

---

## 6. Implementation notes for the scoring script

- **Inputs:** `task_log.csv` and the saved `MethodRun` JSON files from `outputs/runs/<method>/`. Nothing else.
- **Times:** convert everything to tz-aware UTC before comparing. The task log is America/Los_Angeles.
- **Labels:** must match the taxonomy strings exactly. Fail loudly on an unknown label rather than silently mismatching.
- **Fully logged blocks:** parse the `FULL BLOCK START` / `FULL BLOCK END` markers from Notes, and verify there are no unlogged gaps longer than 2 minutes inside a block.
- **Workflow inventory (§5a):** group true tasks into workflows by following "Repeat of" to the first instance; generate one inventory table per method from the B2 matches, B4 scores, B5 links and variant counts (predicted variants reuse the step traces saved in each run's sidecar, `patterns.py`).
- **Output:** one `results/<date>_summary.json` with every metric above, plus the workflow inventory tables, plus a CSV of per-task matches (logged task, matched predicted task, IoU, start/end error, step recall/precision) so any number can be traced back.
- **Run each method once per config.** If the LLM method is re-run, use the cache, so results are reproducible.
