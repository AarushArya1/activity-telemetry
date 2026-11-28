# activity-telemetry

Reconstructing a user's workflow, testing LLMs and also data-driven techniques such as sequence mining. Creating a Streamlit demo to showcase this. Raw JSON data from ActivityWatch, sourced from my own workflows. Proper benchmarking will be done later. Project for Orcapo.

See `PROJECT_CONTEXT.md` (the full brief) and `METHODS_INFO_CONTEXT.md` (what the methods may and may not see).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[app,mining]"     # app = Streamlit; mining = pm4py (optional)
cp .env.example .env                         # then put your key in .env (gitignored)
```

The LLM method uses Gemini by default (`llm.provider: gemini`, key `GEMINI_API_KEY` from
https://aistudio.google.com/apikey). Set `llm.provider: anthropic` to use Claude instead
(`ANTHROPIC_API_KEY`, `pip install -e ".[anthropic]"`). A key already exported in your shell wins over `.env`.

**Free-tier privacy note:** on Gemini's free tier, Google may use inputs/outputs to improve its
products and human reviewers may read them. Only redacted text is sent, but titles and URLs still
describe your activity — consider a paid tier, or adding names to `data/redact_terms.txt`.

Put real data under `data/` (gitignored): `data/aw-buckets-export.json`, optionally `data/task_log.csv`,
and optionally `data/redact_terms.txt` (names to scrub, one per line).

## Demo app

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501 (bound to localhost only). Upload an export (or use the synthetic
sample), pick a day and a method, press **Run**. The two buttons at the top explain the project and
its core concepts.

## Usage (CLI)

```bash
python -m workflow_recon inspect                       # bucket ids, counts, data keys, time ranges
python -m workflow_recon run -v                        # most recent day, all methods
python -m workflow_recon run --date 2026-11-23 -v
python -m workflow_recon run --all-days --methods sequence_mining,baseline
python -m workflow_recon run --mode session            # session windows instead of whole days
python -m workflow_recon patterns                      # variants + DFG per label over all saved runs
python -m workflow_recon patterns --method llm --level sites --label "CS 61A Notes/Homework"

# synthetic data (no real personal data):
python scripts/make_fake_data.py
python -m workflow_recon run --export tests/fixtures/fake_export.json --all-days -v
```

Runs are saved to `outputs/runs/<method>/<window_id>.json` (common schema, `schema.py`) with a
`.sidecar.json` next to them (mined patterns / DFGs, or LLM usage per chunk). Run days oldest-first
(`--all-days`) so repeat detection can use each method's own earlier results.

### Patterns across days

`patterns` pools every saved run of one method into **one PM4Py event log per label** (case = predicted
task, activity = a step's main site(s), or every site switch with `--level sites`) and reports the
**variants** (distinct step orders, how many tasks followed each) and a directly-follows graph. It reads
only the method's own runs, never the task log. Output: terminal summary + `outputs/patterns/<method>_<level>.json`;
in the app, the **Patterns across days** tab.

## Layout

```
config.yaml                      every threshold / model / prompt version (recorded in each run)
prompts/v1_workflow_inference.txt
src/workflow_recon/
  load.py        stage 1  export -> tidy event table (+ Parquet cache), task log loader
  windows.py     stage 2  day (default) / session / padded_task (debug only) windows
  clean.py       stage 3  AFK masking, Chrome dedup, incognito drop, URL/title normalization, segments
  redact.py      stage 4  emails, phones, IDs, chat titles, social profiles, user terms, optional Presidio
  labels.py      taxonomy + transparent rule hints (shared by Method A and the baseline)
  methods/
    sequence_mining.py   Method A: label resolution, change points, steps, DFG, PrefixSpan, repeats
    llm_inference.py     Method B: render -> chunk -> structured output -> citation validation -> merge
    baseline.py          per-segment rule labels, no smoothing or steps
  patterns.py    cross-day process mining: one event log per label, variants, DFG (PM4Py)
  llm/           vendor-neutral provider interface, Gemini + Anthropic providers, on-disk response cache
  io.py, pipeline.py, cli.py
app/streamlit_app.py             demo UI (app/explainers.py holds the popup text)
scripts/make_fake_data.py        synthetic export + task log in tests/fixtures/
tests/                           pytest suite (synthetic data only)
```

## Ground rules (enforced in code)

- Methods never receive the task log. It is read only to drop time inside `Private` tasks.
- Default window is a whole local day; windows are never cut at logged task boundaries.
- `run_llm` refuses unredacted tables and runs a final leak check on the exact prompt text.
- T01–T17 are the development set; tune only on those.
