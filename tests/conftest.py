from pathlib import Path

import pandas as pd
import pytest

from workflow_recon.config import load_config
from workflow_recon.load import flatten_export, load_task_log, read_export
from workflow_recon.pipeline import prepare_window
from workflow_recon.windows import day_window

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def raw_export():
    return read_export(FIX / "fake_export.json")


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"paths": {"cache_dir": str(tmp_path / "cache"),
                                            "runs_dir": str(tmp_path / "runs"),
                                            "redact_terms_file": str(tmp_path / "none.txt")},
                                  "redact": {"use_presidio": False}})


@pytest.fixture(scope="session")
def events(raw_export):
    return flatten_export(raw_export, "America/Los_Angeles")


@pytest.fixture(scope="session")
def fake_log():
    return load_task_log(FIX / "fake_task_log.csv")


@pytest.fixture
def day1(events, cfg):
    return prepare_window(events, day_window("2026-12-01", cfg["timezone"]), cfg)


@pytest.fixture
def day2(events, cfg):
    return prepare_window(events, day_window("2026-12-02", cfg["timezone"]), cfg)


def overlap_s(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, (min(a_end, b_end) - max(a_start, b_start)).total_seconds())


def best_match(tasks, start, end):
    """Predicted task with the largest time overlap with [start, end)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    scored = [(overlap_s(pd.Timestamp(t.start), pd.Timestamp(t.end), start, end), t) for t in tasks]
    scored.sort(key=lambda x: -x[0])
    return scored[0] if scored else (0.0, None)
