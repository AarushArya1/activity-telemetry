"""Helpers shared by the deterministic methods (sequence mining, baseline)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..labels import STRONG, label_hint


def add_label_hints(segments: pd.DataFrame) -> pd.DataFrame:
    df = segments.copy()
    hints = [label_hint(r.app, r.domain, r.path, r.title_norm) for r in df.itertuples(index=False)]
    df["hint_label"] = [h[0] for h in hints]
    df["hint_strong"] = [h[1] == STRONG for h in hints]
    return df


@dataclass
class SegArrays:
    """Numpy view of a segment table for fast interval queries (epoch seconds)."""
    s: np.ndarray
    e: np.ndarray
    tokens: list[str]
    ids: list[str]
    max_dur: float

    @classmethod
    def from_df(cls, df: pd.DataFrame, token_col: str = "token") -> "SegArrays":
        s = np.array([t.timestamp() for t in df["start"]], dtype=float)
        e = np.array([t.timestamp() for t in df["end"]], dtype=float)
        return cls(s, e, list(df[token_col]), list(df["segment_id"]),
                   float((e - s).max()) if len(s) else 0.0)

    def overlapping(self, a: float, b: float) -> list[tuple[int, float]]:
        """(index, overlap seconds) of segments overlapping [a, b)."""
        lo = int(np.searchsorted(self.s, a - self.max_dur, side="left"))
        hi = int(np.searchsorted(self.s, b, side="left"))
        out = []
        for i in range(lo, hi):
            ov = min(b, self.e[i]) - max(a, self.s[i])
            if ov > 0:
                out.append((i, ov))
        return out

    def token_vector(self, a: float, b: float) -> dict[str, float]:
        v: dict[str, float] = defaultdict(float)
        for i, ov in self.overlapping(a, b):
            v[self.tokens[i]] += ov
        return dict(v)


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 1.0


def normalized(v: dict[str, float]) -> dict[str, float]:
    tot = sum(v.values())
    return {k: x / tot for k, x in v.items()} if tot else {}


def add_vec(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, x in b.items():
        out[k] = out.get(k, 0.0) + x
    return out


def runs(values: list) -> list[tuple[int, int]]:
    """[start, end) index ranges of equal consecutive values."""
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


# ------------------------------------------------------------ PrefixSpan

def prefixspan(sequences: list[list[str]], min_support: int = 2, max_len: int = 5) -> list[tuple[list[str], int]]:
    """Frequent sequential patterns (items need not be contiguous). Support = #sequences containing it."""
    results: list[tuple[list[str], int]] = []

    def project(db: list[tuple[int, int]], prefix: list[str]) -> None:
        if len(prefix) >= max_len:
            return
        counts: dict[str, set[int]] = defaultdict(set)
        for sid, pos in db:
            for item in set(sequences[sid][pos:]):
                counts[item].add(sid)
        for item, sids in sorted(counts.items()):
            if len(sids) < min_support:
                continue
            new_prefix = prefix + [item]
            results.append((new_prefix, len(sids)))
            new_db = []
            for sid, pos in db:
                seq = sequences[sid]
                for k in range(pos, len(seq)):
                    if seq[k] == item:
                        new_db.append((sid, k + 1))
                        break
            project(new_db, new_prefix)

    project([(i, 0) for i in range(len(sequences))], [])
    return results


def seq_similarity(a: list[frozenset], b: list[frozenset], item_threshold: float = 0.5) -> float:
    """1 - normalized edit distance, where two steps 'match' if their token sets overlap enough."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def same(x: frozenset, y: frozenset) -> bool:
        return len(x & y) / max(len(x | y), 1) >= item_threshold

    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (0 if same(a[i - 1], b[j - 1]) else 1))
            prev = cur
    return 1.0 - d[m] / max(n, m)
