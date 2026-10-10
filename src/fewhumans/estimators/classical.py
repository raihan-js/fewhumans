"""Estimate results and classical estimators E0/E1 (spec §6.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class EstimateResult:
    """A point estimate with an interval.

    Attributes:
        value: point estimate of the estimand.
        halfwidth: half-width of the two-sided interval.
        lo, hi: interval endpoints.
        n_human: number of human labels used.
        method: estimator id (E0/E1/E2/...).
        confidence: nominal level (e.g. 0.95).
        anytime_valid: True only for sequential confidence sequences.
        note: human-readable caveat (e.g. judge-only is not valid for truth).
    """

    value: float
    halfwidth: float
    lo: float
    hi: float
    n_human: int
    method: str
    confidence: float = 0.95
    anytime_valid: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}] "
            f"({self.method}, {self.n_human} human labels, "
            f"{int(self.confidence * 100)}% CI)"
        )


def _z(confidence: float) -> float:
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


def judge_only(values: list[float], confidence: float = 0.95) -> EstimateResult:
    """E0: estimand computed on panel (judge) labels with a naive CLT interval.

    Estimand: mean of judge-mapped values.
    Assumption: NONE about correctness — this interval covers the *judge* mean,
    not the truth. Included to measure its failure (spec §6.1); never the default.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    hw = _z(confidence) * se if n > 1 else float("nan")
    return EstimateResult(mean, hw, mean - hw, mean + hw, 0, "E0-judge-only",
                          confidence, False, "covers the judge mean, NOT the truth")


def human_only(values: list[float], confidence: float = 0.95) -> EstimateResult:
    """E1: classical estimate on the human sample with a CLT interval.

    Estimand: population mean of the mapped human labels.
    Assumption: human labels are a uniform random sample (known probabilities).
    Angelopoulos et al. PPI (2023) §2 baseline; Wilson/exact variants for
    proportions arrive in Phase 2.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    hw = _z(confidence) * se if n > 1 else float("nan")
    return EstimateResult(mean, hw, mean - hw, mean + hw, n, "E1-human-only",
                          confidence, False, "")
