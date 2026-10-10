"""E6: anytime-valid confidence sequences (spec §6.1).

What is shipped in Phase 2: a confidence sequence (CS) for the human-only mean
of bounded values, built from a union bound over sample sizes ("peeling"):

  ε_t = sqrt(log(2·t·(t+1)/α) / (2·t)),   C_t = [μ̂_t − ε_t, μ̂_t + ε_t].

Validity sketch (self-contained): at each fixed t, Hoeffding's inequality for
[lo, hi]-bounded i.i.d. values gives P(μ ∉ C_t) ≤ α/(t(t+1)); Σ_{t≥1} 1/(t(t+1))
= 1, so P(μ ∉ C_t for some t) ≤ α by the union bound. Hence a researcher may
check after every label and stop whenever the width suffices, with the nominal
miscoverage rate intact. This is the standard confidence-sequence concept
(see e.g. Howard et al., "Time-uniform, nonparametric, nonasymptotic confidence
sequences", Ann. Statist. 2021, for the general theory — no equation claimed).

What is NOT shipped yet: sequential versions of E3 (PPI++) and E5 (active),
and the Sfyraki & Wang Freedman-based bound — they stay behind
experimental=True until the Phase 4 gate validates them.
"""

from __future__ import annotations

import numpy as np

from fewhumans.estimators.classical import EstimateResult


def cs_radius(t: int, alpha: float) -> float:
    """Half-width of the union-bound Hoeffding CS after t samples (unit scale)."""
    if t < 1:
        raise ValueError("t must be ≥ 1")
    return float(np.sqrt(np.log(2.0 * t * (t + 1.0) / alpha) / (2.0 * t)))


def cs_human_only(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    lo: float = 0.0,
    hi: float = 1.0,
) -> EstimateResult:
    """E6: anytime-valid CS for the mean of bounded human-mapped values.

    Args:
        values: human labels mapped to numbers, in arrival order, within [lo, hi].
    """
    arr = np.asarray(values, dtype=float)
    t = len(arr)
    if t < 1:
        raise ValueError("need ≥1 human label")
    if float(arr.min()) < lo or float(arr.max()) > hi:
        raise ValueError(f"values outside [{lo}, {hi}]")
    alpha = 1.0 - confidence
    mean = float(arr.mean())
    hw = (hi - lo) * cs_radius(t, alpha)
    return EstimateResult(mean, hw, mean - hw, mean + hw, t, "E6-human-CS",
                          confidence, True, "anytime-valid (union-bound Hoeffding CS)")


def sequential_mean(values: list[float], confidence: float = 0.95, experimental: bool = False) -> EstimateResult:
    """E6 entry point. Validated construction: pass values in [0, 1].

    The PPI++/active sequential bounds are research stubs: they require
    experimental=True and raise until the Phase 4 gate validates them.
    """
    if not experimental:
        return cs_human_only(values, confidence)
    raise NotImplementedError("Sequential PPI++/active bounds land after the Phase 4 gate.")
