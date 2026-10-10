"""E5: active inference with item-specific sampling probabilities (spec §6.1).

Implements the batch active mean estimator of Zrnic & Candès, "Active
Statistical Inference" (arXiv 2403.03208), §4 eq. (1) and §5 Prop. 1,
adapted from i.i.d. Bernoulli sampling to our fixed-pool setting:

  θ̂ = f̄_N + (1/N)·Σ_{labeled} (y_i − f_i)/π_i            (eq. 1)
  δ_i = f_i + (y_i − f_i)·ξ_i/π_i,  θ̂ = δ̄                  (per-item increments)
  V̂ = S²_δ/N,  CI = θ̂ ± z·√V̂                              (Prop. 1 plug-in)

where ξ_i = 1 if item i was labeled and π_i is its (logged) inclusion
probability. Adaptation notes (logged in docs/decisions.md):
- The paper samples each point independently (ξ_i ∼ Bern(π(X_i))); we sample
  fixed-size batches without replacement (R0/R1) or Poisson batches (R2–R5).
  With E[ξ_i] = π_i the estimator stays unbiased; the plug-in variance ignores
  without-replacement covariances, i.e. it is conservative there.
- With uniform π_i = n/N the estimator reduces to E2/PPI (paper §3 notes PPI
  is the uniform special case) — tested in tests/test_active.py.
- The interval is asymptotic CLT; its actual coverage is measured by the
  Phase 4 gate, which decides whether E5 stays in the default API.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fewhumans.estimators.classical import EstimateResult, _t


def active_mean(
    pred_all: Mapping[str, float],
    y_human: Mapping[str, float],
    probs: Mapping[str, float],
    confidence: float = 0.95,
) -> EstimateResult:
    """E5: active (IPW) estimate of the mean.

    Args:
        pred_all: judge prediction per item, ALL items.
        y_human: human label per labeled item.
        probs: sampling (inclusion) probability per labeled item (must be > 0).
    """
    ids = list(pred_all)
    n_total = len(ids)
    if not y_human:
        raise ValueError("need ≥1 human label")
    deltas = []
    for i in ids:
        f = float(pred_all[i])
        if i in y_human:
            p = float(probs.get(i, 0.0))
            if p <= 0.0:
                raise ValueError(f"item {i} has non-positive sampling probability")
            deltas.append(f + (float(y_human[i]) - f) / p)
        else:
            deltas.append(f)
    arr = np.array(deltas)
    theta = float(arr.mean())
    var = float(arr.var(ddof=1) / n_total) if n_total > 1 else float("nan")
    hw = _t(len(y_human) - 1, confidence) * float(np.sqrt(max(var, 0.0)))
    return EstimateResult(theta, hw, theta - hw, theta + hw, len(y_human),
                          "E5-active", confidence, False, "")
