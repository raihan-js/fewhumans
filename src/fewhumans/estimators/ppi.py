"""PPI estimator E2 (spec §6.1).

Implements prediction-powered inference for a finite population mean,
Angelopoulos et al., "Prediction-Powered Inference" (arXiv 2301.09633),
equations (1)–(3): with N judge predictions f(X_i) and n human labels Y_j,

  theta_hat = mean_N f(X) + mean_n (Y - f(X))            (eq. 1)
  Var_hat   = Var_N(f)/N + Var_n(Y - f)/n                (eq. 3)
  CI        = theta_hat ± z_{1-α/2} sqrt(Var_hat)        (eq. 2)

Assumptions: human labels are a uniform random sample with known probabilities;
judge predictions are fixed (no tuning on the human sample).
"""

from __future__ import annotations

import numpy as np

from fewhumans.estimators.classical import EstimateResult, _z


def ppi_mean(
    pred_all: list[float],
    y_human: list[float],
    pred_human: list[float],
    confidence: float = 0.95,
) -> EstimateResult:
    """E2: PPI estimate of the mean.

    Args:
        pred_all: judge predictions mapped to numbers, all N items.
        y_human: human labels mapped to numbers, n labeled items.
        pred_human: judge predictions on exactly the n human-labeled items.
    """
    f_all = np.asarray(pred_all, dtype=float)
    y = np.asarray(y_human, dtype=float)
    f_lab = np.asarray(pred_human, dtype=float)
    n_all, n = len(f_all), len(y)
    assert n == len(f_lab) and n > 1 and n_all > 1, "need ≥2 judge and ≥2 human labels"
    rectifier = y - f_lab
    theta = float(f_all.mean() + rectifier.mean())
    var = float(f_all.var(ddof=1) / n_all + rectifier.var(ddof=1) / n)
    hw = _z(confidence) * float(np.sqrt(max(var, 0.0)))
    return EstimateResult(theta, hw, theta - hw, theta + hw, n, "E2-ppi",
                          confidence, False, "")
