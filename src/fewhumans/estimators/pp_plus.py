"""E3: PPI++ with power-tuned λ (spec §6.1).

Implements prediction-powered inference with power tuning for a scalar mean,
Angelopoulos, Duchi & Zrnic, "PPI++: Efficient Prediction-Powered Inference"
(arXiv 2311.01453). This module mirrors the reference implementation `ppi_py`
(`ppi.ppi_mean_pointestimate` / `ppi.ppi_mean_ci`, git commit pinned in
docs/decisions.md) so the cross-check in tests/test_pp_plus.py is exact:

  λ̂ = Cov̂(Y, f) / ((1 + n/N)·Var̂(f)), clipped to [0, 1]     (ppi_py `_calc_lam_glm`, scalar case)
  θ̂ = λ̂·f̄_unl + mean(Y − λ̂·f_lab)                            (PPI++ point estimate)
  SE = sqrt(Var(λ̂·f_unl)/N + Var(Y − λ̂·f_lab)/n)              (CLT interval)

where Cov̂ is the centered (1/n) covariance of Y and f on the labeled set,
Var̂(f) is np.cov (ddof, i.e. /(M−1)) over labeled + unlabeled predictions, and the SE uses population (ddof=0) standard deviations — all
exactly as in ppi_py. λ=0 recovers the classical estimate, λ=1 recovers E2/PPI.

Assumptions: human labels are a uniform random sample, independent of the
unlabeled set; judge predictions fixed (no tuning on the human sample).
Validity of the interval is NOT claimed here — the Phase 4 coverage gate decides.
"""

from __future__ import annotations

import numpy as np

from fewhumans.estimators.classical import EstimateResult, _z


def power_tuned_lambda(
    y_human: list[float] | np.ndarray,
    pred_human: list[float] | np.ndarray,
    pred_unlabeled: list[float] | np.ndarray,
) -> float:
    """PPI++ power-tuning parameter λ̂ ∈ [0, 1] (ppi_py `_calc_lam_glm`, scalar).

    Returns 0.0 when predictions carry no signal (constant judge → classical).
    """
    y = np.asarray(y_human, dtype=float)
    f_lab = np.asarray(pred_human, dtype=float)
    f_unl = np.asarray(pred_unlabeled, dtype=float)
    n = len(y)
    n_unl = len(f_unl)
    if n < 2 or n_unl < 1:
        raise ValueError("need ≥2 human labels and ≥1 unlabeled prediction")
    if len(f_lab) != n:
        raise ValueError("pred_human must align with y_human")
    # ppi_py centers gradients at the untuned PPI point and then re-centers at
    # their own means; the net effect is the ordinary centered covariance.
    y_c = y - y.mean()
    f_c = f_lab - f_lab.mean()
    cov = float((y_c * f_c).mean())  # (1/n) normalization, as in ppi_py
    # Note ppi_py's cov has a factor-2 convention absorbed into num AND denom.
    f_all = np.concatenate([f_lab, f_unl])
    var_f = float(np.cov(f_all)) if len(f_all) > 1 else 0.0
    denom = (1.0 + n / n_unl) * var_f
    if denom <= 0.0:
        return 0.0
    lam = (2.0 * cov) / (2.0 * denom)
    return float(np.clip(lam, 0.0, 1.0))


def ppi_plus_mean(
    pred_unlabeled: list[float] | np.ndarray,
    y_human: list[float] | np.ndarray,
    pred_human: list[float] | np.ndarray,
    confidence: float = 0.95,
    lam: float | None = None,
) -> EstimateResult:
    """E3: PPI++ estimate of the mean.

    Args:
        pred_unlabeled: judge predictions on items WITHOUT human labels.
        y_human: human labels mapped to numbers.
        pred_human: judge predictions on exactly the human-labeled items.
        lam: fixed power-tuning parameter; None (default) estimates it.
    """
    f_unl = np.asarray(pred_unlabeled, dtype=float)
    y = np.asarray(y_human, dtype=float)
    f_lab = np.asarray(pred_human, dtype=float)
    n, n_unl = len(y), len(f_unl)
    if n < 2 or n_unl < 1 or len(f_lab) != n:
        raise ValueError("need ≥2 human labels and ≥1 unlabeled prediction")
    tuned = lam is None
    if lam is None:
        lam = power_tuned_lambda(y, f_lab, f_unl)
    lam = float(np.clip(lam, 0.0, 1.0))
    theta = float(lam * f_unl.mean() + (y - lam * f_lab).mean())
    se = float(np.sqrt(f_unl.var() * lam**2 / n_unl + (y - lam * f_lab).var() / n))
    hw = _z(confidence) * se
    note = f"lambda={lam:.4f} ({'tuned' if tuned else 'fixed'})"
    return EstimateResult(theta, hw, theta - hw, theta + hw, n, "E3-ppi-plus",
                          confidence, False, note)
