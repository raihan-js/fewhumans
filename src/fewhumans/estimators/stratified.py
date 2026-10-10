"""E4: stratified PPI++ (spec §6.1).

Within each stratum h (panel-agreement strata or user strata, with a uniform
human sample inside the stratum) run E3/PPI++; combine with population weights:

  θ̂ = Σ_h W_h·θ̂_h,   V̂ = Σ_h W_h²·V̂_h,   W_h = N_h / N.

This is self-contained stratified-sampling algebra (independent stratum
estimates, variance of a weighted sum); the within-stratum estimator is E3
(see estimators/pp_plus.py for its provenance). Strata with fewer than 2 human
labels cannot contribute a variance estimate — the function raises instead of
inventing one (caller should route more labels there, e.g. R1/Neyman).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fewhumans.estimators.classical import EstimateResult, _z
from fewhumans.estimators.pp_plus import power_tuned_lambda


def panel_agreement_strata(panel_outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Split items into 'unanimous' vs 'split' by panel vote (R1/E4 default).

    Args:
        panel_outputs: {item_id: {judge_name: label}}.
    """
    out: dict[str, str] = {}
    for iid, labs in panel_outputs.items():
        vals = {str(v) for v in labs.values()}
        out[iid] = "unanimous" if len(vals) <= 1 else "split"
    return out


def neyman_allocation(
    stratum_sizes: Mapping[str, int],
    stratum_stds: Mapping[str, float] | None,
    batch: int,
) -> dict[str, int]:
    """Split a batch of `batch` labels across strata.

    Neyman (∝ N_h·S_h) when per-stratum stds are known, else proportional
    (∝ N_h). Largest-remainder rounding so the shares sum to exactly `batch`.
    """
    ids = list(stratum_sizes)
    total = sum(stratum_sizes.values())
    if total <= 0 or batch <= 0:
        return {i: 0 for i in ids}
    weights = {
        i: (stratum_sizes[i] * (stratum_stds.get(i, 1.0) if stratum_stds else 1.0))
        for i in ids
    }
    wsum = sum(weights.values())
    if wsum <= 0:
        weights = {i: float(stratum_sizes[i]) for i in ids}
        wsum = float(total)
    raw = {i: batch * weights[i] / wsum for i in ids}
    alloc = {i: int(np.floor(raw[i])) for i in ids}
    remainder = batch - sum(alloc.values())
    for i in sorted(ids, key=lambda k: raw[k] - alloc[k], reverse=True)[:remainder]:
        alloc[i] += 1
    return alloc


def stratified_ppi_plus(
    strata_all: Mapping[str, str],
    pred: Mapping[str, float],
    y_human: Mapping[str, float],
    confidence: float = 0.95,
) -> EstimateResult:
    """E4: stratified PPI++ estimate of the mean.

    Args:
        strata_all: {item_id: stratum} for ALL items (defines weights W_h).
        pred: judge prediction per item (all items).
        y_human: human label for labeled items (uniform sample WITHIN stratum).
    """
    strata = sorted(set(strata_all.values()))
    n_total = len(strata_all)
    theta, var = 0.0, 0.0
    n_human = 0
    details: dict[str, Any] = {}
    for h in strata:
        ids_h = [i for i, s in strata_all.items() if s == h]
        lab_h = [i for i in ids_h if i in y_human]
        if len(lab_h) < 2:
            raise ValueError(
                f"stratum {h!r} has {len(lab_h)} human labels (need ≥2); "
                "route more labels there before estimating"
            )
        unl_h = [i for i in ids_h if i not in y_human]
        if not unl_h:
            raise ValueError(f"stratum {h!r} has no unlabeled items left")
        y = np.array([y_human[i] for i in lab_h])
        f_lab = np.array([pred[i] for i in lab_h])
        f_unl = np.array([pred[i] for i in unl_h])
        lam = power_tuned_lambda(y, f_lab, f_unl)
        th_h = float(lam * f_unl.mean() + (y - lam * f_lab).mean())
        v_h = float(f_unl.var() * lam**2 / len(f_unl) + (y - lam * f_lab).var() / len(y))
        w_h = len(ids_h) / n_total
        theta += w_h * th_h
        var += w_h**2 * v_h
        n_human += len(lab_h)
        details[h] = {"n": len(ids_h), "n_human": len(lab_h), "lambda": lam}
    hw = _z(confidence) * float(np.sqrt(max(var, 0.0)))
    return EstimateResult(theta, hw, theta - hw, theta + hw, n_human,
                          "E4-stratified-ppi-plus", confidence, False,
                          f"strata={details}")
