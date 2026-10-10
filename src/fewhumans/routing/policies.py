"""Routing policies (spec §6.2). All produce known, logged sampling probabilities.

Sampling models (exact — this is what keeps E5 unbiased):
- R0/R1: fixed-size random sample (simple / stratified); π_i = 1/M (·m_h/M_h).
- R2–R5: Poisson (independent Bernoulli) sampling, the paper's own model
  (Zrnic & Candès 2024 §4: ξ_i ∼ Bern(π(X_i))). π_i ∝ signal weight with the
  §6.2 floor π_i ≥ 0.1·budget/N; the realized batch size varies around the
  target (queue() returns the realized set). Every returned item carries its π_i.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RoutingCtx = dict[str, Any]
FLOOR_FRAC = 0.1


def vote_entropy(labels: Sequence[Any]) -> float:
    """Entropy (nats) of the judges' vote distribution (categorical)."""
    vals = [str(v) for v in labels]
    n = len(vals)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return float(-sum((c / n) * math.log(c / n) for c in counts.values()))


def _is_numeric(vals: Sequence[Any]) -> bool:
    try:
        for v in vals:
            float(v)
        return True
    except (TypeError, ValueError):
        return False


def compute_features(
    panel_outputs: Mapping[str, Mapping[str, Any]],
    panel_conf: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict[str, float]]:
    """Per-item disagreement + uncertainty from panel outputs.

    disagreement: vote entropy (categorical) or vote std (numeric).
    uncertainty: 1 − confidence of the first judge (0.5 default when missing).
    """
    feats: dict[str, dict[str, float]] = {}
    for iid, labs in panel_outputs.items():
        vals = list(labs.values())
        if _is_numeric(vals):
            arr = np.array([float(v) for v in vals])
            dis = float(arr.std()) if len(arr) > 1 else 0.0
        else:
            dis = vote_entropy(vals)
        confs = panel_conf.get(iid, {})
        c0 = next(iter(confs.values()), None)
        unc = 1.0 - float(c0) if c0 is not None else 0.5
        feats[iid] = {"disagreement": dis, "uncertainty": unc}
    return feats


def _poisson_probs(
    ids: list[str], weights: list[float], target: int, budget: int, floor_frac: float = FLOOR_FRAC
) -> dict[str, float]:
    """Scale weights to Poisson inclusion probabilities (exact, logged)."""
    n = len(ids)
    if n == 0:
        return {}
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * n
        wsum = float(n)
    floor = floor_frac * budget / n if n else 0.0
    return {i: min(1.0, max(floor, target * w / wsum)) for i, w in zip(ids, weights)}


def _poisson_draw(ids: list[str], probs: dict[str, float], seed: str) -> dict[str, float]:
    rng = random.Random(seed)
    return {i: probs[i] for i in ids if rng.random() < probs[i]}


@dataclass
class Uniform:
    """R0: uniform random fixed-size sample."""

    budget: int = 300
    seed: int = 0
    id: str = "R0"

    def select(
        self, unlabeled_ids: list[str], n: int, round_no: int = 0, ctx: RoutingCtx | None = None
    ) -> dict[str, float]:
        rng = random.Random(f"{self.seed}-{round_no}")
        pool = list(unlabeled_ids)
        rng.shuffle(pool)
        chosen = pool[:n]
        p = 1.0 / max(len(unlabeled_ids), 1)
        return {cid: p for cid in chosen}


@dataclass
class Stratified:
    """R1: stratified by panel agreement (or user strata), fixed allocation."""

    budget: int = 300
    seed: int = 0
    allocation: str = "proportional"  # or "neyman"
    id: str = "R1"

    def select(
        self, unlabeled_ids: list[str], n: int, round_no: int = 0, ctx: RoutingCtx | None = None
    ) -> dict[str, float]:
        from fewhumans.estimators.stratified import neyman_allocation

        strata: dict[str, str] = (ctx or {}).get("stratum", {})
        by_h: dict[str, list[str]] = {}
        for i in unlabeled_ids:
            by_h.setdefault(strata.get(i, "all"), []).append(i)
        stds = (ctx or {}).get("stratum_std") if self.allocation == "neyman" else None
        alloc = neyman_allocation({h: len(v) for h, v in by_h.items()}, stds, n)
        rng = random.Random(f"{self.seed}-{round_no}")
        out: dict[str, float] = {}
        for h, members in by_h.items():
            m = min(alloc.get(h, 0), len(members))
            chosen = rng.sample(members, m) if m else []
            p = m / len(members) if members else 0.0
            for c in chosen:
                out[c] = p
        return out


@dataclass
class Uncertainty:
    """R2: Poisson sampling ∝ single-judge uncertainty (1 − confidence)."""

    budget: int = 300
    seed: int = 0
    floor_frac: float = FLOOR_FRAC
    base: float = 0.05
    id: str = "R2"

    def select(
        self, unlabeled_ids: list[str], n: int, round_no: int = 0, ctx: RoutingCtx | None = None
    ) -> dict[str, float]:
        feats = (ctx or {}).get("features", {})
        weights = [self.base + float(feats.get(i, {}).get("uncertainty", 0.5)) for i in unlabeled_ids]
        probs = _poisson_probs(unlabeled_ids, weights, n, self.budget, self.floor_frac)
        return _poisson_draw(unlabeled_ids, probs, f"{self.seed}-{round_no}")


@dataclass
class Disagreement:
    """R3: Poisson sampling ∝ panel disagreement (vote entropy / variance)."""

    budget: int = 300
    seed: int = 0
    floor_frac: float = FLOOR_FRAC
    base: float = 0.05
    id: str = "R3"

    def select(
        self, unlabeled_ids: list[str], n: int, round_no: int = 0, ctx: RoutingCtx | None = None
    ) -> dict[str, float]:
        feats = (ctx or {}).get("features", {})
        weights = [self.base + float(feats.get(i, {}).get("disagreement", 0.0)) for i in unlabeled_ids]
        probs = _poisson_probs(unlabeled_ids, weights, n, self.budget, self.floor_frac)
        return _poisson_draw(unlabeled_ids, probs, f"{self.seed}-{round_no}")


# Phase-1 name kept as an alias for compatibility.
Active = Disagreement


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))


@dataclass
class LearnedError:
    """R4: Poisson sampling ∝ predicted P(judge wrong) from a tiny logistic model.

    Features per item: [1, disagreement, uncertainty]. The model is trained
    online on human labels collected so far (predicting "judge wrong"), refit
    only between batches via update(); probabilities are logged. Before any
    labels exist it falls back to uniform weights (documented).
    """

    budget: int = 300
    seed: int = 0
    floor_frac: float = FLOOR_FRAC
    base: float = 0.05
    l2: float = 1.0
    lr: float = 0.5
    iters: int = 500
    id: str = "R4"
    coef_: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)

    fitted_: bool = field(default=False, repr=False)

    def _design(self, feats: list[dict[str, float]]) -> np.ndarray:
        x = np.array([[1.0, float(f.get("disagreement", 0.0)), float(f.get("uncertainty", 0.5))] for f in feats])
        s = x[:, 1:].std(axis=0)
        s[s == 0] = 1.0
        x[:, 1:] /= s
        return x

    def update(self, feats: list[dict[str, float]], wrong: list[int]) -> None:
        """Refit the error predictor between batches (deterministic GD)."""
        x = self._design(feats)
        yw = np.array(wrong, dtype=float)
        if len(set(wrong)) < 2 or len(yw) == 0:
            return  # one class only — keep previous coefficients
        w = np.zeros(3)
        for _ in range(self.iters):
            p = _sigmoid(x @ w)
            grad = x.T @ (p - yw) / len(yw) + self.l2 * np.r_[0.0, w[1:]] / len(yw)
            w -= self.lr * grad
        self.coef_ = w
        self.fitted_ = True

    def predict_proba(self, feats: list[dict[str, float]]) -> list[float]:
        x = self._design(feats)
        return [float(v) for v in _sigmoid(x @ self.coef_)]

    def select(
        self, unlabeled_ids: list[str], n: int, round_no: int = 0, ctx: RoutingCtx | None = None
    ) -> dict[str, float]:
        feats = (ctx or {}).get("features", {})
        rows = [feats.get(i, {}) for i in unlabeled_ids]
        if self.fitted_:
            weights = [self.base + p for p in self.predict_proba(rows)]
        else:
            weights = [1.0] * len(unlabeled_ids)
        probs = _poisson_probs(unlabeled_ids, weights, n, self.budget, self.floor_frac)
        return _poisson_draw(unlabeled_ids, probs, f"{self.seed}-{round_no}")


@dataclass
class Mixture:
    """R5: (1−ε)·uniform + ε·disagreement mixture (Poisson), ε adapted online.

    Adaptation is a documented heuristic inspired by the FTRL tuning of
    Sfyraki & Wang (2026 §4): after each batch, ε moves toward the half
    (above/below median disagreement) with the higher observed judge-error
    rate. No regret is claimed; the study (Phase 5) measures whether it helps.
    """

    budget: int = 300
    seed: int = 0
    floor_frac: float = FLOOR_FRAC
    base: float = 0.05
    epsilon: float = 0.5
    eta: float = 0.2
    eps_min: float = 0.05
    eps_max: float = 0.95
    id: str = "R5"

    def update(self, feats: list[dict[str, float]], wrong: list[int]) -> None:
        if not wrong:
            return
        dis = np.array([float(f.get("disagreement", 0.0)) for f in feats])
        w = np.array(wrong, dtype=float)
        med = float(np.median(dis))
        hi = w[dis > med]
        lo = w[dis <= med]
        if len(hi) == 0 or len(lo) == 0:
            return
        step = self.eta * (float(hi.mean()) - float(lo.mean()))
        self.epsilon = float(np.clip(self.epsilon + step, self.eps_min, self.eps_max))

    def select(
        self, unlabeled_ids: list[str], n: int, round_no: int = 0, ctx: RoutingCtx | None = None
    ) -> dict[str, float]:
        feats = (ctx or {}).get("features", {})
        weights = []
        for i in unlabeled_ids:
            d = float(feats.get(i, {}).get("disagreement", 0.0))
            weights.append((1.0 - self.epsilon) * 1.0 + self.epsilon * (self.base + d) / (self.base + 1.0))
        probs = _poisson_probs(unlabeled_ids, weights, n, self.budget, self.floor_frac)
        return _poisson_draw(unlabeled_ids, probs, f"{self.seed}-{round_no}")
