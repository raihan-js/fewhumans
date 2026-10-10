"""Routing R1–R5 unit tests (spec §6.3)."""

import numpy as np

from fewhumans.routing import (
    Disagreement,
    LearnedError,
    Mixture,
    Stratified,
    Uncertainty,
    Uniform,
    compute_features,
    vote_entropy,
)


def _ctx(n=200):
    ids = [f"i{k}" for k in range(n)]
    panel = {i: {"j1": "a" if k % 3 else "b", "j2": "a"} for k, i in enumerate(ids)}
    conf = {i: {"j1": float(0.9 if k % 2 else 0.55)} for k, i in enumerate(ids)}
    feats = compute_features(panel, conf)
    return ids, {"features": feats,
                 "stratum": {i: ("split" if k % 3 == 0 else "unanimous") for k, i in enumerate(ids)}}


def test_vote_entropy_and_features():
    assert vote_entropy(["a", "a"]) == 0.0
    assert abs(vote_entropy(["a", "b"]) - np.log(2)) < 1e-12
    feats = compute_features({"x": {"j1": "a", "j2": "b"}}, {"x": {"j1": 0.7}})
    assert feats["x"]["disagreement"] > 0 and abs(feats["x"]["uncertainty"] - 0.3) < 1e-12


def test_stratified_allocation_exact_probs():
    ids, ctx = _ctx()
    r = Stratified(allocation="proportional", seed=0)
    sel = r.select(ids, 20, 0, ctx)
    assert len(sel) == 20
    # Within-stratum probabilities equal; across strata differ in general.
    by_h: dict[str, set[float]] = {}
    for i, p in sel.items():
        by_h.setdefault(ctx["stratum"][i], set()).add(p)
    assert all(len(v) == 1 for v in by_h.values())


def test_active_policies_prefer_high_signal_and_respect_floor():
    ids, ctx = _ctx()
    for pol in (Uncertainty(seed=0), Disagreement(seed=0)):
        sel = pol.select(ids, 30, 0, ctx)
        assert sel, "Poisson batch realized empty (unlikely at n=30)"
        assert all(p >= 0.1 * pol.budget / len(ids) - 1e-12 for p in sel.values())
    # Disagreement sampling oversamples the split stratum vs uniform rate.
    r = Disagreement(seed=1)
    hits = [ctx["stratum"][i] for i in r.select(ids, 60, 0, ctx)]
    assert sum(h == "split" for h in hits) / len(hits) > 1 / 3


def test_learned_error_fallback_then_learns():
    ids, ctx = _ctx()
    r = LearnedError(seed=0)
    sel0 = r.select(ids, 20, 0, ctx)
    assert sel0  # uniform fallback works
    # Teach: items with high disagreement are wrong.
    rows = [{"disagreement": d, "uncertainty": 0.5} for d in [0.0] * 20 + [0.7] * 20]
    r.update(rows, [0] * 20 + [1] * 20)
    assert r.fitted_
    p = r.predict_proba([{"disagreement": 0.0, "uncertainty": 0.5},
                         {"disagreement": 0.7, "uncertainty": 0.5}])
    assert p[1] > p[0]


def test_mixture_probs_valid_and_epsilon_adapts():
    ids, ctx = _ctx()
    r = Mixture(seed=0, epsilon=0.5)
    sel = r.select(ids, 30, 0, ctx)
    assert sel and all(0.0 < p <= 1.0 for p in sel.values())
    e0 = r.epsilon
    feats = [{"disagreement": 0.0, "uncertainty": 0.5}] * 10 + [{"disagreement": 0.9, "uncertainty": 0.5}] * 10
    r.update(feats, [0] * 10 + [1] * 10)  # errors concentrate in high-disagreement
    assert r.epsilon > e0


def test_uniform_still_exact():
    r = Uniform(seed=0)
    ids = [f"i{k}" for k in range(50)]
    sel = r.select(ids, 5, 0, None)
    assert len(sel) == 5 and all(p == 1 / 50 for p in sel.values())
