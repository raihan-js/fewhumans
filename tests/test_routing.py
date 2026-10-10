"""Routing: known probabilities, floor, reproducibility."""

from fewhumans.routing import Active, Uniform


def test_uniform_probs():
    r = Uniform(budget=300, seed=0)
    ids = [f"i{k}" for k in range(100)]
    probs = r.select(ids, 10, round_no=0)
    assert len(probs) == 10
    assert all(abs(p - 1 / 100) < 1e-12 for p in probs.values())


def test_uniform_reproducible():
    r = Uniform(seed=42)
    ids = [f"i{k}" for k in range(50)]
    assert r.select(ids, 5, 0) == r.select(ids, 5, 0)


def test_active_floor():
    r = Active(signal="disagreement", budget=300, seed=0, prob_floor=0.1)
    ids = [f"i{k}" for k in range(100)]
    probs = r.select(ids, 5, 0)
    floor = 0.1 * 300 / 100
    assert all(p >= floor - 1e-12 for p in probs.values())
