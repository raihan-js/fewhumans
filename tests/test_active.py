"""E5 active inference unit tests (spec §6.3)."""

import numpy as np

from fewhumans.estimators import active_mean
from fewhumans.estimators.ppi import ppi_mean


def test_active_hand_computed():
    # N=4, two labeled with π=0.5. δ = f + (y−f)/0.5 on labeled, f elsewhere.
    pred = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    y = {"a": 1.0, "b": 1.0}
    probs = {"a": 0.5, "b": 0.5}
    r = active_mean(pred, y, probs)
    # δ = [1+(0)/.5, 0+(1)/.5, 1, 0] = [1, 2, 1, 0] -> mean 1.0
    assert abs(r.value - 1.0) < 1e-12
    assert r.n_human == 2
    assert r.method == "E5-active"


def test_active_uniform_equals_ppi_point():
    rng = np.random.default_rng(0)
    N, n = 200, 40
    f = {f"i{k}": float(rng.random()) for k in range(N)}
    lab = [f"i{k}" for k in rng.choice(N, n, replace=False)]
    y = {i: float(rng.random() < 0.5) for i in lab}
    probs = {i: n / N for i in lab}
    r = active_mean(f, y, probs)
    e2 = ppi_mean(list(f.values()), [y[i] for i in lab], [f[i] for i in lab])
    assert abs(r.value - e2.value) < 1e-12


def test_active_unbiased_monte_carlo():
    # Poisson sampling: average of θ̂ over replicates ≈ truth (smoke test).
    rng = np.random.default_rng(1)
    N = 500
    truth = rng.random(N)
    pred = {f"i{k}": float(np.clip(truth[k] + rng.normal(0, 0.2), 0, 1)) for k in range(N)}
    pi = np.where(np.arange(N) % 2 == 0, 0.3, 0.05)  # heterogeneous probs
    ests = []
    for rep in range(400):
        r = np.random.default_rng(1000 + rep)
        lab = [f"i{k}" for k in range(N) if r.random() < pi[k]]
        if len(lab) < 2:
            continue
        ests.append(active_mean(pred, {i: float(truth[int(i[1:])]) for i in lab},
                                {i: float(pi[int(i[1:])]) for i in lab}).value)
    assert abs(float(np.mean(ests)) - float(truth.mean())) < 0.02
