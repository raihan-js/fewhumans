"""E6 sequential CS + Wilson unit tests (spec §6.3)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fewhumans.estimators import human_proportion_wilson
from fewhumans.sequential import cs_human_only, cs_radius, sequential_mean


def test_cs_contains_running_mean_and_shrinks():
    rng = np.random.default_rng(0)
    vals = list(rng.random(200))
    widths = []
    for t in [10, 50, 100, 200]:
        r = cs_human_only(vals[:t])
        assert r.lo <= np.mean(vals[:t]) <= r.hi
        assert r.anytime_valid
        widths.append(r.halfwidth)
    assert widths[-1] < widths[0]


def test_cs_covers_fixed_truth_in_simulation():
    # Fixed (non-peeking) check: CS at t=200 covers 0.5 for fair coins w.h.p.
    rng = np.random.default_rng(2)
    vals = list(rng.integers(0, 2, size=200).astype(float))
    r = cs_human_only(vals)
    assert r.lo <= 0.5 <= r.hi


def test_cs_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="outside"):
        cs_human_only([0.5, 1.5])


def test_sequential_entry_point():
    r = sequential_mean([0.0, 1.0, 1.0])
    assert r.anytime_valid
    with pytest.raises(NotImplementedError):
        sequential_mean([0.0, 1.0], experimental=True)


def test_wilson_hand_computed_and_properties():
    r = human_proportion_wilson(60, 100)
    assert r.lo <= 0.6 <= r.hi
    assert 0.0 <= r.lo and r.hi <= 1.0
    assert r.method == "E1-wilson"
    with pytest.raises(ValueError):
        human_proportion_wilson(5, 4)


@given(st.integers(0, 50), st.integers(1, 50))
@settings(max_examples=100)
def test_wilson_always_valid_interval(k, n):
    k = min(k, n)
    r = human_proportion_wilson(k, n)
    assert 0.0 <= r.lo <= k / n <= r.hi <= 1.0
    assert cs_radius(n, 0.05) > 0
