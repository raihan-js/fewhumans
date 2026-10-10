"""E4 stratified PPI++ unit tests (spec §6.3)."""

import pytest

from fewhumans.estimators import (
    neyman_allocation,
    panel_agreement_strata,
    stratified_ppi_plus,
)


def test_panel_agreement_strata():
    panel = {"a": {"j1": "x", "j2": "x"}, "b": {"j1": "x", "j2": "y"}}
    assert panel_agreement_strata(panel) == {"a": "unanimous", "b": "split"}


def test_neyman_allocation_sums_and_proportional():
    sizes = {"a": 80, "b": 20}
    alloc = neyman_allocation(sizes, None, 10)
    assert sum(alloc.values()) == 10
    assert alloc == {"a": 8, "b": 2}
    alloc_n = neyman_allocation(sizes, {"a": 1.0, "b": 3.0}, 10)
    assert sum(alloc_n.values()) == 10
    assert alloc_n["b"] > alloc["b"]  # high-variance stratum gets more


def test_stratified_hand_computed_two_strata():
    # Stratum A: judge perfect -> rectifier 0, estimate = stratum mean.
    # Stratum B: judge constant -> falls back to human mean within stratum.
    strata = {f"a{i}": "A" for i in range(6)} | {f"b{i}": "B" for i in range(6)}
    pred = {f"a{i}": float(i % 2) for i in range(6)} | {f"b{i}": 0.0 for i in range(6)}
    y = {"a0": 0.0, "a1": 1.0, "b0": 1.0, "b1": 0.0}
    r = stratified_ppi_plus(strata, pred, y)
    # A: perfect judge, mean over 6 items = 0.5; B: human mean 0.5. Total 0.5.
    assert abs(r.value - 0.5) < 1e-9
    assert r.n_human == 4
    assert r.method == "E4-stratified-ppi-plus"


def test_stratified_refuses_singleton_stratum():
    strata = {"a0": "A", "a1": "A", "b0": "B"}
    pred = {"a0": 0.0, "a1": 1.0, "b0": 0.0}
    with pytest.raises(ValueError, match="need ≥2"):
        stratified_ppi_plus(strata, pred, {"a0": 0.0, "b0": 1.0})
