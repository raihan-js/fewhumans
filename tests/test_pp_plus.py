"""E3 cross-check vs ppi_py (reference) + properties (spec §6.1, §6.3)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fewhumans.estimators import human_only, power_tuned_lambda, ppi_plus_mean
from fewhumans.estimators.ppi import ppi_mean


def _synth(seed=0, n=60, n_unl=500):
    rng = np.random.default_rng(seed)
    f_unl = rng.random(n_unl)
    f_lab = rng.random(n)
    y = (rng.random(n) < 0.2 + 0.6 * f_lab).astype(float)
    return f_unl, y, f_lab


def test_matches_ppi_py_point_and_interval():
    ppi = pytest.importorskip("ppi_py.ppi")
    from scipy import stats as _stats

    z = 1.959963984540054
    for seed in range(5):
        f_unl, y, f_lab = _synth(seed)
        lam_ref = float(
            ppi._calc_lam_glm(
                y.reshape(-1, 1), f_lab.reshape(-1, 1), f_unl.reshape(-1, 1),
                np.eye(1), clip=True,
            )
        )
        pt_ref = float(np.asarray(ppi.ppi_mean_pointestimate(y, f_lab, f_unl)).squeeze())
        lo, hi = ppi.ppi_mean_ci(y, f_lab, f_unl, alpha=0.05)
        hw_ref = (float(np.asarray(hi).squeeze()) - float(np.asarray(lo).squeeze())) / 2
        r = ppi_plus_mean(f_unl, y, f_lab, confidence=0.95)
        assert abs(power_tuned_lambda(y, f_lab, f_unl) - lam_ref) < 1e-12
        assert abs(r.value - pt_ref) < 1e-12
        # Same SE as ppi_py (verified ratio 1.0 at z); we use Student-t.
        t = float(_stats.t.ppf(0.975, len(y) - 1))
        assert abs(r.halfwidth - hw_ref * (t / z)) < 1e-9


def test_e2_point_matches_ppi_py_lam1():
    ppi = pytest.importorskip("ppi_py.ppi")
    f_unl, y, f_lab = _synth()
    f_all = np.concatenate([f_lab, f_unl])
    # Same formula (all-N prediction mean + rectifier mean) matches exactly.
    pt_ref = float(np.asarray(ppi.ppi_mean_pointestimate(y, f_lab, f_all, lam=1.0)).squeeze())
    assert abs(ppi_mean(list(f_all), list(y), list(f_lab)).value - pt_ref) < 1e-12


def test_lambda_in_unit_interval_and_constant_judge_gives_classical():
    f_unl, y, _ = _synth()
    c = 0.4
    lam = power_tuned_lambda(y, np.full_like(y, c), f_unl)
    assert lam == 0.0
    r = ppi_plus_mean(f_unl, y, np.full_like(y, c))
    assert abs(r.value - human_only(list(y)).value) < 1e-12


def test_fixed_lam0_is_classical_lam1_is_ppi():
    f_unl, y, f_lab = _synth()
    r0 = ppi_plus_mean(f_unl, y, f_lab, lam=0.0)
    assert abs(r0.value - human_only(list(y)).value) < 1e-12
    r1 = ppi_plus_mean(f_unl, y, f_lab, lam=1.0)
    # E3(λ=1) uses the unlabeled-only prediction mean while E2 uses all-N;
    # the difference is exactly the mean shift (O(n/N), vanishes asymptotically).
    f_all = np.concatenate([f_lab, f_unl])
    expect = float(f_unl.mean() - f_all.mean())
    assert abs((r1.value - ppi_mean(list(f_all), list(y), list(f_lab)).value) - expect) < 1e-12


@given(
    st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1)), min_size=2, max_size=30),
    st.lists(st.floats(0, 1), min_size=1, max_size=30),
)
@settings(max_examples=50)
def test_order_invariance(pairs, f_unl):
    y = [p[0] for p in pairs]
    f_lab = [p[1] for p in pairs]
    y2 = list(reversed(y))
    f2 = list(reversed(f_lab))
    u2 = list(reversed(f_unl))
    r1 = ppi_plus_mean(f_unl, y, f_lab)
    r2 = ppi_plus_mean(u2, y2, f2)
    assert abs(r1.value - r2.value) < 1e-9
