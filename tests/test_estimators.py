"""Hand-computed estimator checks + properties (spec §6.3)."""

import math

from scipy import stats

from fewhumans.estimators import human_only, judge_only, ppi_mean


def _t(df: int, conf: float = 0.95) -> float:
    return float(stats.t.ppf(0.5 + conf / 2.0, df))


def test_judge_only_hand_computed():
    r = judge_only([1.0, 0.0, 1.0, 1.0], confidence=0.95)
    assert r.value == 0.75
    assert r.n_human == 0
    assert "NOT the truth" in r.note
    # std(ddof=1)=0.5, se=0.25, Student-t df=3
    assert abs(r.halfwidth - _t(3) * 0.25) < 1e-9


def test_human_only_hand_computed():
    r = human_only([1.0, 1.0, 0.0, 0.0, 1.0], confidence=0.95)
    assert r.value == 0.6
    assert r.method == "E1-human-only"
    assert r.lo < r.value < r.hi
    # var(ddof=1)=0.3, se=sqrt(0.3/5), Student-t df=4
    assert abs(r.halfwidth - _t(4) * math.sqrt(0.3 / 5)) < 1e-9


def test_ppi_hand_computed():
    # N=4 judge preds, n=2 humans. theta = mean_all + mean(y - f)
    pred_all = [1.0, 1.0, 0.0, 0.0]  # mean 0.5
    y = [1.0, 0.0]
    f = [1.0, 1.0]  # rectifier [-0, -1] mean -0.5 -> theta 0.0
    r = ppi_mean(pred_all, y, f)
    assert r.value == 0.0
    assert r.method == "E2-ppi"
    assert r.n_human == 2


def test_ppi_equals_human_only_when_judge_constant():
    # Judge predicts constant c: PPI = c + mean(Y - c) = mean(Y).
    y = [1.0, 0.0, 1.0, 1.0, 0.0]
    c = 0.3
    r_ppi = ppi_mean([c] * 20, y, [c] * len(y))
    r_h = human_only(y)
    assert abs(r_ppi.value - r_h.value) < 1e-12


def test_intervals_shrink_with_n():
    import random

    rng = random.Random(0)
    small = [float(rng.random() < 0.7) for _ in range(10)]
    big = small + [float(rng.random() < 0.7) for _ in range(90)]
    assert human_only(big).halfwidth < human_only(small).halfwidth


def test_order_invariance():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert human_only(vals).value == human_only(list(reversed(vals))).value
    r1 = ppi_mean([1.0, 0.0, 1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    r2 = ppi_mean([0.0, 1.0, 0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    assert abs(r1.value - r2.value) < 1e-12
