"""Correctness checks for the metric implementations in ``selective.py``.

Each check compares an implementation against a hand computation, a closed-form
answer, or a case whose result is known by construction.  Run with
``python test_selective.py``.
"""
import numpy as np

import selective as S

rng = np.random.RandomState(0)


def test_balanced_error():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .4, .1, .8, .2, .1, .05, .02])
    assert abs(S.balanced_error(y, p, 0.5) - 0.5 * (2 / 3 + 1 / 5)) < 1e-12


def test_youden_is_maximal():
    y = rng.binomial(1, 0.05, 4000)
    p = np.clip(rng.beta(1, 9, 4000) + 0.35 * y, 0, 1)
    thr = S.youden_threshold(y, p)
    J = lambda t: sum(S.sens_spec(y, p, t)) - 1
    assert J(thr) >= max(J(t) for t in np.unique(p)) - 1e-9


def test_temperature_recovers_known_distortion():
    z = rng.normal(0, 2, 20000)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    p = 1 / (1 + np.exp(-z / 2.5))          # over-smoothed; true T is 1/2.5
    assert abs(S.fit_temperature(y, p) - 0.4) < 0.06


def test_ece_vanishes_when_calibrated():
    q = rng.uniform(0, 1, 200000)
    y = rng.binomial(1, q)
    assert S.ece_equal_mass(y, q) < 0.005


def test_delong():
    y = rng.binomial(1, .3, 3000)
    a = rng.normal(y, 1)
    assert S.delong_test(y, a, a + rng.normal(0, 1e-9, 3000))[2] > 0.9
    assert S.delong_test(y, a, rng.normal(y, 3))[2] < 1e-6


def test_margin_is_minimal_at_the_threshold():
    g = np.linspace(.001, .999, 999)
    assert abs(g[np.argmax(S.u_margin_threshold(g, 0.02))] - 0.02) < 0.002


def test_risk_coverage_accepts_easiest_first():
    y = rng.binomial(1, .1, 5000)
    p = np.clip(rng.beta(1, 6, 5000) + 0.4 * y, 0, 1)
    thr = S.youden_threshold(y, p)
    err = ((p >= thr).astype(int) != y).astype(float)
    cov = np.arange(.1, 1.001, .05)
    oracle = S.aurc(S.risk_coverage(y, p, err + rng.uniform(0, 1e-6, 5000), thr, cov), cov)
    random = S.aurc(S.risk_coverage(y, p, rng.uniform(0, 1, 5000), thr, cov), cov)
    assert oracle < random


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
