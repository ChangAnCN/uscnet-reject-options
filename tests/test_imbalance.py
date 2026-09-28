"""Correctness checks for ``imbalance_selective.py``.

Each check compares an implementation against a hand computation, a closed-form
answer, or a case whose result is known by construction. Run with
``python test_imbalance.py``.
"""
import numpy as np

import imbalance_selective as I
import selective as S

rng = np.random.RandomState(0)


def _toy(n=8000, a=-4.0, b=1.0):
    """A faithful miniature of the real setting: a calibrated but weak
    classifier on a low-prevalence target, so the predicted probabilities stay
    small and the Youden threshold lands far below 0.5 -- which is the regime
    the paper is about. A well-separated toy problem would put the threshold
    near 0.5 and hide the effect being tested.
    """
    z = rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(a + b * z)))
    y = rng.binomial(1, p)
    return y, p, S.youden_threshold(y, p)


def test_pooled_risk_matches_selective_module():
    """The benchmark's pooled risk must be the same quantity the rest of the
    paper reports, or the tables are not comparable."""
    y, p, thr = _toy()
    cov = np.arange(0.3, 1.001, 0.1)
    masks = [I.select_global(rng.uniform(0, 1, len(y)), c) for c in cov]
    r = I.evaluate_strategy(y, p, thr, masks, cov)
    for i, keep in enumerate(masks):
        assert abs(r["risk"][i] - S.balanced_error(y[keep], p[keep], thr)) < 1e-12


def test_metrics_are_bounded():
    y, p, thr = _toy()
    cov = np.arange(0.2, 1.001, 0.05)
    u = rng.uniform(0, 1, len(y))
    r = I.evaluate_strategy(y, p, thr, [I.select_global(u, c) for c in cov],
                            cov, u=u)
    for k in ("AURC", "AUGRC", "WC_AURC", "AURC_pos", "AURC_neg"):
        assert 0.0 <= r[k] <= 1.0, (k, r[k])


def test_full_coverage_gives_parity():
    y, p, thr = _toy()
    r = I.evaluate_strategy(y, p, thr, [np.ones(len(y), bool)], [1.0])
    assert abs(r["cov_pos@80"] - 1) < 1e-9
    assert abs(r["cov_neg@80"] - 1) < 1e-9
    assert abs(r["disparity@80"]) < 1e-9


def test_class_conditional_defers_equally():
    y, p, thr = _toy()
    u = rng.uniform(0, 1, len(y))
    pred = (p >= thr).astype(int)
    keep = I.select_class_conditional(u, pred, 0.8)
    for c in (0, 1):
        m = pred == c
        assert abs(keep[m].mean() - 0.8) < 0.02


def test_label_conformal_attains_class_coverage():
    """LABEL must reach at least 1 - alpha coverage in *each* class."""
    n = 20000
    yc = rng.binomial(1, 0.05, n)
    pc = np.clip(rng.beta(1, 8, n) + 0.35 * yc, 0, 1)
    yt = rng.binomial(1, 0.05, n)
    pt = np.clip(rng.beta(1, 8, n) + 0.35 * yt, 0, 1)
    for alpha in (0.05, 0.1, 0.2):
        inset = I.label_conformal(pc, yc, pt, alpha)
        for k in (0, 1):
            assert inset[yt == k, k].mean() > 1 - alpha - 0.02


def test_confidence_score_starves_the_minority_class():
    """The paper's central claim, on synthetic data where the answer is known:
    a score centred on p = 0.5 defers the minority class first when the
    operating threshold is far below 0.5."""
    y, p, thr = _toy(n=40000, a=-4.5, b=1.2)
    assert y.mean() < 0.05 and thr < 0.05, (y.mean(), thr)
    cov = np.arange(0.2, 1.001, 0.05)
    msp = S.u_msp(p)
    marg = S.u_margin_threshold(p, thr)
    r_msp = I.evaluate_strategy(y, p, thr, [I.select_global(msp, c) for c in cov],
                                cov, u=msp)
    r_mar = I.evaluate_strategy(y, p, thr, [I.select_global(marg, c) for c in cov],
                                cov, u=marg)
    assert r_msp["disparity@80"] > 0.10, r_msp["disparity@80"]
    assert r_mar["disparity@80"] < r_msp["disparity@80"]
    assert r_mar["cov_pos@80"] > r_msp["cov_pos@80"]


def test_within_class_curve_is_strategy_independent():
    """The within-class decomposition measures the score, not the policy, so it
    must not change when the selection rule changes."""
    y, p, thr = _toy()
    cov = np.arange(0.2, 1.001, 0.05)
    u = S.u_margin_threshold(p, thr)
    pred = (p >= thr).astype(int)
    a = I.evaluate_strategy(y, p, thr, [I.select_global(u, c) for c in cov], cov, u=u)
    b = I.evaluate_strategy(
        y, p, thr, [I.select_class_conditional(u, pred, c) for c in cov], cov, u=u)
    assert abs(a["WC_AURC"] - b["WC_AURC"]) < 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
