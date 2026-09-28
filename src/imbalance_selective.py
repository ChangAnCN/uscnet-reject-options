"""Imbalance-aware selective-prediction metrics and selection strategies.

Metric definitions follow the recent selective-classification literature:

* **AURC** -- area under the risk-coverage curve, with risk the error rate on the
  accepted subset (Geifman & El-Yaniv, 2017).
* **AUGRC** -- area under the *generalised* risk-coverage curve, where the
  generalised risk normalises accepted errors by the size of the *whole*
  evaluation set rather than by the accepted subset, so that deferring a case is
  not credited as if it had never existed (Traub et al., NeurIPS 2024).
* **WC-AURC** -- *within-class* AURC: for each class the score ranks that
  class's own cases, the error rate among the accepted ones is traced over a
  common class-coverage grid, and the two curves are averaged. This separates
  how well a score ranks *inside* a class from how much coverage a selection
  strategy gives that class. The two components are reported separately as
  ``AURC_pos`` (an FNR curve over the positives) and ``AURC_neg`` (an FPR curve
  over the negatives), because averaging them assumes a symmetric cost that
  screening does not have.

  The argument that pooled metrics hide rejection behaviour under class
  imbalance is due to the class-aware selective-classification literature
  (J Biomed Inform 2026, which proposes CA-AURC / CA-AUGRC / AUIC). We do not
  reimplement those metrics here -- their treatment of class-coverage ranges a
  strategy never reaches is not recoverable from the description -- and instead
  report per-class coverage directly, which needs no convention.

Selection strategies compared:

* ``global``            -- one threshold on the uncertainty score (the classical rule).
* ``class_conditional`` -- coverage matched *within each predicted class*, so the
  same fraction is deferred in each, which prevents abstention from silently
  falling on the minority class.
* ``label``             -- the LABEL class-conditional conformal rule (Sadinle,
  Lei & Wasserman, 2019): a per-class score quantile calibrated on held-out data
  gives prediction sets, and the system abstains when the set is not a singleton.

All of these operate on cached predictions; none requires retraining.
"""
import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# risk definitions
# --------------------------------------------------------------------------
# Three risks are used, and they are kept distinct on purpose:
#
#   balanced error   0.5*FNR + 0.5*FPR recomputed *on the accepted subset*.
#                    This is the pooled risk used everywhere else in the paper,
#                    so the benchmark numbers are comparable to Tables 2 and 6.
#   class risk       the error rate within the accepted cases of one class --
#                    FNR for the positives, FPR for the negatives. Averaging the
#                    two curves is what makes CA-AURC imbalance-aware.
#   generalised risk accepted errors divided by the size of the *whole* class,
#                    so that deferring a case is not credited as if it had never
#                    existed (Traub et al., NeurIPS 2024).


def _err(y, p, thr):
    return ((p >= thr).astype(int) != y).astype(float)


def balanced_error_subset(y, p, thr, keep):
    """0.5*FNR + 0.5*FPR on the accepted subset; NaN if a class is absent."""
    ya, pa = y[keep], p[keep]
    pos, neg = ya == 1, ya == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    e = _err(ya, pa, thr)
    return float(0.5 * (e[pos].mean() + e[neg].mean()))


def _accept_order(u):
    return np.argsort(u, kind="stable")


# --------------------------------------------------------------------------
# selection strategies -> boolean accept mask at a target coverage
# --------------------------------------------------------------------------
def select_global(u, cov):
    """Classical rule: one threshold on the uncertainty score for everyone."""
    keep = np.zeros(len(u), bool)
    keep[_accept_order(u)[:max(int(round(cov * len(u))), 1)]] = True
    return keep


def select_class_conditional(u, pred, cov):
    """Accept the top ``cov`` fraction *within each predicted class*, so the
    same proportion is deferred in each and abstention cannot fall silently on
    the minority class."""
    keep = np.zeros(len(u), bool)
    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            continue
        k = max(int(round(cov * len(idx))), 1)
        keep[idx[np.argsort(u[idx], kind="stable")[:k]]] = True
    return keep


def label_conformal(p_cal, y_cal, p_test, alpha):
    """LABEL class-conditional conformal prediction sets for a binary target.

    For each class k the score is the model's probability of k, and the
    threshold is the alpha-quantile of that score over calibration cases of
    class k, so class-k coverage is at least 1 - alpha (Sadinle, Lei &
    Wasserman, 2019). Returns the two set-membership masks.
    """
    s_cal = np.column_stack([1 - p_cal, p_cal])
    s_test = np.column_stack([1 - p_test, p_test])
    inset = np.zeros_like(s_test, bool)
    for k in (0, 1):
        m = y_cal == k
        if m.sum() < 10:
            inset[:, k] = True
            continue
        n = int(m.sum())
        q = max(0.0, min(1.0, np.floor(alpha * (n + 1)) / n))
        t_k = np.quantile(s_cal[m, k], q, method="lower")
        inset[:, k] = s_test[:, k] >= t_k
    return inset


def select_label(p_cal, y_cal, p_test, alpha):
    """Abstain unless the LABEL prediction set is a singleton."""
    inset = label_conformal(p_cal, y_cal, p_test, alpha)
    singleton = inset.sum(1) == 1
    pred = np.where(inset[:, 1] & singleton, 1, 0)
    return singleton, pred


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------
def _auc(cov, val):
    cov, val = np.asarray(cov, float), np.asarray(val, float)
    ok = ~np.isnan(cov) & ~np.isnan(val)
    if ok.sum() < 2:
        return np.nan
    o = np.argsort(cov[ok])
    c, v = cov[ok][o], val[ok][o]
    span = c[-1] - c[0]
    return float(np.trapezoid(v, c) / span) if span > EPS else np.nan


def within_class_curve(y, p, thr, u, cls, cov_grid):
    """Risk-coverage curve *inside one class*, selecting that class's own cases
    by the score.

    This isolates how well a score ranks within a class from how much coverage a
    selection strategy chooses to give that class -- two things that pooled AURC
    conflates. Because every method is swept over the same class-coverage grid,
    the curves are directly comparable; integrating instead over each method's
    *realised* class coverage would reward a method for barely covering a class
    at all, since it would then be scored only over that class's easiest cases.
    """
    m = np.where(y == cls)[0]
    if len(m) < 5:
        return [np.nan] * len(cov_grid)
    order = m[np.argsort(u[m], kind="stable")]
    e = _err(y, p, thr)
    out = []
    for c in cov_grid:
        k = max(int(round(c * len(order))), 1)
        out.append(float(e[order[:k]].mean()))
    return out


def evaluate_strategy(y, p, thr, masks, cov_grid, u=None):
    """Score a family of accept masks (one per target coverage).

    Returns the pooled balanced-error risk-coverage curve, the two per-class
    curves, and the areas under all of them.
    """
    e = _err(y, p, thr)
    pos, neg = y == 1, y == 0
    n_cls = {1: int(pos.sum()), 0: int(neg.sum())}

    cov, risk, grisk = [], [], []
    ccov = {0: [], 1: []}
    crisk = {0: [], 1: []}
    cgrisk = {0: [], 1: []}

    for keep in masks:
        cov.append(keep.sum() / len(y))
        risk.append(balanced_error_subset(y, p, thr, keep))
        # generalised balanced risk: each class's accepted errors over that
        # class's full size, averaged over classes
        g = []
        for c in (0, 1):
            m = keep & (y == c)
            n_c = max(n_cls[c], 1)
            ccov[c].append(m.sum() / n_c)
            crisk[c].append(float(e[m].mean()) if m.sum() else np.nan)
            cgrisk[c].append(float(e[m].sum() / n_c))
            g.append(e[m].sum() / n_c)
        grisk.append(float(np.mean(g)))

    res = {
        "coverage": cov, "risk": risk, "grisk": grisk,
        "class_coverage": {c: ccov[c] for c in (0, 1)},
        "AURC": _auc(cov, risk),
        "AUGRC": _auc(cov, grisk),
        # realised-coverage class curves, kept for the coverage-fairness view
        "class_risk_realised": {c: crisk[c] for c in (0, 1)},
        "class_grisk_realised": {c: cgrisk[c] for c in (0, 1)},
        # Class-averaged metrics in the sense of Saglam et al. (JBI 2026):
        # each class's risk is integrated against that class's own realised
        # coverage (mean over the realised range) and the areas are averaged.
        # The range a policy never reaches is not extrapolated.
        "CA_AURC": float(np.nanmean([_auc(ccov[c], crisk[c]) for c in (0, 1)])),
        "CA_AUGRC": float(np.nanmean([_auc(ccov[c], cgrisk[c]) for c in (0, 1)])),
    }
    # within-class ranking quality of the score, on a common class-coverage grid
    if u is not None:
        wc = {c: within_class_curve(y, p, thr, u, c, cov_grid) for c in (0, 1)}
        res["WC_AURC"] = float(np.nanmean(
            [_auc(cov_grid, wc[c]) for c in (0, 1)]))
        res["AURC_pos"] = _auc(cov_grid, wc[1])   # FNR curve within positives
        res["AURC_neg"] = _auc(cov_grid, wc[0])   # FPR curve within negatives
        res["class_risk_within"] = wc

    # how differently the two classes are deferred, at ~80% overall coverage
    j = int(np.argmin(np.abs(np.array(cov) - 0.8)))
    res["cov_pos@80"] = ccov[1][j]
    res["cov_neg@80"] = ccov[0][j]
    res["disparity@80"] = ccov[0][j] - ccov[1][j]
    res["balerr@80"] = risk[j]
    return res
