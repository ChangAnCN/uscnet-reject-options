"""Metrics, uncertainty scores and the cue-consistency selective rule.

Conventions used throughout
---------------------------
* ``p``  : predicted probability of the *target* finding (positive class).
* ``y``  : binary ground truth for the target finding.
* Risk for selective prediction is the **balanced error**
  ``0.5*FNR + 0.5*FPR`` evaluated on the accepted subset.  Plain 0/1 error is
  degenerate on CXR8 pneumonia (2.2% prevalence), where predicting "negative"
  everywhere already scores 97.8%.
* Calibration is measured on the positive-class probability with equal-mass
  bins, which is the appropriate form under heavy class imbalance.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


# --------------------------------------------------------------------------
# calibration / accuracy metrics
# --------------------------------------------------------------------------
def ece_equal_mass(y, p, n_bins=15):
    """Expected calibration error of the positive-class probability, using
    equal-mass (quantile) bins."""
    n = len(p)
    order = np.argsort(p)
    e = 0.0
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        e += len(b) / n * abs(p[b].mean() - y[b].mean())
    return float(e)


def ece_confidence(y, p, n_bins=15):
    """Classical Guo et al. confidence-based ECE (equal-width bins)."""
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def balanced_error(y, p, thr):
    """0.5*FNR + 0.5*FPR at threshold ``thr``; NaN if a class is absent."""
    pred = (p >= thr).astype(int)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    fnr = 1.0 - pred[pos].mean()
    fpr = pred[neg].mean()
    return float(0.5 * (fnr + fpr))


def sens_spec(y, p, thr):
    pred = (p >= thr).astype(int)
    pos, neg = y == 1, y == 0
    se = float(pred[pos].mean()) if pos.sum() else np.nan
    sp = float(1 - pred[neg].mean()) if neg.sum() else np.nan
    return se, sp


def youden_threshold(y, p):
    """Operating threshold maximising sensitivity + specificity - 1."""
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P, N = ys.sum(), len(ys) - ys.sum()
    j = tp / max(P, 1) - fp / max(N, 1)
    k = int(np.argmax(j))
    return float(p[order][k])


def fit_temperature(y, p):
    """Single-parameter temperature scaling fitted by NLL on held-out data."""
    z = logit(np.clip(p, EPS, 1 - EPS))

    def obj(logT):
        return nll(y, expit(z / np.exp(logT)))

    r = minimize_scalar(obj, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(p, T):
    return expit(logit(np.clip(p, EPS, 1 - EPS)) / T)


# --------------------------------------------------------------------------
# uncertainty scores  (higher == less trustworthy)
# --------------------------------------------------------------------------
def bin_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def u_msp(p):
    """1 - max softmax response."""
    return 1.0 - np.maximum(p, 1 - p)


def u_entropy(p):
    return bin_entropy(p)


def u_margin_threshold(p, thr):
    """Threshold-aware ambiguity: negative distance from the *operating*
    threshold in logit space.

    MSP and predictive entropy both measure distance from p = 0.5.  Under heavy
    class imbalance the operating threshold sits far from 0.5 (tau ~ 0.005 for
    CXR8 pneumonia), so those scores label every predicted positive "uncertain"
    and abstention preferentially discards the minority class.  Measuring
    ambiguity relative to tau instead makes the score consistent with the
    decision actually being taken.
    """
    z = logit(np.clip(p, EPS, 1 - EPS))
    return -np.abs(z - logit(np.clip(thr, EPS, 1 - EPS)))


def u_mc_entropy(mc):
    """Predictive entropy of the MC-dropout mean.  mc: [T, N]."""
    return bin_entropy(mc.mean(0))


def u_mc_var(mc):
    return mc.var(0)


def u_mc_bald(mc):
    """Mutual information (BALD): total entropy - expected entropy."""
    return bin_entropy(mc.mean(0)) - bin_entropy(mc).mean(0)


def jsd_bernoulli(p, q):
    """Jensen-Shannon divergence between two Bernoulli distributions."""
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    m = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# --------------------------------------------------------------------------
# structured cue bank
# --------------------------------------------------------------------------
class CueBank:
    """Structured clinical cue layer.

    Two sets of cue weights are estimated, both on the **training** split and
    neither hand-set:

    ``fit_association``  L2-regularised logistic regression of the target on the
        ground-truth co-occurring findings.  These coefficients are the
        clinically readable part of the cue bank -- they say which findings the
        data associate with the target -- and are reported for interpretation.

    ``fit_evidence``  the same regression fitted on the *predicted* cue
        probabilities of the training images.  This is what the selective rule
        uses: it yields ``s_cue``, an evidence-only estimate of the target that
        lives on the same probability scale as the image head, so that the two
        can be compared case by case.  Fitting in label space and applying in
        probability space would leave a systematic offset that swamps the
        case-level disagreement the score is meant to detect.
    """

    def __init__(self, target_idx, cue_idx, C=1.0):
        self.t, self.cues, self.C = target_idx, cue_idx, C
        self.lr = None          # evidence model, prediction space
        self.lr_assoc = None    # association model, label space

    @staticmethod
    def _fit(X, y, C):
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        return LogisticRegression(C=C, max_iter=2000).fit(X[ok], y[ok])

    def fit_association(self, Y_train):
        self.lr_assoc = self._fit(Y_train[:, self.cues], Y_train[:, self.t], self.C)
        return self

    def fit_evidence(self, P_train, y_train):
        self.lr = self._fit(P_train[:, self.cues], y_train, self.C)
        return self

    @property
    def weights(self):
        """Evidence-model coefficients (prediction space)."""
        return self.lr.coef_[0]

    @property
    def assoc_weights(self):
        """Association coefficients (label space), for interpretation."""
        return self.lr_assoc.coef_[0]

    def evidence(self, P):
        """Evidence-only target probability from predicted cue probabilities."""
        return self.lr.predict_proba(P[:, self.cues])[:, 1]

    def fit_cue_threshold(self, P_val, y_val):
        """Operating threshold for the evidence-only estimate, so that its
        log-odds can be signed the same way as the image head's."""
        self.thr_cue = youden_threshold(y_val, self.evidence(P_val))
        return self

    def inconsistency(self, P, mode="logit"):
        """Symmetric disagreement between the image head and the cue evidence.

        Kept as a baseline/ablation.  ``logit`` is the scale-free absolute
        log-odds gap; ``jsd`` the bounded probability-space alternative.
        """
        p, s = P[:, self.t], self.evidence(P)
        if mode == "jsd":
            return jsd_bernoulli(p, s)
        return np.abs(logit(np.clip(p, EPS, 1 - EPS)) -
                      logit(np.clip(s, EPS, 1 - EPS)))

    def contradiction(self, P, thr_img):
        """Decision-aware cue contradiction -- the score U-SCNet actually uses.

        A symmetric disagreement is the wrong signal here.  At ~2% prevalence a
        large gap between the image head and the cue evidence is usually a case
        where the cues look abnormal but the target head correctly says
        "not this finding", so |disagreement| is *anti*-correlated with error on
        the majority class and correlates with error on the minority class --
        the two cancel.

        What matters is whether the structured evidence argues *against the
        decision being taken*.  With ``d_img = logit p - logit tau_img`` giving
        the direction of the decision and ``d_cue = logit s_cue - logit
        tau_cue`` the strength and direction of the cue evidence, the
        contradiction score is ``-sign(d_img) * d_cue``: large and positive
        exactly when the cues point the other way.
        """
        d_img = (logit(np.clip(P[:, self.t], EPS, 1 - EPS))
                 - logit(np.clip(thr_img, EPS, 1 - EPS)))
        d_cue = (logit(np.clip(self.evidence(P), EPS, 1 - EPS))
                 - logit(np.clip(self.thr_cue, EPS, 1 - EPS)))
        return -np.sign(d_img) * d_cue


# --------------------------------------------------------------------------
# score fusion
# --------------------------------------------------------------------------
class RankNormalizer:
    """Maps a raw score to its empirical quantile under a reference (validation)
    distribution, so heterogeneous uncertainty terms can be summed."""

    def __init__(self, ref):
        self.ref = np.sort(np.asarray(ref, dtype=np.float64))

    def __call__(self, x):
        return np.searchsorted(self.ref, x, side="right") / max(len(self.ref), 1)


def composite(u_terms, weights, normalizers):
    return sum(w * nz(u) for u, w, nz in zip(u_terms, weights, normalizers))


# --------------------------------------------------------------------------
# risk-coverage analysis
# --------------------------------------------------------------------------
def risk_coverage(y, p, u, thr, cov_grid):
    """Selective balanced error as a function of coverage.

    Cases are accepted in increasing order of uncertainty ``u``.
    """
    order = np.argsort(u, kind="stable")
    ys, ps = y[order], p[order]
    out = []
    for c in cov_grid:
        k = max(int(round(c * len(ys))), 2)
        out.append(balanced_error(ys[:k], ps[:k], thr))
    return np.array(out, dtype=float)


def aurc(risks, cov_grid):
    """Area under the risk-coverage curve (trapezoid over the coverage grid)."""
    m = ~np.isnan(risks)
    if m.sum() < 2:
        return np.nan
    return float(np.trapezoid(risks[m], cov_grid[m]) /
                 (cov_grid[m][-1] - cov_grid[m][0]))


def selective_auc(y, p, u, cov):
    """AUROC recomputed on the accepted subset at coverage ``cov``."""
    order = np.argsort(u, kind="stable")
    k = max(int(round(cov * len(y))), 2)
    ya, pa = y[order][:k], p[order][:k]
    if ya.sum() == 0 or ya.sum() == len(ya):
        return np.nan
    return float(roc_auc_score(ya, pa))


def summarize_at_coverage(y, p, u, thr, cov):
    order = np.argsort(u, kind="stable")
    k = max(int(round(cov * len(y))), 2)
    keep = order[:k]
    ya, pa = y[keep], p[keep]
    se, sp = sens_spec(ya, pa, thr)
    return {
        "coverage": k / len(y),
        "bal_err": balanced_error(ya, pa, thr),
        "sens": se, "spec": sp,
        "auroc": np.nan if ya.sum() in (0, len(ya)) else roc_auc_score(ya, pa),
        "auprc": np.nan if ya.sum() in (0, len(ya)) else average_precision_score(ya, pa),
        "ece": ece_equal_mass(ya, pa),
        "brier": brier(ya, pa),
        "n_accepted": int(k), "n_pos": int(ya.sum()),
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def delong_test(y, p1, p2):
    """DeLong test for two correlated ROC curves. Returns (auc1, auc2, p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def structural(p):
        X, Y = p[pos], p[neg]
        # V10[i] = P(X_i > Y) + 0.5 P(X_i == Y)
        order = np.argsort(Y)
        Ys = Y[order]
        lt = np.searchsorted(Ys, X, side="left")
        le = np.searchsorted(Ys, X, side="right")
        v10 = (lt + 0.5 * (le - lt)) / n
        order = np.argsort(X)
        Xs = X[order]
        gt = len(Xs) - np.searchsorted(Xs, Y, side="right")
        ge = len(Xs) - np.searchsorted(Xs, Y, side="left")
        v01 = (gt + 0.5 * (ge - gt)) / m
        return v10, v01, v10.mean()

    v10a, v01a, a1 = structural(p1)
    v10b, v01b, a2 = structural(p2)
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float(a1), float(a2), 1.0
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(2 * (1 - norm.cdf(abs(z))))


def bootstrap_ci(fn, *arrays, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of ``fn(*arrays)`` resampled over cases."""
    rng = np.random.RandomState(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        try:
            v = fn(*[a[i] for a in arrays])
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
