"""Robustness of the main findings, from cached predictions.

Three analyses that the benchmark does not cover:

1. **Operating-point sensitivity.** Every headline number depends on the
   threshold tau, which is chosen by Youden's J on a validation split with
   ~100 positives. Here tau is re-chosen under several rules (fixed
   sensitivity, prevalence matching, cost-weighted optimum, a factor-of-two
   perturbation, and the test-set optimum as an oracle reference) and the
   class-resolved coverage of the confidence-centred and threshold-aware scores
   is recomputed under each.

2. **Asymmetric cost.** The risk reported elsewhere, 0.5*FNR + 0.5*FPR, is
   symmetric. Screening is not: a missed positive costs more than a false
   alarm. The benchmark is re-scored with risk (c*FNR + FPR)/(c + 1) at
   c = 3, with tau re-chosen as the validation optimum of the same cost, and
   the within-class AURC average of Table 4 is re-weighted the same way.

3. **Case-level bootstrap** of the core comparisons (positive coverage,
   balanced error, AURC, disparity, and the SelectiveNet classifier AUROC),
   resampling test cases with seeds averaged within each resample, so that
   the claims in the text carry intervals rather than five-seed SDs.
"""
import argparse
import os

import numpy as np
import pandas as pd

import imbalance_selective as I
import selective as S
from analyze import (SEEDS, TERMS, apply_combination, fit_combination, load,
                     raw_scores, valid_rows)
from common import LABELS

OUT = "/NHNHOME/uscnet/results"
PRED = "/NHNHOME/uscnet/results/preds"
COV = np.round(np.arange(0.20, 1.0001, 0.05), 4)
J80 = int(np.argmin(np.abs(COV - 0.8)))
COST = 3.0                       # FN:FP cost ratio for the asymmetric analysis

SN_TAGS = [("", "SelectiveNet (weighted)"),
           ("_unw", "SelectiveNet (unweighted)"),
           ("_cb", "CB-SelectiveNet"),
           ("_cbhalf", "CB-SelectiveNet (lambda/2)")]


# --------------------------------------------------------------------------
# threshold rules
# --------------------------------------------------------------------------
def thr_at_sensitivity(y, p, sens):
    """Largest threshold whose validation sensitivity is at least ``sens``."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - sens) * len(pos)))
    return float(pos[max(min(k, len(pos) - 1), 0)])


def thr_prevalence(y, p):
    """Predicted-positive rate equal to the validation prevalence (a quantile
    of the scores; not the same thing as tau = prevalence)."""
    return float(np.quantile(p, 1 - y.mean()))


def thr_cost(y, p, c):
    """Minimise (c*FNR + FPR)/(c+1) -- Youden's J is the c = 1 case."""
    order = np.argsort(-p)
    ys = y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    P, N = ys.sum(), len(ys) - ys.sum()
    fnr, fpr = 1 - tp / max(P, 1), fp / max(N, 1)
    k = int(np.argmin((c * fnr + fpr) / (c + 1)))
    return float(p[order][k])


def tau_rules(yv, pv, yt, pt):
    yo = S.youden_threshold(yv, pv)
    return {
        "Youden (used)": yo,
        "Youden x 0.5": yo * 0.5,
        "Youden x 2": yo * 2.0,
        "sensitivity 90%": thr_at_sensitivity(yv, pv, 0.90),
        "sensitivity 80%": thr_at_sensitivity(yv, pv, 0.80),
        "tau = prevalence": float(yv.mean()),          # cost-optimal if calibrated
        "predicted-positive-rate matched": thr_prevalence(yv, pv),
        f"cost {int(COST)}:1 optimum": thr_cost(yv, pv, COST),
        "test-set Youden (oracle)": S.youden_threshold(yt, pt),
    }


# --------------------------------------------------------------------------
# risks
# --------------------------------------------------------------------------
def cost_error_subset(y, p, thr, keep, c):
    ya, pa = y[keep], p[keep]
    pos, neg = ya == 1, ya == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    e = I._err(ya, pa, thr)
    return float((c * e[pos].mean() + e[neg].mean()) / (c + 1))


def curve(y, p, thr, masks, risk_fn):
    cov = np.array([m.mean() for m in masks])
    risk = np.array([risk_fn(y, p, thr, m) for m in masks])
    ccov1 = np.array([(m & (y == 1)).sum() / max((y == 1).sum(), 1) for m in masks])
    ccov0 = np.array([(m & (y == 0)).sum() / max((y == 0).sum(), 1) for m in masks])
    return dict(AURC=I._auc(cov, risk), risk80=risk[J80],
                cov_pos80=ccov1[J80], cov_neg80=ccov0[J80],
                disparity80=ccov0[J80] - ccov1[J80])


def within_class_aurc(y, p, thr, u):
    wc = {c: I.within_class_curve(y, p, thr, u, c, COV) for c in (0, 1)}
    return I._auc(COV, wc[1]), I._auc(COV, wc[0])       # pos (FNR), neg (FPR)


# --------------------------------------------------------------------------
# per-seed preparation
# --------------------------------------------------------------------------
def sn_scores(target, seed, tag):
    """Selector g(x) and classifier p(x) of a SelectiveNet checkpoint on the
    test cohort, cached as an .npz so the bootstrap needs no GPU."""
    cache = os.path.join(PRED, f"sn{tag}_{target.lower()}_seed{seed}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["g"], z["p"]
    from benchmark import selectivenet_scores
    r = selectivenet_scores(target, seed, tag=tag)
    if r is None:
        return None
    np.savez(cache, g=r[0], p=r[1])
    return r


def prepare(target, need_sn=True):
    t = LABELS.index(target)
    cue_idx = [i for i in range(len(LABELS)) if i != t]
    per = []
    for seed in SEEDS:
        dtr, _, ytr = load("train", seed, need_mc=False)
        cue = S.CueBank(t, cue_idx).fit_evidence(dtr, ytr[:, t])
        del dtr, ytr
        dv, mv, yv = load("val", seed)
        dt, mt, yt = load("test", seed)
        ov, ot = valid_rows(yv, t), valid_rows(yt, t)
        dv, mv, yv_t = dv[ov], mv[:, ov], yv[ov, t]
        dt, mt, yt_t = dt[ot], mt[:, ot], yt[ot, t]
        T = S.fit_temperature(yv_t, dv[:, t])
        pv, pt = S.apply_temperature(dv[:, t], T), S.apply_temperature(dt[:, t], T)
        thr = S.youden_threshold(yv_t, pv)
        cue.fit_cue_threshold(dv, yv_t)
        raw_v, raw_t = raw_scores(dv, mv, cue, t, thr, T), raw_scores(dt, mt, cue, t, thr, T)
        nz = {k: S.RankNormalizer(v) for k, v in raw_v.items()}
        lr3 = fit_combination(raw_v, nz, yv_t, pv, thr, TERMS)
        lr2 = fit_combination(raw_v, nz, yv_t, pv, thr, TERMS[:2])
        U = {"MSP": raw_t["MSP"],
             "MC-BALD": raw_t["MC-BALD"],
             "Margin (threshold-aware)": raw_t["Margin (threshold-aware)"],
             "Composite (no cue)": apply_combination(lr2, raw_t, nz, TERMS[:2]),
             "U-SCNet (cue-aware)": apply_combination(lr3, raw_t, nz, TERMS)}
        d = dict(seed=seed, yv=yv_t, pv=pv, y=yt_t, p=pt, thr=thr, U=U,
                 mc=mt[:, :, t], sn={})
        if need_sn:
            for tag, nm in SN_TAGS:
                r = sn_scores(target, seed, tag)
                if r is not None:
                    d["sn"][nm] = (r[0][ot], r[1][ot])
        per.append(d)
        print(f"  {target} seed {seed}: thr={thr:.4f} n={len(yt_t)} "
              f"pos={int(yt_t.sum())} sn={list(d['sn'])}", flush=True)
    return per


def masks_for(u, pred, strat):
    return [I.select_global(u, c) if strat == "global"
            else I.select_class_conditional(u, pred, c) for c in COV]


def agg(df, keys):
    num = [c for c in df.columns if c not in keys + ["seed"]
           and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(keys, sort=False)[num].agg(["mean", "std"])


# --------------------------------------------------------------------------
# 1. operating-point sensitivity
# --------------------------------------------------------------------------
def tau_sensitivity(target, per):
    rows = []
    for d in per:
        y, p = d["y"], d["p"]
        for rule, thr in tau_rules(d["yv"], d["pv"], y, p).items():
            pred = (p >= thr).astype(int)
            se, sp = S.sens_spec(y, p, thr)
            row = dict(seed=d["seed"], target=target, rule=rule, tau=thr,
                       sens=se, spec=sp, ppr=pred.mean(),
                       balerr=S.balanced_error(y, p, thr))
            scores = {"MSP": d["U"]["MSP"],
                      "Margin": S.u_margin_threshold(p, thr)}
            for sc, u in scores.items():
                for strat, short in (("global", "g"), ("class-conditional", "cc")):
                    r = curve(y, p, thr, masks_for(u, pred, strat),
                              I.balanced_error_subset)
                    row[f"{sc}|{short}|cov_pos80"] = r["cov_pos80"]
                    row[f"{sc}|{short}|risk80"] = r["risk80"]
                    row[f"{sc}|{short}|AURC"] = r["AURC"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{target.lower()}_tau_sensitivity_perseed.csv", index=False)
    a = agg(df, ["rule"])
    a.to_csv(f"{OUT}/{target.lower()}_tau_sensitivity.csv")
    print(f"\n===== {target}: operating-point sensitivity =====")
    cols = [("tau", "mean"), ("sens", "mean"), ("spec", "mean"),
            ("MSP|g|cov_pos80", "mean"), ("Margin|g|cov_pos80", "mean"),
            ("MSP|cc|cov_pos80", "mean"), ("Margin|g|risk80", "mean"),
            ("MSP|g|risk80", "mean")]
    print(a[cols].round(4).to_string())
    return df


# --------------------------------------------------------------------------
# 2. asymmetric cost
# --------------------------------------------------------------------------
def cost_thresholds(target, per, ratios=(1, 2, 3, 5, 10)):
    """Where the cost-optimal threshold lands for several FN:FP ratios; at
    high ratios it degenerates to predicting nearly every case positive."""
    rows = []
    for d in per:
        for c in ratios:
            thr = thr_cost(d["yv"], d["pv"], c)
            se, sp = S.sens_spec(d["y"], d["p"], thr)
            rows.append(dict(seed=d["seed"], target=target, ratio=c, tau=thr,
                             sens=se, spec=sp, ppr=float((d["p"] >= thr).mean())))
    df = pd.DataFrame(rows)
    df.groupby("ratio")[["tau", "sens", "spec", "ppr"]].agg(["mean", "std"]).to_csv(
        f"{OUT}/{target.lower()}_cost_thresholds.csv")
    return df


def cost_benchmark(target, per):
    rows = []
    risk_fn = lambda y, p, thr, m: cost_error_subset(y, p, thr, m, COST)
    for d in per:
        y, p = d["y"], d["p"]
        thr = thr_cost(d["yv"], d["pv"], COST)
        pred = (p >= thr).astype(int)
        U = dict(d["U"])
        # the threshold-aware score is re-centred on the cost-optimal tau
        U["Margin (threshold-aware)"] = S.u_margin_threshold(p, thr)
        for sc, u in U.items():
            if sc in ("Composite (no cue)", "U-SCNet (cue-aware)"):
                continue                    # fitted on the Youden tau; skip
            for strat in ("global", "class-conditional"):
                r = curve(y, p, thr, masks_for(u, pred, strat), risk_fn)
                ap, an = within_class_aurc(y, p, thr, u)
                rows.append(dict(seed=d["seed"], target=target, score=sc,
                                 strategy=strat, tau=thr, **r,
                                 AURC_pos=ap, AURC_neg=an,
                                 WC_weighted=(COST * ap + an) / (COST + 1)))
        for nm, (g, _) in d["sn"].items():
            r = curve(y, p, thr, [I.select_global(-g, c) for c in COV], risk_fn)
            ap, an = within_class_aurc(y, p, thr, -g)
            rows.append(dict(seed=d["seed"], target=target, score=nm,
                             strategy="learned", tau=thr, **r,
                             AURC_pos=ap, AURC_neg=an,
                             WC_weighted=(COST * ap + an) / (COST + 1)))
        for alpha in (0.05, 0.10, 0.20):
            keep, _ = I.select_label(d["pv"], d["yv"], p, alpha)
            rows.append(dict(seed=d["seed"], target=target,
                             score=f"LABEL (alpha={alpha})", strategy="conformal",
                             tau=thr, coverage=keep.mean(),
                             risk_pt=risk_fn(y, p, thr, keep)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{target.lower()}_cost{int(COST)}_perseed.csv", index=False)
    a = agg(df, ["score", "strategy"])
    a.to_csv(f"{OUT}/{target.lower()}_cost{int(COST)}_summary.csv")
    print(f"\n===== {target}: FN:FP = {int(COST)}:1 =====")
    cols = [c for c in [("tau", "mean"), ("AURC", "mean"), ("risk80", "mean"),
                        ("cov_pos80", "mean"), ("AURC_pos", "mean"),
                        ("AURC_neg", "mean"), ("WC_weighted", "mean"),
                        ("coverage", "mean"), ("risk_pt", "mean")] if c in a.columns]
    print(a[cols].round(4).to_string())
    return df


# --------------------------------------------------------------------------
# 3. case-level bootstrap of the core comparisons
# --------------------------------------------------------------------------
def _metrics(d, idx):
    """Everything the core comparisons need, on one resample of one seed."""
    y, p, thr = d["y"][idx], d["p"][idx], d["thr"]
    if y.sum() < 5:
        return None
    pred = (p >= thr).astype(int)
    m = {}
    for sc in ("MSP", "Margin (threshold-aware)", "Composite (no cue)"):
        u = d["U"][sc][idx]
        for strat, short in (("global", "g"), ("class-conditional", "cc")):
            r = curve(y, p, thr, masks_for(u, pred, strat), I.balanced_error_subset)
            for k in ("cov_pos80", "risk80", "AURC", "disparity80"):
                m[f"{sc}|{short}|{k}"] = r[k]
    for nm, (g, psn) in d["sn"].items():
        g, psn = g[idx], psn[idx]
        r = curve(y, p, thr, [I.select_global(-g, c) for c in COV],
                  I.balanced_error_subset)
        for k in ("cov_pos80", "risk80", "AURC", "disparity80"):
            m[f"{nm}|learned|{k}"] = r[k]
        m[f"{nm}|learned|auroc"] = S.roc_auc_score(y, psn)
    return m


# (name, quantity A, quantity B): the interval is for A - B
COMPARISONS = [
    ("threshold-aware vs MSP, global: positive coverage",
     "Margin (threshold-aware)|g|cov_pos80", "MSP|g|cov_pos80"),
    ("class-conditional vs global, MSP: positive coverage",
     "MSP|cc|cov_pos80", "MSP|g|cov_pos80"),
    ("threshold-aware, class-conditional vs global: positive coverage",
     "Margin (threshold-aware)|cc|cov_pos80", "Margin (threshold-aware)|g|cov_pos80"),
    ("threshold-aware vs MSP, global: balanced error at 80%",
     "Margin (threshold-aware)|g|risk80", "MSP|g|risk80"),
    ("composite vs MSP, global: balanced error at 80%",
     "Composite (no cue)|g|risk80", "MSP|g|risk80"),
    ("CB-SelectiveNet vs SelectiveNet (weighted): AURC",
     "CB-SelectiveNet|learned|AURC", "SelectiveNet (weighted)|learned|AURC"),
    ("CB-SelectiveNet vs SelectiveNet (weighted): disparity at 80%",
     "CB-SelectiveNet|learned|disparity80", "SelectiveNet (weighted)|learned|disparity80"),
    ("CB-SelectiveNet vs SelectiveNet (weighted): positive coverage",
     "CB-SelectiveNet|learned|cov_pos80", "SelectiveNet (weighted)|learned|cov_pos80"),
    ("CB-SelectiveNet vs SelectiveNet (weighted): classifier AUROC",
     "CB-SelectiveNet|learned|auroc", "SelectiveNet (weighted)|learned|auroc"),
    ("SelectiveNet unweighted vs weighted: positive coverage",
     "SelectiveNet (unweighted)|learned|cov_pos80", "SelectiveNet (weighted)|learned|cov_pos80"),
    ("SelectiveNet unweighted vs weighted: AURC",
     "SelectiveNet (unweighted)|learned|AURC", "SelectiveNet (weighted)|learned|AURC"),
    ("CB-SelectiveNet lambda/2 vs lambda: AURC",
     "CB-SelectiveNet (lambda/2)|learned|AURC", "CB-SelectiveNet|learned|AURC"),
    ("CB-SelectiveNet lambda/2 vs lambda: disparity at 80%",
     "CB-SelectiveNet (lambda/2)|learned|disparity80", "CB-SelectiveNet|learned|disparity80"),
    ("CB-SelectiveNet vs composite (global): AURC",
     "CB-SelectiveNet|learned|AURC", "Composite (no cue)|g|AURC"),
]


def core_bootstrap(target, per, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(per[0]["y"])
    draws = {}
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        acc = {}
        for d in per:
            m = _metrics(d, idx)
            if m is None:
                continue
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            draws.setdefault(k, []).append(np.nanmean(v))
        if (b + 1) % 100 == 0:
            print(f"    {target} bootstrap {b+1}/{n_boot}", flush=True)
    draws = {k: np.array(v, float) for k, v in draws.items()}
    rows = []
    for name, a, b in COMPARISONS:
        if a not in draws or b not in draws:
            continue
        dlt = draws[a] - draws[b]
        ok = ~np.isnan(dlt)
        dlt = dlt[ok]
        rows.append(dict(target=target, comparison=name, a=a, b=b,
                         a_mean=float(np.nanmean(draws[a])),
                         b_mean=float(np.nanmean(draws[b])),
                         delta=float(dlt.mean()),
                         ci_lo=float(np.percentile(dlt, 2.5)),
                         ci_hi=float(np.percentile(dlt, 97.5)),
                         p_boot=float(2 * min((dlt >= 0).mean(), (dlt <= 0).mean())),
                         n_boot=int(ok.sum())))
    df = pd.DataFrame(rows)
    print(f"\n===== {target}: case-level bootstrap =====")
    print(df[["comparison", "a_mean", "b_mean", "delta", "ci_lo", "ci_hi",
              "p_boot"]].round(4).to_string(index=False))
    return df


# --------------------------------------------------------------------------
# 4. two-level bootstrap: cases AND training seeds
# --------------------------------------------------------------------------
# The core bootstrap above conditions on the five trained models. Resampling
# the seeds with replacement as well (5 of 5, inside each case resample) gives
# intervals that also carry training variability. It is run for the headline
# comparisons only.
TWO_LEVEL = [
    ("threshold-aware vs MSP, global: positive coverage",
     "Margin (threshold-aware)|g|cov_pos80", "MSP|g|cov_pos80"),
    ("class-conditional vs global, MSP: positive coverage",
     "MSP|cc|cov_pos80", "MSP|g|cov_pos80"),
    ("composite vs MSP, global: balanced error at 80%",
     "Composite (no cue)|g|risk80", "MSP|g|risk80"),
    ("CB-SelectiveNet vs SelectiveNet (weighted): disparity at 80%",
     "CB-SelectiveNet|learned|disparity80", "SelectiveNet (weighted)|learned|disparity80"),
    ("CB-SelectiveNet vs SelectiveNet (weighted): AURC",
     "CB-SelectiveNet|learned|AURC", "SelectiveNet (weighted)|learned|AURC"),
]


def _metrics_headline(d, idx):
    y, p, thr = d["y"][idx], d["p"][idx], d["thr"]
    if y.sum() < 5:
        return None
    pred = (p >= thr).astype(int)
    m = {}
    for sc, strat, short in (("MSP", "global", "g"), ("MSP", "class-conditional", "cc"),
                             ("Margin (threshold-aware)", "global", "g"),
                             ("Composite (no cue)", "global", "g")):
        r = curve(y, p, thr, masks_for(d["U"][sc][idx], pred, strat),
                  I.balanced_error_subset)
        m[f"{sc}|{short}|cov_pos80"] = r["cov_pos80"]
        m[f"{sc}|{short}|risk80"] = r["risk80"]
    for nm in ("SelectiveNet (weighted)", "CB-SelectiveNet"):
        if nm not in d["sn"]:
            continue
        g = d["sn"][nm][0][idx]
        r = curve(y, p, thr, [I.select_global(-g, c) for c in COV],
                  I.balanced_error_subset)
        m[f"{nm}|learned|disparity80"] = r["disparity80"]
        m[f"{nm}|learned|AURC"] = r["AURC"]
    return m


_PER = None          # shared with forked workers


def _two_level_one(args):
    b, seed, n, S = args
    rng = np.random.RandomState(seed + 7919 * b)
    idx = rng.randint(0, n, n)
    ms_ = [_metrics_headline(d, idx) for d in _PER]
    w = np.bincount(rng.randint(0, S, S), minlength=S)          # seeds with replacement
    out = {}
    for k in ms_[0]:
        vals = np.array([m[k] if m is not None else np.nan for m in ms_], float)
        ok = ~np.isnan(vals) & (w > 0)
        out[k] = float(np.sum(w[ok] * vals[ok]) / w[ok].sum()) if ok.any() else np.nan
    return out


def two_level_bootstrap(target, per, n_boot=1000, seed=1, workers=32):
    """Resamples are independent, so they are spread over forked workers
    (the per-seed arrays are inherited copy-on-write)."""
    global _PER
    _PER = per
    from multiprocessing import Pool
    n, S = len(per[0]["y"]), len(per)
    with Pool(workers) as pool:
        res = pool.map(_two_level_one, [(b, seed, n, S) for b in range(n_boot)],
                       chunksize=8)
    draws = {k: np.array([r[k] for r in res], float) for k in res[0]}
    print(f"    {target} two-level {n_boot} resamples done", flush=True)
    rows = []
    for name, a, bk in TWO_LEVEL:
        if a not in draws or bk not in draws:
            continue
        dlt = draws[a] - draws[bk]
        dlt = dlt[~np.isnan(dlt)]
        rows.append(dict(target=target, comparison=name, a=a, b=bk,
                         a_mean=float(np.nanmean(draws[a])), b_mean=float(np.nanmean(draws[bk])),
                         delta=float(dlt.mean()), ci_lo=float(np.percentile(dlt, 2.5)),
                         ci_hi=float(np.percentile(dlt, 97.5)),
                         p_boot=float(2 * min((dlt >= 0).mean(), (dlt <= 0).mean())),
                         n_boot=int(len(dlt))))
    df = pd.DataFrame(rows)
    print(f"\n===== {target}: two-level (seed x case) bootstrap =====")
    print(df[["comparison", "delta", "ci_lo", "ci_hi", "p_boot"]].round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["Pneumonia", "Effusion"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["tau", "cost", "boot", "two"])
    a = ap.parse_args()
    boots, twos = [], []
    for tgt in a.targets:
        per = prepare(tgt)
        if "tau" not in a.skip:
            tau_sensitivity(tgt, per)
        if "cost" not in a.skip:
            cost_thresholds(tgt, per)
            cost_benchmark(tgt, per)
        if "boot" not in a.skip:
            boots.append(core_bootstrap(tgt, per, a.n_boot))
        if "two" not in a.skip:
            twos.append(two_level_bootstrap(tgt, per, a.n_boot))
    if boots:
        pd.concat(boots).to_csv(f"{OUT}/core_bootstrap.csv", index=False)
        print(f"wrote {OUT}/core_bootstrap.csv")
    if twos:
        pd.concat(twos).to_csv(f"{OUT}/two_level_bootstrap.csv", index=False)
        print(f"wrote {OUT}/two_level_bootstrap.csv")
