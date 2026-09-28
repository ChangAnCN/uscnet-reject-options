"""Case-level bootstrap comparison of selective-ranking rules.

A paired test over five seeds cannot resolve small effects -- the Wilcoxon
signed-rank statistic on n = 5 has a minimum attainable p of 0.0625, so it can
never reach 0.05 however consistent the effect is.  This script therefore
resamples the *test cases* (the unit that actually carries the statistical
weight), recomputes AURC for each ranking rule on each resample, and reports the
paired difference with a percentile confidence interval.  Seeds are averaged
within each resample so that the interval reflects case-level sampling error for
the method as a whole rather than for one checkpoint.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import selective as S
from analyze import (COV, OPMASK, PRED, SEEDS, TERMS, apply_combination,
                     fit_combination, load, raw_scores, valid_rows)
from common import LABELS
import paths

OUT = paths.RESULTS


def prepare(target):
    """Per-seed test-set scores, with everything fitted on train/validation."""
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
        p_val = S.apply_temperature(dv[:, t], T)
        thr = S.youden_threshold(yv_t, p_val)
        cue.fit_cue_threshold(dv, yv_t)
        p_test = S.apply_temperature(dt[:, t], T)

        raw_v = raw_scores(dv, mv, cue, t, thr, T)
        raw_t = raw_scores(dt, mt, cue, t, thr, T)
        nz = {k: S.RankNormalizer(v) for k, v in raw_v.items()}
        lr3 = fit_combination(raw_v, nz, yv_t, p_val, thr, TERMS)
        lr2 = fit_combination(raw_v, nz, yv_t, p_val, thr, TERMS[:2])

        u = dict(raw_t)
        u["Composite (no cue)"] = apply_combination(lr2, raw_t, nz, TERMS[:2])
        u["U-SCNet (cue-aware)"] = apply_combination(lr3, raw_t, nz, TERMS)
        per.append({"y": yt_t, "p": p_test, "thr": thr, "u": u})
        print(f"  prepared seed {seed} (n={len(yt_t)}, pos={int(yt_t.sum())})",
              flush=True)
    return per


def bootstrap(per, methods, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(per[0]["y"])
    keys = [f"{m}|{r}" for m in methods for r in ("aurc", "aurc_op")]
    draws = {k: [] for k in keys}
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        acc = {k: [] for k in keys}
        for d in per:
            y, p, thr = d["y"][idx], d["p"][idx], d["thr"]
            if y.sum() < 5:
                continue
            for m in methods:
                rc = S.risk_coverage(y, p, d["u"][m][idx], thr, COV)
                acc[f"{m}|aurc"].append(S.aurc(rc, COV))
                acc[f"{m}|aurc_op"].append(S.aurc(rc[OPMASK], COV[OPMASK]))
        for k in keys:
            draws[k].append(np.mean(acc[k]) if acc[k] else np.nan)
        if (b + 1) % 200 == 0:
            print(f"    {b+1}/{n_boot}", flush=True)
    return {k: np.array(v, float) for k, v in draws.items()}


def report(draws, methods, ref, target):
    rows = []
    for m in methods:
        for r in ("aurc", "aurc_op"):
            a, b = draws[f"{ref}|{r}"], draws[f"{m}|{r}"]
            d = a - b                       # negative == reference is better
            ok = ~np.isnan(d)
            d = d[ok]
            rows.append({
                "target": target, "reference": ref, "method": m, "range": r,
                "ref_mean": float(np.nanmean(a)), "method_mean": float(np.nanmean(b)),
                "delta_mean": float(d.mean()),
                "ci_lo": float(np.percentile(d, 2.5)),
                "ci_hi": float(np.percentile(d, 97.5)),
                # two-sided bootstrap p: proportion of resamples on the wrong side
                "p_boot": float(2 * min((d >= 0).mean(), (d <= 0).mean())),
                "n_boot": int(ok.sum()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["Pneumonia", "Effusion"])
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()

    METHODS = ["MSP", "Entropy", "MC-entropy", "MC-BALD", "MC-variance",
               "Margin (threshold-aware)", "Cue disagreement (symmetric)",
               "Cue contradiction", "Composite (no cue)", "U-SCNet (cue-aware)"]
    out = []
    for tgt in a.targets:
        print(f"\n=== bootstrap: {tgt} ===", flush=True)
        per = prepare(tgt)
        draws = bootstrap(per, METHODS, a.n_boot)
        df = report(draws, METHODS, "U-SCNet (cue-aware)", tgt)
        out.append(df)
        print(df[df.range == "aurc"][["method", "method_mean", "delta_mean",
                                      "ci_lo", "ci_hi", "p_boot"]]
              .round(4).to_string(index=False))
    pd.concat(out).to_csv(f"{OUT}/bootstrap_comparison.csv", index=False)
    print(f"\nwrote {OUT}/bootstrap_comparison.csv")
