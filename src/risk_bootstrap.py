"""Case-level bootstrap intervals for the four-risk comparison (Section 4.3).

The core bootstrap in ``robustness.py`` covers positive coverage, balanced
error at 80% and AURC. The reversal between the accepted-subset balanced
error and the three whole-class or class-coverage-integrated risks (AUGRC,
CA-AURC, CA-AUGRC) is a paired comparison on the same test cases and the
same five models, so it gets the same treatment: 1,000 resamples of the test
split, seeds averaged within each resample, percentile intervals for the
difference. Writes ``results/risk_bootstrap.csv``.
"""
import os

import numpy as np
import pandas as pd

import imbalance_selective as I
from robustness import COV, OUT, masks_for, prepare

SCORES = [("MSP", "g"), ("MSP", "cc"),
          ("Margin (threshold-aware)", "g"), ("Margin (threshold-aware)", "cc"),
          ("Composite (no cue)", "g"), ("Composite (no cue)", "cc")]
KEYS = ("AURC", "AUGRC", "CA_AURC", "CA_AUGRC")
STRAT = {"g": "global", "cc": "class-conditional"}

# (name, A, B): interval for A - B
COMPARISONS = [
    ("threshold-aware vs MSP, global", "Margin (threshold-aware)|g", "MSP|g"),
    ("composite vs MSP, global", "Composite (no cue)|g", "MSP|g"),
    ("class-conditional vs global, MSP", "MSP|cc", "MSP|g"),
    ("class-conditional vs global, threshold-aware",
     "Margin (threshold-aware)|cc", "Margin (threshold-aware)|g"),
    ("class-conditional vs global, composite",
     "Composite (no cue)|cc", "Composite (no cue)|g"),
]


def _metrics(d, idx):
    y, p, thr = d["y"][idx], d["p"][idx], d["thr"]
    if y.sum() < 5:
        return None
    pred = (p >= thr).astype(int)
    m = {}
    for sc, short in SCORES:
        u = d["U"][sc][idx]
        r = I.evaluate_strategy(y, p, thr, masks_for(u, pred, STRAT[short]), COV)
        for k in KEYS:
            m[f"{sc}|{short}|{k}"] = r[k]
    return m


def run(target, n_boot=1000, seed=0):
    per = prepare(target, need_sn=False)
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
            print(f"    {target} risk bootstrap {b+1}/{n_boot}", flush=True)
    draws = {k: np.array(v, float) for k, v in draws.items()}
    # point estimate on the full test split (seed mean), as in the tables
    full = {}
    for d in per:
        m = _metrics(d, np.arange(n))
        for k, v in m.items():
            full.setdefault(k, []).append(v)
    full = {k: float(np.mean(v)) for k, v in full.items()}
    rows = []
    for name, a, b in COMPARISONS:
        for k in KEYS:
            dlt = draws[f"{a}|{k}"] - draws[f"{b}|{k}"]
            dlt = dlt[~np.isnan(dlt)]
            rows.append(dict(target=target, comparison=f"{name}: {k}", metric=k,
                             a=a, b=b, a_full=full[f"{a}|{k}"], b_full=full[f"{b}|{k}"],
                             delta_full=full[f"{a}|{k}"] - full[f"{b}|{k}"],
                             delta=float(dlt.mean()),
                             ci_lo=float(np.percentile(dlt, 2.5)),
                             ci_hi=float(np.percentile(dlt, 97.5)),
                             p_boot=float(2 * min((dlt >= 0).mean(), (dlt <= 0).mean())),
                             n_boot=int(len(dlt))))
    df = pd.DataFrame(rows)
    print(f"\n===== {target}: risk bootstrap =====")
    print(df[["comparison", "delta_full", "delta", "ci_lo", "ci_hi", "p_boot"]]
          .round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or ["Pneumonia", "Effusion"]
    for tg in targets:
        run(tg).to_csv(f"{OUT}/risk_bootstrap_{tg.lower()}.csv", index=False)
    out = pd.concat([pd.read_csv(f"{OUT}/risk_bootstrap_{tg.lower()}.csv")
                     for tg in ("Pneumonia", "Effusion")
                     if os.path.exists(f"{OUT}/risk_bootstrap_{tg.lower()}.csv")])
    out.to_csv(f"{OUT}/risk_bootstrap.csv", index=False)
    print("wrote", f"{OUT}/risk_bootstrap.csv")
