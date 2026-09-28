"""Robustness of the class-averaged (CA) metrics to the integration range.

The CA-AURC / CA-AUGRC of Saglam et al. integrate each class's risk against
that class's own coverage. Under the 20-100% overall-coverage grid the
realised class-coverage range differs between designs (a threshold-aware
global rule never takes positive coverage below ~45%, MSP takes it to ~5%).
Two checks: (a) a fine overall-coverage sweep from 2% to 100%, so that every
design's class coverage spans almost the whole [0, 1]; (b) integration
restricted to a common class-coverage window shared by every design.
Writes results/<target>_ca_robustness.csv.
"""
import numpy as np
import pandas as pd

import imbalance_selective as I
from robustness import prepare, masks_for
import paths

OUT = paths.RESULTS
FINE = np.round(np.arange(0.02, 1.0001, 0.02), 4)
WINDOW = (0.50, 1.00)

DESIGNS = [("MSP", "global"), ("MSP", "class-conditional"), ("MC-BALD", "global"),
           ("Margin (threshold-aware)", "global"), ("Margin (threshold-aware)", "class-conditional"),
           ("Composite (no cue)", "global"), ("U-SCNet (cue-aware)", "global")]
LEARNED = ["SelectiveNet (weighted)", "CB-SelectiveNet"]


def _auc_window(cov, val, lo, hi):
    cov, val = np.asarray(cov), np.asarray(val)
    m = (cov >= lo - 1e-9) & (cov <= hi + 1e-9) & ~np.isnan(val)
    return I._auc(cov[m], val[m]) if m.sum() >= 2 else np.nan


def ca(ev, lo=None, hi=None):
    out = {}
    for kind, key in (("CA_AURC", "class_risk_realised"), ("CA_AUGRC", "class_grisk_realised")):
        vals = []
        for c in (0, 1):
            cov, val = ev["class_coverage"][c], ev[key][c]
            vals.append(_auc_window(cov, val, lo, hi) if lo is not None else I._auc(cov, val))
        out[kind] = float(np.nanmean(vals))
    return out


def run(target):
    per = prepare(target)
    rows = []
    for d in per:
        y, p, thr = d["y"], d["p"], d["thr"]
        pred = (p >= thr).astype(int)
        entries = [(sc, st, d["U"][sc], None) for sc, st in DESIGNS if sc in d["U"]]
        entries += [(nm, "learned", -d["sn"][nm][0], None) for nm in LEARNED if nm in d["sn"]]
        for sc, st, u, _ in entries:
            masks = (masks_for(u, pred, st) if st != "learned"
                     else [I.select_global(u, c) for c in np.round(np.arange(0.2, 1.0001, 0.05), 4)])
            ev20 = I.evaluate_strategy(y, p, thr, masks, None, u=None)
            fine = ([I.select_global(u, c) if st == "global" else I.select_class_conditional(u, pred, c)
                     for c in FINE] if st != "learned" else [I.select_global(u, c) for c in FINE])
            evf = I.evaluate_strategy(y, p, thr, fine, None, u=None)
            r = dict(seed=d["seed"], target=target, score=sc, strategy=st,
                     pos_cov_min_20=min(ev20["class_coverage"][1]), neg_cov_min_20=min(ev20["class_coverage"][0]))
            r.update({k + "_grid20": v for k, v in ca(ev20).items()})
            r.update({k + "_fine": v for k, v in ca(evf).items()})
            r.update({k + "_window": v for k, v in ca(evf, *WINDOW).items()})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{target.lower()}_ca_robustness.csv", index=False)
    g = df.groupby(["score", "strategy"]).mean(numeric_only=True).drop(columns="seed") * 100
    print(f"\n===== {target}: CA metrics under three integration conventions (x100) =====")
    print(g.round(2).sort_values("CA_AURC_fine").to_string())


if __name__ == "__main__":
    for tg in ("Pneumonia", "Effusion"):
        run(tg)
