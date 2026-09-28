"""How the two cue scores relate to the decision error, within each class.

The symmetric image-cue disagreement |d_img - d_cue| and the signed
contradiction -sign(d_img) d_cue are compared as stand-alone ranking scores
(AURC over the common 20-100% grid, over the operational 50-100% range, and
positive coverage at 80%), and by the Spearman correlation of each with the
decision error within the positives, within the negatives and pooled. Writes
results/<target>_cue_association.csv (per seed) and the summary printed here.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import imbalance_selective as I
import selective as S
from analyze import COV, OPMASK, SEEDS, load, raw_scores, valid_rows
from common import LABELS
import paths

OUT = paths.RESULTS
SCORES = ["Cue disagreement (symmetric)", "Cue contradiction"]


def run(target):
    t = LABELS.index(target)
    cue_idx = [i for i in range(len(LABELS)) if i != t]
    rows = []
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
        raw_t = raw_scores(dt, mt, cue, t, thr, T)
        err = ((pt >= thr).astype(int) != yt_t).astype(float)
        pos = yt_t == 1
        for nm in SCORES:
            u = raw_t[nm]
            rc = S.risk_coverage(yt_t, pt, u, thr, COV)
            ev = I.evaluate_strategy(yt_t, pt, thr, [I.select_global(u, c) for c in COV], COV, u=u)
            rows.append(dict(
                seed=seed, target=target, score=nm,
                r_pos=spearmanr(u[pos], err[pos]).correlation,
                r_neg=spearmanr(u[~pos], err[~pos]).correlation,
                r_pooled=spearmanr(u, err).correlation,
                aurc=S.aurc(rc, COV), aurc_op=S.aurc(rc[OPMASK], COV[OPMASK]),
                cov_pos80=ev["cov_pos@80"], augrc=ev["AUGRC"],
                aurc_pos=ev["AURC_pos"], aurc_neg=ev["AURC_neg"]))
        print(f"  {target} seed {seed}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{target.lower()}_cue_association.csv", index=False)
    print(df.groupby("score")[["r_pos", "r_neg", "r_pooled", "aurc", "aurc_op",
                               "cov_pos80", "augrc", "aurc_pos", "aurc_neg"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    for tg in ("Pneumonia", "Effusion"):
        run(tg)
