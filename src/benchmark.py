"""Controlled comparison of reject-option designs for imbalanced chest radiography.

The benchmark crosses two axes that the literature usually varies one at a time:

  * the **ranking score** -- MSP, entropy, MC-dropout entropy / variance / BALD,
    the threshold-aware margin, and the composite scores of this work;
  * the **selection rule** -- a single global threshold (the classical rule)
    versus coverage matched within each predicted class.

Two further entrants do not factor into that grid and are evaluated on their own
terms: LABEL class-conditional conformal prediction, and SelectiveNet, whose
selection head is trained jointly with the classifier.

Everything is scored with both the pooled metrics (AURC, AUGRC) and their
class-averaged counterparts (CA-AURC, CA-AUGRC), plus the coverage each class
actually receives, which is what the pooled numbers hide.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import imbalance_selective as I
import selective as S
from analyze import (SEEDS, TERMS, apply_combination, fit_combination, load,
                     raw_scores, valid_rows)
from common import LABELS
import paths

OUT = paths.RESULTS
CKPT = paths.CKPT
COV = np.round(np.arange(0.20, 1.0001, 0.05), 4)

SCORES = ["MSP", "Entropy", "MC-entropy", "MC-variance", "MC-BALD",
          "Margin (threshold-aware)", "Composite (no cue)", "U-SCNet (cue-aware)"]


def selectivenet_scores(target, seed, cohort="test", tag=""):
    """g(x) from a trained SelectiveNet, if the checkpoint exists."""
    import torch
    from train_selectivenet import SelectiveNet
    from common import CACHE_SIZE, CROP, IMAGENET_MEAN, IMAGENET_STD
    from infer import cohort as get_cohort

    path = os.path.join(CKPT, f"selectivenet{tag}_{target.lower()}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    dev = "cuda"
    t = LABELS.index(target)
    m = SelectiveNet(t).to(dev).eval()
    m.load_state_dict(torch.load(path, map_location=dev)["model"])
    imgs, _ = get_cohort(cohort)
    off = (CACHE_SIZE - CROP) // 2
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    G, P = [], []
    with torch.no_grad():
        for s in range(0, len(imgs), 512):
            blk = torch.from_numpy(
                imgs[s:s + 512, off:off + CROP, off:off + CROP].copy())
            x = blk.to(dev).float().div_(255.0).unsqueeze(1)
            x = ((x.expand(-1, 3, -1, -1) - mean) / std).to(
                memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, g, _ = m(x)
            G.append(g.float().cpu().numpy())
            P.append(torch.sigmoid(lo[:, t].float()).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(G), np.concatenate(P)


def _scalars(d):
    """Keep only the scalar metrics; curves and per-class dicts go to JSON."""
    return {k: v for k, v in d.items()
            if not isinstance(v, (list, dict, tuple))}


def run(target, args):
    t = LABELS.index(target)
    cue_idx = [i for i in range(len(LABELS)) if i != t]
    rows, curves, poscov = [], {}, {}

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
        p = S.apply_temperature(dt[:, t], T)
        pred = (p >= thr).astype(int)

        raw_v = raw_scores(dv, mv, cue, t, thr, T)
        raw_t = raw_scores(dt, mt, cue, t, thr, T)
        nz = {k: S.RankNormalizer(v) for k, v in raw_v.items()}
        lr3 = fit_combination(raw_v, nz, yv_t, p_val, thr, TERMS)
        lr2 = fit_combination(raw_v, nz, yv_t, p_val, thr, TERMS[:2])
        U = dict(raw_t)
        U["Composite (no cue)"] = apply_combination(lr2, raw_t, nz, TERMS[:2])
        U["U-SCNet (cue-aware)"] = apply_combination(lr3, raw_t, nz, TERMS)

        # ---- score x strategy grid ---------------------------------------
        for sc in SCORES:
            for strat in ("global", "class-conditional"):
                masks = [I.select_global(U[sc], c) if strat == "global"
                         else I.select_class_conditional(U[sc], pred, c)
                         for c in COV]
                r = I.evaluate_strategy(yt_t, p, thr, masks, COV, u=U[sc])
                rows.append(dict(seed=seed, target=target, score=sc,
                                 strategy=strat, **_scalars(r)))
                curves.setdefault(f"{sc}|{strat}", []).append(r["risk"])
                poscov.setdefault(f"{sc}|{strat}", []).append(
                    r["class_coverage"][1])

        # ---- LABEL conformal (set-valued, its own coverage) ---------------
        for alpha in args.alphas:
            keep, _ = I.select_label(p_val, yv_t, p, alpha)
            r = I.evaluate_strategy(yt_t, p, thr, [keep], [keep.mean()])
            rows.append(dict(seed=seed, target=target,
                             score=f"LABEL (alpha={alpha})", strategy="conformal",
                             coverage_pt=r["coverage"][0], balerr_pt=r["risk"][0],
                             cov_neg_pt=r["cov_neg@80"], cov_pos_pt=r["cov_pos@80"],
                             disparity_pt=r["disparity@80"]))

        # ---- SelectiveNet (both loss variants) ----------------------------
        # _cbhalf: the class-balanced loss with lambda/2, i.e. the per-class
        # constraint written as lambda * (1/2) sum_k, which matches the size of
        # the original single-class penalty when both classes are short of
        # coverage (with lambda unchanged the effective weight is doubled).
        for tag, nm in (("", "SelectiveNet"),
                        ("_unw", "SelectiveNet (unweighted)"),
                        ("_cb", "CB-SelectiveNet"),
                        ("_cbhalf", "CB-SelectiveNet (lambda/2)")):
            sn = selectivenet_scores(target, seed, tag=tag) if args.selectivenet else None
            if sn is None:
                continue
            g, p_sn = sn
            g = g[ot]
            masks = [I.select_global(-g, c) for c in COV]   # higher g == keep
            r = I.evaluate_strategy(yt_t, p, thr, masks, COV, u=-g)
            r["sn_own_auroc"] = roc_auc_score(yt_t, p_sn[ot])
            # Does the learned selector recover the threshold-aware geometry?
            # Positive means g (keep) rises with distance from the operating
            # threshold; negative means it learns the opposite.
            r["sn_corr_margin"] = float(np.corrcoef(
                g, -S.u_margin_threshold(p, thr))[0, 1])
            rows.append(dict(seed=seed, target=target, score=nm,
                             strategy="learned", **_scalars(r)))
            curves.setdefault(f"{nm}|learned", []).append(r["risk"])
            poscov.setdefault(f"{nm}|learned", []).append(r["class_coverage"][1])
        print(f"  {target} seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{target.lower()}_benchmark_perseed.csv", index=False)
    num = [c for c in df.columns
           if c not in ("seed", "target", "score", "strategy")
           and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(["score", "strategy"])[num].agg(["mean", "std"])
    agg.to_csv(f"{OUT}/{target.lower()}_benchmark_summary.csv")
    json.dump({"coverage": COV.tolist(),
               "risk_curves": {k: np.nanmean(np.array(v, float), 0).tolist()
                               for k, v in curves.items()},
               "pos_coverage": {k: np.nanmean(np.array(v, float), 0).tolist()
                                for k, v in poscov.items()}},
              open(f"{OUT}/{target.lower()}_benchmark_curves.json", "w"), indent=2)

    show = df[df.strategy.isin(["global", "class-conditional", "learned"])]
    cols = [c for c in ["AURC", "WC_AURC", "AURC_pos", "AURC_neg",
                        "cov_pos@80", "disparity@80"] if c in show.columns]
    piv = show.groupby(["score", "strategy"])[cols].mean()
    print(f"\n===== {target}: reject-option benchmark =====\n"
          f"AURC       pooled balanced-error risk-coverage (lower better)\n"
          f"WC_AURC    within-class ranking quality, class-averaged (lower better)\n"
          f"cov_pos@80 fraction of true positives still answered at 80% coverage\n"
          f"disparity  negative-class minus positive-class coverage at 80%")
    print(piv.round(4).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["Pneumonia", "Effusion"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--selectivenet", action="store_true")
    a = ap.parse_args()
    for tg in a.targets:
        run(tg, a)
