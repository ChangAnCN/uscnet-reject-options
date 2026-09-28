"""Run the full U-SCNet evaluation and write every table of the paper as JSON/CSV.

Protocol (nothing is fitted on the test data)
---------------------------------------------
* Cue *association* weights: training-split ground truth (interpretation only).
* Cue *evidence* model: each seed's own training-split predictions.
* Temperature, the operating threshold tau, the cue threshold, the rank
  normalisers and the score-combination weights: that seed's validation split.
* Everything is then applied unchanged to the CXR8 test split and to the
  external CheXpert cohorts.

Two AURC ranges are reported for every method and both are always shown:
``aurc`` over the full coverage grid [0.10, 1.00], and ``aurc_op`` over the
operational range [0.50, 1.00] -- abstaining on more than half of all cases is
not a meaningful decision-support mode, but restricting the range is declared
here rather than chosen after seeing the results.
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

import imbalance_selective as I
import selective as S
from common import LABELS

PRED = "/NHNHOME/uscnet/results/preds"
OUT = "/NHNHOME/uscnet/results"
# One coverage grid for every AURC in the paper: overall coverage 20-100% in
# steps of 5% (the same grid as benchmark.py and robustness.py). AURC is the
# trapezoidal area divided by the span, i.e. the mean risk over the grid; the
# "operational range" is 50-100%.
COV = np.round(np.arange(0.20, 1.0001, 0.05), 4)
OPMASK = COV >= 0.50
SEEDS = [42, 123, 3407, 7, 2024]
TARGETS = {"Pneumonia": LABELS.index("Pneumonia"),
           "Effusion": LABELS.index("Effusion")}
TERMS = ["Margin (threshold-aware)", "MC-variance", "Cue contradiction"]


def load(cohort, seed, need_mc=True):
    z = np.load(os.path.join(PRED, f"pred_{cohort}_seed{seed}.npz"))
    mc = z["mc"].astype(np.float64) if need_mc else None
    return z["det"].astype(np.float64), mc, z["y"].astype(np.float64)


def valid_rows(y, t):
    return ~np.isnan(y[:, t])


def both_aurc(rc):
    return S.aurc(rc, COV), S.aurc(rc[OPMASK], COV[OPMASK])


# --------------------------------------------------------------------------
def raw_scores(det, mc, cue, t, thr, T=1.0):
    """Every uncertainty score for one cohort (higher == less trustworthy).

    ``thr`` is chosen on temperature-scaled probabilities, so the scores are
    computed on the same scale; with T = 1 this is the identity.
    """
    p, mct = S.apply_temperature(det[:, t], T), S.apply_temperature(mc[:, :, t], T)
    return {
        "MSP": S.u_msp(p),
        "Entropy": S.u_entropy(p),
        "Margin (threshold-aware)": S.u_margin_threshold(p, thr),
        "MC-entropy": S.u_mc_entropy(mct),
        "MC-variance": S.u_mc_var(mct),
        "MC-BALD": S.u_mc_bald(mct),
        "Cue disagreement (symmetric)": cue.inconsistency(det),
        "Cue contradiction": cue.contradiction(det, thr),
    }


class Fusion:
    """Non-negative class-balanced logistic combination of uncertainty terms.

    Every term is constructed so that larger means *less* trustworthy, so a
    negative coefficient is not a meaningful solution -- it says that cases far
    from the decision boundary are the risky ones.  With only ~100 positive
    validation cases an unconstrained fit does occasionally return one, purely
    from noise, so the weights are constrained to the non-negative orthant.
    Three free parameters, no grid search, and no test data.
    """

    def __init__(self, terms):
        self.terms, self.w, self.b = terms, None, 0.0

    @staticmethod
    def _nll(theta, Z, err, sw):
        z = Z @ theta[1:] + theta[0]
        # numerically stable class-balanced logistic loss
        ll = np.logaddexp(0.0, z) - err * z
        return float(np.sum(sw * ll) / np.sum(sw))

    def fit(self, Z, err, sw):
        from scipy.optimize import minimize
        k = Z.shape[1]
        r = minimize(self._nll, np.r_[0.0, np.ones(k)], args=(Z, err, sw),
                     method="L-BFGS-B",
                     bounds=[(None, None)] + [(0.0, None)] * k)
        self.b, self.w = float(r.x[0]), r.x[1:]
        return self

    def score(self, Z):
        return Z @ self.w + self.b

    @property
    def coef_(self):                      # keeps the reporting code unchanged
        return np.array([self.w])


def _design(raw, nz, terms):
    return np.column_stack([nz[k](raw[k]) for k in terms])


def fit_combination(raw_v, nz, y_val, p_val, thr, terms):
    Z = _design(raw_v, nz, terms)
    err = ((p_val >= thr).astype(int) != y_val).astype(int)
    pi = float(np.clip(y_val.mean(), 1e-6, 1 - 1e-6))
    sw = np.where(y_val == 1, 0.5 / pi, 0.5 / (1 - pi))
    return Fusion(terms).fit(Z, err, sw)


def apply_combination(lr, raw, nz, terms):
    return lr.score(_design(raw, nz, terms))


# --------------------------------------------------------------------------
def run_target(name, t, args):
    print(f"\n{'='*72}\nTARGET: {name}\n{'='*72}", flush=True)
    cue_idx = [i for i in range(len(LABELS)) if i != t]

    meta = pd.read_csv("/NHNHOME/uscnet/data/proc/cxr14_meta.csv")
    Ytr_lab = meta.loc[meta.split == "train", LABELS].values.astype(np.float64)
    assoc = S.CueBank(t, cue_idx).fit_association(Ytr_lab)

    per_seed, sel_rows, ext_rows, cuelevel, sens = [], [], [], [], []
    paired, cue_last = {}, None

    for seed in SEEDS:
        # ---- cue bank: evidence model on this seed's own train predictions --
        dtr, _, ytr = load("train", seed, need_mc=False)
        cue = S.CueBank(t, cue_idx)
        cue.lr_assoc = assoc.lr_assoc
        cue.fit_evidence(dtr, ytr[:, t])
        del dtr, ytr
        cue_last = cue

        dv_all, mv, yv_all = load("val", seed)
        dt_all, mt, yt_all = load("test", seed)
        ov, ot = valid_rows(yv_all, t), valid_rows(yt_all, t)
        dv, mv, yv_t = dv_all[ov], mv[:, ov], yv_all[ov, t]
        dt, mt, yt_t = dt_all[ot], mt[:, ot], yt_all[ot, t]

        # ---- calibration, operating threshold, cue threshold (validation) ---
        T = S.fit_temperature(yv_t, dv[:, t])
        p_val = S.apply_temperature(dv[:, t], T)
        thr = S.youden_threshold(yv_t, p_val)
        cue.fit_cue_threshold(dv, yv_t)
        p_test = S.apply_temperature(dt[:, t], T)

        # ---- scores and validation-fitted fusion ---------------------------
        raw_v = raw_scores(dv, mv, cue, t, thr, T)
        raw_t = raw_scores(dt, mt, cue, t, thr, T)
        nz = {k: S.RankNormalizer(v) for k, v in raw_v.items()}
        lr3 = fit_combination(raw_v, nz, yv_t, p_val, thr, TERMS)
        lr2 = fit_combination(raw_v, nz, yv_t, p_val, thr, TERMS[:2])

        methods = dict(raw_t)
        methods["Composite (no cue)"] = apply_combination(lr2, raw_t, nz, TERMS[:2])
        methods["U-SCNet (cue-aware)"] = apply_combination(lr3, raw_t, nz, TERMS)
        methods["U-SCNet (equal weights)"] = sum(nz[k](raw_t[k]) for k in TERMS)

        # ---- predictor-level discrimination and calibration ----------------
        # Would a validation-only decision have chosen correctly between the
        # two-term and three-term scores?  Recorded so that Section 5.4 can
        # report whether including the cue term can be decided without test data.
        val_u3 = apply_combination(lr3, raw_v, nz, TERMS)
        val_u2 = apply_combination(lr2, raw_v, nz, TERMS[:2])
        va3 = both_aurc(S.risk_coverage(yv_t, p_val, val_u3, thr, COV))
        va2 = both_aurc(S.risk_coverage(yv_t, p_val, val_u2, thr, COV))

        rec = {"seed": seed, "T": T, "thr": thr,
               "w_margin": lr3.coef_[0][0], "w_mcvar": lr3.coef_[0][1],
               "w_cue": lr3.coef_[0][2],
               "val_aurc_cue": va3[0], "val_aurc_nocue": va2[0],
               "val_aurcop_cue": va3[1], "val_aurcop_nocue": va2[1],
               "cue_evidence_auroc": roc_auc_score(yt_t, cue.evidence(dt)),
               "n_test": int(len(yt_t)), "n_pos": int(yt_t.sum())}
        for tag, pp in [("det", dt[:, t]), ("temp", p_test),
                        ("mc_mean", mt[:, :, t].mean(0))]:
            rec[f"auroc_{tag}"] = roc_auc_score(yt_t, pp)
            rec[f"auprc_{tag}"] = average_precision_score(yt_t, pp)
            rec[f"ece_{tag}"] = S.ece_equal_mass(yt_t, pp)
            rec[f"ececonf_{tag}"] = S.ece_confidence(yt_t, pp)
            rec[f"brier_{tag}"] = S.brier(yt_t, pp)
            rec[f"nll_{tag}"] = S.nll(yt_t, pp)
        rec["sens_full"], rec["spec_full"] = S.sens_spec(yt_t, p_test, thr)
        rec["balerr_full"] = S.balanced_error(yt_t, p_test, thr)
        per_seed.append(rec)

        # ---- selective prediction: predictor fixed, ranking varies ---------
        for mname, u in methods.items():
            rc = S.risk_coverage(yt_t, p_test, u, thr, COV)
            a_all, a_op = both_aurc(rc)
            row = {"seed": seed, "method": mname, "aurc": a_all, "aurc_op": a_op,
                   "rc": rc.tolist()}
            for c in args.report_cov:
                for k, v in S.summarize_at_coverage(yt_t, p_test, u, thr, c).items():
                    row[f"{k}@{int(c*100)}"] = v
            sel_rows.append(row)
            paired.setdefault(mname, []).append((a_all, a_op))

        # ---- weight sensitivity (robustness of the fusion) -----------------
        for lam, gam in itertools.product(args.lam_grid, args.gam_grid):
            u = (nz[TERMS[0]](raw_t[TERMS[0]]) + lam * nz[TERMS[1]](raw_t[TERMS[1]])
                 + gam * nz[TERMS[2]](raw_t[TERMS[2]]))
            a_all, a_op = both_aurc(S.risk_coverage(yt_t, p_test, u, thr, COV))
            sens.append({"seed": seed, "lambda": lam, "gamma": gam,
                         "aurc": a_all, "aurc_op": a_op})

        # ---- cue-head quality against real CXR8 labels ---------------------
        for ci in cue_idx:
            yc, pc = yt_all[ot, ci], dt[:, ci]
            if yc.sum() < 10:
                continue
            thr_c = S.youden_threshold(yv_all[ov, ci], dv[:, ci])
            pred = (pc >= thr_c).astype(int)
            cuelevel.append({
                "seed": seed, "cue": LABELS[ci], "prevalence": float(yc.mean()),
                "auroc": roc_auc_score(yc, pc),
                "auprc": average_precision_score(yc, pc),
                "precision": precision_score(yc, pred, zero_division=0),
                "recall": recall_score(yc, pred, zero_division=0),
                "f1": f1_score(yc, pred, zero_division=0),
                "kappa": cohen_kappa_score(yc, pred),
                "evidence_weight": float(cue.weights[cue_idx.index(ci)]),
                "assoc_weight": float(cue.assoc_weights[cue_idx.index(ci)]),
            })

        # ---- external cohorts ----------------------------------------------
        for coh in args.ext_cohorts:
            try:
                de_all, me, ye_all = load(coh, seed)
            except FileNotFoundError:
                continue
            ok = valid_rows(ye_all, t)
            # the official CheXpert validation partition has only 8 pneumonia
            # positives; it is kept, with the caveat stated in the manuscript,
            # precisely because it is often reported as if it were adequate
            if ok.sum() < 50 or np.nansum(ye_all[ok, t]) < 5:
                continue
            de, me, ye_t = de_all[ok], me[:, ok], ye_all[ok, t]
            pe = S.apply_temperature(de[:, t], T)
            raw_e = raw_scores(de, me, cue, t, thr, T)
            r = {"seed": seed, "cohort": coh, "n": int(ok.sum()),
                 "n_pos": int(ye_t.sum()),
                 "auroc": roc_auc_score(ye_t, pe),
                 "auprc": average_precision_score(ye_t, pe),
                 "ece": S.ece_equal_mass(ye_t, pe), "brier": S.brier(ye_t, pe),
                 "balerr_full": S.balanced_error(ye_t, pe, thr)}
            pred_e = (pe >= thr).astype(int)
            r["prevalence"] = float(ye_t.mean())
            r["ppr"] = float(pred_e.mean())
            r["ppv"] = float(ye_t[pred_e == 1].mean()) if pred_e.sum() else np.nan
            for tg, u in [("uscnet", apply_combination(lr3, raw_e, nz, TERMS)),
                          ("nocue", apply_combination(lr2, raw_e, nz, TERMS[:2])),
                          ("margin", raw_e["Margin (threshold-aware)"]),
                          ("mcent", raw_e["MC-entropy"]),
                          ("msp", raw_e["MSP"])]:
                rc = S.risk_coverage(ye_t, pe, u, thr, COV)
                r[f"aurc_{tg}"], r[f"aurcop_{tg}"] = both_aurc(rc)
                for c in args.report_cov:
                    d = S.summarize_at_coverage(ye_t, pe, u, thr, c)
                    r[f"balerr_{tg}@{int(c*100)}"] = d["bal_err"]
                    r[f"auroc_{tg}@{int(c*100)}"] = d["auroc"]
                # class-resolved coverage and the generalised risk, so that the
                # external cohort is reported in the same terms as Table 1
                for strat, short in (("global", "g"), ("class-conditional", "cc")):
                    masks = [I.select_global(u, c) if strat == "global"
                             else I.select_class_conditional(u, pred_e, c) for c in COV]
                    ev = I.evaluate_strategy(ye_t, pe, thr, masks, COV, u=u)
                    r[f"covpos80_{tg}_{short}"] = ev["cov_pos@80"]
                    r[f"disparity80_{tg}_{short}"] = ev["disparity@80"]
                    r[f"augrc_{tg}_{short}"] = ev["AUGRC"]
                    r[f"aurcb_{tg}_{short}"] = ev["AURC"]
            ext_rows.append(r)

        print(f"  seed {seed}: T={T:.3f} tau={thr:.5f} "
              f"w=({lr3.coef_[0][0]:.2f},{lr3.coef_[0][1]:.2f},{lr3.coef_[0][2]:.2f}) "
              f"AUROC={rec['auroc_temp']:.4f} cueAUROC={rec['cue_evidence_auroc']:.4f}",
              flush=True)

    ens = ensemble_analysis(t, cue_idx, args)
    dump(name, per_seed, sel_rows, ext_rows, cuelevel, sens, ens, paired,
         cue_last, cue_idx)


# --------------------------------------------------------------------------
def ensemble_analysis(t, cue_idx, args):
    """Deterministic deep ensemble over the seed checkpoints."""
    dtr = np.stack([load("train", s, need_mc=False)[0] for s in SEEDS]).mean(0)
    ytr = load("train", SEEDS[0], need_mc=False)[2]
    cue = S.CueBank(t, cue_idx).fit_evidence(dtr, ytr[:, t])
    del dtr, ytr

    dv = np.stack([load("val", s, need_mc=False)[0] for s in SEEDS])
    dt = np.stack([load("test", s, need_mc=False)[0] for s in SEEDS])
    yv = load("val", SEEDS[0], need_mc=False)[2]
    yt = load("test", SEEDS[0], need_mc=False)[2]
    ov, ot = valid_rows(yv, t), valid_rows(yt, t)
    dv, dt, yv_t, yt_t = dv[:, ov], dt[:, ot], yv[ov, t], yt[ot, t]

    pv, pt = dv.mean(0), dt.mean(0)
    T = S.fit_temperature(yv_t, pv[:, t])
    pv_c, pt_c = S.apply_temperature(pv[:, t], T), S.apply_temperature(pt[:, t], T)
    thr = S.youden_threshold(yv_t, pv_c)
    cue.fit_cue_threshold(pv, yv_t)

    raw_v = {TERMS[0]: S.u_margin_threshold(pv_c, thr),
             TERMS[1]: dv[:, :, t].var(0),
             TERMS[2]: cue.contradiction(pv, thr)}
    raw_t = {TERMS[0]: S.u_margin_threshold(pt_c, thr),
             TERMS[1]: dt[:, :, t].var(0),
             TERMS[2]: cue.contradiction(pt, thr)}
    nz = {k: S.RankNormalizer(v) for k, v in raw_v.items()}
    lr3 = fit_combination(raw_v, nz, yv_t, pv_c, thr, TERMS)
    lr2 = fit_combination(raw_v, nz, yv_t, pv_c, thr, TERMS[:2])

    res = {"n_models": len(SEEDS), "T": T, "thr": thr,
           "auroc": roc_auc_score(yt_t, pt_c),
           "auprc": average_precision_score(yt_t, pt_c),
           "ece": S.ece_equal_mass(yt_t, pt_c), "brier": S.brier(yt_t, pt_c),
           "nll": S.nll(yt_t, pt_c),
           "balerr_full": S.balanced_error(yt_t, pt_c, thr)}
    for tg, u in [("ensvar", raw_t[TERMS[1]]),
                  ("nocue", apply_combination(lr2, raw_t, nz, TERMS[:2])),
                  ("uscnet", apply_combination(lr3, raw_t, nz, TERMS))]:
        rc = S.risk_coverage(yt_t, pt_c, u, thr, COV)
        res[f"aurc_{tg}"], res[f"aurcop_{tg}"] = both_aurc(rc)
        for c in args.report_cov:
            res[f"balerr_{tg}@{int(c*100)}"] = S.summarize_at_coverage(
                yt_t, pt_c, u, thr, c)["bal_err"]
    return res


# --------------------------------------------------------------------------
def dump(name, per_seed, sel_rows, ext_rows, cuelevel, sens, ens, paired,
         cue, cue_idx):
    os.makedirs(OUT, exist_ok=True)
    tag = name.lower()
    pd.DataFrame(per_seed).to_csv(f"{OUT}/{tag}_perseed.csv", index=False)
    pd.DataFrame(cuelevel).to_csv(f"{OUT}/{tag}_cuelevel.csv", index=False)
    pd.DataFrame(sens).to_csv(f"{OUT}/{tag}_weight_sensitivity.csv", index=False)
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(f"{OUT}/{tag}_external.csv", index=False)

    sel = pd.DataFrame(sel_rows)
    sel.drop(columns=["rc"]).to_csv(f"{OUT}/{tag}_selective_perseed.csv", index=False)
    num = [c for c in sel.columns if c not in ("rc", "method", "seed")]
    sel.groupby("method")[num].agg(["mean", "std"]).to_csv(
        f"{OUT}/{tag}_selective_summary.csv")

    # paired comparison of U-SCNet against every other ranking, across seeds
    ref = "U-SCNet (cue-aware)"
    stats = []
    for m in paired:
        if m == ref or len(paired[m]) < 3:
            continue
        a = np.array([x[0] for x in paired[ref]]) - np.array([x[0] for x in paired[m]])
        b = np.array([x[1] for x in paired[ref]]) - np.array([x[1] for x in paired[m]])
        row = {"method": m, "d_aurc_mean": a.mean(), "d_aurc_sd": a.std(ddof=1),
               "d_aurcop_mean": b.mean(), "d_aurcop_sd": b.std(ddof=1)}
        for k, d in (("aurc", a), ("aurcop", b)):
            row[f"wilcoxon_p_{k}"] = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        stats.append(row)
    pd.DataFrame(stats).to_csv(f"{OUT}/{tag}_paired_stats.csv", index=False)

    rc_mean = {m: np.nanmean(np.stack(g["rc"].map(np.array)), 0).tolist()
               for m, g in sel.groupby("method")}
    json.dump({"coverage": COV.tolist(), "risk_curves": rc_mean, "ensemble": ens,
               "cue_evidence_weights": dict(zip([LABELS[i] for i in cue_idx],
                                                cue.weights.tolist())),
               "cue_assoc_weights": dict(zip([LABELS[i] for i in cue_idx],
                                             cue.assoc_weights.tolist()))},
              open(f"{OUT}/{tag}_curves.json", "w"), indent=2)

    print(f"\n-- {name}: AURC (lower is better) --")
    print(sel.groupby("method")[["aurc", "aurc_op"]].agg(["mean", "std"])
          .round(4).to_string())
    print(f"-- ensemble: AUROC={ens['auroc']:.4f} aurc nocue={ens['aurc_nocue']:.4f} "
          f"uscnet={ens['aurc_uscnet']:.4f}")
    if stats:
        print("\n-- paired vs U-SCNet (negative = U-SCNet better) --")
        print(pd.DataFrame(stats)[["method", "d_aurc_mean", "wilcoxon_p_aurc",
                                   "d_aurcop_mean", "wilcoxon_p_aurcop"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--report_cov", nargs="+", type=float,
                    default=[1.0, 0.9, 0.8, 0.7, 0.5])
    ap.add_argument("--lam_grid", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--gam_grid", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--ext_cohorts", nargs="+",
                    default=["chexpert_ext", "chexpert_expert"])
    a = ap.parse_args()
    for nm in a.targets:
        run_target(nm, TARGETS[nm], a)
