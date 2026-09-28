"""Per-finding backbone evaluation on the CXR8 official test split.

Produces the table and figure data that establish the backbone is a credible
reference point (comparison against Wang et al., 2017 and CheXNet), before any
selective-prediction claim is made.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import selective as S
from common import LABELS
import paths

RES = paths.RESULTS
PRED = f"{RES}/preds"
SEEDS = [42, 123, 3407, 7, 2024]

# Published AUROC on the same official split, for context in the manuscript.
PUBLISHED = {
    "Atelectasis": {"wang2017": 0.700, "chexnet": 0.8094},
    "Cardiomegaly": {"wang2017": 0.810, "chexnet": 0.9248},
    "Effusion": {"wang2017": 0.759, "chexnet": 0.8638},
    "Infiltration": {"wang2017": 0.661, "chexnet": 0.7345},
    "Mass": {"wang2017": 0.693, "chexnet": 0.8676},
    "Nodule": {"wang2017": 0.669, "chexnet": 0.7802},
    "Pneumonia": {"wang2017": 0.658, "chexnet": 0.7680},
    "Pneumothorax": {"wang2017": 0.799, "chexnet": 0.8887},
    "Consolidation": {"wang2017": 0.703, "chexnet": 0.7901},
    "Edema": {"wang2017": 0.805, "chexnet": 0.8878},
    "Emphysema": {"wang2017": 0.833, "chexnet": 0.9371},
    "Fibrosis": {"wang2017": 0.786, "chexnet": 0.8047},
    "Pleural_Thickening": {"wang2017": 0.684, "chexnet": 0.8062},
    "Hernia": {"wang2017": 0.872, "chexnet": 0.9164},
}


def main():
    det = np.stack([np.load(f"{PRED}/pred_test_seed{s}.npz")["det"] for s in SEEDS])
    y = np.load(f"{PRED}/pred_test_seed{SEEDS[0]}.npz")["y"]
    ens = det.mean(0)

    rows = []
    for i, lab in enumerate(LABELS):
        a = [roc_auc_score(y[:, i], det[k, :, i]) for k in range(len(SEEDS))]
        p = [average_precision_score(y[:, i], det[k, :, i]) for k in range(len(SEEDS))]
        lo, hi = S.bootstrap_ci(lambda yy, pp: roc_auc_score(yy, pp),
                                y[:, i], ens[:, i], n_boot=500)
        rows.append({
            "label": lab, "n_pos": int(y[:, i].sum()),
            "prevalence": float(y[:, i].mean()),
            "auroc_mean": float(np.mean(a)), "auroc_std": float(np.std(a, ddof=1)),
            "auprc_mean": float(np.mean(p)), "auprc_std": float(np.std(p, ddof=1)),
            "auroc_ens": float(roc_auc_score(y[:, i], ens[:, i])),
            "auroc_ens_lo": lo, "auroc_ens_hi": hi,
            "wang2017": PUBLISHED[lab]["wang2017"],
            "chexnet": PUBLISHED[lab]["chexnet"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/backbone_perlabel.csv", index=False)

    summ = {
        "mean_auroc_single": float(df.auroc_mean.mean()),
        "mean_auroc_single_sd": float(np.std(
            [np.mean([roc_auc_score(y[:, i], det[k, :, i])
                      for i in range(len(LABELS))]) for k in range(len(SEEDS))], ddof=1)),
        "mean_auroc_ensemble": float(df.auroc_ens.mean()),
        "mean_auroc_wang2017": float(df.wang2017.mean()),
        "mean_auroc_chexnet": float(df.chexnet.mean()),
        "n_test": int(len(y)), "seeds": SEEDS,
    }
    json.dump(summ, open(f"{RES}/backbone_summary.json", "w"), indent=2)

    print(df[["label", "n_pos", "auroc_mean", "auroc_std", "auroc_ens",
              "wang2017", "chexnet"]].round(4).to_string(index=False))
    print("\n", json.dumps(summ, indent=2))


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    main()
