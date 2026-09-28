"""Build the CXR8 (ChestX-ray14) label matrix, patient-disjoint splits, and a
uint8 image cache used by every downstream experiment.

Images are decoded once, resized to 256x256 grayscale, and stored in a single
memmap so that training is not I/O bound.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
CACHE_SIZE = 256


def build_metadata(raw_dir, out_dir, val_frac=0.15, seed=0):
    meta = pd.read_csv(os.path.join(raw_dir, "Data_Entry_2017_v2020.csv"))
    meta = meta.rename(columns={"Image Index": "image", "Finding Labels": "findings",
                                "Patient ID": "patient"})
    for lab in LABELS:
        meta[lab] = meta["findings"].str.contains(lab, regex=False).astype(np.int8)

    test_ids = set(pd.read_csv(os.path.join(raw_dir, "test_list.txt"),
                               header=None)[0].tolist())
    trval_ids = set(pd.read_csv(os.path.join(raw_dir, "train_val_list.txt"),
                                header=None)[0].tolist())

    meta["split"] = "unused"
    meta.loc[meta["image"].isin(test_ids), "split"] = "test"
    trval_mask = meta["image"].isin(trval_ids)

    # Patient-disjoint train/val split of the official train_val partition, so no
    # patient contributes images to both.
    rng = np.random.RandomState(seed)
    trval_patients = np.sort(meta.loc[trval_mask, "patient"].unique())
    rng.shuffle(trval_patients)
    n_val = int(round(val_frac * len(trval_patients)))
    val_patients = set(trval_patients[:n_val].tolist())

    is_val = trval_mask & meta["patient"].isin(val_patients)
    meta.loc[trval_mask, "split"] = "train"
    meta.loc[is_val, "split"] = "val"

    meta = meta[meta["split"] != "unused"].reset_index(drop=True)
    meta["idx"] = np.arange(len(meta))
    os.makedirs(out_dir, exist_ok=True)
    meta.to_csv(os.path.join(out_dir, "cxr14_meta.csv"), index=False)

    # Sanity: the official test partition must share no patient with train/val.
    tr_pat = set(meta.loc[meta.split.isin(["train", "val"]), "patient"])
    te_pat = set(meta.loc[meta.split == "test", "patient"])
    assert not (tr_pat & te_pat), "patient leakage between train/val and test"
    assert not (set(meta.loc[meta.split == "train", "patient"]) & val_patients)

    print("split sizes:\n", meta["split"].value_counts())
    print("\nprevalence (%):")
    print((meta.groupby("split")[LABELS].mean() * 100).round(2).T)
    return meta


def _load_one(args):
    i, path = args
    try:
        im = Image.open(path).convert("L").resize((CACHE_SIZE, CACHE_SIZE),
                                                  Image.BILINEAR)
        return i, np.asarray(im, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - surfaces corrupt files
        print("FAILED", path, exc, flush=True)
        return i, None


def build_cache(meta, png_dir, out_dir, workers=48):
    n = len(meta)
    cache_path = os.path.join(out_dir, f"cxr14_{CACHE_SIZE}.u8")
    arr = np.lib.format.open_memmap(cache_path + ".npy", mode="w+",
                                    dtype=np.uint8, shape=(n, CACHE_SIZE, CACHE_SIZE))
    jobs = [(i, os.path.join(png_dir, im)) for i, im in enumerate(meta["image"])]
    bad = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (i, img) in enumerate(ex.map(_load_one, jobs, chunksize=64)):
            if img is None:
                bad += 1
                continue
            arr[i] = img
            if k % 10000 == 0:
                print(f"  cached {k}/{n}", flush=True)
    arr.flush()
    print(f"cache written -> {cache_path}.npy  (failed: {bad})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/NHNHOME/uscnet/data/cxr14_raw/data")
    ap.add_argument("--png", default="/NHNHOME/uscnet/data/cxr14_png")
    ap.add_argument("--out", default="/NHNHOME/uscnet/data/proc")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    m = build_metadata(a.raw, a.out)
    build_cache(m, a.png, a.out, a.workers)
