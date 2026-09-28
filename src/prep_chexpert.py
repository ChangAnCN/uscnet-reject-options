"""Prepare CheXpert as an external validation cohort for CXR8-trained models.

Two cohorts are built:
  * ``ext``   -- the CheXpert train partition (NLP-derived labels, large N)
  * ``expert``-- the official 234-image validation partition (radiologist labels)

CheXpert class codes in this mirror are 0=unlabeled, 1=uncertain, 2=absent,
3=present.  We keep only definite labels (absent/present); uncertain rows are
dropped for the target finding, and unlabeled is mapped to negative, which is
the standard CheXpert evaluation convention for un-mentioned findings.
"""
import argparse
import glob
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

CACHE_SIZE = 256
# CheXpert finding -> CXR8 label. Only findings that exist in both taxonomies.
CHEX2CXR = {
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
}


def _decode(buf):
    im = Image.open(io.BytesIO(buf)).convert("L").resize(
        (CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def build(files, out_prefix, out_dir, workers=48):
    rows, imgs = [], []
    for f in files:
        t = pq.read_table(f)
        df = t.to_pandas()
        df = df[df["Frontal/Lateral"] == 0]           # frontal views only
        if len(df) == 0:
            continue
        with ProcessPoolExecutor(max_workers=workers) as ex:
            arrs = list(ex.map(_decode, [b["bytes"] for b in df["image"]],
                               chunksize=32))
        imgs.extend(arrs)
        rows.append(df.drop(columns=["image"]).reset_index(drop=True))
        print(f"  {os.path.basename(f)}: +{len(df)} frontal", flush=True)

    meta = pd.concat(rows, ignore_index=True)
    arr = np.stack(imgs)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"chexpert_{out_prefix}_{CACHE_SIZE}.u8.npy"), arr)

    # 3=present -> 1, 2=absent -> 0, 0=unlabeled -> 0, 1=uncertain -> NaN (drop)
    for chex, cxr in CHEX2CXR.items():
        v = meta[chex].astype(float)
        meta[cxr] = np.where(v == 3, 1.0, np.where(v == 1, np.nan, 0.0))
    meta["idx"] = np.arange(len(meta))
    meta.to_csv(os.path.join(out_dir, f"chexpert_{out_prefix}_meta.csv"), index=False)

    print(f"[{out_prefix}] n={len(meta)}  cache={arr.shape}")
    for cxr in CHEX2CXR.values():
        col = meta[cxr]
        print(f"   {cxr:14s} pos={int(np.nansum(col)):6d} "
              f"definite={int(col.notna().sum()):6d} "
              f"prev={100*np.nanmean(col):.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/NHNHOME/uscnet/data/chexpert_raw/data")
    ap.add_argument("--out", default="/NHNHOME/uscnet/data/proc")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    build(sorted(glob.glob(os.path.join(a.raw, "validation-*.parquet"))),
          "expert", a.out, a.workers)
    build(sorted(glob.glob(os.path.join(a.raw, "train-*.parquet"))),
          "ext", a.out, a.workers)
