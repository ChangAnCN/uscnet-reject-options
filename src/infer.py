"""Cache deterministic and MC-dropout predictions for every trained checkpoint.

The whole uint8 cohort is staged on the GPU once and each batch is passed
through the network 1 + T times (deterministic, then T MC-dropout draws), so
the run is compute-bound rather than loader-bound.

Writes one .npz per (seed, cohort) with the deterministic probabilities and the
full MC sample stack, so every selective-prediction variant can be scored later
without re-running the network.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch

from common import (CACHE_SIZE, CROP, IMAGENET_MEAN, IMAGENET_STD, LABELS,
                    DenseNet121Multi, load_cache, load_meta)

PROC = "/NHNHOME/uscnet/data/proc"


def cohort(name):
    """Return (uint8 image array, label matrix) for a named cohort."""
    if name in ("train", "val", "test"):
        meta, cache = load_meta(), load_cache()
        idx = meta.index[meta.split == name].values
        return np.asarray(cache[idx]), meta[LABELS].values.astype(np.float32)[idx]
    if name in ("chexpert_ext", "chexpert_expert"):
        tag = name.split("_")[1]
        meta = pd.read_csv(os.path.join(PROC, f"chexpert_{tag}_meta.csv"))
        arr = np.load(os.path.join(PROC, f"chexpert_{tag}_256.u8.npy"), mmap_mode="r")
        Y = np.full((len(meta), len(LABELS)), np.nan, dtype=np.float32)
        for i, l in enumerate(LABELS):
            if l in meta.columns:
                Y[:, i] = meta[l].values
        return np.asarray(arr), Y
    raise ValueError(name)


@torch.no_grad()
def predict_gpu(model, imgs_u8, device, n_mc, seed, bs):
    """imgs_u8: numpy [N,256,256] uint8 -> (det [N,14], mc [T,N,14])."""
    n = len(imgs_u8)
    off = (CACHE_SIZE - CROP) // 2
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    det = np.empty((n, len(LABELS)), np.float32)
    mc = np.empty((n_mc, n, len(LABELS)), np.float16)
    gen = torch.Generator(device=device).manual_seed(seed)

    for s in range(0, n, bs):
        e = min(s + bs, n)
        blk = torch.from_numpy(imgs_u8[s:e, off:off + CROP, off:off + CROP])
        x = blk.to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)
        x = ((x.expand(-1, 3, -1, -1) - mean) / std).to(memory_format=torch.channels_last)

        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            det[s:e] = torch.sigmoid(model(x).float()).cpu().numpy()
        model.enable_mc_dropout()
        for t in range(n_mc):
            # deterministic dropout masks per (seed, pass) for reproducibility
            torch.cuda.manual_seed(seed * 1000 + t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mc[t, s:e] = torch.sigmoid(model(x).float()).cpu().numpy().astype(np.float16)
    del gen
    return det, mc


def main(a):
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(a.ckpt, "densenet121_seed*.pt")),
                   key=lambda c: int(re.search(r"seed(\d+)", c).group(1)))
    print(f"{len(ckpts)} checkpoints x {len(a.cohorts)} cohorts", flush=True)

    for name in a.cohorts:
        def dst_for(cp):
            sd = re.search(r"seed(\d+)", cp).group(1)
            return os.path.join(a.out, f"pred_{name}_seed{sd}.npz")

        todo = [c for c in ckpts if a.overwrite or not os.path.exists(dst_for(c))]
        if not todo:
            print(f"  {name}: all cached"); continue
        imgs, Y = cohort(name)
        print(f"  cohort {name}: {imgs.shape}", flush=True)
        for cp in todo:
            seed = int(re.search(r"seed(\d+)", cp).group(1))
            dst = os.path.join(a.out, f"pred_{name}_seed{seed}.npz")
            model = DenseNet121Multi(len(LABELS), a.dropout).to(device)
            model.load_state_dict(torch.load(cp, map_location=device)["model"])
            model = model.to(memory_format=torch.channels_last)
            det, mc = predict_gpu(model, imgs, device, a.n_mc, seed, a.bs)
            np.savez_compressed(dst, det=det, mc=mc, y=Y)
            print(f"    seed{seed} -> {os.path.basename(dst)} "
                  f"det{det.shape} mc{mc.shape}", flush=True)
            del model
            torch.cuda.empty_cache()
        del imgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/NHNHOME/uscnet/ckpt")
    ap.add_argument("--out", default="/NHNHOME/uscnet/results/preds")
    ap.add_argument("--cohorts", nargs="+",
                    default=["val", "test", "chexpert_expert", "chexpert_ext"])
    ap.add_argument("--n_mc", type=int, default=20)
    ap.add_argument("--bs", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
