"""Train a 14-label DenseNet121 on the official CXR8 split.

One run == one seed. Checkpoint selection is by mean validation AUROC over the
14 findings, so the backbone is chosen without reference to the test set.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from common import (LABELS, CXRDataset, DenseNet121Multi, load_cache, load_meta,
                    to_input)


def evaluate(model, loader, device, mc=False):
    model.enable_mc_dropout() if mc else model.eval()
    P, Y = [], []
    with torch.no_grad():
        for u8, y in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(to_input(u8, device))
            P.append(torch.sigmoid(logit.float()).cpu())
            Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def mean_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])
            if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(aucs)), aucs


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    meta, cache = load_meta(), load_cache()
    Y = meta[LABELS].values.astype(np.int8)
    tr = meta.index[meta.split == "train"].values
    va = meta.index[meta.split == "val"].values

    dl_tr = DataLoader(CXRDataset(cache, tr, Y[tr], train=True), batch_size=a.bs,
                       shuffle=True, num_workers=a.workers, pin_memory=True,
                       drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(CXRDataset(cache, va, Y[va]), batch_size=a.bs * 2,
                       shuffle=False, num_workers=a.workers, pin_memory=True,
                       persistent_workers=True)

    model = DenseNet121Multi(len(LABELS), a.dropout).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.1)
    crit = nn.BCEWithLogitsLoss()

    best, hist = -1.0, []
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"densenet121_seed{a.seed}.pt")

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for u8, y in dl_tr:
            x = to_input(u8, device).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        p, yv = evaluate(model, dl_va, device)
        m, aucs = mean_auc(yv, p)
        pna = aucs[LABELS.index("Pneumonia")]
        hist.append({"epoch": ep, "loss": tot / n, "val_mean_auc": m,
                     "val_pneumonia_auc": pna, "sec": time.time() - t0})
        print(f"[seed {a.seed}] ep{ep:02d} loss {tot/n:.4f} "
              f"val_mAUC {m:.4f} pneu {pna:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m > best:
            best = m
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_mean_auc": m, "seed": a.seed}, ckpt)

    json.dump({"seed": a.seed, "best_val_mean_auc": best, "history": hist,
               "args": vars(a)},
              open(os.path.join(a.out, f"trainlog_seed{a.seed}.json"), "w"), indent=2)
    print(f"[seed {a.seed}] best val mean AUC {best:.4f} -> {ckpt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="/NHNHOME/uscnet/ckpt")
    main(ap.parse_args())
