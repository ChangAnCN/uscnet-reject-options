"""SelectiveNet (Geifman & El-Yaniv, ICML 2019) baseline for the target condition.

The backbone and training schedule are identical to ``train.py``; the only
change is the head. A selection head g(x) is trained jointly with the prediction
head under the coverage-constrained selective loss

    L = L_(f,g) + lambda * max(0, c - phi(g))^2 + alpha * L_h,
    L_(f,g) = mean_i[ loss(f(x_i), y_i) * g(x_i) ] / phi(g),
    phi(g)  = mean_i[ g(x_i) ],

with an auxiliary head h trained on the unweighted loss.

Two implementation choices keep the comparison fair rather than convenient.

First, the backbone is initialised from the *same* trained checkpoint every
other method in the benchmark uses and fine-tuned at a low learning rate.
Training it from scratch would confound the reject mechanism with backbone
quality; with a 48:1 class weight it also destabilises the shared trunk, and in
pilot runs validation AUROC oscillated between 0.45 and 0.69 across epochs,
which would have understated the baseline.

Second, both loss variants are trained and reported: the original unweighted
objective, and a class-weighted one matched to the balanced risk this paper
evaluates. Reporting only the unweighted variant would penalise SelectiveNet for
optimising a different quantity than the one it is scored on.

At test time g(x) is used as the ranking score, so the full risk-coverage curve
can be traced from a single trained model.
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
import paths


class SelectiveNet(nn.Module):
    def __init__(self, target_idx, p_drop=0.2):
        super().__init__()
        self.base = DenseNet121Multi(len(LABELS), p_drop)
        d = self.base.classifier.in_features
        self.t = target_idx
        self.selector = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Linear(512, 1), nn.Sigmoid())
        self.aux = nn.Linear(d, 1)

    def features(self, x):
        f = self.base.features(x)
        f = torch.nn.functional.relu(f, inplace=True)
        return torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)

    def forward(self, x):
        f = self.features(x)
        logits = self.base.classifier(self.base.drop(f))   # all 14 findings
        g = self.selector(f).squeeze(1)                    # selection head
        h = self.aux(f).squeeze(1)                         # auxiliary head
        return logits, g, h


def selective_loss(logit_t, g, h, y, w, cov, lam):
    """Original SelectiveNet objective (Geifman & El-Yaniv, 2019).

    The selective risk is normalised by the *overall* coverage phi(g) and the
    constraint is on phi(g) alone. Under class imbalance this makes rejecting
    the minority class nearly free: dropping a positive removes a large weighted
    loss from the numerator while moving phi by about 1/N.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    per = bce(logit_t, y, weight=w, reduction="none")
    phi = g.mean()
    l_fg = (per * g).mean() / phi.clamp_min(1e-6)
    pen = lam * torch.clamp(cov - phi, min=0.0) ** 2
    l_h = bce(h, y, weight=w, reduction="mean")
    return l_fg + pen, l_h, phi


def class_balanced_selective_loss(logit_t, g, h, y, cov, lam, eps=1e-6):
    """Class-balanced SelectiveNet -- the objective this paper's analysis implies.

    The risk we report is the class-balanced error on the accepted subset, so
    the training objective is made its empirical counterpart: the selective risk
    is normalised *within each class*, and the coverage constraint is imposed on
    each class separately,

        L = 1/2 * sum_k [ sum_{i in k} l_i g_i / sum_{i in k} g_i ]
            + lambda * sum_k max(0, c - phi_k(g))^2.

    Two things change. Deferring a positive no longer shrinks the numerator for
    free, because that class's risk is divided by its own coverage; and starving
    a class of coverage is now directly penalised instead of invisible. Class
    weights are dropped -- the per-class normalisation supplies the balancing
    they were standing in for, and it does so without inflating the incentive to
    reject the expensive class.

    At 1% prevalence many batches contain no positives; that class's terms are
    skipped for such a batch, so its coverage is driven over the epoch rather
    than within a single step.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    per = bce(logit_t, y, reduction="none")
    risk, pen, phis = [], [], []
    for k in (0, 1):
        m = (y > 0.5) if k == 1 else (y <= 0.5)
        if m.sum() == 0:
            continue
        gk = g[m]
        risk.append((per[m] * gk).sum() / gk.sum().clamp_min(eps))
        pen.append(torch.clamp(cov - gk.mean(), min=0.0) ** 2)
        phis.append(gk.mean())
    if not risk:
        z = g.sum() * 0.0
        return z, bce(h, y, reduction="mean"), g.mean()
    l_fg = torch.stack(risk).mean()
    penalty = lam * torch.stack(pen).sum()
    return l_fg + penalty, bce(h, y, reduction="mean"), torch.stack(phis).mean()


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    t = LABELS.index(a.target)

    meta, cache = load_meta(), load_cache()
    Y = meta[LABELS].values.astype(np.int8)
    tr = meta.index[meta.split == "train"].values
    va = meta.index[meta.split == "val"].values
    pi = float(Y[tr][:, t].mean())
    w_pos, w_neg = 0.5 / pi, 0.5 / (1 - pi)
    print(f"[sn seed {a.seed}] target={a.target} prevalence={pi:.4f} "
          f"w+={w_pos:.1f} w-={w_neg:.2f} target_cov={a.coverage}", flush=True)

    dl_tr = DataLoader(CXRDataset(cache, tr, Y[tr], train=True), batch_size=a.bs,
                       shuffle=True, num_workers=a.workers, pin_memory=True,
                       drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(CXRDataset(cache, va, Y[va]), batch_size=a.bs * 2,
                       shuffle=False, num_workers=a.workers, pin_memory=True,
                       persistent_workers=True)

    model = SelectiveNet(t, a.dropout)
    if a.init_from:
        sd = torch.load(a.init_from, map_location='cpu')['model']
        r = model.base.load_state_dict(sd, strict=False)
        print(f'[sn seed {a.seed}] backbone <- {a.init_from} '
              f'({len(r.missing_keys)} new keys)', flush=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=0.1)
    bce = nn.BCEWithLogitsLoss()

    best, hist = np.inf, []
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(
        a.out, f"selectivenet{a.tag}_{a.target.lower()}_seed{a.seed}.pt")

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for u8, y in dl_tr:
            x = to_input(u8, device).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            yt = y[:, t]
            w = (torch.where(yt > 0.5, w_pos, w_neg) if a.weighted
                 else torch.ones_like(yt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, g, h = model(x)
                if a.loss == "classbalanced":
                    l_sel, l_h, phi = class_balanced_selective_loss(
                        logits[:, t].float(), g.float(), h.float(), yt,
                        a.coverage, a.lam)
                else:
                    l_sel, l_h, phi = selective_loss(
                        logits[:, t].float(), g.float(), h.float(), yt, w,
                        a.coverage, a.lam)
                # keep the other 13 heads supervised so the cue bank stays usable
                l_multi = bce(logits.float(), y)
                loss = a.alpha * l_sel + (1 - a.alpha) * l_h + a.beta * l_multi
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # The selective term divides a 48:1 class-weighted loss by phi(g),
            # which can blow up early when g is small on the positives; without
            # clipping the shared trunk collapses to chance on some seeds.
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)

        # validation: selective balanced error at the target coverage
        model.eval()
        P, G, YY = [], [], []
        with torch.no_grad():
            for u8, y in dl_va:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo, g, _ = model(to_input(u8, device))
                P.append(torch.sigmoid(lo[:, t].float()).cpu())
                G.append(g.float().cpu())
                YY.append(y[:, t])
        p, g, yv = (torch.cat(P).numpy(), torch.cat(G).numpy(), torch.cat(YY).numpy())
        import selective as S
        thr = S.youden_threshold(yv, p)
        k = max(int(round(a.coverage * len(yv))), 2)
        keep = np.argsort(-g)[:k]
        be = S.balanced_error(yv[keep], p[keep], thr)
        auc = roc_auc_score(yv, p)
        hist.append({"epoch": ep, "loss": tot / n, "val_bal_err": be,
                     "val_auc": auc, "phi": float(g.mean()), "sec": time.time() - t0})
        print(f"[sn seed {a.seed}] ep{ep:02d} loss {tot/n:.4f} "
              f"val_balerr {be:.4f} val_auc {auc:.4f} phi {g.mean():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if be < best:
            best = be
            torch.save({"model": model.state_dict(), "epoch": ep, "seed": a.seed,
                        "target": a.target, "coverage": a.coverage}, ckpt)

    json.dump({"seed": a.seed, "target": a.target, "best_val_bal_err": best,
               "history": hist, "args": vars(a)},
              open(os.path.join(a.out,
                   f"snlog{a.tag}_{a.target.lower()}_seed{a.seed}.json"), "w"),
              indent=2)
    print(f"[sn seed {a.seed}] best val balanced error {best:.4f} -> {ckpt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Pneumonia")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--init_from", default="")
    ap.add_argument("--weighted", type=int, default=1)
    ap.add_argument("--loss", choices=["selectivenet", "classbalanced"],
                    default="selectivenet")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=32.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=paths.CKPT)
    main(ap.parse_args())
