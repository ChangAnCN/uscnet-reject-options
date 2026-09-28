"""Shared dataset, model, and I/O helpers for the U-SCNet experiments."""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import paths

cv2.setNumThreads(0)

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
L2I = {l: i for i, l in enumerate(LABELS)}
PROC = paths.PROC
CACHE_SIZE, CROP = 256, 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_meta():
    return pd.read_csv(os.path.join(PROC, "cxr14_meta.csv"))


def load_cache():
    return np.load(os.path.join(PROC, f"cxr14_{CACHE_SIZE}.u8.npy"), mmap_mode="r")


class CXRDataset(Dataset):
    """Serves 224x224 uint8 crops from the in-memory 256x256 cache."""

    def __init__(self, cache, idx, labels, train=False):
        self.cache, self.idx, self.labels, self.train = cache, idx, labels, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = np.asarray(self.cache[self.idx[i]])
        if self.train:
            ang = np.random.uniform(-7, 7)
            sc = np.random.uniform(0.95, 1.10)
            M = cv2.getRotationMatrix2D((CACHE_SIZE / 2, CACHE_SIZE / 2), ang, sc)
            img = cv2.warpAffine(img, M, (CACHE_SIZE, CACHE_SIZE),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            y0 = np.random.randint(0, CACHE_SIZE - CROP + 1)
            x0 = np.random.randint(0, CACHE_SIZE - CROP + 1)
            img = img[y0:y0 + CROP, x0:x0 + CROP]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
        else:
            o = (CACHE_SIZE - CROP) // 2
            img = img[o:o + CROP, o:o + CROP]
        return torch.from_numpy(np.ascontiguousarray(img)), \
            torch.from_numpy(self.labels[i].astype(np.float32))


def to_input(u8, device):
    """uint8 [B,H,W] -> normalised float [B,3,H,W] on device."""
    x = u8.to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)
    x = x.expand(-1, 3, -1, -1)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


class DenseNet121Multi(nn.Module):
    """DenseNet121 with a 14-way multi-label head and a dropout layer kept
    active at inference for MC-dropout uncertainty."""

    def __init__(self, n_out=14, p_drop=0.2, pretrained=True):
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121
        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet121(weights=w)
        self.features = net.features
        self.drop = nn.Dropout(p_drop)
        self.classifier = nn.Linear(net.classifier.in_features, n_out)

    def forward(self, x):
        f = self.features(x)
        f = torch.nn.functional.relu(f, inplace=True)
        f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(self.drop(f))

    def enable_mc_dropout(self):
        self.eval()
        self.drop.train()
