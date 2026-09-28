"""Filesystem layout shared by every script.

All data, checkpoints, results and logs live under one working root. It is
taken from the environment variable ``USCNET_ROOT`` and defaults to the
repository directory itself, so a fresh clone works once ``data/`` (and, to
reuse the released artefacts, ``ckpt/`` and ``results/``) are placed next to
``src/``.
"""
import os

ROOT = os.environ.get(
    "USCNET_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(ROOT, "data")
PROC = os.path.join(DATA, "proc")
CKPT = os.path.join(ROOT, "ckpt")
RESULTS = os.path.join(ROOT, "results")
PREDS = os.path.join(RESULTS, "preds")
LOGS = os.path.join(ROOT, "logs")
