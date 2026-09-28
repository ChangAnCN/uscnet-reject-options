import os

ROOT = os.environ.get(
    "USCNET_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="danjacobellis/chexpert", repo_type="dataset",
    local_dir=os.path.join(ROOT, "data", "chexpert_raw"),
    allow_patterns=["data/validation-*","README.md"], max_workers=8)
print("CHEXPERT_VAL_DONE", p)
