import os

ROOT = os.environ.get(
    "USCNET_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="danjacobellis/chexpert", repo_type="dataset",
    local_dir=os.path.join(ROOT, "data", "chexpert_raw"), max_workers=12)
print("CHEXPERT_FULL_DONE", p)
