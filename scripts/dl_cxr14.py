import os

ROOT = os.environ.get(
    "USCNET_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="alkzar90/NIH-Chest-X-ray-dataset",
    repo_type="dataset",
    local_dir=os.path.join(ROOT, "data", "cxr14_raw"),
    allow_patterns=["data/images/*.zip","data/*.csv","data/*.txt"],
    max_workers=16,
)
print("DOWNLOAD_DONE", p)
