from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="danjacobellis/chexpert", repo_type="dataset",
    local_dir="/NHNHOME/uscnet/data/chexpert_raw",
    allow_patterns=["data/validation-*","README.md"], max_workers=8)
print("CHEXPERT_VAL_DONE", p)
