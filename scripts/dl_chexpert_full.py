from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="danjacobellis/chexpert", repo_type="dataset",
    local_dir="/NHNHOME/uscnet/data/chexpert_raw", max_workers=12)
print("CHEXPERT_FULL_DONE", p)
