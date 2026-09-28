set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n uscnet python=3.11
conda activate uscnet
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn scipy matplotlib seaborn pillow tqdm \
    huggingface_hub[hf_transfer] datasets timm pyarrow opencv-python-headless \
    statsmodels pydicom
python - <<'PY'
import torch, torchvision, sklearn, pandas
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("dev", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(64,64,device='cuda'); print("matmul ok", (x@x).sum().item() is not None)
PY
echo "ENV_SETUP_DONE"
