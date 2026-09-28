source ~/miniconda3/etc/profile.d/conda.sh
conda activate uscnet
ROOT="${USCNET_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"/src
for s in 42 123 3407 7 2024; do
  echo "=== seed $s ==="
  python train.py --seed $s --epochs 15 --bs 128 --workers 40 \
      --out "$ROOT/ckpt"
done
echo TRAIN_ALL_DONE
