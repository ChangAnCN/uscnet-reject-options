# Revision experiments (2026-09-14):
#  (a) SelectiveNet (unweighted) on Effusion -- the control was only run on
#      Pneumonia; (b) CB-SelectiveNet with lambda/2, i.e. the constraint written
#      as lambda * 1/2 sum_k, which matches the *magnitude* of the original
#      single-class penalty when both classes are under coverage.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uscnet
ROOT="${USCNET_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"/src
CK="$ROOT/ckpt"
STREAM=$1
if [ "$STREAM" = A ]; then
  for s in 42 123 3407 7 2024; do
    python train_selectivenet.py --target Effusion --seed $s --epochs 6 \
      --init_from $CK/densenet121_seed$s.pt --weighted 0 --tag "_unw" --out $CK --workers 20
  done
  for s in 42 123 3407 7 2024; do
    python train_selectivenet.py --target Pneumonia --seed $s --epochs 6 \
      --loss classbalanced --lam 16 --init_from $CK/densenet121_seed$s.pt \
      --tag "_cbhalf" --out $CK --workers 20
  done
  echo STREAM_A_DONE
else
  for s in 42 123 3407 7 2024; do
    python train_selectivenet.py --target Effusion --seed $s --epochs 6 \
      --loss classbalanced --lam 16 --init_from $CK/densenet121_seed$s.pt \
      --tag "_cbhalf" --out $CK --workers 20
  done
  echo STREAM_B_DONE
fi
