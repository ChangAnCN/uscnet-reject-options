source ~/miniconda3/etc/profile.d/conda.sh
conda activate uscnet
ROOT="${USCNET_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"/src
CK="$ROOT/ckpt"
# weighted variant (matched to the balanced risk we report) for both targets
for tgt in Pneumonia Effusion; do
  for s in 42 123 3407 7 2024; do
    python train_selectivenet.py --target $tgt --seed $s --epochs 6 \
      --init_from $CK/densenet121_seed$s.pt --weighted 1 --tag "" --out $CK
  done
done
# unweighted variant (faithful to the original objective) on the primary target
for s in 42 123 3407 7 2024; do
  python train_selectivenet.py --target Pneumonia --seed $s --epochs 6 \
    --init_from $CK/densenet121_seed$s.pt --weighted 0 --tag "_unw" --out $CK
done
echo SELECTIVENET_ALL_DONE
