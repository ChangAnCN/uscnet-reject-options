source ~/miniconda3/etc/profile.d/conda.sh
conda activate uscnet
cd /NHNHOME/uscnet/src
for s in 42 123 3407 7 2024; do
  echo "=== seed $s ==="
  python train.py --seed $s --epochs 15 --bs 128 --workers 40 \
      --out /NHNHOME/uscnet/ckpt
done
echo TRAIN_ALL_DONE
