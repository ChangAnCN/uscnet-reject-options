source ~/miniconda3/etc/profile.d/conda.sh
conda activate uscnet
cd /NHNHOME/uscnet/src
CK=/NHNHOME/uscnet/ckpt
for tgt in Pneumonia Effusion; do
  for s in 42 123 3407 7 2024; do
    python train_selectivenet.py --target $tgt --seed $s --epochs 6 \
      --loss classbalanced --init_from $CK/densenet121_seed$s.pt \
      --tag "_cb" --out $CK
  done
done
echo CB_ALL_DONE
