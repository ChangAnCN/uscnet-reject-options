set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uscnet
ROOT="${USCNET_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"/src

# wait for the 5-seed training run to finish
while ! grep -q TRAIN_ALL_DONE "$ROOT"/logs/train.log; do sleep 20; done
echo "=== training done, starting inference ==="

python infer.py --cohorts val test chexpert_expert --n_mc 20 --bs 768
python infer.py --cohorts train --n_mc 2 --bs 768
python infer.py --cohorts chexpert_ext --n_mc 20 --bs 768
echo "=== inference done ==="

python backbone_eval.py 2>&1 | tee "$ROOT"/logs/backbone.log
python analyze.py 2>&1 | tee "$ROOT"/logs/analyze.log
echo PIPELINE_DONE
