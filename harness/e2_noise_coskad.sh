#!/bin/bash
# E2-noise, COSKAD: checkpoint-only eval on a perturbed split-(b) test set.
# Per-seed data root so the 4 seeds can run in parallel on GPUs 0-3.
# Usage: e2_noise_coskad.sh <cond> <seed> <gpu> <pkldir> <dataroot>
set -e
COND=$1; S=$2; GPU=$3; PKLS=$4; ROOT=$5
PS=~/poselift-study
COS=$PS/external/COSKAD
H=$PS/harness
CKDIR=$PS/results/model_runs/coskad/HR-STC/eucl_b_s${S}
DUMP=$PS/results/model_runs/coskad/e2noise/${COND}_s${S}
CFG=$PS/configs/coskad_e2n_s${S}_eval.yaml
rm -rf "$DUMP" && mkdir -p "$DUMP"

CKPT=$(ls -t "$CKDIR"/*.ckpt | head -1); CKN=$(basename "$CKPT")
echo "[COSKAD $COND s$S] ckpt=$CKN gpu=$GPU root=$ROOT"

# per-seed eval config: base = that seed's 4-seed eval cfg, data_dir -> per-seed root
~/.conda/envs/poselift-harness/bin/python - "$S" "$CKN" "$ROOT" "$CFG" <<'PY'
import yaml, os, sys
s, ck, root, out = sys.argv[1:5]
c = yaml.safe_load(open(os.path.expanduser(f'~/poselift-study/configs/coskad_b_s{s}_eval.yaml')))
c['load_ckpt'] = ck; c['split'] = 'test'
c['data_dir'] = root
c['dataset_path_to_robust'] = root
c['test_path'] = os.path.join(root, 'testing', 'test_frame_mask')
yaml.safe_dump(c, open(os.path.expanduser(out), 'w'))
PY

# stage this seed's perturbed test into its own root (train real via fall-through)
POSELIFT_PKL_OVERRIDE=$PKLS ~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py \
  "$ROOT" $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

cd $COS
COSKAD_RAW_DUMP=$DUMP CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/coskad/bin/python eval_COSKAD.py \
  -c $CFG > $DUMP/eval.log 2>&1 || true
echo "[COSKAD $COND s$S] dumps=$(ls $DUMP/*.npy 2>/dev/null | wc -l)"

cd $H
~/.conda/envs/poselift-harness/bin/python noise_ingest.py "COSKAD" "$COND" "$S" "$DUMP" dump
