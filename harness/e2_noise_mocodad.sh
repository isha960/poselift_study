#!/bin/bash
# E2-noise, MoCoDAD: checkpoint-only eval on a perturbed split-(b) test set.
# Per-seed data root so the 4 seeds can run in parallel on GPUs 0-3.
# Usage: e2_noise_mocodad.sh <cond> <seed> <gpu> <convroot> <dataroot>
#   <convroot> has b_test/{trajectories,gt}  (perturbed)   <dataroot> = staging target
set -e
COND=$1; S=$2; GPU=$3; CONV=$4; ROOT=$5
PS=~/poselift-study
MO=$PS/external/MoCoDAD
H=$PS/harness
CKDIR=$PS/results/model_runs/mocodad/HR-STC/b_s${S}
DUMP=$PS/results/model_runs/mocodad/e2noise/${COND}_s${S}
CFG=$PS/configs/mocodad_e2n_s${S}_eval.yaml
rm -rf "$DUMP" && mkdir -p "$DUMP"

CKPT=$(ls -t "$CKDIR"/*.ckpt | head -1); CKN=$(basename "$CKPT")
echo "[MoCoDAD $COND s$S] ckpt=$CKN gpu=$GPU root=$ROOT"

# per-seed data root: symlink clean training/validating, real perturbed testing/
rm -rf "$ROOT"; mkdir -p "$ROOT"
ln -s $PS/data_mocodad/b/training   "$ROOT/training"
ln -s $PS/data_mocodad/b/validating "$ROOT/validating"
~/.conda/envs/poselift-harness/bin/python $H/stage_mocodad_test.py "$ROOT" "$CONV"

~/.conda/envs/poselift-harness/bin/python - "$S" "$CKN" "$ROOT" "$CFG" <<'PY'
import yaml, os, sys
s, ck, root, out = sys.argv[1:5]
c = yaml.safe_load(open(os.path.expanduser(f'~/poselift-study/configs/mocodad_b_s{s}_eval.yaml')))
c['load_ckpt'] = ck; c['split'] = 'test'
c['data_dir'] = root
c['test_path'] = os.path.join(root, 'testing', 'test_frame_mask')
yaml.safe_dump(c, open(os.path.expanduser(out), 'w'))
PY

cd $MO
MOCODAD_RAW_DUMP=$DUMP CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py \
  -c $CFG > $DUMP/eval.log 2>&1 || true
echo "[MoCoDAD $COND s$S] dumps=$(ls $DUMP/*.npy 2>/dev/null | wc -l)"

cd $H
~/.conda/envs/poselift-harness/bin/python noise_ingest.py "MoCoDAD" "$COND" "$S" "$DUMP" dump
rm -rf "$ROOT"
