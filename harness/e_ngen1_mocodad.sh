#!/bin/bash
# Additional experiment (dissertation Ch5/E3): MoCoDAD accuracy at n_generated_samples=1,
# vs the paper-default n_gen=50 used for the main comparison. Same 4-seed checkpoints,
# eval-only, no retrain. variant='ngen1'.
# Usage: e_ngen1_mocodad.sh <seed> <gpu>
set -e
S=$1; GPU=$2
PS=~/poselift-study
MO=$PS/external/MoCoDAD
H=$PS/harness
CKDIR=$PS/results/model_runs/mocodad/HR-STC/b_s${S}
MR=$PS/results/model_runs/mocodad/ngen1
mkdir -p "$MR"

CKPT=$(ls -t "$CKDIR"/*.ckpt | head -1); CKN=$(basename "$CKPT")
echo "[MoCoDAD ngen1 s$S] ckpt=$CKN gpu=$GPU"

~/.conda/envs/poselift-harness/bin/python - "$S" "$CKN" <<'PY'
import yaml, os, sys
s, ck = sys.argv[1], sys.argv[2]
for split, suff in [('test', ''), ('validation', '_val')]:
    c = yaml.safe_load(open(os.path.expanduser(f'~/poselift-study/configs/mocodad_b_s{s}_eval.yaml')))
    c['load_ckpt'] = ck; c['split'] = split; c['n_generated_samples'] = 1
    root = c['data_dir']
    c['test_path'] = os.path.join(root, 'validating' if split == 'validation' else 'testing', 'test_frame_mask')
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_ngen1_s{s}_eval{suff}.yaml'), 'w'))
PY

cd "$MO"
RT=$MR/raw_test_s${S}; RV=$MR/raw_val_s${S}
rm -rf "$RT" "$RV"; mkdir -p "$RT" "$RV"
MOCODAD_RAW_DUMP="$RT" CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py \
  -c $PS/configs/mocodad_ngen1_s${S}_eval.yaml > $MR/eval_test_s${S}.log 2>&1 || true
MOCODAD_RAW_DUMP="$RV" CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py \
  -c $PS/configs/mocodad_ngen1_s${S}_eval_val.yaml > $MR/eval_val_s${S}.log 2>&1 || true
echo "[MoCoDAD ngen1 s$S] test dumps=$(ls $RT/*.npy 2>/dev/null | wc -l)  val dumps=$(ls $RV/*.npy 2>/dev/null | wc -l)"

cd "$H"
~/.conda/envs/poselift-harness/bin/python ingest_raw_variant.py MoCoDAD ngen1 $S "$RV" "$RT"
