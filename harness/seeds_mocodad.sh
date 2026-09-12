#!/bin/bash
# MoCoDAD seeds 1-3 on split (b). Train in parallel (GPUs given as args, default 1 2 3), then evals + ingest.
set -e
PS=~/poselift-study
MO=$PS/external/MoCoDAD
H=$PS/harness
MR=$PS/results/model_runs/mocodad
CKROOT=$MR/HR-STC
G1=${1:-1}; G2=${2:-2}; G3=${3:-3}

# --- per-seed configs from seed-0 ---
~/.conda/envs/poselift-harness/bin/python - <<'PY'
import yaml, os
base = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_train.yaml')))
for s in (1, 2, 3):
    c = dict(base); c['seed'] = s; c['dir_name'] = f'b_s{s}'; c['split'] = 'train'
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_b_s{s}_train.yaml'), 'w'))
print('wrote mocodad seed 1-3 train configs')
PY

declare -A GPU=( [1]=$G1 [2]=$G2 [3]=$G3 )
for S in 1 2 3; do
  rm -rf $CKROOT/b_s$S
  nohup env CUDA_VISIBLE_DEVICES=${GPU[$S]} ~/.conda/envs/mocodad/bin/python \
    "$MO/train_MoCoDAD.py" -c $PS/configs/mocodad_b_s${S}_train.yaml > $MR/train_b_s${S}.log 2>&1 &
  echo "MoCoDAD seed $S training on GPU ${GPU[$S]} (PID $!)"
done
wait
echo "all MoCoDAD seed trainings finished."

cd "$MO"
for S in 1 2 3; do
  CK=$(ls -t $CKROOT/b_s$S/*.ckpt | head -1); CKN=$(basename "$CK")
  ~/.conda/envs/poselift-harness/bin/python - "$S" "$CKN" <<'PY'
import yaml, os, sys
s, ck = sys.argv[1], sys.argv[2]
for split, suff in [('test', ''), ('validation', '_val')]:
    c = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_eval.yaml')))
    c['load_ckpt'] = ck; c['split'] = split; c['dir_name'] = f'b_s{s}'
    root = c['data_dir']
    c['test_path'] = os.path.join(root, 'validating' if split == 'validation' else 'testing', 'test_frame_mask')
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_b_s{s}_eval{suff}.yaml'), 'w'))
PY
  rm -rf $MR/raw_test_s$S $MR/raw_val_s$S && mkdir -p $MR/raw_test_s$S $MR/raw_val_s$S
  MOCODAD_RAW_DUMP=$MR/raw_test_s$S CUDA_VISIBLE_DEVICES=$G1 \
    ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py -c $PS/configs/mocodad_b_s${S}_eval.yaml \
    > $MR/eval_test_b_s$S.log 2>&1 || true
  MOCODAD_RAW_DUMP=$MR/raw_val_s$S CUDA_VISIBLE_DEVICES=$G1 \
    ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py -c $PS/configs/mocodad_b_s${S}_eval_val.yaml \
    > $MR/eval_val_b_s$S.log 2>&1 || true
  echo "seed $S: test $(ls $MR/raw_test_s$S | wc -l)  val $(ls $MR/raw_val_s$S | wc -l)"
done

cd "$H"
for S in 1 2 3; do
  ~/.conda/envs/poselift-harness/bin/python ingest_raw_dump.py MoCoDAD $MR/raw_val_s$S $MR/raw_test_s$S $S
done
echo "MoCoDAD seeds 1-3 complete."
