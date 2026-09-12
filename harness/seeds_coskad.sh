#!/bin/bash
# COSKAD (Euclidean) seeds 1-3 on split (b). Train in parallel (GPUs 0,1,2), then evals + ingest.
set -e
PS=~/poselift-study
COS=$PS/external/COSKAD
H=$PS/harness
MR=$PS/results/model_runs/coskad
CKROOT=$MR/HR-STC

# --- per-seed configs from the seed-0 config ---
~/.conda/envs/poselift-harness/bin/python - <<'PY'
import yaml, os
base = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/coskad_b_s0.yaml')))
for s in (1, 2, 3):
    c = dict(base); c['seed'] = s; c['dir_name'] = f'eucl_b_s{s}'; c['split'] = 'train'
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/coskad_b_s{s}.yaml'), 'w'))
    e = dict(base); e['seed'] = s; e['dir_name'] = f'eucl_b_s{s}'; e['split'] = 'test'; e['load_ckpt'] = 'LAST'
    yaml.safe_dump(e, open(os.path.expanduser(f'~/poselift-study/configs/coskad_b_s{s}_eval.yaml'), 'w'))
print('wrote coskad seed 1-3 configs')
PY

# --- ensure split-b TEST is staged ---
~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

# --- train seeds 1,2,3 in parallel on GPUs 0,1,2 ---
declare -A GPU=( [1]=0 [2]=1 [3]=2 )
for S in 1 2 3; do
  rm -rf $CKROOT/eucl_b_s$S
  nohup env CUDA_VISIBLE_DEVICES=${GPU[$S]} ~/.conda/envs/coskad/bin/python \
    "$COS/train_COSKAD.py" -c $PS/configs/coskad_b_s$S.yaml > $MR/train_b_s$S.log 2>&1 &
  echo "COSKAD seed $S training on GPU ${GPU[$S]} (PID $!)"
done
wait
echo "all COSKAD seed trainings finished."

cd "$COS"
# --- TEST eval for each seed (test still staged) ---
for S in 1 2 3; do
  CK=$(ls -t $CKROOT/eucl_b_s$S/*.ckpt | head -1); CKN=$(basename "$CK")
  ~/.conda/envs/poselift-harness/bin/python - "$S" "$CKN" <<'PY'
import yaml, os, sys
s, ck = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(os.path.expanduser(f'~/poselift-study/configs/coskad_b_s{s}_eval.yaml')))
c['load_ckpt'] = ck
yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/coskad_b_s{s}_eval.yaml'), 'w'))
PY
  rm -rf $MR/raw_test_s$S && mkdir -p $MR/raw_test_s$S
  COSKAD_RAW_DUMP=$MR/raw_test_s$S CUDA_VISIBLE_DEVICES=0 \
    ~/.conda/envs/coskad/bin/python eval_COSKAD.py -c $PS/configs/coskad_b_s${S}_eval.yaml \
    > $MR/eval_test_b_s$S.log 2>&1 || true
  echo "seed $S test dumps: $(ls $MR/raw_test_s$S | wc -l)"
done

# --- restage VAL, val eval for each seed ---
~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
for S in 1 2 3; do
  rm -rf $MR/raw_val_s$S && mkdir -p $MR/raw_val_s$S
  COSKAD_RAW_DUMP=$MR/raw_val_s$S CUDA_VISIBLE_DEVICES=0 \
    ~/.conda/envs/coskad/bin/python eval_COSKAD.py -c $PS/configs/coskad_b_s${S}_eval.yaml \
    > $MR/eval_val_b_s$S.log 2>&1 || true
  echo "seed $S val dumps: $(ls $MR/raw_val_s$S | wc -l)"
done
~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

# --- ingest each seed ---
cd "$H"
for S in 1 2 3; do
  ~/.conda/envs/poselift-harness/bin/python ingest_raw_dump.py COSKAD $MR/raw_val_s$S $MR/raw_test_s$S $S
done
echo "COSKAD seeds 1-3 complete."
