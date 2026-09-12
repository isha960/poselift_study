#!/bin/bash
# COSKAD split-b eval: test pass + val pass (same checkpoint), then ingest.
set -e
PS=~/poselift-study
COS=$PS/external/COSKAD
CKDIR=$PS/results/model_runs/coskad/HR-STC/eucl_b_s0
GPU=${1:-1}

CKPT=$(ls -t "$CKDIR"/*.ckpt | head -1)
CKN=$(basename "$CKPT")
echo "COSKAD eval using checkpoint: $CKN  (GPU $GPU)"

# --- point eval config at that ckpt ---
~/.conda/envs/poselift-harness/bin/python - "$CKN" <<'PY'
import yaml, os, sys
c = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/coskad_b_s0_eval.yaml')))
c['load_ckpt'] = sys.argv[1]
yaml.safe_dump(c, open(os.path.expanduser('~/poselift-study/configs/coskad_b_s0_eval.yaml'), 'w'))
print('eval cfg load_ckpt =', sys.argv[1])
PY

cd "$COS"
# --- TEST pass (data_coskad/b currently staged with testing = split_b_test) ---
rm -rf $PS/results/model_runs/coskad/raw_test && mkdir -p $PS/results/model_runs/coskad/raw_test
COSKAD_RAW_DUMP=$PS/results/model_runs/coskad/raw_test \
CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/coskad/bin/python eval_COSKAD.py \
  -c $PS/configs/coskad_b_s0_eval.yaml > $PS/results/model_runs/coskad/eval_test_b_s0.log 2>&1
echo "test eval exit $?  raw_test dumps: $(ls $PS/results/model_runs/coskad/raw_test | wc -l)"

# --- VAL pass: restage testing=split_b_val, eval, then restore testing=split_b_test ---
~/.conda/envs/poselift-harness/bin/python $PS/harness/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
rm -rf $PS/results/model_runs/coskad/raw_val && mkdir -p $PS/results/model_runs/coskad/raw_val
COSKAD_RAW_DUMP=$PS/results/model_runs/coskad/raw_val \
CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/coskad/bin/python eval_COSKAD.py \
  -c $PS/configs/coskad_b_s0_eval.yaml > $PS/results/model_runs/coskad/eval_val_b_s0.log 2>&1
echo "val eval exit $?  raw_val dumps: $(ls $PS/results/model_runs/coskad/raw_val | wc -l)"
~/.conda/envs/poselift-harness/bin/python $PS/harness/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

# --- ingest through shared harness ---
cd $PS/harness
~/.conda/envs/poselift-harness/bin/python ingest_raw_dump.py COSKAD \
  $PS/results/model_runs/coskad/raw_val $PS/results/model_runs/coskad/raw_test
