#!/bin/bash
# E2-window, COSKAD: retrain at seg_len=T, stride=T/2, 4 seeds parallel on GPUs 0-3.
# Usage: e2_window_coskad.sh <T> <stride>
set -e
T=$1; ST=$2
PS=~/poselift-study; COS=$PS/external/COSKAD; H=$PS/harness
MR=$PS/results/model_runs/coskad/e2win; CKR=$MR/ckpt
mkdir -p "$MR"
VAR="T${T}"

# per-seed train + eval configs from the seed-0 base
~/.conda/envs/poselift-harness/bin/python - "$T" "$ST" <<'PY'
import yaml, os, sys
T, ST = int(sys.argv[1]), int(sys.argv[2])
base = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/coskad_b_s0.yaml')))
for s in range(4):
    c = dict(base)
    c.update(dict(seed=s, dir_name=f'e2w_T{T}_s{s}', split='train',
                  dataset_seg_len=T, dataset_seg_stride=ST,
                  exp_dir=os.path.expanduser('~/poselift-study/results/model_runs/coskad/e2win/ckpt')))
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/coskad_e2w_T{T}_s{s}.yaml'), 'w'))
print(f'wrote coskad E2-window T{T} stride{ST} configs')
PY

# stage split-b TEST (clean data, unchanged)
~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

for S in 0 1 2 3; do
  rm -rf $CKR/HR-STC/e2w_T${T}_s${S}
  nohup env CUDA_VISIBLE_DEVICES=$S ~/.conda/envs/coskad/bin/python \
    "$COS/train_COSKAD.py" -c $PS/configs/coskad_e2w_T${T}_s${S}.yaml > $MR/train_T${T}_s${S}.log 2>&1 &
done
wait
echo "[COSKAD $VAR] trainings done"

cd "$COS"
for S in 0 1 2 3; do
  CK=$(ls -t $CKR/HR-STC/e2w_T${T}_s${S}/*.ckpt 2>/dev/null | head -1); CKN=$(basename "$CK")
  ~/.conda/envs/poselift-harness/bin/python - "$T" "$S" "$CKN" <<'PY'
import yaml, os, sys
T, s, ck = sys.argv[1], sys.argv[2], sys.argv[3]
c = yaml.safe_load(open(os.path.expanduser(f'~/poselift-study/configs/coskad_e2w_T{T}_s{s}.yaml')))
c['split'] = 'test'; c['load_ckpt'] = ck
yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/coskad_e2w_T{T}_s{s}_eval.yaml'), 'w'))
PY
  RT=$MR/raw_test_T${T}_s${S}; RV=$MR/raw_val_T${T}_s${S}
  rm -rf "$RT" "$RV"; mkdir -p "$RT" "$RV"
  COSKAD_RAW_DUMP="$RT" CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/coskad/bin/python eval_COSKAD.py \
    -c $PS/configs/coskad_e2w_T${T}_s${S}_eval.yaml > $MR/eval_test_T${T}_s${S}.log 2>&1 || true
done
# val pass
~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
for S in 0 1 2 3; do
  RV=$MR/raw_val_T${T}_s${S}
  COSKAD_RAW_DUMP="$RV" CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/coskad/bin/python eval_COSKAD.py \
    -c $PS/configs/coskad_e2w_T${T}_s${S}_eval.yaml > $MR/eval_val_T${T}_s${S}.log 2>&1 || true
done
~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $PS/data_coskad/b \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

cd "$H"
for S in 0 1 2 3; do
  ~/.conda/envs/poselift-harness/bin/python ingest_raw_variant.py COSKAD "$VAR" $S \
    $MR/raw_val_T${T}_s${S} $MR/raw_test_T${T}_s${S} 2>&1 | grep -E "seed|sig=0"
done
echo "[COSKAD $VAR] done"
