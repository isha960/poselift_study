#!/bin/bash
# E2-noise, STG-NF: checkpoint-only pass on a perturbed split-(b) test set.
# Usage: e2_noise_stgnf.sh <cond> <seed> <gpu> <pkldir>
#   <pkldir> = dir of perturbed {video}.pkl (test videos only)
set -e
COND=$1; S=$2; GPU=$3; PKLS=$4
PS=~/poselift-study
STG=~/PoseLift/STG-NF
MR=$PS/results/model_runs/stgnf
H=$PS/harness
OUT=$MR/e2noise/${COND}_s${S}
mkdir -p "$OUT"

CK=$(find $MR/exp_b_s${S} -name '*checkpoint.pth.tar' | head -1)
echo "[STG-NF $COND s$S] ckpt=$CK  gpu=$GPU"

# stage: train falls through to REAL pkls (not in override dir), test = perturbed
POSELIFT_PKL_OVERRIDE=$PKLS ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

cd $STG
rm -rf results/csv_files && mkdir -p results/csv_files
CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
  --dataset PoseLift --seed $S --seg_len 24 --seg_stride 6 --device cuda:0 \
  --checkpoint "$CK" --exp_dir $OUT > $OUT/eval.log 2>&1
n=$(ls results/csv_files/*.csv 2>/dev/null | wc -l)
echo "[STG-NF $COND s$S] eval exit $?  csv=$n"
rm -rf $OUT/csv && mkdir -p $OUT/csv && cp results/csv_files/*.csv $OUT/csv/

cd $H
~/.conda/envs/poselift-harness/bin/python noise_ingest.py "STG-NF" "$COND" "$S" "$OUT/csv" stgnf
