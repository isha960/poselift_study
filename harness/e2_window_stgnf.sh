#!/bin/bash
# E2-window, STG-NF: retrain at seg_len=T, seg_stride=T/2, 4 seeds serial on one GPU (fast).
# Usage: e2_window_stgnf.sh <T> <stride> [gpu]
set -e
T=$1; ST=$2; GPU=${3:-0}
PS=~/poselift-study; STG=~/PoseLift/STG-NF; H=$PS/harness
MR=$PS/results/model_runs/stgnf/e2win; mkdir -p "$MR"
VAR="T${T}"

~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

for S in 0 1 2 3; do
  echo "[STG-NF $VAR s$S]"
  cd $STG
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --epochs 3 --seg_len $T --seg_stride $ST \
    --batch_size 256 --model_lr 5e-4 --model_optimizer adamx --device cuda:0 \
    --exp_dir $MR/exp_T${T}_s$S > $MR/train_T${T}_s$S.log 2>&1
  mkdir -p $MR/csv_test_T${T}_s$S $MR/csv_val_T${T}_s$S
  cp results/csv_files/*.csv $MR/csv_test_T${T}_s$S/ 2>/dev/null || true
  CK=$(find $MR/exp_T${T}_s$S -name '*checkpoint.pth.tar' | head -1)
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --seg_len $T --seg_stride $ST --device cuda:0 \
    --checkpoint "$CK" --exp_dir $MR/exp_T${T}_s${S}_valpass > $MR/val_T${T}_s$S.log 2>&1 || true
  cp results/csv_files/*.csv $MR/csv_val_T${T}_s$S/ 2>/dev/null || true
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null
  cd $H
  ~/.conda/envs/poselift-harness/bin/python ingest_stgnf_variant.py "$VAR" $S \
    $MR/csv_val_T${T}_s$S $MR/csv_test_T${T}_s$S 2>&1 | grep -E "clips|sig=0"
done
echo "[STG-NF $VAR] done"
