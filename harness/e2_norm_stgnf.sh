#!/bin/bash
# E2-norm, STG-NF: the only STG-NF normalization knob is on/off (--global_pose_segs makes it
# use UNnormalized pose segments). variant = norm_raw. 4 seeds serial. Native = normalised (main).
# Usage: e2_norm_stgnf.sh [gpu]
set -e
GPU=${1:-0}
PS=~/poselift-study; STG=~/PoseLift/STG-NF; H=$PS/harness
MR=$PS/results/model_runs/stgnf/e2norm; mkdir -p "$MR"
VAR="norm_raw"

~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

for S in 0 1 2 3; do
  echo "[STG-NF $VAR s$S]"
  cd $STG
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --epochs 3 --seg_len 24 --seg_stride 6 --global_pose_segs \
    --batch_size 256 --model_lr 5e-4 --model_optimizer adamx --device cuda:0 \
    --exp_dir $MR/exp_${VAR}_s$S > $MR/train_${VAR}_s$S.log 2>&1 || echo "  train FAILED s$S"
  mkdir -p $MR/csv_test_${VAR}_s$S $MR/csv_val_${VAR}_s$S
  cp results/csv_files/*.csv $MR/csv_test_${VAR}_s$S/ 2>/dev/null || true
  CK=$(find $MR/exp_${VAR}_s$S -name '*checkpoint.pth.tar' | head -1)
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --seg_len 24 --seg_stride 6 --global_pose_segs --device cuda:0 \
    --checkpoint "$CK" --exp_dir $MR/exp_${VAR}_s${S}_valpass > $MR/val_${VAR}_s$S.log 2>&1 || true
  cp results/csv_files/*.csv $MR/csv_val_${VAR}_s$S/ 2>/dev/null || true
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null
  cd $H
  ~/.conda/envs/poselift-harness/bin/python ingest_stgnf_variant.py "$VAR" $S \
    $MR/csv_val_${VAR}_s$S $MR/csv_test_${VAR}_s$S 2>&1 | grep -E "clips|sig=0"
done
echo "[STG-NF $VAR] done"
