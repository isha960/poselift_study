#!/bin/bash
# One-off addendum (dissertation Ch5 sec.5.5/5.9/6.5): STG-NF at MoCoDAD's native window
# (T=6, stride=3) to test whether MoCoDAD's advantage survives at a shared window length.
# STG-NF's default temporal_kernel_size = T//2+1 is EVEN at T=6 (=4), which fails the
# architecture's odd-kernel assertion (models/STG_NF/stgcn.py: assert kernel_size[0]%2==1).
# This is a genuine structural limit, not a bug: --temporal_kernel 3 (smallest valid odd
# kernel <= T) is used, and this deviation is recorded in RUNLOG/results.md, not hidden.
set -e
GPU=${1:-0}
PS=~/poselift-study; STG=~/PoseLift/STG-NF; H=$PS/harness
MR=$PS/results/model_runs/stgnf/e2win; mkdir -p "$MR"
VAR="T6"

~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

for S in 0 1 2 3; do
  echo "[STG-NF $VAR s$S] (temporal_kernel=3)"
  cd $STG
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --epochs 3 --seg_len 6 --seg_stride 3 --temporal_kernel 3 \
    --batch_size 256 --model_lr 5e-4 --model_optimizer adamx --device cuda:0 \
    --exp_dir $MR/exp_T6_s$S > $MR/train_T6_s$S.log 2>&1
  mkdir -p $MR/csv_test_T6_s$S $MR/csv_val_T6_s$S
  cp results/csv_files/*.csv $MR/csv_test_T6_s$S/ 2>/dev/null || true
  CK=$(find $MR/exp_T6_s$S -name '*checkpoint.pth.tar' | head -1)
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --seg_len 6 --seg_stride 3 --temporal_kernel 3 --device cuda:0 \
    --checkpoint "$CK" --exp_dir $MR/exp_T6_${S}_valpass > $MR/val_T6_s$S.log 2>&1 || true
  cp results/csv_files/*.csv $MR/csv_val_T6_s$S/ 2>/dev/null || true
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null
  cd $H
  ~/.conda/envs/poselift-harness/bin/python ingest_stgnf_variant.py "$VAR" $S \
    $MR/csv_val_T6_s$S $MR/csv_test_T6_s$S 2>&1 | grep -E "clips|sig=0"
done
echo "[STG-NF $VAR] done"
