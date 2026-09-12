#!/bin/bash
# STG-NF seeds 1-3 on split (b): train+test, val pass, ingest. Serial on one GPU (fast, ~2min/seed).
set -e
PS=~/poselift-study
STG=~/PoseLift/STG-NF
MR=$PS/results/model_runs/stgnf
GPU=${1:-3}
H=~/poselift-study/harness

# make sure split-b test staging is in place
~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
  $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null

for S in 1 2 3; do
  echo "===== STG-NF seed $S ====="
  cd $STG
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --epochs 3 --seg_len 24 --seg_stride 6 \
    --batch_size 256 --model_lr 5e-4 --model_optimizer adamx --device cuda:0 \
    --exp_dir $MR/exp_b_s$S > $MR/train_b_s$S.log 2>&1
  mkdir -p $MR/csv_test_b_s$S $MR/csv_val_b_s$S
  cp results/csv_files/*.csv $MR/csv_test_b_s$S/
  CK=$(find $MR/exp_b_s$S -name '*checkpoint.pth.tar' | head -1)
  # val pass
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_val.json >/dev/null
  rm -rf results/csv_files && mkdir -p results/csv_files
  CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
    --dataset PoseLift --seed $S --seg_len 24 --seg_stride 6 --device cuda:0 \
    --checkpoint "$CK" --exp_dir $MR/exp_b_s${S}_valpass > $MR/val_b_s$S.log 2>&1 || true
  cp results/csv_files/*.csv $MR/csv_val_b_s$S/
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/split_b_train.json $PS/results/splits/split_b_test.json >/dev/null
  # ingest
  cd $H
  ~/.conda/envs/poselift-harness/bin/python run_stgnf_ingest.py $S $MR/csv_val_b_s$S $MR/csv_test_b_s$S
done
echo "STG-NF seeds 1-3 done."
