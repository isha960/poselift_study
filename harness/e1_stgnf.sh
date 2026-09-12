#!/bin/bash
# E1 LOCO — STG-NF, 6 folds x 4 seeds. Serial on one GPU (fast). No val.
set -e
PS=~/poselift-study; STG=~/PoseLift/STG-NF; H=$PS/harness
MR=$PS/results/model_runs/stgnf/e1; mkdir -p "$MR"
GPU=${1:-3}
for K in 1 2 3 4 5 6; do
  ~/.conda/envs/poselift-harness/bin/python $H/stage_stgnf.py \
    $PS/results/splits/loco_f${K}_train.json $PS/results/splits/loco_f${K}_test.json >/dev/null
  for S in 0 1 2 3; do
    cd $STG; rm -rf results/csv_files; mkdir -p results/csv_files
    CUDA_VISIBLE_DEVICES=$GPU ~/.conda/envs/stgnf-poselift/bin/python train_eval.py \
      --dataset PoseLift --seed $S --epochs 3 --seg_len 24 --seg_stride 6 \
      --batch_size 256 --model_lr 5e-4 --model_optimizer adamx --device cuda:0 \
      --exp_dir $MR/f${K}_s${S} > $MR/f${K}_s${S}.log 2>&1
    D=$MR/csv_f${K}_s${S}; rm -rf "$D"; mkdir -p "$D"; cp results/csv_files/*.csv "$D"/
    cd $H
    ~/.conda/envs/poselift-harness/bin/python loco_ingest.py STG-NF $K $S "$D" stgnf 2>&1 | grep -E 'LOCO fold|sig=0'
  done
done
echo "E1 STG-NF done."
