#!/bin/bash
# E1 LOCO — COSKAD (Euclidean), 6 folds x 4 seeds. 4 seeds run in parallel per fold (GPUs 0-3). No val.
set -e
PS=~/poselift-study; COS=$PS/external/COSKAD; H=$PS/harness
MR=$PS/results/model_runs/coskad/e1; mkdir -p "$MR"
ROOT=$PS/data_coskad/e1
CKR=$MR/ckpt

for K in 1 2 3 4 5 6; do
  echo "===== fold $K ====="
  ~/.conda/envs/poselift-harness/bin/python $H/stage_coskad.py $ROOT \
    $PS/results/splits/loco_f${K}_train.json $PS/results/splits/loco_f${K}_test.json >/dev/null
  # per-(fold,seed) train configs
  ~/.conda/envs/poselift-harness/bin/python - "$K" <<'PY'
import yaml, os, sys
k = sys.argv[1]
base = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/coskad_b_s0.yaml')))
for s in range(4):
    c = dict(base)
    c.update(dict(data_dir=os.path.expanduser('~/poselift-study/data_coskad/e1'),
                  test_path=os.path.expanduser('~/poselift-study/data_coskad/e1/testing/test_frame_mask'),
                  dataset_path_to_robust=os.path.expanduser('~/poselift-study/data_coskad/e1'),
                  seed=s, dir_name=f'e1_f{k}_s{s}', split='train',
                  exp_dir=os.path.expanduser('~/poselift-study/results/model_runs/coskad/e1/ckpt')))
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/coskad_e1_f{k}_s{s}.yaml'), 'w'))
PY
  # train 4 seeds in parallel
  for S in 0 1 2 3; do
    rm -rf $CKR/HR-STC/e1_f${K}_s${S}
    nohup env CUDA_VISIBLE_DEVICES=$S ~/.conda/envs/coskad/bin/python \
      "$COS/train_COSKAD.py" -c $PS/configs/coskad_e1_f${K}_s${S}.yaml > $MR/train_f${K}_s${S}.log 2>&1 &
  done
  wait
  # eval + ingest each seed (test set already staged for this fold)
  cd "$COS"
  for S in 0 1 2 3; do
    CK=$(ls -t $CKR/HR-STC/e1_f${K}_s${S}/*.ckpt 2>/dev/null | head -1); CKN=$(basename "$CK")
    ~/.conda/envs/poselift-harness/bin/python - "$K" "$S" "$CKN" <<'PY'
import yaml, os, sys
k, s, ck = sys.argv[1], sys.argv[2], sys.argv[3]
c = yaml.safe_load(open(os.path.expanduser(f'~/poselift-study/configs/coskad_e1_f{k}_s{s}.yaml')))
c['split'] = 'test'; c['load_ckpt'] = ck
yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/coskad_e1_f{k}_s{s}_eval.yaml'), 'w'))
PY
    RD=$MR/raw_f${K}_s${S}; rm -rf "$RD"; mkdir -p "$RD"
    COSKAD_RAW_DUMP="$RD" CUDA_VISIBLE_DEVICES=0 \
      ~/.conda/envs/coskad/bin/python eval_COSKAD.py -c $PS/configs/coskad_e1_f${K}_s${S}_eval.yaml \
      > $MR/eval_f${K}_s${S}.log 2>&1 || true
    cd "$H"
    ~/.conda/envs/poselift-harness/bin/python loco_ingest.py COSKAD $K $S "$RD" dump 2>&1 | grep -E 'LOCO fold|sig=0'
    cd "$COS"
  done
done
echo "E1 COSKAD done."
