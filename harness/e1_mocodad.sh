#!/bin/bash
# E1 LOCO — MoCoDAD, 6 folds x 4 seeds. 4 seeds parallel per fold (GPUs 0-3). No val.
set -e
PS=~/poselift-study; MO=$PS/external/MoCoDAD; H=$PS/harness
MR=$PS/results/model_runs/mocodad/e1; mkdir -p "$MR"
ROOT=$PS/data_mocodad/e1
CKR=$MR/ckpt

for K in 1 2 3 4 5 6; do
  echo "===== fold $K ====="
  # stage: training/ = fold train ; testing/ = fold test  (reuse converted trajectories + GT)
  ~/.conda/envs/poselift-harness/bin/python - "$K" <<'PY'
import os, shutil, json, sys
k = sys.argv[1]
PS = os.path.expanduser('~/poselift-study')
CV = f'{PS}/data_converted'
root = f'{PS}/data_mocodad/e1'
shutil.rmtree(root, ignore_errors=True)
# map every video -> which converted split holds its trajectory folder + gt
def find_src(v):
    for s in ('b_train', 'b_val', 'b_test'):
        if os.path.isdir(f'{CV}/{s}/trajectories/{v}'):
            return s
    return None
for part, sj in [('training', f'{PS}/results/splits/loco_f{k}_train.json'),
                 ('testing',  f'{PS}/results/splits/loco_f{k}_test.json')]:
    os.makedirs(f'{root}/{part}/trajectories')
    if part == 'testing':
        os.makedirs(f'{root}/{part}/test_frame_mask')
    for d in json.load(open(sj))['videos']:
        v = d['video']; src = find_src(v)
        dash = v.replace('_', '-', 1)
        shutil.copytree(f'{CV}/{src}/trajectories/{v}', f'{root}/{part}/trajectories/{dash}')
        if part == 'testing':
            shutil.copy(f'{CV}/{src}/gt/{v}.npy', f'{root}/{part}/test_frame_mask/{v}.npy')
print(f'fold {k} staged: train {len(os.listdir(root+"/training/trajectories"))} test {len(os.listdir(root+"/testing/trajectories"))}')
PY
  ~/.conda/envs/poselift-harness/bin/python - "$K" <<'PY'
import yaml, os, sys
k = sys.argv[1]
base = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_train.yaml')))
for s in range(4):
    c = dict(base)
    c.update(dict(data_dir=os.path.expanduser('~/poselift-study/data_mocodad/e1'),
                  test_path=os.path.expanduser('~/poselift-study/data_mocodad/e1/testing/test_frame_mask'),
                  seed=s, dir_name=f'e1_f{k}_s{s}', split='train',
                  exp_dir=os.path.expanduser('~/poselift-study/results/model_runs/mocodad/e1/ckpt')))
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_e1_f{k}_s{s}_train.yaml'), 'w'))
PY
  for S in 0 1 2 3; do
    rm -rf $CKR/HR-STC/e1_f${K}_s${S}
    nohup env CUDA_VISIBLE_DEVICES=$S ~/.conda/envs/mocodad/bin/python \
      "$MO/train_MoCoDAD.py" -c $PS/configs/mocodad_e1_f${K}_s${S}_train.yaml > $MR/train_f${K}_s${S}.log 2>&1 &
  done
  wait
  cd "$MO"
  for S in 0 1 2 3; do
    CK=$(ls -t $CKR/HR-STC/e1_f${K}_s${S}/*.ckpt 2>/dev/null | head -1); CKN=$(basename "$CK")
    ~/.conda/envs/poselift-harness/bin/python - "$K" "$S" "$CKN" <<'PY'
import yaml, os, sys
k, s, ck = sys.argv[1], sys.argv[2], sys.argv[3]
c = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_eval.yaml')))
c.update(dict(data_dir=os.path.expanduser('~/poselift-study/data_mocodad/e1'),
              test_path=os.path.expanduser('~/poselift-study/data_mocodad/e1/testing/test_frame_mask'),
              split='test', load_ckpt=ck, dir_name=f'e1_f{k}_s{s}',
              exp_dir=os.path.expanduser('~/poselift-study/results/model_runs/mocodad/e1/ckpt')))
yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_e1_f{k}_s{s}_eval.yaml'), 'w'))
PY
    RD=$MR/raw_f${K}_s${S}; rm -rf "$RD"; mkdir -p "$RD"
    MOCODAD_RAW_DUMP="$RD" CUDA_VISIBLE_DEVICES=0 \
      ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py -c $PS/configs/mocodad_e1_f${K}_s${S}_eval.yaml \
      > $MR/eval_f${K}_s${S}.log 2>&1 || true
    cd "$H"
    ~/.conda/envs/poselift-harness/bin/python loco_ingest.py MoCoDAD $K $S "$RD" dump 2>&1 | grep -E 'LOCO fold|sig=0'
    cd "$MO"
  done
done
echo "E1 MoCoDAD done."
