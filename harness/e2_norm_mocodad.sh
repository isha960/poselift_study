#!/bin/bash
# E2-norm, MoCoDAD: retrain under a native normalization strategy, 4 seeds parallel GPUs 0-3.
# Usage: e2_norm_mocodad.sh <strategy>   strategy in {markovitz, stan, bbox}  (robust = main run)
set -e
STRAT=$1
PS=~/poselift-study; MO=$PS/external/MoCoDAD; H=$PS/harness
MR=$PS/results/model_runs/mocodad/e2norm; CKR=$MR/ckpt
mkdir -p "$MR"
VAR="norm_${STRAT}"

~/.conda/envs/poselift-harness/bin/python - "$STRAT" <<'PY'
import yaml, os, sys
strat = sys.argv[1]
base = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_train.yaml')))
for s in range(4):
    c = dict(base)
    c.update(dict(seed=s, dir_name=f'e2n_{strat}_s{s}', split='train',
                  normalization_strategy=strat,
                  exp_dir=os.path.expanduser('~/poselift-study/results/model_runs/mocodad/e2norm/ckpt')))
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_e2n_{strat}_s{s}_train.yaml'), 'w'))
print(f'wrote mocodad E2-norm {strat} train configs')
PY

for S in 0 1 2 3; do
  rm -rf $CKR/HR-STC/e2n_${STRAT}_s${S}
  nohup env CUDA_VISIBLE_DEVICES=$S ~/.conda/envs/mocodad/bin/python \
    "$MO/train_MoCoDAD.py" -c $PS/configs/mocodad_e2n_${STRAT}_s${S}_train.yaml > $MR/train_${STRAT}_s${S}.log 2>&1 &
done
wait
echo "[MoCoDAD $VAR] trainings done"

cd "$MO"
for S in 0 1 2 3; do
  CK=$(ls -t $CKR/HR-STC/e2n_${STRAT}_s${S}/*.ckpt 2>/dev/null | head -1); CKN=$(basename "$CK")
  ~/.conda/envs/poselift-harness/bin/python - "$STRAT" "$S" "$CKN" <<'PY'
import yaml, os, sys
strat, s, ck = sys.argv[1], sys.argv[2], sys.argv[3]
for split, suff in [('test', ''), ('validation', '_val')]:
    c = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_eval.yaml')))
    c['load_ckpt'] = ck; c['split'] = split; c['dir_name'] = f'e2n_{strat}_s{s}'
    c['normalization_strategy'] = strat
    c['exp_dir'] = os.path.expanduser('~/poselift-study/results/model_runs/mocodad/e2norm/ckpt')
    root = c['data_dir']
    c['test_path'] = os.path.join(root, 'validating' if split == 'validation' else 'testing', 'test_frame_mask')
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/mocodad_e2n_{strat}_s{s}_eval{suff}.yaml'), 'w'))
PY
  RT=$MR/raw_test_${STRAT}_s${S}; RV=$MR/raw_val_${STRAT}_s${S}
  rm -rf "$RT" "$RV"; mkdir -p "$RT" "$RV"
  MOCODAD_RAW_DUMP="$RT" CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py \
    -c $PS/configs/mocodad_e2n_${STRAT}_s${S}_eval.yaml > $MR/eval_test_${STRAT}_s${S}.log 2>&1 || true
  MOCODAD_RAW_DUMP="$RV" CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py \
    -c $PS/configs/mocodad_e2n_${STRAT}_s${S}_eval_val.yaml > $MR/eval_val_${STRAT}_s${S}.log 2>&1 || true
done

cd "$H"
for S in 0 1 2 3; do
  ~/.conda/envs/poselift-harness/bin/python ingest_raw_variant.py MoCoDAD "$VAR" $S \
    $MR/raw_val_${STRAT}_s${S} $MR/raw_test_${STRAT}_s${S} 2>&1 | grep -E "seed|sig=0"
done
echo "[MoCoDAD $VAR] done"
