#!/bin/bash
# MoCoDAD split-b eval: test pass + val pass (same checkpoint), then ingest.
# MoCoDAD's data_dir has training/testing/validating subfolders already staged, so the
# test vs val pass is selected by --split (test | validation), no restaging needed.
set -e
PS=~/poselift-study
MO=$PS/external/MoCoDAD
CKDIR=$PS/results/model_runs/mocodad/HR-STC/b_s0
GPU=${1:-3}

CKPT=$(ls -t "$CKDIR"/*.ckpt | head -1); CKN=$(basename "$CKPT")
echo "MoCoDAD eval checkpoint: $CKN  (GPU $GPU)"

# test-pass config
~/.conda/envs/poselift-harness/bin/python - "$CKN" <<'PY'
import yaml, os, sys
for split, out in [('test','mocodad_b_s0_eval.yaml'), ('validation','mocodad_b_s0_eval_val.yaml')]:
    c = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_eval.yaml')))
    c['load_ckpt'] = sys.argv[1]; c['split'] = split
    # validation split reads gt from {data_dir}/validating/test_frame_mask via init_args when validation:True;
    # but we keep validation:False, so point test_path at the right mask dir per pass:
    root = c['data_dir']
    c['test_path'] = os.path.join(root, 'validating' if split=='validation' else 'testing', 'test_frame_mask')
    yaml.safe_dump(c, open(os.path.expanduser(f'~/poselift-study/configs/{out}'), 'w'))
print('wrote test + val eval configs, ckpt', sys.argv[1])
PY

cd "$MO"
rm -rf $PS/results/model_runs/mocodad/raw_test $PS/results/model_runs/mocodad/raw_val
mkdir -p $PS/results/model_runs/mocodad/raw_test $PS/results/model_runs/mocodad/raw_val

MOCODAD_RAW_DUMP=$PS/results/model_runs/mocodad/raw_test CUDA_VISIBLE_DEVICES=$GPU \
  ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py -c $PS/configs/mocodad_b_s0_eval.yaml \
  > $PS/results/model_runs/mocodad/eval_test_b_s0.log 2>&1
echo "test eval exit $?  raw_test: $(ls $PS/results/model_runs/mocodad/raw_test | wc -l)"

MOCODAD_RAW_DUMP=$PS/results/model_runs/mocodad/raw_val CUDA_VISIBLE_DEVICES=$GPU \
  ~/.conda/envs/mocodad/bin/python eval_MoCoDAD.py -c $PS/configs/mocodad_b_s0_eval_val.yaml \
  > $PS/results/model_runs/mocodad/eval_val_b_s0.log 2>&1
echo "val eval exit $?  raw_val: $(ls $PS/results/model_runs/mocodad/raw_val | wc -l)"

cd $PS/harness
~/.conda/envs/poselift-harness/bin/python ingest_raw_dump.py MoCoDAD \
  $PS/results/model_runs/mocodad/raw_val $PS/results/model_runs/mocodad/raw_test
