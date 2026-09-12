"""Ingest a model's raw per-clip score .npy dumps -> shared harness metrics.

Both COSKAD (eval_COSKAD patch) and MoCoDAD (mocodad.post_processing patch) write, per
test/val clip, {scene}_{clip}.npy = per-frame anomaly score, higher = more anomalous,
PRE-smoothing, already averaged over the model's test-time-augmentation transforms and
max-over-persons. So no sign flip, no extra aggregation -- the shared harness only adds
the sigma grid + metrics, identically for every model.

Usage:
  ingest_raw_dump.py <MODEL> <val_raw_dir> <test_raw_dir> [SEED=0]
"""
import sys, os
import numpy as np
from io_poselift import load_split
import score_run

CV = os.path.expanduser('~/poselift-study/data_converted')
SPL = os.path.expanduser('~/poselift-study/results/splits')

def build(split_json, raw_dir, gt_dir):
    pv = {}
    for d in load_split(split_json):
        v = d['video']; s, c = v.split('_')
        f = f'{raw_dir}/{int(s)}_{int(c)}.npy'
        y = np.load(f'{gt_dir}/{v}.npy').astype(int)
        if not os.path.exists(f):
            # clip had no scored segments at all -> neutral (all-zero) anomaly score
            sc = np.zeros(len(y), float)
        else:
            sc = np.load(f).astype(float)
        # align to GT length
        if len(sc) > len(y):
            sc = sc[:len(y)]
        elif len(sc) < len(y):
            sc = np.concatenate([sc, np.full(len(y) - len(sc), sc[-1] if len(sc) else 0.0)])
        pv[v] = dict(n_frames=len(y), labels=y, frame_scores=sc)
    return pv

if __name__ == '__main__':
    model, val_raw, test_raw = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    pv_val = build(f'{SPL}/split_b_val.json',  val_raw,  f'{CV}/b_val/gt')
    pv_test = build(f'{SPL}/split_b_test.json', test_raw, f'{CV}/b_test/gt')
    miss_v = sum(1 for v in pv_val.values() if not np.any(v['frame_scores']))
    miss_t = sum(1 for v in pv_test.values() if not np.any(v['frame_scores']))
    print(f'{model} seed {seed} ingest: val {len(pv_val)} clips ({miss_v} empty), test {len(pv_test)} clips ({miss_t} empty)')
    score_run.score_all(model, pv_val, pv_test, seed=seed, split='b', variant='main')
