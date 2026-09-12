"""Ingest raw per-clip .npy score dumps (COSKAD/MoCoDAD) under an arbitrary variant tag.
Same as ingest_raw_dump.py but variant is a CLI arg (for E2-norm / E2-window). Has a val
pass so fixed-FPR thresholds are calibrated exactly as the main runs.

Usage: ingest_raw_variant.py <MODEL> <variant> <seed> <val_raw_dir> <test_raw_dir>
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
        sc = np.load(f).astype(float) if os.path.exists(f) else np.zeros(len(y), float)
        if len(sc) > len(y):
            sc = sc[:len(y)]
        elif len(sc) < len(y):
            sc = np.concatenate([sc, np.full(len(y) - len(sc), sc[-1] if len(sc) else 0.0)])
        pv[v] = dict(n_frames=len(y), labels=y, frame_scores=sc)
    return pv


if __name__ == '__main__':
    model, variant, seed, val_raw, test_raw = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    pv_val = build(f'{SPL}/split_b_val.json', val_raw, f'{CV}/b_val/gt')
    pv_test = build(f'{SPL}/split_b_test.json', test_raw, f'{CV}/b_test/gt')
    mv = sum(1 for x in pv_val.values() if not np.any(x['frame_scores']))
    mt = sum(1 for x in pv_test.values() if not np.any(x['frame_scores']))
    print(f'{model} {variant} seed {seed}: val {len(pv_val)} ({mv} empty), test {len(pv_test)} ({mt} empty)')
    score_run.score_all(model, pv_val, pv_test, seed=seed, split='b', variant=variant)
