"""STG-NF per-frame CSV -> shared harness metrics, under an arbitrary variant tag (E2).
Same as run_stgnf_ingest.py but variant is a CLI arg. anomaly_score = -col1.

Usage: ingest_stgnf_variant.py <variant> <seed> <val_csv_dir> <test_csv_dir>
"""
import sys, os, csv
import numpy as np
from io_poselift import load_split
import score_run

CV = os.path.expanduser('~/poselift-study/data_converted')
SPL = os.path.expanduser('~/poselift-study/results/splits')


def fid(v):
    s, c = v.split('_'); return f'{int(s):02d}_{int(c):04d}'


def build(split_json, csv_dir, gt_dir):
    pv = {}
    for d in load_split(split_json):
        v = d['video']
        p = f'{csv_dir}/{fid(v)}.csv'
        y = np.load(f'{gt_dir}/{v}.npy').astype(int)
        if os.path.exists(p):
            arr = np.array(list(csv.reader(open(p))), float)
            sc = -arr[:, 1]
            if len(sc) > len(y):
                sc = sc[:len(y)]
            elif len(sc) < len(y):
                sc = np.concatenate([sc, np.full(len(y) - len(sc), sc[-1] if len(sc) else 0.0)])
        else:
            sc = np.zeros(len(y), float)
        pv[v] = dict(n_frames=len(y), labels=y, frame_scores=sc)
    return pv


if __name__ == '__main__':
    variant, seed, val_dir, test_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    pv_val = build(f'{SPL}/split_b_val.json', val_dir, f'{CV}/b_val/gt')
    pv_test = build(f'{SPL}/split_b_test.json', test_dir, f'{CV}/b_test/gt')
    print(f'STG-NF {variant} seed {seed}: val {len(pv_val)} clips, test {len(pv_test)} clips')
    score_run.score_all('STG-NF', pv_val, pv_test, seed=seed, split='b', variant=variant)
