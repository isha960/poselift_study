"""Ingest STG-NF per-frame CSVs -> shared harness metrics.  seed-parameterised.

STG-NF working copy writes results/csv_files/<scene:02d>_<clip:04d>.csv per test clip,
BEFORE its internal smoothing, rows = (inv_gt, normality_score):
  col0 = 1 - true_label      (STG-NF: 1 = normal)
  col1 = min-over-persons normality score, higher = MORE NORMAL; gap frames = clip max
  -> shared protocol anomaly_score = -col1
Labels taken from our converted GT; length asserted equal.

Usage: run_stgnf_ingest.py <SEED> <val_csv_dir> <test_csv_dir>
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
        arr = np.array(list(csv.reader(open(f'{csv_dir}/{fid(v)}.csv'))), float)
        y = np.load(f'{gt_dir}/{v}.npy').astype(int)
        assert len(arr) == len(y), f'{v}: csv {len(arr)} vs gt {len(y)}'
        pv[v] = dict(n_frames=len(y), labels=y, frame_scores=-arr[:, 1])
    return pv

if __name__ == '__main__':
    seed = int(sys.argv[1]); val_dir, test_dir = sys.argv[2], sys.argv[3]
    pv_val = build(f'{SPL}/split_b_val.json',  val_dir,  f'{CV}/b_val/gt')
    pv_test = build(f'{SPL}/split_b_test.json', test_dir, f'{CV}/b_test/gt')
    print(f'STG-NF seed {seed} ingest: val {len(pv_val)} clips, test {len(pv_test)} clips')
    score_run.score_all('STG-NF', pv_val, pv_test, seed=seed, split='b', variant='main')
