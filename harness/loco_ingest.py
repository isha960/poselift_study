"""E1 ingest: one (model, fold, seed) -> shared harness metrics, no val (threshold-free).

Usage: loco_ingest.py <MODEL> <fold 1-6> <seed> <score_dir> <fmt: stgnf|dump>
  stgnf : score_dir has <scene:02d>_<clip:04d>.csv rows (inv_gt, normality); anomaly = -col1
  dump  : score_dir has <scene>_<clip>.npy per-frame anomaly score (higher = anomalous)
Labels come straight from io_poselift.gt_labels (videos with no GT file -> all-normal).
"""
import sys, os, csv
import numpy as np
from io_poselift import load_split, load_video, gt_labels
import score_run

SPL = os.path.expanduser('~/poselift-study/results/splits')

def build(fold, score_dir, fmt):
    pv = {}
    for d in load_split(f'{SPL}/loco_f{fold}_test.json'):
        v = d['video']; s, c = v.split('_')
        n, _ = load_video(v)
        y, n_used, _ = gt_labels(v, n)
        if fmt == 'stgnf':
            p = f'{score_dir}/{int(s):02d}_{int(c):04d}.csv'
            arr = np.array(list(csv.reader(open(p))), float)
            sc = -arr[:, 1]
        else:
            p = f'{score_dir}/{int(s)}_{int(c)}.npy'
            sc = np.load(p).astype(float) if os.path.exists(p) else np.zeros(len(y))
        if len(sc) > len(y):
            sc = sc[:len(y)]
        elif len(sc) < len(y):
            sc = np.concatenate([sc, np.full(len(y) - len(sc), sc[-1] if len(sc) else 0.0)])
        pv[v] = dict(n_frames=len(y), labels=y, frame_scores=sc)
    return pv

if __name__ == '__main__':
    model, fold, seed, score_dir, fmt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    pv = build(fold, score_dir, fmt)
    na = sum(1 for x in pv.values() if x['labels'].any())
    print(f'{model} LOCO fold {fold} seed {seed}: {len(pv)} test clips ({na} anomaly)')
    score_run.score_all(model, {}, pv, seed=seed, split='b', variant='main', fold=f'f{fold}')
