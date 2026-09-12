"""E2-noise ingest: one (model, condition, seed) -> shared harness metrics.

No val pass (perturbation is eval-time only; primary signal is threshold-free frame
ROC-AUC / PR-AUC / EER / event-ROC-AUC on the degraded split-(b) test set, exactly as E1).
Rows land in raw/metrics.csv with variant=<cond>, fold='none', seed=<trained-model seed>.

Usage: noise_ingest.py <MODEL> <cond> <seed> <score_dir> <fmt: stgnf|dump>
  stgnf : score_dir has <scene:02d>_<clip:04d>.csv rows (inv_gt, normality); anomaly = -col1
  dump  : score_dir has <scene>_<clip>.npy per-frame anomaly score (higher = anomalous)
"""
import sys, os, csv
import numpy as np
from io_poselift import load_split, load_video, gt_labels
import score_run

SPL = os.path.expanduser('~/poselift-study/results/splits')


def build(score_dir, fmt):
    pv = {}
    for d in load_split(f'{SPL}/split_b_test.json'):
        v = d['video']; s, c = v.split('_')
        n, _ = load_video(v)
        y, _, _ = gt_labels(v, n)
        if fmt == 'stgnf':
            p = f'{score_dir}/{int(s):02d}_{int(c):04d}.csv'
            sc = (-np.array(list(csv.reader(open(p))), float)[:, 1]) if os.path.exists(p) \
                else np.zeros(len(y))
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
    model, cond, seed, score_dir, fmt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    pv = build(score_dir, fmt)
    na = sum(1 for x in pv.values() if x['labels'].any())
    print(f'{model} E2-noise {cond} seed {seed}: {len(pv)} test clips ({na} anomaly)')
    score_run.score_all(model, {}, pv, seed=seed, split='b', variant=cond, fold='none')
