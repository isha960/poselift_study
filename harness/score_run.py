"""Common scoring entry for every model.

A per-model runner builds two dicts (val + test):
    per_video[video] = dict(n_frames=int, labels=1d-int,
                            window_records=[dict(track,start,length,score)]  # OR
                            frame_scores=1d-float)
score oriented higher = more anomalous.

score_all() then, IDENTICALLY for every model:
  * builds per-frame series (aggregate.build_frame_series or given frame_scores)
  * for sigma in {0,1,2,4,8}: smooth, calibrate fixed-FPR thresholds on the
    all-normal VAL frames, evaluate on TEST, append tidy rows to raw/metrics.csv
    and save raw/scores/<tag>.npz
  * sigma=0 rows are the pre-registered primary; sigma>0 rows are exploratory
    (see preregistration Amendment 1).
"""
import os, json
import numpy as np
from aggregate import build_frame_series, smooth, gap_fill
import metrics as M
import run_eval

SIGMAS = [0, 1, 2, 4, 8]
FPRS = (0.01, 0.05, 0.10)

def _series(v):
    if 'frame_scores' in v:
        fs, nf = gap_fill(np.asarray(v['frame_scores'], float))
        return fs, nf
    return build_frame_series(v['n_frames'], v['window_records'])

def score_all(model, per_video_val, per_video_test, *, seed=0, split='b', variant='main', fold='none'):
    # raw (unsmoothed) per-video series, computed once
    val_raw = {k: _series(v)[0] for k, v in per_video_val.items()}
    test_series = {k: _series(v) for k, v in per_video_test.items()}
    all_rows = []
    for sigma in SIGMAS:
        val_cat = np.concatenate([smooth(s, sigma) for s in val_raw.values()]) if val_raw else np.array([])
        vth = {fpr: M.threshold_at_fpr_neg_only(val_cat, fpr) for fpr in FPRS} if len(val_cat) else None
        pv = {}
        for k, v in per_video_test.items():
            fs, nf = test_series[k]
            pv[k] = dict(n_frames=v['n_frames'], labels=v['labels'],
                         frame_scores=fs)  # already raw; run_eval will smooth
        rows = run_eval.evaluate(pv, model=model, seed=seed, split=split,
                                 variant=variant, smoothing_sigma=sigma, fold=fold,
                                 val_thresholds=vth, write_scores=(sigma == 0),
                                 run_tag=f'{model}_s{seed}_{split}_{fold}_{variant}_sig{sigma}' if fold!='none' else f'{model}_s{seed}_{split}_{variant}_sig{sigma}')
        all_rows += rows
        prim = next(r for r in rows if r['metric'] == 'frame_roc_auc')
        print(f'  [{model} {variant} sig={sigma}] frame_roc_auc={prim["value"]:.4f}'
              + ('   <-- PRIMARY (n=1, preliminary)' if sigma == 0 else '   (exploratory)'))
    return all_rows
