"""Orchestrator: (frame scores per test video) -> all metrics, with & without smoothing.
Writes tidy rows to results/raw/metrics.csv and the frame arrays to results/raw/scores/.

A "run" = one (model, seed, split, score_variant). The per-model wrapper produces
`per_video`: dict{video -> dict(n_frames, labels(1d), window_records[list] OR frame_scores(1d))}.
Everything downstream is identical for every model.
"""
import os, csv, json
import numpy as np
from aggregate import build_frame_series, smooth
import metrics as M

RES = os.path.expanduser('~/poselift-study/results')
SIGMA_GRID = [0, 1, 2, 4, 8]

def _series_for_video(v):
    n = v['n_frames']
    if 'frame_scores' in v:
        fs = np.asarray(v['frame_scores'], float)
        from aggregate import gap_fill
        fs, nf = gap_fill(fs)
        return fs, nf
    return build_frame_series(n, v['window_records'])

def evaluate(per_video, *, model, seed, split, variant, smoothing_sigma, fold='none',
             val_thresholds=None, write_scores=True, run_tag=None):
    """per_video: {video: {n_frames, labels(1d,int), window_records|frame_scores}}
    val_thresholds: optional dict{fpr: thr} calibrated on val; if None, fixed-FPR
                    metrics are computed self-referentially on test and flagged.
    Returns list of tidy metric rows (also appended to metrics.csv)."""
    run_tag = run_tag or f'{model}_s{seed}_{split}_{variant}_sig{smoothing_sigma}'
    all_s, all_y = [], []
    events, anom_events = [], []
    gap_total = 0
    per_video_scores = {}

    for vid, v in per_video.items():
        y = np.asarray(v['labels'], int)
        fs, nf = _series_for_video(v)
        gap_total += nf
        # align score length to label length
        if len(fs) > len(y):
            fs = fs[:len(y)]
        elif len(fs) < len(y):
            fs = np.concatenate([fs, np.full(len(y) - len(fs), fs[-1] if len(fs) else 0.0)])
        fs_s = smooth(fs, smoothing_sigma)
        per_video_scores[vid] = fs_s
        all_s.append(fs_s); all_y.append(y)
        is_anom = bool(y.any())
        if is_anom:
            on = int(np.argmax(y))
            off = len(y) - int(np.argmax(y[::-1]))
            events.append(dict(is_anomaly=True, clip_max_score=float(fs_s[on:off].max())))
            anom_events.append(dict(onset=on, frame_scores_in_window=fs_s[on:off]))
        else:
            events.append(dict(is_anomaly=False, clip_max_score=float(fs_s.max())))

    S = np.concatenate(all_s); Y = np.concatenate(all_y)
    rows = []
    def add(metric, value, extra=''):
        rows.append(dict(model=model, seed=seed, split=split, fold=fold,
                         variant=variant, smoothed=int(smoothing_sigma and smoothing_sigma > 0),
                         sigma=smoothing_sigma, metric=metric, value=value, note=extra))

    add('frame_roc_auc', M.frame_roc_auc(Y, S))
    add('frame_pr_auc', M.frame_pr_auc(Y, S), f'base_rate={M.base_rate(Y):.4f}')
    add('base_rate', M.base_rate(Y))
    add('eer', M.eer(Y, S))
    add('event_roc_auc', M.event_roc_auc(events))
    add('gap_filled_frames', float(gap_total))
    for fpr in (0.01, 0.05, 0.10):
        if val_thresholds and fpr in val_thresholds:
            thr = val_thresholds[fpr]; src = 'val'
        else:
            thr = M.threshold_at_fpr(Y, S, fpr); src = 'test_selfref'
        add(f'recall_at_fpr{fpr}', M.recall_at_threshold(Y, S, thr), f'thr_src={src}')
        add(f'fpr_at_thr_fpr{fpr}', M.fpr_at_threshold(Y, S, thr), f'thr_src={src}')
        add(f'f1_at_fpr{fpr}', M.f1_at_threshold(Y, S, thr), f'thr_src={src}')
        lat = M.detection_latencies(anom_events, thr)
        add(f'median_latency_at_fpr{fpr}', M.median_latency(lat), f'thr_src={src}')

    # persist
    os.makedirs(f'{RES}/raw/scores', exist_ok=True)
    if write_scores:
        np.savez_compressed(f'{RES}/raw/scores/{run_tag}.npz',
                            scores=S, labels=Y,
                            videos=json.dumps({k: per_video_scores[k].tolist() for k in per_video_scores}))
    mfile = f'{RES}/raw/metrics.csv'
    new = not os.path.exists(mfile)
    with open(mfile, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        for r in rows: w.writerow(r)
    return rows
