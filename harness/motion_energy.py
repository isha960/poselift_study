"""Motion-energy baseline (pre-registration sec.4).
Per window, anomaly score = mean per-joint speed (primary) or mean per-joint
acceleration (secondary variant), computed on the SHARED normalized representation.
No training. Runs entirely in the harness env.
"""
import numpy as np
from io_poselift import load_video, gt_labels
from repr import normalize_window, bbox_diag

T_DEFAULT, STRIDE_DEFAULT = 24, 12

def _tracks_of_video(video):
    n, frames = load_video(video)
    seqs = {}   # track -> dict(frame -> (kp(17,3), bbox(4,)))
    for fr, ppl in frames.items():
        for pid, (bb, kp) in ppl.items():
            seqs.setdefault(pid, {})[fr] = (kp, bb)
    return n, seqs

def score_video(video, variant='speed', T=T_DEFAULT, stride=STRIDE_DEFAULT):
    """-> list of window_records dict(track,start,length,score)."""
    n, seqs = _tracks_of_video(video)
    recs = []
    for pid, fmap in seqs.items():
        frs = sorted(fmap)
        if len(frs) < 3:
            continue
        f0, f1 = frs[0], frs[-1]
        # dense array over the track span; missing frames get NaN then interp inside normalize
        span = f1 - f0 + 1
        kp = np.zeros((span, 17, 2)); cf = np.zeros((span, 17))
        for fr in frs:
            k, _ = fmap[fr]
            kp[fr - f0] = k[:, :2]; cf[fr - f0] = k[:, 2]
        diag = bbox_diag(fmap[frs[len(frs) // 2]][1])
        norm = normalize_window(kp, cf, scheme='mid-hip+torso', tau=0.0, bbox_diag=diag)[..., :2]
        for st in range(0, max(1, span - T + 1), stride):
            w = norm[st:st + T]
            if len(w) < 3:
                continue
            vel = np.linalg.norm(np.diff(w, axis=0), axis=-1)                 # (T-1,17)
            if variant == 'speed':
                sc = float(np.nanmean(vel))
            else:  # acceleration
                acc = np.linalg.norm(np.diff(w, n=2, axis=0), axis=-1)
                sc = float(np.nanmean(acc))
            recs.append(dict(track=pid, start=f0 + st, length=len(w), score=sc))
    return n, recs

def frame_count_and_labels(video):
    n, _ = load_video(video)
    y, n_used, note = gt_labels(video, n)
    return n, y, n_used, note
