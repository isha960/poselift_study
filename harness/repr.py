"""Shared pose representation (pre-registration sec.4).

Primary scheme 'mid-hip+torso':
  translate so mid-hip is origin (fallback mid-shoulder if a hip conf<tau),
  scale by torso length ||mid-shoulder - mid-hip|| per window (fallback bbox diag),
  keep conf as 3rd channel.
Low-conf joints (conf<tau) are linearly interpolated WITHIN a track only; tau=0.0
for primary runs (PoseLift confs are already sparse).

These functions are used by the motion-energy baseline and by E2. The three
external repos (STG-NF/COSKAD/MoCoDAD) keep their NATIVE preprocessing for the
reproduction pass; the deviation is logged in Table 3.1.
"""
import numpy as np
from io_poselift import COCO17

L_SHO, R_SHO, L_HIP, R_HIP = COCO17['l_sho'], COCO17['r_sho'], COCO17['l_hip'], COCO17['r_hip']

def interp_track(seq, conf, tau=0.0):
    """seq (T,17,2), conf (T,17) -> seq with conf<=tau linearly interpolated over time
    within this track. Ends held constant. Joints never valid stay 0."""
    seq = seq.copy()
    T, J, _ = seq.shape
    for j in range(J):
        good = np.where(conf[:, j] > tau)[0]
        if len(good) == 0:
            continue
        for d in range(2):
            seq[:, j, d] = np.interp(np.arange(T), good, seq[good, j, d])
    return seq

def normalize_window(seq, conf, scheme='mid-hip+torso', tau=0.0, bbox_diag=None):
    """seq (T,17,2) pixel coords, conf (T,17). -> (T,17,3) normalized (x,y,conf)."""
    seq = interp_track(seq, conf, tau)
    T = seq.shape[0]
    mid_hip = 0.5 * (seq[:, L_HIP] + seq[:, R_HIP])            # (T,2)
    mid_sho = 0.5 * (seq[:, L_SHO] + seq[:, R_SHO])
    hips_ok = (conf[:, L_HIP] > tau) & (conf[:, R_HIP] > tau)

    if scheme == 'none':
        out = seq
    elif scheme == 'bbox-minmax':
        lo = seq.reshape(-1, 2).min(0); hi = seq.reshape(-1, 2).max(0)
        rng = np.where(hi - lo > 1e-6, hi - lo, 1.0)
        out = (seq - lo) / rng * 2 - 1
    else:
        if scheme == 'mid-shoulder+bboxdiag':
            origin = mid_sho
            scale = np.full(T, bbox_diag if bbox_diag else 1.0)
        else:  # 'mid-hip+torso' (primary)
            origin = np.where(hips_ok[:, None], mid_hip, mid_sho)
            torso = np.linalg.norm(mid_sho - mid_hip, axis=1)          # (T,)
            med = np.median(torso[torso > 1e-6]) if np.any(torso > 1e-6) else 0.0
            fallback = bbox_diag if bbox_diag else (med if med > 0 else 1.0)
            scale = np.where(torso > 1e-6, torso, fallback)
        out = (seq - origin[:, None, :]) / scale[:, None, None]

    return np.concatenate([out, conf[..., None]], axis=-1).astype(np.float32)

def bbox_diag(bbox):
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))
