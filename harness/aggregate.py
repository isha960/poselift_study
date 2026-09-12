"""Window/person -> per-frame score series, gap-fill, smoothing (pre-registration sec.4).
Identical for every model. Models only ever emit raw per-(video,track,window) or
per-(video,frame,person) anomaly scores oriented higher=more anomalous.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

def windows_to_person_frames(n_frames, window_records):
    """window_records: list of dict(track:int, start:int, length:int, score:float).
    -> dict{track -> np.array(n_frames)} : per-frame score = MEAN over windows covering
       that frame (NaN where a track has no covering window)."""
    acc = {}
    for w in window_records:
        t = w['track']
        if t not in acc:
            acc[t] = (np.zeros(n_frames), np.zeros(n_frames))
        s, c = acc[t]
        a, b = w['start'], min(w['start'] + w['length'], n_frames)
        s[a:b] += w['score']; c[a:b] += 1
    out = {}
    for t, (s, c) in acc.items():
        f = np.full(n_frames, np.nan)
        nz = c > 0
        f[nz] = s[nz] / c[nz]
        out[t] = f
    return out

def persons_to_frame(person_frame_scores, n_frames):
    """dict{track->(n_frames,) with NaN} -> (n_frames,) = MAX over persons present."""
    if not person_frame_scores:
        return np.full(n_frames, np.nan)
    M = np.vstack(list(person_frame_scores.values()))          # (P, n_frames)
    allnan = np.all(np.isnan(M), axis=0)
    out = np.full(n_frames, np.nan)
    if (~allnan).any():
        out[~allnan] = np.nanmax(M[:, ~allnan], axis=0)
    return out

def gap_fill(frame_scores):
    """NaN frames (no detected person / no covering window) -> carry forward previous;
    leading NaNs -> first finite value; all-NaN -> zeros. Returns (filled, n_filled)."""
    s = frame_scores.copy()
    nan = np.isnan(s)
    n_filled = int(nan.sum())
    if nan.all():
        return np.zeros_like(s), n_filled
    idx = np.where(~nan)[0]
    s[:idx[0]] = s[idx[0]]
    for i in range(1, len(s)):
        if np.isnan(s[i]):
            s[i] = s[i - 1]
    return s, n_filled

def smooth(frame_scores, sigma):
    if sigma is None or sigma <= 0:
        return frame_scores
    return gaussian_filter1d(frame_scores, sigma=float(sigma), mode='nearest')

def build_frame_series(n_frames, window_records):
    """full pipeline for one video: windows -> person frames -> max -> gap fill.
    Returns (frame_scores (n_frames,), n_gap_filled)."""
    pf = windows_to_person_frames(n_frames, window_records)
    fr = persons_to_frame(pf, n_frames)
    return gap_fill(fr)
