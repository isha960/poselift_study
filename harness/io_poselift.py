"""PoseLift raw I/O + split iteration. One place that knows the .pkl layout.

.pkl  : dict{frame:int -> dict{person_id:int -> [bbox_xyxy(list4), kp ndarray(17,3)=(x,y,conf)]}}
        frames 0..N-1 contiguous; some frames empty.
GT .npy: per-video int{0,1} frame labels. 4 videos have len(pkl)!=len(gt); we align to
         the GT length (truncate extra score frames, carry-forward-pad missing) and log it.
COCO-17 order: 0 nose,1 Leye,2 Reye,3 Lear,4 Rear,5 Lsho,6 Rsho,7 Lelb,8 Relb,9 Lwri,
              10 Rwri,11 Lhip,12 Rhip,13 Lkne,14 Rkne,15 Lank,16 Rank
"""
import os, glob, json, pickle
import numpy as np

RAW = os.path.expanduser('~/PoseLift/Pickle_files')
COCO17 = dict(nose=0, l_sho=5, r_sho=6, l_hip=11, r_hip=12)

def _find_pkl(video):
    # E2-noise: POSELIFT_PKL_OVERRIDE points at a dir of perturbed {video}.pkl for the
    # split-(b) TEST videos only; anything not found there (train/val) falls through to
    # the real pickles, so train/val staging is unaffected even with the env var set.
    ovr = os.environ.get('POSELIFT_PKL_OVERRIDE')
    if ovr:
        p = f'{ovr}/{video}.pkl'
        if os.path.exists(p):
            return p
    for sub in ('Train', 'Test'):
        p = f'{RAW}/{sub}/{video}.pkl'
        if os.path.exists(p):
            return p
    raise FileNotFoundError(video)

def load_video(video):
    """-> (n_frames, dict{frame:{pid:(bbox(4,), kp(17,3))}})  ; n_frames from the pkl."""
    d = pickle.load(open(_find_pkl(video), 'rb'))
    n = (max(d) + 1) if d else 0
    out = {}
    for fr, ppl in d.items():
        out[fr] = {int(pid): (np.asarray(rec[0], float), np.asarray(rec[1], float))
                   for pid, rec in ppl.items()}
    return n, out

def gt_labels(video, n_frames_pkl):
    """Frame labels aligned to whatever length we will score.
    Returns (labels int8 array, n_used, note-or-None)."""
    p = f'{RAW}/GT/{video}.npy'
    if not os.path.exists(p):
        return np.zeros(n_frames_pkl, np.int8), n_frames_pkl, None      # normal video, no GT
    g = np.load(p).astype(np.int8)
    if len(g) == n_frames_pkl:
        return g, len(g), None
    note = f'{video}: pkl_frames={n_frames_pkl} gt_frames={len(g)} -> scored on gt length'
    return g, len(g), note

def load_split(split_json):
    """results/splits/*.json -> list of dicts {video,camera,n_frames,anomaly,...}."""
    return json.load(open(os.path.expanduser(split_json)))['videos']

def split_a():
    """Release split: Train/ (normal) vs Test/ (41 anomaly + 6 normal)."""
    tr = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{RAW}/Train/*.pkl'))
    te = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{RAW}/Test/*.pkl'))
    return tr, te
