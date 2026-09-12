"""E2 keypoint-degradation perturbations (pre-registration sec.7, "E2 keypoint noise").

EVAL-TIME ONLY, NO RETRAIN. Perturbs the raw PoseLift keypoints of the split-(b) TEST
videos only; train + val stay clean (fixed-FPR thresholds are still calibrated on the
clean all-normal val split, then applied to the degraded test set -- the realistic
"model deployed, pose detector degrades at inference" scenario). Primary metric is the
threshold-free frame ROC-AUC, exactly as in E1.

Pre-registered grid:
  gaussN  : additive Gaussian noise on (x,y), std = N pixels, N in {1,2,4,8}
  jdropP  : each (frame, joint) independently dropped with prob P in {0.05,0.10,0.20}
            (dropped joint -> x=y=conf=0, i.e. detector failed to localise it)
  fdropP  : each (track, frame) independently dropped with prob P in {0.05,0.10}
            (whole-person miss -> that frame removed from the track)

Noise is applied in RAW PIXEL space (the literal pre-registration spec "{1,2,4,8}px");
each external repo then applies its own native normalisation downstream, unchanged.
Only keypoints with conf>0 in the clean data are perturbed / eligible to be dropped.

RNG: np.random.default_rng([20260901, COND_INDEX, seed]) -- deterministic, one stream
per (condition, trained-model seed) pair.
"""
import os, pickle
import numpy as np
from io_poselift import _find_pkl

# (name, kind, param) -- COND_INDEX is the position in this list, frozen.
CONDITIONS = [
    ('gauss1',  'gauss', 1.0),
    ('gauss2',  'gauss', 2.0),
    ('gauss4',  'gauss', 4.0),
    ('gauss8',  'gauss', 8.0),
    ('jdrop05', 'jdrop', 0.05),
    ('jdrop10', 'jdrop', 0.10),
    ('jdrop20', 'jdrop', 0.20),
    ('fdrop05', 'fdrop', 0.05),
    ('fdrop10', 'fdrop', 0.10),
]
COND = {c[0]: c for c in CONDITIONS}
SEED_ROOT = 20260901


def _rng(cond_name, seed):
    idx = [i for i, c in enumerate(CONDITIONS) if c[0] == cond_name][0]
    return np.random.default_rng([SEED_ROOT, idx, int(seed)])


def perturb_frames(frames, cond_name, seed):
    """frames: {frame:int -> {pid:int -> (bbox(4,), kp(17,3))}}  (as io_poselift.load_video).
    Returns a NEW dict, same structure, perturbed per `cond_name`. bbox is left untouched
    (models here consume keypoints only; STG-NF/COSKAD ignore bbox, MoCoDAD uses x/y cols)."""
    _, kind, p = COND[cond_name]
    rng = _rng(cond_name, seed)
    out = {}
    for fr in sorted(frames):
        ppl = frames[fr]
        new_ppl = {}
        for pid, (bb, kp) in ppl.items():
            kp = np.array(kp, float)                       # (17,3) copy
            valid = kp[:, 2] > 0
            if kind == 'gauss':
                noise = rng.normal(0.0, p, size=(17, 2))
                kp[valid, :2] += noise[valid]
            elif kind == 'jdrop':
                drop = (rng.random(17) < p) & valid
                kp[drop, :] = 0.0
            elif kind == 'fdrop':
                if rng.random() < p:
                    continue                               # whole-person miss this frame
            new_ppl[int(pid)] = (np.array(bb, float), kp)
        if new_ppl:
            out[int(fr)] = new_ppl
    return out


def write_perturbed_pkls(videos, cond_name, seed, out_dir):
    """Write perturbed {video}.pkl for each test `video` into out_dir, byte-compatible
    with io_poselift.load_video (dict{frame:{pid:[bbox_list4, kp_ndarray(17,3)]}})."""
    os.makedirs(out_dir, exist_ok=True)
    for v in videos:
        d = pickle.load(open(_find_pkl(v), 'rb'))
        frames = {int(fr): {int(pid): (np.asarray(rec[0], float), np.asarray(rec[1], float))
                            for pid, rec in ppl.items()}
                  for fr, ppl in d.items()}
        pert = perturb_frames(frames, cond_name, seed)
        ser = {fr: {pid: [bb.tolist(), kp] for pid, (bb, kp) in ppl.items()}
               for fr, ppl in pert.items()}
        pickle.dump(ser, open(f'{out_dir}/{v}.pkl', 'wb'))
    return len(videos)


if __name__ == '__main__':
    # self-test: shapes, determinism, expected effect direction
    import numpy as np
    frames = {0: {1: (np.array([0, 0, 100, 200.]), np.hstack([np.full((17, 1), 50.),
                                                              np.full((17, 1), 60.),
                                                              np.ones((17, 1))]))}}
    a = perturb_frames(frames, 'gauss4', 0)[0][1][1]
    b = perturb_frames(frames, 'gauss4', 0)[0][1][1]
    assert np.allclose(a, b), 'not deterministic'
    c = perturb_frames(frames, 'gauss4', 1)[0][1][1]
    assert not np.allclose(a, c), 'seed had no effect'
    assert abs(np.std(a[:, :2] - 50 - np.array([0, 10.])[None]) - 4) < 3, 'gauss magnitude off'
    jd = perturb_frames(frames, 'jdrop20', 0)[0][1][1]
    assert ((jd == 0).all(axis=1)).sum() >= 1, 'jdrop dropped nothing'
    rec = frames[0][1]
    many = {i: {1: rec} for i in range(2000)}
    kept = len(perturb_frames(many, 'fdrop10', 0))
    assert 1750 < kept < 1850, f'fdrop rate off: {kept}/2000'
    print('perturb.py self-test OK')
