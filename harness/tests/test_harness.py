"""Harness unit tests (pre-registration: 'Unit-test the harness').
Run:  cd harness && ../../.conda/envs/poselift-harness/bin/python -m pytest tests -q
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.metrics import roc_auc_score
from delong import delong_test, auc_variance
from aggregate import windows_to_person_frames, persons_to_frame, gap_fill, smooth, build_frame_series
import metrics as M


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 400)
        if y.min() == y.max():
            continue
        s = rng.normal(y * 0.7, 1.0)
        a_dl, _ = auc_variance(y, s)
        assert abs(a_dl - roc_auc_score(y, s)) < 1e-9


def test_delong_identical_scorers_diff_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500); s = rng.normal(y, 1)
    r = delong_test(y, s, s)
    assert abs(r['diff']) < 1e-12 and r['p'] == 1.0


def test_random_scorer_auc_near_half():
    rng = np.random.default_rng(2)
    aucs = []
    for _ in range(200):
        y = rng.integers(0, 2, 300); s = rng.random(300)
        if y.min() != y.max():
            aucs.append(roc_auc_score(y, s))
    assert abs(np.mean(aucs) - 0.5) < 0.02


def test_windows_to_frames_mean_and_max():
    recs = [dict(track=1, start=0, length=4, score=1.0),
            dict(track=1, start=2, length=4, score=3.0),   # frames 2,3 covered twice -> mean 2.0
            dict(track=2, start=0, length=2, score=10.0)]
    pf = windows_to_person_frames(6, recs)
    assert np.allclose(pf[1][:2], 1.0) and np.allclose(pf[1][2:4], 2.0)
    fr = persons_to_frame(pf, 6)
    assert fr[0] == 10.0 and fr[4] == 3.0                   # max over persons; track2 gone by fr4


def test_gap_fill_carry_forward():
    s = np.array([np.nan, np.nan, 2.0, np.nan, 5.0, np.nan])
    f, n = gap_fill(s)
    assert n == 4 and np.allclose(f, [2, 2, 2, 2, 5, 5])
    z, n2 = gap_fill(np.full(4, np.nan))
    assert n2 == 4 and np.allclose(z, 0)


def test_smoothing_monotone_variance():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 500)
    assert np.var(smooth(x, 8)) < np.var(smooth(x, 2)) < np.var(x) + 1e-9


def test_motion_energy_constant_vs_jerk():
    """On mid-hip-centred coords, whole-body translation is (correctly) removed, so
    the probe must be ARTICULATION: distal joints moving relative to the torso."""
    from repr import normalize_window
    T = 24
    rng = np.random.default_rng(0)
    # a non-degenerate skeleton: 17 joints scattered around a ~80px torso
    base = rng.normal(0, 40, (17, 2)) + np.array([100.0, 100.0])
    base[11] = [90, 160]; base[12] = [110, 160]      # hips
    base[5] = [88, 80];  base[6] = [112, 80]         # shoulders
    still = np.tile(base, (T, 1, 1)) + rng.normal(0, 0.02, (T, 17, 2))
    conf = np.ones((T, 17))
    jerk = still.copy()
    jerk[T // 2:, [9, 10, 7, 8]] += 35.0             # wrists+elbows jerk, hips/shoulders fixed
    def energy(seq):
        n = normalize_window(seq, conf, scheme='mid-hip+torso', bbox_diag=50.0)[..., :2]
        return float(np.nanmean(np.linalg.norm(np.diff(n, axis=0), axis=-1)))
    assert energy(jerk) > 5 * energy(still)


def test_event_roc_auc_separable():
    ev = [dict(is_anomaly=True, clip_max_score=0.9),
          dict(is_anomaly=True, clip_max_score=0.8),
          dict(is_anomaly=False, clip_max_score=0.2),
          dict(is_anomaly=False, clip_max_score=0.1)]
    assert M.event_roc_auc(ev) == 1.0


def test_detection_latency():
    ev = [dict(onset=0, frame_scores_in_window=np.array([0, 0, 0, 1, 1.0])),
          dict(onset=0, frame_scores_in_window=np.array([0, 0, 0, 0, 0.0]))]
    lat = M.detection_latencies(ev, thr=0.5)
    assert lat[0] == 3.0 and np.isnan(lat[1])
    assert M.median_latency(lat) == 3.0
