"""Model: motion-energy baseline. Runs entirely in poselift-harness env. No training.
Usage: python run_motion.py [--smoke]
"""
import sys, os, json, time
import numpy as np
from io_poselift import load_split
import motion_energy as ME
import score_run

SPL = os.path.expanduser('~/poselift-study/results/splits')

def build(videos, variant):
    pv = {}
    for v in videos:
        n, recs = ME.score_video(v, variant=variant, T=24, stride=12)
        _, y, n_used, note = ME.frame_count_and_labels(v)
        pv[v] = dict(n_frames=n, labels=y, window_records=recs)
    return pv

def main(smoke=False, seeds=(0,)):
    val_v = [d['video'] for d in load_split(f'{SPL}/split_b_val.json')]
    test_v = [d['video'] for d in load_split(f'{SPL}/split_b_test.json')]
    if smoke:
        val_v, test_v = val_v[:3], test_v[:4] + [d['video'] for d in load_split(f'{SPL}/split_b_test.json') if d['anomaly']][:2]
        test_v = list(dict.fromkeys(test_v))
    t0 = time.time()
    # motion-energy is a fixed deterministic computation -> identical for every seed
    # (seed rows written so multi-seed stats have a complete grid; variance is exactly 0)
    for variant in ('speed', 'accel'):
        pv_val = build(val_v, variant)
        pv_test = build(test_v, variant)
        for s in seeds:
            score_run.score_all('motion-energy', pv_val, pv_test, seed=s, split='b', variant=variant)
    print(f'motion-energy done in {time.time()-t0:.1f}s  (smoke={smoke}, seeds={seeds})')

if __name__ == '__main__':
    sd = [int(x) for x in sys.argv[1:] if x.isdigit()] or [0]
    main(smoke='--smoke' in sys.argv, seeds=sd)
