"""Paired cluster bootstrap: 95% CI on the AUC DIFFERENCE between two models, resampling
the SAME 69 test clips (with replacement) for both models on each draw, so the resampling
respects the pairing (identical frames/clips scored by both models) as well as the
within-clip dependence structure (pre-registration sec.8; dissertation Ch5 sec.5.2/5.9,
decides pre-registered criterion 2 for MoCoDAD vs STG-NF).

Usage: bootstrap_paired.py <MODEL_A> <MODEL_B> [seed]   -> CI on AUC_A - AUC_B
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score

RES = os.path.expanduser('~/poselift-study/results')
CV = os.path.expanduser('~/poselift-study/data_converted')
N_BOOT = 2000
RNG = np.random.default_rng(20260829)

TAGS = {
    'STG-NF': 'STG-NF_s{s}_b_main_sig0',
    'COSKAD': 'COSKAD_s{s}_b_main_sig0',
    'MoCoDAD': 'MoCoDAD_s{s}_b_main_sig0',
    'motion-energy(speed)': 'motion-energy_s{s}_b_speed_sig0',
    'motion-energy(accel)': 'motion-energy_s{s}_b_accel_sig0',
}


def load(model, seed):
    tag = TAGS[model].format(s=seed)
    d = np.load(f'{RES}/raw/scores/{tag}.npz', allow_pickle=True)
    return d['scores'].astype(float), d['labels'].astype(int), json.loads(str(d['videos']))


def main(model_a, model_b, seed=0):
    sa, ya, va = load(model_a, seed)
    sb, yb, vb = load(model_b, seed)
    vids = sorted(va)
    assert vids == sorted(vb), 'the two models were not scored on the same clip set'
    gt_dir = f'{CV}/b_test/gt'
    ys = {v: np.load(f'{gt_dir}/{v}.npy').astype(int) for v in vids}
    assert all(len(va[v]) == len(ys[v]) for v in vids), 'clip length mismatch vs GT'

    auc_a0 = roc_auc_score(ya, sa); auc_b0 = roc_auc_score(yb, sb)
    diffs, aucs_a, aucs_b = [], [], []
    for _ in range(N_BOOT):
        samp = RNG.choice(len(vids), len(vids), replace=True)
        chosen = [vids[i] for i in samp]
        y = np.concatenate([ys[v] for v in chosen])
        if y.min() == y.max():
            continue
        s_a = np.concatenate([va[v] for v in chosen])
        s_b = np.concatenate([vb[v] for v in chosen])
        aa, ab = roc_auc_score(y, s_a), roc_auc_score(y, s_b)
        aucs_a.append(aa); aucs_b.append(ab); diffs.append(aa - ab)
    diffs = np.array(diffs)
    lo, med, hi = np.percentile(diffs, [2.5, 50, 97.5])
    # two-sided bootstrap p-value: 2 * min(P(diff<=0), P(diff>=0)), floored at 1/N_BOOT
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_boot = max(p_boot, 1.0 / len(diffs))
    print(f'{model_a} seed{seed} AUC (point) = {auc_a0:.4f}   bootstrap median {np.median(aucs_a):.4f}')
    print(f'{model_b} seed{seed} AUC (point) = {auc_b0:.4f}   bootstrap median {np.median(aucs_b):.4f}')
    print(f'paired cluster-bootstrap, {len(diffs)}/{N_BOOT} valid resamples (69 clips, with replacement)')
    print(f'  {model_a} - {model_b} point diff = {auc_a0 - auc_b0:+.4f}')
    print(f'  95% CI on the difference: [{lo:+.4f}, {hi:+.4f}]  (median {med:+.4f})')
    print(f'  two-sided bootstrap p (uncorrected) = {p_boot:.4g}')
    print(f'  excludes zero: {lo > 0 or hi < 0}')
    import csv
    out = f'{RES}/stats/bootstrap_paired.csv'
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['model_a', 'model_b', 'seed', 'auc_a', 'auc_b', 'diff', 'ci95_lo', 'ci95_hi',
                        'p_boot_uncorrected', 'n_valid_resamples', 'note'])
        w.writerow([model_a, model_b, seed, f'{auc_a0:.4f}', f'{auc_b0:.4f}', f'{auc_a0-auc_b0:+.4f}',
                    f'{lo:+.4f}', f'{hi:+.4f}', f'{p_boot:.4g}', len(diffs),
                    'paired cluster bootstrap, same 69-clip resample applied to both models, N_BOOT=2000'])
    print(f'appended {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
