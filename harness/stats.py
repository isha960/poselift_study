"""Statistics for the model comparison (pre-registration sec.8).

Multi-seed aware. For each model with >=1 seed present it emits, and once >=2 seeds
are present it adds the seed-level tests.

Per (model): per-seed frame ROC-AUC (sig=0), mean + 95% t-interval.
Confirmatory vs STG-NF (each contemporary method):
  * DeLong on the POOLED test frames of seed 0 (p + 95% CI of AUC diff)   [pooled-frame view]
  * cluster bootstrap over TEST CLIPS on seed 0 -> CI of AUC and of the diff [clip view]
  * Wilcoxon signed-rank on the matched per-seed AUCs                       [seed view]
  * Cliff's delta + rank-biserial on the per-seed AUC pairs
  * Holm-Bonferroni across the contemporary-method comparisons
Verdict per model in stats/verdicts.csv per the pre-registered rule (sec.2).

Outputs: stats/{perseed.csv, delong.csv, bootstrap.csv, wilcoxon.csv, verdicts.csv}
"""
import os, glob, json, csv, itertools
import numpy as np
from scipy import stats as ss
from sklearn.metrics import roc_auc_score
from delong import delong_test

RES = os.path.expanduser('~/poselift-study/results')
SPL = os.path.expanduser('~/poselift-study/results/splits')
CV = os.path.expanduser('~/poselift-study/data_converted')
BASELINE = 'STG-NF'
MARGIN = 0.03
N_BOOT = 2000
RNG = np.random.default_rng(20260829)

MODEL_TAGS = {   # model -> npz tag pattern (seed substituted)
    'STG-NF':  'STG-NF_s{s}_b_main_sig0',
    'COSKAD':  'COSKAD_s{s}_b_main_sig0',
    'MoCoDAD': 'MoCoDAD_s{s}_b_main_sig0',
    'motion-energy(speed)': 'motion-energy_s{s}_b_speed_sig0',
    'motion-energy(accel)': 'motion-energy_s{s}_b_accel_sig0',
}

def load(tag):
    f = f'{RES}/raw/scores/{tag}.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return d['scores'].astype(float), d['labels'].astype(int), json.loads(str(d['videos']))

def perseed_aucs(model):
    out = {}
    for s in range(4):
        r = load(MODEL_TAGS[model].format(s=s))
        if r is not None:
            out[s] = roc_auc_score(r[1], r[0])
    return out

def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x[0]), float(x[0]), float(x[0]))
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    h = ss.t.ppf(0.975, len(x) - 1) * se
    return (float(m), float(m - h), float(m + h))

def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def cluster_bootstrap(per_video_scores, gt_dir):
    vids = list(per_video_scores)
    ys = {v: np.load(f'{gt_dir}/{v}.npy').astype(int) for v in vids}
    aucs = []
    for _ in range(N_BOOT):
        samp = RNG.choice(len(vids), len(vids), replace=True)
        s = np.concatenate([per_video_scores[vids[i]] for i in samp])
        y = np.concatenate([ys[vids[i]] for i in samp])
        if y.min() != y.max():
            aucs.append(roc_auc_score(y, s))
    return np.percentile(aucs, [2.5, 50, 97.5])

def main():
    os.makedirs(f'{RES}/stats', exist_ok=True)
    models = [m for m in MODEL_TAGS if perseed_aucs(m)]
    per = {m: perseed_aucs(m) for m in models}
    gt_dir = f'{CV}/b_test/gt'

    # ---- perseed.csv ----
    with open(f'{RES}/stats/perseed.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'n_seeds', 'seeds', 'auc_per_seed', 'mean', 'ci95_lo', 'ci95_hi'])
        for m in models:
            a = [per[m][s] for s in sorted(per[m])]
            mean, lo, hi = tci(a)
            w.writerow([m, len(a), ';'.join(map(str, sorted(per[m]))),
                        ';'.join(f'{x:.4f}' for x in a), f'{mean:.4f}', f'{lo:.4f}', f'{hi:.4f}'])
            print(f'  {m:22s} n={len(a)}  AUC/seed {[round(x,3) for x in a]}  mean {mean:.4f} [{lo:.4f},{hi:.4f}]')

    if BASELINE not in per:
        print('no baseline seeds; stop'); return
    base = per[BASELINE]
    contemp = [m for m in models if m != BASELINE]

    # ---- DeLong + bootstrap on seed 0 ----
    b0 = load(MODEL_TAGS[BASELINE].format(s=0))
    dl_rows, boot_rows = [], []
    if b0 is not None:
        sb, yb, vb = b0
        lo, md, hi = cluster_bootstrap(vb, gt_dir)
        boot_rows.append([BASELINE, f'{roc_auc_score(yb,sb):.4f}', f'{lo:.4f}', f'{md:.4f}', f'{hi:.4f}'])
        for m in contemp:
            r0 = load(MODEL_TAGS[m].format(s=0))
            if r0 is None:
                continue
            s0, y0, v0 = r0
            lo, md, hi = cluster_bootstrap(v0, gt_dir)
            boot_rows.append([m, f'{roc_auc_score(y0,s0):.4f}', f'{lo:.4f}', f'{md:.4f}', f'{hi:.4f}'])
            if np.array_equal(y0, yb):
                dd = delong_test(yb, s0, sb)
                dl_rows.append([m, f'{dd["auc_a"]:.4f}', f'{dd["auc_b"]:.4f}', f'{dd["diff"]:.4f}',
                                f'{dd["ci95"][0]:.4f}', f'{dd["ci95"][1]:.4f}', f'{dd["p"]:.4g}'])
    with open(f'{RES}/stats/bootstrap.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'auc_seed0', 'boot_lo', 'boot_med', 'boot_hi', 'note'])
        for r in boot_rows: w.writerow(r + ['seed0; cluster bootstrap over 69 test clips'])
    with open(f'{RES}/stats/delong.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model_vs_STGNF', 'auc_model', 'auc_stgnf', 'auc_diff', 'ci95_lo', 'ci95_hi', 'p', 'note'])
        for r in dl_rows: w.writerow(r + ['seed0; DeLong on pooled test frames'])

    # ---- Wilcoxon + Cliff's delta + Holm across contemporary models ----
    wil = []
    for m in contemp:
        common = sorted(set(per[m]) & set(base))
        a = np.array([per[m][s] for s in common]); c = np.array([base[s] for s in common])
        d = a - c
        if len(common) >= 2 and np.any(d != 0):
            try:
                st, p = ss.wilcoxon(a, c)
            except ValueError:
                st, p = np.nan, np.nan
        else:
            st, p = np.nan, np.nan
        wil.append(dict(model=m, n=len(common), mean_dAUC=float(np.mean(d)),
                        per_seed_dAUC=';'.join(f'{x:+.4f}' for x in d),
                        wilcoxon_stat=st, wilcoxon_p=p,
                        cliffs_delta=cliffs_delta(a, c),
                        all_seeds_positive=bool(np.all(d > 0)),
                        all_seeds_negative=bool(np.all(d < 0))))
    ps = [w['wilcoxon_p'] for w in wil if np.isfinite(w['wilcoxon_p'])]
    order = np.argsort(ps) if ps else []
    holm = {}
    kk = len(ps)
    for rank, idx in enumerate([i for i, w in enumerate(wil) if np.isfinite(w['wilcoxon_p'])][:0] or
                               sorted([i for i, w in enumerate(wil) if np.isfinite(w['wilcoxon_p'])],
                                      key=lambda i: wil[i]['wilcoxon_p'])):
        holm[idx] = min(1.0, wil[idx]['wilcoxon_p'] * (kk - rank))
    with open(f'{RES}/stats/wilcoxon.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model_vs_STGNF', 'n_common_seeds', 'mean_dAUC', 'per_seed_dAUC',
                    'wilcoxon_p', 'holm_p', 'cliffs_delta', 'all_seeds_positive'])
        for i, ww in enumerate(wil):
            w.writerow([ww['model'], ww['n'], f'{ww["mean_dAUC"]:+.4f}', ww['per_seed_dAUC'],
                        f'{ww["wilcoxon_p"]:.4g}' if np.isfinite(ww['wilcoxon_p']) else 'na',
                        f'{holm.get(i, float("nan")):.4g}' if i in holm else 'na',
                        f'{ww["cliffs_delta"]:+.3f}', ww['all_seeds_positive']])

    # ---- verdicts.csv (pre-registration sec.2) ----
    dlp = {r[0]: (float(r[3]), float(r[4]), float(r[5]), float(r[6])) for r in dl_rows}  # model -> (diff, lo, hi, p)
    bootd = {r[0]: (float(r[2]), float(r[4])) for r in boot_rows}  # model -> (lo, hi) of its own AUC
    base_pt = roc_auc_score(b0[1], b0[0]) if b0 is not None else np.nan
    with open(f'{RES}/stats/verdicts.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'mean_dAUC_seeds', 'n_seeds', 'c1_margin>=0.03', 'c2_delong_CI_excl_0',
                    'c2_bootstrap_lb>base', 'c4_all_seeds_positive', 'verdict', 'note'])
        for ww in wil:
            m = ww['model']
            c1 = ww['mean_dAUC'] >= MARGIN
            dd = dlp.get(m)
            c2_delong = bool(dd and (dd[1] > 0 or dd[2] < 0))
            bl = bootd.get(m)
            c2_boot = bool(bl and bl[0] > base_pt)
            c4 = ww['all_seeds_positive']
            n = ww['n']
            allc = c1 and c2_delong and c2_boot and c4 and n >= 4
            if allc:
                v = 'OUTPERFORMS STG-NF'
            elif ww['mean_dAUC'] <= -MARGIN and ww['all_seeds_negative']:
                v = 'BELOW STG-NF'
            else:
                v = 'comparable / within noise'
            note = f'criteria 1..4 (n<4 -> cannot fully confirm)' if n < 4 else 'full pre-registered check'
            w.writerow([m, f'{ww["mean_dAUC"]:+.4f}', n, c1, c2_delong, c2_boot, c4, v, note])
            print(f'  VERDICT {m:22s} mean dAUC {ww["mean_dAUC"]:+.4f} (n={n}) -> {v}')

if __name__ == '__main__':
    main()
