"""Assemble aggregated/summary.csv + vector figures from raw/scores + raw/metrics.csv.
Run after all model ingests are in metrics.csv. Idempotent.
"""
import os, csv, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

RES = os.path.expanduser('~/poselift-study/results')
CV = os.path.expanduser('~/poselift-study/data_converted')
BASELINE = 'STG-NF'

MODELS = [  # (label, npz tag)
    ('motion-energy (speed)', 'motion-energy_s0_b_speed_sig0'),
    ('motion-energy (accel)', 'motion-energy_s0_b_accel_sig0'),
    ('STG-NF',                'STG-NF_s0_b_main_sig0'),
    ('COSKAD',                'COSKAD_s0_b_main_sig0'),
    ('MoCoDAD',               'MoCoDAD_s0_b_main_sig0'),
]

def load(tag):
    f = f'{RES}/raw/scores/{tag}.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return d['scores'].astype(float), d['labels'].astype(int)

# ---------- summary.csv : mean (+ std) across seeds, per (model,variant,sigma) ----------
def summary():
    rows = list(csv.DictReader(open(f'{RES}/raw/metrics.csv')))
    acc = {}   # (model,variant,sigma,metric) -> {seed: value}
    for r in rows:
        if r.get('fold', 'none') != 'none':      # LOCO fold rows belong to §5.4, not the main/E2 aggregates
            continue
        try:
            acc.setdefault((r['model'], r['variant'], r['sigma'], r['metric']), {})[r['seed']] = float(r['value'])
        except ValueError:
            pass
    keys = ['frame_roc_auc', 'frame_pr_auc', 'base_rate', 'eer', 'event_roc_auc',
            'recall_at_fpr0.01', 'recall_at_fpr0.05', 'recall_at_fpr0.1',
            'fpr_at_thr_fpr0.05', 'median_latency_at_fpr0.05']
    combos = sorted({(m, v, s) for (m, v, s, _) in acc}, key=lambda x: (x[0], x[1], int(x[2])))
    with open(f'{RES}/aggregated/summary.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        head = ['model', 'variant', 'sigma', 'n_seeds']
        for k in keys:
            head += [k + '_mean', k + '_std']
        w.writerow(head)
        for (m, v, s) in combos:
            if int(s) not in (0, 8):
                continue
            ns = max((len(acc.get((m, v, s, k), {})) for k in keys), default=0)
            row = [m, v, s, ns]
            for k in keys:
                vals = list(acc.get((m, v, s, k), {}).values())
                if vals:
                    row += [f'{np.mean(vals):.4f}', f'{(np.std(vals, ddof=1) if len(vals) > 1 else 0):.4f}']
                else:
                    row += ['', '']
            w.writerow(row)
    print('wrote aggregated/summary.csv (seed means +/- std)')

# ---------- ROC + PR curves ----------
def curves():
    have = [(lab, load(tag)) for lab, tag in MODELS]
    have = [(lab, d) for lab, d in have if d is not None]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for lab, (s, y) in have:
        fpr, tpr, _ = roc_curve(y, s)
        ax[0].plot(fpr, tpr, lw=1.6, label=f'{lab} ({roc_auc_score(y, s):.3f})')
        pr, rc, _ = precision_recall_curve(y, s)
        ax[1].plot(rc, pr, lw=1.6, label=f'{lab} ({average_precision_score(y, s):.3f})')
    br = float(np.mean(have[0][1][1]))
    ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax[1].axhline(br, ls='--', c='k', lw=0.8, label=f'chance (base rate {br:.3f})')
    ax[0].set(xlabel='FPR', ylabel='TPR', title='ROC — split (b) test, n=1, sigma=0')
    ax[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall')
    for a in ax:
        a.legend(fontsize=7, loc='lower right'); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{RES}/figures/roc_pr_curves.pdf'); plt.close(fig)
    print('wrote figures/roc_pr_curves.pdf')

# ---------- forest plot: per-seed mean AUC ± 95% t-CI, and Δ vs STG-NF ----------
def forest():
    ps = {r['model']: r for r in csv.DictReader(open(f'{RES}/stats/perseed.csv'))}
    dl = {r['model_vs_STGNF']: r for r in csv.DictReader(open(f'{RES}/stats/delong.csv'))} \
        if os.path.exists(f'{RES}/stats/delong.csv') else {}
    if BASELINE not in ps:
        print('skip forest (no baseline seeds)'); return
    order = [lab for lab, _ in MODELS if lab in ps]
    y = np.arange(len(order))
    means = np.array([float(ps[m]['mean']) for m in order])
    lo = np.array([float(ps[m]['ci95_lo']) for m in order])
    hi = np.array([float(ps[m]['ci95_hi']) for m in order])
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.9 + 0.55 * len(order)))
    # left: absolute mean ROC-AUC ± 95% t-CI over 4 seeds
    ax[0].errorbar(means, y, xerr=[means - lo, hi - means], fmt='o', capsize=4, color='#1f77b4')
    for i, m in enumerate(order):
        seeds = ps[m]['auc_per_seed']
        ax[0].scatter([float(x) for x in seeds.split(';')], [i] * len(seeds.split(';')),
                      s=14, color='#888', zorder=3)
    ax[0].axvline(0.5, c='k', ls='--', lw=.8)
    base_mean = float(ps[BASELINE]['mean'])
    ax[0].axvline(base_mean, c='#d62728', ls=':', lw=1, label=f'STG-NF mean ({base_mean:.3f})')
    ax[0].set_yticks(y); ax[0].set_yticklabels(order)
    ax[0].set_xlabel('frame ROC-AUC (4-seed mean ± 95% t-CI; grey = per-seed)')
    ax[0].set_title('Per-model, split (b), σ=0'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, axis='x')
    # right: Δ vs STG-NF — per-seed Δ + DeLong(seed0) CI
    contemp = [m for m in order if m != BASELINE]
    yy = np.arange(len(contemp))
    for i, m in enumerate(contemp):
        w = {r['model_vs_STGNF']: r for r in csv.DictReader(open(f'{RES}/stats/wilcoxon.csv'))}[m]
        dd = [float(x) for x in w['per_seed_dAUC'].split(';')]
        ax[1].scatter(dd, [i] * len(dd), s=16, color='#888', zorder=3)
        ax[1].scatter([np.mean(dd)], [i], s=60, marker='D', color='#1f77b4', zorder=4)
        if m in dl:
            r = dl[m]
            ax[1].plot([float(r['ci95_lo']), float(r['ci95_hi'])], [i - .18, i - .18], lw=2, color='#2ca02c')
    ax[1].axvline(0, c='k', lw=1)
    ax[1].axvline(0.03, c='r', ls=':', lw=1, label='±0.03 margin'); ax[1].axvline(-0.03, c='r', ls=':', lw=1)
    ax[1].set_yticks(yy); ax[1].set_yticklabels(contemp)
    ax[1].set_xlabel('Δ frame ROC-AUC vs STG-NF  (grey=per-seed, ◆=mean, green=DeLong seed-0 CI)')
    ax[1].set_title('vs STG-NF'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis='x')
    fig.tight_layout(); fig.savefig(f'{RES}/figures/forest_auc_diff.pdf'); plt.close(fig)
    print('wrote figures/forest_auc_diff.pdf')

# ---------- E3 accuracy-efficiency Pareto ----------
def pareto():
    ef = f'{RES}/stats/efficiency.csv'
    ps = f'{RES}/stats/perseed.csv'
    if not (os.path.exists(ef) and os.path.exists(ps)):
        print('skip pareto'); return
    eff = {r['model'].split(' ')[0]: r for r in csv.DictReader(open(ef))}
    acc = {r['model']: r for r in csv.DictReader(open(ps))}
    pts = []
    for key, disp in [('STG-NF', 'STG-NF'), ('COSKAD', 'COSKAD'), ('MoCoDAD', 'MoCoDAD')]:
        if key in eff and disp in acc:
            pts.append((disp, float(eff[key]['windows_per_s']), float(acc[disp]['mean']),
                        float(acc[disp]['ci95_hi']) - float(acc[disp]['mean']),
                        int(eff[key]['params'])))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, wps, m, err, npar in pts:
        ax.errorbar(wps, m, yerr=err, fmt='o', ms=8, capsize=4)
        ax.annotate(f'{name}\n{npar/1000:.1f}K params', (wps, m),
                    textcoords='offset points', xytext=(8, 6), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('throughput  (windows / s, CPU 1-thread, batch 1)  — higher = faster')
    ax.set_ylabel('frame ROC-AUC  (4-seed mean ± 95% t-CI, split b, σ=0)')
    ax.set_title('E3 — accuracy vs efficiency Pareto')
    ax.grid(alpha=.3, which='both')
    fig.tight_layout(); fig.savefig(f'{RES}/figures/efficiency_pareto.pdf'); plt.close(fig)
    print('wrote figures/efficiency_pareto.pdf')


# ---------- E1 LOCO heatmap ----------
def loco():
    rows = [r for r in csv.DictReader(open(f'{RES}/raw/metrics.csv'))
            if r['fold'].startswith('f') and r['metric'] == 'frame_roc_auc' and r['sigma'] == '0']
    if not rows:
        print('skip loco (no fold rows)'); return
    d = {}
    for r in rows:
        d.setdefault((r['model'], r['fold']), []).append(float(r['value']))
    models = ['STG-NF', 'COSKAD', 'MoCoDAD']
    folds = [f'f{k}' for k in range(1, 7)]
    M = np.array([[np.mean(d.get((m, fk), [np.nan])) for fk in folds] for m in models])
    # write csv
    with open(f'{RES}/stats/loco_matrix.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model'] + [f'holdout_C{k}' for k in range(1, 7)] + ['loco_mean', 'min', 'max', 'folds_below_0.5'])
        for i, m in enumerate(models):
            row = M[i]
            w.writerow([m] + [f'{x:.4f}' for x in row] +
                       [f'{np.nanmean(row):.4f}', f'{np.nanmin(row):.4f}', f'{np.nanmax(row):.4f}', int((row < 0.5).sum())])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(M, cmap='RdYlGn', vmin=0.4, vmax=0.9, aspect='auto')
    ax.set_xticks(range(6)); ax.set_xticklabels([f'hold C{k}' for k in range(1, 7)])
    ax.set_yticks(range(3)); ax.set_yticklabels(models)
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=9,
                    color='k')
    ax.set_title('E1 — leave-one-camera-out frame ROC-AUC (4-seed mean, σ=0)')
    fig.colorbar(im, ax=ax, shrink=.8, label='ROC-AUC')
    fig.tight_layout(); fig.savefig(f'{RES}/figures/loco_heatmap.pdf'); plt.close(fig)
    print('wrote figures/loco_heatmap.pdf + stats/loco_matrix.csv')


# ---------- E2-noise degradation curves ----------
def e2noise():
    CONDS = ['gauss1', 'gauss2', 'gauss4', 'gauss8', 'jdrop05', 'jdrop10', 'jdrop20', 'fdrop05', 'fdrop10']
    CLEAN = {'STG-NF': 0.6604, 'COSKAD': 0.5990, 'MoCoDAD': 0.7364}
    rows = [r for r in csv.DictReader(open(f'{RES}/raw/metrics.csv'))
            if r['metric'] == 'frame_roc_auc' and r['sigma'] == '0' and r['fold'] == 'none'
            and r['variant'] in CONDS]
    if not rows:
        print('skip e2noise (no rows)'); return
    d = {}
    for r in rows:
        d.setdefault((r['model'], r['variant']), []).append(float(r['value']))
    models = ['STG-NF', 'COSKAD', 'MoCoDAD']
    with open(f'{RES}/stats/e2noise_matrix.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'clean'] + CONDS)
        for m in models:
            w.writerow([m, f'{CLEAN[m]:.4f}'] +
                       [f'{np.mean(d.get((m, c), [np.nan])):.4f}' for c in CONDS])
    groups = [('Gaussian px', ['gauss1', 'gauss2', 'gauss4', 'gauss8'], [1, 2, 4, 8]),
              ('joint dropout p', ['jdrop05', 'jdrop10', 'jdrop20'], [.05, .10, .20]),
              ('frame dropout p', ['fdrop05', 'fdrop10'], [.05, .10])]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    col = {'STG-NF': '#d62728', 'COSKAD': '#1f77b4', 'MoCoDAD': '#2ca02c'}
    for j, (title, conds, xs) in enumerate(groups):
        for m in models:
            ys = [np.mean(d.get((m, c), [np.nan])) for c in conds]
            es = [np.std(d.get((m, c), [np.nan]), ddof=1) if len(d.get((m, c), [])) > 1 else 0 for c in conds]
            ax[j].errorbar([0] + xs, [CLEAN[m]] + ys, yerr=[0] + es, marker='o', ms=4,
                           capsize=3, color=col[m], label=m)
        ax[j].axhline(0.5, c='k', ls='--', lw=.8)
        ax[j].set_title(title); ax[j].set_xlabel(title); ax[j].grid(alpha=.3)
    ax[0].set_ylabel('frame ROC-AUC (4-seed mean ± std, σ=0)')
    ax[0].legend(fontsize=8)
    fig.suptitle('E2-noise — robustness to eval-time keypoint degradation (x=0 is clean)')
    fig.tight_layout(); fig.savefig(f'{RES}/figures/e2noise_curves.pdf'); plt.close(fig)
    print('wrote figures/e2noise_curves.pdf + stats/e2noise_matrix.csv')


# ---------- E2-window + E2-norm (retraining arms) ----------
def e2retrain():
    rows = [r for r in csv.DictReader(open(f'{RES}/raw/metrics.csv'))
            if r['metric'] == 'frame_roc_auc' and r['sigma'] == '0' and r['fold'] == 'none']
    def mean_of(model, variant):
        v = [float(r['value']) for r in rows if r['model'] == model and r['variant'] == variant]
        return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)) if v else (np.nan, np.nan, 0)
    MAIN = {'STG-NF': 0.6604, 'COSKAD': 0.5990, 'MoCoDAD': 0.7364}
    # --- E2-window ---
    Ts = [12, 24, 32, 48]
    with open(f'{RES}/stats/e2window_matrix.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'T12', 'T24_main', 'T32', 'T48'])
        for m in ['STG-NF', 'COSKAD']:
            r = [m]
            for T in Ts:
                r.append(f'{MAIN[m]:.4f}' if T == 24 else f'{mean_of(m, f"T{T}")[0]:.4f}')
            w.writerow(r)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, c in [('STG-NF', '#d62728'), ('COSKAD', '#1f77b4')]:
        ys = [MAIN[m] if T == 24 else mean_of(m, f'T{T}')[0] for T in Ts]
        es = [0 if T == 24 else mean_of(m, f'T{T}')[1] for T in Ts]
        ax.errorbar(Ts, ys, yerr=es, marker='o', capsize=3, color=c, label=m)
    ax.axhline(0.5, c='k', ls='--', lw=.8)
    ax.set_xlabel('window length T (frames), stride T/2  [T=24 = main run, native stride]')
    ax.set_ylabel('frame ROC-AUC (4-seed mean ± std, σ=0)')
    ax.set_title('E2-window — MoCoDAD excluded (native seg_len 6 architecturally fixed)')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{RES}/figures/e2window_curves.pdf'); plt.close(fig)
    # --- E2-norm ---
    NORM = {'STG-NF': [('default (norm)', None), ('raw / unnormalised', 'norm_raw')],
            'COSKAD': [('markovitz (default)', None), ('stan', 'norm_stan')],
            'MoCoDAD': [('robust (default)', None), ('markovitz', 'norm_markovitz'), ('stan', 'norm_stan')]}
    with open(f'{RES}/stats/e2norm_matrix.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'scheme', 'roc_auc_mean', 'std', 'n_seeds', 'delta_vs_default'])
        for m, schemes in NORM.items():
            for label, var in schemes:
                mu, sd, n = (MAIN[m], 0.0, 4) if var is None else mean_of(m, var)
                w.writerow([m, label, f'{mu:.4f}', f'{sd:.4f}', n, f'{mu - MAIN[m]:+.4f}'])
    print('wrote figures/e2window_curves.pdf + stats/e2{window,norm}_matrix.csv')


if __name__ == '__main__':
    os.makedirs(f'{RES}/figures', exist_ok=True)
    summary(); curves(); forest(); pareto(); loco(); e2noise(); e2retrain()
