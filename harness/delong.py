"""DeLong (1988) fast AUC + variance / two-correlated-ROC test.
Implementation after Sun & Xu (2014) fast algorithm. Unit-tested against
sklearn.roc_auc_score in tests/test_harness.py.
"""
import numpy as np
from scipy import stats

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out

def _structural(preds_pos, preds_neg):
    m, n = len(preds_pos), len(preds_neg)
    pos = preds_pos[None, :]; neg = preds_neg[None, :]
    k = 1
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    cov = s01 / m + s10 / n
    return aucs[0], float(cov)

def auc_variance(y_true, score):
    y = np.asarray(y_true).astype(bool); s = np.asarray(score, float)
    a, v = _structural(s[y], s[~y])
    return a, v

def delong_test(y_true, score_a, score_b):
    """H0: AUC_a == AUC_b for two scorers on the SAME samples.
    Returns dict(auc_a, auc_b, diff, ci95=(lo,hi), z, p) (two-sided)."""
    y = np.asarray(y_true).astype(bool)
    sa = np.asarray(score_a, float); sb = np.asarray(score_b, float)
    pos = np.stack([sa[y], sb[y]]); neg = np.stack([sa[~y], sb[~y]])
    m, n = pos.shape[1], neg.shape[1]
    tx = np.stack([_midrank(pos[r]) for r in range(2)])
    ty = np.stack([_midrank(neg[r]) for r in range(2)])
    tz = np.stack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(2)])
    aucs = tz[:, :m].sum(1) / m - (m + 1) / 2.0
    aucs = aucs / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    S = s01 / m + s10 / n
    d = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(var) if var > 0 else 0.0
    z = d / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z)) if se > 0 else (0.0 if d != 0 else 1.0)
    lo, hi = (d - 1.96 * se, d + 1.96 * se) if se > 0 else (d, d)
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=float(d),
                ci95=(float(lo), float(hi)), z=float(z), p=float(p), se=float(se))
