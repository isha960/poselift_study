"""Metrics module (pre-registration sec.1). Every function returns a scalar.
Accuracy is deliberately NOT provided.

Inputs are always the concatenated-across-videos frame arrays for one split's TEST
set, plus (for event metrics) a per-video description.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def frame_roc_auc(y, s):
    return float(roc_auc_score(y, s))

def frame_pr_auc(y, s):
    return float(average_precision_score(y, s))

def base_rate(y):
    return float(np.mean(y))

def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)

def threshold_at_fpr(y, s, target_fpr):
    """Largest threshold whose val FPR <= target (for applying to test)."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])

def threshold_at_fpr_neg_only(neg_scores, target_fpr):
    """Calibrate a threshold from an all-NORMAL (all-negative) validation set:
    the value exceeded by exactly `target_fpr` of normal frames = (1-fpr) quantile."""
    return float(np.quantile(np.asarray(neg_scores, float), 1.0 - target_fpr))

def recall_at_threshold(y, s, thr):
    y = np.asarray(y).astype(bool)
    pred = s >= thr
    tp = np.sum(pred & y); fn = np.sum(~pred & y)
    return float(tp / (tp + fn)) if (tp + fn) else float('nan')

def fpr_at_threshold(y, s, thr):
    y = np.asarray(y).astype(bool)
    pred = s >= thr
    fp = np.sum(pred & ~y); tn = np.sum(~pred & ~y)
    return float(fp / (fp + tn)) if (fp + tn) else float('nan')

def f1_at_threshold(y, s, thr):
    y = np.asarray(y).astype(bool); pred = s >= thr
    tp = np.sum(pred & y); fp = np.sum(pred & ~y); fn = np.sum(~pred & y)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * p * r / (p + r)) if (p + r) else 0.0

# ---------- event / clip level ----------
def event_roc_auc(events):
    """events: list of dict(is_anomaly:bool, clip_max_score:float). One row per test video
    (anomaly videos use the max smoothed score inside the labelled window; normal videos
    use the whole-clip max)."""
    y = np.array([e['is_anomaly'] for e in events], int)
    s = np.array([e['clip_max_score'] for e in events], float)
    if y.min() == y.max():
        return float('nan')
    return float(roc_auc_score(y, s))

def detection_latencies(anomaly_events, thr):
    """anomaly_events: list of dict(onset:int, frame_scores_in_window:1d array starting at onset).
    latency = frames from onset to first score>=thr; NaN if never. Returns list."""
    out = []
    for e in anomaly_events:
        w = np.asarray(e['frame_scores_in_window'], float)
        hit = np.where(w >= thr)[0]
        out.append(float(hit[0]) if len(hit) else float('nan'))
    return out

def median_latency(latencies):
    a = np.asarray(latencies, float)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if len(a) else float('nan')
