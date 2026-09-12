"""E3 efficiency — generic CPU profiler. Each per-model wrapper builds (model, dummy_input)
and calls profile(). Writes one row to results/stats/efficiency.csv.

Metrics: parameter count; FLOPs/window (ptflops, best-effort); CPU single-thread batch-1
latency = median of 200 timed forwards after 20 warmup -> windows/s; peak RSS (MB).
"""
import os, time, csv, resource
import numpy as np
import torch

OUT = os.path.expanduser('~/poselift-study/results/stats/efficiency.csv')

def profile(name, model, dummy_input, note='', forward=None):
    torch.set_num_threads(1)
    model.eval()
    fwd = forward or (lambda: model(dummy_input))
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for _ in range(20):
            fwd()
        T = []
        for _ in range(200):
            t0 = time.perf_counter(); fwd(); T.append((time.perf_counter() - t0) * 1e3)
    med = float(np.median(T)); p90 = float(np.percentile(T, 90))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    flops = 'nan'
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, tuple(dummy_input.shape[1:]),
                                            as_strings=False, print_per_layer_stat=False, verbose=False)
        flops = f'{2 * macs:.0f}'
    except Exception as e:
        note = (note + f' | ptflops_failed:{type(e).__name__}').strip(' |')
    row = dict(model=name, params=n_params, flops_per_window=flops,
               cpu_ms_median=f'{med:.3f}', cpu_ms_p90=f'{p90:.3f}',
               windows_per_s=f'{1000/med:.1f}', peak_rss_mb=f'{rss:.0f}', note=note)
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print('RESULT', row)
    return row
