"""Isolated-process peak-RSS + latency measurement for MoCoDAD at n_gen=1 only (Table 5.6
row 4). eff_mocodad.py measures n_gen=1 then n_gen=50 in the SAME process, so its reported
722MB peak RSS is the process-wide peak (dominated by the n_gen=50 pass), not a clean n_gen=1
figure. This script repeats only the n_gen=1 timing in a fresh process for a clean reading.
"""
import os, sys, glob, yaml, time, resource, csv, argparse
import numpy as np
sys.path.insert(0, os.path.expanduser('~/poselift-study/external/MoCoDAD'))
import torch
torch.set_num_threads(1)
from models.mocodad import MoCoDAD
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader

cfg = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/mocodad_b_s0_eval.yaml')))
cfg['split'] = 'test'; cfg['batch_size'] = 1
a = init_args(argparse.Namespace(**cfg))
ck = sorted(glob.glob(os.path.expanduser('~/poselift-study/results/model_runs/mocodad/HR-STC/b_s0/*.ckpt')))[-1]
lit = MoCoDAD.load_from_checkpoint(ck, args=a, map_location='cpu').eval().cpu()
lit.device_ = 'cpu'
for h in ('on_test_start', 'on_test_epoch_start'):
    getattr(lit, h, lambda: None)()
if not hasattr(lit, '_test_output_list'):
    lit._test_output_list = []

ds, loader, _, _ = get_dataset_and_loader(a, split='test')
batch = next(iter(loader))
batch = [b.cpu() if torch.is_tensor(b) else b for b in batch]

lit.n_generated_samples = 1
with torch.no_grad():
    for _ in range(2):
        lit.test_step(batch, 0)
    Tms = []
    for _ in range(8):
        t0 = time.perf_counter(); lit.test_step(batch, 0); Tms.append((time.perf_counter() - t0) * 1e3)
ms1 = float(np.median(Tms))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

row = dict(model='MoCoDAD (diffusion, n_gen=1)', params=sum(p.numel() for p in lit.parameters()),
           flops_per_window='nan', cpu_ms_median=f'{ms1:.1f}', cpu_ms_p90='',
           windows_per_s=f'{1000/ms1:.2f}', peak_rss_mb=f'{rss:.0f}',
           note=f'isolated-process n_gen=1 only (cf. eff_mocodad.py process-wide 722MB, '
                f'which is dominated by the n_gen=50 pass run in the same process)')
OUT = os.path.expanduser('~/poselift-study/results/stats/efficiency.csv')
with open(OUT, 'a', newline='') as fh:
    csv.DictWriter(fh, fieldnames=list(row.keys())).writerow(row)
print('RESULT', row)
