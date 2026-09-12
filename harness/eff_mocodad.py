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
for h in ('on_test_start','on_test_epoch_start'):
    getattr(lit, h, lambda: None)()
if not hasattr(lit,'_test_output_list'): lit._test_output_list=[]
n_total = sum(p.numel() for p in lit.parameters())
n_unet = sum(p.numel() for p in lit.model.parameters())
n_cond = sum(p.numel() for p in lit.condition_encoder.parameters())

ds, loader, _, _ = get_dataset_and_loader(a, split='test')
batch = next(iter(loader))
batch = [b.cpu() if torch.is_tensor(b) else b for b in batch]

def time_gen(n_gen, reps):
    lit.n_generated_samples = n_gen
    with torch.no_grad():
        for _ in range(2):
            lit.test_step(batch, 0)
        Tms = []
        for _ in range(reps):
            t0 = time.perf_counter(); lit.test_step(batch, 0); Tms.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(Tms))

ms1 = time_gen(1, 8)
ms50 = time_gen(50, 4)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
steps = cfg['noise_steps']
row = dict(model='MoCoDAD (diffusion)', params=n_total, flops_per_window='nan',
           cpu_ms_median=f'{ms50:.1f}', cpu_ms_p90='',
           windows_per_s=f'{1000/ms50:.2f}', peak_rss_mb=f'{rss:.0f}',
           note=f'per-window generative inference, CPU 1-thread, batch 1: n_gen=1 -> {ms1:.1f} ms; '
                f'n_gen=50 (paper) -> {ms50:.1f} ms ({1000/ms50:.2f} win/s); noise_steps={steps}; '
                f'U-Net {n_unet} params, condition-encoder {n_cond} params')
OUT = os.path.expanduser('~/poselift-study/results/stats/efficiency.csv')
new = not os.path.exists(OUT)
with open(OUT, 'a', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(row.keys()))
    if new:
        w.writeheader()
    w.writerow(row)
print('RESULT', row)
