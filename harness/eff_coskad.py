import os, sys, glob, yaml
sys.path.insert(0, os.path.expanduser('~/poselift-study/harness'))
sys.path.insert(0, os.path.expanduser('~/poselift-study/external/COSKAD'))
import torch
from models.stse.stse_hidden_hypersphere import STSE
from eff_profile import profile

cfg = yaml.safe_load(open(os.path.expanduser('~/poselift-study/configs/coskad_b_s0.yaml')))
T = cfg['dataset_seg_len']; V = 17
m = STSE(c_in=2, h_dim=cfg['h_dim'], latent_dim=cfg['latent_dim'], n_frames=T, n_joints=V,
         dropout=cfg['dropout'], channels=cfg['channels'], encoder_type='STS_GCN').cpu()
ck = sorted(glob.glob(os.path.expanduser('~/poselift-study/results/model_runs/coskad/HR-STC/eucl_b_s0/*.ckpt')))[-1]
try:
    _ck = torch.load(ck, map_location='cpu', weights_only=False)
except TypeError:
    _ck = torch.load(ck, map_location='cpu')
sd = {k[6:]: v for k, v in _ck['state_dict'].items() if k.startswith('model.')}
info = m.load_state_dict(sd, strict=False)

x = None
for shp in [(1, 2, T, V), (1, 2, T, V, 1), (1, T, V, 2)]:
    try:
        xi = torch.randn(*shp)
        with torch.no_grad():
            m(xi)
        x = xi
        break
    except Exception as e:
        last = e
if x is None:
    raise SystemExit(f'COSKAD shape probe failed: {last}')
profile('COSKAD (STSE encoder)', m, x,
        note=f'window (2,{T},{V}); missing_keys={len(info.missing_keys)}; anomaly score = dist to hypersphere centre (1 fwd)')
