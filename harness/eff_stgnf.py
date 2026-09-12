import os, sys, glob, json
sys.path.insert(0, os.path.expanduser('~/poselift-study/harness'))
sys.path.insert(0, os.path.expanduser('~/PoseLift/STG-NF'))
import torch
from models.STG_NF.model_pose import STG_NF
from eff_profile import profile

# reconstruct model_args from the saved run's args.json
aj = sorted(glob.glob(os.path.expanduser('~/poselift-study/results/model_runs/stgnf/exp_b_s0/PoseLift/*/args.json')))[-1]
A = json.load(open(aj))
# STG_NF pose_shape = (C, T, V) ; STG-NF uses kp18 -> V=18, C=2 (no confidence), T=seg_len
pose_shape = (2, A.get('seg_len', 24), 18)
margs = dict(pose_shape=pose_shape, hidden_channels=A.get('model_hidden_dim', 0) or 0,
             K=A.get('K', 8), L=A.get('L', 1), R=A.get('R', 3.0),
             actnorm_scale=1.0, flow_permutation=A.get('flow_permutation', 'permute'),
             flow_coupling='affine', LU_decomposed=False, learn_top=False,
             edge_importance=A.get('edge_importance', False),
             temporal_kernel_size=A.get('temporal_kernel') or None,
             strategy=A.get('adj_strategy', 'uniform'), max_hops=A.get('max_hops', 8),
             device='cpu')
try:
    m = STG_NF(**margs).cpu()
except TypeError as e:
    # fall back: minimal required kwargs
    m = STG_NF(pose_shape=pose_shape, K=margs['K'], L=margs['L'], R=margs['R'],
               device='cpu')
ck = sorted(glob.glob(os.path.expanduser('~/poselift-study/results/model_runs/stgnf/exp_b_s0/PoseLift/*/*.tar')))[-1]
try:
    st = torch.load(ck, map_location='cpu', weights_only=False)
except TypeError:
    st = torch.load(ck, map_location='cpu')
st = st.get('model', st.get('state_dict', st))
m.load_state_dict(st, strict=False)

x = torch.randn(1, 2, pose_shape[1], pose_shape[2])
def fwd():
    return m(x)
try:
    with torch.no_grad():
        fwd()
except Exception as e:
    # STG_NF forward may want (x, cond) or dict
    def fwd():
        return m(x, torch.zeros(1))
profile('STG-NF (normalizing flow)', m, x, forward=fwd,
        note=f'window (2,{pose_shape[1]},{pose_shape[2]}); 1 flow forward (nll)')
