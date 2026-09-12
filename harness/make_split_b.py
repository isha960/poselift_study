"""Stage 1 - build the pre-registered split (b) and verify zero track leakage.

Rules (from results/preregistration.md sec.6):
  * 41 videos with any anomaly frame  -> TEST ONLY
  * 110 all-normal videos             -> 60/15/25 train/val/test, split BY VIDEO,
                                         stratified by camera (C1..C6 in each part)
  * globally-unique track id = "<video>::<person_id>"; verify train INTERSECT
    {val,test} == empty (true by construction of a by-video split, checked anyway)
  * split RNG seed = 20260828
Outputs: results/splits/split_b_{train,val,test}.json  + split_b_counts.csv
"""
import pickle, numpy as np, glob, os, json, csv, random
from collections import Counter, defaultdict

ROOT = os.path.expanduser('~/PoseLift/Pickle_files')
OUT  = os.path.expanduser('~/poselift-study/results/splits')
SEED = 20260828
FRAC = dict(train=0.60, val=0.15, test=0.25)

def vid(f): return os.path.basename(f)[:-4]
def cam(v): return v.split('_')[0]

pkls = sorted(glob.glob(f'{ROOT}/Train/*.pkl')) + sorted(glob.glob(f'{ROOT}/Test/*.pkl'))
gt   = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(f'{ROOT}/GT/*.npy')}

def tracks(f):
    d = pickle.load(open(f, 'rb'))
    s = set()
    for _, ppl in d.items():
        s |= set(int(p) for p in ppl.keys())
    return sorted(s), len(d)

info = {}
for f in pkls:
    v = vid(f)
    tids, nfr = tracks(f)
    g = gt.get(v)
    info[v] = dict(path=f, cam=cam(v), n_frames=nfr, n_tracks=len(tids),
                   gtracks=[f"{v}::{t}" for t in tids],
                   anomaly=(g is not None and int(g.sum()) > 0),
                   anomaly_frames=(int(g.sum()) if g is not None else 0),
                   has_gt=(g is not None))

anomaly_vids = sorted(v for v, d in info.items() if d['anomaly'])
normal_vids  = sorted(v for v, d in info.items() if not d['anomaly'])
assert len(anomaly_vids) == 41, len(anomaly_vids)
assert len(normal_vids) == 110, len(normal_vids)

# camera-stratified by-video split of the NORMAL videos
rng = random.Random(SEED)
split = dict(train=[], val=[], test=[])
by_cam = defaultdict(list)
for v in normal_vids:
    by_cam[info[v]['cam']].append(v)
for c in sorted(by_cam):
    vs = sorted(by_cam[c]); rng.shuffle(vs)
    n = len(vs); n_tr = round(n*FRAC['train']); n_va = round(n*FRAC['val'])
    # guarantee >=1 per part per camera when the camera has >=3 normal videos
    if n >= 3:
        n_tr = max(1, min(n_tr, n-2)); n_va = max(1, min(n_va, n-1-n_tr))
    split['train'] += vs[:n_tr]
    split['val']   += vs[n_tr:n_tr+n_va]
    split['test']  += vs[n_tr+n_va:]
split['test'] += anomaly_vids            # all anomaly videos -> test
for k in split: split[k] = sorted(split[k])

# ---- leakage check on globally-unique track ids ----
tset = {k: set(t for v in split[k] for t in info[v]['gtracks']) for k in split}
leak_tr_va = tset['train'] & tset['val']
leak_tr_te = tset['train'] & tset['test']
leak_va_te = tset['val']   & tset['test']
vid_overlap = (set(split['train']) & set(split['val'])) | (set(split['train']) & set(split['test'])) | (set(split['val']) & set(split['test']))
assert not vid_overlap, f"VIDEO overlap: {vid_overlap}"
assert not leak_tr_va and not leak_tr_te and not leak_va_te, "TRACK-ID leakage detected"

# ---- write split files ----
os.makedirs(OUT, exist_ok=True)
for k in split:
    payload = {"seed": SEED, "n_videos": len(split[k]),
               "videos": [{"video": v, "camera": info[v]['cam'],
                           "n_frames": info[v]['n_frames'],
                           "n_tracks": info[v]['n_tracks'],
                           "anomaly": info[v]['anomaly'],
                           "anomaly_frames": info[v]['anomaly_frames'],
                           "track_ids": info[v]['gtracks']} for v in split[k]]}
    json.dump(payload, open(f'{OUT}/split_b_{k}.json', 'w'), indent=1)

# ---- counts table ----
def agg(vs):
    fr = sum(info[v]['n_frames'] for v in vs)
    an = sum(info[v]['anomaly_frames'] for v in vs)
    tk = sum(info[v]['n_tracks'] for v in vs)
    return len(vs), fr, an, tk
with open(f'{OUT}/split_b_counts.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['split', 'camera', 'videos', 'frames', 'anomaly_frames', 'tracks', 'base_rate'])
    for k in ['train', 'val', 'test']:
        for c in ['1','2','3','4','5','6','ALL']:
            vs = [v for v in split[k] if (c == 'ALL' or info[v]['cam'] == c)]
            nv, fr, an, tk = agg(vs)
            w.writerow([k, c, nv, fr, an, tk, f'{(an/fr if fr else 0):.4f}'])

print("split (b) written. video counts:",
      {k: len(v) for k, v in split.items()},
      "| leakage: train-val", len(leak_tr_va), "train-test", len(leak_tr_te), "val-test", len(leak_va_te))
print("\nsplit_b_counts.csv:")
print(open(f'{OUT}/split_b_counts.csv').read())
