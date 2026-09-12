"""E1 — build 6 leave-one-camera-out splits (pre-registration sec.7).
Fold k: train = every NORMAL video from cameras != k ; test = every video from camera k.
No val (E1 reports threshold-free metrics: ROC-AUC / PR-AUC / EER / event-ROC-AUC).
Writes results/splits/loco_f{1..6}_{train,test}.json in the same schema as split_b_*.json.
"""
import json, os
inv = json.load(open(os.path.expanduser('~/poselift-study/results/splits/_video_inventory.json')))
OUT = os.path.expanduser('~/poselift-study/results/splits')

def rec(v):
    return dict(video=v['video'], camera=v['cam'], n_frames=v['n_frames'],
               n_tracks=v['n_tracks'], anomaly=bool(v['anomaly_frames']),
               anomaly_frames=int(v['anomaly_frames'] or 0),
               track_ids=[f"{v['video']}::{t}" for t in v['track_ids']])

normal = [v for v in inv if not (v['anomaly_frames'] and v['anomaly_frames'] > 0)]
rows = []
for k in '123456':
    tr = [rec(v) for v in normal if v['cam'] != k]
    te = [rec(v) for v in inv if v['cam'] == k]
    for name, lst in [('train', tr), ('test', te)]:
        json.dump({'fold': int(k), 'held_out_camera': int(k), 'n_videos': len(lst), 'videos': lst},
                  open(f'{OUT}/loco_f{k}_{name}.json', 'w'), indent=1)
    tr_tracks = set(t for r in tr for t in r['track_ids'])
    te_tracks = set(t for r in te for t in r['track_ids'])
    assert not (tr_tracks & te_tracks), f'fold {k}: track leakage'
    rows.append((k, len(tr), len(te), sum(r['anomaly'] for r in te), sum(r['anomaly_frames'] for r in te)))

print('fold | train_norm_vids | test_vids | test_anom_vids | test_anom_frames | track_leak')
for k, ntr, nte, na, af in rows:
    print(f'  {k}  |  {ntr:3d}  |  {nte:3d}  |  {na:2d}  |  {af:5d}  |  0')
