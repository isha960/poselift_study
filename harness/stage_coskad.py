"""Stage a split into COSKAD's expected layout (dataset_choice='HR-STC', JSON path).
  {root}/pose/training/tracked_person/{scene}_{clip}_x.json
  {root}/pose/testing/tracked_person/{scene}_{clip}_x.json
  {root}/testing/test_frame_mask/{scene}_{clip}.npy
  {root}/validating/test_frame_mask/{scene}_{clip}.npy   (copy of testing; some code paths read it)
JSON: {"<pid>": {"<frame:06d>": {"keypoints":[x,y,c ...51], "scores": null}}}  (6-digit frame keys)
Usage: stage_coskad.py <root> <train_split_json> <test_split_json>
"""
import sys, os, json, shutil
import numpy as np
from io_poselift import load_video, gt_labels, load_split

def write_pose(video, dst):
    n, frames = load_video(video)
    tr = {}
    for fr, ppl in frames.items():
        for pid, (bb, kp) in ppl.items():
            tr.setdefault(str(int(pid)), {})[f'{int(fr):06d}'] = {
                "keypoints": [float(x) for x in kp.reshape(-1)], "scores": None}
    s, c = video.split('_')
    json.dump(tr, open(f'{dst}/{int(s)}_{int(c)}_x.json', 'w'))
    return n

def write_gt(video, n, dst):
    y, _, _ = gt_labels(video, n)
    s, c = video.split('_')
    np.save(f'{dst}/{int(s)}_{int(c)}.npy', y.astype(np.int8))

def stage(root, train_json, test_json):
    root = os.path.expanduser(root)
    d_tr = f'{root}/pose/training/tracked_person'
    d_te = f'{root}/pose/testing/tracked_person'
    d_gt = f'{root}/testing/test_frame_mask'
    d_gv = f'{root}/validating/test_frame_mask'
    for d in (d_tr, d_te, d_gt, d_gv):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    tr = [x['video'] for x in load_split(os.path.expanduser(train_json))]
    te = [x['video'] for x in load_split(os.path.expanduser(test_json))]
    for v in tr:
        write_pose(v, d_tr)
    for v in te:
        n = write_pose(v, d_te); write_gt(v, n, d_gt); write_gt(v, n, d_gv)
    print(f'COSKAD staged at {root}: train {len(tr)}, test {len(te)}')

if __name__ == '__main__':
    stage(sys.argv[1], sys.argv[2], sys.argv[3])
