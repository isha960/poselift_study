"""Stage a split into the STG-NF working copy's expected layout.
STG-NF (this working copy) requires:
  pose file  : {scene:02d}_{clip:04d}_alphapose_tracked_person.json
               { "<pid>": { "<frame>": {"keypoints":[x,y,c ...51], "scores": null} } }
               frame keys UNPADDED str(int)  (matches the proven Aug-23 split-a staging)
  gt file    : {scene:02d}_{clip:04d}.npy   (int8 frame labels; 1 = anomaly)
Hardcoded read roots in utils/scoring_utils.py:
  data/PoseLift/pose/{train,test}   and   data/PoseLift/gt/test_frame_mask/
Usage: python stage_stgnf.py <split_json_for_train> <split_json_for_test>
"""
import sys, os, json, shutil
import numpy as np
from io_poselift import load_video, gt_labels, load_split

STG = os.path.expanduser('~/PoseLift/STG-NF/data/PoseLift')

def fid(video):
    s, c = video.split('_')
    return f'{int(s):02d}_{int(c):04d}'

def write_pose(video, dst_dir):
    n, frames = load_video(video)
    tr = {}
    for fr, ppl in frames.items():
        for pid, (bb, kp) in ppl.items():
            tr.setdefault(str(int(pid)), {})[str(int(fr))] = {
                "keypoints": [float(x) for x in kp.reshape(-1)], "scores": None}
    json.dump(tr, open(f'{dst_dir}/{fid(video)}_alphapose_tracked_person.json', 'w'))
    return n

def write_gt(video, n, dst_dir):
    y, _, _ = gt_labels(video, n)
    np.save(f'{dst_dir}/{fid(video)}.npy', y.astype(np.int8))

def stage(train_json, test_json):
    for sub in ('pose/train', 'pose/test', 'gt/test_frame_mask'):
        d = f'{STG}/{sub}'
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    tr = [d['video'] for d in load_split(train_json)]
    te = [d['video'] for d in load_split(test_json)]
    for v in tr:
        write_pose(v, f'{STG}/pose/train')
    for v in te:
        n = write_pose(v, f'{STG}/pose/test')
        write_gt(v, n, f'{STG}/gt/test_frame_mask')
    print(f'staged: train {len(tr)} pose, test {len(te)} pose + {len(te)} gt')

if __name__ == '__main__':
    stage(os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2]))
