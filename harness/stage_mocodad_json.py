"""Stage COCO-JSON pose into MoCoDAD's non-'robust' layout (E2-norm recovery).
  {root}/pose/{training,testing,validating}/tracked_person/{s}_{c}_x.json  (6-digit frame keys)
  {root}/{testing,validating}/test_frame_mask/{s}_{c}.npy
Usage: stage_mocodad_json.py <root>   (uses split_b_{train,test,val}.json)
"""
import sys, os, json, shutil
import numpy as np
from io_poselift import load_video, gt_labels, load_split

SPL = os.path.expanduser('~/poselift-study/results/splits')
MAP = {'training': 'train', 'testing': 'test', 'validating': 'val'}


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


def stage(root):
    root = os.path.expanduser(root)
    for part, key in MAP.items():
        dp = f'{root}/pose/{part}/tracked_person'
        shutil.rmtree(dp, ignore_errors=True); os.makedirs(dp)
        vids = [d['video'] for d in load_split(f'{SPL}/split_b_{key}.json')]
        for v in vids:
            n = write_pose(v, dp)
            if part in ('testing', 'validating'):
                dg = f'{root}/{part}/test_frame_mask'
                os.makedirs(dg, exist_ok=True)
                write_gt(v, n, dg)
        print(f'  {part}: {len(vids)} pose json')
    print(f'MoCoDAD JSON staged at {root}/pose')


if __name__ == '__main__':
    stage(sys.argv[1])
