"""PoseLift .pkl  ->  the input formats the external repos consume.
Applied identically to split (a) and split (b); output trees are study artefacts.

COCO-JSON  (STG-NF, COSKAD):
    {outdir}/{split}/pose/{video}.json
    = { "<pid>": { "<frame>": {"keypoints": [x,y,c ...51], "scores": null}, ... }, ... }
    frame keys zero-padded to 6 digits so lexicographic == numeric (COSKAD sorts str keys).
CSV-trajectory  (MoCoDAD, normalization_strategy='robust' path -> 34 x/y cols):
    {outdir}/{split}/trajectories/{video}/{pid}.csv   rows: frame,x0,y0,...,x16,y16
GT (all):
    {outdir}/{split}/gt/{video}.npy  (int8 frame labels; normal videos -> zeros)
    {outdir}/{split}/test_frame_mask/{video}.npy  (same, name MoCoDAD/COSKAD expect)
Plus {outdir}/{split}/manifest.json
"""
import os, json, csv
import numpy as np
from io_poselift import load_video, gt_labels

def _video_tracks(video):
    n, frames = load_video(video)
    tr = {}   # pid -> {frame -> kp(17,3)}
    for fr, ppl in frames.items():
        for pid, (bb, kp) in ppl.items():
            tr.setdefault(int(pid), {})[int(fr)] = kp
    return n, tr

def convert_split(name, videos, outdir, pad=6):
    base = os.path.join(os.path.expanduser(outdir), name)
    for sub in ('pose', 'trajectories', 'gt', 'test_frame_mask'):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    manifest = dict(split=name, n_videos=len(videos), videos=[])

    for v in videos:
        n, tr = _video_tracks(v)
        # ---- COCO-JSON ----
        jd = {}
        for pid, fmap in tr.items():
            jd[str(pid)] = {f"{fr:0{pad}d}": {"keypoints": [float(x) for x in kp.reshape(-1)],
                                              "scores": None}
                            for fr, kp in sorted(fmap.items())}
        json.dump(jd, open(os.path.join(base, 'pose', f'{v}.json'), 'w'))
        # ---- CSV-trajectory (x,y only, 34 cols) ----
        vdir = os.path.join(base, 'trajectories', v)
        os.makedirs(vdir, exist_ok=True)
        for pid, fmap in tr.items():
            rows = []
            for fr, kp in sorted(fmap.items()):
                rows.append([fr] + [float(c) for c in kp[:, :2].reshape(-1)])
            with open(os.path.join(vdir, f'{pid}.csv'), 'w', newline='') as fh:
                csv.writer(fh).writerows(rows)
        # ---- GT ----
        y, n_used, note = gt_labels(v, n)
        np.save(os.path.join(base, 'gt', f'{v}.npy'), y.astype(np.int8))
        np.save(os.path.join(base, 'test_frame_mask', f'{v}.npy'), y.astype(np.int8))
        manifest['videos'].append(dict(video=v, pkl_frames=n, scored_frames=int(n_used),
                                       n_tracks=len(tr), anomaly_frames=int(y.sum()),
                                       align_note=note))
    json.dump(manifest, open(os.path.join(base, 'manifest.json'), 'w'), indent=1)
    return manifest


if __name__ == '__main__':
    import sys
    from io_poselift import load_split, split_a
    OUT = os.path.expanduser('~/poselift-study/data_converted')
    SPL = os.path.expanduser('~/poselift-study/results/splits')

    tr_a, te_a = split_a()
    print('split (a) train', convert_split('a_train', tr_a, OUT)['n_videos'],
          '| test', convert_split('a_test', te_a, OUT)['n_videos'])

    for part in ('train', 'val', 'test'):
        vids = [d['video'] for d in load_split(f'{SPL}/split_b_{part}.json')]
        m = convert_split(f'b_{part}', vids, OUT)
        print(f'split (b) {part}: {m["n_videos"]} videos,',
              f'{sum(x["scored_frames"] for x in m["videos"])} scored frames,',
              f'{sum(x["anomaly_frames"] for x in m["videos"])} anomaly frames')
