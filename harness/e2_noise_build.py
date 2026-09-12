"""Build one E2-noise test set: perturbed pickles + a perturbed converted tree.

Usage: e2_noise_build.py <cond> <seed> <outroot>
  <outroot>/pkls/{video}.pkl          perturbed split-(b) TEST pickles (STG-NF, COSKAD
                                      pick these up via POSELIFT_PKL_OVERRIDE)
  <outroot>/conv/b_test/{pose,trajectories,gt,test_frame_mask}/   (MoCoDAD staging source)

Train + val are never perturbed. GT is copied straight through (labels unchanged).
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
from io_poselift import load_split
import perturb

SPL = os.path.expanduser('~/poselift-study/results/splits')


def main(cond, seed, outroot):
    assert cond in perturb.COND, f'unknown condition {cond}'
    outroot = os.path.expanduser(outroot)
    vids = [d['video'] for d in load_split(f'{SPL}/split_b_test.json')]
    pkl_dir = f'{outroot}/pkls'
    n = perturb.write_perturbed_pkls(vids, cond, seed, pkl_dir)
    print(f'[{cond} s{seed}] wrote {n} perturbed test pkls -> {pkl_dir}')

    # perturbed converted tree for MoCoDAD (reuses convert.py unchanged, via the override)
    os.environ['POSELIFT_PKL_OVERRIDE'] = pkl_dir
    import convert
    m = convert.convert_split('b_test', vids, f'{outroot}/conv')
    print(f'[{cond} s{seed}] converted {m["n_videos"]} test clips, '
          f'{sum(x["anomaly_frames"] for x in m["videos"])} anomaly frames -> {outroot}/conv/b_test')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
