"""Repopulate ONLY MoCoDAD's testing/ subtree from a given converted tree (E2-noise).
Leaves training/ and validating/ untouched (they stay clean).
Usage: stage_mocodad_test.py <mocodad_root> <conv_root>   (conv_root has b_test/{trajectories,gt})
"""
import sys, os, shutil


def stage(root, conv_root):
    root = os.path.expanduser(root); conv = os.path.expanduser(conv_root)
    s_tr = f'{conv}/b_test/trajectories'
    d_tr = f'{root}/testing/trajectories'
    shutil.rmtree(d_tr, ignore_errors=True); os.makedirs(d_tr)
    for vfolder in os.listdir(s_tr):
        dash = vfolder.replace('_', '-', 1)
        shutil.copytree(f'{s_tr}/{vfolder}', f'{d_tr}/{dash}')
    d_gt = f'{root}/testing/test_frame_mask'
    shutil.rmtree(d_gt, ignore_errors=True); os.makedirs(d_gt)
    for fn in os.listdir(f'{conv}/b_test/gt'):
        shutil.copy(f'{conv}/b_test/gt/{fn}', f'{d_gt}/{fn}')
    print(f'  MoCoDAD testing/ repopulated from {conv}/b_test: {len(os.listdir(d_tr))} clip folders')


if __name__ == '__main__':
    stage(sys.argv[1], sys.argv[2])
