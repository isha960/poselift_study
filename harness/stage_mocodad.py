"""Arrange the pre-converted CSV trajectories + GT into MoCoDAD's expected tree
(normalization_strategy='robust' -> data_of_combined_model reads {data_dir}/{sub}/trajectories/).
  {root}/training/trajectories/{video}/{pid}.csv        rows: frame,x0,y0,...,x16,y16
  {root}/testing/trajectories/{video}/{pid}.csv
  {root}/validating/trajectories/{video}/{pid}.csv
  {root}/testing/test_frame_mask/{video}.npy
  {root}/validating/test_frame_mask/{video}.npy
Source: ~/poselift-study/data_converted/b_{train,val,test}/{trajectories,gt}
Usage: stage_mocodad.py <root>
"""
import sys, os, shutil

CV = os.path.expanduser('~/poselift-study/data_converted')
MAP = {'training': 'b_train', 'testing': 'b_test', 'validating': 'b_val'}

def stage(root):
    root = os.path.expanduser(root)
    shutil.rmtree(root, ignore_errors=True)
    for sub, src in MAP.items():
        # trajectories -- MoCoDAD parses folder as "{scene}-{clip}" (DASH); our videos are "{scene}_{clip}"
        s_tr = f'{CV}/{src}/trajectories'
        d_tr = f'{root}/{sub}/trajectories'
        os.makedirs(d_tr, exist_ok=True)
        for vfolder in os.listdir(s_tr):
            dash = vfolder.replace('_', '-', 1)
            shutil.copytree(f'{s_tr}/{vfolder}', f'{d_tr}/{dash}')
        # gt masks (testing + validating need them; training doesn't)
        if sub in ('testing', 'validating'):
            d_gt = f'{root}/{sub}/test_frame_mask'
            os.makedirs(d_gt, exist_ok=True)
            for fn in os.listdir(f'{CV}/{src}/gt'):
                shutil.copy(f'{CV}/{src}/gt/{fn}', f'{d_gt}/{fn}')
        n_clips = len(os.listdir(d_tr))
        print(f'  {sub}: {n_clips} clip folders')
    print(f'MoCoDAD staged at {root}')

if __name__ == '__main__':
    stage(sys.argv[1])
