"""Untuned 30-node six-unit torus probe using frozen five/seven-bar source models.

Goals D2/D3 are specified before running. No six-unit labels, checkpoints,
hyperparameter tuning or adaptation are used. Start coverage is only the
provided reference shape; absence of success does not prove goal infeasibility.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graph_transfer import SpatialGeometry,GraphModalStudent
from transfer_study import path_rollout
from supervised_folding import BASE,write_csv


def run(root,seeds):
    g=SpatialGeometry(BASE/'model/6unit_torus');output=root/'six_unit_zero_probe';output.mkdir(exist_ok=True)
    design=dict(source='five/seven-bar mixed; 5,000 updates',target='6unit_torus',nodes=len(g.initial),members=len(g.member),
        source_seeds=seeds,goals=[2.,3.],max_steps=400,initial_shapes=1,adaptation_updates=0,target_teacher_labels=0,
        initial_sha256=hashlib.sha256((g.directory/'node_initial.dat').read_bytes()).hexdigest(),
        limitations='Reference start only; candidate goal feasibility has not been independently established. Report all attempted goals and seeds.')
    (output/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    for seed in seeds:
        model=GraphModalStudent();model.set_geometry(g)
        checkpoint=root/f'graph_source_mixed_seed{seed}_u5000.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
        for goal in (2.,3.):
            prefix=f'seed{seed}_d{goal:g}'
            if (output/f'{prefix}.csv').exists(): continue
            path,metrics=path_rollout(g,g.initial,goal,model=model,max_steps=400)
            np.savez_compressed(output/f'{prefix}.npz',nodes=path)
            write_csv(output/f'{prefix}.csv',[dict(seed=seed,goal=goal,**metrics)])
            print('six-unit zero-shot',seed,goal,metrics,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503,5504,5505])
    a=p.parse_args();run(a.root,a.seeds)
