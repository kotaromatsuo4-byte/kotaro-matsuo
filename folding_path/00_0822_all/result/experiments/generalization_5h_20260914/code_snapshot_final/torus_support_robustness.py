"""Repeat one-trajectory torus transfer with four different fixed teacher paths."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent,SpatialGeometry
from transfer_study import train_fixed,evaluate_model
from supervised_folding import BASE,write_csv


def run(root,seeds):
    folder=root/'torus';output=folder/'support_path_robustness';output.mkdir(exist_ok=True)
    g=SpatialGeometry(BASE/'model/4unit_torus');support=dict(np.load(folder/'support_data.npz'));starts=np.load(folder/'splits.npz')['test']
    for seed in seeds:
        torch.manual_seed(seed);model=GraphModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        learned=torch.load(root/f'graph_source_seed{seed}_u5000.pt',weights_only=True)['model'];model.set_geometry(g)
        for path_id in range(4):
            mask=support['path_id']==path_id;data={k:v[mask] for k,v in support.items()}
            for arm,weights,lr in [('scratch',initial,.003),('pretrained',learned,.0003)]:
                prefix=f'seed{seed}_{arm}_path{path_id:02d}';destination=output/f'{prefix}_evaluation.csv'
                if destination.exists(): continue
                if path_id==0:
                    suffix='_lr0.003' if arm=='scratch' else ''
                    with (folder/f'graph_src5000_u2000_test{suffix}'/f'seed{seed}_{arm}_n01_evaluation.csv').open() as stream:
                        rows=list(csv.DictReader(stream))
                    rows=[{k:v for k,v in r.items() if k not in ('seed','arm','budget','labels')} for r in rows]
                else:
                    model.load_state_dict(weights);logs=train_fixed(model,data,g.scale,2000,seed+200,batch=32,lr=lr)
                    write_csv(output/f'{prefix}_learning.csv',logs)
                    torch.save(dict(model=model.state_dict(),seed=seed,arm=arm,support_path=path_id,updates=2000,lr=lr),output/f'{prefix}.pt')
                    rows=evaluate_model(g,model,starts,2.)
                write_csv(destination,[dict(seed=seed,arm=arm,budget=1,labels=int(mask.sum()),support_path=path_id,**r) for r in rows])
                print('torus support path',seed,path_id,arm,'labels',int(mask.sum()),'success',np.mean([int(r['success']) for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503]);a=p.parse_args();run(a.root,a.seeds)
