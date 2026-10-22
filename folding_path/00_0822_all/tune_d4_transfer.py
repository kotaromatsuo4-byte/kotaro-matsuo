"""Independently tune D4 adaptation rates on validation; test only after selection."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from transfer_study import ModalStudent,train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds,budgets,lr,split,arms):
    g=Geometry(BASE/'model/seven_bar_linkage');data=dict(np.load(root/'support_data.npz'))
    starts=np.load(root/'splits.npz')[split]
    output=root/f'modal_tuned_lr{lr:g}_u2000_{split}';output.mkdir(exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed);model=ModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        pretrained=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for budget in budgets:
            mask=data['path_id']<budget;target={k:v[mask] for k,v in data.items()}
            for arm,weights in (('scratch',initial),('pretrained',pretrained)):
                if arm not in arms: continue
                prefix=f'seed{seed}_{arm}_n{budget:02d}'
                if (output/f'{prefix}_evaluation.csv').exists(): continue
                model.load_state_dict(weights)
                logs=train_fixed(model,target,g.scale,2000,seed+200,lr=lr)
                write_csv(output/f'{prefix}_learning.csv',logs)
                torch.save(dict(model=model.state_dict(),seed=seed,budget=budget,lr=lr,arm=arm,updates=2000),output/f'{prefix}.pt')
                rows=evaluate_model(g,model,starts,4.)
                write_csv(output/f'{prefix}_evaluation.csv',[dict(seed=seed,arm=arm,budget=budget,labels=int(mask.sum()),**r) for r in rows])
                print('D4 tuned',lr,prefix,'success',np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--budgets',type=int,nargs='+',default=[1,4,16,32])
    p.add_argument('--lr',type=float,nargs='+',default=[.001,.003]);p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--arms',nargs='+',choices=('scratch','pretrained'),default=['scratch','pretrained'])
    a=p.parse_args()
    for lr in a.lr: run(a.root,a.seeds,a.budgets,lr,a.split,a.arms)
