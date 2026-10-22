"""Adapt with nearest-branch D4 teacher labels; NN still receives only state/modes/D.

Separate dataset condition. Learning rates and replay coefficient reuse previous
validation choices, without further tuning to this teacher or to test outcomes.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from transfer_study import ModalStudent,train_fixed,evaluate_model
from replay_transfer import train_replay
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds,budgets,split):
    g=Geometry(BASE/'model/seven_bar_linkage');source=dict(np.load(root/'source_data.npz'))
    support=dict(np.load(root/'nearest_branch_teacher'/'support_data.npz'));starts=np.load(root/'splits.npz')[split]
    output=root/f'nearest_branch_u2000_{split}';output.mkdir(exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed);model=ModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        learned=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for budget in budgets:
            mask=support['path_id']<budget;data={k:v[mask] for k,v in support.items()}
            for arm in ('scratch','pretrained','replay'):
                prefix=f'seed{seed}_{arm}_n{budget:02d}';result=output/f'{prefix}_evaluation.csv'
                if result.exists(): continue
                model.load_state_dict(initial if arm=='scratch' else learned)
                if arm=='replay': logs=train_replay(model,data,source,g.scale,2000,seed+200,.5)
                else: logs=train_fixed(model,data,g.scale,2000,seed+200,lr=.001 if arm=='scratch' else .0003)
                write_csv(output/f'{prefix}_learning.csv',logs)
                torch.save(dict(model=model.state_dict(),seed=seed,arm=arm,budget=budget,teacher='nearest of three D4 targets',
                    target_labels=int(mask.sum()),updates=2000,lr=.001 if arm=='scratch' else .0003,replay_weight=.5 if arm=='replay' else 0.),output/f'{prefix}.pt')
                rows=[]
                for goal in (2.,3.,4.): rows.extend(evaluate_model(g,model,starts,goal))
                write_csv(result,[dict(seed=seed,arm=arm,budget=budget,labels=int(mask.sum()),**r) for r in rows])
                print('nearest branch NN',split,prefix,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in (2.,3.,4.)},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--budgets',type=int,nargs='+',default=[4,16])
    p.add_argument('--split',choices=('validation','test'),default='validation');a=p.parse_args();run(a.root,a.seeds,a.budgets,a.split)
