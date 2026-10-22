"""Separate teacher-reference evaluation and preregistered wider-start stress test.

All generated teacher labels here are evaluation only and must never enter
source/support training archives.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from graph_transfer import SpatialGeometry,GraphModalStudent
from transfer_study import path_rollout,target_d4,evaluate_model
from supervised_folding import BASE,Geometry,load_targets,generate_starts,write_csv


def teachers(root):
    output=root/'teacher_evaluation';output.mkdir(exist_ok=True)
    for structure in ('seven','torus'):
        target_root=root if structure=='seven' else root/'torus'
        g=Geometry(BASE/'model/seven_bar_linkage') if structure=='seven' else SpatialGeometry(BASE/'model/4unit_torus')
        targets=load_targets(g) if structure=='seven' else {2.:np.loadtxt(target_root/'target.dat',delimiter=',')}
        if structure=='seven': targets[4.]=target_d4()
        splits=np.load(target_root/'splits.npz')
        for split in ('validation','test'):
            filename=output/f'{structure}_{split}.csv'
            if filename.exists(): continue
            rows=[]
            for goal,target in targets.items():
                for i,initial in enumerate(splits[split]):
                    _,metrics=path_rollout(g,initial,goal,target=target)
                    rows.append(dict(structure=structure,split=split,start_id=i,goal=goal,**metrics))
                print('teacher evaluation',structure,split,goal,np.mean([r['success'] for r in rows if r['goal']==goal]),flush=True)
            write_csv(filename,rows)


def wider(root):
    target_root=root/'torus';output=target_root/'wider_starts';output.mkdir(exist_ok=True)
    g=SpatialGeometry(BASE/'model/4unit_torus')
    saved=output/'starts.npz'
    if not saved.exists():
        original=np.load(target_root/'splits.npz')
        avoid=np.concatenate([original[k] for k in original.files])
        starts=generate_starts(g,16,np.random.default_rng(66004),100,300,avoid)
        np.savez_compressed(saved,test=starts)
        write_csv(output/'initial_metrics.csv',[dict(start_id=i,**g.metrics(x,2.)) for i,x in enumerate(starts)])
    starts=np.load(saved)['test']
    if not (output/'teacher.csv').exists():
        target=np.loadtxt(target_root/'target.dat',delimiter=',');rows=[]
        for i,initial in enumerate(starts):
            _,metrics=path_rollout(g,initial,2.,target=target)
            rows.append(dict(start_id=i,**metrics))
        write_csv(output/'teacher.csv',rows)
        print('wider torus teacher',np.mean([r['success'] for r in rows]),flush=True)


def evaluate_wider(root,seeds,condition='baseline'):
    target_root=root/'torus';parent=target_root/'wider_starts';output=parent/condition;output.mkdir(exist_ok=True)
    g=SpatialGeometry(BASE/'model/4unit_torus');starts=np.load(parent/'starts.npz')['test']
    settings={
        'baseline':('graph_src5000_u2000_test',(1,16),('scratch','pretrained')),
        'tuned_scratch':('graph_src5000_u2000_test_lr0.003',(1,16),('scratch',)),
        'dagger':('graph_src5000_u2000_test_dagger',(0,16),('pretrained',)),
        'mixed':('graph_src5000_u2000_test_mixed',(0,1,16),('pretrained',)),
        'mixed_zero':('graph_src5000_u2000_test_mixed',(0,),('pretrained',)),
        'mixed_static_zero':('graph_src5000_u2000_test_mixed_static',(0,),('pretrained',)),
        'dagger16':('graph_src5000_u2000_test_dagger',(16,),('pretrained',)),
        'replay16':('graph_dagger_replay0.25_u2000_test',(16,),('pretrained',)),
    }
    directory,budgets,arms=settings[condition]
    support=np.load(target_root/'support_data.npz')
    for seed in seeds:
        for budget in budgets:
            for arm in arms:
                prefix=f'seed{seed}_{arm}_n{budget:02d}'
                if (output/f'{prefix}_evaluation.csv').exists(): continue
                checkpoint=target_root/directory/f'{prefix}.pt'
                if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
                model=GraphModalStudent();model.set_geometry(g)
                model.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
                rows=evaluate_model(g,model,starts,2.)
                write_csv(output/f'{prefix}_evaluation.csv',[dict(seed=seed,budget=budget,arm=arm,
                    labels=int((support['path_id']<budget).sum()),**r) for r in rows])
                print('wider torus',prefix,np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('teachers','wider','evaluate-wider'))
    p.add_argument('root',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503,5504,5505])
    p.add_argument('--condition',choices=('baseline','tuned_scratch','dagger','mixed','mixed_zero','mixed_static_zero','dagger16','replay16'),default='baseline')
    a=p.parse_args()
    if a.command=='teachers': teachers(a.root)
    elif a.command=='wider': wider(a.root)
    else: evaluate_wider(a.root,a.seeds,a.condition)
