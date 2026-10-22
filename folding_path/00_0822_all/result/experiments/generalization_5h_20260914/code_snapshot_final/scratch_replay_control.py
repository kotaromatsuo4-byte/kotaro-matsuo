"""Matched old-data replay from random initialization, without source pretraining.

This separates pretraining initialization from access to old labels during target
adaptation. Batch draws, source datasets and replay weights match pretrained
replay; scratch LR reuses the earlier validation-selected target-only rate.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from graph_transfer import GraphModalStudent,SpatialGeometry
from transfer_study import ModalStudent,tensors,evaluate_model
from replay_transfer import batch_loss
from supervised_folding import BASE,Geometry,write_csv


def run(root,structure,seeds,split,lr_override=None):
    seven=Geometry(BASE/'model/seven_bar_linkage');torus=structure=='torus'
    g=SpatialGeometry(BASE/'model/4unit_torus') if torus else seven;folder=root/'torus' if torus else root
    lr=(.003 if torus else .001) if lr_override is None else lr_override
    weight=.25 if torus else .5;batch=32 if torus else 64;source_batch=16 if torus else 32
    support=dict(np.load(folder/'support_data.npz'));mask=support['path_id']<16
    target=tensors({k:v[mask] for k,v in support.items()},g.scale)
    output=folder/f'scratch_replay{weight:g}_lr{lr:g}_u2000_{split}';output.mkdir(exist_ok=True)
    for seed in seeds:
        sourcefile=root/'source_data.npz'
        if torus:
            selected=torch.load(root/f'graph_source_dagger_seed{seed}_best.pt',weights_only=True)
            if selected['round']: sourcefile=root/f'graph_dagger_seed{seed}'/f'data_round{selected["round"]}.npz'
        source=tensors(dict(np.load(sourcefile)),seven.scale)
        torch.manual_seed(seed);model=GraphModalStudent() if torus else ModalStudent()
        prefix=f'seed{seed}_scratch_n16';saved=output/f'{prefix}.pt'
        if not saved.exists():
            target_rng=torch.Generator().manual_seed(seed+200);source_rng=torch.Generator().manual_seed(seed+10200)
            optimizer=torch.optim.Adam(model.parameters(),lr=lr);recent=[];logs=[];start=time.monotonic();model.train()
            for update in range(1,2001):
                if torus: model.set_geometry(g)
                tl,dl,wl=batch_loss(model,target,torch.randint(len(target['inputs']),(batch,),generator=target_rng))
                if torus: model.set_geometry(seven)
                sl,sdl,swl=batch_loss(model,source,torch.randint(len(source['inputs']),(source_batch,),generator=source_rng))
                optimizer.zero_grad();(tl+weight*sl).backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                recent.append([float(v.detach()) for v in (dl,wl,sdl,swl)])
                if update%100==0:
                    avg=np.mean(recent,0);recent=[]
                    logs.append(dict(update=update,direction_loss=avg[0],width_loss=avg[1],source_direction_loss=avg[2],source_width_loss=avg[3],elapsed=time.monotonic()-start))
            write_csv(output/f'{prefix}_learning.csv',logs)
            torch.save(dict(model=model.state_dict(),seed=seed,arm='scratch',updates=2000,budget=16,replay_weight=weight,lr=lr,
                target_batch=batch,source_batch=source_batch,source_data=str(sourcefile),source_pretraining_updates=0),saved)
        model.load_state_dict(torch.load(saved,weights_only=True)['model']);model.eval()
        tasks=[('seven',seven,root,(2.,3.,4.))] if not torus else [('torus',g,folder,(2.,)),('seven',seven,root,(2.,3.))]
        for name,geometry,taskfolder,goals in tasks:
            destination=output/f'{prefix}_{name}_evaluation.csv'
            if destination.exists(): continue
            if torus: model.set_geometry(geometry)
            starts=np.load(taskfolder/'splits.npz')[split];rows=[]
            for goal in goals: rows.extend(evaluate_model(geometry,model,starts,goal))
            write_csv(destination,[dict(seed=seed,arm='scratch',budget=16,labels=int(mask.sum()),structure=name,**r) for r in rows])
            print('scratch replay',structure,seed,name,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in goals},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('structure',choices=('seven','torus'));p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503,5504,5505])
    p.add_argument('--split',choices=('validation','test'),default='test');p.add_argument('--lr',type=float)
    a=p.parse_args();run(a.root,a.structure,a.seeds,a.split,a.lr)
