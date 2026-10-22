"""One graph network retains seven-bar tasks while adapting to torus.

Target draws/updates match graph_transfer; replay adds computation, no new labels.
The coefficient .25 is a fixed exploratory choice, evaluated on validation first.
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
from replay_transfer import batch_loss
from transfer_study import tensors,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds,split,weight,updates,budget):
    source_g=Geometry(BASE/'model/seven_bar_linkage');target_g=SpatialGeometry(BASE/'model/4unit_torus')
    target_root=root/'torus';support=dict(np.load(target_root/'support_data.npz'))
    mask=support['path_id']<budget;target=tensors({k:v[mask] for k,v in support.items()},target_g.scale)
    output=target_root/f'graph_dagger_replay{weight:g}_u{updates}_{split}';output.mkdir(exist_ok=True)
    for seed in seeds:
        checkpoint=torch.load(root/f'graph_source_dagger_seed{seed}_best.pt',weights_only=True)
        datafile=(root/f'graph_dagger_seed{seed}'/f'data_round{checkpoint["round"]}.npz'
                  if checkpoint['round'] else root/'source_data.npz')
        source=tensors(dict(np.load(datafile)),source_g.scale)
        model=GraphModalStudent();model.load_state_dict(checkpoint['model'])
        prefix=f'seed{seed}_pretrained_n{budget:02d}';saved=output/f'{prefix}.pt'
        if not saved.exists():
            target_rng=torch.Generator().manual_seed(seed+200);source_rng=torch.Generator().manual_seed(seed+10200)
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);logs=[];recent=[];start=time.monotonic()
            model.train()
            for update in range(1,updates+1):
                model.set_geometry(target_g)
                target_loss,dl,wl=batch_loss(model,target,torch.randint(len(target['inputs']),(32,),generator=target_rng))
                model.set_geometry(source_g)
                source_loss,sdl,swl=batch_loss(model,source,torch.randint(len(source['inputs']),(16,),generator=source_rng))
                optimizer.zero_grad();(target_loss+weight*source_loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                recent.append([float(v.detach()) for v in (dl,wl,sdl,swl)])
                if update%100==0 or update==updates:
                    avg=np.mean(recent,0);recent=[]
                    logs.append(dict(update=update,direction_loss=avg[0],width_loss=avg[1],source_direction_loss=avg[2],source_width_loss=avg[3],elapsed=time.monotonic()-start))
            write_csv(output/f'{prefix}_learning.csv',logs)
            torch.save(dict(model=model.state_dict(),seed=seed,budget=budget,updates=updates,replay_weight=weight,
                source_round=checkpoint['round'],source_labels=len(source['inputs']),target_labels=int(mask.sum()),
                target_batch=32,source_batch=16,source_data=str(datafile)),saved)
        model.load_state_dict(torch.load(saved,weights_only=True)['model']);model.eval()
        for structure,g,folder,goals in [('torus',target_g,target_root,(2.,)),('seven',source_g,root,(2.,3.))]:
            destination=output/f'{prefix}_{structure}_evaluation.csv'
            if destination.exists(): continue
            model.set_geometry(g);starts=np.load(folder/'splits.npz')[split];rows=[]
            for goal in goals: rows.extend(evaluate_model(g,model,starts,goal))
            write_csv(destination,[dict(seed=seed,arm='pretrained',budget=budget,labels=int(mask.sum()),structure=structure,**r) for r in rows])
            print('graph replay',split,seed,structure,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in goals},flush=True)
        # Same target-adapted, no-replay checkpoint: measure source retention separately.
        baseline=target_root/f'graph_src5000_u{updates}_{split}_dagger'/f'{prefix}.pt'
        destination=output/f'{prefix}_baseline_seven_evaluation.csv'
        if baseline.exists() and not destination.exists():
            model.load_state_dict(torch.load(baseline,weights_only=True)['model']);model.set_geometry(source_g)
            rows=[]
            for goal in (2.,3.): rows.extend(evaluate_model(source_g,model,np.load(root/'splits.npz')[split],goal))
            write_csv(destination,[dict(seed=seed,arm='pretrained',budget=budget,labels=int(mask.sum()),structure='seven',**r) for r in rows])
            print('graph no replay retention',seed,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in (2.,3.)},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--weight',type=float,default=.25);p.add_argument('--updates',type=int,default=2000);p.add_argument('--budget',type=int,default=16)
    a=p.parse_args();run(a.root,a.seeds,a.split,a.weight,a.updates,a.budget)
