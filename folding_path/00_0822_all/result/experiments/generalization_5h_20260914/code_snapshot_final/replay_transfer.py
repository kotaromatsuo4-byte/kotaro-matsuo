"""Source replay during D4 adaptation, keeping target batches/updates identical.

Extra source labels are the already collected D2/D3 data. Source replay costs
extra computation and is not counted as free target-data improvement.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transfer_study import ModalStudent, tensors, evaluate_model
from supervised_folding import BASE, Geometry, write_csv


def batch_loss(model,t,indices):
    prediction,width=model(t['inputs'][indices],t['modes'][indices],t['direction'][indices])
    direction=(prediction-t['direction'][indices]).square().sum(-1).mean()
    valid=t['width_valid'][indices]
    width_loss=((width-t['width'][indices]).square()*valid).sum()/valid.sum().clamp(min=1)
    return direction+.2*width_loss,direction,width_loss


def train_replay(model,target,source,scale,updates,seed,replay_weight):
    target,source=tensors(target,scale),tensors(source,scale)
    # Target draws exactly match train_fixed. Replay has a separate generator.
    target_rng=torch.Generator().manual_seed(seed)
    source_rng=torch.Generator().manual_seed(seed+10000)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    logs=[];recent=[];start=time.monotonic()
    model.train()
    for update in range(1,updates+1):
        indices=torch.randint(len(target['inputs']),(64,),generator=target_rng)
        target_loss,dl,wl=batch_loss(model,target,indices)
        indices=torch.randint(len(source['inputs']),(32,),generator=source_rng)
        source_loss,sdl,swl=batch_loss(model,source,indices)
        loss=target_loss+replay_weight*source_loss
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        recent.append([float(x.detach()) for x in (dl,wl,sdl,swl)])
        if update%100==0 or update==updates:
            avg=np.mean(recent,0);recent=[]
            logs.append(dict(update=update,direction_loss=avg[0],width_loss=avg[1],
                             source_direction_loss=avg[2],source_width_loss=avg[3],elapsed=time.monotonic()-start))
    model.eval()
    return logs


def study(root,seeds,budgets,updates,weight,split):
    g=Geometry(BASE/'model/seven_bar_linkage')
    source=dict(np.load(root/'source_data.npz'));support=dict(np.load(root/'support_data.npz'))
    starts=np.load(root/'splits.npz')[split]
    output=root/f'modal_replay{weight:g}_u{updates}_{split}'
    output.mkdir(exist_ok=True)
    for seed in seeds:
        checkpoint=root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt'
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        weights=torch.load(checkpoint,weights_only=True)['model']
        for budget in budgets:
            prefix=f'seed{seed}_pretrained_n{budget:02d}'
            if (output/f'{prefix}_evaluation.csv').exists(): continue
            model=ModalStudent();model.load_state_dict(weights)
            mask=support['path_id']<budget
            data={k:v[mask] for k,v in support.items()}
            logs=train_replay(model,data,source,g.scale,updates,seed+200,weight)
            write_csv(output/f'{prefix}_learning.csv',logs)
            torch.save(dict(model=model.state_dict(),seed=seed,budget=budget,replay_weight=weight),output/f'{prefix}.pt')
            rows=[]
            for goal in (2.,3.,4.): rows.extend(evaluate_model(g,model,starts,goal))
            write_csv(output/f'{prefix}_evaluation.csv',[dict(seed=seed,arm='pretrained',budget=budget,labels=int(mask.sum()),**r) for r in rows])
            print('replay',weight,prefix,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in (2.,3.,4.)},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--budgets',type=int,nargs='+',default=[1,4,16,32])
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--weight',type=float,default=.5)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    a=p.parse_args();study(a.root,a.seeds,a.budgets,a.updates,a.weight,a.split)
