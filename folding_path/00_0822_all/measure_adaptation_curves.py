"""Validation rollout success versus optimizer updates; final weights are verified.

These post-hoc diagnostic curves do not select checkpoints for the primary test.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replay_transfer import batch_loss
from transfer_study import ModalStudent,tensors,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds):
    output=root/'adaptation_update_curves';output.mkdir(exist_ok=True)
    g=Geometry(BASE/'model/seven_bar_linkage');starts=np.load(root/'splits.npz')['validation']
    support=dict(np.load(root/'support_data.npz'));mask=support['path_id']<16
    target=tensors({k:v[mask] for k,v in support.items()},g.scale);source=tensors(dict(np.load(root/'source_data.npz')),g.scale)
    conditions=[('scratch',.001,0.,'modal_tuned_lr0.001_u2000_test','scratch'),
                ('pretrained',.0003,0.,'modal_u2000_src10000_test','pretrained'),
                ('replay',.0003,.5,'modal_replay0.5_u2000_test','pretrained')]
    for seed in seeds:
        torch.manual_seed(seed);model=ModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        learned=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for name,lr,weight,directory,arm in conditions:
            done=output/f'seed{seed}_{name}_verified.json'
            if done.exists(): continue
            model.load_state_dict(initial if name=='scratch' else learned)
            optimizer=torch.optim.Adam(model.parameters(),lr=lr)
            target_rng=torch.Generator().manual_seed(seed+200);source_rng=torch.Generator().manual_seed(seed+10200)
            rows=[]
            for update in range(2001):
                if update:
                    model.train();loss,_,_=batch_loss(model,target,torch.randint(len(target['inputs']),(64,),generator=target_rng))
                    if weight:
                        extra,_,_=batch_loss(model,source,torch.randint(len(source['inputs']),(32,),generator=source_rng));loss=loss+weight*extra
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if update in (0,100,250,500,1000,2000):
                    model.eval();evaluations=[]
                    for goal in (2.,3.,4.): evaluations.extend(evaluate_model(g,model,starts,goal))
                    rows.extend(dict(seed=seed,condition=name,update=update,budget=16,labels=int(mask.sum()),**r) for r in evaluations)
                    write_csv(output/f'seed{seed}_{name}_evaluation.csv',rows)
                    print('adaptation curve',seed,name,update,{d:np.mean([r['success'] for r in evaluations if r['goal']==d]) for d in (2.,3.,4.)},flush=True)
            final=torch.load(root/directory/f'seed{seed}_{arm}_n16.pt',weights_only=True)['model']
            difference=max(float((v-final[k]).abs().max()) for k,v in model.state_dict().items())
            if difference>1e-6: raise AssertionError(f'Final checkpoint mismatch: {difference}')
            done.write_text(json.dumps(dict(seed=seed,condition=name,max_absolute_weight_difference=difference,
                comparison_checkpoint=str(root/directory/f'seed{seed}_{arm}_n16.pt'),checkpoint_selection=False),indent=2),encoding='utf-8')
    plot(output)


def plot(output):
    rows=[]
    for file in output.glob('seed*_evaluation.csv'):
        with file.open() as f: rows.extend(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
    summary=[]
    for axis,goal in zip(axes,(2.,3.,4.)):
        for condition,color in [('scratch','#444444'),('pretrained','#2266aa'),('replay','#2b9474')]:
            selected=[r for r in rows if r['condition']==condition and float(r['goal'])==goal]
            updates=sorted({int(r['update']) for r in selected});means=[];lows=[];highs=[]
            for update in updates:
                part=[r for r in selected if int(r['update'])==update];seeds=sorted({r['seed'] for r in part})
                rates=[np.mean([int(r['success']) for r in part if r['seed']==seed]) for seed in seeds]
                means.append(np.mean(rates)*100);lows.append(min(rates)*100);highs.append(max(rates)*100)
                summary.append(dict(condition=condition,goal=goal,update=update,seeds=len(seeds),mean_success=np.mean(rates),seed_min=min(rates),seed_max=max(rates)))
            axis.plot(updates,means,'o-',color=color,label=condition)
            axis.fill_between(updates,lows,highs,color=color,alpha=.1)
        axis.set(title=f'D{goal:g}',xlabel='Target optimizer updates',ylabel='Validation success (%)',ylim=(-3,103));axis.grid(alpha=.2)
    axes[0].legend();fig.suptitle('16 D4 teacher paths: acquisition and retention during adaptation')
    fig.supxlabel('Diagnostic only; no checkpoint selection. Lines: seed means; shading: seed range.',fontsize=9)
    for extension in ('png','pdf'): fig.savefig(output/f'adaptation_updates.{extension}',dpi=180)
    plt.close(fig);write_csv(output/'summary.csv',summary)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503]);p.add_argument('--plot-only',action='store_true')
    a=p.parse_args()
    if a.plot_only: plot(a.root/'adaptation_update_curves')
    else: run(a.root,a.seeds)
