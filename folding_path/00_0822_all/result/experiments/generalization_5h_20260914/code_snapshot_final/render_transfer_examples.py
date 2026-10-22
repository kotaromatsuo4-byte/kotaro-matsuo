"""Illustrative held-out folding paths with an explicit, reproducible selection rule."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graph_transfer import GraphModalStudent,SpatialGeometry
from transfer_study import ModalStudent,path_rollout
from supervised_folding import BASE,Geometry,write_csv


def run(root,structure,seed,budget):
    torus=structure=='torus';folder=root/'torus' if torus else root
    goal=2. if torus else 4.
    g=SpatialGeometry(BASE/'model/4unit_torus') if torus else Geometry(BASE/'model/seven_bar_linkage')
    conditions=([('Scratch, tuned LR','graph_src5000_u2000_test_lr0.003','scratch','#555555'),
                 ('D2/D3 + source aggregation','graph_src5000_u2000_test_dagger','pretrained','#cb722a')] if torus else
                [('Scratch, tuned LR','modal_tuned_lr0.001_u2000_test','scratch','#555555'),
                 ('D2/D3 pretrained','modal_u2000_src10000_test','pretrained','#2266aa')])
    outcomes=[]
    for name,directory,arm,color in conditions:
        with (folder/directory/f'seed{seed}_{arm}_n{budget:02d}_evaluation.csv').open() as f:
            outcomes.append({int(r['start_id']):int(r['success']) for r in csv.DictReader(f) if float(r['goal'])==goal})
    candidates=[i for i in sorted(outcomes[1]) if outcomes[1][i] and not outcomes[0][i]]
    rule='First test index with pretrained success and scratch failure, for illustration only.'
    if not candidates:
        candidates=[i for i in sorted(outcomes[1]) if outcomes[1][i]] or [0]
        rule='First pretrained-success test index, otherwise index 0, for illustration only.'
    index=candidates[0];initial=np.load(folder/'splits.npz')['test'][index]
    output=folder/'examples';output.mkdir(exist_ok=True)
    paths=[];results=[]
    for name,directory,arm,color in conditions:
        model=GraphModalStudent() if torus else ModalStudent()
        if torus: model.set_geometry(g)
        checkpoint=folder/directory/f'seed{seed}_{arm}_n{budget:02d}.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
        path,metrics=path_rollout(g,initial,goal,model=model)
        if metrics['success']!=outcomes[len(paths)][index]: raise AssertionError('Checkpoint replay differs from recorded evaluation')
        paths.append(path);results.append(dict(condition=name,start_id=index,**metrics))
        np.savez_compressed(output/f'seed{seed}_{arm}_n{budget:02d}_start{index:02d}.npz',nodes=path)
    write_csv(output/f'seed{seed}_n{budget:02d}_metrics.csv',results)
    (output/f'seed{seed}_n{budget:02d}_selection.json').write_text(json.dumps(dict(seed=seed,budget=budget,start_id=index,
        selection=rule,warning='Selected illustrative example, not an estimate of average performance.'),indent=2),encoding='utf-8')
    fig=plt.figure(figsize=(13,9),constrained_layout=True);grid=fig.add_gridspec(3,4,height_ratios=[1,1,.85])
    coordinates=np.concatenate(paths);low=coordinates.min((0,1));high=coordinates.max((0,1))
    center=(low+high)/2;radius=max(high-low)*.55
    for row,(path,(name,directory,arm,color)) in enumerate(zip(paths,conditions)):
        for col,fraction in enumerate((0.,1/3,2/3,1.)):
            step=round((len(path)-1)*fraction);node=path[step]
            axis=fig.add_subplot(grid[row,col],projection='3d' if torus else None)
            for a,b in g.member:
                points=node[[a,b]]
                if torus: axis.plot(*points.T,color=color,linewidth=1.3)
                else: axis.plot(points[:,0],points[:,2],'-o',color=color,linewidth=2,markersize=3)
            if torus:
                axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
                axis.set_box_aspect((1,1,1));axis.view_init(elev=20,azim=-55)
            else:
                axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-radius,center[2]+radius),aspect='equal')
            axis.set_axis_off();axis.set_title(f'{name if col==0 else ""}\nstep {step}',fontsize=9)
    flat=fig.add_subplot(grid[2,:2]);spread=fig.add_subplot(grid[2,2:])
    for path,(name,directory,arm,color) in zip(paths,conditions):
        metrics=[g.metrics(node,goal) for node in path]
        flat.plot([max(r['flatness'],1e-12) for r in metrics],label=name,color=color)
        spread.plot([r['spread'] for r in metrics],label=name,color=color)
    flat.axhline(1e-3,color='black',linestyle='--',linewidth=.8);flat.set(yscale='log',xlabel='Folding steps',ylabel='Normalized flatness')
    spread.axhspan(goal-.05,goal+.05,color='green',alpha=.12);spread.axhline(goal,color='black',linestyle='--',linewidth=.8)
    spread.set(xlabel='Folding steps',ylabel='Projected spread D')
    for axis in (flat,spread): axis.grid(alpha=.2)
    flat.legend(fontsize=8)
    fig.suptitle(f'{structure}: seed {seed}, {budget} support paths, held-out start {index} (selected example)',fontsize=13)
    for extension in ('png','pdf'): fig.savefig(output/f'seed{seed}_n{budget:02d}_example.{extension}',dpi=180)
    plt.close(fig);print(results,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('structure',choices=('seven','torus'));p.add_argument('--seed',type=int,default=5501);p.add_argument('--budget',type=int,default=16)
    a=p.parse_args();run(a.root,a.structure,a.seed,a.budget)
