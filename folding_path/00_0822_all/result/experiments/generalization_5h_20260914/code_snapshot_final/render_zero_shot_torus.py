"""Replay and illustrate a genuine, explicitly selected bar-only zero-shot path."""
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
from matplotlib.animation import FuncAnimation,PillowWriter
from graph_transfer import GraphModalStudent,SpatialGeometry
from transfer_study import path_rollout
from supervised_folding import BASE,write_csv


def run(root):
    folder=root/'torus';seed=5501
    with (folder/'graph_src5000_u2000_test_mixed'/f'seed{seed}_pretrained_n00_evaluation.csv').open() as f:
        rows=list(csv.DictReader(f))
    index=min(int(r['start_id']) for r in rows if int(r['success']))
    checkpoint=root/f'graph_source_mixed_seed{seed}_u5000.pt'
    g=SpatialGeometry(BASE/'model/4unit_torus');model=GraphModalStudent();model.set_geometry(g)
    model.load_state_dict(torch.load(checkpoint,weights_only=True)['model']);model.eval()
    initial=np.load(folder/'splits.npz')['test'][index]
    path,metrics=path_rollout(g,initial,2.,model=model)
    assert metrics['success']==1,'The original bar-only checkpoint must reproduce the success.'
    output=folder/'zero_shot_example';output.mkdir(exist_ok=True)
    np.savez_compressed(output/'path.npz',nodes=path)
    (output/'provenance.json').write_text(json.dumps(dict(seed=seed,start_id=index,source_checkpoint=str(checkpoint),
        source_structures=['five_bar_linkage','seven_bar_linkage'],source_goals=[2,3],target_updates=0,target_labels=0,
        selection='First successful test index at seed 5501; selected illustration, not an average.',
        caveat='Architecture exploration used torus validation. Five-seed normal-test mean is 16.25%, two seeds have successes.',
        **metrics),indent=2),encoding='utf-8')
    history=[dict(step=i,**g.metrics(node,2.)) for i,node in enumerate(path)]
    write_csv(output/'metrics.csv',history)
    low=path.min((0,1));high=path.max((0,1));center=(low+high)/2;radius=max(high-low)*.55
    def draw(axis,node,step):
        axis.clear()
        for a,b in g.member: axis.plot(*node[[a,b]].T,color='#2266aa',linewidth=1.4)
        axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        axis.set_box_aspect((1,1,1));axis.view_init(elev=23,azim=-55);axis.set_axis_off();axis.set_title(f'Step {step}',fontsize=10)
    fig=plt.figure(figsize=(13,6.5),constrained_layout=True);grid=fig.add_gridspec(2,4,height_ratios=[1.3,1])
    for col,fraction in enumerate((0,1/3,2/3,1)):
        step=round((len(path)-1)*fraction);draw(fig.add_subplot(grid[0,col],projection='3d'),path[step],step)
    ax=fig.add_subplot(grid[1,:2]);ax.plot([r['flatness'] for r in history]);ax.axhline(1e-3,color='black',linestyle='--');ax.set(yscale='log',xlabel='Folding steps',ylabel='Normalized flatness');ax.grid(alpha=.2)
    ax=fig.add_subplot(grid[1,2:]);ax.plot([r['spread'] for r in history]);ax.axhspan(1.95,2.05,color='green',alpha=.15);ax.set(xlabel='Folding steps',ylabel='Projected spread D');ax.grid(alpha=.2)
    fig.suptitle(f'Bar-only NN folds 4-unit torus: no target training (seed {seed}, held-out start {index})')
    fig.supxlabel('Selected success example. Across five training seeds: 16.25% success; successes in two of five seeds.',fontsize=9)
    for extension in ('png','pdf'): fig.savefig(output/f'zero_shot.{extension}',dpi=180)
    plt.close(fig)
    fig=plt.figure(figsize=(6,6));axis=fig.add_subplot(111,projection='3d')
    frames=np.unique(np.round(np.linspace(0,len(path)-1,min(45,len(path)))).astype(int))
    def animate(frame):
        draw(axis,path[frame],frame)
        fig.suptitle('Bar-only NN: selected zero-shot torus example\nNo torus labels or NN updates',fontsize=11)
    animation=FuncAnimation(fig,animate,frames=frames,interval=100)
    animation.save(output/'zero_shot.gif',writer=PillowWriter(fps=10),dpi=100);plt.close(fig)
    print('zero-shot replay verified',seed,index,metrics,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    run(p.parse_args().root)
