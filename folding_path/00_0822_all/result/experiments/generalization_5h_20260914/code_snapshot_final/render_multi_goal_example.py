"""One fixed NN and one held-out initial state, three scalar goals."""
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
from transfer_study import ModalStudent,path_rollout
from supervised_folding import BASE,Geometry,write_csv


def run(root,nearest=False):
    seed=5501;budget=4 if nearest else 16
    folder=root/('nearest_branch_u2000_test' if nearest else 'modal_replay0.5_u2000_test')
    arm='replay' if nearest else 'pretrained';prefix=f'seed{seed}_{arm}_n{budget:02d}';checkpoint=folder/f'{prefix}.pt'
    with (folder/f'{prefix}_evaluation.csv').open() as stream: rows=list(csv.DictReader(stream))
    common=set.intersection(*[{int(r['start_id']) for r in rows if float(r['goal'])==goal and int(r['success'])} for goal in (2,3,4)])
    index=min(common);initial=np.load(root/'splits.npz')['test'][index]
    g=Geometry(BASE/'model/seven_bar_linkage');model=ModalStudent();model.load_state_dict(torch.load(checkpoint,weights_only=True)['model']);model.eval()
    output=root/('multi_goal_nearest_example' if nearest else 'multi_goal_example');output.mkdir(exist_ok=True);paths=[];results=[]
    np.savetxt(output/'node_initial.dat',initial,delimiter=',')
    for goal in (2.,3.,4.):
        path,metrics=path_rollout(g,initial,goal,model=model)
        assert metrics['success']==1,'All three goals must reproduce the recorded success'
        paths.append(path);results.append(dict(goal=goal,start_id=index,**metrics))
        np.savez_compressed(output/f'd{goal:g}_path.npz',nodes=path)
    write_csv(output/'results.csv',results)
    (output/'selection.json').write_text(json.dumps(dict(seed=seed,start_id=index,checkpoint=str(checkpoint),
        selection='First held-out initial state successful for all three goals at seed 5501; selected example, not an average.',
        target_teacher_paths=budget,teacher_rule='nearest branch' if nearest else 'fixed target',network_changes_between_goals=False,target_coordinates_at_inference=False),indent=2),encoding='utf-8')
    allnodes=np.concatenate(paths);low=allnodes.min((0,1));high=allnodes.max((0,1));center=(low+high)/2;radius=max((high-low)[[0,2]])*.58
    vertical_radius=max((high[2]-low[2])*.58,radius*.6)
    fig,axes=plt.subplots(3,3,figsize=(11.5,8),constrained_layout=True)
    colors=['#2266aa','#2b9474','#cb722a']
    for row,(goal,path,color) in enumerate(zip((2,3,4),paths,colors)):
        for col,fraction in enumerate((0,.5,1)):
            step=round((len(path)-1)*fraction);node=path[step];axis=axes[row,col]
            for a,b in g.member: axis.plot(node[[a,b],0],node[[a,b],2],'-o',color=color,linewidth=2,markersize=4)
            axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-vertical_radius,center[2]+vertical_radius),aspect='equal')
            axis.set_axis_off();axis.set_title(f'D{goal}: '+('same initial state' if col==0 else f'step {step}'),fontsize=11)
    fig.suptitle(f'One NN, three goals: {budget} D4 teacher paths (seed {seed}, held-out start {index})',fontsize=14)
    fig.supxlabel('Only scalar D changes. Source replay retains D2/D3 after D4 learning. Selected successful example.',fontsize=10)
    for extension in ('png','pdf'): fig.savefig(output/f'multi_goal.{extension}',dpi=180)
    plt.close(fig);print(results,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--nearest-teacher',action='store_true')
    a=p.parse_args();run(a.root,a.nearest_teacher)
