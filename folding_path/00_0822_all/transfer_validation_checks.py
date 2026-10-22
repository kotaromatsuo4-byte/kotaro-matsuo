"""Validation-only checks to avoid an under-optimized scratch baseline."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from supervised_folding import BASE,Geometry,write_csv
from graph_transfer import GraphModalStudent
from transfer_study import evaluate_model


def main(root):
    torch.set_num_threads(1)
    g=Geometry(BASE/'model/seven_bar_linkage')
    model=GraphModalStudent()
    model.set_geometry(g)
    model.load_state_dict(torch.load(root/'graph_source_seed5501_u5000.pt',weights_only=True)['model'])
    starts=np.load(root/'splits.npz')['validation']
    rows=[]
    for goal in (2.,3.):
        rows.extend(evaluate_model(g,model,starts,goal))
    write_csv(root/'graph_source_validation.csv',rows)
    print('Graph source success', {goal:np.mean([r['success'] for r in rows if r['goal']==goal]) for goal in (2.,3.)},flush=True)
    for lr,updates in ((.001,2000),(.003,2000),(.0003,8000)):
        command=[sys.executable,str(BASE/'graph_transfer.py'),'study',str(root),
                 '--target','torus','--seeds','5501','--budgets','1','4','16',
                 '--source-updates','5000','--updates',str(updates),'--split','validation',
                 '--adapt-lr',str(lr),'--arms','scratch']
        print('VALIDATION GRID',lr,updates,flush=True)
        subprocess.run(command,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    main(p.parse_args().root.resolve())
