"""Validate the joint-learning scratch control and freeze its LR before testing."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scratch_replay_control import run


def study(root,structure):
    torus=structure=='torus';folder=root/'torus' if torus else root;weight=.25 if torus else .5;goal=2. if torus else 4.;candidates=[]
    for lr in (.0003,.001,.003):
        run(root,structure,[5501],'validation',lr)
        file=folder/f'scratch_replay{weight:g}_lr{lr:g}_u2000_validation'/f'seed5501_scratch_n16_{structure}_evaluation.csv'
        with file.open() as stream: rows=[r for r in csv.DictReader(stream) if float(r['goal'])==goal]
        candidates.append(dict(lr=lr,validation_target_success=float(np.mean([int(r['success']) for r in rows]))))
    best=max(candidates,key=lambda r:(r['validation_target_success'],-r['lr']))
    selection=dict(structure=structure,candidates=candidates,selected_lr=best['lr'],selection='Highest target-task validation success at seed5501 and budget16; lower LR breaks ties. Test does not select LR.')
    (folder/'scratch_replay_rate_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8');print(selection,flush=True)
    default=.003 if torus else .001
    # The default-rate five-seed test is already running independently. Do not race it.
    if best['lr']!=default: run(root,structure,[5501,5502,5503,5504,5505],'test',best['lr'])


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('structure',choices=('seven','torus'))
    a=p.parse_args();study(a.root,a.structure)
