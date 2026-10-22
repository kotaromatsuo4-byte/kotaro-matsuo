"""Hold-out source competence and zero-shot D4 of graph checkpoints."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent,GraphProjectedStudent
from transfer_study import evaluate_model
from supervised_folding import BASE,Geometry,write_csv,generate_starts


def run(root,seeds,kinds,split,structure='seven'):
    g=Geometry(BASE/f'model/{structure}_bar_linkage')
    if structure=='five':
        file=root/'five_quality_splits.npz'
        if not file.exists():
            source_starts=np.load(root/'five_source_starts.npz')['source']
            validation=generate_starts(g,12,np.random.default_rng(66012),20,160,source_starts)
            test=generate_starts(g,16,np.random.default_rng(66013),20,160,[*source_starts,*validation])
            np.savez_compressed(file,validation=validation,test=test)
        starts=np.load(file)[split]
    else: starts=np.load(root/'splits.npz')[split]
    for kind in kinds:
        parent=root/f'graph_source_evaluation_{split}'
        if structure=='five': parent=parent/'five'
        output=parent/kind;output.mkdir(parents=True,exist_ok=True)
        for seed in seeds:
            file=(root/f'graph_source_{kind}_seed{seed}_best.pt' if kind in ('dagger','static','mixed_static') else
                  root/f'graph_source{("_"+kind) if kind!="seven" else ""}_seed{seed}_u5000.pt')
            if not file.exists(): continue
            if kind in ('dagger','static','mixed_static'):
                final_selection=root/f'graph_{kind}_seed{seed}'/'selection.json'
                if not final_selection.exists(): continue
            checkpoint=torch.load(file,weights_only=True)
            checkpoint_hash=hashlib.sha256(file.read_bytes()).hexdigest()
            destination=output/f'seed{seed}_pretrained_n00_evaluation.csv'
            if destination.exists():
                with destination.open() as stream: prior=next(csv.DictReader(stream))
                if int(prior['source_round'])==checkpoint.get('round',0) and prior.get('checkpoint_sha256',checkpoint_hash)==checkpoint_hash: continue
                archive=output/'provisional_evaluations';archive.mkdir(exist_ok=True)
                (archive/f'seed{seed}_round{prior["source_round"]}.csv').write_bytes(destination.read_bytes())
            model=GraphProjectedStudent() if kind=='projected' else GraphModalStudent();model.set_geometry(g)
            model.load_state_dict(checkpoint['model'])
            rows=[]
            goals=(2.,3.) if structure=='five' else (2.,3.,4.)
            for goal in goals: rows.extend(evaluate_model(g,model,starts,goal))
            source_labels=checkpoint.get('labels',5688+(3370 if kind in ('mixed','mixed_static') else 0))
            write_csv(destination,[dict(seed=seed,arm='pretrained',budget=0,labels=0,source_labels=source_labels,
                source_kind=kind,source_round=checkpoint.get('round',0),checkpoint_sha256=checkpoint_hash,**r) for r in rows])
            print('source quality',structure,kind,seed,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in goals},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503,5504,5505])
    p.add_argument('--kinds',nargs='+',choices=('seven','dagger','static','mixed','mixed_static','projected'),default=['seven','dagger'])
    p.add_argument('--split',choices=('validation','test'),default='test')
    p.add_argument('--structure',choices=('five','seven'),default='seven')
    a=p.parse_args();run(a.root,a.seeds,a.kinds,a.split,a.structure)
