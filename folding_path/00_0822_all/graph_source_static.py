"""Compute-matched source control: extra updates without adding on-policy data."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent
from transfer_study import train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds):
    g=Geometry(BASE/'model/seven_bar_linkage');data=dict(np.load(root/'source_data.npz'))
    validation=np.load(root/'splits.npz')['validation']
    for seed in seeds:
        output=root/f'graph_static_seed{seed}';output.mkdir(exist_ok=True)
        if (output/'selection.json').exists(): continue
        model=GraphModalStudent();model.set_geometry(g)
        model.load_state_dict(torch.load(root/f'graph_source_seed{seed}_u5000.pt',weights_only=True)['model'])
        rows=[];best=-1.;best_round=0
        for round_id in range(4):
            if round_id:
                logs=train_fixed(model,data,g.scale,2000,seed+300+round_id,batch=32)
                write_csv(output/f'learning_round{round_id}.csv',logs)
            results=[]
            for goal in (2.,3.): results.extend(evaluate_model(g,model,validation,goal))
            rate=np.mean([r['success'] for r in results]);rows.extend(dict(round=round_id,labels=len(data['node']),**r) for r in results)
            write_csv(output/'validation.csv',rows)
            checkpoint=dict(model=model.state_dict(),seed=seed,round=round_id,labels=len(data['node']))
            torch.save(checkpoint,output/f'round{round_id}.pt')
            if rate>best:
                best=float(rate);best_round=round_id
                torch.save(checkpoint,root/f'graph_source_static_seed{seed}_best.pt')
            print('graph static source',seed,'round',round_id,'success',rate,'best',best,flush=True)
        (output/'selection.json').write_text(json.dumps(dict(seed=seed,best_round=best_round,validation_rate=best,
            selection='highest mean D2/D3 validation success; earliest tie',source_only=True,
            initial_updates=5000,additional_updates=6000,added_teacher_labels=0),indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503,5504,5505])
    a=p.parse_args();run(a.root,a.seeds)
