"""Aggregate on-policy D2/D3 states without looking at D4 or torus data."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent
from transfer_study import path_rollout,build_records,pack,train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,load_targets,write_csv


def run(root,seed,rounds,updates,trajectories):
    g=Geometry(BASE/'model/seven_bar_linkage');targets=load_targets(g)
    splits=np.load(root/'splits.npz');source_starts=splits['source'];validation=splits['validation']
    data=dict(np.load(root/'source_data.npz'))
    torch.manual_seed(seed)
    model=GraphModalStudent();model.set_geometry(g)
    base=root/f'graph_source_seed{seed}_u5000.pt'
    if base.exists():
        model.load_state_dict(torch.load(base,weights_only=True)['model'])
    else:
        logs=train_fixed(model,data,g.scale,5000,seed+100,batch=32)
        temporary=root/f'graph_source_seed{seed}_u5000_dagger_tmp.pt'
        torch.save(dict(model=model.state_dict(),seed=seed),temporary)
        temporary.replace(base)
        write_csv(base.with_suffix('.csv'),logs)
    output=root/f'graph_dagger_seed{seed}';output.mkdir(exist_ok=True)
    rows=[];best=-1.;best_round=0
    for round_id in range(rounds+1):
        if round_id:
            records=[]
            for j in range(trajectories):
                i=((round_id-1)*trajectories+j)%len(source_starts)
                for goal in (2.,3.):
                    path,metrics=path_rollout(g,source_starts[i],goal,model=model,max_steps=300)
                    np.savez_compressed(output/f'round{round_id}_start{i:02d}_d{goal:g}.npz',nodes=path)
                    for label_goal in (2.,3.):
                        records.extend(build_records(g,path,targets[label_goal],label_goal,i,spacing=.04))
                print('graph DAgger',seed,round_id,j+1,'/',trajectories,'new labels',len(records),flush=True)
            new=pack(records)
            data={k:np.concatenate([data[k],new[k]]) for k in data}
            np.savez_compressed(output/f'data_round{round_id}.npz',**data)
            logs=train_fixed(model,data,g.scale,updates,seed+300+round_id,batch=32)
            write_csv(output/f'learning_round{round_id}.csv',logs)
        results=[]
        for goal in (2.,3.): results.extend(evaluate_model(g,model,validation,goal))
        rate=np.mean([r['success'] for r in results])
        rows.extend(dict(round=round_id,labels=len(data['node']),**r) for r in results)
        write_csv(output/'validation.csv',rows)
        checkpoint=dict(model=model.state_dict(),seed=seed,round=round_id,labels=len(data['node']))
        torch.save(checkpoint,output/f'round{round_id}.pt')
        if rate>best:
            best=float(rate);best_round=round_id
            torch.save(checkpoint,root/f'graph_source_dagger_seed{seed}_best.pt')
        print('graph DAgger validation',seed,'round',round_id,'mean success',rate,'best',best,flush=True)
    (output/'selection.json').write_text(json.dumps(dict(seed=seed,best_round=best_round,validation_rate=best,
        selection='highest mean D2/D3 validation success; earliest tie',source_only=True,
        initial_updates=5000,additional_updates=rounds*updates),indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--trajectories',type=int,default=16)
    a=p.parse_args()
    for seed in a.seeds: run(a.root,seed,a.rounds,a.updates,a.trajectories)
