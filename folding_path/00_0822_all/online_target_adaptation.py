"""Query the teacher only at student-visited D4 states, with equal query budgets.

By default no expert D4 trajectory is used. Optional bootstrap trajectories and
their update/label costs are recorded separately. Four fixed support starts are
cycled; pretrained and scratch arms collect their own on-policy states.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from transfer_study import ModalStudent,path_rollout,target_d4,pad_modes,oracle,pack,train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds,rounds,queries,updates,split,bootstrap_paths=0,bootstrap_updates=0,balanced_states=False):
    g=Geometry(BASE/'model/seven_bar_linkage');target=target_d4();splits=np.load(root/'splits.npz')
    suffix=f'_boot{bootstrap_paths}u{bootstrap_updates}' if bootstrap_paths else ''
    if balanced_states: suffix+='_balanced'
    output=root/f'online_d4_q{queries}_u{updates}_{split}{suffix}';output.mkdir(exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed);model=ModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        pretrained=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for arm,weights in (('scratch',initial),('pretrained',pretrained)):
            final=output/f'seed{seed}_{arm}_n{rounds*queries:02d}_evaluation.csv'
            if final.exists(): continue
            model.load_state_dict(weights);records=[];history=[];teacher_calls=0
            bootstrap_labels=0
            if bootstrap_paths:
                support=dict(np.load(root/'support_data.npz'));indices=np.flatnonzero(support['path_id']<bootstrap_paths)
                records=[dict(**{k:v[i] for k,v in support.items()},round_id=0) for i in indices]
                bootstrap_labels=len(records)
                logs=train_fixed(model,pack(records),g.scale,bootstrap_updates,seed+650,lr=3e-4)
                history.extend(dict(round=0,teacher_queries=0,**r) for r in logs)
            rng=np.random.default_rng(seed+600)
            for round_id in range(1,rounds+1):
                candidates=[];by_path=[]
                for i in range(4):
                    path,_=path_rollout(g,splits['support'][i],4.,model=model,max_steps=300)
                    np.savez_compressed(output/f'seed{seed}_{arm}_r{round_id}_start{i}.npz',nodes=path)
                    # Include every visited state, including the start, exactly once per rollout.
                    if balanced_states:
                        kept=[path[0]]
                        for node in path[1:]:
                            if np.linalg.norm(node-kept[-1])>=.03*g.scale: kept.append(node)
                        if np.linalg.norm(path[-1]-kept[-1])>1e-10: kept.append(path[-1])
                        by_path.append([(node,i) for node in kept])
                    else: candidates.extend((node,i) for node in path)
                # With replacement only when the controller visits too few distinct states.
                if balanced_states:
                    selected=[]
                    for i,pool in enumerate(by_path):
                        count=queries//4+int(i<queries%4)
                        indices=rng.choice(len(pool),count,replace=len(pool)<count)
                        selected.extend(pool[index] for index in indices)
                else:
                    indices=rng.choice(len(candidates),queries,replace=len(candidates)<queries)
                    selected=[candidates[index] for index in indices]
                for node,path_id in selected:
                    direction,width,_,status,calls=oracle(g,node,target)
                    teacher_calls+=calls
                    if np.linalg.norm(direction)<1e-8: continue
                    records.append(dict(node=node.copy(),goal=4.,modes=pad_modes(g,node),direction=direction,
                        width=width/g.step_width,width_valid=float(status=='accepted'),path_id=path_id,
                        teacher_calls=calls,round_id=round_id))
                data=pack(records)
                logs=train_fixed(model,data,g.scale,updates,seed+700+round_id,lr=3e-4)
                history.extend(dict(round=round_id,teacher_queries=round_id*queries,**r) for r in logs)
                budget=round_id*queries;prefix=f'seed{seed}_{arm}_n{budget:02d}'
                np.savez_compressed(output/f'{prefix}_data.npz',**data)
                torch.save(dict(model=model.state_dict(),seed=seed,arm=arm,queries=budget,round=round_id,
                    bootstrap_paths=bootstrap_paths,bootstrap_labels=bootstrap_labels,bootstrap_updates=bootstrap_updates),output/f'{prefix}.pt')
                write_csv(output/f'seed{seed}_{arm}_learning.csv',history)
                if round_id in (1,2,4,8,rounds):
                    rows=[]
                    for goal in (2.,3.,4.): rows.extend(evaluate_model(g,model,splits[split],goal))
                    write_csv(output/f'{prefix}_evaluation.csv',[dict(seed=seed,arm=arm,budget=budget,labels=len(records),
                        updates=bootstrap_updates+round_id*updates,bootstrap_labels=bootstrap_labels,
                        bootstrap_paths=bootstrap_paths,teacher_projection_calls=teacher_calls,**r) for r in rows])
                    print('online D4',prefix,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in (2.,3.,4.)},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--queries',type=int,default=128);p.add_argument('--updates',type=int,default=250)
    p.add_argument('--bootstrap-paths',type=int,default=0);p.add_argument('--bootstrap-updates',type=int,default=0)
    p.add_argument('--balanced-states',action='store_true')
    p.add_argument('--split',choices=('validation','test'),default='validation')
    a=p.parse_args();run(a.root,a.seeds,a.rounds,a.queries,a.updates,a.split,a.bootstrap_paths,a.bootstrap_updates,a.balanced_states)
