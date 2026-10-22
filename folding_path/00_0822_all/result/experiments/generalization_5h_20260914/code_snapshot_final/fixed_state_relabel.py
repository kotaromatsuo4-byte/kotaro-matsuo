"""Separate the teacher label rule from changes in visited-state coverage.

At four original support trajectories, keep identical NN inputs and replace only
the direction/width labels by the nearest-branch oracle. Three fixed seeds.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from transfer_study import ModalStudent,oracle,train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,teacher_direction,write_csv


def run(root):
    output=root/'fixed_state_relabel_n04_test';output.mkdir(exist_ok=True)
    g=Geometry(BASE/'model/seven_bar_linkage');support=dict(np.load(root/'support_data.npz'))
    source_data={k:v[support['path_id']<4] for k,v in support.items()};original_count=len(source_data['node'])
    if not (output/'nearest_labels.npz').exists():
        targets=np.array([np.loadtxt(root/'branch_diagnostic'/f'd4_branch{i}.dat',delimiter=',') for i in range(3)])
        indices=[];directions=[];widths=[];valids=[];calls=[]
        for i,node in enumerate(source_data['node']):
            branch=int(np.argmin(np.linalg.norm(targets-node,axis=(1,2))));modes=g.modes(node)
            direction,norm=teacher_direction(node,targets[branch],modes)
            if norm<=1e-8: continue
            _,width,_,status,attempts=oracle(g,node,targets[branch],modes)
            indices.append(i);directions.append(direction);widths.append(width/g.step_width);valids.append(float(status=='accepted'));calls.append(attempts)
        original={k:v[indices] for k,v in source_data.items()};nearest={k:v.copy() for k,v in original.items()}
        nearest.update(direction=np.asarray(directions),width=np.asarray(widths),width_valid=np.asarray(valids),teacher_calls=np.asarray(calls))
        np.savez_compressed(output/'original_labels.npz',**original);np.savez_compressed(output/'nearest_labels.npz',**nearest)
        for key in ('node','goal','modes','path_id'): assert np.array_equal(original[key],nearest[key])
        (output/'design.json').write_text(json.dumps(dict(teacher_paths=4,original_states=original_count,common_states=len(indices),
            identical_inputs=['node','goal','modes','path_id'],changed_labels=['direction','width','width_valid'],seeds=[5501,5502,5503],
            updates=2000,scratch_lr=.001,pretrained_lr=.0003,selection='Fixed three-seed label-rule ablation; no hyperparameter selection'),indent=2),encoding='utf-8')
    starts=np.load(root/'splits.npz')['test']
    for seed in (5501,5502,5503):
        torch.manual_seed(seed);model=ModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        learned=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for condition in ('original','nearest'):
            data=dict(np.load(output/f'{condition}_labels.npz'));directory=output/condition;directory.mkdir(exist_ok=True)
            for arm,weights in [('scratch',initial),('pretrained',learned)]:
                prefix=f'seed{seed}_{arm}_n04';destination=directory/f'{prefix}_evaluation.csv'
                if destination.exists(): continue
                if condition=='original' and len(data['node'])==original_count:
                    prior=root/('modal_tuned_lr0.001_u2000_test' if arm=='scratch' else 'modal_u2000_src10000_test')/f'{prefix}_evaluation.csv'
                    with prior.open() as stream: rows=[r for r in csv.DictReader(stream) if float(r['goal'])==4.]
                    rows=[{k:v for k,v in r.items() if k not in ('seed','arm','budget','labels','updates')} for r in rows]
                else:
                    model.load_state_dict(weights);logs=train_fixed(model,data,g.scale,2000,seed+200,lr=.001 if arm=='scratch' else .0003)
                    write_csv(directory/f'{prefix}_learning.csv',logs)
                    torch.save(dict(model=model.state_dict(),seed=seed,arm=arm,condition=condition,updates=2000),directory/f'{prefix}.pt')
                    rows=evaluate_model(g,model,starts,4.)
                write_csv(destination,[dict(seed=seed,arm=arm,budget=4,labels=len(data['node']),label_rule=condition,**r) for r in rows])
                print('fixed states',seed,condition,arm,np.mean([int(r['success']) for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
