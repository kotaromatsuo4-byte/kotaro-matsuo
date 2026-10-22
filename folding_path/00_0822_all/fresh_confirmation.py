"""Freeze two contrasts, then generate fresh held-out initial states.

This confirms initial-state generalization only: source/support and topologies
are the same as the exploratory study. It does not claim an unseen topology.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent,SpatialGeometry
from transfer_study import ModalStudent,evaluate_model,path_rollout,target_d4
from supervised_folding import BASE,Geometry,generate_starts,write_csv


def run(root,structure,teacher_only=False):
    folder=root if structure=='seven' else root/'torus';output=root/'fresh_confirmation';output.mkdir(exist_ok=True)
    settings=( [('scratch','modal_tuned_lr0.001_u2000_test','scratch'),('replay','modal_replay0.5_u2000_test','pretrained')]
              if structure=='seven' else [('scratch','graph_src5000_u2000_test_lr0.003','scratch'),('replay','graph_dagger_replay0.25_u2000_test','pretrained')] )
    files=[folder/directory/f'seed{seed}_{arm}_n16.pt' for seed in range(5501,5506) for _,directory,arm in settings]
    protocol=output/f'{structure}_protocol.json'
    if not protocol.exists():
        hashes={str(file.relative_to(root)):hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
        protocol.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),structure=structure,
            conditions=settings,seeds=list(range(5501,5506)),teacher_paths=16,target_updates=2000,
            initial_seed=88001 if structure=='seven' else 88002,initial_count=32 if structure=='seven' else 16,
            random_walk=[20,160] if structure=='seven' else [5,25],checkpoints=hashes,
            purpose='Fresh initial-state confirmation of the frozen scratch versus source-replay contrast; same source/support data and topology.'),indent=2),encoding='utf-8')
    else:
        frozen=json.loads(protocol.read_text(encoding='utf-8'))
        for name,expected in frozen['checkpoints'].items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    g=Geometry(BASE/'model/seven_bar_linkage') if structure=='seven' else SpatialGeometry(BASE/'model/4unit_torus')
    goal=4. if structure=='seven' else 2.;saved=output/f'{structure}_starts.npz'
    if not saved.exists():
        prior=np.load(folder/'splits.npz');avoid=np.concatenate([prior[key] for key in prior.files])
        if structure=='torus': avoid=np.concatenate([avoid,np.load(folder/'wider_starts'/'starts.npz')['test']])
        starts=generate_starts(g,32 if structure=='seven' else 16,np.random.default_rng(88001 if structure=='seven' else 88002),
            20 if structure=='seven' else 5,160 if structure=='seven' else 25,avoid)
        np.savez_compressed(saved,test=starts)
    starts=np.load(saved)['test']
    if teacher_only:
        target=target_d4() if structure=='seven' else np.loadtxt(folder/'target.dat',delimiter=',');rows=[]
        for i,initial in enumerate(starts):
            _,metrics=path_rollout(g,initial,goal,target=target);rows.append(dict(start_id=i,goal=goal,**metrics))
        write_csv(output/f'{structure}_teacher_evaluation.csv',rows)
        print('fresh teacher',structure,np.mean([r['success'] for r in rows]),flush=True);return
    for seed in range(5501,5506):
        for condition,directory,arm in settings:
            destination=output/f'{structure}_seed{seed}_{condition}_evaluation.csv'
            if destination.exists(): continue
            model=ModalStudent() if structure=='seven' else GraphModalStudent()
            if structure=='torus': model.set_geometry(g)
            model.load_state_dict(torch.load(folder/directory/f'seed{seed}_{arm}_n16.pt',weights_only=True)['model'])
            rows=evaluate_model(g,model,starts,goal)
            write_csv(destination,[dict(seed=seed,arm=condition,budget=16,structure=structure,**r) for r in rows])
            print('fresh confirmation',structure,seed,condition,np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('structure',choices=('seven','torus'));p.add_argument('--teacher-only',action='store_true');a=p.parse_args();run(a.root,a.structure,a.teacher_only)
