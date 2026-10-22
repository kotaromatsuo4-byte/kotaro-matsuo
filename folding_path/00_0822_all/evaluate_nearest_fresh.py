"""Supplementary fresh-start evaluation of the already trained nearest-teacher NNs.

No weights or labels change. This is additional to the originally frozen two-arm
fresh confirmation, and is reported separately rather than replacing it.
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
from nearest_branch_teacher import rollout
from transfer_study import ModalStudent,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root):
    parent=root/'fresh_confirmation';output=parent/'nearest_teacher_supplement';output.mkdir(exist_ok=True)
    checkpoints=[root/'nearest_branch_u2000_test'/f'seed{seed}_{arm}_n04.pt' for seed in range(5501,5506) for arm in ('scratch','pretrained','replay')]
    protocol=output/'protocol.json'
    if not protocol.exists():
        protocol.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoint_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints},
            initial_hash=hashlib.sha256((parent/'seven_starts.npz').read_bytes()).hexdigest(),teacher_paths=4,target_labels=174,target_updates=2000,
            note='Supplementary analysis added after the first fresh comparison. No adaptation to these fresh states; does not replace the originally frozen two-arm confirmation.'),indent=2),encoding='utf-8')
    frozen=json.loads(protocol.read_text(encoding='utf-8'))
    for file,expected in frozen['checkpoint_hashes'].items(): assert hashlib.sha256((root/file).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((parent/'seven_starts.npz').read_bytes()).hexdigest()==frozen['initial_hash']
    starts=np.load(parent/'seven_starts.npz')['test'];g=Geometry(BASE/'model/seven_bar_linkage')
    teacher_file=output/'teacher_evaluation.csv'
    if not teacher_file.exists():
        targets=np.array([np.loadtxt(root/'branch_diagnostic'/f'd4_branch{i}.dat',delimiter=',') for i in range(3)]);rows=[]
        for i,initial in enumerate(starts):
            _,metrics=rollout(g,initial,targets);rows.append(dict(start_id=i,goal=4.,**metrics))
        write_csv(teacher_file,rows);print('fresh nearest teacher',np.mean([r['success'] for r in rows]),flush=True)
    for seed in range(5501,5506):
        for arm in ('scratch','pretrained','replay'):
            prefix=f'seed{seed}_{arm}_n04';file=output/f'{prefix}_evaluation.csv'
            if file.exists(): continue
            model=ModalStudent();model.load_state_dict(torch.load(root/'nearest_branch_u2000_test'/f'{prefix}.pt',weights_only=True)['model'])
            rows=evaluate_model(g,model,starts,4.)
            write_csv(file,[dict(seed=seed,arm=arm,budget=4,labels=174,**r) for r in rows])
            print('fresh nearest NN',seed,arm,np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
