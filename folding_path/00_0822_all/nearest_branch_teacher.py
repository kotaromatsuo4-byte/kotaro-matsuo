"""D4 teacher with the nearest of three valid targets, chosen from current state.

An exploratory teacher diagnostic: one current-state branch decision, not an
oracle union of independently completed trajectories. No test labels train NN.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
from transfer_study import oracle,build_records,pack
from supervised_folding import BASE,Geometry,write_csv


def rollout(g,initial,targets):
    node=initial.copy();nodes=[node.copy()];branches=[];calls=0;smallest=g.step_width;reason='timeout'
    for _ in range(400):
        if g.metrics(node,4.)['success']: reason='success';break
        branch=int(np.argmin(np.linalg.norm(targets-node,axis=(1,2))));branches.append(branch)
        _,width,candidate,reason,attempts=oracle(g,node,targets[branch]);calls+=attempts
        if not width or np.linalg.norm(candidate-node)<1e-12: break
        smallest=min(smallest,width);node=candidate;nodes.append(node.copy())
        if len(nodes)>32 and np.linalg.norm(node-nodes[-25])<1e-6: reason='cycle';break
    metrics=g.metrics(node,4.)
    if metrics['success']: reason='success'
    elif reason=='accepted': reason='timeout'
    return np.asarray(nodes),dict(**metrics,steps=len(nodes)-1,reason=reason,projection_calls=calls,min_width=smallest,
        branch_switches=sum(a!=b for a,b in zip(branches[:-1],branches[1:])),initial_branch=branches[0] if branches else -1)


def run(root,split):
    output=root/'nearest_branch_teacher';output.mkdir(exist_ok=True)
    targets=np.array([np.loadtxt(root/'branch_diagnostic'/f'd4_branch{i}.dat',delimiter=',') for i in range(3)])
    g=Geometry(BASE/'model/seven_bar_linkage');rows=[];records=[]
    for i,initial in enumerate(np.load(root/'splits.npz')[split]):
        path,metrics=rollout(g,initial,targets)
        np.savez_compressed(output/f'{split}_start{i:02d}.npz',nodes=path)
        rows.append(dict(split=split,start_id=i,goal=4.,**metrics))
        if split=='support':
            last=None
            for j,node in enumerate(path):
                if last is not None and np.linalg.norm(node-last)<.03*g.scale and j!=len(path)-1: continue
                if last is not None and np.linalg.norm(node-last)<1e-10: continue
                last=node;branch=int(np.argmin(np.linalg.norm(targets-node,axis=(1,2))))
                records.extend(build_records(g,[node],targets[branch],4.,i))
        write_csv(output/f'{split}_evaluation.csv',rows)
        print('nearest branch teacher',split,i,metrics['success'],flush=True)
    if split=='support': np.savez_compressed(output/'support_data.npz',**pack(records))
    print('nearest branch teacher total',split,np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--split',choices=('validation','test','support'),default='validation');a=p.parse_args();run(a.root,a.split)
