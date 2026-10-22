"""Validation-only diagnostic of multiple collinear states with the same D4.

These teacher rollouts do not enter any adaptation dataset. Three candidate
coordinate targets are possible for this six-member, fixed-endpoint chain.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from transfer_study import path_rollout
from supervised_folding import BASE,Geometry,write_csv


def run(root):
    g=Geometry(BASE/'model/seven_bar_linkage');starts=np.load(root/'splits.npz')['validation']
    output=root/'branch_diagnostic';output.mkdir(exist_ok=True)
    targets=[]
    for signs in itertools.product((-1,1),repeat=6):
        if sum(signs)!=2: continue
        x=np.r_[-1,-1+np.cumsum(signs)]
        if np.ptp(x)!=4: continue
        target=np.zeros((7,3));target[:,0]=x;targets.append(target)
    rows=[]
    for branch,target in enumerate(targets):
        np.savetxt(output/f'd4_branch{branch}.dat',target,delimiter=',')
        for i,initial in enumerate(starts):
            path,metrics=path_rollout(g,initial,4.,target=target)
            rows.append(dict(branch=branch,start_id=i,**metrics))
            np.savez_compressed(output/f'branch{branch}_start{i:02d}.npz',nodes=path)
        print('teacher D4 branch',branch,'success',sum(r['success'] for r in rows if r['branch']==branch),'/',len(starts),flush=True)
    write_csv(output/'validation.csv',rows)
    union=sum(any(r['success'] for r in rows if r['start_id']==i) for i in range(len(starts)))
    summary=dict(split='validation',initial_shapes=len(starts),any_target_successes=union,
        original_target_branch=2,note='Union of separate oracle attempts, not a single deployable student or controller')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
