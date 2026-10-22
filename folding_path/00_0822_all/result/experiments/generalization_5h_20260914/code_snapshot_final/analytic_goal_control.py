"""No-learning baseline using flatness + full-diameter goal energy.

The energy uses geometry and scalar D only, never target coordinates. Lambda=1
is fixed before evaluation. A geometric-energy line search is explicit and
differs from the student's feasibility-only backtracking.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from graph_transfer import SpatialGeometry
from transfer_study import propose
from supervised_folding import BASE,write_csv


def energy_gradient(g,node,goal):
    centered=node-node.mean(0)
    _,_,right=np.linalg.svd(centered,full_matrices=False)
    residual=centered@(np.eye(3)-right[:g.dimension].T@right[:g.dimension])
    energy=float(np.sum(residual**2)/g.initial_potential)
    gradient=2*residual/g.initial_potential
    delta=node[:,None]-node[None,:];squared=(delta**2).sum(-1)
    a,b=np.unravel_index(np.argmax(squared),squared.shape);distance=np.sqrt(squared[a,b])
    diameter=distance/g.scale;error=diameter-goal;energy+=error**2
    if distance>1e-12:
        force=2*error*delta[a,b]/(distance*g.scale)
        gradient[a]+=force;gradient[b]-=force
    return energy,gradient.reshape(-1)


def rollout(g,initial,goal,max_steps=400):
    node=initial.copy();nodes=[node.copy()];reason='timeout';calls=0
    for _ in range(max_steps):
        if g.metrics(node,goal)['success']: reason='success';break
        energy,gradient=energy_gradient(g,node,goal);modes=g.modes(node)
        direction=-modes@(modes.T@gradient);norm=np.linalg.norm(direction)
        if norm<1e-10: reason='zero_gradient';break
        direction/=norm;width=g.step_width;accepted=False
        while width>=1e-4:
            candidate,actual,attempts,status=propose(g,node,direction,width)
            calls+=attempts
            if actual and energy_gradient(g,candidate,goal)[0]<energy-1e-12:
                accepted=True;break
            width*=.5
        if not accepted: reason='energy_or_feasibility_stalled';break
        node=candidate;nodes.append(node.copy())
    metrics=g.metrics(node,goal)
    if metrics['success']: reason='success'
    return np.array(nodes),dict(**metrics,steps=len(nodes)-1,reason=reason,projection_calls=calls)


def run(root,structure,split,goals):
    g=SpatialGeometry(BASE/'model'/structure)
    output=root/f'analytic_{structure}_{split}';output.mkdir(exist_ok=True)
    if split=='reference': starts=g.initial[None]
    elif split=='wider': starts=np.load(root/'torus/wider_starts/starts.npz')['test']
    else: starts=np.load(root/'torus/splits.npz')[split]
    (output/'design.json').write_text(json.dumps(dict(structure=structure,split=split,goals=goals,
        energy='normalized squared distance to best plane + (full Euclidean diameter / member scale - D)^2',
        lambda_goal=1.,max_steps=400,training_labels=0,training_updates=0,
        acceptance='member constraints, collision check, and decrease of the stated geometric energy'),indent=2),encoding='utf-8')
    for goal in goals:
        file=output/f'd{goal:g}.csv'
        if file.exists(): continue
        rows=[]
        for i,initial in enumerate(starts):
            path,metrics=rollout(g,initial,goal)
            rows.append(dict(start_id=i,goal=goal,**metrics));np.savez_compressed(output/f'd{goal:g}_start{i:02d}.npz',nodes=path)
        write_csv(file,rows)
        print('analytic control',structure,split,goal,'success',np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--structure',choices=('4unit_torus','6unit_torus'),default='4unit_torus')
    p.add_argument('--split',choices=('validation','test','wider','reference'),default='test')
    p.add_argument('--goals',type=float,nargs='+',default=[2.])
    a=p.parse_args()
    if a.structure=='6unit_torus' and a.split!='reference': raise ValueError('Six-unit probe uses reference only')
    run(a.root,a.structure,a.split,a.goals)
