"""Run a saved folding student with current geometry and scalar D, without a target shape.

Example: python rollout_saved_student.py CHECKPOINT --structure seven --goal 4
         --initial model/seven_bar_linkage/node_initial.dat --output result/my_rollout
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent,GraphProjectedStudent,SpatialGeometry
from transfer_study import ModalStudent,CategoricalModalStudent,path_rollout
from supervised_folding import BASE,Geometry,write_csv


def run(checkpoint,structure,goal,initial_path,output):
    directory=BASE/'model'/dict(five='five_bar_linkage',seven='seven_bar_linkage',torus='4unit_torus',torus6='6unit_torus')[structure]
    g=SpatialGeometry(directory) if structure.startswith('torus') else Geometry(directory)
    saved=torch.load(checkpoint,weights_only=True);weights=saved['model']
    if 'node_encoder.0.weight' in weights:
        hidden=weights['node_encoder.0.weight'].shape[0]
        model=(GraphProjectedStudent(hidden) if weights['coefficient.weight'].shape[0]==3 else GraphModalStudent(hidden))
        model.set_geometry(g)
    else:
        if 'encoder.0.weight' not in weights: raise ValueError('This runner supports ModalStudent and graph students, not adapter checkpoints')
        size=g.coordinate_size;hidden,features=weights['encoder.0.weight'].shape
        if features==2*size+1: model=ModalStudent(size,hidden)
        elif features==2*size+3:
            if goal not in (2.,3.,4.): raise ValueError('This categorical checkpoint supports D2/D3/D4 only')
            model=CategoricalModalStudent(size,hidden)
        else: raise ValueError('Fixed-size checkpoint does not match this structure')
    model.load_state_dict(weights);model.eval()
    initial=g.initial.copy() if initial_path is None else np.loadtxt(initial_path,delimiter=',')
    if initial.shape!=g.initial.shape: raise ValueError('Initial node array does not match structure')
    if output.exists() and any(output.iterdir()): raise FileExistsError('Choose an empty output directory to preserve earlier runs')
    output.mkdir(parents=True,exist_ok=True)
    nodes,metrics=path_rollout(g,initial,goal,model=model)
    np.savez_compressed(output/'path.npz',nodes=nodes)
    np.savetxt(output/'node_final.dat',nodes[-1],delimiter=',')
    write_csv(output/'metrics.csv',[dict(step=i,**g.metrics(node,goal)) for i,node in enumerate(nodes)])
    provenance=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        structure=structure,goal=goal,initial=str(initial_path) if initial_path else 'model reference initial',
        target_coordinates_used=False,model_type=type(model).__name__,**metrics)
    (output/'result.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--structure',choices=('five','seven','torus','torus6'),required=True);p.add_argument('--goal',type=float,required=True)
    p.add_argument('--initial',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.checkpoint,a.structure,a.goal,a.initial,a.output)
