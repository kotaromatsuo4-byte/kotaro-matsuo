"""Does an equivalent orthonormal nullspace basis change the physical action?"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent,GraphProjectedStudent
from transfer_study import pad_modes
from supervised_folding import BASE,Geometry,write_csv


@torch.no_grad()
def run(root):
    torch.set_num_threads(1)
    g=Geometry(BASE/'model/seven_bar_linkage');starts=np.load(root/'splits.npz')['validation']
    rows=[]
    for kind,file in [('seven',root/'graph_source_seed5501_u5000.pt'),
                      ('dagger',root/'graph_source_dagger_seed5501_best.pt'),
                      ('projected',root/'graph_source_projected_seed5501_u5000.pt')]:
        if not file.exists(): continue
        model=GraphProjectedStudent() if kind=='projected' else GraphModalStudent();model.set_geometry(g)
        model.load_state_dict(torch.load(file,weights_only=True)['model'])
        generator=torch.Generator().manual_seed(77011)
        for i,node in enumerate(starts):
            for goal in (2.,3.):
                x=torch.tensor(np.append(node.reshape(-1)/g.scale,goal)[None],dtype=torch.float32)
                modes=torch.from_numpy(pad_modes(g,node))[None]
                k=int((modes.square().sum(-1)>1e-10).sum())
                d,w=model(x,modes)
                for rotation_id in range(10):
                    rotation,_=torch.linalg.qr(torch.randn(k,k,generator=generator))
                    rotated=modes.clone();rotated[:,:k]=rotation@modes[:,:k]
                    d2,w2=model(x,rotated)
                    rows.append(dict(source_kind=kind,start_id=i,goal=goal,rotation_id=rotation_id,
                        direction_cosine=float((d*d2).sum()),width_difference=float((w-w2).abs())))
        print('basis sensitivity',kind,'mean direction cosine',np.mean([r['direction_cosine'] for r in rows if r['source_kind']==kind]),flush=True)
    write_csv(root/'basis_sensitivity_validation.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
