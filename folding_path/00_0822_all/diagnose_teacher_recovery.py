"""Can the original local teacher move away from a failed student's terminal state?"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from transfer_study import ModalStudent,path_rollout,target_d4,oracle
from supervised_folding import BASE,Geometry,teacher_direction,write_csv


def run(root):
    torch.set_num_threads(1);g=Geometry(BASE/'model/seven_bar_linkage')
    starts=np.load(root/'splits.npz')['validation'];target=target_d4()
    directory=root/'modal_u2000_src10000_test'
    checkpoints=[('zero_shot',directory/'source_seed5501.pt'),
                 ('adapted_16',directory/'seed5501_pretrained_n16.pt')]
    rows=[];output=root/'teacher_recovery_diagnostic';output.mkdir(exist_ok=True)
    for name,file in checkpoints:
        model=ModalStudent();model.load_state_dict(torch.load(file,weights_only=True)['model'])
        for i,initial in enumerate(starts):
            path,metrics=path_rollout(g,initial,4.,model=model)
            terminal=path[-1]
            _,norm=teacher_direction(terminal,target,g.modes(terminal))
            _,width,_,status,calls=oracle(g,terminal,target)
            rows.append(dict(condition=name,start_id=i,teacher_gradient_norm=norm,teacher_width=width,
                teacher_status=status,teacher_projection_calls=calls,**metrics))
            np.savez_compressed(output/f'{name}_start{i:02d}.npz',nodes=path)
        subset=[r for r in rows if r['condition']==name and not r['success']]
        print('teacher recovery',name,'student failures',len(subset),'teacher can take descending step',sum(r['teacher_width']>0 for r in subset),flush=True)
    write_csv(output/'validation.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
