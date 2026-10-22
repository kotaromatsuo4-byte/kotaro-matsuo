"""Continue mixed-source learning; select using five/seven source validation only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graph_transfer import GraphModalStudent,train_mixed_source
from transfer_study import evaluate_model
from supervised_folding import BASE,Geometry,generate_starts,write_csv


def run(root,seeds):
    five=Geometry(BASE/'model/five_bar_linkage');seven=Geometry(BASE/'model/seven_bar_linkage')
    file=root/'five_quality_splits.npz'
    if not file.exists():
        source=np.load(root/'five_source_starts.npz')['source']
        validation=generate_starts(five,12,np.random.default_rng(66012),20,160,source)
        test=generate_starts(five,16,np.random.default_rng(66013),20,160,[*source,*validation])
        np.savez_compressed(file,validation=validation,test=test)
    groups=[('five',five,np.load(file)['validation']),('seven',seven,np.load(root/'splits.npz')['validation'])]
    for seed in seeds:
        output=root/f'graph_mixed_static_seed{seed}';output.mkdir(exist_ok=True)
        if (output/'selection.json').exists(): continue
        model=GraphModalStudent();model.set_geometry(seven)
        model.load_state_dict(torch.load(root/f'graph_source_mixed_seed{seed}_u5000.pt',weights_only=True)['model'])
        rows=[];best=-1.;best_round=0
        for round_id in range(4):
            if round_id:
                logs=train_mixed_source(model,root,5000,seed+900+round_id)
                write_csv(output/f'learning_round{round_id}.csv',logs)
            current=[]
            for structure,g,starts in groups:
                model.set_geometry(g)
                for goal in (2.,3.):
                    current.extend(dict(source_structure=structure,round=round_id,**r) for r in evaluate_model(g,model,starts,goal))
            rows.extend(current);write_csv(output/'validation.csv',rows)
            rate=np.mean([r['success'] for r in current])
            checkpoint=dict(model=model.state_dict(),seed=seed,round=round_id,labels=9058,
                source_updates=5000+round_id*5000,source_kind='mixed_static')
            torch.save(checkpoint,output/f'round{round_id}.pt')
            if rate>best:
                best=float(rate);best_round=round_id
                torch.save(checkpoint,root/f'graph_source_mixed_static_seed{seed}_best.pt')
            print('mixed continued',seed,'round',round_id,'source validation success',rate,'best',best,flush=True)
        (output/'selection.json').write_text(json.dumps(dict(seed=seed,best_round=best_round,validation_rate=best,
            selection='highest mean success of five/seven D2/D3, 12 validation starts per structure; earliest tie',
            initial_updates=5000,additional_updates=15000,new_teacher_labels=0,target_torus_used=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503,5504,5505])
    a=p.parse_args();run(a.root,a.seeds)
