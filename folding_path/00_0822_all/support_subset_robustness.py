"""Repeat four-trajectory D4 adaptation on eight disjoint support subsets.

Teacher paths, source weights, test starts and validation-selected learning rates
are frozen. Repetition over support subsets is separate from training-seed
repetition; failure paths are retained and each subset's label count is reported.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from transfer_study import ModalStudent,train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


def run(root,seeds):
    g=Geometry(BASE/'model/seven_bar_linkage');support=dict(np.load(root/'support_data.npz'))
    starts=np.load(root/'splits.npz')['test'];rates=json.loads((root/'d4_rate_selection.json').read_text())['selection']
    output=root/'support_subset_robustness_g4';output.mkdir(exist_ok=True)
    (output/'design.json').write_text(json.dumps(dict(seeds=seeds,groups=8,paths_per_group=4,grouping='disjoint contiguous path IDs 0..31',
        updates=2000,learning_rates=rates,test_starts=32,source='same frozen D2/D3 checkpoints as primary experiment'),indent=2),encoding='utf-8')
    for seed in seeds:
        torch.manual_seed(seed);model=ModalStudent();initial={k:v.clone() for k,v in model.state_dict().items()}
        pretrained=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for group in range(8):
            ids=list(range(4*group,4*group+4));mask=np.isin(support['path_id'],ids);data={k:v[mask] for k,v in support.items()}
            for arm,weights in (('scratch',initial),('pretrained',pretrained)):
                prefix=f'seed{seed}_{arm}_g{group:02d}'
                file=output/f'{prefix}_evaluation.csv'
                if file.exists(): continue
                if group==0:
                    directory=('modal_u2000_src10000_test' if rates[arm]==.0003 else f'modal_tuned_lr{rates[arm]:g}_u2000_test')
                    prior=root/directory/f'seed{seed}_{arm}_n04_evaluation.csv'
                    with prior.open() as f: rows=[r for r in csv.DictReader(f) if float(r['goal'])==4.]
                    rows=[{k:v for k,v in r.items() if k not in ('seed','arm','budget','labels','updates')} for r in rows]
                else:
                    model.load_state_dict(weights)
                    logs=train_fixed(model,data,g.scale,2000,seed+200,lr=rates[arm])
                    write_csv(output/f'{prefix}_learning.csv',logs)
                    torch.save(dict(model=model.state_dict(),seed=seed,arm=arm,subset=group,path_ids=ids,lr=rates[arm]),output/f'{prefix}.pt')
                    rows=evaluate_model(g,model,starts,4.)
                rows=[dict(seed=seed,arm=arm,budget=4,subset=group,path_ids='|'.join(map(str,ids)),labels=int(mask.sum()),updates=2000,**r) for r in rows]
                write_csv(file,rows)
                print('support subset',seed,group,arm,'labels',mask.sum(),'success',np.mean([int(r['success']) for r in rows]),flush=True)
    summarize(output)


def summarize(output):
    raw=[]
    for file in output.glob('seed*_evaluation.csv'):
        with file.open() as f: raw.extend(csv.DictReader(f))
    rows=[]
    for group in range(8):
        for arm in ('scratch','pretrained'):
            selected=[r for r in raw if int(r['subset'])==group and r['arm']==arm]
            if not selected: continue
            seeds=sorted({int(r['seed']) for r in selected})
            values=[np.mean([int(r['success']) for r in selected if int(r['seed'])==seed]) for seed in seeds]
            rows.append(dict(subset=group,arm=arm,seeds=len(seeds),labels=int(selected[0]['labels']),mean_success=np.mean(values),seed_min=min(values),seed_max=max(values)))
    if rows: write_csv(output/'summary.csv',rows)


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503]);p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    if a.summarize_only: summarize(a.root/'support_subset_robustness_g4')
    else: run(a.root,a.seeds)
