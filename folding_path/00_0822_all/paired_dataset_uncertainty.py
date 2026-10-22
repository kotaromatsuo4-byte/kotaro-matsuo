"""Exploratory transfer uncertainty over support choice, training seed and start."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def run(root,structure):
    directory=root/'support_subset_robustness_g4' if structure=='seven' else root/'torus'/'support_path_robustness'
    group_key='subset' if structure=='seven' else 'support_path';rows=[]
    for file in directory.glob('seed*_evaluation.csv'):
        with file.open() as stream: rows.extend(csv.DictReader(stream))
    groups=sorted({int(r[group_key]) for r in rows});seeds=sorted({int(r['seed']) for r in rows});starts=sorted({int(r['start_id']) for r in rows})
    assert len(groups)==(8 if structure=='seven' else 4) and len(seeds)==3
    lookup={(int(r[group_key]),int(r['seed']),int(r['start_id']),r['arm']):int(r['success']) for r in rows}
    difference=np.array([[[lookup[g,s,i,'pretrained']-lookup[g,s,i,'scratch'] for i in starts] for s in seeds] for g in groups],float)
    rng=np.random.default_rng(77123);bootstrap=[]
    for _ in range(10000):
        gi=rng.integers(len(groups),size=len(groups));si=rng.integers(len(seeds),size=len(seeds));ii=rng.integers(len(starts),size=len(starts))
        bootstrap.append(difference[np.ix_(gi,si,ii)].mean())
    low,high=np.quantile(bootstrap,[.025,.975]);cells=difference.mean(2);group_means=cells.mean(1)
    result=dict(structure=structure,support_groups=len(groups),seeds=len(seeds),shared_test_starts=len(starts),mean_difference=float(difference.mean()),
        bootstrap_low=float(low),bootstrap_high=float(high),positive_groups=int((group_means>0).sum()),negative_groups=int((group_means<0).sum()),
        positive_group_seed_pairs=int((cells>0).sum()),negative_group_seed_pairs=int((cells<0).sum()),
        note='Exploratory percentile interval, independently resampling support groups, training seeds and shared test starts. Source data and topology remain fixed; small group count and unequal label counts limit interpretation. No multiplicity correction.')
    (directory/'paired_uncertainty.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('structure',choices=('seven','torus'))
    a=p.parse_args();run(a.root,a.structure)
