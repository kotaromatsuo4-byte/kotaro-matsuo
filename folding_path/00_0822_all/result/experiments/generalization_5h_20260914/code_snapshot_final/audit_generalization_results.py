"""Verify frozen data, reported terminal success, and zero-shot weight provenance."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def run(root):
    hashes=json.loads((root/'frozen_data_and_source_hashes.json').read_text())
    for name,expected in hashes.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert actual==expected,f'Frozen file changed: {name}'
    provenance=[]
    for kind in ('seven','dagger','static','mixed','mixed_static'):
        suffix='' if kind=='seven' else '_'+kind
        for seed in range(5501,5506):
            original=(root/f'graph_source_{kind}_seed{seed}_best.pt' if kind in ('dagger','static','mixed_static') else
                      root/f'graph_source{suffix}_seed{seed}_u5000.pt')
            copied=root/'torus'/f'graph_src5000_u2000_test{suffix}'/f'seed{seed}_pretrained_n00.pt'
            if not copied.exists(): continue
            a=torch.load(original,weights_only=True)['model'];b=torch.load(copied,weights_only=True)['model']
            difference=max(float((v-b[k]).abs().max()) for k,v in a.items())
            assert difference==0.,f'Zero-shot source provenance mismatch: {copied}'
            provenance.append(dict(kind=kind,seed=seed,source=str(original.relative_to(root)),zero_shot=str(copied.relative_to(root)),max_difference=difference))
    files=0;rows=0;successes=0
    for file in root.rglob('*evaluation.csv'):
        if 'provisional_evaluations' in file.parts: continue
        with file.open(encoding='utf-8') as stream:
            reader=csv.DictReader(stream)
            required={'flatness','spread_error','strain','success'}
            if not required.issubset(reader.fieldnames or []): continue
            files+=1
            for r in reader:
                values=[float(r[k]) for k in ('flatness','spread_error','strain')]
                assert np.isfinite(values).all(),f'Non-finite metric: {file}'
                expected=int(values[0]<=1e-3 and values[1]<=.05 and values[2]<=1e-8)
                assert int(r['success'])==expected,f'Inconsistent success flag: {file}'
                rows+=1;successes+=expected
    result=dict(frozen_hashes_verified=len(hashes),zero_shot_checkpoints_verified=len(provenance),
        evaluation_files=files,evaluation_rows=rows,successful_rows=successes,
        note='Row counts include diagnostic and validation evaluations; they are not independent experiments.',provenance=provenance)
    (root/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='provenance'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
