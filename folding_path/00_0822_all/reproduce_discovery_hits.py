"""Replay saved positive conditions; this is verification, not new trials."""
import os
import json
import shutil
import subprocess
import sys
import numpy as np
from discovery_transfer import ROOT,BASE,write_json,digest


def main():
    output=ROOT/'reproduction_checks';output.mkdir(exist_ok=True)
    geometry=output/'holdout_nine';geometry.mkdir(exist_ok=True)
    for name in ('node_initial.dat','member.dat','fix.dat'):
        shutil.copy2(ROOT/'holdout_nine'/name,geometry/name)
    env=dict(os.environ,FOLDING_DISCOVERY_OUTPUT=str(output),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    checks=[]
    for task,seed,series,label,tag in [('seven_d4',5701,'projected_transfer','projected_water','main'),
                                    ('nine_d4',5702,'projected_nine','projected_water','main'),
                                    ('seven_d4',5701,'projected_bars_transfer','projected_bars','bars_only')]:
        source=seed-100;checkpoint=ROOT/'source'/f'{label}_seed{source}'/'best.pt'
        name=f'{task}_projected_frozen_s{seed}'
        directory=output/tag/name;original=ROOT/series/name
        if not (directory/'result.json').exists():
            args=[sys.executable,str(BASE/'discovery_transfer.py'),'run','--task',task,'--architecture','projected',
                  '--arm','frozen','--seed',str(seed),'--source-seed',str(source),'--budget','6400','--checkpoint',str(checkpoint),'--tag',tag]
            with (output/f'{tag}_{name}.log').open('w',encoding='utf-8') as f:
                subprocess.run(args,cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        expected=json.loads((original/'result.json').read_text(encoding='utf-8'))
        actual=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        before=np.load(original/'first_hit_path.npz')['nodes']
        after=np.load(directory/'first_hit_path.npz')['nodes']
        same=before.shape==after.shape and np.array_equal(before,after)
        record=dict(task=task,source_label=label,source_seed=source,search_seed=seed,source_sha256=digest(checkpoint),
            expected_steps=expected['steps'],replayed_steps=actual['steps'],success=actual['success'],
            identical_success_path=bool(same),updates=actual['updates'],
            valid=bool(actual['success'] and actual['steps']==expected['steps'] and same and actual['updates']==0))
        checks.append(record)
    write_json(ROOT/'reproduction_status.json',dict(checks=checks,all_valid=all(r['valid'] for r in checks),
        interpretation='Exact replay of already-observed successes in a separate output tree, not independent validation seeds'))
    print(json.dumps(checks,indent=2))
    if not all(r['valid'] for r in checks):raise RuntimeError('Positive replay mismatch')


if __name__=='__main__':main()
