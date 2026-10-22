"""Freeze D4 learning-rate choices from validation, then optionally evaluate test."""
import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def run(root,run_tests):
    rows=[];budgets=[1,4,16,32]
    for lr in (.0003,.001,.003):
        folder=root/('modal_u2000_src10000_validation' if lr==.0003 else f'modal_tuned_lr{lr:g}_u2000_validation')
        for arm in ('scratch','pretrained'):
            rates=[]
            for budget in budgets:
                with (folder/f'seed5501_{arm}_n{budget:02d}_evaluation.csv').open() as f:
                    subset=[r for r in csv.DictReader(f) if float(r['goal'])==4.]
                if len(subset)!=12: raise ValueError('Incomplete validation arm')
                rates.append(float(np.mean([int(r['success']) for r in subset])))
            rows.append(dict(arm=arm,lr=lr,mean_success=float(np.mean(rates)),rates=rates))
    selected={arm:sorted([r for r in rows if r['arm']==arm],key=lambda r:(-round(r['mean_success'],12),r['lr']))[0]['lr'] for arm in ('scratch','pretrained')}
    result=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection=selected,grid=rows,budgets=budgets,
        selection_rule='Mean validation success over budgets 1/4/16/32; lower rate wins ties; no test outcomes used',
        validation_seed=5501,validation_initial_shapes=12,limitations='One training seed used for tuning; not exhaustive hyperparameter optimization')
    file=root/'d4_rate_selection.json'
    if file.exists():
        prior=json.loads(file.read_text(encoding='utf-8'))
        if prior['selection']!=selected: raise ValueError('Refusing to change a frozen selection')
    else: file.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if run_tests:
        for lr in sorted(set(selected.values())):
            if lr==.0003: continue # Existing primary five-seed test supplies this condition.
            arms=[arm for arm in selected if selected[arm]==lr]
            command=[sys.executable,'tune_d4_transfer.py',str(root),'--seeds','5501','5502','5503','5504','5505',
                     '--budgets',*[str(n) for n in budgets],'--lr',str(lr),'--split','test','--arms',*arms]
            subprocess.run(command,check=True,cwd=Path(__file__).resolve().parent)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--run-tests',action='store_true')
    a=p.parse_args();run(a.root,a.run_tests)
