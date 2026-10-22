"""Sequential exploratory validation jobs; each child retains its own artifacts."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

root='result/experiments/generalization_5h_20260914'
jobs=[
    ['discrete_goal_adapter.py',root,'--seeds','5501','--budgets','1','4','16','32','--split','validation'],
    ['graph_transfer.py','study',root,'--target','torus','--seeds','5501','--budgets','0','1','4','16','--split','validation','--architecture','projected'],
    ['graph_transfer.py','study',root,'--target','torus','--seeds','5501','--budgets','0','1','4','16','--split','validation','--source-kind','mixed','--arms','pretrained'],
    ['online_target_adaptation.py',root,'--seeds','5501','--split','validation'],
    ['generalization_oracles.py','teachers',root],
    ['target_branch_diagnostic.py',root],
    ['generalization_oracles.py','wider',root],
    ['evaluate_graph_sources.py',root,'--kinds','seven','dagger','mixed','projected'],
]
if __name__=='__main__':
    base=Path(__file__).resolve().parent
    name='validation_queue'
    if len(sys.argv)>1:
        specification=Path(sys.argv[1]);jobs=json.loads(specification.read_text(encoding='utf-8'));name=specification.stem
    journal=base/root/f'{name}.jsonl'
    for arguments in jobs:
        event=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),arguments=arguments,status='start')
        with journal.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
        print(name,'START',arguments,flush=True)
        result=subprocess.run([sys.executable,*arguments],cwd=base)
        event.update(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
        with journal.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
        if result.returncode: raise SystemExit(result.returncode)
    print(name,'COMPLETE',flush=True)
