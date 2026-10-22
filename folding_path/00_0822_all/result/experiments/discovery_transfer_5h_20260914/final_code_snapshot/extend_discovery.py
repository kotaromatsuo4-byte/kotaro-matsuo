"""Prespecified deeper D4 screen for the first three paired seeds."""
import csv,json,shutil,time
from discovery_transfer import ROOT,DEADLINE,write_json
from discovery_queue import launch

def main():
    ledger=ROOT/'extended_ledger.jsonl'
    for seed in range(5701,5704):
        for arm in ('scratch','finetune'):
            previous=ROOT/'main'/f'seven_d4_dense_{arm}_s{seed}'
            while not (previous/'result.json').exists() and time.time()<DEADLINE-180:time.sleep(1)
            if time.time()>=DEADLINE-180:return
            result=json.loads((previous/'result.json').read_text())
            if result['success']:
                destination=ROOT/'extended'/previous.name
                shutil.copytree(previous,destination)
                result.update(budget=256000,inherited_success_from=str(previous),additional_steps=0)
                write_json(destination/'result.json',result)
                design=json.loads((destination/'design.json').read_text());design.update(budget=256000,inherited_success_from=str(previous))
                write_json(destination/'design.json',design)
                outcome=dict(seed=seed,arm=arm,status='inherited_earlier_first_hit')
            else:
                with (previous/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
                resumable=result['episodes']==result['updates']*8 and rows[-1]['reason']=='timeout'
                job=dict(task='seven_d4',architecture='dense',arm=arm,seed=seed,source_seed=seed-100,budget=256000,tag='extended')
                if resumable:job['resume']=str(previous)
                outcome=launch(job);outcome['resumable']=resumable
            with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(outcome)+'\n')
            print(json.dumps(outcome),flush=True)

if __name__=='__main__':main()
