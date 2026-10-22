"""Bounded two-process experiment queue, with per-run logs and hard deadline."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from discovery_transfer import ROOT,BASE,DEADLINE,write_json

def launch(job):
    if time.time()>=DEADLINE-180:return dict(job=job,status='not_started_deadline')
    if job['arm']!='scratch' and not job.get('checkpoint'):
        ready=ROOT/'source'/f"{job['architecture']}_seed{job['source_seed']}"/'summary.json'
        while not ready.exists():
            if time.time()>=DEADLINE-180:return dict(job=job,status='source_not_ready_deadline')
            time.sleep(1)
    tag=job.get('tag','main')
    name=f"{tag}_{job['task']}_{job['architecture']}_{job['arm']}_s{job['seed']}"
    directory=ROOT/tag/f"{job['task']}_{job['architecture']}_{job['arm']}_s{job['seed']}"
    if (directory/'result.json').exists():return dict(job=job,status='already_complete')
    args=[sys.executable,str(BASE/'discovery_transfer.py'),'run']
    for key,value in job.items():args.extend(['--'+key.replace('_','-'),str(value)])
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs'/f'{name}.log').open('w',encoding='utf-8') as f:
        process=subprocess.Popen(args,cwd=BASE,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=max(1,DEADLINE-60-time.time()))
        except subprocess.TimeoutExpired:process.terminate();process.wait();code=-99
    return dict(job=job,status='completed' if code==0 else 'failed',returncode=code)

def main():
    p=argparse.ArgumentParser();p.add_argument('queue');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    jobs=json.loads(Path(a.queue).read_text(encoding='utf-8'))
    ledger=ROOT/(Path(a.queue).stem+'_ledger.jsonl')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(launch,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result()
            with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps(result),flush=True)

if __name__=='__main__':main()
