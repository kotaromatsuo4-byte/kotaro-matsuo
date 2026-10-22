"""Separate basis-invariant representation screen, source-only selection."""
import subprocess,sys
from discovery_transfer import *

def queue(name,jobs):
    file=ROOT/(name+'.json');write_json(file,jobs)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(file),'--workers','1'],cwd=BASE,check=True)

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'projected_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Mode-basis-invariant nodal field, same legacy Gaussian action sampler and RL update',
        sources=['five D2,D3','seven D2,D3','water D1,sqrt2'],source_seeds=[5601,5602,5603],
        updates=20000,selection='minimum held-out source-path loss',unknowns=['seven D4','torus D2'],
        target_budget=51200,teacher_on_unknown=False,primary_comparison_unchanged=True))
    for seed in range(5601,5604):pretrain('projected',seed,20000,'projected_water',ROOT/'water_source',True)
    quality=[];zero=[];adapt=[]
    for i in range(3):
        checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt')
        common=dict(architecture='projected',seed=5701+i,source_seed=5601+i,checkpoint=checkpoint)
        for task in ('five_d2','five_d3','seven_d2','seven_d3','water_d1','water_sqrt2'):
            quality.append(dict(**common,task=task,arm='deterministic',budget=400,tag='source_projected_water'))
        for task in ('seven_d4','torus_d2'):
            zero.append(dict(**common,task=task,arm='deterministic',budget=400,tag='projected_zero'))
        for task in ('seven_d4','torus_d2'):
            for arm in ('scratch','finetune','frozen'):
                job=dict(**common,task=task,arm=arm,budget=51200,tag='projected_transfer')
                if arm=='scratch':job.pop('checkpoint')
                adapt.append(job)
    queue('projected_quality_queue',quality);queue('projected_zero_queue',zero);queue('projected_adaptation_queue',adapt)

if __name__=='__main__':main()
