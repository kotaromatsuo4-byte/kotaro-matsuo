"""One compute lane: source-only water training, quality, legacy, water transfer."""
import subprocess,sys,time
from discovery_transfer import ROOT,BASE,DEADLINE,pretrain,write_json
import torch

def queue(name,jobs,workers=1):
    file=ROOT/(name+'.json');write_json(file,jobs)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(file),'--workers',str(workers)],cwd=BASE,check=True)

def main():
    torch.set_num_threads(1)
    while not (ROOT/'water_source/manifest.json').exists():
        if time.time()>=DEADLINE-300:return
        time.sleep(2)
    for seed in range(5601,5606):pretrain('graph',seed,20000,'graph_water',ROOT/'water_source',True)
    quality=[];zero=[];adapt=[]
    for i in range(5):
        checkpoint=str(ROOT/'source'/f'graph_water_seed{5601+i}'/'best.pt')
        common=dict(architecture='graph',seed=5701+i,source_seed=5601+i,checkpoint=checkpoint)
        for task in ('five_d2','five_d3','seven_d2','seven_d3','water_d1','water_sqrt2'):
            quality.append(dict(**common,task=task,arm='deterministic',budget=400,tag='source_graph_water'))
        for task in ('seven_d4','torus_d2'):
            zero.append(dict(**common,task=task,arm='deterministic',budget=400,tag='water_zero'))
        for arm in ('finetune','frozen'):
            adapt.append(dict(**common,task='torus_d2',arm=arm,budget=51200,tag='water_transfer'))
    queue('water_quality_queue',quality);queue('water_zero_queue',zero)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(ROOT/'legacy_torus_queue.json'),'--workers','1'],cwd=BASE,check=True)
    queue('water_adaptation_queue',adapt)

if __name__=='__main__':main()
