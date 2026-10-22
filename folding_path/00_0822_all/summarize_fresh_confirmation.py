"""Summarize the frozen-contrast, fresh-initial-state confirmation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_generalization_report import read,values
from supervised_folding import write_csv


def run(root):
    output=root/'fresh_confirmation';summary=[];uncertainty=[]
    fig,axes=plt.subplots(1,2,figsize=(9,4.6),constrained_layout=True)
    for axis,structure,goal in zip(axes,('seven','torus'),(4.,2.)):
        rows=read(output,f'{structure}_seed*_evaluation.csv')
        seeds=sorted({int(r['seed']) for r in rows});starts=sorted({int(r['start_id']) for r in rows})
        lookup={(int(r['seed']),int(r['start_id']),r['arm']):int(r['success']) for r in rows}
        assert len(seeds)==5 and len(lookup)==5*len(starts)*2,f'{structure} confirmation incomplete'
        rates=[]
        for arm in ('scratch','replay'):
            v=values(rows,goal,16,arm);rates.append(v)
            summary.append(dict(structure=structure,goal=goal,arm=arm,seeds=5,starts=len(starts),mean_success=np.mean(v),seed_min=min(v),seed_max=max(v)))
        difference=np.array([[lookup[s,i,'replay']-lookup[s,i,'scratch'] for i in starts] for s in seeds],float)
        rng=np.random.default_rng(88100);si=rng.integers(5,size=(10000,5));ii=rng.integers(len(starts),size=(10000,len(starts)))
        boot=difference[si[:,:,None],ii[:,None,:]].mean((1,2));low,high=np.quantile(boot,[.025,.975]);sd=difference.mean(1)
        uncertainty.append(dict(structure=structure,mean_difference=float(difference.mean()),bootstrap_low=float(low),bootstrap_high=float(high),
            improved_seeds=int((sd>0).sum()),regressed_seeds=int((sd<0).sum()),
            note='Exploratory percentile interval over training seed and fresh shared initial state. Source/support and topology remain fixed.'))
        for seed in range(5): axis.plot((0,1),[r[seed]*100 for r in rates],'-o',color='#999999',alpha=.5,markersize=4,linewidth=1)
        axis.plot((0,1),[np.mean(r)*100 for r in rates],'-o',color='#2266aa',linewidth=3,markersize=7,label='Five-seed mean')
        axis.set_xticks((0,1),('Scratch, tuned LR','Pretrained + source replay'))
        axis.set(title=f'{structure}: D{goal:g}, {len(starts)} fresh starts',ylabel='Success (%)',ylim=(-3,103));axis.grid(alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('Settings frozen before generating new evaluation initial states')
    fig.supxlabel('16 target teacher paths, 2,000 target updates. Replay adds old-data computation.',fontsize=9)
    for extension in ('png','pdf'): fig.savefig(output/f'confirmation.{extension}',dpi=180)
    plt.close(fig);write_csv(output/'summary.csv',summary)
    (output/'paired_uncertainty.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8');print(summary);print(uncertainty)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
