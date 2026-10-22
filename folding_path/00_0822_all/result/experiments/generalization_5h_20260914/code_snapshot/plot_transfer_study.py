"""Aggregate completed controlled-transfer arms without conflating seeds and rollouts."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(directory):
    directory=Path(directory)
    raw=[]
    for path in sorted(directory.glob('seed*_evaluation.csv')):
        with path.open(encoding='utf-8') as f: raw.extend(csv.DictReader(f))
    if not raw: return None
    keys=sorted({(int(r['seed']),r['arm'],int(r['budget']),float(r['goal'])) for r in raw})
    summaries=[]
    for seed,arm,budget,goal in keys:
        rows=[r for r in raw if (int(r['seed']),r['arm'],int(r['budget']),float(r['goal']))==(seed,arm,budget,goal)]
        summaries.append(dict(seed=seed,arm=arm,budget=budget,goal=goal,labels=int(rows[0]['labels']),
                             trials=len(rows),successes=sum(int(r['success']) for r in rows),
                             rate=np.mean([int(r['success']) for r in rows]),
                             median_flatness=np.median([float(r['flatness']) for r in rows]),
                             median_spread_error=np.median([float(r['spread_error']) for r in rows])))
    with (directory/'summary_by_seed.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    goals=sorted({r['goal'] for r in summaries})
    fig,axes=plt.subplots(1,len(goals),figsize=(5*len(goals),4.7),squeeze=False,constrained_layout=True)
    aggregate=[]
    for axis,goal in zip(axes[0],goals):
        for arm,color in [('scratch','#7b7b7b'),('pretrained','#2869a2')]:
            budgets=sorted({r['budget'] for r in summaries if r['goal']==goal and r['arm']==arm})
            means,lows,highs=[],[],[]
            for budget in budgets:
                rows=[r for r in summaries if r['goal']==goal and r['arm']==arm and r['budget']==budget]
                values=[r['rate']*100 for r in rows]
                means.append(np.mean(values));lows.append(min(values));highs.append(max(values))
                aggregate.append(dict(goal=goal,arm=arm,budget=budget,seeds=len(rows),mean_rate=np.mean(values)/100,
                                      seed_min=min(values)/100,seed_max=max(values)/100,
                                      labels=rows[0]['labels']))
                axis.scatter([budget]*len(values),values,color=color,s=12,alpha=.45)
            axis.plot(budgets,means,'o-',color=color,label=arm)
            axis.fill_between(budgets,lows,highs,color=color,alpha=.12)
        axis.set(xscale='symlog',xlabel='Target teacher trajectories (nested prefixes)',ylabel='Success rate (%)',
                 title=f'D={goal:g}',ylim=(-4,104))
        ticks=sorted({r['budget'] for r in summaries})
        axis.set_xticks(ticks,[str(t) for t in ticks]);axis.grid(alpha=.2);axis.legend()
    fig.suptitle(directory.name.replace('_',' '),fontsize=12)
    fig.supxlabel('Dots: individual training seeds. Line: seed mean. Shading: seed min-max, not confidence intervals.',fontsize=9)
    for ext in ('png','pdf'): fig.savefig(directory/f'learning_transfer.{ext}',dpi=180)
    plt.close(fig)
    with (directory/'summary_across_seeds.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    paired=[]
    for goal in goals:
        for budget in sorted({r['budget'] for r in summaries}):
            a={r['seed']:r['rate'] for r in summaries if r['goal']==goal and r['budget']==budget and r['arm']=='scratch'}
            b={r['seed']:r['rate'] for r in summaries if r['goal']==goal and r['budget']==budget and r['arm']=='pretrained'}
            common=sorted(a.keys()&b.keys())
            if common:
                delta=[b[s]-a[s] for s in common]
                paired.append(dict(goal=goal,budget=budget,paired_seeds=len(common),
                                   mean_improvement=float(np.mean(delta)),improved_seeds=int(sum(d>0 for d in delta)),
                                   regressed_seeds=int(sum(d<0 for d in delta)),unchanged_seeds=int(sum(d==0 for d in delta))))
    with (directory/'paired_differences.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    return paired


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    print(json.dumps(summarize(p.parse_args().directory),indent=2))
