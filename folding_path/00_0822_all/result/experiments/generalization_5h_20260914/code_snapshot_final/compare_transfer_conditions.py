"""Compare fixed evaluation conditions and paired crossed-bootstrap differences.

Resample training seeds and test initial states as separate crossed axes. This
does not estimate uncertainty over newly collected support/source datasets.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_condition(directory,arm,goal):
    rows=[]
    for file in sorted(directory.glob(f'seed*_{arm}_n*_evaluation.csv')):
        with file.open(encoding='utf-8') as f:
            rows.extend(r for r in csv.DictReader(f) if float(r['goal'])==goal)
    return rows


def compare(root,structure):
    if structure=='torus':
        folder=root/'torus';goal=2.
        settings=[('Scratch, LR 0.0003','graph_src5000_u2000_test','scratch','#a1a1a1'),
                  ('Scratch, tuned LR 0.003','graph_src5000_u2000_test_lr0.003','scratch','#202020'),
                  ('D2/D3 pretrained','graph_src5000_u2000_test','pretrained','#2266aa'),
                  ('D2/D3 + source aggregation','graph_src5000_u2000_test_dagger','pretrained','#cb722a'),
                  ('D2/D3 + extra static training','graph_src5000_u2000_test_static','pretrained','#2b9474')]
    else:
        folder=root;goal=4.
        settings=[('Scratch, LR 0.0003','modal_u2000_src10000_test','scratch','#aaaaaa'),
                  ('Scratch, tuned LR 0.001','modal_tuned_lr0.001_u2000_test','scratch','#222222'),
                  ('D2/D3, full fine-tuning','modal_u2000_src10000_test','pretrained','#2266aa'),
                  ('D2/D3, source replay','modal_replay0.5_u2000_test','pretrained','#2b9474'),
                  ('D2/D3, frozen + D4 adapter','discrete_adapter_r8_u2000_test','pretrained','#cb722a')]
    output=folder/'comparison';output.mkdir(exist_ok=True)
    all_rows={name:read_condition(folder/directory,arm,goal) for name,directory,arm,color in settings}
    summary=[]
    fig,axis=plt.subplots(figsize=(9,5.2),constrained_layout=True)
    for name,directory,arm,color in settings:
        rows=all_rows[name]
        if not rows: continue
        budgets=sorted({int(r['budget']) for r in rows});means=[];lows=[];highs=[]
        for budget in budgets:
            subset=[r for r in rows if int(r['budget'])==budget]
            seeds=sorted({int(r['seed']) for r in subset})
            values=[np.mean([int(r['success']) for r in subset if int(r['seed'])==s]) for s in seeds]
            means.append(np.mean(values)*100);lows.append(min(values)*100);highs.append(max(values)*100)
            summary.append(dict(condition=name,budget=budget,labels=int(subset[0]['labels']),seeds=len(seeds),
                starts_per_seed=len(subset)//len(seeds),mean_success=np.mean(values),seed_min=min(values),seed_max=max(values)))
        axis.plot(budgets,means,'o-',label=name,color=color,linewidth=2)
        axis.fill_between(budgets,lows,highs,color=color,alpha=.08)
    axis.set(xscale='symlog',xlabel='Target teacher trajectories',ylabel='Success rate (%)',ylim=(-3,103),
        title=f'{structure}: held-out initial states, 2,000 target updates')
    ticks=sorted({r['budget'] for r in summary});axis.set_xticks(ticks,[str(x) for x in ticks])
    axis.grid(alpha=.2);axis.legend(fontsize=9,loc='best')
    fig.supxlabel('Lines: training-seed means. Shading: seed min-max; the same test starts are reused across seeds.',fontsize=8)
    for extension in ('png','pdf'): fig.savefig(output/f'comparison.{extension}',dpi=180)
    plt.close(fig)
    if summary:
        with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    # Primary pretraining versus standard scratch; additional contrasts versus tuned scratch.
    contrasts=([(settings[2][0],settings[1][0]),(settings[3][0],settings[1][0]),
                (settings[4][0],settings[1][0]),(settings[3][0],settings[4][0])] if structure=='torus' else
               [(settings[2][0],settings[0][0]),(settings[2][0],settings[1][0])])
    paired=[];rng=np.random.default_rng(77001)
    for treatment,control in contrasts:
        a,b=all_rows[treatment],all_rows[control]
        budgets=sorted({int(r['budget']) for r in a}&{int(r['budget']) for r in b})
        for budget in budgets:
            av={(int(r['seed']),int(r['start_id'])):int(r['success']) for r in a if int(r['budget'])==budget}
            bv={(int(r['seed']),int(r['start_id'])):int(r['success']) for r in b if int(r['budget'])==budget}
            common=av.keys()&bv.keys();seeds=sorted({s for s,i in common});starts=sorted({i for s,i in common})
            if not seeds or not starts: continue
            if any((s,i) not in common for s in seeds for i in starts): raise ValueError('Unbalanced paired evaluations')
            differences=np.array([[av[s,i]-bv[s,i] for i in starts] for s in seeds],float)
            si=rng.integers(len(seeds),size=(10000,len(seeds)))
            ii=rng.integers(len(starts),size=(10000,len(starts)))
            boot=differences[si[:,:,None],ii[:,None,:]].mean((1,2))
            low,high=np.quantile(boot,[.025,.975]);seed_delta=differences.mean(1)
            paired.append(dict(treatment=treatment,control=control,budget=budget,seeds=len(seeds),starts=len(starts),
                mean_difference=differences.mean(),bootstrap_low=low,bootstrap_high=high,
                improved_seeds=int((seed_delta>0).sum()),regressed_seeds=int((seed_delta<0).sum())))
    if paired:
        with (output/'paired_crossed_bootstrap.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    (output/'uncertainty_note.txt').write_text('Exploratory percentile intervals from independently resampling training seeds and shared test starts. The support/source split is fixed; uncertainty over alternative training datasets or topologies is not included. Multiple budget comparisons are not multiplicity-adjusted.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('structure',choices=('seven','torus'))
    a=p.parse_args();compare(a.root,a.structure)
