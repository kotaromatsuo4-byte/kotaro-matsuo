"""Plot all three seven-bar goals, including shared pre-adaptation checkpoints."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(root):
    baseline=root/'modal_u2000_src10000_test';replay=root/'modal_replay0.5_u2000_test'
    output=root/'comparison';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True);summaries=[]
    for name,folder,color in [('Full fine-tuning',baseline,'#2266aa'),('Source replay',replay,'#2b9474')]:
        for axis,goal in zip(axes,(2.,3.,4.)):
            means=[];lows=[];highs=[];budgets=[]
            for budget in (0,4,16,32):
                current=baseline if budget==0 else folder
                rates=[]
                for file in sorted(current.glob(f'seed*_pretrained_n{budget:02d}_evaluation.csv')):
                    with file.open() as f: values=[int(r['success']) for r in csv.DictReader(f) if float(r['goal'])==goal]
                    rates.append(np.mean(values))
                if not rates: continue
                budgets.append(budget);means.append(np.mean(rates)*100);lows.append(min(rates)*100);highs.append(max(rates)*100)
                summaries.append(dict(condition=name,goal=goal,budget=budget,seeds=len(rates),mean_success=np.mean(rates),seed_min=min(rates),seed_max=max(rates)))
            axis.plot(budgets,means,'o-',color=color,label=name)
            axis.fill_between(budgets,lows,highs,color=color,alpha=.12)
    for axis,goal in zip(axes,(2,3,4)):
        axis.set(title=f'D{goal}',xlabel='D4 teacher trajectories',ylabel='Success rate (%)',ylim=(-3,103),xscale='symlog')
        axis.set_xticks([0,4,16,32],['0','4','16','32']);axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('Seven-bar: D4 adaptation and retention of D2/D3',fontsize=13)
    fig.supxlabel('Same 2,000 target updates and 64 target examples per step; replay adds 32 existing source examples (loss weight 0.5).\nLines: seed means; shading: seed min-max. At budget 0 both methods use the same source checkpoints.',fontsize=8)
    for extension in ('png','pdf'): fig.savefig(output/f'replay_retention.{extension}',dpi=180)
    plt.close(fig)
    with (output/'replay_retention.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
