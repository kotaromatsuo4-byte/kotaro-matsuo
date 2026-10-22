"""Training-loss curves, separate from held-out folding-success curves."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root,structure,budget):
    if structure=='torus':
        folder=root/'torus'
        conditions=[('Scratch, LR 0.0003','graph_src5000_u2000_test','scratch','#aaaaaa'),
                    ('Scratch, tuned LR 0.003','graph_src5000_u2000_test_lr0.003','scratch','#222222'),
                    ('D2/D3 pretrained','graph_src5000_u2000_test','pretrained','#2266aa'),
                    ('D2/D3 + source aggregation','graph_src5000_u2000_test_dagger','pretrained','#cb722a'),
                    ('D2/D3 + extra static training','graph_src5000_u2000_test_static','pretrained','#2b9474')]
    else:
        folder=root
        conditions=[('Scratch, LR 0.0003','modal_u2000_src10000_test','scratch','#aaaaaa'),
                    ('Scratch, tuned LR 0.001','modal_tuned_lr0.001_u2000_test','scratch','#222222'),
                    ('D2/D3 pretrained','modal_u2000_src10000_test','pretrained','#2266aa'),
                    ('D2/D3 + replay','modal_replay0.5_u2000_test','pretrained','#2b9474'),
                    ('Frozen D2/D3 + adapter','discrete_adapter_r8_u2000_test','pretrained','#cb722a')]
    output=folder/'comparison';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for name,directory,arm,color in conditions:
        rows=[]
        for file in (folder/directory).glob(f'seed*_{arm}_n{budget:02d}_learning.csv'):
            with file.open(encoding='utf-8') as f: rows.extend(csv.DictReader(f))
        if not rows: continue
        steps=sorted({int(r['update']) for r in rows})
        for axis,metric in zip(axes,('direction_loss','width_loss')):
            values=[[float(r[metric]) for r in rows if int(r['update'])==step] for step in steps]
            axis.plot(steps,[np.mean(v) for v in values],label=name,color=color)
            axis.fill_between(steps,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.08)
    for axis,title in zip(axes,('Direction squared error','Normalized step-width squared error')):
        axis.set(xlabel='Target optimizer updates',ylabel=title,yscale='log');axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{structure}: training loss with {budget} target teacher trajectories')
    fig.supxlabel('Line: training-seed mean. Shading: seed min-max. Losses are sampled-batch training errors, not test success.',fontsize=8)
    for extension in ('png','pdf'): fig.savefig(output/f'training_losses_n{budget:02d}.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('structure',choices=('seven','torus'));p.add_argument('--budget',type=int,default=16)
    a=p.parse_args();plot(a.root,a.structure,a.budget)
