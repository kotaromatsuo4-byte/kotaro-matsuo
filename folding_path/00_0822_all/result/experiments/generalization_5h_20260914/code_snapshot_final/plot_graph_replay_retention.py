"""Show the same graph weights retaining bar tasks after torus adaptation."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_generalization_report import read,values


def run(root):
    folder=root/'torus';output=folder/'comparison';output.mkdir(exist_ok=True)
    source=read(root/'graph_source_evaluation_test'/'dagger')
    target=read(folder/'graph_src5000_u2000_test_dagger')
    replay_dir=folder/'graph_dagger_replay0.25_u2000_test'
    retained=read(replay_dir,'seed*_n16_seven_evaluation.csv');forgotten=read(replay_dir,'seed*_baseline_seven_evaluation.csv')
    replayed=read(replay_dir,'seed*_torus_evaluation.csv')
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.6),constrained_layout=True)
    for axis,title,goal,groups in [(axes[0],'Seven-bar D2',2,[(source,0),(forgotten,16),(retained,16)]),
        (axes[1],'Seven-bar D3',3,[(source,0),(forgotten,16),(retained,16)]),
        (axes[2],'Torus D2',2,[(target,0),(target,16),(replayed,16)])]:
        rates=[values(rows,goal,budget,'pretrained') for rows,budget in groups]
        assert all(len(v)==5 for v in rates),'Five-seed graph replay comparison is incomplete'
        for seed in range(5): axis.plot(range(3),[v[seed]*100 for v in rates],'-o',color='#999999',alpha=.4,linewidth=.8,markersize=3)
        axis.plot(range(3),[np.mean(v)*100 for v in rates],'-o',color='#2266aa',linewidth=2.5,label='Five-seed mean')
        axis.set_xticks(range(3),['Before torus','Torus fine-tuning','Torus + bar replay'],rotation=15)
        axis.set(title=title,ylabel='Test success (%)',ylim=(-3,103));axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('One graph network: source retention and structural transfer (16 torus teacher paths)')
    fig.supxlabel('Adaptation arms independently start from the same source weights. Gray: matched seeds. Replay adds computation.',fontsize=9)
    for extension in ('png','pdf'): fig.savefig(output/f'graph_replay_retention.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
