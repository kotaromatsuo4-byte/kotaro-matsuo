"""Progress at episode endpoints; independent minima are not joint success."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discovery_transfer import ROOT,write_json


def main():
    fig,axes=plt.subplots(1,3,figsize=(14,4));summary=[]
    for arm,color in [('scratch','#555555'),('finetune','#0072B2'),('frozen','#D55E00')]:
        directory=ROOT/('projected_transfer' if arm=='frozen' else 'torus_depth')/f'torus_d2_projected_{arm}_s5701'
        file=directory/'episodes.csv'
        if not file.exists():continue
        records=[]
        with file.open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:records.append([int(r['cumulative_steps']),float(r['flatness']),float(r['spread_error'])])
                except (ValueError,TypeError):continue
        if not records:continue
        a=np.asarray(records);metrics=[a[:,1],a[:,2],np.maximum(a[:,1]/.001,a[:,2]/.05)]
        for ax,y in zip(axes,metrics):
            ax.plot(a[:,0],np.maximum(np.minimum.accumulate(y),1e-10),color=color,label=f'{arm}: {int(a[-1,0]):,} observed')
        best=int(np.argmin(metrics[2]));summary.append(dict(arm=arm,observed_proposals=int(a[-1,0]),
            minimum_terminal_flatness=float(metrics[0].min()),minimum_terminal_D_error=float(metrics[1].min()),
            minimum_joint_threshold_ratio=float(metrics[2][best]),joint_best_proposals=int(a[best,0]),
            joint_best_flatness=float(a[best,1]),joint_best_D_error=float(a[best,2])))
    for ax,title,threshold in zip(axes,('Best terminal flatness','Best terminal D error','Best joint threshold ratio'),(.001,.05,1.)):
        ax.axhline(threshold,color='#009E73',ls='--',lw=1,label='Success threshold')
        ax.set(title=title,xlabel='Cumulative proposals',xscale='log',yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Torus D2: same projected actor; matched seed; episode endpoints',fontsize=13)
    fig.tight_layout();fig.savefig(ROOT/'torus_depth_progress.png',dpi=180);plt.close(fig)
    write_json(ROOT/'torus_depth_progress.json',dict(rows=summary,
        interpretation='First two minima may come from different states. Joint ratio is max(flatness/.001,D_error/.05) at the same episode endpoint. This diagnostic is not a substitute for first-hit detection at every proposal.'))


if __name__=='__main__':main()
