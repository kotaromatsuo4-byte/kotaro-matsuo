"""Common-horizon comparison on all eight fresh starts, with failures marked."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from discovery_transfer import ROOT

def main():
    fig,ax=plt.subplots(figsize=(10,4.8));horizon=400;handles=[]
    for series,arm,base,label,color,offset in [('fresh_bar','frozen',6201,'Seven-only NN (frozen)','#D55E00',-.18),
            ('gradient_fresh_bar','gradient_w1p0',6401,'Local geometric gradient','#009E73',0),
            ('fresh_bar','scratch',6201,'Untrained graph + RL','#555555',.18)]:
        rows=[]
        for p in (ROOT/series).glob('*/result.json'):
            r=json.loads(p.read_text())
            if r['arm']==arm:rows.append(r)
        handles.append(Line2D([],[],color=color,marker='o',linestyle='None',label=label))
        for r in rows:
            hit=r['success'] and r['steps']<=horizon
            ax.scatter(r['seed']-base+1+offset,r['steps'] if hit else horizon,
                marker='o' if hit else 'x',s=70,color=color,zorder=3)
    ax.axhline(horizon,color='black',ls=':',alpha=.5)
    ax.text(8.55,horizon,'No hit\nwithin 400',va='center',fontsize=9)
    ax.set(xlim=(.5,9.5),ylim=(15,600),yscale='log',xticks=range(1,9),xlabel='Fresh initial shape',ylabel='Proposals to first discovery',
        title='Same eight five-bar D3 starts; common 400-proposal horizon')
    ax.grid(alpha=.2);fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(ROOT/'fresh_bar_controls.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
