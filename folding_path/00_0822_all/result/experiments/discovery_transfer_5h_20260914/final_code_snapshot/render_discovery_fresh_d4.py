"""Show every new D4 initial condition, including any unsuccessful one."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discovery_transfer import ROOT,BASE,write_json
from report_discovery_findings import read_results,curve,COLORS


def main():
    member=np.loadtxt(BASE/'model/seven_bar_linkage/member.dat',delimiter=',',dtype=int).reshape(-1,2)
    entries=[];all_nodes=[]
    for i in range(4):
        initial=np.loadtxt(ROOT/'fresh_d4_inputs'/f'start{i:02d}.dat',delimiter=',')
        directory=ROOT/'fresh_d4'/f'seven_d4_projected_frozen_s{6901+i}'
        file=directory/'result.json';result=json.loads(file.read_text()) if file.exists() else None
        nodes=np.load(directory/'first_hit_path.npz')['nodes'] if result and result['success'] else initial[None]
        entries.append((i,nodes,result));all_nodes.extend(nodes)
    all_nodes=np.asarray(all_nodes);limits=(all_nodes[:,:,0].min()-.25,all_nodes[:,:,0].max()+.25,all_nodes[:,:,2].min()-.25,all_nodes[:,:,2].max()+.25)
    fig,axes=plt.subplots(2,2,figsize=(9,7));summary=[]
    for ax,(i,nodes,result) in zip(axes.ravel(),entries):
        frames=[(0,'#666666',.7,'Initial')]
        if result and result['success']:frames.extend([((len(nodes)-1)//2,'#56B4E9',.65,'Halfway'),(len(nodes)-1,'#D55E00',1.,'First hit')])
        for index,color,alpha,label in frames:
            node=nodes[index]
            for j,(a,b) in enumerate(member):ax.plot(node[[a,b],0],node[[a,b],2],'-o',color=color,alpha=alpha,ms=3,label=label if j==0 else None)
        status=f"{result['steps']} proposals" if result and result['success'] else (f"no hit: {result['steps']} observed" if result else 'pending')
        ax.set_aspect('equal');ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='x',ylabel='z',title=f'Start {i+1}: {status}');ax.grid(alpha=.15)
        summary.append(dict(start=i,success=result['success'] if result else None,steps=result['steps'] if result else None))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('One D2/D3-only NN; all four new seven-bar starts; unseen D4\nPaths: successful episode only. Counts include previous failed episodes.',fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/'fresh_d4_shapes.png',dpi=180);plt.close(fig)
    write_json(ROOT/'fresh_d4_shapes.json',summary)
    rows=read_results();fig,ax=plt.subplots(figsize=(7,4))
    for arm in ('scratch','frozen'):
        group=[r for r in rows if r['series']=='fresh_d4' and r['arm']==arm]
        if group:curve(ax,group,arm,COLORS[arm],6400)
    ax.set_title('Four new seven-bar starts; one fixed D2/D3-only source NN')
    fig.tight_layout();fig.savefig(ROOT/'fresh_d4_first_discovery.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
