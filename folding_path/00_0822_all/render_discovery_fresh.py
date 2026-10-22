"""All eight held-out initial shapes, without selecting successful examples."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discovery_transfer import ROOT,BASE,write_json

def main():
    member=np.loadtxt(BASE/'model/five_bar_linkage/member.dat',delimiter=',',dtype=int).reshape(-1,2)
    reference=np.loadtxt(BASE/'model/five_bar_linkage/node_initial.dat',delimiter=',')
    fig,axes=plt.subplots(2,4,figsize=(13,6));summary=[]
    for i,ax in enumerate(axes.ravel()):
        directory=ROOT/'fresh_bar'/f'five_d3_graph_frozen_s{6201+i}'
        result=json.loads((directory/'result.json').read_text())
        nodes=np.load(directory/'first_hit_path.npz')['nodes']
        for index,color,alpha,label in [(0,'#666666',.8,'Initial'),((len(nodes)-1)//2,'#56B4E9',.65,'Halfway'),(len(nodes)-1,'#D55E00',1,'First hit')]:
            node=nodes[index]
            for j,(a,b) in enumerate(member):ax.plot(node[[a,b],0],node[[a,b],2],'-o',color=color,alpha=alpha,ms=3,label=label if j==0 else None)
        ax.set_aspect('equal');ax.set(xlim=(-1.4,2.4),ylim=(-.35,2),xlabel='x',ylabel='z',title=f'Start {i+1}: {result["steps"]} proposals')
        ax.grid(alpha=.15)
        if i==0:ax.legend(fontsize=8)
        summary.append(dict(start=i,first_hit_steps=result['steps'],rms_node_offset_from_reference=float(np.sqrt(np.mean(np.sum((nodes[0]-reference)**2,axis=1)))),initial_z_max=float(nodes[0,:,2].max())))
    fig.suptitle('One seven-only NN; all eight fresh five-bar initial shapes; no weight updates')
    fig.tight_layout();fig.savefig(ROOT/'fresh_bar_shapes.png',dpi=180);plt.close(fig)
    write_json(ROOT/'fresh_bar_geometry_summary.json',summary)

if __name__=='__main__':main()
