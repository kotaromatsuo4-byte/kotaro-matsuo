"""Recorded seven-only actor -> unseen five-bar zero-shot success example."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from discovery_transfer import ROOT,BASE,write_json

def main():
    entries=[('Known structure: seven bar, D3',ROOT/'source_graph_seven/seven_d3_graph_deterministic_s5701','seven_bar_linkage'),
             ('Unseen structure: five bar, D3',ROOT/'bar_zero/five_d3_graph_deterministic_s5701','five_bar_linkage')]
    data=[];hashes=[]
    for label,directory,structure in entries:
        result=json.loads((directory/'result.json').read_text());assert result['success'] and result['updates']==0
        design=json.loads((directory/'design.json').read_text());hashes.append(design['source_sha256'])
        nodes=np.load(directory/'first_hit_path.npz')['nodes']
        member=np.loadtxt(BASE/'model'/structure/'member.dat',delimiter=',',dtype=int).reshape(-1,2)
        data.append((label,nodes,member))
    assert len(set(hashes))==1,'The policy weights must be identical across structures'
    def draw(ax,nodes,member,index,label):
        node=nodes[index]
        for a,b in member:ax.plot(node[[a,b],0],node[[a,b],2],'-',color='#0072B2',lw=2)
        ax.scatter(node[:,0],node[:,2],s=30,color='#D55E00',zorder=3)
        ax.set_aspect('equal');ax.set_xlim(nodes[:,:,0].min()-.25,nodes[:,:,0].max()+.25);ax.set_ylim(-.2,max(.3,nodes[:,:,2].max()+.2))
        ax.set_title(f'{label}\nstep {index}/{len(nodes)-1}',fontsize=10);ax.set_xlabel('x');ax.set_ylabel('z');ax.grid(alpha=.15)
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for row,(label,nodes,member) in zip(axes,data):
        for ax,index in zip(row,(0,(len(nodes)-1)//2,len(nodes)-1)):draw(ax,nodes,member,index,label)
    fig.suptitle('Same source-trained actor weights; no five-bar teacher data or weight updates',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'bar_transfer_example.png',dpi=180);plt.close(fig)
    frames=[]
    for fraction in np.linspace(0,1,36):
        fig,axes=plt.subplots(1,2,figsize=(8,3.3),dpi=100)
        for ax,(label,nodes,member) in zip(axes,data):draw(ax,nodes,member,round(fraction*(len(nodes)-1)),label)
        fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    frames[0].save(ROOT/'bar_transfer_example.gif',save_all=True,append_images=frames[1:],duration=90,loop=0)
    write_json(ROOT/'bar_transfer_example.json',dict(selection='First prespecified source seed 5601 / target seed 5701, not the fastest seed',actor_checkpoint_sha256=hashes[0],source_reference=str(entries[0][1]),unseen_structure_run=str(entries[1][1]),target_teacher_used=False))

if __name__=='__main__':main()
