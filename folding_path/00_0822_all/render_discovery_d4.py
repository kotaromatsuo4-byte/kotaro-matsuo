"""Same actor: learned D3 and first discovery of previously unseen D4."""
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from discovery_transfer import ROOT,BASE,write_json

def main(bars_only=False):
    source_series='source_projected_bars' if bars_only else 'source_projected_water'
    target_series='projected_bars_transfer' if bars_only else 'projected_transfer'
    stem='bars_only_d4_example' if bars_only else 'unseen_d4_example'
    entries=[('Known D3: mean policy',ROOT/source_series/'seven_d3_projected_deterministic_s5701'),
        ('Unseen D4: frozen stochastic policy',ROOT/target_series/'seven_d4_projected_frozen_s5701')]
    data=[];hashes=[]
    member=np.loadtxt(BASE/'model/seven_bar_linkage/member.dat',delimiter=',',dtype=int).reshape(-1,2)
    for label,path in entries:
        result=json.loads((path/'result.json').read_text());assert result['success'] and result['updates']==0
        hashes.append(result['source_sha256']);data.append((label,np.load(path/'first_hit_path.npz')['nodes'],result))
    assert len(set(hashes))==1
    allnodes=np.concatenate([x[1] for x in data])
    limits=(allnodes[:,:,0].min()-.25,allnodes[:,:,0].max()+.25,allnodes[:,:,2].min()-.2,allnodes[:,:,2].max()+.2)
    def draw(ax,label,nodes,result,index):
        node=nodes[index]
        for a,b in member:ax.plot(node[[a,b],0],node[[a,b],2],'-o',color='#0072B2',lw=2,ms=4)
        ax.set_aspect('equal');ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='x',ylabel='z')
        ax.set_title(f'{label}\nstep {index}/{len(nodes)-1} in successful episode',fontsize=9);ax.grid(alpha=.15)
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for row,(label,nodes,result) in zip(axes,data):
        for ax,index in zip(row,(0,(len(nodes)-1)//2,len(nodes)-1)):draw(ax,label,nodes,result,index)
    hit=data[1][2]
    description='Same D2/D3-only actor' if bars_only else 'Same source-trained actor'
    fig.suptitle(f"{description}; D4 first hit after {hit['steps']} total proposals ({hit['episodes']} episodes), 0 RL updates",fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/(stem+'.png'),dpi=180);plt.close(fig)
    frames=[]
    for fraction in np.linspace(0,1,36):
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),dpi=100)
        for ax,(label,nodes,result) in zip(axes,data):draw(ax,label,nodes,result,round(fraction*(len(nodes)-1)))
        fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    frames[0].save(ROOT/(stem+'.gif'),save_all=True,append_images=frames[1:],duration=100,loop=0)
    write_json(ROOT/(stem+'.json'),dict(source_checkpoint_sha256=hashes[0],source_tasks=['five/seven D2,D3']+([] if bars_only else ['water D1,sqrt2']),
        D4_teacher_used=False,first_hit_total_proposals=hit['steps'],successful_episode_proposals=len(data[1][1])-1,RL_updates=0,
        selection='First prespecified source seed 5601',runs=[str(x[1]) for x in entries]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bars-only',action='store_true');a=p.parse_args();main(a.bars_only)
