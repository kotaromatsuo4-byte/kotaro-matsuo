"""Same actor on known seven D3 and previously untaught nine D4."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from discovery_transfer import ROOT,BASE,write_json

def main():
    entries=[('Known seven-bar D3',ROOT/'source_projected_water/seven_d3_projected_deterministic_s5702',BASE/'model/seven_bar_linkage'),
        ('Unseen nine-bar D4',ROOT/'projected_nine/nine_d4_projected_frozen_s5702',ROOT/'holdout_nine')]
    data=[];hashes=[]
    for label,path,geometry in entries:
        result=json.loads((path/'result.json').read_text());assert result['success'] and result['updates']==0
        hashes.append(result['source_sha256']);data.append((label,np.load(path/'first_hit_path.npz')['nodes'],
            np.loadtxt(geometry/'member.dat',delimiter=',',dtype=int).reshape(-1,2),result))
    assert len(set(hashes))==1
    xmin=min(x[1][:,:,0].min() for x in data)-.25;xmax=max(x[1][:,:,0].max() for x in data)+.25
    zmin=min(x[1][:,:,2].min() for x in data)-.2;zmax=max(x[1][:,:,2].max() for x in data)+.2
    def draw(ax,label,nodes,member,result,index):
        node=nodes[index]
        for a,b in member:ax.plot(node[[a,b],0],node[[a,b],2],'-o',color='#0072B2',lw=2,ms=4)
        ax.set_aspect('equal');ax.set(xlim=(xmin,xmax),ylim=(zmin,zmax),xlabel='x',ylabel='z')
        ax.set_title(f'{label}\nstep {index}/{len(nodes)-1} in successful episode',fontsize=9);ax.grid(alpha=.15)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,(label,nodes,member,result) in zip(axes,data):
        for ax,index in zip(row,(0,(len(nodes)-1)//2,len(nodes)-1)):draw(ax,label,nodes,member,result,index)
    fig.suptitle('Same frozen actor; no nine-bar or D4 teacher data; nine D4 first hit: 1,583 proposals, 0 updates',fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/'unseen_nine_example.png',dpi=180);plt.close(fig)
    frames=[]
    for fraction in np.linspace(0,1,36):
        fig,axes=plt.subplots(1,2,figsize=(8,4),dpi=100)
        for ax,(label,nodes,member,result) in zip(axes,data):draw(ax,label,nodes,member,result,round(fraction*(len(nodes)-1)))
        fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    frames[0].save(ROOT/'unseen_nine_example.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    write_json(ROOT/'unseen_nine_example.json',dict(source_checkpoint_sha256=hashes[0],source_seed=5602,search_seed=5702,
        source_tasks=['five/seven D2,D3','water D1,sqrt2'],nine_teacher_used=False,D4_teacher_used=False,
        first_hit_total_proposals=1583,successful_episode_proposals=len(data[1][1])-1,RL_updates=0,
        selection='First observed nine-D4 success; all three source seeds are reported separately'))

if __name__=='__main__':main()
