"""Freeze a newly constructed nine-node linkage before any policy evaluation."""
import subprocess,sys
from discovery_transfer import *

def main():
    directory=ROOT/'holdout_nine';directory.mkdir(exist_ok=False)
    dx=np.array([.04,.16,.30,.50])
    left=np.column_stack([np.r_[-1,-1+np.cumsum(dx)],np.zeros(5),np.r_[0,np.cumsum(np.sqrt(1-dx**2))]])
    right=left[-2::-1].copy();right[:,0]*=-1
    nodes=np.vstack([left,right]);members=np.column_stack([np.arange(8),np.arange(1,9)])
    fixed=np.array(sorted([(i,1) for i in range(9)]+[(0,0),(0,2),(8,0),(8,2)]))
    assert np.allclose(np.linalg.norm(nodes[members[:,0]]-nodes[members[:,1]],axis=1),1,atol=1e-12)
    np.savetxt(directory/'node_initial.dat',nodes,delimiter=',',fmt='%.15g')
    np.savetxt(directory/'member.dat',members,delimiter=',',fmt='%d');np.savetxt(directory/'fix.dat',fixed,delimiter=',',fmt='%d')
    write_json(directory/'manifest.json',dict(rule='Unit links; left x-increments [.04,.16,.30,.50], positive z-increments sqrt(1-dx^2); mirror about x=0.',goal=3,no_target_coordinates_created=True,source='seven-only graph actors, source seeds 5601..5603',frozen_before_evaluation=True,hashes={n:digest(directory/n) for n in ('node_initial.dat','member.dat','fix.dat')}))
    jobs=[]
    for i in range(3):
        checkpoint=str(ROOT/'source'/f'graph_seven_seed{5601+i}'/'best.pt')
        for arm in ('untrained_deterministic','deterministic'):
            job=dict(task='nine_d3',architecture='graph',arm=arm,seed=5701+i,source_seed=5601+i,budget=400,tag='nine_zero')
            if arm=='deterministic':job['checkpoint']=checkpoint
            jobs.append(job)
    for i in range(3):
        for arm in ('scratch','finetune','frozen'):
            job=dict(task='nine_d3',architecture='graph',arm=arm,seed=5701+i,source_seed=5601+i,budget=6400,tag='nine_transfer')
            if arm!='scratch':job['checkpoint']=str(ROOT/'source'/f'graph_seven_seed{5601+i}'/'best.pt')
            jobs.append(job)
    queue=ROOT/'nine_queue.json';write_json(queue,jobs)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(queue),'--workers','1'],cwd=BASE,check=True)

if __name__=='__main__':main()
