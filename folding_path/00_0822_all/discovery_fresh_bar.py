"""Fresh initial conditions, frozen source seed 5601, paired scratch controls."""
import subprocess,sys
from discovery_transfer import *
from supervised_folding import Geometry,generate_starts

def main():
    output=ROOT/'fresh_bar_inputs';output.mkdir(exist_ok=False)
    checkpoint=ROOT/'source/graph_seven_seed5601/best.pt'
    write_json(output/'source_frozen_before_starts.json',dict(source_seed=5601,selection='First prespecified source seed, not fastest source seed',checkpoint=str(checkpoint),sha256=digest(checkpoint),initial_seed=59501))
    g=Geometry(BASE/'model/five_bar_linkage')
    starts=generate_starts(g,8,np.random.default_rng(59501),20,80,excluded=(g.initial,))
    jobs=[];manifest=[]
    for i,node in enumerate(starts):
        path=output/f'start{i:02d}.dat';np.savetxt(path,node,delimiter=',',fmt='%.15g')
        manifest.append(dict(index=i,path=str(path),sha256=digest(path),initial_metrics=g.metrics(node,3)))
        for arm in ('scratch','frozen'):
            job=dict(task='five_d3',architecture='graph',arm=arm,seed=6201+i,source_seed=5601,budget=6400,tag='fresh_bar',initial=str(path))
            if arm=='frozen':job['checkpoint']=str(checkpoint)
            jobs.append(job)
    write_json(output/'manifest.json',manifest)
    queue=ROOT/'fresh_bar_queue.json';write_json(queue,jobs)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(queue),'--workers','1'],cwd=BASE,check=True)

if __name__=='__main__':
    torch.set_num_threads(1);main()
