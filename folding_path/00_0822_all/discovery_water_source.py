"""Plane-folding source experience from archived WATERBOMB RL first hits.

No torus success file or target coordinate is read by this module.
"""
import shutil
from discovery_transfer import *
from graph_transfer import SpatialGeometry
from supervised_folding import generate_starts
from transfer_study import path_rollout,build_records,pack

def prepare():
    output=ROOT/'water_source'
    if (output/'manifest.json').exists():raise FileExistsError('Completed source data already exists')
    output.mkdir(exist_ok=True)
    for name in ('source_data.npz','five_source_data.npz'):shutil.copy2(ROOT/'expanded_source'/name,output/name)
    archive=BASE/'result/experiments/20260831_water_alpha0p05_tuning_v1/full'
    selections=[(1.,archive/'water_d1/C_conservative/seed8102',111,395),
                (float(np.sqrt(2)),archive/'water_sqrt2/A_base/seed8101',23,256)]
    g=SpatialGeometry(BASE/'model/waterbomb',step_width=.3)
    targets={};provenance=[];records=[];rollouts=[]
    for goal,directory,episode,steps in selections:
        summary=json.loads((directory/'summary.json').read_text())
        if summary['first_strict_success_episode']!=episode or summary['first_strict_success_episode_steps']!=steps:raise ValueError('Not the archived first hit')
        pathdir=directory/f'best_episode{episode:05d}'
        path=pathdir/f'step{steps}.dat';target=np.loadtxt(path)
        if not g.metrics(target,goal)['success']:raise ValueError('Archived source target fails strict criteria')
        targets[goal]=target
        provenance.append(dict(goal=goal,source=str(path),sha256=digest(path),summary=str(directory/'summary.json'),summary_sha256=digest(directory/'summary.json'),metrics=g.metrics(target,goal)))
    starts=generate_starts(g,16,np.random.default_rng(59001),5,40)
    np.savez_compressed(output/'starts.npz',source=starts)
    for i,initial in [(-1,g.initial),*enumerate(starts)]:
        for goal,target in targets.items():
            path,metrics=path_rollout(g,initial,goal,target=target,max_steps=400)
            for label,other in targets.items():records.extend(build_records(g,path,other,label,i))
            rollouts.append(dict(path_id=i,goal=goal,**metrics))
        print('water source',i,'records',len(records),flush=True)
    # Teacher labels along the known RL first-hit routes cover source states
    # outside the gradient teacher's own basin. These remain training-only.
    np.savez_compressed(output/'teacher_only_records.npz',**pack(records))
    write_csv(output/'teacher_rollouts.csv',rollouts)
    for goal,directory,episode,steps in selections:
        pathdir=directory/f'best_episode{episode:05d}'
        indices=np.unique(np.linspace(0,steps,32).round().astype(int))
        nodes=[np.loadtxt(pathdir/f'step{i}.dat') for i in indices]
        for label,target in targets.items():records.extend(build_records(g,nodes,target,label,-2))
        print('archived water route',goal,'sampled states',len(nodes),'records',len(records),flush=True)
    np.savez_compressed(output/'water_source_data.npz',**pack(records))
    write_csv(output/'teacher_rollouts.csv',rollouts)
    write_json(output/'manifest.json',dict(provenance=provenance,labels=len(records),seed=59001,starts=16,archived_route_sampling='32 evenly spaced states per known RL first-hit route',train_path_ids='-2,-1,0..11',validation_path_ids='12..15',target_torus_used=False,sha256=digest(output/'water_source_data.npz')))

if __name__=='__main__':
    torch.set_num_threads(1);prepare()
