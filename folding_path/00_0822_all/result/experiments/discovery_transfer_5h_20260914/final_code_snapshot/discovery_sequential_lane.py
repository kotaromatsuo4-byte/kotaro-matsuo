"""New RL first hit -> known teacher experience -> still-unseen nine-node task.

The newly discovered seven-D4 coordinates are used only as SOURCE coordinates
in this separate stage. No nine-node target coordinates or teacher are used.
"""
import shutil,subprocess,sys
from discovery_transfer import *
from supervised_folding import Geometry,generate_starts
from transfer_study import build_records,pack,path_rollout

def prepare():
    output=ROOT/'seven_discovered_d4_source';output.mkdir(exist_ok=False)
    discovery=ROOT/'main/seven_d4_dense_scratch_s5704'
    result=json.loads((discovery/'result.json').read_text());assert result['success']
    g=Geometry(BASE/'model/seven_bar_linkage');target=np.loadtxt(discovery/'first_hit.dat',delimiter=',')
    assert g.metrics(target,4.)['success']
    targetfile=output/'known_d4_from_first_hit.dat';shutil.copy2(discovery/'first_hit.dat',targetfile)
    original=dict(np.load(ROOT/'seven_only_source/source_data.npz'))
    starts=generate_starts(g,32,np.random.default_rng(59701),20,80,excluded=(g.initial,))
    np.savez_compressed(output/'starts.npz',starts=starts)
    records=[];rollouts=[]
    for index,node in [(-1,g.initial),*enumerate(starts)]:
        if time.time()>=DEADLINE-300:raise RuntimeError('No time to finish source generation')
        nodes,metrics=path_rollout(g,node,4.,target=target,max_steps=400)
        records.extend(build_records(g,nodes,target,4.,index));rollouts.append(dict(path_id=index,**metrics))
        print('New known D4 teacher',index,'records',len(records),'success',metrics['success'],flush=True)
    new=pack(records);combined={k:np.concatenate([original[k],new[k]]) for k in original}
    np.savez_compressed(output/'source_data.npz',**combined);write_csv(output/'teacher_rollouts.csv',rollouts)
    write_json(output/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        discovery_result=str(discovery/'result.json'),discovery_result_sha256=digest(discovery/'result.json'),
        discovery_coordinates=str(discovery/'first_hit.dat'),discovery_coordinates_sha256=digest(discovery/'first_hit.dat'),
        copied_target=str(targetfile),copied_target_sha256=digest(targetfile),RL_weights_used=False,
        original_source=str(ROOT/'seven_only_source/source_data.npz'),original_source_sha256=digest(ROOT/'seven_only_source/source_data.npz'),
        original_labels=len(original['node']),new_D4_labels=len(new['node']),source_goals=[2,3,4],source_nodes=7,
        starts_seed=59701,train_path_ids='-1,0..23 plus original training groups',validation_path_ids='24..31',
        unknown_nine_teacher_used=False,source_data_sha256=digest(output/'source_data.npz')))

def queue(name,jobs):
    path=ROOT/(name+'.json');write_json(path,jobs)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(path),'--workers','1'],cwd=BASE,check=True)

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'sequential_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_discovery='main/seven_d4_dense_scratch_s5704',source_seeds=[5601,5602,5603],updates=20000,
        source_goals=[2,3,4],source_structure='seven',unknown_structure='nine',unknown_goals=[3,4],budget=6400,
        selection='First three source seeds; source-path validation only',
        warning='D4 is known source experience ONLY in this stage; this is not a D2/D3-to-D4 claim.',
        RL_discovery_weights_transferred=False,target_nine_teacher_used=False))
    prepare()
    for seed in range(5601,5604):pretrain('graph',seed,20000,'graph_seven_d4',ROOT/'seven_discovered_d4_source',False,True,True)
    quality=[];zero=[];adapt=[]
    for i in range(3):
        common=dict(architecture='graph',seed=5701+i,source_seed=5601+i,
            checkpoint=str(ROOT/'source'/f'graph_seven_d4_seed{5601+i}'/'best.pt'))
        for task in ('seven_d2','seven_d3','seven_d4'):
            quality.append(dict(**common,task=task,arm='deterministic',budget=400,tag='source_seven_discovered_d4'))
        for task in ('nine_d3','nine_d4'):
            zero.append(dict(**common,task=task,arm='deterministic',budget=400,tag='sequential_nine_zero'))
            for arm in ('finetune','frozen'):
                adapt.append(dict(**common,task=task,arm=arm,budget=6400,tag='sequential_nine'))
            if task=='nine_d4':
                base=dict(task=task,architecture='graph',seed=5701+i,source_seed=5601+i,budget=6400)
                adapt.append(dict(**base,arm='scratch',tag='sequential_nine_control'))
                adapt.append(dict(**base,arm='frozen',tag='sequential_nine_old',
                    checkpoint=str(ROOT/'source'/f'graph_seven_seed{5601+i}'/'best.pt')))
    queue('sequential_quality_queue',quality);queue('sequential_zero_queue',zero);queue('sequential_adaptation_queue',adapt)

if __name__=='__main__':main()
