"""Smaller structural holdout: seven-only experience -> unseen five bar.

Even source-state collection uses only dense seven-only source policies.
The expanded_source dataset is NOT used: its collectors had seen five bar.
"""
import subprocess,sys
from discovery_transfer import *
from supervised_folding import Geometry,load_targets
from transfer_study import build_records,pack,path_rollout

def main():
    torch.set_num_threads(1)
    output=ROOT/'seven_only_source';output.mkdir(exist_ok=False)
    g=Geometry(BASE/'model/seven_bar_linkage');targets=load_targets(g);records=[];collectors=[]
    for goal,target in targets.items():
        nodes,_=path_rollout(g,g.initial,goal,target=target)
        for label,other in targets.items():records.extend(build_records(g,nodes,other,label,-1))
    for seed in range(5601,5604):
        checkpoint=ROOT/'source'/f'dense_seed{seed}'/'best.pt'
        saved=torch.load(checkpoint,weights_only=False)
        if len(saved['source'])!=1 or Path(saved['source'][0]['path']).name!='source_data.npz':raise ValueError('Collector contains held-out structure experience')
        collectors.append(dict(path=str(checkpoint),sha256=digest(checkpoint),source=saved['source']))
        for goal in (2,3):
            config=config_for(f'seven_d{goal}');env=GoalEnv(config);model=make_model('dense',21,seed)
            model.load_state_dict(saved['actor'],strict=False);agent=make_agent(model,config)
            state=env.reset();nodes=[env.node.copy()];done=False
            with torch.no_grad():
                while not done and len(nodes)<=400:
                    modes=env.get_modes()
                    if modes.shape[1]==0:break
                    a=agent.get_action(state,modes,deterministic=True)
                    state,_,done,_=env.step(a.physical_direction,a.requested_width,a.mode_coefficients);nodes.append(env.node.copy())
            for label,target in targets.items():records.extend(build_records(g,nodes,target,label,-1))
    original=dict(np.load(SOURCE/'source_data.npz'));new=pack(records)
    _,idx=np.unique(np.column_stack([new['node'].reshape(len(new['node']),-1).round(9),new['goal']]),axis=0,return_index=True)
    idx=np.sort(idx);new={k:v[idx] for k,v in new.items()}
    data={k:np.concatenate([original[k],new[k]]) for k in original}
    np.savez_compressed(output/'source_data.npz',**data)
    write_json(output/'manifest.json',dict(original_sha256=digest(SOURCE/'source_data.npz'),added_labels=len(idx),collectors=collectors,held_out_structure='five_bar_linkage',source_structures=['seven_bar_linkage']))
    for seed in range(5601,5604):pretrain('graph',seed,20000,'graph_seven',output,seven_only=True)
    jobs=[]
    for i in range(3):
        checkpoint=str(ROOT/'source'/f'graph_seven_seed{5601+i}'/'best.pt')
        for task in ('seven_d2','seven_d3'):
            jobs.append(dict(task=task,architecture='graph',arm='deterministic',seed=5701+i,source_seed=5601+i,budget=400,tag='source_graph_seven',checkpoint=checkpoint))
        for task in ('five_d2','five_d3'):
            jobs.append(dict(task=task,architecture='graph',arm='deterministic',seed=5701+i,source_seed=5601+i,budget=400,tag='bar_zero',checkpoint=checkpoint))
    for i in range(3):
        for task in ('five_d2','five_d3'):
            for arm in ('scratch','finetune','frozen'):
                job=dict(task=task,architecture='graph',arm=arm,seed=5701+i,source_seed=5601+i,budget=6400,tag='bar_transfer')
                if arm!='scratch':job['checkpoint']=str(ROOT/'source'/f'graph_seven_seed{5601+i}'/'best.pt')
                jobs.append(job)
    queue=ROOT/'bar_transfer_queue.json';write_json(queue,jobs)
    subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(queue),'--workers','1'],cwd=BASE,check=True)

if __name__=='__main__':main()
