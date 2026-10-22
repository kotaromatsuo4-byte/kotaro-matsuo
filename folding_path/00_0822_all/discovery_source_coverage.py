"""Add known-task teacher paths and source-student visited states only."""
from discovery_transfer import *
from supervised_folding import Geometry,load_targets
from transfer_study import build_records,pack,path_rollout

def generate():
    output=ROOT/'expanded_source';output.mkdir(exist_ok=False)
    manifest=[]
    for structure,name in [('five_bar_linkage','five_source_data.npz'),('seven_bar_linkage','source_data.npz')]:
        g=Geometry(BASE/'model'/structure);targets=load_targets(g)
        original=dict(np.load(SOURCE/name));records=[];stats=[]
        for goal in (2.,3.):
            nodes,metrics=path_rollout(g,g.initial,goal,target=targets[goal])
            for label in (2.,3.):records.extend(build_records(g,nodes,targets[label],label,-1))
            stats.append(dict(controller='teacher',goal=goal,states=len(nodes),success=metrics['success']))
        for seed in range(5601,5606):
            for goal in (2,3):
                task=('five' if structure.startswith('five') else 'seven')+f'_d{goal}'
                config=config_for(task);env=GoalEnv(config);model=make_model('graph',g.coordinate_size,seed)
                model.set_geometry(g.directory)
                checkpoint=ROOT/'source'/f'graph_seed{seed}'/'best.pt'
                saved=torch.load(checkpoint,weights_only=False)
                model.load_state_dict(saved['actor'],strict=False);agent=make_agent(model,config)
                state=env.reset();nodes=[env.node.copy()];done=False
                with torch.no_grad():
                    while not done and len(nodes)<=400:
                        modes=env.get_modes()
                        if modes.shape[1]==0:break
                        a=agent.get_action(state,modes,deterministic=True)
                        state,_,done,info=env.step(a.physical_direction,a.requested_width,a.mode_coefficients)
                        nodes.append(env.node.copy())
                for label in (2.,3.):records.extend(build_records(g,nodes,targets[label],label,-1))
                stats.append(dict(controller=f'graph{seed}',goal=goal,states=len(nodes),success=info.get('strict_success',False)))
                print(structure,seed,goal,'new records',len(records),flush=True)
        new=pack(records)
        # Remove exact duplicate state-goal observations across source students.
        keys=np.column_stack([new['node'].reshape(len(new['node']),-1).round(9),new['goal']])
        _,indices=np.unique(keys,axis=0,return_index=True);indices=np.sort(indices)
        new={k:v[indices] for k,v in new.items()}
        combined={k:np.concatenate([original[k],new[k]]) for k in original}
        np.savez_compressed(output/name,**combined)
        manifest.append(dict(structure=structure,original_labels=len(original['node']),added_labels=len(new['node']),
            source_checkpoints=[dict(path=str(ROOT/'source'/f'graph_seed{s}'/'best.pt'),sha256=digest(ROOT/'source'/f'graph_seed{s}'/'best.pt')) for s in range(5601,5606)],
            target_files=[dict(path=str(g.directory/f'node_target_d{goal}.dat'),sha256=digest(g.directory/f'node_target_d{goal}.dat')) for goal in (2,3)],
            generated_sha256=digest(output/name),rollouts=stats))
    write_json(output/'manifest.json',manifest)

if __name__=='__main__':
    torch.set_num_threads(1);generate()
