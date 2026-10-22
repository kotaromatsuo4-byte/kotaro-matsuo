"""Counted random prefix + frozen mean actor, no unknown target teacher.

Each attempt starts at the original torus initial shape, spends up to 40
proposals on random feasible exploration, then uses the mean actor for 200.
All rejected and prefix proposals count toward first discovery and the budget.
"""
from discovery_transfer import *

def run_restart(architecture,arm,seed,source_seed,budget=6400):
    task='torus_d2';directory=ROOT/'restart_torus'/f'{task}_{architecture}_{arm}_s{seed}'
    directory.mkdir(parents=True,exist_ok=False)
    config=config_for(task);config['max_steps']=240
    env=GoalEnv(config);model=make_model(architecture,env.coordinate_size,seed)
    model.set_geometry(config['model_directory'],2)
    sourcefile=None
    if arm=='deterministic':
        label='graph_water' if architecture=='graph' else 'projected_water'
        sourcefile=ROOT/'source'/f'{label}_seed{source_seed}'/'best.pt'
        saved=torch.load(sourcefile,weights_only=False);missing,unexpected=model.load_state_dict(saved['actor'],strict=False)
        assert not unexpected and all(n.startswith('critic') for n in missing)
    agent=make_agent(model,config)
    before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('critic')}
    write_json(directory/'design.json',dict(task=task,architecture=architecture,arm=arm,seed=seed,source_seed=source_seed,
        config=config,budget=budget,source_checkpoint=sourcefile,source_sha256=digest(sourcefile) if sourcefile else None,
        target_coordinates_used=False,random_prefix_proposals=40,actor_proposals_per_attempt=200,prefix_seed_rule='seed+200000+episode',
        prefix_counted=True,initial_file=None,initial_sha256=None,variance_override=None,align_width=False))
    total=0;projection_calls=0;episode=0;hit=None;rows=[];start=time.monotonic()
    with torch.no_grad():
        while total<budget and time.time()<DEADLINE-120:
            episode+=1;env.configure_episode_tolerance(False);state=env.reset();nodes=[env.node.copy()]
            rng=np.random.default_rng(seed+200000+episode);reward_sum=0
            info=dict(strict_success=False,termination_reason='no_modes')
            for local in range(240):
                if total>=budget or time.time()>=DEADLINE-120:break
                modes=env.get_modes()
                if not modes.shape[1]:break
                if local<40:
                    coefficients=rng.normal(size=modes.shape[1]);direction=modes@coefficients
                    direction/=max(np.linalg.norm(direction),1e-12)
                    width=rng.uniform(-config['max_step_width'],config['max_step_width'])
                else:
                    action=agent.get_action(state,modes,deterministic=True)
                    direction=action.physical_direction;width=action.requested_width;coefficients=action.mode_coefficients
                state,reward,done,info=env.step(direction,width,coefficients)
                total+=1;projection_calls+=int(info.get('projection_attempts',0));reward_sum+=reward;nodes.append(env.node.copy())
                if info.get('strict_success'):
                    hit=dict(steps=total,projection_calls=projection_calls,episode=episode,seconds=time.monotonic()-start,updates=0,found_during_random_prefix=local<40)
                    np.savez_compressed(directory/'first_hit_path.npz',nodes=np.array(nodes))
                    np.savetxt(directory/'first_hit.dat',env.node,delimiter=',');break
                if done:break
            metrics=env.compactness_metrics()
            rows.append(dict(episode=episode,steps=env.step_count,cumulative_steps=total,projection_calls=projection_calls,
                reward=reward_sum,flatness=metrics['flatness_ratio'],spread=metrics['spread'],spread_error=metrics['spread_error'],
                success=int(hit is not None),updates_before=0,seconds=time.monotonic()-start,reason=info['termination_reason']))
            write_csv(directory/'episodes.csv',rows);write_json(directory/'progress.json',rows[-1])
            if hit or env.step_count==0:break
    torch.save(dict(model=model.state_dict()),directory/'final.pt')
    result=dict(task=task,architecture=architecture,arm=arm,seed=seed,source_seed=source_seed,budget=budget,steps=total,
        episodes=episode,success=hit is not None,first_hit=hit,projection_calls=projection_calls,seconds=time.monotonic()-start,updates=0,
        stop_reason='success' if hit else ('deadline' if time.time()>=DEADLINE-120 else 'budget'),
        frozen_actor_unchanged=all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()),
        source_sha256=digest(sourcefile) if sourcefile else None,prefix_proposals_counted=True)
    write_json(directory/'result.json',result);print(json.dumps(result),flush=True)
    return result

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'restart_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),task='torus_d2',
        source_seeds=[5601,5602,5603],target_seeds=[6301,6302,6303],budget=6400,random_prefix=40,mean_actor_steps=200,
        source_labels=['graph_water','projected_water'],arms=['untrained_deterministic','deterministic'],
        target_teacher_used=False,prefix_proposals_counted=True,selection='First three source seeds, frozen before evaluating these prefixes'))
    for i in range(3):
        for architecture in ('graph','projected'):
            for arm in ('untrained_deterministic','deterministic'):
                if time.time()>=DEADLINE-180:return
                run_restart(architecture,arm,6301+i,5601+i)

if __name__=='__main__':main()
