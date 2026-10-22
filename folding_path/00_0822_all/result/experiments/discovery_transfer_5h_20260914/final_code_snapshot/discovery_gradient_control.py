"""Teacher-free geometric baseline, with all modal objective calls counted."""
from discovery_transfer import *

def run_gradient(task,weight,initial=None,index=0,tag=None,seed=None):
    arm='gradient_w'+str(weight).replace('.','p');seed=6401+index if seed is None else seed
    series=tag or ('gradient_fresh_bar' if initial else 'gradient_controls')
    directory=ROOT/series/f'{task}_analytic_{arm}_s{seed}';directory.mkdir(parents=True,exist_ok=False)
    config=config_for(task);env=GoalEnv(config)
    if initial:env=GoalEnv(config,data=(np.loadtxt(initial,delimiter=','),env.member,env.fix))
    write_json(directory/'design.json',dict(task=task,architecture='analytic',arm=arm,seed=seed,source_seed=None,config=config,
        budget=config['max_steps'],source_checkpoint=None,source_sha256=None,target_coordinates_used=False,
        initial_file=initial,initial_sha256=digest(initial) if initial else None,
        objective='reset-normalized flatness + weight * squared D error',weight=weight,finite_difference_epsilon=1e-5,
        width_rule='max width initially; halve after rejection/no motion, regrow by 1.1 after motion',teacher_used=False))
    env.configure_episode_tolerance(False);env.reset();nodes=[env.node.copy()];calls=0;total=0;projection_calls=0;hit=None
    width=config['max_step_width'];start=time.monotonic();reason='budget';reward_sum=0
    def objective(node):
        nonlocal calls
        calls+=1;m=env.compactness_metrics(node)
        return m['flatness_ratio']+weight*m['spread_error']**2
    while total<config['max_steps'] and time.time()<DEADLINE-120:
        modes=env.get_modes();eps=1e-5*env.coordinate_scale
        gradient=np.array([(objective(env.node+eps*v.reshape(env.node.shape))-objective(env.node-eps*v.reshape(env.node.shape)))/(2*eps) for v in modes.T])
        coefficients=-gradient;direction=modes@coefficients;norm=np.linalg.norm(direction)
        if norm<1e-10:reason='stationary';break
        direction/=norm
        _,reward,done,info=env.step(direction,width,coefficients)
        total+=1;projection_calls+=int(info.get('projection_attempts',0));reward_sum+=reward;nodes.append(env.node.copy())
        if info.get('rejected_crossing') or info.get('actual_displacement',0)<1e-10:width*=.5
        else:width=min(config['max_step_width'],width*1.1)
        if info.get('strict_success'):
            hit=dict(steps=total,projection_calls=projection_calls,episode=1,seconds=time.monotonic()-start,updates=0,objective_evaluations=calls)
            np.savez_compressed(directory/'first_hit_path.npz',nodes=np.array(nodes));np.savetxt(directory/'first_hit.dat',env.node,delimiter=',')
            reason='success';break
        if done:reason=info['termination_reason'];break
        if width<1e-8:reason='width_exhausted';break
    if time.time()>=DEADLINE-120 and hit is None:reason='deadline'
    metrics=env.compactness_metrics()
    write_csv(directory/'episodes.csv',[dict(episode=1,steps=total,cumulative_steps=total,projection_calls=projection_calls,reward=reward_sum,
        flatness=metrics['flatness_ratio'],spread=metrics['spread'],spread_error=metrics['spread_error'],success=int(hit is not None),updates_before=0,
        seconds=time.monotonic()-start,reason=reason)])
    result=dict(task=task,architecture='analytic',arm=arm,seed=seed,source_seed=None,budget=config['max_steps'],steps=total,episodes=1,
        success=hit is not None,first_hit=hit,projection_calls=projection_calls,seconds=time.monotonic()-start,updates=0,stop_reason=reason,
        frozen_actor_unchanged=True,source_sha256=None,objective_evaluations=calls)
    write_json(directory/'result.json',result);print(json.dumps(result),flush=True)

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'gradient_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),weights=[0.,.1,1.,10.],
        tasks=['five_d2','five_d3','seven_d4','nine_d3','torus_d2'],fresh_five_weight=1.,
        target_teacher_used=False,source_experience_used=False,objective_calls_counted=True,
        interpretation='Short deterministic local-search reference, not a tuned optimizer benchmark; report every weight.'))
    for task in ('five_d2','five_d3','seven_d4','nine_d3','torus_d2'):
        for weight in (0.,.1,1.,10.):
            if time.time()>=DEADLINE-180:return
            run_gradient(task,weight)
    for i in range(8):run_gradient('five_d3',1.,ROOT/'fresh_bar_inputs'/f'start{i:02d}.dat',i)

if __name__=='__main__':main()
