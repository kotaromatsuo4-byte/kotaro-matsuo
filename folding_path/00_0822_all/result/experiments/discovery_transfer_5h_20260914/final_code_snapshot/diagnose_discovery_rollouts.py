"""Reproduce frozen first episodes and record why motion stops. No teachers."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1);summary=[]
    output=ROOT/'rollout_diagnostics';output.mkdir(exist_ok=True)
    specifications=[('dense',('seven_d2','seven_d3','seven_d4')),
        ('graph_coverage',('seven_d4','torus_d2')),
        ('graph_water',('water_d1','water_sqrt2','torus_d2')),
        ('graph_seven',('seven_d2','seven_d3','five_d3','nine_d3'))]
    for label,tasks in specifications:
        for source_seed in range(5601,5604):
            checkpoint=ROOT/'source'/f'{label}_seed{source_seed}'/'best.pt'
            saved=torch.load(checkpoint,weights_only=False)
            for task in tasks:
                for deterministic in (True,False):
                    if time.time()>=DEADLINE-120:return
                    config=config_for(task);env=GoalEnv(config)
                    architecture='dense' if label=='dense' else 'graph'
                    model=make_model(architecture,env.coordinate_size,source_seed+100)
                    if architecture=='graph':model.set_geometry(config['model_directory'],config['flatness_subspace_dimension'])
                    model.load_state_dict(saved['actor'],strict=False);agent=make_agent(model,config)
                    torch.manual_seed(source_seed+100+100000);np.random.seed(source_seed+100+100000)
                    env.configure_episode_tolerance(False);state=env.reset();rows=[]
                    for step in range(config['max_steps']):
                        modes=env.get_modes()
                        if not modes.shape[1]:break
                        with torch.no_grad():a=agent.get_action(state,modes,deterministic=deterministic)
                        state,reward,done,info=env.step(a.physical_direction,a.requested_width,a.mode_coefficients)
                        keys=('flatness_ratio','spread','spread_error','strict_success','wrong_flat','rejected_crossing','correction_failed','actual_displacement','accepted_width','projection_attempts','termination_reason')
                        row={k:info.get(k) for k in keys};row.update(step=step+1,requested_width=a.requested_width,width_mean=a.width_mean,concentration=a.mixture_concentration,reward=reward)
                        rows.append(row)
                        if done:break
                    name=f'{label}_{task}_s{source_seed}_{"mean" if deterministic else "sample"}'
                    write_csv(output/(name+'.csv'),rows)
                    summary.append(dict(source=label,task=task,source_seed=source_seed,deterministic=deterministic,steps=len(rows),
                        success=bool(rows[-1]['strict_success']),reason=rows[-1]['termination_reason'],
                        crossing_rejections=sum(bool(r['rejected_crossing']) for r in rows),
                        correction_failures=sum(bool(r['correction_failed']) for r in rows),
                        no_motion=sum(abs(r['actual_displacement'] or 0)<1e-10 for r in rows),
                        final_flatness=rows[-1]['flatness_ratio'],final_spread_error=rows[-1]['spread_error'],
                        best_joint_ratio=min(max(r['flatness_ratio']/.001,r['spread_error']/.05) for r in rows),
                        source_sha256=digest(checkpoint)))
    write_csv(ROOT/'rollout_diagnostics.csv',summary)
    write_json(output/'protocol.json',dict(teachers_used=False,episodes_per_condition=1,source_seeds=[5601,5602,5603],purpose='Descriptive failure analysis; repeated episodes are not additional independent success trials.'))
    print('Saved',len(summary),'diagnostic replays.')

if __name__=='__main__':main()
