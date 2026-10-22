"""Read-only initial policy statistics; no target teacher or environment step."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1);rows=[]
    for label in ('dense','graph','graph_long','graph_coverage','graph_water','graph_seven'):
        architecture='dense' if label=='dense' else 'graph'
        for seed in range(5601,5606):
            checkpoint=ROOT/'source'/f'{label}_seed{seed}'/'best.pt'
            if not (checkpoint.parent/'summary.json').exists():continue
            saved=torch.load(checkpoint,weights_only=False)
            tasks=('seven_d2','seven_d3','seven_d4') if architecture=='dense' else ('five_d2','five_d3','seven_d2','seven_d3','seven_d4','water_d1','water_sqrt2','torus_d2')
            for task in tasks:
                config=config_for(task);env=GoalEnv(config);model=make_model(architecture,env.coordinate_size,seed)
                if architecture=='graph':model.set_geometry(config['model_directory'],config['flatness_subspace_dimension'])
                model.load_state_dict(saved['actor'],strict=False);agent=make_agent(model,config)
                state=env.get_state();modes=env.get_modes()
                with torch.no_grad():
                    a=agent.get_action(state,modes,deterministic=True)
                    mean,_,_=model.modal_distribution_parameters(torch.tensor(state[None]),torch.tensor(modes.T,dtype=torch.float32))
                    raw=modes@mean.numpy();raw/=max(np.linalg.norm(raw),1e-12)
                alignment=float(np.dot(raw,a.physical_direction)*np.sign(a.requested_width))
                rows.append(dict(source=label,seed=seed,task=task,modes=a.mode_count,mean_norm=a.mixture_mean_norm,std_norm=a.mixture_standard_deviation_norm,concentration=a.mixture_concentration,width_mean=a.width_mean,width_variance=a.width_variance,requested_width=a.requested_width,alignment_with_directed_mean=alignment))
    write_csv(ROOT/'initial_policy_diagnostics.csv',rows)
    print('Saved',len(rows),'initial policy observations; no folding transitions executed.')

if __name__=='__main__':main()
