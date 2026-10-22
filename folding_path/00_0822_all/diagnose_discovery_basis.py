"""Basis sensitivity at initial states, without teacher labels or target steps."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1);rows=[]
    for label,architecture in [('graph_water','graph'),('projected_water','projected')]:
        checkpoint=ROOT/'source'/f'{label}_seed5601'/'best.pt'
        saved=torch.load(checkpoint,weights_only=False)
        for task in ('five_d3','seven_d2','seven_d3','seven_d4','water_d1','water_sqrt2','torus_d2'):
            config=config_for(task);env=GoalEnv(config);model=make_model(architecture,env.coordinate_size,5601)
            model.set_geometry(config['model_directory'],config['flatness_subspace_dimension']);model.load_state_dict(saved['actor'],strict=False)
            state=torch.tensor(env.get_state()[None]);modes=torch.tensor(env.get_modes().T,dtype=torch.float32)
            generator=torch.Generator().manual_seed(59601)
            with torch.no_grad():
                mean,_,_=model.modal_distribution_parameters(state,modes);physical=mean@modes
                for change in ('sign','rotation'):
                    cosines=[];errors=[]
                    for _ in range(100):
                        rotation=(torch.diag(torch.randint(0,2,(len(modes),),generator=generator)*2-1).float() if change=='sign' else torch.linalg.qr(torch.randn(len(modes),len(modes),generator=generator))[0])
                        basis=rotation@modes
                        other,_,_=model.modal_distribution_parameters(state,basis);other=other@basis
                        cosines.append(float(F.cosine_similarity(physical,other,dim=0)))
                        errors.append(float((other-physical).norm()/physical.norm().clamp(min=1e-8)))
                    rows.append(dict(source=label,task=task,change=change,samples=100,mean_cosine=float(np.mean(cosines)),min_cosine=min(cosines),mean_relative_error=float(np.mean(errors)),max_relative_error=max(errors),source_sha256=digest(checkpoint)))
    write_csv(ROOT/'basis_sensitivity.csv',rows);print('Saved',len(rows),'basis-sensitivity conditions; no environment transitions.')

if __name__=='__main__':main()
