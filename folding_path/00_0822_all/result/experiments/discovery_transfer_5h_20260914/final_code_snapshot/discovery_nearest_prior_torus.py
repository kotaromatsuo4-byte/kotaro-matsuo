"""Reuse a known plane-folding condition as the prior for actual torus D2."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1);prior=float(np.sqrt(2.))
    write_json(ROOT/'nearest_prior_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        rule='Nearest observed source D within the same flatness subspace dimension',
        known_plane_goals=[1.,prior],actual_target_D=2.,policy_condition_D=prior,
        task='torus_d2',source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],budget=6400,
        architecture='projected',arms=['frozen','scratch','finetune'],
        source='Previously frozen projected_water checkpoints; no checkpoint reselection',
        motivation='D4 goal-input controls showed that a known D3 folding prior can discover D4',
        target_teacher_used=False,reward_and_success_goal_unchanged=True))
    expected=[ROOT/'projected_nine'/f'{task}_projected_{arm}_s{seed}'/'result.json'
        for task in ('nine_d3','nine_d4') for arm in ('frozen','scratch') for seed in (5701,5702,5703)]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for i in range(3):
        for arm in ('frozen','scratch','finetune'):
            if time.time()>=DEADLINE-180:return
            run('torus_d2','projected',arm,5701+i,5601+i,6400,
                checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt') if arm!='scratch' else None,
                tag='nearest_prior_torus',policy_goal=prior)

if __name__=='__main__':main()
