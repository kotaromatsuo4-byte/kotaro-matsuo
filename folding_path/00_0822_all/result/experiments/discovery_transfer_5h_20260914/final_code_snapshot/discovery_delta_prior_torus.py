"""Choose an existing source prior by the required change in spread."""
from discovery_transfer import *


def main():
    torch.set_num_threads(1)
    target = GoalEnv(config_for('torus_d2'))
    source = GoalEnv(config_for('water_d1'))
    target_delta = target.target_spread - target.compactness_metrics()['spread']
    source_initial = source.compactness_metrics()['spread']
    candidates = [1., float(np.sqrt(2.))]
    prior = min(candidates, key=lambda d: abs((d-source_initial)-target_delta))
    write_json(ROOT/'delta_prior_protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        rule='Choose known plane-source D whose target-minus-initial spread is closest to that of the new problem',
        motivation='Absolute-nearest known D uses an expanding water path for a contracting torus objective',
        source_initial_spread=source_initial, target_initial_spread=target.compactness_metrics()['spread'],
        target_delta=target_delta, known_plane_goals=candidates, policy_condition_D=prior,
        actual_target_D=2., task='torus_d2', architecture='projected',
        source_seeds=[5601,5602,5603], search_seeds=[5701,5702,5703], budget=6400,
        arms=['frozen','scratch'], checkpoint_reselection=False, target_teacher_used=False,
        interpretation='Exploratory alternative prior, all three fixed source seeds; report alongside absolute-nearest trials'))
    expected = [ROOT/'nearest_prior_torus'/f'torus_d2_projected_{arm}_s{seed}'/'result.json'
                for arm in ('frozen','scratch','finetune') for seed in (5701,5702,5703)]
    while not all(p.exists() for p in expected):
        if time.time() >= DEADLINE-180:
            return
        time.sleep(3)
    for i in range(3):
        for arm in ('frozen','scratch'):
            if time.time() >= DEADLINE-180:
                return
            run('torus_d2','projected',arm,5701+i,5601+i,6400,
                checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt') if arm=='frozen' else None,
                tag='delta_prior_torus',policy_goal=prior)


if __name__ == '__main__':
    main()
