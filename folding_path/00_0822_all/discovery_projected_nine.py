"""Reuse the already-frozen multi-structure projected actor on unseen nine."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'projected_nine_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],tasks=['nine_d3','nine_d4'],budget=6400,
        architecture='projected',arms=['frozen','scratch'],source='Previously frozen projected_water checkpoints',
        source_goals=['five/seven D2,D3','water D1,sqrt2'],nine_teacher_used=False,D4_teacher_used=False,
        purpose='Check the same representation that found unseen seven D4 on the pre-fixed nine-node linkage',
        interpretation='Different from sequential_nine, which intentionally includes newly known seven D4 experience'))
    expected=[ROOT/f'd4_goal_ablation_d{goal}'/f'seven_d4_projected_frozen_s{seed}'/'result.json'
        for goal in (2,3) for seed in (5701,5702,5703)]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for i in range(3):
        for task in ('nine_d3','nine_d4'):
            for arm in ('frozen','scratch'):
                if time.time()>=DEADLINE-180:return
                run(task,'projected',arm,5701+i,5601+i,6400,
                    checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt') if arm=='frozen' else None,
                    tag='projected_nine')

if __name__=='__main__':main()
