"""Mean-action controls and crossed search seeds for the nine-node discovery."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'nine_followup_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='Frozen projected source seed5602 found nine D4 in 1583 proposals',
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],
        mean_tasks=['nine_d3','nine_d4'],mean_budget=400,crossed_task='nine_d4',crossed_budget=6400,
        mean_control='untrained projected actor, same initial model/search seeds',
        diagonal_reused_from='projected_nine',target_teacher_used=False,checkpoint_reselection=False,
        interpretation='Crossed source/search seeds; not nine independently trained source models'))
    for i in range(3):
        for task in ('nine_d3','nine_d4'):
            for arm in ('deterministic','untrained_deterministic'):
                if time.time()>=DEADLINE-180:return
                run(task,'projected',arm,5701+i,5601+i,400,
                    checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt') if arm=='deterministic' else None,
                    tag='projected_nine_zero' if arm=='deterministic' else 'projected_nine_zero_control')
    expected=[ROOT/'d4_graph_matched'/f'seven_d4_graph_{arm}_s{seed}'/'result.json'
        for arm in ('frozen','scratch') for seed in (5701,5702,5703)]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for source_seed in (5601,5602,5603):
        for seed in (5701,5702,5703):
            if source_seed+100==seed:continue
            if time.time()>=DEADLINE-180:return
            run('nine_d4','projected','frozen',seed,source_seed,6400,
                checkpoint=str(ROOT/'source'/f'projected_water_seed{source_seed}'/'best.pt'),
                tag=f'crossed_nine_source{source_seed}')

if __name__=='__main__':main()
