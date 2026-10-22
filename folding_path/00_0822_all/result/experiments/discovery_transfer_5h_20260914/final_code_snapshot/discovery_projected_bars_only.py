"""Remove water experience: strictly D2/D3 source goals for the projected actor."""
from discovery_transfer import *


def main():
    torch.set_num_threads(1)
    write_json(ROOT/'projected_bars_protocol.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Test the original D2/D3-only experience claim separately from the mixed water-source positive result',
        sources=['five D2,D3','seven D2,D3'],data_root=str(ROOT/'water_source'),
        water_data_used=False,D4_teacher_used=False,source_seeds=[5601,5602,5603],updates=20000,
        selection='Minimum source-path validation loss; all three NNs trained before unknown-task evaluation',
        targets=['seven D4','nine D4'],budget=6400,
        comparisons_reused=['projected D4 scratch at common6400','projected_nine nine D4 scratch'],
        note='Same bar records as projected_water, but geometry sampling now contains two rather than three geometries'))
    expected=[ROOT/'delta_prior_torus'/f'torus_d2_projected_{arm}_s{seed}'/'result.json'
              for arm in ('frozen','scratch') for seed in (5701,5702,5703)]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for seed in (5601,5602,5603):
        if time.time()>=DEADLINE-180:return
        pretrain('projected',seed,20000,'projected_bars',ROOT/'water_source',False)
    for i in range(3):
        checkpoint=str(ROOT/'source'/f'projected_bars_seed{5601+i}'/'best.pt')
        for task in ('five_d2','five_d3','seven_d2','seven_d3'):
            if time.time()>=DEADLINE-180:return
            run(task,'projected','deterministic',5701+i,5601+i,400,checkpoint=checkpoint,tag='source_projected_bars')
        for task in ('seven_d4','nine_d4'):
            if time.time()>=DEADLINE-180:return
            run(task,'projected','deterministic',5701+i,5601+i,400,checkpoint=checkpoint,tag='projected_bars_zero')
    for task,arm in [('seven_d4','frozen'),('nine_d4','frozen'),('seven_d4','finetune')]:
        for i in range(3):
            if time.time()>=DEADLINE-180:return
            run(task,'projected',arm,5701+i,5601+i,6400,
                checkpoint=str(ROOT/'source'/f'projected_bars_seed{5601+i}'/'best.pt'),tag='projected_bars_transfer')


if __name__=='__main__':main()
