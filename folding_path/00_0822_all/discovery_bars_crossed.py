"""Replicate the strict D2/D3-only D4 discovery across search streams."""
from discovery_transfer import *


def main():
    torch.set_num_threads(1)
    write_json(ROOT/'bars_crossed_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='First two D2/D3-only projected NNs found seven D4 in 673 and 565 proposals',
        task='seven_d4',architecture='projected',arm='frozen',budget=6400,
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],
        source='All three projected_bars models already trained and selected by source validation',
        diagonal_reused_from='projected_bars_transfer',D4_teacher_used=False,checkpoint_reselection=False,
        interpretation='Three source NNs crossed with three search seeds; six extra pairs, no independent-model count inflation'))
    expected=[ROOT/'projected_bars_transfer'/f'{task}_projected_{arm}_s{seed}'/'result.json'
              for task,arm in [('seven_d4','frozen'),('nine_d4','frozen'),('seven_d4','finetune')]
              for seed in (5701,5702,5703)]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for source in (5601,5602,5603):
        for seed in (5701,5702,5703):
            if source+100==seed:continue
            if time.time()>=DEADLINE-180:return
            run('seven_d4','projected','frozen',seed,source,6400,
                checkpoint=str(ROOT/'source'/f'projected_bars_seed{source}'/'best.pt'),
                tag=f'crossed_bars_d4_source{source}')


if __name__=='__main__':main()
