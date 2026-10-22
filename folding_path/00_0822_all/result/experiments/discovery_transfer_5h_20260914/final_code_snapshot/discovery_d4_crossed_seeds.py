"""Cross frozen source NN seeds with search seeds; reuse diagonal trials."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'d4_crossed_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],budget=6400,arm='frozen',
        purpose='Separate source-weight variability from stochastic search variability',
        diagonal_reused_from=['projected_transfer','projected_d4_replication'],
        interpretation='Three source NNs crossed with three search seeds, not nine independently trained NNs',
        target_teacher_used=False,checkpoint_reselection=False))
    for source_seed in (5601,5602,5603):
        for seed in (5701,5702,5703):
            if source_seed+100==seed:continue
            if time.time()>=DEADLINE-180:return
            run('seven_d4','projected','frozen',seed,source_seed,6400,
                checkpoint=str(ROOT/'source'/f'projected_water_seed{source_seed}'/'best.pt'),
                tag=f'crossed_d4_source{source_seed}')

if __name__=='__main__':main()
