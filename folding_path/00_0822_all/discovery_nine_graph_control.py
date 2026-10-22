"""Same mixed source experience, older graph representation, unseen nine D4."""
from discovery_transfer import *


def main():
    torch.set_num_threads(1)
    write_json(ROOT/'nine_graph_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        task='nine_d4',architecture='graph',arm='frozen',budget=6400,
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],
        source='Fixed graph_water checkpoints with the same teachers and 20000 updates as projected_water',
        scratch_control_reused='sequential_nine_control: same graph initial weights, target and search seeds',
        D4_teacher_used=False,target_teacher_used=False,checkpoint_reselection=False))
    expected=[ROOT/f'crossed_nine_source{s}'/f'nine_d4_projected_frozen_s{t}'/'result.json'
              for s in (5601,5602,5603) for t in (5701,5702,5703) if s+100!=t]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for i in range(3):
        if time.time()>=DEADLINE-180:return
        run('nine_d4','graph','frozen',5701+i,5601+i,6400,
            checkpoint=str(ROOT/'source'/f'graph_water_seed{5601+i}'/'best.pt'),tag='nine_graph_matched')


if __name__=='__main__':main()
