"""Match source data, updates and search budget for the older graph actor."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'d4_representation_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],budget=6400,task='seven_d4',
        architecture='graph',arms=['frozen','scratch'],source='Already frozen graph_water models, 20000 updates',
        comparison='projected_water uses the identical known-task datasets and update count',
        purpose='Do not attribute success to the projected representation without a stochastic graph control',
        D4_teacher_used=False,checkpoint_reselection=False))
    expected=[]
    for seed in (5701,5702,5703):
        for task in ('nine_d3','nine_d4'):
            for arm in ('finetune','frozen'):expected.append(ROOT/'sequential_nine'/f'{task}_graph_{arm}_s{seed}'/'result.json')
        expected.extend([ROOT/'sequential_nine_control'/f'nine_d4_graph_scratch_s{seed}'/'result.json',
            ROOT/'sequential_nine_old'/f'nine_d4_graph_frozen_s{seed}'/'result.json'])
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for i in range(3):
        for arm in ('frozen','scratch'):
            if time.time()>=DEADLINE-180:return
            run('seven_d4','graph',arm,5701+i,5601+i,6400,
                checkpoint=str(ROOT/'source'/f'graph_water_seed{5601+i}'/'best.pt') if arm=='frozen' else None,
                tag='d4_graph_matched')

if __name__=='__main__':main()
