"""Only the NN goal input is changed; the environment still requires D4."""
from discovery_transfer import *

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'goal_ablation_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_seeds=[5601,5602,5603],search_seeds=[5701,5702,5703],policy_goals=[3.,2.],environment_goal=4.,
        task='seven_d4',budget=6400,arm='frozen',correct_goal_reference='projected_transfer and projected_d4_replication',
        purpose='Check whether unseen-goal conditioning matters beyond a generic learned folding prior',
        target_teacher_used=False,parameter_updates=0))
    expected=[ROOT/f'crossed_d4_source{s}'/f'seven_d4_projected_frozen_s{t}'/'result.json'
        for s in (5601,5602,5603) for t in (5701,5702,5703) if s+100!=t]
    while not all(p.exists() for p in expected):
        if time.time()>=DEADLINE-180:return
        time.sleep(3)
    for goal in (3.,2.):
        for i in range(3):
            if time.time()>=DEADLINE-180:return
            run('seven_d4','projected','frozen',5701+i,5601+i,6400,
                checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt'),
                tag=f'd4_goal_ablation_d{int(goal)}',policy_goal=goal)

if __name__=='__main__':main()
