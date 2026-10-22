"""Four newly generated seven-bar states; no D4 teachers or policy filtering."""
from discovery_transfer import *
from supervised_folding import Geometry,generate_starts
from discovery_gradient_control import run_gradient


def main():
    torch.set_num_threads(1)
    output=ROOT/'fresh_d4_inputs';output.mkdir(exist_ok=False)
    checkpoint=ROOT/'source/projected_bars_seed5601/best.pt'
    write_json(output/'source_frozen_before_starts.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint),source_sha256=digest(checkpoint),source_seed=5601,
        selection='First prespecified source seed; fixed before generating new starts',
        initial_seed=69701,count=4,search_seeds=[6901,6902,6903,6904],walk_range=[20,80],
        target_task='seven_d4',target_teacher_used=False,policy_based_initial_selection=False,
        source_state_exclusion_distance=.03,source_state_file=str(ROOT/'water_source/source_data.npz'),
        comparison='Frozen NN and matched scratch RL:6400; local geometric gradient w=1:400',
        interpretation='New initial-condition problems, not free warm starts for the original initial-condition search; generation is setup cost for both arms'))
    g=Geometry(BASE/'model/seven_bar_linkage')
    source_nodes=np.load(ROOT/'water_source/source_data.npz')['node']
    starts=generate_starts(g,4,np.random.default_rng(69701),20,80,excluded=[g.initial,*source_nodes])
    manifest=[]
    for i,node in enumerate(starts):
        file=output/f'start{i:02d}.dat';np.savetxt(file,node,delimiter=',',fmt='%.15g')
        manifest.append(dict(index=i,path=str(file),sha256=digest(file),metrics=g.metrics(node,4),
            nearest_source_state_distance=float(np.linalg.norm((source_nodes-node).reshape(len(source_nodes),-1),axis=1).min()/g.scale)))
    write_json(output/'manifest.json',manifest)
    # Cheap analytic controls are fixed before observing any NN result here.
    for i in range(4):
        if time.time()>=DEADLINE-180:return
        run_gradient('seven_d4',1.,output/f'start{i:02d}.dat',i,tag='gradient_fresh_d4',seed=6901+i)
    for i in range(4):
        for arm in ('frozen','scratch'):
            if time.time()>=DEADLINE-180:return
            run('seven_d4','projected',arm,6901+i,5601,6400,
                checkpoint=str(checkpoint) if arm=='frozen' else None,
                tag='fresh_d4',initial=output/f'start{i:02d}.dat')


if __name__=='__main__':main()
