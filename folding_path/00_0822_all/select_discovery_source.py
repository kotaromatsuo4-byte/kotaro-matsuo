"""Freeze an extended source variant using source-task performance only."""
import json
from discovery_transfer import ROOT,digest,write_json

def select():
    candidates=[]
    for label in ('graph_long','graph_coverage'):
        results=list((ROOT/('source_'+label)).glob('*/result.json'))
        if len(results)!=20:raise RuntimeError(f'{label}: need all 20 known-task reference evaluations, have {len(results)}')
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in results]
        losses=[json.loads((ROOT/'source'/f'{label}_seed{s}'/'summary.json').read_text())['best_validation'] for s in range(5601,5606)]
        candidates.append(dict(label=label,known_reference_successes=sum(r['success'] for r in rows),known_reference_trials=20,mean_heldout_source_loss=sum(losses)/5))
    best=sorted(candidates,key=lambda x:(-x['known_reference_successes'],x['mean_heldout_source_loss']))[0]
    selected=best['label']
    manifest=dict(rule='Highest known-task reference success count across all five source seeds; tie: lowest mean held-out source loss. No unknown-task result used.',candidates=candidates,selected=selected,
        checkpoints=[dict(seed=s,path=str(ROOT/'source'/f'{selected}_seed{s}'/'best.pt'),sha256=digest(ROOT/'source'/f'{selected}_seed{s}'/'best.pt')) for s in range(5601,5606)])
    write_json(ROOT/'source_selection.json',manifest)
    jobs=[]
    for i in range(5):
        checkpoint=str(ROOT/'source'/f'{selected}_seed{5601+i}'/'best.pt')
        for task in ('seven_d4','torus_d2'):
            jobs.append(dict(task=task,architecture='graph',arm='deterministic',seed=5701+i,source_seed=5601+i,budget=400,tag='selected_zero',checkpoint=checkpoint))
    for i in range(5):
        checkpoint=str(ROOT/'source'/f'{selected}_seed{5601+i}'/'best.pt')
        for arm in ('finetune','frozen'):
            jobs.append(dict(task='torus_d2',architecture='graph',arm=arm,seed=5701+i,source_seed=5601+i,budget=51200,tag='selected',checkpoint=checkpoint))
    write_json(ROOT/'selected_queue.json',jobs)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':select()
