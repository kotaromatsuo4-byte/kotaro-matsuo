"""Prioritize remaining frozen source seeds after the first D4 first hit.

Short prefixes of the same planned 51,200-proposal trials, not new independent
replicates. A success can be inherited by its longer-budget counterpart.
"""
import shutil
from discovery_transfer import *

def promote_success(source,destination):
    result=json.loads((source/'result.json').read_text(encoding='utf-8'))
    if not result['success'] or destination.exists():return
    temporary=ROOT/'promotion_cache'/source.name/'full_budget_copy'
    temporary.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,temporary)
    result.update(budget=51200,inherited_success_from=str(source),additional_steps=0)
    write_json(temporary/'result.json',result)
    design=json.loads((temporary/'design.json').read_text(encoding='utf-8'))
    design.update(budget=51200,inherited_success_from=str(source),inherited_result_sha256=digest(source/'result.json'))
    write_json(temporary/'design.json',design)
    try:temporary.rename(destination)
    except OSError as error:
        print('Promotion skipped; original prefix retained:',repr(error),flush=True)
        return # A concurrent run or transient Windows lock must not stop replication.

def main():
    torch.set_num_threads(1)
    write_json(ROOT/'d4_replication_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        trigger='First prespecified source seed found unseen D4 in 612 proposals without RL updates',
        source_seeds=[5602,5603],target_seeds=[5702,5703],budget=6400,
        selection='All remaining source seeds, already trained and frozen before this confirmation',
        same_trials_as='projected_transfer, first 6400 proposals; do not count twice',
        source_goals=['five/seven D2,D3','water D1,sqrt2'],D4_teacher_used=False))
    for i in (1,2):
        for arm in ('frozen','scratch','finetune'):
            if time.time()>=DEADLINE-180:return
            result=run('seven_d4','projected',arm,5701+i,5601+i,6400,
                checkpoint=str(ROOT/'source'/f'projected_water_seed{5601+i}'/'best.pt') if arm!='scratch' else None,
                tag='projected_d4_replication')
            name=f'seven_d4_projected_{arm}_s{5701+i}'
            promote_success(ROOT/'projected_d4_replication'/name,ROOT/'projected_transfer'/name)

if __name__=='__main__':main()
