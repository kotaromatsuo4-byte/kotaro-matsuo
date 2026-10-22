"""Reuse main workers for a deeper paired torus test, with exact resumes."""
import csv,json,shutil,time
from concurrent.futures import ThreadPoolExecutor
from discovery_transfer import ROOT,DEADLINE,write_json,datetime,timezone
from discovery_queue import launch

def extend(arm):
    previous=ROOT/'projected_transfer'/f'torus_d2_projected_{arm}_s5701'
    while not (previous/'result.json').exists() and time.time()<DEADLINE-180:time.sleep(2)
    if time.time()>=DEADLINE-180:return dict(arm=arm,status='not_started_deadline')
    result=json.loads((previous/'result.json').read_text())
    if result['success']:
        destination=ROOT/'torus_depth'/previous.name;shutil.copytree(previous,destination)
        result.update(budget=256000,inherited_success_from=str(previous),additional_steps=0);write_json(destination/'result.json',result)
        design=json.loads((destination/'design.json').read_text());design.update(budget=256000,inherited_success_from=str(previous));write_json(destination/'design.json',design)
        return dict(arm=arm,status='inherited_earlier_success')
    with (previous/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
    resumable=result['episodes']==result['updates']*8 and rows[-1]['reason']=='timeout'
    job=dict(task='torus_d2',architecture='projected',arm=arm,seed=5701,source_seed=5601,budget=256000,tag='torus_depth')
    if arm=='finetune':job['checkpoint']=str(ROOT/'source/projected_water_seed5601/best.pt')
    if resumable:job['resume']=str(previous)
    outcome=launch(job);outcome['resumable']=resumable
    return outcome

def main():
    write_json(ROOT/'torus_depth_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        reason='Allocate freed main workers to first-discovery depth after short torus tests remained censored',
        replaces_unstarted_queue='selected_adaptation_queue.json',task='torus_d2',architecture='projected',
        arms=['scratch','finetune'],source_seed=5601,target_seed=5701,budget=256000,
        source_checkpoint_selection='Existing frozen source-path validation checkpoint; no torus teacher or checkpoint reselection',
        interpretation='One matched-seed depth study, not an independent replicate of the earlier 51200-proposal prefixes'))
    while time.time()<DEADLINE-180:
        ledger=ROOT/'main_queue_ledger.jsonl'
        if ledger.exists() and ledger.read_text().count('\n')>=40:break
        time.sleep(3)
    if time.time()>=DEADLINE-180:return
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(extend,('scratch','finetune')))
    write_json(ROOT/'torus_depth_status.json',results)

if __name__=='__main__':main()
