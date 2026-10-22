"""Account for scheduled artifacts, preserve code, and record bounded completion.

Artifact counts are not independent statistical trials: resumed and promoted
prefixes are explicitly marked, and inferential tables deduplicate separately.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from discovery_transfer import ROOT,BASE,DEADLINE,digest,write_json,write_csv


def inventory():
    jobs={}
    def add(tag,task,architecture,arm,seed,budget,source_seed=None,plan='explicit protocol'):
        key=f'{tag}/{task}_{architecture}_{arm}_s{seed}'
        jobs[key]=dict(series=tag,task=task,architecture=architecture,arm=arm,seed=seed,
            source_seed=source_seed,budget=budget,plan=plan,path=key)
    for file in ROOT.glob('*queue.json'):
        records=json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(records,list):continue
        for j in records:
            if not isinstance(j,dict) or not {'task','architecture','arm','seed','budget'}<=j.keys():continue
            add(j.get('tag','main'),j['task'],j['architecture'],j['arm'],j['seed'],j['budget'],j.get('source_seed'),file.name)
    def cube(tag,tasks,arch,arms,seeds,budget):
        for task in tasks:
            for arm in arms:
                for seed in seeds:add(tag,task,arch,arm,seed,budget,seed-100)
    seeds=range(5701,5704)
    cube('extended',['seven_d4'],'dense',['scratch','finetune'],seeds,256000)
    cube('torus_depth',['torus_d2'],'projected',['scratch','finetune'],[5701],256000)
    cube('projected_nine',['nine_d3','nine_d4'],'projected',['scratch','frozen'],seeds,6400)
    cube('projected_nine_zero',['nine_d3','nine_d4'],'projected',['deterministic'],seeds,400)
    cube('projected_nine_zero_control',['nine_d3','nine_d4'],'projected',['untrained_deterministic'],seeds,400)
    cube('projected_d4_replication',['seven_d4'],'projected',['scratch','finetune','frozen'],[5702,5703],6400)
    cube('d4_graph_matched',['seven_d4'],'graph',['scratch','frozen'],seeds,6400)
    cube('nine_graph_matched',['nine_d4'],'graph',['frozen'],seeds,6400)
    cube('nearest_prior_torus',['torus_d2'],'projected',['scratch','finetune','frozen'],seeds,6400)
    cube('delta_prior_torus',['torus_d2'],'projected',['scratch','frozen'],seeds,6400)
    cube('source_projected_bars',['five_d2','five_d3','seven_d2','seven_d3'],'projected',['deterministic'],seeds,400)
    cube('projected_bars_zero',['seven_d4','nine_d4'],'projected',['deterministic'],seeds,400)
    cube('projected_bars_transfer',['seven_d4'],'projected',['frozen','finetune'],seeds,6400)
    cube('projected_bars_transfer',['nine_d4'],'projected',['frozen'],seeds,6400)
    for goal in (2,3):cube(f'd4_goal_ablation_d{goal}',['seven_d4'],'projected',['frozen'],seeds,6400)
    for seed in range(6901,6905):
        for arm in ('frozen','scratch'):add('fresh_d4','seven_d4','projected',arm,seed,6400,5601)
        add('gradient_fresh_d4','seven_d4','analytic','gradient_w1p0',seed,400)
    for task,prefix in [('seven_d4','crossed_d4_source'),('nine_d4','crossed_nine_source'),('seven_d4','crossed_bars_d4_source')]:
        for source in (5601,5602,5603):
            for seed in seeds:
                if source+100!=seed:add(f'{prefix}{source}',task,'projected','frozen',seed,6400,source)
    # Include completed direct-run studies and any started design even if their
    # controller did not use a queue file. Do not count smoke tests.
    for file in ROOT.glob('*/*/design.json'):
        if file.parent.parent.name=='smoke':continue
        d=json.loads(file.read_text(encoding='utf-8'));key=str(file.parent.relative_to(ROOT)).replace('\\','/')
        if key not in jobs:add(file.parent.parent.name,d['task'],d['architecture'],d['arm'],d['seed'],d['budget'],d.get('source_seed'),'saved design')
    rows=[]
    for key,j in sorted(jobs.items()):
        directory=ROOT/key;result=directory/'result.json'
        row=dict(**j,status='not_started',steps=0,success=False,stop_reason='',inherited_success_from='',resume_from='')
        if result.exists():
            r=json.loads(result.read_text(encoding='utf-8'))
            row.update(status='saved',steps=r['steps'],success=r['success'],stop_reason=r['stop_reason'],
                inherited_success_from=r.get('inherited_success_from') or '',resume_from=r.get('resume_from') or '')
        elif (directory/'design.json').exists():
            row['status']='started_without_final_result'
            episodes=directory/'episodes.csv'
            if episodes.exists():
                with episodes.open(encoding='utf-8') as f:records=list(csv.DictReader(f))
                if records:row['steps']=int(records[-1]['cumulative_steps'])
        elif j['series']=='selected':row['status']='not_started_reprioritized_to_torus_depth'
        rows.append(row)
    write_csv(ROOT/'job_inventory.csv',rows)
    sources=[]
    profiles={'dense':(5,5000),'graph':(5,5000),'graph_long':(5,20000),'graph_coverage':(5,20000),
              'graph_water':(5,20000),'graph_seven':(3,20000),'graph_seven_d4':(3,20000),
              'projected_water':(3,20000),'projected_bars':(3,20000)}
    for label,(count,updates) in profiles.items():
        for seed in range(5601,5601+count):
            file=ROOT/'source'/f'{label}_seed{seed}'/'summary.json'
            r=json.loads(file.read_text(encoding='utf-8')) if file.exists() else {}
            sources.append(dict(label=label,seed=seed,planned_updates=updates,observed_updates=r.get('updates',0),
                status='saved' if r else ('started' if file.parent.exists() else 'not_started')))
    write_csv(ROOT/'source_inventory.csv',sources)
    out=dict(updated_utc=datetime.now(timezone.utc).isoformat(),artifact_counts=dict(Counter(r['status'] for r in rows)),
        saved_stop_reasons=dict(Counter(r['stop_reason'] for r in rows if r['status']=='saved')),
        inherited_success_artifacts=sum(bool(r['inherited_success_from']) for r in rows),
        resumed_artifacts=sum(bool(r['resume_from']) for r in rows),source_counts=dict(Counter(r['status'] for r in sources)),
        interpretation='Directory/artifact counts, NOT independent trial counts. Success prefixes and resumed runs must be deduplicated for comparisons.')
    write_json(ROOT/'job_inventory.json',out)
    return out


def snapshot():
    target=ROOT/'final_code_snapshot';target.mkdir(exist_ok=True)
    files=set()
    for pattern in ('discovery*.py','extend_discovery.py','report_discovery*.py','render_discovery*.py',
                    'diagnose_discovery*.py','audit_discovery*.py','test_discovery_transfer.py','finalize_discovery_study.py','reproduce_discovery_hits.py'):
        files.update(BASE.glob(pattern))
    for name in ('main.py','agent.py','policy.py','env.py','mechanics.py','mechanics_origin.py',
                 'supervised_folding.py','transfer_study.py'):
        file=BASE/name
        if file.exists():files.add(file)
    for file in files:shutil.copy2(file,target/file.name)
    geometry=[]
    for model in ('five_bar_linkage','seven_bar_linkage','waterbomb','4unit_torus'):
        for name in ('node_initial.dat','member.dat','fix.dat'):
            src=BASE/'model'/model/name
            if src.exists():
                dst=target/'model'/model/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                geometry.append(str(dst.relative_to(target)))
    for file in (ROOT/'holdout_nine').glob('*.dat'):
        if file.name not in ('node_initial.dat','member.dat','fix.dat'):continue
        dst=target/'holdout_nine'/file.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(file,dst)
    original=json.loads((ROOT/'code_snapshot/manifest.json').read_text(encoding='utf-8'))
    core={name:dict(original_sha256=original[name],final_sha256=digest(BASE/name),unchanged=original[name]==digest(BASE/name))
          for name in ('main.py','agent.py','policy.py','env.py','mechanics.py')}
    write_json(ROOT/'core_code_audit.json',core)
    manifest={str(f.relative_to(target)).replace('\\','/'):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='manifest.json'}
    write_json(target/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=manifest,
        core_unchanged=all(r['unchanged'] for r in core.values()),
        purpose='Final code and initial geometry snapshot; no unknown target coordinates are copied. Restore scripts to the original code directory layout for reruns.'))
    if not all(r['unchanged'] for r in core.values()):raise RuntimeError('Original RL core changed')
    return dict(files=len(manifest),core_unchanged=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');p.add_argument('--complete',action='store_true');a=p.parse_args()
    out=inventory()
    if a.snapshot:out['snapshot']=snapshot()
    if a.complete:
        now=datetime.now(timezone.utc);start=datetime(2026,9,14,8,5,7,tzinfo=timezone.utc)
        tests=json.loads((ROOT/'test_status.json').read_text(encoding='utf-8'))
        audit=json.loads((ROOT/'audit.json').read_text(encoding='utf-8'))
        core=json.loads((ROOT/'core_code_audit.json').read_text(encoding='utf-8'))
        replay=json.loads((ROOT/'reproduction_status.json').read_text(encoding='utf-8'))
        holdout=json.loads((ROOT/'source_holdout_geometry_audit.json').read_text(encoding='utf-8'))
        if (tests['returncode'] or audit['failures'] or not replay['all_valid'] or
                not all(r['unchanged'] for r in core.values()) or holdout['already_successful_D4_source_states'] or
                digest(holdout['source_file'])!=holdout['source_sha256'] or out['artifact_counts'].get('started_without_final_result',0)):
            raise RuntimeError('Final validation failed or started runs are still missing results')
        write_json(ROOT/'completion_status.json',dict(start_utc=start.isoformat(),end_utc=now.isoformat(),
            elapsed_seconds=(now-start).total_seconds(),deadline_utc=datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat(),
            finished_within_requested_window=now.timestamp()<=DEADLINE,inventory=out,
            tests=tests,audit_failures=len(audit['failures']),success_paths_audited=audit['success_paths'],
            positive_replay_checks_valid=replay['all_valid'],
            D4_solutions_in_D2_D3_source_states=holdout['already_successful_D4_source_states'],
            frozen_runs_audited=audit['frozen_runs_checked'],core_unchanged=all(r['unchanged'] for r in core.values()),
            interpretation='Bounded exploratory study complete. Deadline-censored and unstarted jobs are retained; unresolved torus discovery is not claimed as success.'))
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
