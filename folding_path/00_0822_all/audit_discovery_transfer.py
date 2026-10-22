"""Check saved success paths, source provenance and frozen-actor invariants."""
import os
for _thread_key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_thread_key]='1'
import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
import mechanics
from discovery_transfer import ROOT,BASE,digest,config_for,GoalEnv,write_json

def audit():
    torch.set_num_threads(1)
    successes=[];failures=[];frozen=0;source_hashes=set();checked_data=set()
    cache_file=ROOT/'physical_audit_cache.json'
    physical_cache=json.loads(cache_file.read_text(encoding='utf-8')) if cache_file.exists() else {}
    cache_hits=0
    physical_code=[digest(BASE/name) for name in ('mechanics.py','env.py','main.py','discovery_transfer.py','audit_discovery_transfer.py')]
    for file in ROOT.glob('*/*/result.json'):
        if file.parent.parent.name in ('smoke',):continue
        result=json.loads(file.read_text(encoding='utf-8'))
        design=json.loads((file.parent/'design.json').read_text(encoding='utf-8'))
        if result['stop_reason']=='budget' and result['steps']<result['budget']:
            failures.append(str(file)+': budget censor recorded before the proposal budget')
        if result['success'] and result['first_hit']['steps']!=result['steps']:
            failures.append(str(file)+': first-hit cost differs from stopped proposal count')
        episodes=file.parent/'episodes.csv'
        if episodes.exists():
            with episodes.open(encoding='utf-8') as f:episode_rows=list(csv.DictReader(f))
            if episode_rows and 'cumulative_steps' in episode_rows[-1] and int(episode_rows[-1]['cumulative_steps'])!=result['steps']:
                failures.append(str(file)+': episode trace and final proposal count differ')
        if design['target_coordinates_used']:failures.append(str(file)+': target coordinates enabled')
        checkpoint=design['source_checkpoint']
        final_file=file.parent/'final.pt'
        if final_file.exists():
            final_state=torch.load(final_file,weights_only=False)['model']
            if any(not torch.isfinite(v).all() for v in final_state.values()):failures.append(str(file)+': non-finite model weights')
        if checkpoint:
            if digest(checkpoint)!=design['source_sha256']:failures.append(str(file)+': source checkpoint changed')
            source_hashes.add(design['source_sha256'])
            saved=torch.load(checkpoint,weights_only=False)
            for item in saved.get('source',[]):
                discovered_d4=bool(item.get('discovery_manifest'))
                if discovered_d4:
                    series=file.parent.parent.name
                    permitted=(series=='source_seven_discovered_d4' and result['task'] in ('seven_d2','seven_d3','seven_d4')) or (series in ('sequential_nine','sequential_nine_zero') and result['task'] in ('nine_d3','nine_d4'))
                    if not permitted:failures.append(str(file)+': newly known D4 source leaked into another evaluation')
                    manifest=json.loads(Path(item['discovery_manifest']).read_text(encoding='utf-8'))
                    if digest(item['discovery_manifest'])!=item['discovery_manifest_sha256']:failures.append(str(file)+': sequential source manifest changed')
                    discovery=json.loads(Path(manifest['discovery_result']).read_text(encoding='utf-8'))
                    if not discovery['success'] or discovery['task']!='seven_d4' or manifest['RL_weights_used']:failures.append(str(file)+': invalid sequential experience provenance')
                    for key in ('discovery_result','discovery_coordinates','copied_target','original_source'):
                        if digest(manifest[key])!=manifest[key+'_sha256']:failures.append(str(file)+': sequential '+key+' changed')
                if file.parent.parent.name in ('bar_zero','bar_transfer','fresh_bar','nine_zero','nine_transfer') and np.load(item['path'])['node'].shape[1]!=7:
                    failures.append(str(file)+': five-bar holdout contaminated')
                if item['path'] in checked_data:continue
                checked_data.add(item['path'])
                if digest(item['path'])!=item['sha256']:failures.append(str(file)+': source data changed')
                data=np.load(item['path'])
                water=Path(item['path']).name=='water_source_data.npz'
                allowed={1.,float(np.sqrt(2))} if water else ({2.,3.,4.} if discovered_d4 else {2.,3.})
                if set(np.unique(data['goal']))!=allowed:failures.append(str(file)+': forbidden source goals')
                expected=len(np.loadtxt(BASE/'model/waterbomb/node_initial.dat',delimiter=',')) if water else None
                if data['node'].shape[1] not in ((expected,) if water else (5,7)):failures.append(str(file)+': forbidden source structure')
                if discovered_d4 and data['node'].shape[1]!=7:failures.append(str(file)+': sequential source must contain seven-node states only')
        if result['arm'] in ('frozen','deterministic'):
            frozen+=1
            if not result['frozen_actor_unchanged']:failures.append(str(file)+': frozen actor changed')
            final=final_state
            if any(not torch.equal(value,final[key]) for key,value in saved['actor'].items()):failures.append(str(file)+': frozen artifact differs from source')
        if not result['success']:continue
        config=config_for(result['task']);geometry=Path(config['model_directory'])
        fingerprint_items=[physical_code,config,digest(file.parent/'first_hit_path.npz'),
            [digest(geometry/name) for name in ('node_initial.dat','member.dat','fix.dat')],
            digest(design['initial_file']) if design.get('initial_file') else None,design.get('initial_sha256')]
        fingerprint=hashlib.sha256(json.dumps(fingerprint_items,sort_keys=True,default=str).encode()).hexdigest()
        cache_key=str(file.relative_to(ROOT))
        cached=physical_cache.get(cache_key)
        if cached and cached['fingerprint']==fingerprint and cached['record']['valid']:
            successes.append(cached['record']);cache_hits+=1;continue
        env=GoalEnv(config)
        if design.get('initial_file'):
            if digest(design['initial_file'])!=design['initial_sha256']:failures.append(str(file)+': initial condition changed')
            env=GoalEnv(config_for(result['task']),data=(np.loadtxt(design['initial_file'],delimiter=','),env.member,env.fix))
        nodes=np.load(file.parent/'first_hit_path.npz')['nodes']
        initial_ok=np.allclose(nodes[0],env.node0,rtol=0,atol=1e-9)
        max_strain=max(mechanics.current_strain(node,env.member,env.lengths) for node in nodes)
        fixed_error=float(np.max(np.abs(nodes[:,env.fix[:,0],env.fix[:,1]]-env.node0[env.fix[:,0],env.fix[:,1]])))
        metrics=env.compactness_metrics(nodes[-1])
        success=metrics['flatness_ratio']<=env.evaluation_flatness_tolerance and metrics['spread_error']<=env.spread_tolerance
        crossings=sum(env._has_crossing(a,b) for a,b in zip(nodes[:-1],nodes[1:]))
        good=initial_ok and max_strain<=env.strain_tolerance*1.001 and fixed_error<1e-8 and success and crossings==0
        record=dict(path=str(file.relative_to(ROOT)),initial_matches=initial_ok,max_strain=max_strain,fixed_error=fixed_error,flatness=metrics['flatness_ratio'],spread_error=metrics['spread_error'],crossings=crossings,valid=bool(good))
        successes.append(record)
        physical_cache[cache_key]=dict(fingerprint=fingerprint,record=record)
        if not good:failures.append(record)
    seven_manifest=ROOT/'seven_only_source/manifest.json'
    if seven_manifest.exists():
        manifest=json.loads(seven_manifest.read_text())
        for collector in manifest['collectors']:
            if digest(collector['path'])!=collector['sha256']:failures.append('Seven-only source collector changed')
            for item in collector['source']:
                if np.load(item['path'])['node'].shape[1]!=7:failures.append('Seven-only collector used held-out five-bar data')
    write_json(cache_file,physical_cache)
    output=dict(success_paths=len(successes),paths=successes,frozen_runs_checked=frozen,source_checkpoints_checked=len(source_hashes),
        unchanged_physical_paths_reused=cache_hits,physical_paths_recomputed=len(successes)-cache_hits,failures=failures)
    write_json(ROOT/'audit.json',output);print(json.dumps(output,indent=2))
    if failures:raise RuntimeError('Audit failures')

if __name__=='__main__':audit()
