"""Preserve final experiment code, local dependencies, geometry and file hashes."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil


ENTRYPOINTS='''supervised_folding transfer_study graph_transfer replay_transfer discrete_goal_adapter
graph_source_dagger graph_source_static mixed_source_continued online_target_adaptation tune_d4_transfer select_d4_rates
support_subset_robustness generalization_oracles target_branch_diagnostic evaluate_graph_sources diagnose_mode_basis
diagnose_teacher_recovery zero_six_unit_probe analytic_goal_control run_generalization_validation_queue plot_transfer_study
compare_transfer_conditions plot_transfer_losses plot_replay_retention summarize_transfer_failures render_transfer_examples
graph_replay_transfer measure_adaptation_curves render_zero_shot_torus audit_generalization_results nearest_branch_teacher
branch_transfer torus_support_robustness make_generalization_report paired_dataset_uncertainty plot_graph_replay_retention
rollout_saved_student render_multi_goal_example fresh_confirmation summarize_fresh_confirmation scratch_replay_control
tune_scratch_replay fixed_state_relabel evaluate_nearest_fresh archive_generalization_study test_supervised_folding test_transfer_study
test_graph_transfer test_discrete_goal_adapter test_analytic_goal_control'''.split()


def run(root):
    base=Path(__file__).resolve().parent;output=root/'code_snapshot_final';output.mkdir(exist_ok=True)
    pending=list(ENTRYPOINTS);seen=set();files=[]
    while pending:
        name=pending.pop()
        if name in seen: continue
        seen.add(name);file=base/f'{name}.py'
        if not file.exists(): raise FileNotFoundError(file)
        files.append(file)
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8-sig'))):
            imports=([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else
                     [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [])
            for module in imports:
                if (base/f'{module}.py').exists() and module not in seen: pending.append(module)
    files.extend(base/name for name in ('generalization_report_template.md','final_stress_queue.json','supplementary_queue.json') if (base/name).exists())
    for name in ('five_bar_linkage','seven_bar_linkage','4unit_torus','6unit_torus'):
        files.extend(f for f in (base/'model'/name).glob('*') if f.is_file() and f.suffix.lower() in ('.dat','.csv','.txt','.json','.npy','.npz'))
    hashes={}
    for file in sorted(set(files)):
        destination=output/file.relative_to(base);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,destination)
        hashes[str(destination.relative_to(root))]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        file_count=len(hashes),hashes=hashes),indent=2),encoding='utf-8')
    data={}
    for file in root.rglob('*'):
        if file.is_file() and 'code_snapshot_final' not in file.parts and file.suffix in ('.npz','.pt','.dat'):
            data[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
    (root/'final_data_and_checkpoint_hashes.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(dict(code_and_geometry_files=len(hashes),data_and_checkpoint_files=len(data)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
