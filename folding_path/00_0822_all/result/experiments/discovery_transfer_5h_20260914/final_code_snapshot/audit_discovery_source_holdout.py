"""Check that D2/D3 source states did not already include a flat D4 solution."""
import json
from discovery_transfer import ROOT,np,GoalEnv,config_for,digest,write_json


def main():
    file=ROOT/'water_source/source_data.npz';data=np.load(file);env=GoalEnv(config_for('seven_d4'))
    metrics=[env.compactness_metrics(node) for node in data['node']]
    hits=[i for i,m in enumerate(metrics) if m['flatness_ratio']<=.001 and m['spread_error']<=.05]
    best=min(metrics,key=lambda m:max(m['flatness_ratio']/.001,m['spread_error']/.05))
    out=dict(source_file=str(file),source_sha256=digest(file),source_goals=np.unique(data['goal']).tolist(),
        states=len(metrics),unknown_task='seven_d4',already_successful_D4_source_states=len(hits),matching_indices=hits,
        best_joint_threshold_ratio=max(best['flatness_ratio']/.001,best['spread_error']/.05),best_metrics=best)
    write_json(ROOT/'source_holdout_geometry_audit.json',out);print(json.dumps(out))
    if hits or set(out['source_goals'])!={2.,3.}:raise RuntimeError('D4 holdout/source-goal audit failed')


if __name__=='__main__':main()
