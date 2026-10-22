"""Summarize all attempts, keeping censored runs in first-discovery curves."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'result/experiments/discovery_transfer_5h_20260914'

def report():
    rows=[]
    for file in sorted(ROOT.glob('*/*/result.json')):
        r=json.loads(file.read_text(encoding='utf-8'))
        hit=r['first_hit'] or {}
        rows.append(dict(series=file.parent.parent.name,task=r['task'],architecture=r['architecture'],arm=r['arm'],seed=r['seed'],source_seed=r['source_seed'],success=int(r['success']),steps=r['steps'],budget=r['budget'],first_hit_steps=hit.get('steps',''),first_hit_seconds=hit.get('seconds',''),seconds=r['seconds'],projection_calls=r.get('projection_calls',''),updates=r['updates'],stop_reason=r['stop_reason'],path=str(file.relative_to(ROOT))))
    if not rows:return
    with (ROOT/'all_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    main=[r for r in rows if r['series']=='main']
    tasks=sorted(set(r['task'] for r in main))
    if tasks:
        fig,axes=plt.subplots(1,len(tasks),figsize=(6*len(tasks),4),squeeze=False)
        for ax,task in zip(axes[0],tasks):
            for arm,color in [('scratch','#555555'),('finetune','#0072B2'),('frozen','#D55E00')]:
                group=[r for r in main if r['task']==task and r['arm']==arm]
                if not group:continue
                budget=min([r['budget'] for r in group]+[r['steps'] for r in group if not r['success']])
                x=np.unique([0,budget]+[r['steps'] for r in group if r['success'] and r['steps']<=budget])
                y=[sum(r['success'] and r['steps']<=v for r in group)/len(group) for v in x]
                ax.step(x,y,where='post',label=f'{arm} (completed n={len(group)})',color=color)
            ax.set(title=task,xlabel='Environment proposals, including rejections',ylabel='Fraction with first strict success',ylim=(-.02,1.02))
            ax.legend();ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(ROOT/'first_discovery.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,len(tasks),figsize=(6*len(tasks),4),squeeze=False)
        colors={'scratch':'#555555','finetune':'#0072B2','frozen':'#D55E00'}
        for ax,task in zip(axes[0],tasks):
            seen=set()
            for r in main:
                if r['task']!=task:continue
                path=ROOT/Path(r['path']).parent/'episodes.csv'
                with path.open(encoding='utf-8') as f:episodes=list(csv.DictReader(f))
                x=[float(e['cumulative_steps']) for e in episodes]
                q=np.minimum.accumulate([max(float(e['flatness'])/.001,float(e['spread_error'])/.05) for e in episodes])
                arm=r['arm'];ax.plot(x,q,color=colors[arm],alpha=.55,label=arm if arm not in seen else None);seen.add(arm)
            ax.axhline(1,color='black',ls=':',label='Success threshold')
            ax.set_yscale('log');ax.set(title=task,xlabel='Environment proposals',ylabel='Best terminal max(F / 0.001, D error / 0.05)')
            ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(ROOT/'terminal_progress.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,architecture in zip(axes,('dense','graph')):
        for file in sorted((ROOT/'source').glob(f'{architecture}*/learning.csv')):
            if '5699' in str(file):continue
            with file.open(encoding='utf-8') as f:learning=list(csv.DictReader(f))
            ax.plot([int(r['update']) for r in learning],[float(r['validation_loss']) for r in learning],lw=1,alpha=.7,label=file.parent.name)
        ax.set(title=f'{architecture}: source-task supervision only',xlabel='Supervised updates',ylabel='Held-out source-path loss');ax.set_yscale('log');ax.grid(alpha=.2)
        ax.legend(fontsize=5,ncol=2)
    fig.tight_layout();fig.savefig(ROOT/'source_learning.png',dpi=180);plt.close(fig)
    summary=[]
    for key in sorted(set((r['series'],r['task'],r['architecture'],r['arm']) for r in rows)):
        group=[r for r in rows if (r['series'],r['task'],r['architecture'],r['arm'])==key]
        summary.append(dict(series=key[0],task=key[1],architecture=key[2],arm=key[3],n=len(group),successes=sum(r['success'] for r in group),steps=[r['steps'] if r['success'] else None for r in group],mean_observed_steps=float(np.mean([r['steps'] for r in group])),seeds=[r['seed'] for r in group]))
    (ROOT/'aggregate.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps([s for s in summary if s['series']=='main'],indent=2))

if __name__=='__main__':report()
