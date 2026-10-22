"""Source-model/search-seed matrices; missing and censored runs stay explicit."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_discovery_findings import ROOT,read_results,d4_common_rows,write_json


def crossed(rows,task,bars_only=False):
    if bars_only:
        diagonal=[r for r in rows if r['series']=='projected_bars_transfer' and r['task']==task]
        prefix='crossed_bars_d4_source'
    else:
        diagonal=d4_common_rows(rows) if task=='seven_d4' else [r for r in rows if r['series']=='projected_nine' and r['task']==task]
        prefix='crossed_d4_source' if task=='seven_d4' else 'crossed_nine_source'
    candidates=diagonal+[r for r in rows if r['series'].startswith(prefix)]
    result=[]
    for source in (5601,5602,5603):
        for search in (5701,5702,5703):
            matches=[r for r in candidates if r['arm']=='frozen' and r['source_seed']==source and r['seed']==search]
            if not matches:
                result.append(dict(source_seed=source,search_seed=search,status='not_finished'))
                continue
            r=max(matches,key=lambda x:x['steps'])
            result.append(dict(source_seed=source,search_seed=search,
                status='success' if r['success'] else ('budget' if r['steps']>=6400 else 'deadline'),
                steps=r['steps'],updates=r.get('updates',0),series=r['series']))
    return result


def matrix(ax,entries,title):
    values=np.full((3,3),np.nan)
    for e in entries:
        i=e['source_seed']-5601;j=e['search_seed']-5701
        status=e['status']
        values[i,j]=np.log10(e['steps']) if status=='success' else 4.1
        label=str(e['steps']) if status=='success' else ('>= 6,400' if status=='budget' else (f"censored\n{e['steps']}" if status=='deadline' else 'pending'))
        ax.text(j,i,label,ha='center',va='center',fontsize=12,color='white' if values[i,j]<3.2 else 'black')
    ax.imshow(values,cmap='YlGnBu_r',vmin=2.,vmax=4.2)
    ax.set(xticks=range(3),xticklabels=[5701,5702,5703],yticks=range(3),yticklabels=[5601,5602,5603],
           xlabel='Search seed',ylabel='Supervised source seed',title=title)


def main():
    rows=read_results();out={t:crossed(rows,t) for t in ('seven_d4','nine_d4')}
    write_json(ROOT/'crossed_discovery.json',dict(budget=6400,conditions=out,
        interpretation='Three trained source NNs crossed with three search seeds; diagonals reused, not nine independent NNs'))
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,task,title in zip(axes,out,('Unseen D4, seven nodes','Unseen structure + D4, nine nodes')):
        matrix(ax,out[task],title)
    fig.suptitle('Frozen NN: proposals to first discovery (no RL updates)',fontsize=13)
    fig.tight_layout();fig.savefig(ROOT/'crossed_discovery.png',dpi=180);plt.close(fig)
    bars=crossed(rows,'seven_d4',bars_only=True)
    fig,ax=plt.subplots(figsize=(5.5,4.8));matrix(ax,bars,'D2/D3-only experience → unseen D4')
    fig.tight_layout();fig.savefig(ROOT/'bars_crossed_discovery.png',dpi=180);plt.close(fig)
    write_json(ROOT/'bars_crossed_discovery.json',dict(budget=6400,rows=bars,
        interpretation='Three D2/D3-only source NNs crossed with three search seeds; no water or D4 teacher data'))
    d4=d4_common_rows(rows);entries=[]
    fig,ax=plt.subplots(figsize=(7,3.8));values=np.full((3,3),4.1)
    for i,goal in enumerate((2,3,4)):
        for j,seed in enumerate((5701,5702,5703)):
            candidates=d4 if goal==4 else [r for r in rows if r['series']==f'd4_goal_ablation_d{goal}']
            found=[r for r in candidates if r['arm']=='frozen' and r['seed']==seed]
            r=found[0] if found else None
            label='pending'
            if r:
                label=str(r['steps']) if r['success'] else ('>= 6,400' if r['steps']>=6400 else f"censored {r['steps']}")
                values[i,j]=np.log10(r['steps']) if r['success'] else 4.1
                entries.append(dict(policy_D=goal,source_seed=5601+j,search_seed=seed,success=r['success'],steps=r['steps']))
            ax.text(j,i,label,ha='center',va='center',fontsize=12,color='white' if values[i,j]<3.2 else 'black')
    ax.imshow(values,cmap='YlGnBu_r',vmin=2.,vmax=4.2,aspect='auto')
    ax.set(xticks=range(3),xticklabels=['5601 / 5701','5602 / 5702','5603 / 5703'],
        yticks=range(3),yticklabels=[2,3,4],xlabel='Source NN / search seed',ylabel='D supplied to NN',
        title='Actual success goal stays D4; only the NN input changes')
    fig.tight_layout();fig.savefig(ROOT/'d4_goal_input.png',dpi=180);plt.close(fig)
    write_json(ROOT/'d4_goal_input.json',dict(actual_goal=4,budget=6400,rows=entries))


if __name__=='__main__':main()
