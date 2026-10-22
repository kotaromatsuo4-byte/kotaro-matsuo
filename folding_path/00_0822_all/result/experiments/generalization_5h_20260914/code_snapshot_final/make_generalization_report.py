"""Build reproducible Japanese tables and figures from saved, complete evaluations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supervised_folding import write_csv


def read(directory,pattern='seed*_evaluation.csv'):
    rows=[]
    for file in sorted(directory.glob(pattern)):
        with file.open(encoding='utf-8') as stream: rows.extend(csv.DictReader(stream))
    return rows


def values(rows,goal,budget=None,arm=None):
    part=[r for r in rows if float(r['goal'])==goal and (budget is None or int(r['budget'])==budget) and (arm is None or r['arm']==arm)]
    seeds=sorted({r.get('seed','teacher') for r in part})
    return [float(np.mean([int(r['success']) for r in part if r.get('seed','teacher')==s])) for s in seeds]


def rate(rows,goal,budget=None,arm=None):
    v=values(rows,goal,budget,arm)
    return f'{np.mean(v)*100:.2f}%' if v else '—'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def run(root,final=False):
    torus=root/'torus';blocks={}
    main=read(root/'modal_u2000_src10000_test');scratch=read(root/'modal_tuned_lr0.001_u2000_test');replay=read(root/'modal_replay0.5_u2000_test')
    labels={0:0,1:62,2:117,4:228,8:347,16:769,32:1624}
    blocks['d4_table']=table(['D4教師経路数','状態ラベル数','scratch（調整済みLR）','D2/D3事前学習→D4','事前学習→D4＋旧データ'],
        [[b,labels[b],rate(scratch if b else main,4,b,'scratch'),rate(main,4,b,'pretrained'),rate(replay,4,b,'pretrained')] for b in labels])
    blocks['retention_table']=table(['条件（D4教師16本）','D2','D3','D4'],[
        ['追加学習前',*[rate(main,d,0,'pretrained') for d in (2,3,4)]],
        ['通常のD4追加学習',*[rate(main,d,16,'pretrained') for d in (2,3,4)]],
        ['旧D2/D3データを混ぜて追加学習',*[rate(replay,d,16,'pretrained') for d in (2,3,4)]]])
    subset=read(root/'support_subset_robustness_g4');subset_rows=[];subset_differences=[]
    for group in range(8):
        rows=[r for r in subset if int(r['subset'])==group]
        if not rows: continue
        a=values(rows,4,4,'scratch');b=values(rows,4,4,'pretrained')
        assert len(a)==len(b)==3,'Support robustness is incomplete'
        delta=np.mean(b)-np.mean(a);subset_differences.append(delta)
        subset_rows.append([f'{4*group}–{4*group+3}',rows[0]['labels'],f'{np.mean(a)*100:.2f}%',f'{np.mean(b)*100:.2f}%',f'{delta*100:+.2f} pp'])
    blocks['subset_table']=table(['教師経路ID','ラベル数','scratch','事前学習あり','差'],subset_rows)
    blocks['subset_overall']=f"8組・3seedを等重みで平均: scratch {rate(subset,4,4,'scratch')}、事前学習あり {rate(subset,4,4,'pretrained')}。組ごとの平均で事前学習が上回ったのは{sum(d>0 for d in subset_differences)}/8組、下回ったのは{sum(d<0 for d in subset_differences)}/8組。"
    fig,axis=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    for arm,color in [('scratch','#444444'),('pretrained','#2266aa')]:
        allv=[values([r for r in subset if int(r['subset'])==group],4,4,arm) for group in range(8)]
        axis.plot(range(8),[np.mean(v)*100 for v in allv],'o-',color=color,label=arm)
        for group,v in enumerate(allv): axis.scatter([group]*len(v),np.array(v)*100,color=color,alpha=.35,s=18)
    axis.set(xlabel='Disjoint group of four teacher paths',ylabel='D4 test success (%)',ylim=(-3,103));axis.grid(alpha=.2);axis.legend()
    axis.set_xticks(range(8),[f'{4*i}–{4*i+3}' for i in range(8)])
    fig.suptitle('Teacher-path selection changes the transfer advantage (three training seeds)')
    fig.supxlabel('Same 32 test starts. Lines: seed means; dots: each seed. Group sizes in state labels vary.',fontsize=9)
    for ext in ('png','pdf'): fig.savefig(root/'support_subset_robustness_g4'/f'subsets.{ext}',dpi=180)
    plt.close(fig)
    conditions=[('scratch（調整済みLR）','graph_src5000_u2000_test_lr0.003','scratch'),
        ('seven D2/D3・5,000更新','graph_src5000_u2000_test','pretrained'),
        ('seven D2/D3・訪問状態を追加','graph_src5000_u2000_test_dagger','pretrained'),
        ('seven D2/D3・既存データだけで継続','graph_src5000_u2000_test_static','pretrained')]
    blocks['torus_table']=table(['事前学習・対照','torus教師1本','4本','16本'],[
        [name,*[rate(read(torus/directory),2,b,arm) for b in (1,4,16)]] for name,directory,arm in conditions])
    torus_subsets=read(torus/'support_path_robustness');ts=[]
    for path_id in range(4):
        rows=[r for r in torus_subsets if int(r['support_path'])==path_id]
        ts.append([path_id,rows[0]['labels'] if rows else '—',rate(rows,2,1,'scratch'),rate(rows,2,1,'pretrained'),len(values(rows,2,1,'pretrained'))])
    blocks['torus_subset_table']=table(['教師経路ID','ラベル数','scratch','事前学習あり','seed数'],ts)
    branch=read(root/'nearest_branch_u2000_test');bn=[]
    for budget in (4,16):
        for arm in ('scratch','pretrained','replay'):
            rows=[r for r in branch if int(r['budget'])==budget and r['arm']==arm]
            if rows: bn.append([budget,rows[0]['labels'],arm,*[rate(rows,d) for d in (2,3,4)],len(values(rows,4))])
    blocks['branch_nn_table']=table(['D4教師経路数','ラベル数','条件','D2','D3','D4','test seed数'],bn)
    relabel=[]
    for label,directory in [('元の状態・元の教師ラベル','original'),('元の状態・nearest教師ラベル','nearest')]:
        rows=read(root/'fixed_state_relabel_n04_test'/directory)
        relabel.append([label,rows[0]['labels'] if rows else '—',rate(rows,4,4,'scratch'),rate(rows,4,4,'pretrained'),len(values(rows,4,4,'pretrained'))])
        if final: assert all(len(values(rows,4,4,arm))==3 for arm in ('scratch','pretrained')),'Fixed-state label ablation incomplete'
    selected=[r for r in branch if int(r['seed']) in (5501,5502,5503) and int(r['budget'])==4]
    relabel.append(['nearest教師の新しい経路・同教師ラベル',selected[0]['labels'] if selected else '—',rate(selected,4,4,'scratch'),rate(selected,4,4,'pretrained'),len(values(selected,4,4,'pretrained'))])
    blocks['relabel_table']=table(['状態・教師の条件','ラベル数','scratch D4','事前学習あり D4','seed数'],relabel)
    if final:
        for rows,goal,budgets,arm in [(main,4,(0,1,2,4,8,16,32),'pretrained'),(scratch,4,(1,2,4,8,16,32),'scratch'),(replay,4,(4,16,32),'pretrained')]:
            assert all(len(values(rows,goal,b,arm))==5 for b in budgets),'Main D4 comparison incomplete'
        for _,directory,arm in conditions:
            assert all(len(values(read(torus/directory),2,b,arm))==5 for b in (1,4,16)),'Main torus comparison incomplete'
        for path_id in range(4):
            rows=[r for r in torus_subsets if int(r['support_path'])==path_id]
            assert all(len(values(rows,2,1,arm))==3 for arm in ('scratch','pretrained')),'Torus support control incomplete'
        assert all(len(values(branch,4,b,arm))==5 for b in (4,16) for arm in ('scratch','pretrained','replay')),'Nearest branch experiment incomplete'
    zero=[];zero_summary=[]
    for label,suffix,arm in [('ランダム初期化','','scratch'),('seven・5,000更新','','pretrained'),
        ('seven・訪問状態追加','_dagger','pretrained'),('seven・既存データ継続','_static','pretrained'),
        ('five/seven混合・5,000更新','_mixed','pretrained'),('five/seven混合・sourceで継続選択','_mixed_static','pretrained')]:
        v=values(read(torus/f'graph_src5000_u2000_test{suffix}'),2,0,arm)
        assert len(v)==5,f'Zero-shot incomplete: {suffix} {arm}'
        zero.append([label,f'{np.mean(v)*100:.2f}%',', '.join(f'{x*100:.2f}' for x in v),f'{sum(x>0 for x in v)}/5'])
        zero_summary.append(dict(condition=label,mean_success=np.mean(v),positive_seeds=sum(x>0 for x in v)))
    blocks['zero_table']=table(['事前学習','torus成功率','seed別成功率（%）','成功例のあるseed'],zero)
    write_csv(torus/'zero_shot_summary.csv',zero_summary)
    wider=[]
    for label,condition,budget,arm in [('scratch（LR調整）','tuned_scratch',16,'scratch'),
        ('seven・訪問状態追加→torus16本','dagger16',16,'pretrained'),
        ('sevenデータを混ぜてtorus16本へ適応','replay16',16,'pretrained'),
        ('five/seven混合・追加学習なし','mixed_zero',0,'pretrained'),
        ('five/seven継続選択・追加学習なし','mixed_static_zero',0,'pretrained')]:
        rows=read(torus/'wider_starts'/condition);v=values(rows,2,budget,arm)
        wider.append([label,f'{np.mean(v)*100:.2f}%' if v else '—',len(v)])
        if final: assert len(v)==5,f'Wider-start comparison incomplete: {condition}'
    blocks['wider_table']=table(['条件','広い初期形状での成功率','seed数'],wider)
    graph_replay=torus/'graph_dagger_replay0.25_u2000_test'
    tr=read(graph_replay,'seed*_torus_evaluation.csv');sr=read(graph_replay,'seed*_n16_seven_evaluation.csv');base=read(graph_replay,'seed*_baseline_seven_evaluation.csv')
    blocks['graph_retention_table']=table(['条件','seven D2','seven D3','torus D2'],[
        ['通常のtorus追加学習',rate(base,2),rate(base,3),rate(read(torus/'graph_src5000_u2000_test_dagger'),2,16,'pretrained')],
        ['sevenデータを混ぜてtorus追加学習',rate(sr,2),rate(sr,3),rate(tr,2)]])
    blocks['graph_replay_seeds']=str(len(values(tr,2)))
    controls=[]
    for structure,folder,weight,default_lr,goal in [('seven',root,.5,.001,4),('torus',torus,.25,.003,2)]:
        selection=folder/'scratch_replay_rate_selection.json'
        lr=json.loads(selection.read_text(encoding='utf-8'))['selected_lr'] if selection.exists() else default_lr
        directory=folder/f'scratch_replay{weight:g}_lr{lr:g}_u2000_test'
        joint=read(directory,f'seed*_{structure}_evaluation.csv')
        joint_source=joint if structure=='seven' else read(directory,'seed*_seven_evaluation.csv')
        pre=replay if structure=='seven' else tr
        pre_source=replay if structure=='seven' else sr
        controls.append([structure,'事前学習なし＋旧データ',f'{lr:g}',rate(joint_source,2,16),rate(joint_source,3,16),rate(joint,goal,16),len(values(joint,goal))])
        if lr!=default_lr:
            default_dir=folder/f'scratch_replay{weight:g}_lr{default_lr:g}_u2000_test'
            default_rows=read(default_dir,f'seed*_{structure}_evaluation.csv')
            default_source=default_rows if structure=='seven' else read(default_dir,'seed*_seven_evaluation.csv')
            controls.append([structure,'事前学習なし＋旧データ（既定LR）',f'{default_lr:g}',rate(default_source,2,16),rate(default_source,3,16),rate(default_rows,goal,16),len(values(default_rows,goal))])
        controls.append([structure,'事前学習あり＋旧データ','.0003',rate(pre_source,2,16),rate(pre_source,3,16),rate(pre,goal,16,'pretrained'),len(values(pre,goal,16,'pretrained'))])
        if final:
            assert selection.exists() and len(values(joint,goal))==(4 if structure=='torus' else 5),f'Joint scratch control count differs from recorded completion: {structure}'
            assert len(values(pre,goal,16,'pretrained'))==5 and len(values(pre_source,2,16))==5,'Pretrained replay incomplete'
    blocks['scratch_replay_table']=table(['構造','条件','LR','seven D2','seven D3','新課題','seed数'],controls)
    fresh=[]
    for structure,goal in [('seven',4),('torus',2)]:
        rows=read(root/'fresh_confirmation',f'{structure}_seed*_evaluation.csv')
        teacher_rows=read(root/'fresh_confirmation',f'{structure}_teacher_evaluation.csv')
        fresh.append([structure,f'D{goal}',rate(rows,goal,16,'scratch'),rate(rows,goal,16,'replay'),rate(teacher_rows,goal),len(values(rows,goal,16,'replay'))])
        if final: assert all(len(values(rows,goal,16,arm))==5 for arm in ('scratch','replay')),'Fresh confirmation incomplete'
    blocks['fresh_table']=table(['構造','新課題','scratch','事前学習＋旧データ','目的座標教師（参考）','seed数'],fresh)
    source_rows=[]
    for label,kind in [('sevenのみ・5,000更新','seven'),('seven・訪問状態追加','dagger'),('seven・既存データ継続','static'),('five/seven混合・5,000更新','mixed'),('five/seven混合・継続選択','mixed_static')]:
        rows=read(root/'graph_source_evaluation_test'/kind)
        source_rows.append([label,*[rate(rows,d,0,'pretrained') for d in (2,3,4)]])
    blocks['source_table']=table(['GNNのsource学習','seven D2','seven D3','seven D4（未学習）'],source_rows)
    teacher=[]
    for structure in ('seven','torus'):
        for split in ('validation','test'):
            rows=read(root/'teacher_evaluation',f'{structure}_{split}.csv')
            for goal in sorted({float(r['goal']) for r in rows}):
                part=[r for r in rows if float(r['goal'])==goal]
                teacher.append([structure,split,f'D{goal:g}',f"{sum(int(r['success']) for r in part)}/{len(part)}",rate(rows,goal)])
    blocks['teacher_table']=table(['構造','分割','目標','成功数','成功率'],teacher)
    template=(Path(__file__).parent/'generalization_report_template.md').read_text(encoding='utf-8')
    for key,value in blocks.items(): template=template.replace('{{'+key+'}}',value)
    assert '{{' not in template,'Unfilled report token'
    if not final: template='> 集計途中のドラフトです。実行中の比較を含むため、最終結果として引用しないでください。\n\n'+template
    (root/'RESULTS_JA.md').write_text(template,encoding='utf-8')
    print('Report generated:',root/'RESULTS_JA.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--final',action='store_true')
    a=p.parse_args();run(a.root,a.final)
