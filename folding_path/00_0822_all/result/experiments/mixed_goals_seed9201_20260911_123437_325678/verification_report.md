# 探索強化付きD2/D3同時学習：D2 step予算を単独試験にそろえた延長

2026-09-11。学習seed9201。学習・固定評価とも完了。

## 結論

D2に203,363 stepを使うまで延長したが、学習中のD2成功は0/600 episode、near-miss到達は4/600だった。通常NNの最終評価ではD2成功・near-missとも0/100、D3は両方100/100。D2指定でも全100回がD≈3のwrong-flatとなった。

D2に探索補助e=1を付けた最終評価でも成功0/100、near-miss2/100であり、D3側への偏りは解消していない。単独試験程度のD2 step予算まで延長すれば両立する、という結果にはならなかった。ただし、1学習seed・この探索制御での結果であり、同時学習一般の不可能性やNNの表現力不足を確定しない。

## 延長条件

- 再開元: `mixed_goals_seed9201_20260911_122009_729405`の256 episode・32更新後。
- 事前学習なしで始めた同時学習の継続。D2単独checkpointへの切替・初期化は行っていない。
- NN、actor/critic optimizer、行動乱数、目標順序乱数、成功履歴を引き継ぐ。
- D2×4＋D3×4のbatch、同じ報酬、LR、探索制御を維持。
- 各目標の直近32学習episodeの成功率pから、batch開始時にe=max(0,1−p/0.2)を計算。モード係数・符号付きstep幅のGaussianをmu/(1+e)、v+eへ変換し、同じ変換後分布のlog-probabilityで更新。
- D2は全期間e=1。D3は学習が進むとe=0。評価結果を学習・controllerへ戻さない。
- actor LR=0.0005、critic LR=0.00015、batch=8、actor損失の固定参照horizon400、batch全体advantage centering。
- spectral重み0.05、joint potential差、wrong-flat罰則−2とshaping取消。crossはpenaltyなしreject・episode続行。物理step幅上限0.1、最大400 step/episode。
- 終了条件は再開前を含むD2累積step≥202,970となった最初の完全なbatch。安全上のepisode上限10,000には到達していない。

```text
python RL_mixed_goals.py --resume result/experiments/mixed_goals_seed9201_20260911_122009_729405 --episodes 10000 --d2-step-budget 202970 --eval-every 512 --workers 2
```

## 学習量・初回成功

| 対象 | 学習episode | step | 学習中成功 | near-miss到達 |
| --- | ---: | ---: | ---: | ---: |
| D2 | 600 | 203,363 | 0 | 4 |
| D3 | 600 | 85,538 | 580 | 582 |
| 合計 | 1,200 | 288,901 | 580 | 586 |

actor/criticとも150更新。今回追加した分は944 episode、118更新、231,740 step（D2は165,202 step追加）。

最終batch直前のD2累積stepは201,975、最終値203,363。目標202,970との差393 step（約0.194%）は4+4のbatchを完了して停止したため。

| 目標 | 初回成功 | 混合episode | その目標のepisode | 成功前NN更新 | 成功まで総step | その目標のstep |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D2 | 未発見 | — | — | — | — | — |
| D3 | 発見済み | 45 | 21 | 5 | 17,766 | 8,224 |

D2の未発見打切りは混合1,200 episode・150更新・総288,901 step時点（D2のみ600 episode・203,363 step）。D3初回成功は再開前の記録であり、今回の延長で発見し直したものではない。

D2の終了理由はwrong-flat319回、timeout281回。D3はsuccess580回、wrong-flat1回、timeout19回。

near-missは、経路中に一度でもF≤0.1かつ|D−D*|≤0.05を満たしたrollout/episodeの割合・数。successを含み、状態数の割合ではない。

## 通常NNの固定評価（探索補助なし）

各checkpointで各目標100 rollout、評価seed950000–950099。評価seedは学習と分離し、checkpoint間では共通。Q*はrollout中のmax(qF×qD)の中央値。

| 混合episode | D2成功/100 | D2 near-miss/100 | D2 Q*中央値 | D3成功/100 | D3 near-miss/100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.06767 | 0 | 2 |
| 256 | 0 | 0 | 0.07091 | 100 | 100 |
| 512 | 0 | 0 | 0.07447 | 100 | 100 |
| 1024 | 0 | 0 | 0.07369 | 100 | 100 |
| 1200 | 0 | 0 | 0.07247 | 100 | 100 |

D2のQ*中央値にも、この区間で明確に改善し続ける傾向は見られない。最終D2終端D中央値2.999126、wrong-flat100回、平均138.67 step。D3終端D中央値2.999280、成功100回、平均121.48 step。

![最終通常NNの終端形状](terminal_distribution.png)

## 探索補助付き固定評価

| 混合episode | D2のe | D2成功/100 | D2 near-miss/100 | D2 Q*中央値 | D3のe | D3成功/100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 0 | 0.08248 | 1 | 0 |
| 256 | 1 | 1 | 2 | 0.08664 | 0 | 100 |
| 512 | 1 | 0 | 1 | 0.09072 | 0 | 100 |
| 1200 | 1 | 0 | 2 | 0.08483 | 0 | 100 |

D2は学習前後で同じe=1、評価seedも同じ。探索補助を付けても到達頻度は低いままで、延長による明確な改善は確認できなかった。

256 episode時点のD2成功1回は評価専用で、学習へ戻していない。保存済みの`five_d2/evaluation_only_success_seed950048`はその過去checkpointによる成功例であり、最終NNの成功や学習中first hitとして扱わない。

## D2単独試験との対比

| 学習方法 | D2 step | D2学習episode | actor更新 | D2学習中初回成功 | 最終通常NN成功/100 | 最終near-miss/100 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| D2単独・探索補助なし | 202,970 | 512 | 64 | episode291、116,214 step | 11 | 72 |
| D2/D3同時・適応探索あり | 203,363 | 600 | 150 | 未発見 | 0 | 0 |

単独試験は`single_goal_five_d2_seed9201_20260910_220547_379979`。D2に使った環境stepはほぼ同じだが、全体step・更新数・他目標の経験・探索分布は同じではない。したがって混合学習だけが唯一の原因と確定する比較ではない。D2単独にも同じ適応探索を加えた対照は未実施。

少なくとも「D2経験量を単独試験と同程度に増やせば解決する」という仮説は今回支持されなかった。探索強化がD2に有効な経路の経験へ十分つながらないのか、経験後の条件別学習・共有更新の干渉が問題なのかは、別途切り分けが必要。探索制御の値や罰則をこの延長中に調整してはいない。

## 検証・実装変更

- 再開処理に過去の`advantage_diagnostics.csv`の読込と`exploration_probe`のコピーを追加。保存履歴を維持する変更で、学習式の変更ではない。
- 元の256 episode分のepisode/診断記録と全33checkpointが完全一致して引き継がれていることを確認。
- 保存U32から再開直後の8 episodeを再現し、全episode結果・次のNN・Torch乱数状態が一致。
- 全150batchがD2×4＋D3×4、episode/診断件数1,200件、step合計288,901と整合。
- 最終actor/critic optimizerはいずれも150更新。終了境界201,975<202,970≤203,363を確認。
- 実行ソースのSHA-256はdesign.jsonと一致。元runは上書きしていない。
- 図8枚を保存し、最終終端分布を目視確認。PDFは作成していない。

0/100は真の成功確率ゼロという意味ではない（95% Wilson区間0–3.70%）。同じ評価seedの時系列は独立した学習seedの反復ではない。今回の結論は1学習seed・約20万D2 stepの範囲に限る。

## 保存物

[学習集計](training_summary.json)、[評価集計](evaluation_summary.csv)、[設定](design.json)。

[Learning curve](learning_curve.png)、[D2 near-miss](five_d2_near_miss.png)、[D3 near-miss](five_d3_near_miss.png)、[D2 Q*](five_d2_quality.png)、[D3 Q*](five_d3_quality.png)、[D2成功率](five_d2_strict_success.png)、[D3成功率](five_d3_strict_success.png)。

[D2探索補助付き最終評価](exploration_probe/five_d2_update00150.csv)、[D3探索補助付き最終評価](exploration_probe/five_d3_update00150.csv)。
