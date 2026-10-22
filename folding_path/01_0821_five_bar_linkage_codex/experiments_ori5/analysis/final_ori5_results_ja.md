# ori_codex5 最終検証結果

共通の報酬定数を固定し、optimizer等のハイパーパラメータのみを調整して5条件を検証した。five-bar D=2/D=3、waterbomb D=1/D=sqrt(2)は10,000 episodeを完走した。torus D=2はユーザー停止のため、最後のatomic checkpointであるepisode 8,192を正式な打切り点とし、stdoutだけに残ったepisode 8,193–8,800は集計から除外した。

## 結果一覧

| 条件 | endpoint | 初回hit: episode / mechanics / updates | random/RL mechanics RMST比 | final 100: success / crossing | spread error |
|---|---|---:|---:|---:|---:|
| five_d2 | complete | 585 / 233,824 / 73 | 15.898x | 0.880 / 0.120 | 0.00018577 |
| five_d3 | complete | 3 / 1,033 / 0 | 1579.374x | 1.000 / 0.000 | 0.000712964 |
| torus2 | user_stopped_at_checkpoint | 未発見 | -- | 0.000 / 0.000 | 0.400037 |
| water1 | complete | 未発見 | -- | 0.000 / 1.000 | 0.0170814 |
| watersqrt2 | complete | 47 / 17,324 / 5 | 1.047x | 0.930 / 0.000 | 0.000118852 |

## 条件別の解釈

- **five-bar D=2**: episode 585、233,824 mechanics step、73 policy updates後に初回成功した。random mechanics RMST比は15.898倍、直接計時したRLは1176.80秒、校正random RMSTは8174.40秒（記述比6.95倍）。final 100は成功0.88、自己交差0.12、平均spread error 0.000186。学習による初回発見、学習後の安定化、spread制御を確認したが、交差モードは残った。
- **five-bar D=3**: final 100は成功1.00、自己交差0。初回成功はepisode 3、1,033 mechanics stepだが、policy updateは0回である。したがって速い初回発見は未更新NNの初期方策分布によるもので、報酬学習の効果とは主張できない。
- **waterbomb D=sqrt(2)**: episode 47、17,324 mechanics step、5 updates後に初回成功し、final 100は成功0.93、自己交差0、平均spread error 0.000119。ただしrandom mechanics RMSTは18,132.5で優位は1.047倍にすぎず、random中央値10,718.5はRLより早い。random 10試行中6試行もRLより早かった。さらにRL直接計時110.83秒に対し校正random RMSTは20.71秒で、wall上はRLが約5.35倍遅い。
- **torus D=2**: episode 8,192まで成功・自己交差・wrong-flatはすべて0で、全episodeがtimeoutだった。qF=q=0、joint milestoneは0、spread-only milestoneは3/8,192だけで、final-seed 100評価も全timeoutだった。flatness near-miss閾値へ入れないdead zoneであり、この共通報酬では誘導信号が不足した。
- **waterbomb D=1**: 成功0。last 1,000の自己交差率0.999、final 100の自己交差率1.00で、自己交差attractorへ収束した。spread milestoneを安定して得る交差経路がtimeoutより高評価になり、回避方策を学べなかった。

## 初回発見とrandom比較

episode/mechanics比較には既存の互換な一様random試行を用い、各条件で共通の行政打切り上限に再censorしたRMSTを使った。five-bar D=2では明確にrandomより早かった。一方、D=3は0 updateなので学習効果を示さず、waterbomb D=sqrt(2)のmechanics優位は弱くwallでは負けた。torusとwaterbomb D=1はRL・randomとも発見なしであり、速度比は定義しない。

wall比較のRL値は、seed 2101を同一設定で再実行し、full runの保存済みepisode/update/first-success pathと一致した場合の rollout + Actor–Critic update時間である。random値は、同じidle machine上の固定uniform prefix throughputでcommon-censor mechanics RMSTを換算した校正wall-equivalentで、random first-discovery wallを直接計時した値ではない。単一RL seed対複数random seed RMSTの記述比較であり、推測統計的なspeedupではない。

## 総合結論

共通報酬はfive-bar D=2、five-bar D=3、waterbomb D=sqrt(2)で学習後の安定折り畳みとspread制御を実現した。しかし、全構造を包括して「初回発見をrandomより高速化する」とは結論できない。学習による明確な初回高速化を確認できたのはfive-bar D=2だけである。torusのflatness dead zoneとwaterbomb D=1の自己交差attractorを解消するには、報酬形態自体の追加改訂が必要であり、ハイパーパラメータ調整だけでは不十分だった。

## 図

- [five-bar D=2 learning curve](full_five_d2_profiles.png)
- [five-bar D=3 learning curve](full_five_d3_profiles.png)
- [torus D=2 learning curve（episode 8,192で打切り）](full_torus2_profiles.png)
- [waterbomb D=1 learning curve](full_water1_profiles.png)
- [waterbomb D=sqrt(2) learning curve](full_watersqrt2_profiles.png)
- [全5条件の初回発見mechanics比較](full_random_first_hit_comparison.png)
- [成功3条件の校正wall比較](final_ori5_calibrated_wall_comparison.png)
- episode/mechanics ECDF: [five D=2](full_five_d2_random_ecdf_survival.png), [five D=3](full_five_d3_random_ecdf_survival.png), [torus](full_torus2_random_ecdf_survival.png), [water D=1](full_water1_random_ecdf_survival.png), [water D=sqrt(2)](full_watersqrt2_random_ecdf_survival.png)

## 機械可読データ

- [最終比較CSV](final_ori5_comparison.csv)
- [最終比較JSON](final_ori5_comparison.json)
- [校正wall比較CSV](final_ori5_calibrated_wall_comparison.csv)
- [wall replay検証](final_ori5_wall_replay_validation.json)
- [artifact QA](final_ori5_artifact_qa.json)
- [torus派生評価](derived_torus_user_stopped_ep8192/evaluation_summary.json)
