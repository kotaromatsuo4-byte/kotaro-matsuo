# zero-reject二段階実験 最終報告

生成時刻（UTC）: 2026-08-26T23:26:30.216687+00:00

## 結論

- torus D=2 はR1の4096 episode・3 seed gateを通過してseed4203を選び、最終ep10000まで到達した。最終1024 episodeの成功率は89.4%、平均returnは12.89、terminal F中央値は0.0006303、spread error中央値は7.783e-05。
- water D=1 はR1（near-miss spread far=0.40）が4096 gateで正式棄却された。その後、保存trajectoryの反実仮想だけを根拠にfar=0.12へ一項目変更したfresh fallbackを実行し、2/3 seedでgateを通過してseed4301を選択、最終ep9728まで解析した。最終1024 episodeの成功率は51.2%、平均returnは8.339、terminal F中央値は0.0009884、spread error中央値は3.73e-05。
- 4096までは3 seedのscreen evidence、4096以降は選抜1 seedのproduction diagnosticであり、3 seed安定性の確認ではない。
- 旧P0X2RCS05 seed4101@6144との比較はseedも設定も一致しない記述的比較であり、因果効果量として扱わない。

## 因果的な検討順序

1. 旧 P0X2RCS05 seed4101 の6144 episodeを記述的基準として固定。
2. 通常crossing rejectをreward=0のno-op継続に変更。R1はtorusのactor LR初回milestone=1528、waterのnear-miss far=0.40をtask-specific設定として開始し、run途中では変更しなかった。
3. zero-reject R1では torus が4096 gateを通過してseed4203を選択し、water far=0.40は2/3条件を満たさず正式棄却。
4. 保存trajectory上の反実仮想で境界 basin の過大報酬を確認し、最小変更 far=0.12を事前選択。これは再学習結果ではない。
5. R1 water正式棄却、CSV再計算、反実仮想成果物・source prefix hashをfail-closed確認してfresh fallbackを開始。
6. water far=0.12は4096で2/3 seedがpromisingとなりseed4301を選択。
7. torus seed4203はep10000、water seed4301はep9728を最終authorityとして解析。

この順序により、water far=0.12はR1 far=0.40の4096正式棄却後にのみ導入されており、同じrun内の途中変更ではない。したがって両者はfresh cohortとして分離して読む。

## 学習曲線の数値要約

| task / run | endpoint | first1024 return / F / eD | last1024 return / F / eD | success | constructive | terminal joint | strict hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| torus2 / R1 zero-reject seed4203 selected | 10000 | -0.2402 / 1.032 / 0.1113 | 12.89 / 0.0006303 / 7.783e-05 | 89.4% | 96.4% | 95.9% | 915 |
| water1 / fallback far=0.12 seed4301 selected | 9728 | 0.01699 / 1.81 / 0.04158 | 8.339 / 0.0009884 / 3.73e-05 | 51.2% | 95.2% | 69.9% | 524 |
| torus2 / old P0X2RCS05 seed4101@6144 | 6144 | -1.379 / 1.01 / 0.09677 | -0.9628 / 1.014 / 0.4014 | 0.0% | 0.0% | 0.0% | 0 |
| water1 / old P0X2RCS05 seed4101@6144 | 6144 | -1.577 / 0.8232 / 0.3947 | -0.3625 / 0.337 / 0.4041 | 0.0% | 0.0% | 0.0% | 0 |

### basin占有率（最終1024 episode）

| task / run | wrong-plane | boundary-plane 0.10<eD<=0.15 | torus collapse F>=0.9,eD>=0.30 |
|---|---:|---:|---:|
| torus2 / R1 zero-reject seed4203 selected | 0.0% | 0.0% | 0.1% |
| water1 / fallback far=0.12 seed4301 selected | 0.0% | 0.0% | 0.0% |
| torus2 / old P0X2RCS05 seed4101@6144 | 0.0% | 0.0% | 100.0% |
| water1 / old P0X2RCS05 seed4101@6144 | 81.2% | 0.0% | 0.0% |

wrong-planeはwaterでF<=0.5かつeD>=0.30、boundary-planeはwaterでF<=0.5かつ0.10<eD<=0.15、torus collapseはF>=0.9かつeD>=0.30として集計した。

FとeDは小さいほどよい。learning curveではreturn、F、eD、success/constructive、reject率を同じepisode軸で表示し、terminal jointとbasin占有率も追加し、4096 gateを縦線で示した。

## 学習前半→後半と旧runとの差（記述的）

| task | first1024→last1024 Δreturn / ΔF / ΔeD / Δsuccess | current last1024 - old last1024 Δreturn / ΔF / ΔeD / Δsuccess |
|---|---:|---:|
| torus2 | 13.13 / -1.031 / -0.1113 / 89.4% | 13.86 / -1.013 / -0.4013 / 89.4% |
| water1 | 8.322 / -1.809 / -0.04154 / 48.1% | 8.701 / -0.336 / -0.4041 / 51.2% |

returnとsuccessは正、FとeDは負が改善方向。ただしoldとの差はseed4101対選抜seedで、task-specific設定も異なるため、あくまで記述的比較である。

## 4096 seed gateと選択

| phase | task | seed | promising | last1024 return | F median | eD median | constructive | terminal joint | strict |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| R1 | torus2 | 4201 | False | 0.4644 | 0.6559 | 0.05486 | 22.7% | 0.5% | 0 |
| R1 | torus2 | 4202 | True | 0.3858 | 0.7785 | 0.04686 | 17.5% | 0.0% | 0 |
| R1 | torus2 | 4203 | True | 2.448 | 0.08127 | 0.00882 | 90.8% | 55.4% | 64 |
| R1 far=0.40 | water1 | 4201 | False | -0.01589 | 0.1852 | 0.1165 | 0.1% | 0.0% | 0 |
| R1 far=0.40 | water1 | 4202 | False | 0.2359 | 0.1309 | 0.1166 | 0.0% | 0.0% | 0 |
| R1 far=0.40 | water1 | 4203 | True | 0.6545 | 0.2265 | 0.1137 | 8.8% | 0.0% | 0 |
| fallback far=0.12 | water1 | 4301 | True | 6.609 | 0.06105 | 0.000688 | 87.3% | 57.0% | 387 |
| fallback far=0.12 | water1 | 4302 | False | -0.1104 | 0.1871 | 0.116 | 0.1% | 0.0% | 0 |
| fallback far=0.12 | water1 | 4303 | True | 6.67 | 0.06358 | 0.0007295 | 89.6% | 55.8% | 391 |

R1 torusはseed4203、fallback waterはseed4301が、promising seed内でlast512/last1024 persistenceを優先する事前登録順序により選択された。R1 waterはpromising seedが2未満だったため選択せず停止した。

## zero-reject no-opとreward整合性

- 記録された通常reject action数: 875,996
- `ori6_zero_reject_total_reward` の最大絶対値: 0.000e+00
- 通常crossing候補はlast-safe stateを維持し、accepted width=0、displacement=0、reward=0としてstepを消費する。
- horizonでrejectした場合だけ通常timeout `-1 + q(last_safe)` を別成分として一度与える。通常no-op reward=0とは区別した。

## optimizer診断（各runの最終512 episode）

| run | critic loss mean/p95/max | critic grad mean/p95/max; >1 | actor grad mean/p95/max; >0.5 | entropy | actor LR | reject rate |
|---|---:|---:|---:|---:|---:|---:|
| R1 zero-reject seed4201 screen@4096 | 0.1306/0.2871/0.4771 | 0.4293/0.8862/1.411; 3.1% | 0.6942/1.365/1.851; 62.5% | 2.318 | 0.0001 | 0.2% |
| R1 zero-reject seed4202 screen@4096 | 0.03825/0.07523/0.09205 | 0.2005/0.4566/0.7078; 0.0% | 0.2838/0.4882/0.5245; 1.6% | 2.147 | 0.0001 | 0.0% |
| R1 zero-reject seed4203 selected | 15.14/42.86/50.86 | 13.51/32.15/38.92; 100.0% | 0.474/0.8133/1.372; 37.5% | 2.115 | 3e-05 | 6.5% |
| R1 far=0.40 seed4201 rejected@4096 | 0.9195/1.466/1.703 | 0.681/1.352/1.75; 18.8% | 0.0908/0.1549/0.2453; 0.0% | 2.261 | 5e-05 | 4.4% |
| R1 far=0.40 seed4202 rejected@4096 | 0.8992/1.331/9.664 | 0.8845/2.123/3.287; 25.0% | 0.08302/0.1241/0.1631; 0.0% | 2.042 | 5e-05 | 5.2% |
| R1 far=0.40 seed4203 rejected@4096 | 0.2745/0.606/1.115 | 0.4346/0.9838/1.77; 4.7% | 0.1044/0.1849/0.2355; 0.0% | 2.12 | 5e-05 | 6.4% |
| fallback far=0.12 seed4301 selected | 35.38/48.43/56.54 | 10.97/25.26/37.93; 100.0% | 0.1395/0.2419/0.3519; 0.0% | 2.168 | 1.5e-05 | 12.1% |
| fallback far=0.12 seed4302 screen@4096 | 1.129/2.136/3.276 | 0.9701/2.103/4.064; 40.6% | 0.08518/0.1451/0.1941; 0.0% | 2.24 | 5e-05 | 5.2% |
| fallback far=0.12 seed4303 screen@4096 | 29.63/58.6/63.15 | 8.043/21.39/23.73; 95.3% | 0.1292/0.2072/0.2867; 0.0% | 2.144 | 5e-05 | 13.0% |

gradient normはclip前の記録であり、`>1`（critic）と`>0.5`（actor）は閾値超過更新の割合。critic lossの一時的shock、actor更新量、entropy、適用LRの変化を別々に確認できる。

## 最終policy評価：stochasticとdeterministicの区別

| run | mode | episode | n | success | return | F | eD | 解釈 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R1 zero-reject seed4203 selected | stochastic_heldout | 10000 | 64 | 87.5% | 12.61 | 0.03378 | 0.01314 | 64-rollout final held-out stochastic estimate |
| R1 zero-reject seed4203 selected | deterministic_final_single | 0 | 1 | 0.0% | -0.4676 | 1.467 | 0.07703 | one deterministic final rollout; not a success-rate estimate |
| fallback far=0.12 seed4301 selected | periodic_stochastic | 9728 | 16 | 31.2% | 5.94 | 0.1754 | 0.00717 | latest periodic stochastic diagnostic; not final held-out |
| fallback far=0.12 seed4301 selected | periodic_deterministic_single | 9728 | 1 | 0.0% | -0.1009 | 1.127 | 0.03546 | one latest periodic deterministic rollout; not held-out rate evidence |
| old P0X2RCS05 seed4101@6144 | periodic_stochastic | 6000 | 25 | 0.0% | -0.96 | 1.02 | 0.3958 | latest periodic stochastic diagnostic; not final held-out |
| old P0X2RCS05 seed4101@6144 | periodic_deterministic_single | 6000 | 1 | 0.0% | -0.8959 | 1.068 | 0.3615 | one latest periodic deterministic rollout; not held-out rate evidence |
| old P0X2RCS05 seed4101@6144 | periodic_stochastic | 6000 | 25 | 0.0% | -0.2858 | 0.3044 | 0.4036 | latest periodic stochastic diagnostic; not final held-out |
| old P0X2RCS05 seed4101@6144 | periodic_deterministic_single | 6000 | 1 | 0.0% | -0.7225 | 0.7342 | 0.3736 | one latest periodic deterministic rollout; not held-out rate evidence |

`stochastic_heldout`はfinal evaluation seedでの64 rollout平均である。`deterministic_final_single`は平均成功率ではなく、mean actionによる1 trajectoryの結果である。したがって、両者が異なっても矛盾ではなく、探索分散を含むpolicyとmean policyの挙動差を示す。10000未満で停止したrunについてはfinal held-outと呼ばず、最新periodic diagnosticと明記した。

## endpoint authority

- R1 zero-reject seed4203 selected: ep10000 / `complete_final_policy_history_update_summary_config`; checkpoint=9728, policy=10000, history=10000, update=10000。
  - checkpoint.pt=9728でも、policy.pt・training_history・update_history・summary・configがすべて10000で完結しているため、自然終了10000を優先して回収した。
- fallback far=0.12 seed4301 selected: ep9728 / `periodic_atomic_checkpoint`; checkpoint=9728, policy=10000, history=10000, update=10000。

## 整合性・再現可能性

- 全integrity check: PASS
- input hash set SHA-256: `c6c84aad5e807aaf58bace4b760892b2852be5bfac5cb859c559bb352d561904`
- `final_metrics.json`に全input fileの絶対pathとSHA-256、endpoint authority、gate、selection、window統計を保存した。
- `integrity.json` / `integrity_checks.csv`に個別checkを保存した。
- `artifact_sha256.json`にこの報告・CSV・JSON・PNGとpostprocessor/test sourceの最終SHA-256を保存する。

主要authority fileのSHA-256:

- r1_execution_manifest: `108588a09a1b5a2af7f4fb9eefaf9b3b162da52ff1ea2596a208ed9ed56c14d2` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\experiments_ori6\manifests\zero_reject_staged_dfc53efe4698.json`
- fallback_execution_manifest: `09ff80da2c88855a4efee85fb8cf935de14be0858ff45613bf52f58a34af2a8d` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\experiments_ori6\manifests\zero_reject_water_fallback_3e8c4c36ffce.json`
- r1_torus_formal_gate: `c7cfb7a6cde33939ed669a2a5060931a2d491f4ae60e8124c167d40edf241e86` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\experiments_ori6\staged\zero_reject_reward\P0X2RCS05ZR0R1\analysis\checkpoint_4096_torus2\checkpoint_analysis.json`
- r1_water_formal_gate: `18e1441083ae80e9ca10f0407e335ab3a7b1867c21ab53782e465de49ea224e9` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\experiments_ori6\staged\zero_reject_reward\P0X2RCS05ZR0R1\analysis\checkpoint_4096_water1\checkpoint_analysis.json`
- fallback_water_formal_gate: `b3440061e67d39e029ef2d7a24826c7b8b8000d1ead9e40941a6c18419b35d9b` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\experiments_ori6\staged\zero_reject_water_fallback\P0X2RCS05ZR0WF12\analysis\checkpoint_4096_water1\checkpoint_analysis.json`
- combined_postprocessor_source: `e1a263705f429a5c5113166df1350e94c19f7bbb637f820616ad888c75b1a989` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\finalize_ori_codex6_zero_reject_combined.py`
- combined_postprocessor_test: `ab3258fc63e16db0c524a480ffe775e949a8a54a3db1250e5092349ce66a32f2` — `C:\Users\Kotaro Matsuo\OneDrive\Documents\2026\master_thesis\program\folding_path\01_0821_five_bar_linkage_codex\test_finalize_ori_codex6_zero_reject_combined.py`

## 出力

- `learning_curve_episode_comparison.png`: return、F、eD、strict/joint/constructive、basin占有率、reject率
- `optimizer_reject_diagnostics.png`: critic loss/grad、actor grad、entropy、LR、reject率
- `policy_evaluation_comparison.png`: stochastic estimateとdeterministic single trajectory
- `final_windows_512.csv`: 全runの非重複512 window統計（端数endpointでは真の最終512 overlap行も追加）
- `final_seed_gates.csv`: gate・selection・collapse判定履歴
- `final_evaluations.csv`: evaluation modeとsample数を分離したpolicy評価
- `causal_chronology.csv`: 二段階実験の検討順序とauthority
