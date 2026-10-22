# 根拠資料の索引

整理日：2026-10-22。以下はローカルの `folding_path/` を起点とする相対パス。ソースや元報告をGitHubへ転載せず、共有ノートから再調査できるように記録する。コードは関連する設定・制御・関数を確認したもので、全行の監査ではない。保存報告の数値は今回再実行していない。

会話の根拠C01–C19は [会話索引](conversations.md) を参照。2026年5月・7月という初期の時期の大枠は、2026-10-22の本人の振り返りによる。

## 初期探索・多様解

| ID | パス | 確認した役割 |
|---|---|---|
| S01 | `01_0511_five_bar_linkage/ori_reinfore.py`、`01_0511_five_bar_linkage/ori_actor_critic.py` | REINFORCEとActor–Criticの初期実装。実験の成功率の証明ではない |
| S02 | `01_0516_five_bar_linkage_every_step_opt/ori_CMAES_randomize+gravity_potential.py` | 初期不整、疑似重力ポテンシャルの局所最小化。CMA-ES等の探索要素まで決定論的という意味ではない |
| S03 | `02_4unit_torus/ori_reinfore.py`、`02_4unit_torus_kai/ori_reinfore.py`、`02_4unit_torus_random/ori_reinfore.py`、`03_1dof_tri_panel/ori_reinfore.py`、`04_yoshi_unit/ori_reinfore.py` | 構造を変えた試行。random版は学習関数が存在しても実行ループの更新が無効で、行動はrandom |
| S04 | `00_0804_codex2/README.md`、`00_0804_codex2/qd_folding/protocol_search.py` | 候補集合と選択法の二段階比較。旧MAP-Elitesコードの存在と現在の入口を区別 |
| S05 | `01_0804_five_bar_linkage/main4.py`、`01_0813_five_bar_linkage/main5.py` | NN／GPによる終端探索。物理的変位方向を用いる設計 |
| S06 | `01_0813_five_bar_linkage/MAIN5_RESULTS.md` | random・GP-UCB・GP-posteriorのheld-out比較 |
| S07 | `01_0813_five_bar_linkage/MULTISTEP_REACHABILITY_RESULTS.md` | 複数step摂動による到達可能性 |
| S08 | `01_0813_five_bar_linkage/GP_SELECTION_PRESSURE_RESULTS.md`、`01_0813_five_bar_linkage/MAIN6_ONLINE_AGGREGATE.md`、`01_0813_five_bar_linkage/MAIN7_B10000_RESULTS.md` | pilotとオンライン検証。GP予測精度と選択効果を区別 |
| S09 | `01_0813_five_bar_linkage/MAIN8_FINAL_RESULTS.md`、`01_0813_five_bar_linkage/main8.py` | 可変長方向持続macro。行動設計の効果とGP選択の限界 |
| S10 | `01_0813_five_bar_linkage/MAIN9_FINAL_RESULTS.md`、`01_0813_five_bar_linkage/main9.py`、`01_0813_five_bar_linkage/collision_continue.py` | 衝突継続、kinematic／feasible archive、forest shadow |
| S11 | `01_0813_five_bar_linkage/MAIN9_UNBOUNDED_ONLINE_RESULTS.md` | v1.1、10seed、suffix局所上限撤廃後のfilter比較 |
| S12 | `01_0813_five_bar_linkage/meeting_materials/FACTS_AND_PROVENANCE.md` | raw／canonical、衝突の扱い、pilot／held-outなどの来歴確認 |

## 報酬・初回発見

| ID | パス | 確認した役割 |
|---|---|---|
| S13 | `01_0821_five_bar_linkage_codex/experiments_ori5/analysis/final_ori5_results_ja.md` | 共通報酬による複数課題、初回成功と更新数、失敗条件 |
| S14 | `01_0821_five_bar_linkage_codex/docs/ori_codex6_zero_reject_design_ja.md` | ori2–6の報酬・衝突処理の変遷。MC Actor–CriticとPPOを区別 |
| S15 | `01_0821_five_bar_linkage_codex/experiments_ori6/analysis/zero_reject_reward_final/final_report_ja.md` | zero-reject、gate選抜とproductionの結果 |
| S16 | `01_0821_five_bar_linkage_codex/experiments_ori7/analysis/C7TC2A_2100/report_ja.md` | crossing終了を用いた報酬比較、不採用・未実行条件 |
| S17 | `01_0821_five_bar_linkage_codex/experiments_ori8/analysis/C8SP2R1_1000/report_ja.md` | scale報酬の選抜、初回発見とwaterの不安定性 |
| S18 | `01_0821_five_bar_linkage_codex/experiments_ori10/analysis/C10NS1_final/report_ja.md` | C10のoptimizer・batch等の選抜と結果 |
| S19 | `01_0821_five_bar_linkage_codex/experiments_ori10/analysis/C10NS1_vs_C10R2_random/report_ja.md` | RL3seed／random10seed、共通打切りでの初回発見比較 |
| S20 | `00_0822_all/result/experiments/20260830_torus_spectral_screen_v1/summary.md` | spectral重みのpaired screen。選抜済み設定の独立検証とは別 |
| S21 | `00_0829_all/README.md`、`00_0829_all/main.py`、`00_0829_all/RL_main.py`、`00_0829_all/agent.py` | 初回成功で停止、成功batchは学習しない、更新前後のNNを固定して100rollout評価 |

## 汎化・今後の拡張

| ID | パス | 確認した役割 |
|---|---|---|
| S22 | `00_0822_all/MIXED_GOALS.md` | 目標D*を入力しD2／D3を混合するActor–Critic設計 |
| S23 | `00_0822_all/result/experiments/mixed_advantage_study_20260911.md` | advantage補正・critic追加学習でもD2探索が回復しない比較 |
| S24 | `00_0822_all/SUPERVISED_FOLDING.md`、`00_0822_all/supervised_folding.py` | 既知目標で教師を作り、studentには完全な目標座標を渡さない学習 |
| S25 | `00_0822_all/result/experiments/generalization_5h_20260914/RESULTS_JA.md` | target教師ありの少量データ転移、忘却とreplay、構造転移の限定的結果 |
| S26 | `00_0822_all/result/experiments/discovery_transfer_5h_20260914/RESULTS_JA.md` | 未知課題の教師なし初回発見。frozen NN、更新回数、失敗例、探索的な実験設計 |
| S27 | `00_0822_all/result/experiments/discovery_transfer_5h_20260914/EXTENSIONS_JA.md` | 物理場をモードへ射影する拡張。変更点を不変性だけの効果としない |
| S28 | `00_0822_all/discovery_transfer.py`、`00_0822_all/graph_transfer.py`、`00_0822_all/transfer_study.py` | 転移実験の入口・設計。教師あり転移と未知課題探索を分ける |
| S29 | `00_0822_all/env.py`、`00_0822_all/mechanics.py`、`00_0822_all/main.py` | 力学拘束と許容誤差。数値上のstrain許容を弾性変形経路の実装とは呼ばない |

## フォルダーの扱い

- 初期linkage／torus／tri-panel／yoshi系列は試作群。フォルダー名だけから実験実施順・成否を決めない。
- `00_0804_codex2` → `01_0804_five_bar_linkage`／`01_0813_five_bar_linkage` は多様解探索の検討を対応付ける資料群。単純な後継版の一直線とは限らない。
- `01_0821_five_bar_linkage_codex`、`00_0822_all`、`00_0829_all` は初回発見・汎化の各実験を個別に参照する。
- `02_4unit_torus_CMAES` は初期最適化の関連試作として所在を確認。独立の性能主張には使わない。
- `00_0813_codex/a.tex` はGP・確率の説明資料で、探索性能の実証資料としては使わない。
- `output`、`tmp`、`experiments_ori6`、`.vscode` 等を独立した研究段階とは扱わない。
- **本人指定により `00_0930_all` は整理の根拠から除外。`shell_analysis` も対象外。**

OneDriveのオンライン専用だった転移報告S25–S27は、ローカルで内容を読み取れた範囲を反映した。今後内容が更新された場合、古い要約と混同せず、報告名・実験条件・更新理由を残して共有文書を改訂する。
