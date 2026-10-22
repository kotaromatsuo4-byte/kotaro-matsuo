# Codex7 21時 最終レポート

- 生成時刻（UTC）: 2026-08-27T11:46:09.362701+00:00
- manifest fingerprint: `14af9b73e5c1c8726bab6f39217a1ac2a430d26053299c4ff552df1035b79f47`

## 解釈上の注意

atomic endpoint が4096未満のrunはearly evidenceであり、正式なarm比較とは解釈しません。
未起動runは右打切りとして保持し、成功・失敗のどちらにも数えません。
crossing terminal episodeはstrict/jointの正の証拠から除外しています。

jointはnoncrossingかつ terminal F≤0.1 かつ spread error≤0.1 と定義しています。

cohort待機中に4096を超えたscreened rowはpost_gate_overrun_descriptiveとして保存しますが、primary/formal集計とarm選択比較から除外します。

## Task / arm / seed

| task | arm | seed | launched | endpoint | phase | integrity | selected | reward(last1024) | F(last1024) | spread err(last1024) | strict | joint | crossing rate | censor |
|---|---|---:|:---:|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| torus2 | no_fixed | 7101 | yes | 4608 | formal_exact_4096_available | verified | no | -0.976238 | 1.0073 | 0.407536 | 0 | 0 | 0 | rejected_at_4096_arm_gate |
| torus2 | no_fixed | 7102 | yes | 5120 | formal_exact_4096_available | verified | no | 1.46452 | 0.617324 | 0.0466604 | 0 | 0 | 0.878906 | rejected_at_4096_arm_gate |
| torus2 | no_fixed | 7103 | yes | 4608 | formal_exact_4096_available | verified | no | 1.41463 | 0.63477 | 0.0518783 | 0 | 0 | 0.893555 | rejected_at_4096_arm_gate |
| torus2 | historical_1p25 | 7101 | yes | 4096 | formal_exact_4096_available | verified | no | 0.1437 | 0.758311 | 0.0615953 | 0 | 0 | 0.240234 | rejected_at_4096_arm_gate |
| torus2 | historical_1p25 | 7102 | yes | 4096 | formal_exact_4096_available | verified | no | 1.01305 | 0.281782 | 0.0214997 | 0 | 65 | 0.276367 | rejected_at_4096_arm_gate |
| torus2 | historical_1p25 | 7103 | yes | 4096 | formal_exact_4096_available | verified | no | 0.91465 | 0.347306 | 0.0227605 | 0 | 3 | 0.22168 | rejected_at_4096_arm_gate |
| water1 | no_fixed | 7101 | yes | 4608 | formal_exact_4096_available | verified | no | 0.340986 | 1.63979 | 0.017491 | 0 | 0 | 0.999023 | rejected_at_4096_arm_gate |
| water1 | no_fixed | 7102 | yes | 10000 | formal_exact_4096_available | verified | no | 2.06862 | 0.292996 | 0.00436648 | 4 | 4 | 0.988281 | natural_completion_10000 |
| water1 | no_fixed | 7103 | yes | 10000 | formal_exact_4096_available | verified | no | 0.321755 | 1.66009 | 0.0225631 | 0 | 0 | 1 | natural_completion_10000 |
| water1 | historical_1p25 | 7101 | yes | 6656 | formal_exact_4096_available | verified | no | -0.904507 | 1.63999 | 0.0166298 | 0 | 0 | 1 | rejected_at_4096_arm_gate |
| water1 | historical_1p25 | 7102 | yes | 9728 | formal_exact_4096_available | verified | no | -0.935269 | 1.66349 | 0.0238556 | 0 | 0 | 1 | rejected_at_4096_arm_gate |
| water1 | historical_1p25 | 7103 | yes | 4096 | formal_exact_4096_available | verified | no | -0.922591 | 1.65602 | 0.0173698 | 0 | 0 | 0.993164 | rejected_at_4096_arm_gate |
| five_d2 | no_fixed | 7101 | yes | 512 | early_evidence_only | verified | no | -0.371185 | 0.398466 | 0.122043 | 0 | 1 | 0 | deadline_right_censored_before_4096 |
| five_d2 | no_fixed | 7102 | yes | 512 | early_evidence_only | verified | no | -0.177458 | 0.38865 | 0.0276823 | 1 | 6 | 0 | deadline_right_censored_before_4096 |
| five_d2 | no_fixed | 7103 | yes | 512 | early_evidence_only | verified | no | -0.237651 | 0.441204 | 0.0164227 | 0 | 1 | 0 | deadline_right_censored_before_4096 |
| five_d2 | historical_1p25 | 7101 | yes | 512 | early_evidence_only | verified | no | -0.371185 | 0.398466 | 0.122043 | 0 | 1 | 0 | deadline_right_censored_before_4096 |
| five_d2 | historical_1p25 | 7102 | yes | 512 | early_evidence_only | verified | no | -0.177458 | 0.38865 | 0.0276823 | 1 | 6 | 0 | deadline_right_censored_before_4096 |
| five_d2 | historical_1p25 | 7103 | yes | 512 | early_evidence_only | verified | no | -0.237651 | 0.441204 | 0.0164227 | 0 | 1 | 0 | deadline_right_censored_before_4096 |
| five_d3 | no_fixed | 7101 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| five_d3 | no_fixed | 7102 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| five_d3 | no_fixed | 7103 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| five_d3 | historical_1p25 | 7101 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| five_d3 | historical_1p25 | 7102 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| five_d3 | historical_1p25 | 7103 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| watersqrt2 | no_fixed | 7101 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| watersqrt2 | no_fixed | 7102 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| watersqrt2 | no_fixed | 7103 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| watersqrt2 | historical_1p25 | 7101 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| watersqrt2 | historical_1p25 | 7102 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |
| watersqrt2 | historical_1p25 | 7103 | no | 0 | unlaunched_censored | not_applicable_unlaunched | no |  |  |  |  |  |  | queued_at_training_deadline |

## Paired seed差（no_fixed − historical_1p25）

| task | seed | decision cap | observed common | phase | Δreward | ΔF | Δspread error | Δstrict rate | Δjoint rate | Δcrossing rate |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| torus2 | 7101 | 4096 | 4096 | formal_exact_4096 | -1.11994 | 0.248988 | 0.345941 | 0 | 0 | -0.240234 |
| torus2 | 7102 | 4096 | 4096 | formal_exact_4096 | 0.451475 | 0.335542 | 0.0251607 | 0 | -0.0634766 | 0.602539 |
| torus2 | 7103 | 4096 | 4096 | formal_exact_4096 | 0.499975 | 0.287464 | 0.0291178 | 0 | -0.00292969 | 0.671875 |
| water1 | 7101 | 4096 | 4608 | formal_exact_4096 | 1.24549 | -0.000192915 | 0.000861257 | 0 | 0 | -0.000976562 |
| water1 | 7102 | 4096 | 9728 | formal_exact_4096 | 3.00389 | -1.3705 | -0.0194891 | 0.00390625 | 0.00390625 | -0.0117188 |
| water1 | 7103 | 4096 | 4096 | formal_exact_4096 | 1.24435 | 0.00406698 | 0.00519332 | 0 | 0 | 0.00683594 |
| five_d2 | 7101 | 512 | 512 | early_evidence_only | 0 | 0 | 0 | 0 | 0 | 0 |
| five_d2 | 7102 | 512 | 512 | early_evidence_only | 0 | 0 | 0 | 0 | 0 | 0 |
| five_d2 | 7103 | 512 | 512 | early_evidence_only | 0 | 0 | 0 | 0 | 0 | 0 |
| five_d3 | 7101 | 0 | 0 | unavailable_censored |  |  |  |  |  |  |
| five_d3 | 7102 | 0 | 0 | unavailable_censored |  |  |  |  |  |  |
| five_d3 | 7103 | 0 | 0 | unavailable_censored |  |  |  |  |  |  |
| watersqrt2 | 7101 | 0 | 0 | unavailable_censored |  |  |  |  |  |  |
| watersqrt2 | 7102 | 0 | 0 | unavailable_censored |  |  |  |  |  |  |
| watersqrt2 | 7103 | 0 | 0 | unavailable_censored |  |  |  |  |  |  |

## Integrity

入力ファイル 202 件をSHA-256で固定しました。
報酬式・transition式・Actor–Critic設定・crossing reward分解・parent/spectral分解を各atomic historyで再検証しています。
