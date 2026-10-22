# water training-only episode 10000 非authority補遺

> **重要:** これは正式な最終結果ではない。final held-out 64-rolloutと`summary.json`が安全期限までに完了しなかったため、正式authorityはatomic checkpoint **9728** のままである。

正式報告: [final_report_ja.md](final_report_ja.md)

## training-only 10000 の再計算値

- last1024 (ep8977–10000): mean return 8.2554, terminal F median 0.00099383, spread error median 4.06949e-05, strict 516/1024 (50.4%)。
- last512 (ep9489–10000): strict 265/512 (51.8%)。
- periodic stochastic ep10000: n=16, success 62.5%, self-crossing 0.0%, mean return 9.7392。これはfinal held-outではない。
- periodic deterministic ep10000: success 0%, termination=timeout, F=2.08953, eD=0.115619。1 trajectoryであり成功率推定ではない。

## authorityを9728に維持する理由

policy.pt、training_history.csv、update_history.csvは10000まで到達したが、final held-out評価中にdeadline stopとなりsummary.jsonが存在しない。したがってcomplete final-policy setの条件を満たさず、checkpoint.pt=9728を正式authorityとして保守的に採用する。10000値は学習trajectoryの補足証拠に限定する。

## 整合性

- supplement integrity: PASS
- input hash set SHA-256: `df32422d6e7e78d1c4614024cd11f399cf3c5683b3ae152cf5bd208089ec2353`
- main artifact manifest SHA-256: `6af4321be9c67aae404ecd1ad2bddc89d523790d060aa454a3cacae58393fe11`
- raw training/update/evaluation CSV、policy/checkpoint、formal final summary、main report/metrics/manifest、supplement source/testをSHA-256でbindした。
