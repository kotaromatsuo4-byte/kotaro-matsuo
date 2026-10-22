# main9 特徴量アブレーション

## 結論

`last_nonflat_context` を除いて入力を文字どおりの `(q,v,K)` にしても、
6 cell 発見前の学習効率は改善しなかった。canonical T6 時点の独立 macro 数は
seed 800--802 で `11 / 14 / 17` である。この範囲に対応する 12, 16 macro では、
13 次元 `(q,v,K)` と 19 次元 `(q,context,v,K)` の優劣がモデルと学習点によって
反転し、どちらの精度も prevalence baseline に近い。

320 macro では context を除いた group-kNN の closed-set 精度が 0.484 となり、
context ありの 0.464 より 0.021 高かった。しかし未知 cell を隠す LOCO 評価では、
context なしの厳格 filter は未知 cell を 91.7% しか保持できず、安全基準 95% を
満たさなかった。context ありは 95.8% を保持し、既知 duplicate を 5.7% 棄却した。
したがって context は単純な closed-set 正解率を少し下げる場合があっても、
未知 basin を誤棄却しないための情報として有用である。

特徴量の次元削減だけでは、T6 前に有用な選択圧を発生させるには不十分である。

## 評価条件

- 保存済み `result/main9_random_corpus_b20000/seed_800..802/training_data.npz`
  のみを使用し、物理計算と label は変更していない。
- 保存入力の配置を `q:6, context:6, v:6, logK:1` と検査した。
- 比較入力は `(q,v,K)` 13 次元と `(q,context,v,K)` 19 次元。
- 各 seed の末尾 64 macro を全 learning-curve 点で共通の chronological holdout
  とした。同一 macro の hindsight 行は train/test を跨がず、総重みも1である。
- Random Forest、ExtraTrees、group-kNN、feature-blind prevalence を比較した。
  forest は16 trees、seedは固定した。
- 分類曲線は 8, 12, 16, 32, 64, 128, 320 macro。LOCO は best model である
  group-kNN の 32, 320 macro に限定した。
- filter 採用基準は precision >= 0.90、LOCO unknown retention >= 0.95、
  duplicate rejection >= 0.05、かつ3 seed中半数以上で基準を満たすこと。

## T6 付近の分類結果

括弧内は macro-F1 である。

| model | macro | `(q,v,K)` | `(q,context,v,K)` | prevalence accuracy |
|---|---:|---:|---:|---:|
| Random Forest | 12 | 0.219 (0.208) | 0.212 (0.186) | 0.199 |
| ExtraTrees | 12 | 0.278 (0.252) | 0.311 (0.265) | 0.199 |
| group-kNN | 12 | 0.291 (0.240) | 0.291 (0.226) | 0.199 |
| Random Forest | 16 | 0.210 (0.167) | 0.234 (0.179) | 0.222 |
| ExtraTrees | 16 | 0.222 (0.212) | 0.210 (0.180) | 0.222 |
| group-kNN | 16 | 0.240 (0.166) | 0.222 (0.158) | 0.222 |

12 macro では context 除去による accuracy 差が RF `+0.007`、ExtraTrees
`-0.033`、kNN `0.000`、16 macro では `-0.024 / +0.012 / +0.018` だった。
一貫した改善ではなく、少数 macro によるばらつきの範囲である。

## 長時間側の分類結果

| model | macro | `(q,v,K)` | `(q,context,v,K)` | accuracy差 | F1差 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 320 | 0.401 | 0.411 | -0.010 | -0.017 |
| ExtraTrees | 320 | 0.417 | 0.422 | -0.005 | +0.001 |
| group-kNN | 320 | 0.484 | 0.464 | +0.021 | +0.026 |

context 除去が明確に有利なのは group-kNN の closed-set 指標だけで、改善幅も
約2 pointである。RF/ExtraTreesではわずかに悪化した。

## group-kNN filter の LOCO 結果

| feature | macro | threshold | precision | duplicate rejection | LOCO retention | 採用基準 |
|---|---:|---|---:|---:|---:|---|
| `(q,v,K)` | 32 | strict | -- | 0.000 | 1.000 | 不合格 |
| `(q,context,v,K)` | 32 | strict | -- | 0.000 | 1.000 | 不合格 |
| `(q,v,K)` | 32 | loose | 1.000 | 0.026 | 0.984 | 不合格 |
| `(q,context,v,K)` | 32 | loose | 1.000 | 0.036 | 0.984 | 不合格 |
| `(q,v,K)` | 320 | strict | 1.000 | 0.052 | 0.917 | 不合格 |
| `(q,context,v,K)` | 320 | strict | 1.000 | 0.057 | 0.958 | 合格 (2/3 seed) |
| `(q,v,K)` | 320 | loose | 1.000 | 0.234 | 0.797 | 不合格 |
| `(q,context,v,K)` | 320 | loose | 1.000 | 0.234 | 0.771 | 不合格 |

32 macro はすでに全 corpus seed の T6 より後であるが、厳格 filter は一件も
棄却しない。320 macro で初めて現行19次元kNNが採用基準を満たす。これは
「十分長く探索すれば既知 duplicate filter を学習できる」ことは示す一方、
「filter によって最初の6 cellを早める」ためには遅すぎる。

## 再現方法と成果物

```powershell
python main9_feature_ablation.py result/main9_random_corpus_b20000 `
  --training-sizes 8,12,16,32,64,128,320 --holdout-macros 64 `
  --model-kinds random_forest extra_trees knn prevalence `
  --n-estimators 16 --workers 3 --skip-loco `
  --output-directory result/main9_feature_ablation_classification

python main9_feature_ablation.py result/main9_random_corpus_b20000 `
  --training-sizes 32,320 --holdout-macros 64 --model-kinds knn `
  --n-estimators 16 --workers 3 `
  --output-directory result/main9_feature_ablation_knn_loco
```

- 実装: `main9_feature_ablation.py`
- 全分類結果: `result/main9_feature_ablation_classification/`
- targeted LOCO結果: `result/main9_feature_ablation_knn_loco/`
- `result/main9_feature_ablation_smoke/` は実行確認用であり、結論には使用していない。

