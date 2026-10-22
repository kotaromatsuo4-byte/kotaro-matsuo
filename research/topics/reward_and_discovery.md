# Actor–Critic：報酬・衝突処理・初回発見の評価

この系列は「未知終端を一つ見つける」ことが中心。[研究史04](../history/04_first_discovery.md) を先に読む。`ori_codexN` の番号だけで全条件が単純に置換されたと解釈しない。

## 改訂の問題意識

| 検討 | 問題 | 変更・比較 | 保存結果の解釈 |
|---|---|---|---|
| ori3/ori4から共通報酬ori5 | flatnessだけでは折り上がりの広がりを選べず、惜しい状態への信号も不足 | D*、near-miss、endpoint potential、固定milestone | fiveなどで成功。torusはdead zone、water D1は交差経路への集中が残る |
| ori6系 | spreadや近似平面の変化を、より連続的に評価したい | continuous spread-frontier、spectral subspace、rollback | 報酬だけが増えるbasinや、reject報酬の意味を吟味 |
| zero-reject | 衝突で毎回終了する・棄却に負報酬を与える影響 | 候補をlast-safe stateへ戻し、通常報酬0でstepを消費して継続 | torus選抜seedで成功。waterはnear-miss範囲を変更したfresh cohortで改善 |
| ori7 | 交差終端の報酬扱いを再検討 | no_fixed / historical penalty | torus・waterは正式gateで棄却。未起動・打切りを失敗と数えない |
| ori8 | spectralの形・尺度の影響 | legacy / scale / phase_v2 | scaleを選択。ただし初回発見と長期安定性は一致しない |
| C10 | 非plane-swap課題でoptimizerを調整 | five D2/D3、water √2のarm比較 | 複数seedで初回発見。random対照と共通capで比較 |
| 00_0822_all / 00_0829_all | 初回発見の証拠を整理し、簡潔な実装へ | zero-reject、joint potential差、固定horizonのloss正規化、checkpoint評価 | 成功を生んだNNまで評価。成功後の追加学習と切り離す |

出典：S13–S21。通常のzero-rejectの即時報酬0と、horizonでのtimeout終端成分は区別する。最終候補の更新は **episode-separated Monte-Carlo Actor–Critic**。PPOパラメータやPPOコードが存在しても、実行設定がactor_criticの結果をPPOと呼ばない。[S14]

## 比較で分けるべき三つの成果

1. **初回発見の高速化**：初めてstrict条件を満たすまでの累積proposal、episode、更新回数。棄却提案も数える。未発見は共通上限で打ち切る。
2. **方策が有望な領域を探すようになったか**：初回成功前のcheckpointを固定し、学習とは別の乱数で評価。near-miss到達率、Q*など。
3. **発見後の安定折り畳み**：追加学習後の固定方策の成功率。現在の主対象ではなく、過去の検証成果として保存する。

更新0回での初回成功は初期NN分布の寄与を含む。「学習が速めた」証拠と分ける。frozenとfinetuneが最初の更新前に同じ経路で成功した場合も二重に数えない。

## 代表的な保存結果と限界

- ori5：five D2は585 episode・233,824 step・73更新で初回成功。five D3は3 episode・1,033 step・更新0。water √2はrandomとの差が弱く、校正wall比較ではRLが遅い。torus D2とwater D1は未発見。[S13]
- zero-reject：torus選抜seedの最終1024 episode成功率89.4%、water fallback選抜seed51.2%。4096以降は選抜1 seedであり、複数seed安定性の証拠ではない。[S15]
- C10対exact uniform：共通proposal capでrandom/RL RMST比はfive D2=4.780、D3=11.600、water √2=0.519。RL 3 seed、random 10 seedの小標本。arm選択と最終評価の独立性にも注意し、全構造への優位としない。[S18–S19]
- torus spectral screen：α=0.05/0.25/0.5の成功は4/5、4/5、2/5。打切り込み平均stepは676,236.4、715,958.4、1,311,394.0。成功したseedだけならα=0.5が速く見えるため、条件付き中央値だけで選ばない。α=0.05はこのscreenからの候補であり独立最終testではない。[S20]

## 固定checkpoint評価の意味

`00_0829_all` のnear-missは、1 rollout中に一度でもF≤0.1とD誤差≤0.05を同時に満たした割合。Q*は同一stepのqF・qDの積を経路内で最大化する。異なるstepの最良qFと最良qDを掛けない。

READMEの実装確認（2026-09-09）では、five D3 seed9102が9 episode・3,501 step・1更新で初回成功。初期NNと更新後NN各100 rolloutのnear-missは0/100→22/100、Q*中央値は0.0462→0.2645。ただし一つのrunの確認であり、全taskの検証ではない。[S21]

## 今後の比較設計に残すもの

同じ入力情報・幾何・幅・horizon・衝突遷移・成功判定のbaselineを用意する。mechanics-step/proposalとwall timeを両方記録し、学習費用を無視しない。成功例だけの平均、選抜seedだけの一般化、異なる版や報酬値の単純比較を避ける。
