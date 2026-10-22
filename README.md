# 折り畳み経路探索：研究共有ノート

ChatGPTでの研究相談と、ローカルCodexでの実装・検証をつなぐ軽量な記録です。研究コード・実験データはローカルに保持します。

最初に [研究の全体像](research/overview.md) を読んでください。対象は `folding_path`、`00_0930_all` は除外します。

| 読みたい内容 | 入口 |
|---|---|
| 何を問題にし、なぜ方針を変えたか | [研究史](research/history/01_early_rl.md) ／ [判断記録](research/decisions.md) |
| 実験から何が言えるか | [実験一覧](research/experiments.md) ／ [資料一覧](research/sources.md) |
| 過去のCodex会話にある意図 | [会話索引](research/conversations.md) |
| 次に何を決めるか | [未解決事項](research/questions.md) |
| ChatGPTへの引き継ぎ | [CHATGPT.md](CHATGPT.md) |
| Codexの作業報告 | [CODEX.md](CODEX.md) |
| 更新・共有のルール | [AGENTS.md](AGENTS.md) |

研究史はmain上の文書に時系列で蓄積し、Gitのコミットで改訂履歴を残します。並行する別案を試す場合にテーマ別branchを使います。過去の出来事の日付と、今回文書を整理した日付は分けます。

ノートの更新だけで別のChatGPT／Codex会話が自動起動するわけではありません。各作業の開始時に最新ノートを読み、終了時に判断・結果を保存する運用です。
