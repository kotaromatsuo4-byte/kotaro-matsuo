# 折り畳み経路探索の研究ノート

**普段は [current.md](research/current.md) だけ読んで、そこへ書きます。** ChatGPT・Codex共通の議論と作業の場所です。

```text
research/
├── current.md       今の問い・議論・結果・次にすること
├── overview.md      研究全体の流れ
└── history/
    ├── 2026-05.md   日付が月まで分かる過去の記録
    ├── …            版比較・実験条件・判断理由を詳しく保存
    └── 2026-10-26.md
```

- [今の議論と次の作業](research/current.md)
- [全体の流れと履歴の目次](research/overview.md)
- [共同編集・アーカイブ・Gitのルール](AGENTS.md)

研究史はfolding_pathが対象で、00_0930_allとshell_analysisは除外します。Gitではリポジトリ全体のPython・Markdown・TeXとルート.gitignoreを保存し、実験データ・モデル・画像等はローカルに保持します。

議論の区切りにcurrentを日付別historyへ保存し、未解決事項はcurrentに引き継ぎます。ノートの更新で別会話が自動起動する仕組みは設定していません。
