# 警告サインの変更

## ❌ 従来の間違ったアプローチ
```python
if drawdown < -20%:
    stop_trading()  # ただ停止 → 何も学ばない
if consecutive_losses >= 3:
    stop_trading()  # ただ停止 → 何も学ばない
```

## ✅ 新しい正しいアプローチ
```python
if drawdown < -15%:
    # 緊急学習トリガー！
    retrain_models(
        emergency=True,
        focus_on_failures=True,
        failure_weight=3.0  # 失敗に3倍の重み
    )
    # → 失敗から学び、改善する

if consecutive_losses >= 5:
    # パターン学習トリガー
    analyze_loss_pattern()
    retrain_with_new_insights()
    # → 連敗の原因を学習
```

## 学習トリガー一覧

| 条件 | 従来 | 新システム |
|------|------|-----------|
| ドローダウン-15% | 停止 | 🧠 緊急学習（失敗に3倍重み） |
| 5連敗 | 停止 | 🧠 パターン分析＋再訓練 |
| 勝率30%以下 | 停止 | 🧠 モデル根本的見直し |
| 100トレード | - | 🧠 定期再訓練 |
| パフォーマンス悪化 | - | 🧠 学習率2倍で集中学習 |
