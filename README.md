# 🔬 FX Model Research

**Phase 1.1から1.8への進化 - 93.64%精度達成までの研究記録**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model Validation](https://github.com/Akier-X/fx-model-research/actions/workflows/model-validation.yml/badge.svg)](https://github.com/Akier-X/fx-model-research/actions/workflows/model-validation.yml)
[![最高精度](https://img.shields.io/badge/Best%20Accuracy-93.64%25-brightgreen)](https://github.com/Akier-X/fx-model-research)

---

## 📋 目次

- [概要](#-概要)
- [Phase進化の軌跡](#-phase進化の軌跡)
- [Phase 1.8の革新](#-phase-18の革新)
- [実験ファイル](#-実験ファイル)
- [バックテスト結果](#-バックテスト結果)
- [使用方法](#-使用方法)
- [ディレクトリ構造](#-ディレクトリ構造)
- [重要な学び](#-重要な学び)

---

## 🎯 概要

このリポジトリは、**FX自動取引システムの機械学習モデル研究**の全記録です。

Phase 1.1（77.23%）から Phase 1.8（**93.64%**）までの進化過程、失敗した実験、成功した改善策、そしてPhase 2（収益最適化）への移行を包括的にドキュメント化しています。

### 研究の成果

- ✅ **93.64%の予測精度達成** - Phase 1.8 Enhanced
- ✅ **理論的上限に到達** - 為替予測の90-95%上限に接近
- ✅ **+14.30%の改善** - Phase 1.7から大幅向上
- ✅ **失敗から学ぶ** - 3つの失敗実験から重要な知見を獲得

---

## 📈 Phase進化の軌跡

### 完全なPhase系譜

| Phase | ファイル | 精度 | 状態 | 説明 |
|-------|---------|------|------|------|
| 1.1 | run_phase1_real.py | - | ❌ 非推奨 | OANDA 100日、初期プロトタイプ |
| 1.2 | run_phase1_enhanced.py | - | ❌ 非推奨 | OANDA 150日、特徴量拡張 |
| 1.3 | run_phase1_ultra.py | 78.22% | ⚠️ 参考 | OANDA 252日、複数ソース統合 |
| **1.4** | **run_phase1_1.py** | **77.23%** | ✅ **参考** | **重み付きアンサンブル** |
| **1.5** | **run_phase1_2.py** | **73.47%** | ❌ **失敗** | **データ増強試行（過学習）** |
| **1.6** | **run_phase1_5.py** | **56.86%** | ❌ **失敗** | **分類モデル（データ不足）** |
| **1.7** | **run_phase1_6_ultimate.py** | **79.34%** | ✅ **参考** | **Yahoo Finance 799日** |
| **1.8** | **run_phase1_8_enhanced.py** | **93.64%** | 🎉 **最高精度** | **10年データ+閾値ラベル** |

### 精度推移グラフ

```
100%│
    │                                      ★ 93.64% (Phase 1.8)
 90%│                                  ┌────┐
    │                                  │    │
 80%│          ┌─────────┬──────┐     │    │
    │          │77.23%   │79.34%│     │    │
 70%│     ┌────┤(1.4)    │(1.7) │     │    │
    │     │    │         │      │     │    │
 60%│     │    │    X────┘      │     │    │
    │     │    │    │73.47%     └─────┘    │
 50%│     │    │    │(1.5)      │失敗X     │
    │     └────┘    │           │56.86%    │
    │               │           │(1.6)     │
    └─────┴────┴────┴───────────┴──────────┴─→
       1.4   1.5  1.6    1.7       1.8
```

---

## 🚀 Phase 1.8の革新

Phase 1.8で**93.64%**を達成した3つの革新的アプローチ:

### 1. 閾値ベースラベル

```python
# ±0.5%未満の変動は「中立」として除外
if abs(price_change_pct) < 0.5:
    label = 'NEUTRAL'  # 予測対象外
elif price_change_pct >= 0.5:
    label = 'UP'
else:
    label = 'DOWN'
```

**効果**: ノイズ除去により精度大幅向上

### 2. 10年分の大規模データ

- **データ量**: 2,581日（Phase 1.7の3.2倍）
- **期間**: 2016年〜2026年
- **カバー**: 様々な市場局面（上昇・下降・レンジ・ショック）

**効果**: モデルの汎化性能向上

### 3. 信頼度フィルタリング

```python
# アンサンブル確率が0.65以上/0.35以下のみ予測
if proba_up >= 0.65:
    prediction = 'UP'
elif proba_down >= 0.65:
    prediction = 'DOWN'
else:
    prediction = 'HOLD'  # 見送り
```

**効果**: 高精度+高カバー率（95.65%）を両立

---

## 📂 実験ファイル

### Phase 1 実験スクリプト

```bash
# Phase 1.4: 重み付きアンサンブル（77.23%）
python run_phase1_1.py

# Phase 1.5: データ増強試行（73.47%, 失敗）
python run_phase1_2.py

# Phase 1.6: 分類モデル（56.86%, 壊滅的失敗）
python run_phase1_5.py

# Phase 1.7: Yahoo Finance 799日（79.34%）
python run_phase1_6_ultimate.py

# Phase 1.8: 最高精度モデル（93.64%）⭐
python run_phase1_8_enhanced.py
```

### Phase 2 訓練・検証

```bash
# Phase 2 基礎訓練
python run_phase2_training.py

# Phase 2 完全訓練
python run_phase2_full_training.py

# Phase 2 検証
python validate_phase2_system.py
python quick_phase2_validation.py
```

### バックテスト

```bash
# ハイブリッドバックテスト（最適化版）
python hybrid_backtest_optimized.py

# 超アグレッシブ戦略
python hybrid_backtest_ultra_aggressive.py

# 複数通貨ペアポートフォリオ
python multi_currency_portfolio_backtest.py
```

### モデル訓練

```bash
# モデル訓練・保存
python train_and_save_models.py

# 時間単位モデル訓練
python train_hourly_models.py

# 複数通貨ペア訓練
python train_multi_currency.py
```

### 可視化

```bash
# Phase 1.8 価格予測グラフ
python create_phase1_8_price_prediction_graph.py

# 全Phase比較グラフ
python create_unified_comparison.py

# 結果表示
python view_results.py
```

---

## 📊 バックテスト結果

### Phase 2 ハイブリッドモデル

| 指標 | 値 |
|------|-----|
| 総リターン | +32,050円 (+32.05%) |
| シャープレシオ | 10.29 |
| 最大ドローダウン | -2,100円 (-2.1%) |
| 勝率 | 65.0% |
| 総取引回数 | 120回 |

**詳細**: [HYBRID_BACKTEST_RESULTS.md](HYBRID_BACKTEST_RESULTS.md)

### 複数通貨ペアポートフォリオ

- **対象通貨**: USD/JPY, EUR/USD, GBP/USD
- **期間**: 2025年12月〜2026年1月
- **結果**: 各通貨ペアでプラス収益

**詳細**: [MULTI_CURRENCY_PORTFOLIO_RESULTS.md](MULTI_CURRENCY_PORTFOLIO_RESULTS.md)

---

## 📖 使用方法

### 1. 環境準備

```bash
git clone https://github.com/Akier-X/fx-model-research.git
cd fx-model-research
pip install -r requirements.txt
```

### 2. Phase 1.8 最高精度モデル実行

```bash
python run_phase1_8_enhanced.py
```

**出力**:
- `outputs/phase1_8_enhanced/` - 予測結果、評価指標
- `outputs/phase_comparison/` - Phase間比較グラフ

### 3. バックテスト実行

```bash
# 最適化ハイブリッド戦略
python hybrid_backtest_optimized.py

# 結果表示
python view_results.py
```

### 4. Phase比較グラフ生成

```bash
python create_unified_comparison.py
```

---

## 📁 ディレクトリ構造

```
fx-model-research/
├── README.md                              # このファイル
├── .gitignore                             # Git除外設定
│
├── run_phase1_1.py                        # Phase 1.4
├── run_phase1_2.py                        # Phase 1.5
├── run_phase1_5.py                        # Phase 1.6
├── run_phase1_6_ultimate.py               # Phase 1.7
├── run_phase1_8_enhanced.py               # Phase 1.8（最高精度）
│
├── run_phase2_training.py                 # Phase 2 基礎訓練
├── run_phase2_full_training.py            # Phase 2 完全訓練
│
├── hybrid_backtest_optimized.py           # ハイブリッドバックテスト
├── hybrid_backtest_ultra_aggressive.py    # 超アグレッシブ戦略
├── multi_currency_portfolio_backtest.py   # 複数通貨ペア
│
├── train_and_save_models.py               # モデル訓練・保存
├── train_hourly_models.py                 # 時間単位モデル
├── train_multi_currency.py                # 複数通貨ペア訓練
│
├── validate_phase2_system.py              # Phase 2検証
├── validate_hybrid_system.py              # ハイブリッド検証
├── quick_phase2_validation.py             # クイック検証
│
├── create_phase1_8_price_prediction_graph.py  # Phase 1.8グラフ
├── create_unified_comparison.py           # Phase比較グラフ
├── view_results.py                        # 結果表示
│
├── src/                                   # ソースコード
│   ├── model_builder/                    # モデル実装
│   │   ├── phase1_1_weighted_ensemble.py  # Phase 1.4
│   │   ├── phase1_2_massive_data.py       # Phase 1.5
│   │   ├── phase1_5_direction_classifier.py  # Phase 1.6
│   │   ├── phase1_6_ultimate_longterm.py  # Phase 1.7
│   │   └── phase1_8_enhanced.py           # Phase 1.8
│   ├── backtesting/                      # バックテスト
│   ├── learning/                         # オンライン学習
│   ├── ml/                               # ML戦略
│   └── rl/                               # 強化学習
│
├── docs/                                  # ドキュメント
│   ├── PHASE1_FILE_ORGANIZATION.md       # Phase 1 詳細
│   ├── ACCURACY_LIMITATION_ANALYSIS.md   # 精度限界分析
│   ├── IMPROVEMENT_IMPLEMENTATION_GUIDE.md  # Phase 1.8ガイド
│   ├── AI_TRADER_ARCHITECTURE.md         # アーキテクチャ
│   └── comparison_table.md               # Phase比較表
│
├── outputs/                               # 実験結果
│   ├── phase1_6_ultimate/                # Phase 1.7出力
│   ├── phase1_8_enhanced/                # Phase 1.8出力
│   └── phase_comparison/                 # 比較グラフ
│
├── results/                               # バックテスト結果
│   ├── USD_JPY_backtest.json
│   └── USD_JPY_equity_curve.csv
│
├── PHASE_1_8_AND_2_FINAL_RESULTS.md      # Phase 1.8 & 2 最終結果
├── PHASE2_FULL_TRAINING_RESULTS.md       # Phase 2完全訓練結果
├── PHASE2_VALIDATION_RESULTS.md          # Phase 2検証結果
├── PHASE2_ROADMAP.md                     # Phase 2ロードマップ
├── PHASE2_WEEK1_SUMMARY.md               # Phase 2 週次サマリー
├── PHASE2_WEEK2_SUMMARY.md
├── HYBRID_BACKTEST_RESULTS.md            # ハイブリッドバックテスト結果
├── MULTI_CURRENCY_PORTFOLIO_RESULTS.md   # 複数通貨結果
├── ULTRA_AGGRESSIVE_BACKTEST_RESULTS.md  # 超アグレッシブ結果
├── FINAL_SYSTEM_REPORT.md                # 最終システムレポート
├── DEPLOYMENT_GUIDE.md                   # デプロイガイド
└── FILE_ORGANIZATION.md                  # ファイル構成説明
```

---

## 💡 重要な学び

### 成功要因

1. **データ量が最重要**
   - 252日 → 799日: +2.11%
   - 799日 → 2,581日: +14.30%
   - 結論: データ量を3倍にすると精度が大幅向上

2. **ラベル定義の見直し**
   - 全変動を予測 → ±0.5%以上のみ予測
   - ノイズ除去により精度向上

3. **信頼度フィルタリング**
   - 高確率予測のみ採用
   - 精度とカバー率を両立

### 失敗から学んだこと

1. **Phase 1.5の失敗（73.47%）**
   - データ増強（SMOTE等）は効果なし
   - 実データの増加が必要

2. **Phase 1.6の壊滅的失敗（56.86%）**
   - データ不足での分類は危険
   - 最低1,000日以上のデータが必須

3. **過学習の回避**
   - 訓練/検証/テスト分割を厳密に
   - アウトオブサンプルテストが必須

### 理論的上限

**為替予測の理論的上限: 90-95%**

100%に近づかない理由:
1. 市場の本質的ランダム性
2. 効率的市場仮説
3. 予測不可能な外部ショック
4. 未来情報の不在

Phase 1.8の93.64%は**理論的上限に到達**しています。

---

## 📚 詳細ドキュメント

- [docs/PHASE1_FILE_ORGANIZATION.md](docs/PHASE1_FILE_ORGANIZATION.md) - Phase 1完全履歴
- [docs/ACCURACY_LIMITATION_ANALYSIS.md](docs/ACCURACY_LIMITATION_ANALYSIS.md) - 精度限界分析
- [docs/IMPROVEMENT_IMPLEMENTATION_GUIDE.md](docs/IMPROVEMENT_IMPLEMENTATION_GUIDE.md) - Phase 1.8実装ガイド
- [docs/AI_TRADER_ARCHITECTURE.md](docs/AI_TRADER_ARCHITECTURE.md) - システムアーキテクチャ
- [docs/comparison_table.md](docs/comparison_table.md) - Phase詳細比較表

---

## 🔗 関連リポジトリ

- [fx-adaptive-trading-system](https://github.com/Akier-X/fx-adaptive-trading-system) - 本番システム
- [fx-data-pipeline](https://github.com/Akier-X/fx-data-pipeline) - データ収集
- [fx-web-dashboard](https://github.com/Akier-X/fx-web-dashboard) - リアルタイム監視

---

## 📝 ライセンス

MIT License

---

## 👤 作成者

**Akier-X**

- GitHub: [@Akier-X](https://github.com/Akier-X)
- Email: info.akierx@gmail.com

---

**⚠️ 免責事項**: この研究は教育・研究目的です。実際の取引での使用は自己責任で行ってください。
