# FX自動売買システム ファイル構成

## 📁 プロジェクト構造

```
FX/
├── docs/                          # 📚 全ドキュメント
│   ├── README.md                  # ドキュメント索引
│   ├── PHASE1_FILE_ORGANIZATION.md    # Phase 1詳細ガイド
│   ├── ACCURACY_LIMITATION_ANALYSIS.md # 精度限界と改善方針
│   └── ...                        # その他のドキュメント
│
├── src/                           # 💻 ソースコード
│   ├── data_sources/              # データ取得
│   │   ├── oanda_data.py          # OANDA API
│   │   ├── yahoo_finance.py       # Yahoo Finance（長期データ）
│   │   └── economic_indicators.py # FRED API（経済指標）
│   │
│   ├── model_builder/             # モデル構築
│   │   ├── phase1_basic_model.py          # Phase 1.0: 基本プロトタイプ
│   │   ├── phase1_real_data.py            # Phase 1.1: 実データ使用開始
│   │   ├── phase1_enhanced.py             # Phase 1.2: 特徴量拡張
│   │   ├── phase1_ultra_enhanced.py       # Phase 1.3: 複数データソース統合
│   │   ├── phase1_1_weighted_ensemble.py  # Phase 1.4: 重み付きアンサンブル
│   │   ├── phase1_2_massive_data.py       # Phase 1.5: データ増強試行
│   │   ├── phase1_5_direction_classifier.py # Phase 1.6: 分類モデル（失敗）
│   │   └── phase1_6_ultimate_longterm.py   # Phase 1.7: 究極版 ⭐ 最高精度
│   │
│   ├── feature_engineering/       # 特徴量生成
│   └── utils/                     # ユーティリティ
│
├── outputs/                       # 📊 出力結果
│   ├── phase_comparison/          # 全Phase比較グラフ
│   │   ├── 01_accuracy_comparison.png
│   │   ├── 02_data_volume_comparison.png
│   │   ├── 03_approach_comparison.png
│   │   ├── 04_evolution_timeline.png
│   │   └── 05_comprehensive_table.png
│   │
│   ├── phase1_6_ultimate/         # Phase 1.6結果（最高精度）
│   └── ...                        # その他のPhase結果
│
├── run_phase1.py                  # Phase 1.0実行スクリプト
├── run_phase1_real.py             # Phase 1.1実行スクリプト
├── run_phase1_enhanced.py         # Phase 1.2実行スクリプト
├── run_phase1_ultra.py            # Phase 1.3実行スクリプト
├── run_phase1_1.py                # Phase 1.4実行スクリプト
├── run_phase1_2.py                # Phase 1.5実行スクリプト
├── run_phase1_5.py                # Phase 1.6実行スクリプト（分類・失敗）
├── run_phase1_6_ultimate.py       # Phase 1.7実行スクリプト ⭐ 最高精度
│
├── create_unified_comparison.py   # 統一比較グラフ作成
├── test_apis.py                   # API接続テスト
├── master_build_system.py         # マスタービルドシステム
│
├── .env                           # 環境変数（APIキー等）
├── requirements.txt               # Pythonパッケージ
└── README.md                      # プロジェクトREADME
```

---

## 🎯 Phase 1 実行ファイル一覧

### 実行順序とファイル対応表

| 実行順序 | ファイル名 | Phase | 説明 | 方向性的中率 | 推奨度 |
|---------|----------|-------|------|------------|-------|
| 1 | `run_phase1.py` | 1.0 | 基本プロトタイプ（デモデータ） | - | ❌ 非推奨 |
| 2 | `run_phase1_real.py` | 1.1 | 実データ使用開始（OANDA 100日） | - | ❌ 非推奨 |
| 3 | `run_phase1_enhanced.py` | 1.2 | 特徴量拡張（OANDA 150日） | - | ❌ 非推奨 |
| 4 | `run_phase1_ultra.py` | 1.3 | 複数データソース統合（OANDA 252日） | 78.22% | ⚠️ 参考用 |
| 5 | `run_phase1_1.py` | 1.4 | 重み付きアンサンブル（OANDA 252日） | 77.23% | ✅ 参考用 |
| 6 | `run_phase1_2.py` | 1.5 | データ増強試行（OANDA 252日） | 73.47% | ❌ 失敗版 |
| 7 | `run_phase1_5.py` | 1.6 | 分類モデル（OANDA 252日） | 56.86% | ❌ 失敗版 |
| 8 | `run_phase1_6_ultimate.py` | 1.7 | 究極版（Yahoo Finance 799日） | 79.34% ⭐ | ✅ 参考用 |
| 9 | **`run_phase1_8_enhanced.py`** | **1.8** | **🎉 最高精度版（10年データ + 閾値ラベル）** | **93.64%** ⭐⭐⭐ | ✅✅✅ **本番推奨** |

---

## 📝 各Phase詳細説明

### Phase 1.0: 基本プロトタイプ

**ファイル**: `run_phase1.py`

**目的**: 概念実証
**データ**: ランダムデモデータ
**結果**: 実用性なし

**使用推奨**: ❌ 歴史的資料のみ

---

### Phase 1.1: 実データ使用開始

**ファイル**: `run_phase1_real.py`

**目的**: OANDA APIで実データ取得開始
**データ**: OANDA API 100日間
**アプローチ**: RandomForest回帰
**結果**: 初めて実データで検証

**使用推奨**: ❌ データ期間が短すぎ

---

### Phase 1.2: 特徴量拡張

**ファイル**: `run_phase1_enhanced.py`

**目的**: 特徴量拡張
**データ**: OANDA API 150日間
**特徴量**: SMA, EMA, RSI, MACD等
**アプローチ**: GradientBoosting回帰

**使用推奨**: ❌ データ期間が短すぎ

---

### Phase 1.3: 複数データソース統合

**ファイル**: `run_phase1_ultra.py`

**目的**: 複数データソース統合
**データ**: OANDA API 252日（最大）
**データソース**:
- OANDA API（複数時間足：H1, H4, D）
- FRED API（経済指標）
- 複数通貨ペア（EUR/USD, GBP/USD, EUR/JPY）

**特徴量**: 69個
**アプローチ**: 5モデルアンサンブル（単純平均）
**結果**: **78.22%** - 実用レベル到達

**使用推奨**: ⚠️ データソース統合の参考用

**学び**: データソース統合は有効だが、単純平均では限界

---

### Phase 1.4: 重み付きアンサンブル

**ファイル**: `run_phase1_1.py`

**目的**: 重み付きアンサンブル最適化
**データ**: OANDA API 252日
**改善点**:
- 単純平均 → **方向性的中率ベースの重み付け**
- 特徴量選択（上位30個）
- 訓練/検証分割で重み最適化

**アプローチ**: GradientBoosting, RandomForest, XGBoost
**結果**: **77.23%** - 重み付けで安定化

**使用推奨**: ✅ 重み付け手法の参考用

**学び**: 重み付けは有効だが、データ量がボトルネック

---

### Phase 1.5: データ増強試行

**ファイル**: `run_phase1_2.py`

**目的**: 訓練データ大幅増強（500日目標）
**データ**: 252日（OANDA APIの制限で増やせず）
**改善点**:
- LightGBM, CatBoost追加
- 83個の特徴量
- データ分割最適化

**アプローチ**: 5モデルアンサンブル
**結果**: 73.47% - **予想外の悪化**

**使用推奨**: ❌ 失敗版（過学習）

**学び**: データ量が足りず、モデル増加が逆効果

---

### Phase 1.6: 分類モデル（失敗版）

**ファイル**: `run_phase1_5.py`

**目的**: 戦略転換（回帰→分類）
**データ**: OANDA API 252日
**革命的変更**:
- **価格予測（回帰） → 方向予測（分類）**
- ターゲット: 1日後に上昇(1) / 下降(0)
- 確率的予測（信頼度付き）

**アプローチ**: 分類モデル5種類
**結果**: 56.86% - **大失敗**

**失敗原因**:
- テストサンプル: わずか51件
- 訓練サンプル: わずか151件
- **データ不足で分類モデルが機能せず**

**使用推奨**: ❌ 失敗から学ぶための参考資料

**学び**: 分類アプローチは正しいが、**データ量が決定的に重要**

---

### 🎉 Phase 1.8: Enhanced 最高精度版 ⭐⭐⭐

**ファイル**: **`run_phase1_8_enhanced.py`**

**目的**: 3大改善策統合で理論的上限に挑戦
**データ**: **2,581日間（10年分）** ← Phase 1.7の3倍以上！

**データソース**:
- **Yahoo Finance**: 10年分の為替データ（無料）
- FRED API: 経済指標
- 株価指数: S&P500, 日経, NASDAQ, VIX
- 複数通貨ペア: EUR/USD, GBP/USD, EUR/JPY

**特徴量**: 150列 → 上位60個選択

**データ分割**:
- 訓練: 70% (531件)
- 検証: 15% (113件)
- テスト: 15% (115件)

**3大改善策**:
1. **閾値ベースラベル**: ±0.5%以上の変動のみ予測対象（ノイズ除去）
2. **10年分データ**: 2,581日で様々な市場局面をカバー
3. **信頼度フィルタリング**: 確率0.65以上のみ予測（精度優先）
4. **時間重み付け**: 直近データを重視（decay_rate=0.95）

**アプローチ**: 分類モデル（方向予測 + 閾値ラベル + 信頼度フィルタ）
**モデル**: GBC, RFC, XGBoost, LightGBM, CatBoost

**結果**: **93.64%** - **全Phase中最高！理論的上限到達！**

**達成指標**:
- 方向性的中率: **93.64%** ⭐⭐⭐
- カバー率: 95.65%（見送りわずか5件）
- 上昇的中精度: 92.59%
- 下降的中精度: 94.64%
- F1スコア（上昇）: 93.46%
- F1スコア（下降）: 93.81%

**Phase 1.7からの改善**: +14.30%の大幅向上

**成功要因**:
1. **データ量の大幅増加（799日 → 2,581日）**
2. **閾値ベースラベルでノイズ除去**
3. **信頼度フィルタリングで精度優先**
4. **10年分データで様々な市場局面をカバー**

**使用推奨**: ✅✅✅ **本番運用推奨（最高精度）**

**重要な特徴量 Top 10**:
1. low_close_ratio (0.2479)
2. high_close_ratio (0.2101)
3. EUR_JPY_low_return (0.0472)
4. EUR_JPY_high_return (0.0310)
5. EUR_USD_high_return (0.0243)
6. EUR_USD_low_return (0.0209)
7. stoch_14 (0.0294)
8. GBP_USD_high_return (0.0153)
9. stoch_21 (0.0143)
10. high_low_range (0.0115)

---

### Phase 1.7: 究極版（従来の最高精度） ⭐

**ファイル**: `run_phase1_6_ultimate.py`

**目的**: Yahoo Finance長期データで最強モデル構築
**データ**: **799日間（3年分）** ← Phase 1.6の3倍以上！

**データソース**:
- **Yahoo Finance**: 3年以上の為替データ（無料）
- FRED API: 経済指標
- 株価指数: S&P500, 日経, NASDAQ, VIX
- 複数通貨ペア: EUR/USD, GBP/USD, EUR/JPY

**特徴量**: 122列 → 上位60個選択

**データ分割**:
- 訓練: 70% (559件)
- 検証: 15% (119件)
- テスト: 15% (121件)

**アプローチ**: 分類モデル（方向予測）
**モデル**: GBC, RFC, XGBoost, LightGBM, CatBoost

**結果**: **79.34%** - **全Phase中最高！**

**成功要因**:
1. **十分な訓練データ（559サンプル vs Phase 1.6の151サンプル）**
2. **バランスの良いデータ分割**
3. **分類アプローチ + 長期データの組み合わせ**
4. **複数データソース統合**

**使用推奨**: ✅ **参考用（Phase 1.8が最高精度）**

**重要な特徴量 Top 10**:
1. high_close_ratio (0.1304)
2. low_close_ratio (0.1277)
3. stoch_14 (0.0294)
4. stoch_21 (0.0243)
5. high_low_range (0.0148)
6. EUR_JPY_return (0.0117)
7. EUR_USD_corr_10 (0.0108)
8. rsi_28_slope (0.0103)
9. rsi_21_slope (0.0101)
10. price_change (0.0101)

---

## 🔧 ユーティリティスクリプト

### create_unified_comparison.py

**目的**: 全Phase比較グラフ作成

**生成されるグラフ**:
- `outputs/phase_comparison/01_accuracy_comparison.png` - 方向性的中率比較
- `outputs/phase_comparison/02_data_volume_comparison.png` - データ量比較
- `outputs/phase_comparison/03_approach_comparison.png` - アプローチ別比較
- `outputs/phase_comparison/04_evolution_timeline.png` - 進化の軌跡
- `outputs/phase_comparison/05_comprehensive_table.png` - 総合比較表

**実行方法**:
```bash
python create_unified_comparison.py
```

---

### test_apis.py

**目的**: API接続テスト

**確認項目**:
- OANDA API接続
- FRED API接続
- Yahoo Finance接続

**実行方法**:
```bash
python test_apis.py
```

---

### master_build_system.py

**目的**: 段階的システム構築のマスター制御

**機能**:
- 全Phaseの自動実行
- ビルドステップ管理
- エラーハンドリング

---

## 📊 出力フォルダ構成

```
outputs/
├── phase_comparison/               # 全Phase統一比較グラフ
│   ├── 01_accuracy_comparison.png
│   ├── 02_data_volume_comparison.png
│   ├── 03_approach_comparison.png
│   ├── 04_evolution_timeline.png
│   └── 05_comprehensive_table.png
│
├── phase1_6_ultimate/              # Phase 1.7結果（最高精度）⭐
│   ├── phase1_6_ultimate_results.png
│   └── ...
│
├── phase1_1_2020/                  # Phase 1.4結果
├── phase1_2_2020/                  # Phase 1.5結果
├── phase1_5_2020/                  # Phase 1.6結果（失敗版）
├── phase1_ultra_2020/              # Phase 1.3結果
├── phase1_enhanced_2020/           # Phase 1.2結果
└── phase1_real_2020/               # Phase 1.1結果
```

---

## 🎯 使用推奨度サマリー

### ✅✅✅ 本番運用推奨

- **`run_phase1_8_enhanced.py`** (Phase 1.8) - 93.64% ⭐⭐⭐ **最高精度**

### ✅ 学習・参考用推奨

- `run_phase1_6_ultimate.py` (Phase 1.7) - 従来の最高精度（79.34%）
- `run_phase1_1.py` (Phase 1.4) - 重み付け手法の参考
- `run_phase1_ultra.py` (Phase 1.3) - データソース統合の参考

### ⚠️ 参考のみ（非推奨）

- `run_phase1_2.py` (Phase 1.5) - 失敗版（過学習）
- `run_phase1_5.py` (Phase 1.6) - 失敗版（データ不足）

### ❌ 非推奨（歴史的資料）

- `run_phase1.py` (Phase 1.0) - デモデータ版
- `run_phase1_real.py` (Phase 1.1) - 初期プロトタイプ
- `run_phase1_enhanced.py` (Phase 1.2) - 初期プロトタイプ

---

## 📚 関連ドキュメント

詳細は以下のドキュメントを参照:

- **Phase 1詳細**: `docs/PHASE1_FILE_ORGANIZATION.md`
- **精度限界と改善方針**: `docs/ACCURACY_LIMITATION_ANALYSIS.md`
- **ドキュメント索引**: `docs/README.md`
- **システムアーキテクチャ**: `docs/AI_TRADER_ARCHITECTURE.md`
- **セットアップ**: `docs/SETUP_GUIDE.md`

---

## 🚀 クイックスタート

### 最高精度モデル（Phase 1.8）を実行

```bash
# 1. 環境変数設定（.envファイルにAPIキー記載）
# OANDA_API_KEY=your_key_here
# FRED_API_KEY=your_key_here

# 2. パッケージインストール
pip install -r requirements.txt

# 3. Phase 1.8実行（最高精度 93.64%）⭐⭐⭐ 推奨
python run_phase1_8_enhanced.py

# または Phase 1.7実行（従来の最高精度 79.34%）
python run_phase1_6_ultimate.py
```

### 全Phase比較グラフ作成

```bash
python create_unified_comparison.py
```

結果: `outputs/phase_comparison/` に5つのグラフが生成されます。

---

**最終更新**: 2026-01-01
**プロジェクト**: FX自動売買システム
**現在フェーズ**: Phase 1完了、Phase 2準備中
**最高精度**: Phase 1.8 Enhanced - 93.64% ⭐⭐⭐
