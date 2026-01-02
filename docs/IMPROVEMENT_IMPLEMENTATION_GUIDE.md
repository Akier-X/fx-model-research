# Phase 1 精度改善 実装ガイド

## 📋 概要

このドキュメントは、Phase 1の方向性的中率79.34%からさらに精度を向上させるための**具体的な実装ガイド**です。

**現在の到達点**: 79.34%（Phase 1.7 Ultimate）
**目標**: 85-90%（理論的上限: 90-95%）

---

## 🎯 優先順位付き改善策

| 優先度 | 施策 | 期待効果 | 実装難易度 | 実装期間 |
|-------|------|---------|-----------|---------|
| **1** | ラベル定義の見直し（閾値導入） | +5〜10% | 低 | 1-2日 |
| **2** | データ量の拡張（10年分） | +3〜5% | 低 | 1-2日 |
| **3** | 信頼度フィルタリング | +5〜8% | 低 | 1日 |
| **4** | COTレポート追加 | +2〜4% | 中 | 3-5日 |
| **5** | スタッキングアンサンブル | +2〜4% | 中 | 2-3日 |
| **6** | センチメント分析（Twitter/ニュース） | +3〜5% | 高 | 1-2週 |
| **7** | LSTM追加 | +1〜3% | 高 | 1週 |

---

## 1️⃣ ラベル定義の見直し（優先度：最高）

### 現状の問題

```python
# 現在の実装（phase1_6_ultimate_longterm.py）
future_price = df['close'].shift(-lookahead_days)
df['future_direction'] = (future_price > df['close']).astype(int)
```

**問題点**:
- 0.01円の微小な上昇も、3円の大幅上昇も同じ「上昇(1)」
- ノイズレベルの値動きを予測しようとして精度低下
- **市場のランダムウォーク成分を学習してしまう**

---

### 改善策A：閾値ベース分類（推奨）

#### コンセプト
有意な変動（±0.5%以上）のみを予測対象とし、微小な変動は除外する。

#### 実装コード

```python
def create_threshold_labels(df, lookahead_days=1, threshold=0.005):
    """
    閾値ベースのラベル作成

    Parameters:
    -----------
    df : pd.DataFrame
        価格データ
    lookahead_days : int
        何日先を予測するか
    threshold : float
        閾値（0.005 = 0.5%）

    Returns:
    --------
    labels : pd.Series
        1=有意な上昇, 0=有意な下降, -1=除外（中立）
    """
    future_price = df['close'].shift(-lookahead_days)
    future_return = (future_price - df['close']) / df['close']

    # ラベル作成
    labels = pd.Series(index=df.index, dtype=int)
    labels[future_return > threshold] = 1      # 有意な上昇
    labels[future_return < -threshold] = 0     # 有意な下降
    labels[np.abs(future_return) <= threshold] = -1  # 中立（除外）

    return labels, future_return

# 使用例
df['label'], df['return'] = create_threshold_labels(df, lookahead_days=1, threshold=0.005)

# 中立を除外して学習
mask = df['label'] != -1
X_train = X[mask]
y_train = df['label'][mask]

print(f"元のサンプル数: {len(df)}")
print(f"学習サンプル数: {mask.sum()} ({mask.sum()/len(df)*100:.1f}%)")
print(f"除外サンプル数: {(~mask).sum()} ({(~mask).sum()/len(df)*100:.1f}%)")
```

#### 閾値の最適化

```python
def optimize_threshold(df, X, thresholds=[0.003, 0.005, 0.007, 0.01, 0.015]):
    """
    複数の閾値で精度を比較
    """
    results = []

    for thresh in thresholds:
        labels, _ = create_threshold_labels(df, threshold=thresh)
        mask = labels != -1

        if mask.sum() < 100:  # サンプル数が少なすぎる場合はスキップ
            continue

        X_filtered = X[mask]
        y_filtered = labels[mask]

        # Train/Test分割
        split_idx = int(len(X_filtered) * 0.85)
        X_train, X_test = X_filtered[:split_idx], X_filtered[split_idx:]
        y_train, y_test = y_filtered[:split_idx], y_filtered[split_idx:]

        # モデル訓練
        model = XGBClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)

        # 評価
        accuracy = model.score(X_test, y_test)
        coverage = mask.sum() / len(mask)

        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'coverage': coverage,
            'samples': mask.sum()
        })

    return pd.DataFrame(results)

# 実行
results = optimize_threshold(df, X)
print(results)
```

#### 期待される結果

```
閾値0.3%:  精度82%, カバー率85%（サンプル数多、精度中）
閾値0.5%:  精度85%, カバー率70%（推奨バランス）⭐
閾値0.7%:  精度87%, カバー率55%（精度高、サンプル少）
閾値1.0%:  精度90%, カバー率40%（精度最高、実用性低）
```

**推奨**: 閾値0.5%（カバー率70%、精度85%）

---

### 改善策B：3クラス分類

#### コンセプト
大幅上昇・横ばい・大幅下降の3クラスに分類。

#### 実装コード

```python
def create_3class_labels(df, lookahead_days=1, threshold=0.01):
    """
    3クラス分類ラベル作成

    Returns:
    --------
    labels : 2=大幅上昇, 1=横ばい, 0=大幅下降
    """
    future_price = df['close'].shift(-lookahead_days)
    future_return = (future_price - df['close']) / df['close']

    labels = pd.Series(index=df.index, dtype=int)
    labels[future_return > threshold] = 2       # 大幅上昇
    labels[future_return < -threshold] = 0      # 大幅下降
    labels[(future_return >= -threshold) & (future_return <= threshold)] = 1  # 横ばい

    return labels

# 使用例
df['label_3class'] = create_3class_labels(df, threshold=0.01)

# 分布確認
print(df['label_3class'].value_counts())

# モデル訓練
model = XGBClassifier(n_estimators=100, max_depth=5, objective='multi:softmax', num_class=3)
model.fit(X_train, y_train)

# 予測
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 「横ばい」を除外した精度も計算
mask = (y_test != 1)  # 横ばいを除外
accuracy_without_neutral = accuracy_score(y_test[mask], predictions[mask])
print(f"横ばい除外精度: {accuracy_without_neutral:.2%}")
```

---

## 2️⃣ データ量の拡張（優先度：高）

### 現状
- 799日（約3年）のデータ
- 訓練サンプル: 559件

### 改善目標
- **2,500日（約10年）のデータ**
- 訓練サンプル: 1,750件（3.1倍）

---

### 実装コード

```python
from src.data_sources.yahoo_finance import YahooFinanceData
from datetime import datetime, timedelta

def get_extended_data(years_back=10, instrument="USD/JPY"):
    """
    10年分のデータを取得
    """
    yf_data = YahooFinanceData()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    # メイン通貨ペア
    main_data = yf_data.get_forex_data(
        pair=instrument.replace('_', '/'),
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )

    print(f"取得データ期間: {start_date.date()} - {end_date.date()}")
    print(f"データ日数: {len(main_data)}日")

    return main_data

# 実行例
df_10years = get_extended_data(years_back=10, instrument="USD/JPY")
```

---

### データ期間別の重み付け（推奨）

古いデータは市場構造が異なるため、**直近データを重視**する重み付けを行う。

```python
def create_time_weighted_sample(df, decay_rate=0.95):
    """
    時系列に沿って重みを付ける

    直近データほど重みが大きい（指数減衰）

    Parameters:
    -----------
    df : pd.DataFrame
        時系列データ（古い順にソート済み）
    decay_rate : float
        減衰率（0.95推奨）

    Returns:
    --------
    weights : np.array
        各サンプルの重み
    """
    n = len(df)
    # 直近が1、最古がdecay_rate^nとなる重み
    weights = np.array([decay_rate ** (n - i - 1) for i in range(n)])

    # 正規化（合計を1に）
    weights = weights / weights.sum()

    return weights

# 使用例
sample_weights = create_time_weighted_sample(df, decay_rate=0.95)

# XGBoostで重み付き学習
model = XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train, sample_weight=sample_weights[:len(X_train)])
```

---

### 市場レジーム分割（高度な手法）

市場を「上昇トレンド期」「下降トレンド期」「レンジ相場期」に分割し、各期間から均等にサンプリング。

```python
def identify_market_regime(df, window=60):
    """
    市場レジームの識別

    Returns:
    --------
    regime : 'uptrend', 'downtrend', 'range'
    """
    # 60日移動平均の傾き
    ma60 = df['close'].rolling(window=window).mean()
    slope = ma60.diff(window)

    # ボラティリティ
    volatility = df['close'].pct_change().rolling(window=window).std()

    # レジーム判定
    regime = pd.Series(index=df.index, dtype=str)
    regime[slope > 0.02] = 'uptrend'      # 上昇トレンド
    regime[slope < -0.02] = 'downtrend'   # 下降トレンド
    regime[(slope >= -0.02) & (slope <= 0.02)] = 'range'  # レンジ

    return regime

# 使用例
df['regime'] = identify_market_regime(df, window=60)

# レジーム別のサンプル数確認
print(df['regime'].value_counts())

# 各レジームから均等にサンプリング
from sklearn.utils import resample

regime_samples = []
for regime in ['uptrend', 'downtrend', 'range']:
    regime_data = df[df['regime'] == regime]
    # 各レジームから200サンプルずつ
    sampled = resample(regime_data, n_samples=min(200, len(regime_data)), random_state=42)
    regime_samples.append(sampled)

balanced_df = pd.concat(regime_samples)
```

---

## 3️⃣ 信頼度ベースのフィルタリング（優先度：高）

### コンセプト
確信度の高い予測のみを採用し、**精度を優先**する。

---

### 実装コード

```python
def predict_with_confidence(model, X, threshold=0.65):
    """
    信頼度ベースの予測

    Parameters:
    -----------
    model : sklearn model
        訓練済みモデル
    X : array-like
        特徴量
    threshold : float
        信頼度閾値（0.65推奨）

    Returns:
    --------
    predictions : np.array
        1=上昇予測, 0=下降予測, -1=見送り
    probabilities : np.array
        確率値
    """
    probabilities = model.predict_proba(X)[:, 1]  # 上昇確率

    predictions = np.full(len(probabilities), -1)  # デフォルト: 見送り
    predictions[probabilities > threshold] = 1      # 上昇予測（確信あり）
    predictions[probabilities < (1 - threshold)] = 0  # 下降予測（確信あり）

    return predictions, probabilities

# 使用例
predictions, probabilities = predict_with_confidence(model, X_test, threshold=0.65)

# 見送りを除外した精度計算
mask = predictions != -1
accuracy = accuracy_score(y_test[mask], predictions[mask])
coverage = mask.sum() / len(mask)

print(f"信頼度閾値: 0.65")
print(f"精度: {accuracy:.2%}")
print(f"カバー率: {coverage:.2%}")
print(f"見送り: {(~mask).sum()}件 ({(~mask).sum()/len(mask)*100:.1f}%)")
```

---

### 閾値最適化

```python
def optimize_confidence_threshold(model, X_test, y_test):
    """
    信頼度閾値を最適化
    """
    thresholds = np.arange(0.5, 0.85, 0.05)
    results = []

    for thresh in thresholds:
        predictions, probabilities = predict_with_confidence(model, X_test, threshold=thresh)

        mask = predictions != -1
        if mask.sum() == 0:
            continue

        accuracy = accuracy_score(y_test[mask], predictions[mask])
        coverage = mask.sum() / len(mask)

        # 期待値計算（精度 × カバー率）
        expected_value = accuracy * coverage

        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'coverage': coverage,
            'expected_value': expected_value,
            'predictions': mask.sum()
        })

    df_results = pd.DataFrame(results)

    # 期待値が最大の閾値を推奨
    best_idx = df_results['expected_value'].idxmax()
    best_threshold = df_results.loc[best_idx, 'threshold']

    print("信頼度閾値別の結果:")
    print(df_results)
    print(f"\n推奨閾値: {best_threshold}")

    return df_results

# 実行
results = optimize_confidence_threshold(model, X_test, y_test)
```

---

### グラフ可視化

```python
import matplotlib.pyplot as plt

def plot_confidence_analysis(results_df):
    """
    信頼度分析の可視化
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 精度とカバー率
    ax1 = axes[0]
    ax1.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='精度', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['coverage'], 's-', label='カバー率', linewidth=2)
    ax1.set_xlabel('信頼度閾値', fontsize=12)
    ax1.set_ylabel('値', fontsize=12)
    ax1.set_title('信頼度閾値 vs 精度・カバー率', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右: 期待値
    ax2 = axes[1]
    ax2.plot(results_df['threshold'], results_df['expected_value'], 'D-', color='green', linewidth=2)
    best_idx = results_df['expected_value'].idxmax()
    best_thresh = results_df.loc[best_idx, 'threshold']
    best_ev = results_df.loc[best_idx, 'expected_value']
    ax2.scatter([best_thresh], [best_ev], color='red', s=200, zorder=5, label=f'最適: {best_thresh}')
    ax2.set_xlabel('信頼度閾値', fontsize=12)
    ax2.set_ylabel('期待値（精度 × カバー率）', fontsize=12)
    ax2.set_title('信頼度閾値 vs 期待値', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/confidence_analysis.png', dpi=150)
    plt.close()

# 実行
plot_confidence_analysis(results)
```

---

## 4️⃣ COTレポートの追加（優先度：中）

### COT（Commitment of Traders）レポートとは

CFTC（米商品先物取引委員会）が毎週発表する、大口投機筋のポジション情報。

**重要性**:
- 大口投機筋（ヘッジファンド等）の動向がわかる
- **先行指標**として機能（大口が先に動く）
- 極端なポジション偏りは反転のサイン

---

### データ取得

```python
import pandas as pd
import requests
from io import StringIO

def fetch_cot_data(start_year=2014):
    """
    COTレポートを取得

    CFTCの公式サイトからダウンロード
    """
    # CFTC COT Historical Data
    # Japanese Yen Futures (Code: 097741)
    url = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{}.zip"

    all_data = []
    current_year = datetime.now().year

    for year in range(start_year, current_year + 1):
        try:
            # データダウンロード
            response = requests.get(url.format(year))
            # ZIP解凍と読み込み
            # (実装省略 - 実際はzipファイルの解凍が必要)

            # Japanese Yen（コード097741）のみ抽出
            # ...

        except Exception as e:
            print(f"Error fetching {year}: {e}")
            continue

    df_cot = pd.concat(all_data)
    return df_cot

# 簡易版: 手動ダウンロードしたCSVを読み込み
df_cot = pd.read_csv('cot_jpy_data.csv', parse_dates=['date'])
```

---

### 特徴量作成

```python
def create_cot_features(df_cot):
    """
    COTレポートから特徴量を作成
    """
    df_cot = df_cot.sort_values('date').reset_index(drop=True)

    # 1. ネットポジション（ロング - ショート）
    df_cot['speculator_net'] = df_cot['speculator_long'] - df_cot['speculator_short']
    df_cot['commercial_net'] = df_cot['commercial_long'] - df_cot['commercial_short']

    # 2. ネットポジションの変化率
    df_cot['speculator_net_change'] = df_cot['speculator_net'].pct_change()
    df_cot['commercial_net_change'] = df_cot['commercial_net'].pct_change()

    # 3. ポジションの偏り度（極端な偏りは反転のサイン）
    df_cot['speculator_net_zscore'] = (
        df_cot['speculator_net'] - df_cot['speculator_net'].rolling(52).mean()
    ) / df_cot['speculator_net'].rolling(52).std()

    # 4. ロングとショートの比率
    df_cot['speculator_long_short_ratio'] = (
        df_cot['speculator_long'] / (df_cot['speculator_short'] + 1e-10)
    )

    return df_cot

# 使用例
df_cot = create_cot_features(df_cot)
```

---

### メインデータとの統合

```python
def merge_cot_with_price_data(df_price, df_cot):
    """
    価格データとCOTレポートを統合

    COTは週次データなので、日次データに前方埋め
    """
    # 日付でマージ
    df_merged = pd.merge_asof(
        df_price.sort_values('date'),
        df_cot[['date', 'speculator_net', 'speculator_net_change',
                'speculator_net_zscore', 'speculator_long_short_ratio']],
        on='date',
        direction='backward'  # 直近のCOTデータを使用
    )

    return df_merged

# 実行
df = merge_cot_with_price_data(df_price, df_cot)

print("追加されたCOT特徴量:")
print(['speculator_net', 'speculator_net_change', 'speculator_net_zscore', 'speculator_long_short_ratio'])
```

---

### COT特徴量の有効性検証

```python
from sklearn.ensemble import RandomForestClassifier

def evaluate_cot_features(df, X_original, y):
    """
    COT特徴量の重要度を評価
    """
    # COT特徴量のみでモデル訓練
    cot_features = ['speculator_net', 'speculator_net_change',
                    'speculator_net_zscore', 'speculator_long_short_ratio']

    X_cot = df[cot_features].fillna(0)

    # Train/Test分割
    split_idx = int(len(X_cot) * 0.85)
    X_train, X_test = X_cot[:split_idx], X_cot[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    accuracy_cot_only = model.score(X_test, y_test)

    # 全特徴量と比較
    model_full = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    X_full = pd.concat([X_original, X_cot], axis=1)
    model_full.fit(X_full[:split_idx], y_train)
    accuracy_full = model_full.score(X_full[split_idx:], y_test)

    print("COT特徴量の評価:")
    print(f"  COT特徴量のみ: {accuracy_cot_only:.2%}")
    print(f"  全特徴量: {accuracy_full:.2%}")
    print(f"  改善: {(accuracy_full - accuracy_cot_only):.2%}")

    return model_full

# 実行
model_with_cot = evaluate_cot_features(df, X_original, y)
```

---

## 5️⃣ スタッキングアンサンブル（優先度：中）

### コンセプト
2段階のアンサンブル学習：
1. **Level 1**: 多様なベースモデル群
2. **Level 2**: Level 1の出力を入力とするメタモデル

**メリット**: モデルの多様性が増し、過学習を防ぎつつ精度向上

---

### 実装コード

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_stacking_model():
    """
    スタッキングアンサンブルモデル構築
    """
    # Level 1: ベースモデル（多様性を重視）
    base_models = [
        ('gbc', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)),
        ('cat', CatBoostClassifier(iterations=100, depth=5, random_state=42, verbose=False)),
    ]

    # Level 2: メタモデル（シンプルなモデルを推奨）
    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    # スタッキング
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,  # 5-fold Cross Validation
        stack_method='predict_proba',  # 確率を使用
        passthrough=True  # 元の特徴量もメタモデルに渡す
    )

    return stacking_clf

# 使用例
stacking_model = build_stacking_model()
stacking_model.fit(X_train, y_train)

# 評価
accuracy = stacking_model.score(X_test, y_test)
print(f"スタッキングアンサンブル精度: {accuracy:.2%}")
```

---

### 特徴量のサブセット化（高度）

各ベースモデルに異なる特徴量セットを使用し、さらに多様性を増す。

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class FeatureSubsetClassifier(BaseEstimator, ClassifierMixin):
    """
    特定の特徴量サブセットのみを使用するラッパー
    """
    def __init__(self, estimator, feature_indices):
        self.estimator = estimator
        self.feature_indices = feature_indices

    def fit(self, X, y):
        X_subset = X[:, self.feature_indices]
        self.estimator.fit(X_subset, y)
        return self

    def predict(self, X):
        X_subset = X[:, self.feature_indices]
        return self.estimator.predict(X_subset)

    def predict_proba(self, X):
        X_subset = X[:, self.feature_indices]
        return self.estimator.predict_proba(X_subset)

# 特徴量を3グループに分割
n_features = X_train.shape[1]
group1 = list(range(0, n_features // 3))                    # テクニカル指標
group2 = list(range(n_features // 3, 2 * n_features // 3))  # 統計特徴
group3 = list(range(2 * n_features // 3, n_features))       # クロス通貨・経済指標

# 各モデルに異なる特徴量セットを割り当て
base_models = [
    ('xgb_tech', FeatureSubsetClassifier(XGBClassifier(), group1)),
    ('lgb_stat', FeatureSubsetClassifier(LGBMClassifier(), group2)),
    ('cat_econ', FeatureSubsetClassifier(CatBoostClassifier(verbose=False), group3)),
    ('rf_all', RandomForestClassifier()),  # 全特徴量使用
]

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
```

---

## 6️⃣ LSTM追加（優先度：低〜中）

### 注意事項
LSTMは時系列の長期依存関係を捕捉できるが、**大量データ（1,000+サンプル）が必要**。

現在のデータ量（799日）では効果薄。**10年分データ取得後**に実装推奨。

---

### 実装コード

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_sequences(X, y, sequence_length=20):
    """
    時系列シーケンスを作成

    Parameters:
    -----------
    X : np.array
        特徴量 (samples, features)
    y : np.array
        ラベル
    sequence_length : int
        シーケンス長（過去何日分を見るか）

    Returns:
    --------
    X_seq : np.array (samples - sequence_length, sequence_length, features)
    y_seq : np.array (samples - sequence_length,)
    """
    X_seq, y_seq = [], []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(sequence_length, n_features):
    """
    LSTMモデル構築
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(32, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),

        Dense(16, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# 使用例
sequence_length = 20
n_features = X_train.shape[1]

# シーケンス作成
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

print(f"X_train_seq shape: {X_train_seq.shape}")  # (samples, 20, n_features)
print(f"y_train_seq shape: {y_train_seq.shape}")  # (samples,)

# モデル構築
lstm_model = build_lstm_model(sequence_length, n_features)
lstm_model.summary()

# コールバック
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# 訓練
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 評価
loss, accuracy = lstm_model.evaluate(X_test_seq, y_test_seq)
print(f"LSTM精度: {accuracy:.2%}")
```

---

### LSTMと決定木のハイブリッド

```python
def build_hybrid_model(X_train, y_train, X_test, y_test, sequence_length=20):
    """
    LSTM + 決定木のハイブリッドアンサンブル
    """
    # 1. LSTMモデル
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

    lstm_model = build_lstm_model(sequence_length, X_train.shape[1])
    lstm_model.fit(X_train_seq, y_train_seq, validation_split=0.15, epochs=50, verbose=0)

    # LSTM予測（確率）
    lstm_proba = lstm_model.predict(X_test_seq)

    # 2. XGBoostモデル
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5)
    # シーケンス長分をスキップ
    xgb_model.fit(X_train[sequence_length:], y_train[sequence_length:])
    xgb_proba = xgb_model.predict_proba(X_test[sequence_length:])[:, 1]

    # 3. アンサンブル（重み付き平均）
    ensemble_proba = 0.5 * lstm_proba.flatten() + 0.5 * xgb_proba
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test[sequence_length:], ensemble_pred)
    print(f"ハイブリッドアンサンブル精度: {accuracy:.2%}")

    return lstm_model, xgb_model

# 実行
lstm_model, xgb_model = build_hybrid_model(X_train, y_train, X_test, y_test)
```

---

## 7️⃣ センチメント分析（優先度：中〜高）

### Twitter感情分析

```python
# 注: Twitter API v2が必要（無料枠あり）
import tweepy
from textblob import TextBlob

def fetch_forex_sentiment(api_key, api_secret, keywords=['USDJPY', 'forex', '円高', '円安']):
    """
    Twitterから為替関連ツイートを取得し、感情分析
    """
    # Twitter API認証
    client = tweepy.Client(bearer_token='YOUR_BEARER_TOKEN')

    sentiments = []
    for keyword in keywords:
        # ツイート検索
        tweets = client.search_recent_tweets(
            query=keyword,
            max_results=100,
            tweet_fields=['created_at', 'public_metrics']
        )

        for tweet in tweets.data:
            # 感情分析
            analysis = TextBlob(tweet.text)
            sentiment = analysis.sentiment.polarity  # -1（ネガティブ） ~ 1（ポジティブ）
            sentiments.append({
                'date': tweet.created_at.date(),
                'keyword': keyword,
                'sentiment': sentiment
            })

    df_sentiment = pd.DataFrame(sentiments)

    # 日次集計
    daily_sentiment = df_sentiment.groupby('date')['sentiment'].mean()

    return daily_sentiment

# 使用例
# daily_sentiment = fetch_forex_sentiment(api_key, api_secret)
```

---

### ニュース感情分析（FinBERT）

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def analyze_news_sentiment(news_text):
    """
    FinBERTで金融ニュースの感情分析

    FinBERT: 金融ニュース特化のBERTモデル
    """
    # FinBERTモデル読み込み
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

    # トークン化
    inputs = tokenizer(news_text, return_tensors='pt', truncation=True, max_length=512)

    # 予測
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # ラベル: 0=negative, 1=neutral, 2=positive
    sentiment_score = predictions[0][2].item() - predictions[0][0].item()  # positive - negative

    return sentiment_score

# 使用例
news = "米FRB、利上げペース加速を示唆。ドル円は急騰し、150円台に到達。"
sentiment = analyze_news_sentiment(news)
print(f"ニュース感情スコア: {sentiment:.3f}")
```

---

## 📊 実装優先順位と期待効果まとめ

### フェーズ1: 即座に実装可能（1週間以内）

```
☐ 1. ラベル定義の見直し（閾値0.5%）
   期待効果: +5〜10%
   実装難易度: 低

☐ 2. 信頼度フィルタリング（閾値0.65）
   期待効果: +5〜8%
   実装難易度: 低

☐ 3. データ量拡張（10年分）
   期待効果: +3〜5%
   実装難易度: 低
```

**期待される合計改善**: +13〜23%
**目標精度**: 79.34% + 15% = **94.34%**（理論上限に到達）

---

### フェーズ2: 中期実装（2-3週間）

```
☐ 4. COTレポート追加
   期待効果: +2〜4%
   実装難易度: 中

☐ 5. スタッキングアンサンブル
   期待効果: +2〜4%
   実装難易度: 中

☐ 6. 時間重み付けサンプリング
   期待効果: +1〜2%
   実装難易度: 低
```

---

### フェーズ3: 長期実装（1-2ヶ月）

```
☐ 7. センチメント分析（Twitter/ニュース）
   期待効果: +3〜5%
   実装難易度: 高

☐ 8. LSTM追加
   期待効果: +1〜3%
   実装難易度: 高
   前提条件: 10年分データ必須
```

---

## 🚀 次のアクション

### 今すぐ実装すべきTop 3

1. **ラベル定義の見直し**
   - `src/model_builder/phase1_7_threshold_labels.py` を作成
   - 閾値0.5%で実装
   - 期待精度: 84-89%

2. **データ量拡張**
   - Yahoo Financeから10年分取得
   - 時間重み付けサンプリング実装
   - 期待精度: 82-84%

3. **信頼度フィルタリング**
   - 閾値0.65で実装
   - 精度優先戦略
   - 期待精度: 84-87%

### 実装コード例（統合版）

次のメッセージで、これら3つを統合した **Phase 1.7 Enhanced** の実装コードを提供します。

---

**作成日**: 2026-01-01
**対象**: Phase 1精度改善
**目標**: 79.34% → 85-90%
