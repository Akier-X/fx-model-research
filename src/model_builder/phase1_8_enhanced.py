"""
Phase 1.8 Enhanced - 最強モデル

3大改善策を統合:
1. 閾値ベースラベル (0.5%以上の変動のみ予測)
2. 10年分データ (Yahoo Finance)
3. 信頼度フィルタリング (確率0.65以上のみ採用)

目標精度: 85-90%
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.data_sources.yahoo_finance import YahooFinanceData
from src.data_sources.economic_indicators import EconomicIndicators


class Phase1_8_Enhanced:
    """
    Phase 1.8 Enhanced - 最強モデル

    改善点:
    1. 閾値ベースラベル: ±0.5%以上の変動のみ予測対象
    2. 10年分データ: 2,500日以上の訓練データ
    3. 信頼度フィルタリング: 確率0.65以上のみ予測
    4. 時間重み付け: 直近データを重視
    """

    def __init__(self, years_back=10, instrument="USD/JPY", top_features=60,
                 lookahead_days=1, threshold=0.005, confidence_threshold=0.65):
        """
        Parameters:
        -----------
        years_back : int
            過去何年分のデータを取得するか（10年推奨）
        instrument : str
            通貨ペア
        top_features : int
            使用する特徴量の数（重要度上位）
        lookahead_days : int
            何日先を予測するか
        threshold : float
            ラベル閾値（0.005 = 0.5%）
        confidence_threshold : float
            信頼度閾値（0.65推奨）
        """
        self.years_back = years_back
        self.instrument = instrument
        self.top_features = top_features
        self.lookahead_days = lookahead_days
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold

        # データソース初期化
        self.yf_data = YahooFinanceData()
        self.econ_indicators = EconomicIndicators()

        # 出力ディレクトリ
        self.output_dir = Path("outputs/phase1_8_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 1.8 Enhanced 初期化")
        logger.info(f"  データ期間: 過去{years_back}年")
        logger.info(f"  ラベル閾値: ±{threshold*100:.1f}%")
        logger.info(f"  信頼度閾値: {confidence_threshold:.2f}")

    def fetch_all_data(self):
        """
        全データソースからデータ取得（10年分）
        """
        logger.info("=" * 70)
        logger.info("データ取得開始（10年分）")
        logger.info("=" * 70)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years_back * 365)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # 1. Yahoo Finance: メイン通貨ペア（10年分）
        logger.info(f"1. Yahoo Finance: {self.instrument} ({start_str} - {end_str})")
        main_data = self.yf_data.get_forex_data(
            pair=self.instrument.replace('_', '/'),
            start_date=start_str,
            end_date=end_str,
            interval='1d'
        )
        logger.info(f"   取得: {len(main_data)}日分")

        # 価格データを保存（可視化用）
        self.price_data = main_data.copy()

        # 2. Yahoo Finance: 複数通貨ペア
        logger.info("2. Yahoo Finance: 複数通貨ペア")
        multi_pairs_dict = self.yf_data.get_multiple_pairs(
            pairs=["EUR/USD", "GBP/USD", "EUR/JPY"],
            start_date=start_str,
            end_date=end_str
        )

        # dictをDataFrameに変換
        multi_pairs = pd.DataFrame()
        for pair, pair_data in multi_pairs_dict.items():
            pair_name = pair.replace('/', '_')
            for col in pair_data.columns:
                if col != 'volume':  # volumeは除外
                    multi_pairs[f'{pair_name}_{col}'] = pair_data[col]

        logger.info(f"   取得: {len(multi_pairs.columns)}列")

        # 3. Yahoo Finance: 株価指数
        logger.info("3. Yahoo Finance: 株価指数")
        indices_dict = self.yf_data.get_stock_indices(
            start_date=start_str,
            end_date=end_str
        )

        # dictをDataFrameに変換
        indices = pd.DataFrame()
        for index_name, index_data in indices_dict.items():
            for col in index_data.columns:
                if col != 'volume':  # volumeは除外
                    indices[f'{index_name}_{col}'] = index_data[col]

        logger.info(f"   取得: {len(indices.columns)}列")

        # 4. FRED API: 経済指標
        logger.info("4. FRED API: 経済指標")
        econ_data = self.econ_indicators.get_all_indicators(
            start_date=start_str,
            end_date=end_str
        )
        logger.info(f"   取得: {len(econ_data.columns)}列")

        # データ統合
        df = main_data.copy()
        df = df.join(multi_pairs, how='left')
        df = df.join(indices, how='left')
        df = df.join(econ_data, how='left')

        # 前方補完（経済指標は月次/週次のため）
        df = df.fillna(method='ffill')
        df = df.dropna()

        logger.info(f"\n統合後データ: {len(df)}日分, {len(df.columns)}列")
        logger.info("=" * 70)

        return df

    def create_threshold_labels(self, df):
        """
        閾値ベースのラベル作成

        ±threshold以上の変動のみを予測対象とする

        Returns:
        --------
        labels : pd.Series
            1=有意な上昇, 0=有意な下降, -1=除外（中立）
        returns : pd.Series
            実際のリターン
        """
        future_price = df['close'].shift(-self.lookahead_days)
        future_return = (future_price - df['close']) / df['close']

        # ラベル作成
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_return > self.threshold] = 1      # 有意な上昇
        labels[future_return < -self.threshold] = 0     # 有意な下降
        labels[np.abs(future_return) <= self.threshold] = -1  # 中立（除外）

        # 統計情報
        n_up = (labels == 1).sum()
        n_down = (labels == 0).sum()
        n_neutral = (labels == -1).sum()
        total = len(labels)

        logger.info(f"\nラベル統計（閾値±{self.threshold*100:.1f}%）:")
        logger.info(f"  有意な上昇: {n_up}件 ({n_up/total*100:.1f}%)")
        logger.info(f"  有意な下降: {n_down}件 ({n_down/total*100:.1f}%)")
        logger.info(f"  中立（除外）: {n_neutral}件 ({n_neutral/total*100:.1f}%)")
        logger.info(f"  カバー率: {(n_up+n_down)/total*100:.1f}%")

        return labels, future_return

    def prepare_features_and_labels(self, df):
        """
        特徴量エンジニアリングとラベル作成
        """
        logger.info("\n特徴量エンジニアリング開始")

        feature_df = pd.DataFrame(index=df.index)

        # ========== 基本的な特徴量 ==========
        feature_df['open'] = df['open']
        feature_df['high'] = df['high']
        feature_df['low'] = df['low']
        feature_df['close'] = df['close']
        feature_df['volume'] = df['volume']

        # 価格変化
        feature_df['price_change'] = df['close'].pct_change()
        feature_df['high_low_range'] = (df['high'] - df['low']) / df['close']
        feature_df['high_close_ratio'] = df['high'] / df['close']
        feature_df['low_close_ratio'] = df['low'] / df['close']

        # ========== 移動平均 ==========
        for window in [5, 10, 20, 30, 50, 100, 200]:
            feature_df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            feature_df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            feature_df[f'sma_{window}_ratio'] = df['close'] / feature_df[f'sma_{window}']

        # ========== RSI ==========
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            feature_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # RSIの傾き
            feature_df[f'rsi_{period}_slope'] = feature_df[f'rsi_{period}'].diff(5)

        # ========== MACD ==========
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        feature_df['macd'] = ema12 - ema26
        feature_df['macd_signal'] = feature_df['macd'].ewm(span=9).mean()
        feature_df['macd_histogram'] = feature_df['macd'] - feature_df['macd_signal']
        feature_df['macd_slope'] = feature_df['macd'].diff(5)

        # ========== ボリンジャーバンド ==========
        for window in [20, 50]:
            sma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            feature_df[f'bb_upper_{window}'] = sma + (2 * std)
            feature_df[f'bb_lower_{window}'] = sma - (2 * std)
            feature_df[f'bb_position_{window}'] = (df['close'] - feature_df[f'bb_lower_{window}']) / \
                                                   (feature_df[f'bb_upper_{window}'] - feature_df[f'bb_lower_{window}'] + 1e-10)

        # ========== ATR（ボラティリティ） ==========
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for period in [14, 28]:
            feature_df[f'atr_{period}'] = true_range.rolling(window=period).mean()
            feature_df[f'atr_{period}_ratio'] = feature_df[f'atr_{period}'] / df['close']

        # ========== ストキャスティクス ==========
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            feature_df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)

        # ========== ボラティリティ ==========
        for window in [10, 20, 30]:
            feature_df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std()

        # ========== リターン ==========
        for days in [1, 5, 10, 20]:
            feature_df[f'return_{days}d'] = df['close'].pct_change(days)

        # ========== 外部データ（通貨ペア、株価指数、経済指標） ==========
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                if 'return' in col or 'rate' in col or 'index' in col:
                    feature_df[col] = df[col]
                elif col.startswith(('EUR', 'GBP', 'SP500', 'NIKKEI', 'NASDAQ', 'VIX')):
                    # リターンと相関を計算
                    feature_df[f'{col}_return'] = df[col].pct_change()
                    feature_df[f'{col}_corr_10'] = df['close'].rolling(10).corr(df[col])
                    feature_df[f'{col}_corr_20'] = df['close'].rolling(20).corr(df[col])

        # 欠損値処理
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info(f"特徴量数: {len(feature_df.columns)}個")

        # ラベル作成（閾値ベース）
        labels, returns = self.create_threshold_labels(df)

        # 中立を除外
        mask = labels != -1
        feature_df_filtered = feature_df[mask]
        labels_filtered = labels[mask]
        returns_filtered = returns[mask]

        # NaN除去
        nan_mask = ~(labels_filtered.isna() | feature_df_filtered.isna().any(axis=1))
        feature_df_filtered = feature_df_filtered[nan_mask]
        labels_filtered = labels_filtered[nan_mask]
        returns_filtered = returns_filtered[nan_mask]

        logger.info(f"フィルタ後サンプル数: {len(feature_df_filtered)}件")

        return feature_df_filtered, labels_filtered, returns_filtered

    def select_top_features(self, X, y, n_features):
        """
        Random Forestで特徴量重要度を計算し、上位を選択
        """
        logger.info(f"\n特徴量選択（上位{n_features}個）")

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # 重要度
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        top_features = importances.head(n_features).index.tolist()

        logger.info(f"Top 10特徴量:")
        for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")

        return top_features, importances

    def create_time_weights(self, n_samples, decay_rate=0.95):
        """
        時間重み付け（直近データを重視）
        """
        weights = np.array([decay_rate ** (n_samples - i - 1) for i in range(n_samples)])
        weights = weights / weights.sum()
        return weights

    def train_models(self, X_train, y_train, X_val, y_val):
        """
        複数モデルを訓練（時間重み付き）
        """
        logger.info("\n" + "=" * 70)
        logger.info("モデル訓練開始（時間重み付き）")
        logger.info("=" * 70)

        # 時間重み
        sample_weights = self.create_time_weights(len(X_train), decay_rate=0.95)

        models = {
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(iterations=100, depth=5, random_state=42, verbose=False),
        }

        trained_models = {}
        val_accuracies = {}

        for name, model in models.items():
            logger.info(f"\n訓練中: {name}")

            # 時間重み付き訓練
            if name in ['XGBoost', 'LightGBM', 'CatBoost']:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)

            # 検証精度
            val_acc = model.score(X_val, y_val)
            val_accuracies[name] = val_acc
            trained_models[name] = model

            logger.info(f"  検証精度: {val_acc*100:.2f}%")

        return trained_models, val_accuracies

    def predict_with_confidence(self, models, model_weights, X, y_true=None):
        """
        アンサンブル予測 + 信頼度フィルタリング

        Returns:
        --------
        predictions : np.array
            1=上昇, 0=下降, -1=見送り
        probabilities : np.array
            アンサンブル確率
        """
        # 各モデルの確率予測
        all_probabilities = []
        for name, model in models.items():
            proba = model.predict_proba(X)[:, 1]  # 上昇確率
            all_probabilities.append(proba * model_weights[name])

        # 重み付き平均
        ensemble_proba = np.sum(all_probabilities, axis=0)

        # 信頼度フィルタリング
        predictions = np.full(len(ensemble_proba), -1)  # デフォルト: 見送り
        predictions[ensemble_proba > self.confidence_threshold] = 1      # 上昇予測
        predictions[ensemble_proba < (1 - self.confidence_threshold)] = 0  # 下降予測

        # 統計
        n_up = (predictions == 1).sum()
        n_down = (predictions == 0).sum()
        n_skip = (predictions == -1).sum()

        logger.info(f"\n予測統計（信頼度閾値{self.confidence_threshold}）:")
        logger.info(f"  上昇予測: {n_up}件 ({n_up/len(predictions)*100:.1f}%)")
        logger.info(f"  下降予測: {n_down}件 ({n_down/len(predictions)*100:.1f}%)")
        logger.info(f"  見送り: {n_skip}件 ({n_skip/len(predictions)*100:.1f}%)")
        logger.info(f"  カバー率: {(n_up+n_down)/len(predictions)*100:.1f}%")

        if y_true is not None:
            # 見送りを除外した精度
            mask = predictions != -1
            if mask.sum() > 0:
                accuracy = accuracy_score(y_true[mask], predictions[mask])
                logger.info(f"  精度（見送り除外）: {accuracy*100:.2f}%")

        return predictions, ensemble_proba

    def calculate_metrics(self, y_true, y_pred, probabilities):
        """
        評価指標を計算（見送りを除外）
        """
        # 見送りを除外
        mask = y_pred != -1
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        if len(y_true_filtered) == 0:
            logger.warning("予測サンプルが0件です")
            return {}

        # 基本指標
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)

        # 混同行列
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # クラス別精度
        precision_up = precision_score(y_true_filtered, y_pred_filtered, pos_label=1, zero_division=0)
        precision_down = precision_score(y_true_filtered, y_pred_filtered, pos_label=0, zero_division=0)
        recall_up = recall_score(y_true_filtered, y_pred_filtered, pos_label=1, zero_division=0)
        recall_down = recall_score(y_true_filtered, y_pred_filtered, pos_label=0, zero_division=0)
        f1_up = f1_score(y_true_filtered, y_pred_filtered, pos_label=1, zero_division=0)
        f1_down = f1_score(y_true_filtered, y_pred_filtered, pos_label=0, zero_division=0)

        metrics = {
            'Accuracy': accuracy * 100,
            'Coverage': mask.sum() / len(mask) * 100,
            'Skipped': (~mask).sum(),
            'True_Negatives': int(tn),
            'False_Positives': int(fp),
            'False_Negatives': int(fn),
            'True_Positives': int(tp),
            'Precision_Up': precision_up * 100,
            'Precision_Down': precision_down * 100,
            'Recall_Up': recall_up * 100,
            'Recall_Down': recall_down * 100,
            'F1_Up': f1_up * 100,
            'F1_Down': f1_down * 100,
        }

        return metrics

    def create_visualizations(self, y_test, y_pred, probabilities, metrics, model_weights):
        """
        結果の可視化
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 見送りを除外
        mask = y_pred != -1
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        proba_filtered = probabilities[mask]

        # 1. 累積精度
        ax1 = fig.add_subplot(gs[0, :2])
        correct = (y_test_filtered == y_pred_filtered).astype(int)
        cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)

        ax1.plot(cumulative_accuracy * 100, linewidth=2, color='#2ecc71')
        ax1.axhline(y=metrics['Accuracy'], color='red', linestyle='--', linewidth=2, label=f"最終精度 {metrics['Accuracy']:.2f}%")
        ax1.axhline(y=90, color='orange', linestyle=':', linewidth=2, label='目標 90%')
        ax1.set_xlabel('予測サンプル数', fontsize=12)
        ax1.set_ylabel('累積的中率 (%)', fontsize=12)
        ax1.set_title('Phase 1.8 Enhanced - 累積的中率の推移', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 2. モデル別重み
        ax2 = fig.add_subplot(gs[0, 2])
        models = list(model_weights.keys())
        weights = list(model_weights.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        ax2.barh(models, weights, color=colors)
        ax2.set_xlabel('重み', fontsize=11)
        ax2.set_title('モデル別重み分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. 混同行列
        ax3 = fig.add_subplot(gs[1, 0])
        cm = np.array([[metrics['True_Negatives'], metrics['False_Positives']],
                       [metrics['False_Negatives'], metrics['True_Positives']]])

        im = ax3.imshow(cm, cmap='Blues', aspect='auto')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['予測: 下降', '予測: 上昇'])
        ax3.set_yticklabels(['実際: 下降', '実際: 上昇'])

        for i in range(2):
            for j in range(2):
                text = ax3.text(j, i, cm[i, j], ha="center", va="center",
                               color="white" if cm[i, j] > cm.max()/2 else "black",
                               fontsize=16, fontweight='bold')

        ax3.set_title('混同行列', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax3)

        # 4. 確率分布
        ax4 = fig.add_subplot(gs[1, 1])
        proba_up = proba_filtered[y_test_filtered == 1]
        proba_down = proba_filtered[y_test_filtered == 0]

        ax4.hist(proba_up, bins=30, alpha=0.7, label='実際: 上昇', color='#2ecc71', edgecolor='black')
        ax4.hist(proba_down, bins=30, alpha=0.7, label='実際: 下降', color='#e74c3c', edgecolor='black')
        ax4.axvline(x=self.confidence_threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'信頼度閾値 {self.confidence_threshold}')
        ax4.axvline(x=(1-self.confidence_threshold), color='blue', linestyle='--', linewidth=2)
        ax4.set_xlabel('予測確率（上昇）', fontsize=11)
        ax4.set_ylabel('頻度', fontsize=11)
        ax4.set_title('確率分布（信頼度フィルタ適用）', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. クラス別精度
        ax5 = fig.add_subplot(gs[1, 2])
        metrics_names = ['精度\n(上昇)', '精度\n(下降)', '再現率\n(上昇)', '再現率\n(下降)']
        metrics_values = [metrics['Precision_Up'], metrics['Precision_Down'],
                         metrics['Recall_Up'], metrics['Recall_Down']]
        colors_bar = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

        bars = ax5.bar(metrics_names, metrics_values, color=colors_bar, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, metrics_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

        ax5.set_ylabel('%', fontsize=11)
        ax5.set_title('クラス別性能指標', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 105)
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. サマリーテーブル
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        summary_data = [
            ['指標', '値', '備考'],
            ['方向性的中率', f"{metrics['Accuracy']:.2f}%", '⭐ 見送り除外'],
            ['カバー率', f"{metrics['Coverage']:.2f}%", f"{int(metrics['Coverage'])}%の相場で予測"],
            ['見送り件数', f"{metrics['Skipped']}件", f"信頼度<{self.confidence_threshold}"],
            ['上昇的中精度', f"{metrics['Precision_Up']:.2f}%", '上昇予測の正確性'],
            ['下降的中精度', f"{metrics['Precision_Down']:.2f}%", '下降予測の正確性'],
            ['', '', ''],
            ['改善策', '', ''],
            ['1. 閾値ベースラベル', f"±{self.threshold*100:.1f}%以上", 'ノイズ除去'],
            ['2. データ量拡張', f'{self.years_back}年分', '市場局面カバー'],
            ['3. 信頼度フィルタ', f'{self.confidence_threshold:.2f}', '精度優先'],
        ]

        table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.3, 0.2, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # ヘッダー
        for i in range(3):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # データ行
        for i in range(1, len(summary_data)):
            for j in range(3):
                if i == 6:  # 空行
                    continue
                if i == 7:  # 改善策ヘッダー
                    table[(i, j)].set_facecolor('#2ecc71')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                elif i == 1:  # 精度行を強調
                    table[(i, j)].set_facecolor('#ffffcc')
                    table[(i, j)].set_text_props(weight='bold')
                elif i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        # タイトル
        fig.suptitle(f'Phase 1.8 Enhanced - 最強モデル結果\n目標精度85-90% | 達成精度{metrics["Accuracy"]:.2f}%',
                    fontsize=16, fontweight='bold', y=0.98)

        # 保存
        output_path = self.output_dir / "phase1_8_enhanced_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"\nグラフ保存: {output_path}")

        return output_path

    def run(self):
        """
        Phase 1.8 Enhanced 実行
        """
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1.8 Enhanced - 最強モデル 実行開始")
        logger.info("=" * 70)
        logger.info(f"目標精度: 85-90%")
        logger.info(f"改善策: 閾値ラベル + 10年データ + 信頼度フィルタ")
        logger.info("=" * 70 + "\n")

        # 1. データ取得
        df = self.fetch_all_data()

        # 2. 特徴量とラベル作成
        X, y, returns = self.prepare_features_and_labels(df)

        # 3. 特徴量選択
        top_features, importances = self.select_top_features(X, y, self.top_features)
        X_selected = X[top_features]

        # 4. データ分割
        train_size = int(len(X_selected) * 0.70)
        val_size = int(len(X_selected) * 0.15)

        X_train = X_selected.iloc[:train_size].values
        y_train = y.iloc[:train_size].values

        X_val = X_selected.iloc[train_size:train_size+val_size].values
        y_val = y.iloc[train_size:train_size+val_size].values

        X_test = X_selected.iloc[train_size+val_size:].values
        y_test = y.iloc[train_size+val_size:].values

        logger.info(f"\nデータ分割:")
        logger.info(f"  訓練: {len(X_train)}件 (70%)")
        logger.info(f"  検証: {len(X_val)}件 (15%)")
        logger.info(f"  テスト: {len(X_test)}件 (15%)")

        # 5. モデル訓練
        models, val_accuracies = self.train_models(X_train, y_train, X_val, y_val)

        # 6. モデル重み計算
        total_acc = sum(val_accuracies.values())
        model_weights = {name: acc/total_acc for name, acc in val_accuracies.items()}

        logger.info("\nモデル重み:")
        for name, weight in model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")

        # 7. テスト予測（信頼度フィルタ適用）
        logger.info("\n" + "=" * 70)
        logger.info("テストセット予測（信頼度フィルタ適用）")
        logger.info("=" * 70)

        y_pred, probabilities = self.predict_with_confidence(models, model_weights, X_test, y_test)

        # 8. 評価指標計算
        metrics = self.calculate_metrics(y_test, y_pred, probabilities)

        # 9. 結果表示
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1.8 Enhanced 最終結果")
        logger.info("=" * 70)
        logger.info(f"方向性的中率:     {metrics['Accuracy']:.2f}%  ⭐")
        logger.info(f"カバー率:         {metrics['Coverage']:.2f}%")
        logger.info(f"見送り:           {metrics['Skipped']}件")
        logger.info(f"上昇的中精度:     {metrics['Precision_Up']:.2f}%")
        logger.info(f"下降的中精度:     {metrics['Precision_Down']:.2f}%")
        logger.info(f"上昇再現率:       {metrics['Recall_Up']:.2f}%")
        logger.info(f"下降再現率:       {metrics['Recall_Down']:.2f}%")
        logger.info("=" * 70)

        # 10. 可視化
        graph_path = self.create_visualizations(y_test, y_pred, probabilities, metrics, model_weights)

        # テストデータのインデックスを取得
        test_indices = X_selected.iloc[train_size+val_size:].index

        # 結果返却
        results = {
            'metrics': metrics,
            'model_weights': model_weights,
            'selected_features': top_features,
            'feature_importances': importances,
            'graph_path': str(graph_path),
            'predictions': y_pred,
            'probabilities': probabilities,
            'actual_labels': y_test,
            'test_indices': test_indices,
            'X_test': X_test,
        }

        return results
