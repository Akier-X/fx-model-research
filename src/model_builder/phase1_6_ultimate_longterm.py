"""
Phase 1.6: 究極の長期データ版（最終版）

データソース:
- Yahoo Finance: 3年以上の長期データ（782日+）
- OANDA API: リアルタイムデータ
- FRED API: 経済指標
- 株価指数: S&P500, 日経, VIX

目標: 方向性的中率 90%+ 達成
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from loguru import logger
from typing import Tuple, Dict, List
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from ..data_sources.yahoo_finance import YahooFinanceData
from ..data_sources.economic_indicators import EconomicIndicators


class Phase1_6_UltimateLongTerm:
    """
    Phase 1.6: 究極の長期データモデル

    特徴:
    - 3年以上の訓練データ
    - 複数データソース統合
    - 分類モデル（方向予測特化）
    - クロスバリデーション
    """

    def __init__(
        self,
        years_back: int = 3,
        instrument: str = "USD/JPY",
        top_features: int = 60,
        lookahead_days: int = 1
    ):
        self.years_back = years_back
        self.instrument = instrument
        self.top_features = top_features
        self.lookahead_days = lookahead_days

        # データソース初期化
        self.yf_data = YahooFinanceData()
        self.econ_indicators = EconomicIndicators()

        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.selected_features = []

        self.output_dir = Path(f"outputs/phase1_6_ultimate")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*70)
        logger.info(f"Phase 1.6: 究極の長期データモデル")
        logger.info("="*70)
        logger.info(f"訓練データ期間: 過去{years_back}年")
        logger.info(f"データソース: Yahoo Finance (長期), FRED (経済指標)")
        logger.info(f"目標: 方向性的中率 90%+")

    def fetch_all_data(self) -> pd.DataFrame:
        """
        全データソースから統合データ取得
        """
        logger.info("[データ取得] Yahoo Finance + 経済指標")

        # 期間設定（過去N年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years_back * 365 + 30)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"期間: {start_str} ~ {end_str}")

        # Yahoo Financeからメイン通貨ペア取得
        main_data = self.yf_data.get_forex_data(
            pair=self.instrument,
            start_date=start_str,
            end_date=end_str
        )

        if main_data.empty:
            raise ValueError(f"{self.instrument} データが取得できませんでした")

        logger.info(f"{self.instrument}: {len(main_data)}件取得")

        # 複数通貨ペア取得
        pairs = ["EUR/USD", "GBP/USD", "EUR/JPY"]
        multi_pairs = self.yf_data.get_multiple_pairs(pairs, start_str, end_str)

        # 株価指数取得
        indices = self.yf_data.get_stock_indices(start_str, end_str)

        # 経済指標取得
        try:
            econ_data = self.econ_indicators.get_all_indicators(start_str, end_str)
            logger.info(f"経済指標: {len(econ_data.columns)}個取得")
        except Exception as e:
            logger.warning(f"経済指標取得エラー: {e}")
            econ_data = pd.DataFrame()

        return main_data, multi_pairs, indices, econ_data

    def prepare_features_and_labels(
        self,
        main_data: pd.DataFrame,
        multi_pairs: dict,
        indices: dict,
        econ_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特徴量エンジニアリング + ラベル作成
        """
        logger.info("特徴量エンジニアリング開始...")

        df = main_data.copy()

        # === 基本特徴量 ===
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = df['high'] - df['low']
        df['high_close_ratio'] = df['high'] / (df['close'] + 1e-10)
        df['low_close_ratio'] = df['low'] / (df['close'] + 1e-10)
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day

        # === 移動平均 ===
        for period in [5, 10, 20, 30, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_sma_{period}_diff'] = df['close'] - df[f'sma_{period}']
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5)
            df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(5)

        # === RSI ===
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{period}_slope'] = df[f'rsi_{period}'].diff(3)

        # === MACD ===
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['macd_slope'] = df['macd'].diff(3)

        # === ボリンジャーバンド ===
        for period in [20, 50]:
            bb_sma = df['close'].rolling(period).mean()
            bb_std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_sma + 2 * bb_std
            df[f'bb_lower_{period}'] = bb_sma - 2 * bb_std
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']

        # === ATR ===
        for period in [14, 28]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(period).mean()

        # === モメンタム ===
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['close'].diff(period)
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100

        # === ストキャスティクス ===
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)

        # === 複数通貨ペア ===
        for pair_name, pair_data in multi_pairs.items():
            aligned = pair_data.reindex(df.index, method='ffill')
            clean_name = pair_name.replace('/', '_')
            df[f'{clean_name}_close'] = aligned['close']
            df[f'{clean_name}_return'] = aligned['close'].pct_change()
            df[f'{clean_name}_sma_20'] = aligned['close'].rolling(20).mean()
            # 相関
            for window in [10, 20]:
                df[f'{clean_name}_corr_{window}'] = df['close'].rolling(window).corr(aligned['close'])

        # === 株価指数 ===
        for idx_name, idx_data in indices.items():
            aligned = idx_data.reindex(df.index, method='ffill')
            df[f'{idx_name}_close'] = aligned['close']
            df[f'{idx_name}_return'] = aligned['close'].pct_change()
            df[f'{idx_name}_sma_20'] = aligned['close'].rolling(20).mean()

        # === 経済指標 ===
        if not econ_data.empty:
            econ_aligned = econ_data.reindex(df.index, method='ffill')
            for col in econ_aligned.columns:
                df[f'econ_{col}'] = econ_aligned[col]
            logger.info(f"  経済指標 {len(econ_aligned.columns)}個追加")

        # === ラベル作成（方向） ===
        future_price = df['close'].shift(-self.lookahead_days)
        df['future_direction'] = (future_price > df['close']).astype(int)

        # NaN除去
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        df = df[df['future_direction'].notna()]

        logger.info(f"特徴量エンジニアリング完了: {len(df.columns)}列")
        logger.info(f"データ件数: {len(df)}")
        logger.info(f"ラベル分布: 上昇={df['future_direction'].sum()} ({df['future_direction'].mean()*100:.1f}%), 下降={(1-df['future_direction']).sum()} ({(1-df['future_direction'].mean())*100:.1f}%)")

        # 特徴量とラベルを分離
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_direction']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df['future_direction']

        return X, y

    def select_top_features(self, X_train, y_train) -> List[str]:
        """重要特徴量選択"""
        logger.info(f"特徴量選択中（上位{self.top_features}個）...")

        from sklearn.ensemble import RandomForestClassifier
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)

        selected = importance_df.head(self.top_features)['feature'].tolist()

        logger.info(f"選択された特徴量 Top 15:")
        for i, feat in enumerate(selected[:15], 1):
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            logger.info(f"  {i}. {feat}: {imp:.4f}")

        return selected

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        訓練と評価

        データ分割: 70% train, 15% val, 15% test
        """
        logger.info("データ分割中...")

        total_len = len(X)
        train_size = int(total_len * 0.70)
        val_size = int(total_len * 0.15)

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]

        logger.info(f"訓練データ: {len(X_train)}件 ({len(X_train)/total_len*100:.1f}%)")
        logger.info(f"検証データ: {len(X_val)}件 ({len(X_val)/total_len*100:.1f}%)")
        logger.info(f"テストデータ: {len(X_test)}件 ({len(X_test)/total_len*100:.1f}%)")

        # 特徴量選択
        self.selected_features = self.select_top_features(X_train, y_train)
        X_train_selected = X_train[self.selected_features]
        X_val_selected = X_val[self.selected_features]
        X_test_selected = X_test[self.selected_features]

        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)

        logger.info(f"\n使用特徴量: {len(self.selected_features)}個")

        # モデル訓練
        logger.info("分類モデル訓練中...")

        # Gradient Boosting
        self.models['gbc'] = GradientBoostingClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, random_state=42
        )
        self.models['gbc'].fit(X_train_scaled, y_train)
        logger.info("  Gradient Boosting Classifier 完了")

        # Random Forest
        self.models['rfc'] = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1
        )
        self.models['rfc'].fit(X_train_scaled, y_train)
        logger.info("  Random Forest Classifier 完了")

        # XGBoost
        try:
            import xgboost as xgb
            self.models['xgbc'] = xgb.XGBClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.05, random_state=42
            )
            self.models['xgbc'].fit(X_train_scaled, y_train)
            logger.info("  XGBoost Classifier 完了")
        except ImportError:
            logger.warning("XGBoostスキップ")

        # LightGBM
        try:
            import lightgbm as lgb
            self.models['lgbc'] = lgb.LGBMClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.05, random_state=42, verbose=-1
            )
            self.models['lgbc'].fit(X_train_scaled, y_train)
            logger.info("  LightGBM Classifier 完了")
        except ImportError:
            logger.warning("LightGBMスキップ")

        # CatBoost
        try:
            from catboost import CatBoostClassifier
            self.models['catc'] = CatBoostClassifier(
                iterations=500, depth=10, learning_rate=0.05, random_state=42, verbose=0
            )
            self.models['catc'].fit(X_train_scaled, y_train)
            logger.info("  CatBoost Classifier 完了")
        except ImportError:
            logger.warning("CatBoostスキップ")

        # 検証データで重み計算
        logger.info("\n検証データでの各モデル精度:")
        model_accuracies = {}
        for name, model in self.models.items():
            y_val_pred = model.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_val_pred) * 100
            model_accuracies[name] = acc
            logger.info(f"  {name}: {acc:.2f}%")

        # 重み計算
        total_acc = sum(model_accuracies.values())
        self.model_weights = {name: acc / total_acc for name, acc in model_accuracies.items()}

        logger.info("\nアンサンブル重み:")
        for name, weight in self.model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")

        # テストデータで予測
        logger.info("\nテストデータで最終評価...")
        predictions_proba = {}
        for name, model in self.models.items():
            proba = model.predict_proba(X_test_scaled)[:, 1]
            predictions_proba[name] = proba

        # 重み付き確率平均
        weighted_proba = np.zeros(len(X_test))
        for name, proba in predictions_proba.items():
            weighted_proba += self.model_weights[name] * proba

        final_predictions = (weighted_proba >= 0.5).astype(int)

        return {
            'y_test': y_test,
            'predictions': final_predictions,
            'predictions_proba': weighted_proba,
            'model_accuracies': model_accuracies,
            'model_weights': self.model_weights,
            'selected_features': self.selected_features,
            'X_test_index': X_test.index,
            'X_train_index': X_train.index,
            'X_val_index': X_val.index
        }

    def calculate_metrics(self, results: Dict) -> Dict:
        """評価指標計算"""
        y_true = results['y_test'].values
        y_pred = results['predictions']

        accuracy = accuracy_score(y_true, y_pred) * 100
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_up = 2 * precision_up * recall_up / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0

        precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_down = 2 * precision_down * recall_down / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0

        logger.info("\n混同行列:")
        logger.info(f"              予測下降  予測上昇")
        logger.info(f"実際下降:      {tn:4d}    {fp:4d}")
        logger.info(f"実際上昇:      {fn:4d}    {tp:4d}")

        return {
            'Accuracy': accuracy,
            'True_Positives': int(tp),
            'True_Negatives': int(tn),
            'False_Positives': int(fp),
            'False_Negatives': int(fn),
            'Precision_Up': precision_up * 100,
            'Recall_Up': recall_up * 100,
            'F1_Up': f1_up * 100,
            'Precision_Down': precision_down * 100,
            'Recall_Down': recall_down * 100,
            'F1_Down': f1_down * 100
        }

    def create_visualization(self, results: Dict, metrics: Dict) -> str:
        """結果可視化"""
        logger.info("結果グラフ作成中...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # グラフ1: 累積精度推移
        ax1 = axes[0, 0]
        y_true = results['y_test'].values
        y_pred = results['predictions']
        cumulative_accuracy = [np.mean(y_true[:i+1] == y_pred[:i+1]) * 100 for i in range(len(y_true))]
        ax1.plot(cumulative_accuracy, linewidth=2, color='blue')
        ax1.axhline(y=90, color='r', linestyle='--', linewidth=2, label='目標 90%')
        ax1.axhline(y=metrics['Accuracy'], color='g', linestyle='--', linewidth=2, label=f'最終精度 {metrics["Accuracy"]:.2f}%')
        ax1.set_title('累積精度の推移', fontsize=14, fontweight='bold')
        ax1.set_xlabel('予測数', fontsize=12)
        ax1.set_ylabel('累積精度 (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # グラフ2: モデル別検証精度
        ax2 = axes[0, 1]
        model_names = list(results['model_accuracies'].keys())
        accuracies = list(results['model_accuracies'].values())
        colors = plt.cm.Set3(range(len(model_names)))
        bars = ax2.bar(model_names, accuracies, color=colors)
        ax2.axhline(y=90, color='r', linestyle='--', linewidth=2, label='目標 90%')
        ax2.set_title('各モデルの検証精度', fontsize=14, fontweight='bold')
        ax2.set_ylabel('精度 (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # グラフ3: 混同行列
        ax3 = axes[1, 0]
        cm = np.array([[metrics['True_Negatives'], metrics['False_Positives']],
                      [metrics['False_Negatives'], metrics['True_Positives']]])
        im = ax3.imshow(cm, cmap='Blues', aspect='auto')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['予測: 下降', '予測: 上昇'], fontsize=11)
        ax3.set_yticklabels(['実際: 下降', '実際: 上昇'], fontsize=11)
        ax3.set_title('混同行列', fontsize=14, fontweight='bold')

        for i in range(2):
            for j in range(2):
                text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=18, color=text_color, fontweight='bold')

        plt.colorbar(im, ax=ax3)

        # グラフ4: 予測確率分布
        ax4 = axes[1, 1]
        proba_up = results['predictions_proba']
        ax4.hist(proba_up[results['y_test'] == 1], bins=25, alpha=0.6, label='実際上昇', color='green', edgecolor='black')
        ax4.hist(proba_up[results['y_test'] == 0], bins=25, alpha=0.6, label='実際下降', color='red', edgecolor='black')
        ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='判定閾値')
        ax4.set_title('予測確率の分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('上昇確率', fontsize=12)
        ax4.set_ylabel('頻度', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        graph_path = self.output_dir / "phase1_6_ultimate_results.png"
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"グラフ保存: {graph_path}")
        return str(graph_path)

    def run(self) -> Dict:
        """Phase 1.6 実行"""
        logger.info("\n" + "="*70)
        logger.info("Phase 1.6: 究極の長期データモデル")
        logger.info("="*70 + "\n")

        # データ取得
        logger.info("[1/4] 全データソース統合取得...")
        main_data, multi_pairs, indices, econ_data = self.fetch_all_data()

        # 特徴量準備
        logger.info("\n[2/4] 特徴量エンジニアリング...")
        X, y = self.prepare_features_and_labels(main_data, multi_pairs, indices, econ_data)

        # 訓練と評価
        logger.info("\n[3/4] モデル訓練と評価...")
        results = self.train_and_evaluate(X, y)

        # 評価指標
        logger.info("\n[4/4] 最終評価...")
        metrics = self.calculate_metrics(results)

        logger.info(f"\n方向性的中率: {metrics['Accuracy']:.2f}%")
        logger.info(f"上昇的中精度: {metrics['Precision_Up']:.2f}%")
        logger.info(f"下降的中精度: {metrics['Precision_Down']:.2f}%")

        # グラフ作成
        logger.info("\nグラフ作成...")
        graph_path = self.create_visualization(results, metrics)

        results['metrics'] = metrics
        results['graph_path'] = graph_path

        # モデル保存
        model_path = self.output_dir / "phase1_6_ultimate_model.pkl"
        joblib.dump({
            'models': self.models,
            'weights': self.model_weights,
            'scaler': self.scaler,
            'selected_features': self.selected_features
        }, model_path)

        logger.info("\n" + "="*70)
        logger.info("Phase 1.6 完了！")
        logger.info(f"方向性的中率: {metrics['Accuracy']:.2f}%")
        logger.info(f"グラフ: {graph_path}")
        logger.info("="*70 + "\n")

        return results
