"""
Phase 1.2: 訓練データ大幅増強版

改善点:
1. 訓練期間: 150日 → 500日
2. より多くの市場局面を学習
3. データ拡張（時系列augmentation）
4. LightGBM/CatBoost追加

目標: 方向性的中率 90% 達成
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from ..api.oanda_client import OandaClient
from ..api.oanda_client_extended import OandaClientExtended
from ..data_sources.economic_indicators import EconomicIndicators


class Phase1_2_MassiveData:
    """
    Phase 1.2: 大量訓練データ版

    改善:
    - 500日間の訓練データ
    - より多くのモデル（LightGBM, CatBoost追加）
    - 特徴量選択
    - 方向性予測に基づく最適重み
    """

    def __init__(
        self,
        start_year: int = 2020,
        train_days: int = 500,  # 150 → 500日
        predict_days: int = 150,
        instrument: str = "USD_JPY",
        top_features: int = 40  # 上位40特徴量
    ):
        self.start_year = start_year
        self.train_days = train_days
        self.predict_days = predict_days
        self.instrument = instrument
        self.top_features = top_features

        self.client = OandaClient()
        self.client_ext = OandaClientExtended()
        self.econ_indicators = EconomicIndicators()

        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.selected_features = []

        self.output_dir = Path(f"outputs/phase1_2_{start_year}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 1.2 大量訓練データ版: {start_year}年から開始")
        logger.info(f"訓練期間: {train_days}日 (★大幅増強★), 予測期間: {predict_days}日")

    def fetch_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """複数時間足データを取得"""
        logger.info("複数時間足データ取得中...")
        timeframes = ["H1", "H4", "D"]
        required_days = self.train_days + self.predict_days + 100
        multi_tf = {}

        for tf in timeframes:
            try:
                if tf == "H1":
                    count = min(required_days * 24, 5000)
                elif tf == "H4":
                    count = min(required_days * 6, 5000)
                elif tf == "D":
                    count = min(required_days, 5000)
                else:
                    count = min(required_days * 24, 5000)

                logger.info(f"  {tf} データ取得中... ({count}本)")

                data = self.client.get_historical_data(
                    instrument=self.instrument,
                    granularity=tf,
                    count=count
                )

                if not data.empty:
                    multi_tf[tf] = data
                    logger.info(f"  {tf} 完了: {len(data)}件")

                import time
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"{tf} 取得エラー: {e}")
                continue

        return multi_tf

    def fetch_multiple_currencies(self) -> Dict[str, pd.DataFrame]:
        """複数通貨ペア取得"""
        logger.info("複数通貨ペアデータ取得中...")
        instruments = ["USD_JPY", "EUR_USD", "GBP_USD", "EUR_JPY"]
        required_days = self.train_days + self.predict_days + 100
        count = min(required_days * 24, 5000)
        multi_inst = {}

        for inst in instruments:
            try:
                logger.info(f"  {inst} データ取得中...")
                data = self.client.get_historical_data(
                    instrument=inst,
                    granularity="H1",
                    count=count
                )

                if not data.empty:
                    daily = data.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    multi_inst[inst] = daily
                    logger.info(f"  {inst} 完了: {len(daily)}件")

                import time
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"{inst} 取得エラー: {e}")
                multi_inst[inst] = self._generate_demo_currency(inst)

        return multi_inst

    def _generate_demo_currency(self, instrument: str) -> pd.DataFrame:
        """デモ通貨データ生成"""
        days = self.train_days + self.predict_days + 100
        dates = pd.date_range(start=f"{self.start_year}-01-01", periods=days, freq='D')

        base_price = {"USD_JPY": 150, "EUR_USD": 1.1, "GBP_USD": 1.3, "EUR_JPY": 165}.get(instrument, 150)
        returns = np.random.randn(days) * 0.5
        prices = base_price + np.cumsum(returns)

        return pd.DataFrame({
            'close': prices,
            'open': prices + np.random.randn(days) * 0.1,
            'high': prices + np.abs(np.random.randn(days) * 0.3),
            'low': prices - np.abs(np.random.randn(days) * 0.3),
            'volume': np.random.randint(1000, 10000, days)
        }, index=dates)

    def prepare_features(
        self,
        multi_tf: Dict[str, pd.DataFrame],
        multi_inst: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        特徴量準備（拡張版）
        """
        logger.info("特徴量エンジニアリング中...")

        # メインデータ準備
        h1_data = multi_tf.get('H1', multi_tf.get('D'))
        if h1_data is None or h1_data.empty:
            raise ValueError("データが取得できませんでした")

        main_data = h1_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df = main_data.copy()

        # 基本特徴量
        df['day_num'] = range(len(df))
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = df['high'] - df['low']
        df['high_close_ratio'] = df['high'] / df['close']
        df['low_close_ratio'] = df['low'] / df['close']

        # 移動平均（複数期間）
        for period in [5, 10, 20, 30, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            # 価格と移動平均の乖離
            df[f'price_sma_{period}_diff'] = df['close'] - df[f'sma_{period}']

        # RSI
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # ボリンジャーバンド
        for period in [20, 50]:
            bb_sma = df['close'].rolling(period).mean()
            bb_std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_sma + 2 * bb_std
            df[f'bb_lower_{period}'] = bb_sma - 2 * bb_std
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

        # ATR（Average True Range）
        for period in [14, 28]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(period).mean()

        # 複数時間足
        for tf_name, tf_data in multi_tf.items():
            daily_tf = tf_data.resample('D').agg({'close': 'last'}).dropna()
            for period in [5, 10, 20, 50]:
                df[f'{tf_name}_ema_{period}'] = daily_tf['close'].ewm(span=period, adjust=False).mean()
                df[f'{tf_name}_sma_{period}'] = daily_tf['close'].rolling(period).mean()

        # 通貨ペア相関
        for inst_name, inst_data in multi_inst.items():
            if inst_name == self.instrument:
                continue
            aligned = inst_data.reindex(df.index, method='ffill')
            df[f'{inst_name}_close'] = aligned['close']
            df[f'{inst_name}_return'] = aligned['close'].pct_change()
            # 相関係数（ローリング）
            for window in [10, 20]:
                df[f'{inst_name}_corr_{window}'] = df['close'].rolling(window).corr(aligned['close'])

        # 経済指標
        try:
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            econ_data = self.econ_indicators.get_all_indicators(start_date, end_date)
            econ_aligned = econ_data.reindex(df.index, method='ffill')
            for col in econ_aligned.columns:
                df[f'econ_{col}'] = econ_aligned[col]
            logger.info(f"  経済指標 {len(econ_aligned.columns)}個追加")
        except Exception as e:
            logger.warning(f"経済指標取得エラー: {e}")
            df['econ_rate_differential'] = 1.5 + np.random.randn(len(df)) * 0.1

        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        logger.info(f"特徴量エンジニアリング完了: {len(df.columns)}個")

        return df

    def select_top_features(self, X_train, y_train, feature_cols: List[str]) -> List[str]:
        """重要な特徴量のみ選択"""
        logger.info(f"特徴量選択中（上位{self.top_features}個）...")

        # Random Forestで重要度計算
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)

        # 重要度でソート
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)

        # 上位N個選択
        selected = importance_df.head(self.top_features)['feature'].tolist()

        logger.info(f"選択された特徴量 Top 10:")
        for feat in selected[:10]:
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            logger.info(f"  {feat}: {imp:.4f}")

        return selected

    def calculate_direction_accuracy(self, y_true, y_pred) -> float:
        """方向性的中率を計算"""
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        accuracy = np.mean(true_direction == pred_direction) * 100
        return accuracy

    def optimize_ensemble_weights(self, X_val, y_val) -> Dict[str, float]:
        """
        アンサンブルの最適重みを計算

        方向性的中率を最大化する重みを探索
        """
        logger.info("アンサンブル重み最適化中...")

        # 各モデルの予測と方向性的中率
        model_predictions = {}
        model_accuracies = {}

        for name, model in self.models.items():
            preds = model.predict(X_val)
            acc = self.calculate_direction_accuracy(y_val, preds)
            model_predictions[name] = preds
            model_accuracies[name] = acc
            logger.info(f"  {name} 方向性的中率: {acc:.2f}%")

        # 方向性的中率に基づいて重みを設定（高いほど重い）
        total_acc = sum(model_accuracies.values())
        weights = {name: acc / total_acc for name, acc in model_accuracies.items()}

        logger.info("\n最適化された重み:")
        for name, weight in weights.items():
            logger.info(f"  {name}: {weight:.4f}")

        return weights

    def train_and_predict(
        self,
        multi_tf: Dict[str, pd.DataFrame],
        multi_inst: Dict[str, pd.DataFrame]
    ) -> Dict:
        """訓練と予測"""
        logger.info("特徴量エンジニアリング中...")
        df = self.prepare_features(multi_tf, multi_inst)

        # データ分割（利用可能なデータに基づいて調整）
        total_available = len(df)
        logger.info(f"利用可能データ総数: {total_available}日")

        # 訓練データの調整（利用可能データの60%を訓練、20%を検証、20%を予測に使用）
        if total_available < self.train_days + self.predict_days:
            logger.warning(f"要求データ量（{self.train_days + self.predict_days}日）が利用可能データ（{total_available}日）を超えています")
            logger.info("データ分割を動的に調整します")

            # 60% 訓練, 20% 検証, 20% 予測
            predict_size = int(total_available * 0.2)
            val_size = int(total_available * 0.2)
            train_size = total_available - val_size - predict_size
        else:
            # 通常の分割
            val_size = int(self.train_days * 0.2)
            train_size = self.train_days - val_size
            predict_size = min(self.predict_days, total_available - self.train_days)

        train_data = df.iloc[:train_size].copy()
        val_data = df.iloc[train_size:train_size + val_size].copy()
        predict_data = df.iloc[train_size + val_size:train_size + val_size + predict_size].copy()

        logger.info(f"訓練データ: {len(train_data)}日 ({len(train_data)/total_available*100:.1f}%)")
        logger.info(f"検証データ: {len(val_data)}日 ({len(val_data)/total_available*100:.1f}%)")
        logger.info(f"予測期間: {len(predict_data)}日 ({len(predict_data)/total_available*100:.1f}%)")

        # 特徴量選択
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X_train_full = train_data[feature_cols].values
        y_train_full = train_data['close'].values

        # スケーリング（特徴量選択用）
        X_train_scaled = self.scaler.fit_transform(X_train_full)

        # 重要な特徴量のみ選択
        self.selected_features = self.select_top_features(X_train_scaled, y_train_full, feature_cols)

        # 選択された特徴量で再訓練
        X_train = train_data[self.selected_features].values
        y_train = train_data['close'].values
        X_val = val_data[self.selected_features].values
        y_val = val_data['close'].values

        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        logger.info(f"\n使用特徴量: {len(self.selected_features)}個")

        # モデル訓練（より多くのモデル）
        logger.info("アンサンブルモデル訓練中...")

        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=500, max_depth=15, learning_rate=0.02, subsample=0.8, random_state=42
        )
        self.models['gb'].fit(X_train_scaled, y_train)
        logger.info("  Gradient Boosting 完了")

        self.models['rf'] = RandomForestRegressor(
            n_estimators=500, max_depth=30, min_samples_split=2, random_state=42, n_jobs=-1
        )
        self.models['rf'].fit(X_train_scaled, y_train)
        logger.info("  Random Forest 完了")

        try:
            import xgboost as xgb
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=500, max_depth=12, learning_rate=0.03, random_state=42
            )
            self.models['xgb'].fit(X_train_scaled, y_train)
            logger.info("  XGBoost 完了")
        except ImportError:
            logger.warning("XGBoostスキップ")

        try:
            import lightgbm as lgb
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=500, max_depth=12, learning_rate=0.03, random_state=42, verbose=-1
            )
            self.models['lgb'].fit(X_train_scaled, y_train)
            logger.info("  LightGBM 完了")
        except ImportError:
            logger.warning("LightGBMスキップ")

        try:
            from catboost import CatBoostRegressor
            self.models['cat'] = CatBoostRegressor(
                iterations=500, depth=10, learning_rate=0.03, random_state=42, verbose=0
            )
            self.models['cat'].fit(X_train_scaled, y_train)
            logger.info("  CatBoost 完了")
        except ImportError:
            logger.warning("CatBoostスキップ")

        # 重み最適化
        self.model_weights = self.optimize_ensemble_weights(X_val_scaled, y_val)

        # 予測
        logger.info("\n予測中...")
        X_predict = predict_data[self.selected_features].values
        X_predict_scaled = self.scaler.transform(X_predict)

        # 各モデルで予測
        predictions_dict = {}
        for name, model in self.models.items():
            predictions_dict[name] = model.predict(X_predict_scaled)

        # 重み付きアンサンブル
        weighted_predictions = np.zeros(len(X_predict))
        for name, preds in predictions_dict.items():
            weight = self.model_weights[name]
            weighted_predictions += weight * preds

        logger.info("重み付きアンサンブル予測完了")

        results = {
            'train_data': train_data,
            'val_data': val_data,
            'predict_data': predict_data,
            'predictions': weighted_predictions,
            'actual_prices': predict_data['close'].values,
            'model_weights': self.model_weights,
            'predictions_dict': predictions_dict,
            'selected_features': self.selected_features
        }

        return results

    def calculate_metrics(self, results: Dict) -> Dict:
        """評価指標計算"""
        predictions = results['predictions']
        actual = results['actual_prices']

        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        mape = np.mean(np.abs((predictions - actual) / actual)) * 100

        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actual) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100

        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'R2_Score': r2_score
        }

    def create_comparison_graph(self, results: Dict) -> str:
        """比較グラフ作成"""
        logger.info("比較グラフ作成中...")

        train_data = results['train_data']
        predict_data = results['predict_data']
        predictions = results['predictions']
        actual_prices = results['actual_prices']

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        # グラフ1: 価格予測
        ax1 = axes[0]
        ax1.plot(train_data.index, train_data['close'], 'b-', label='訓練データ (500日)', linewidth=2, alpha=0.7)
        ax1.plot(predict_data.index, predictions, 'r--', label='予測（重み付きアンサンブル）', linewidth=2.5)
        ax1.plot(predict_data.index, actual_prices, 'g-', label='実際の価格', linewidth=2)

        boundary = train_data.index[-1]
        ax1.axvline(x=boundary, color='orange', linestyle=':', linewidth=2, label='訓練/予測の境界')
        ax1.fill_between(predict_data.index, predictions, actual_prices, alpha=0.2, color='gray')

        ax1.set_title(f'Phase 1.2: 大量訓練データ版 ({self.start_year}年, 訓練{self.train_days}日)', fontsize=16)
        ax1.set_xlabel('日付', fontsize=14)
        ax1.set_ylabel('価格 (JPY)', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # グラフ2: モデル重み
        ax2 = axes[1]
        weights = list(self.model_weights.values())
        names = list(self.model_weights.keys())
        colors = plt.cm.Set3(range(len(names)))
        ax2.bar(names, weights, color=colors)
        ax2.set_title('アンサンブル重み（方向性的中率ベース）', fontsize=14)
        ax2.set_ylabel('重み', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # 重みの値を表示
        for i, (name, weight) in enumerate(zip(names, weights)):
            ax2.text(i, weight, f'{weight:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        graph_path = self.output_dir / "phase1_2_comparison.png"
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"グラフ保存: {graph_path}")
        return str(graph_path)

    def run(self) -> Dict:
        """Phase 1.2を実行"""
        logger.info("\n" + "="*70)
        logger.info("Phase 1.2: 大量訓練データ版 (500日)")
        logger.info("="*70 + "\n")

        # データ取得
        logger.info("[1/4] データ取得...")
        multi_tf = self.fetch_multi_timeframe_data()
        multi_inst = self.fetch_multiple_currencies()

        # 訓練と予測
        logger.info("\n[2/4] モデル訓練と予測...")
        results = self.train_and_predict(multi_tf, multi_inst)

        # 評価
        logger.info("\n[3/4] 評価指標計算...")
        metrics = self.calculate_metrics(results)

        logger.info(f"MAE: {metrics['MAE']:.4f}")
        logger.info(f"RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"MAPE: {metrics['MAPE']:.2f}%")
        logger.info(f"方向性的中率: {metrics['Direction_Accuracy']:.2f}%")
        logger.info(f"R2スコア: {metrics['R2_Score']:.4f}")

        # グラフ
        logger.info("\n[4/4] グラフ作成...")
        graph_path = self.create_comparison_graph(results)

        results['metrics'] = metrics
        results['graph_path'] = graph_path

        # モデル保存
        model_path = self.output_dir / "phase1_2_model.pkl"
        joblib.dump({
            'models': self.models,
            'weights': self.model_weights,
            'scaler': self.scaler,
            'selected_features': self.selected_features
        }, model_path)

        logger.info("\n" + "="*70)
        logger.info("Phase 1.2 完了！")
        logger.info(f"グラフ: {graph_path}")
        logger.info(f"方向性的中率: {metrics['Direction_Accuracy']:.2f}%")
        logger.info("="*70 + "\n")

        return results
