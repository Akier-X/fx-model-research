"""
ハイブリッドシステム用モデル訓練・保存スクリプト

Phase 1.8（93.64%方向予測）とPhase 2（Sharpe 4.07収益予測）の
両方を訓練し、ハイブリッドシステムで使用できるように保存
"""

from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import sys
import joblib
from typing import Dict, Tuple

from src.data_sources.yahoo_finance import YahooFinanceData
from src.data_sources.economic_indicators import EconomicIndicators
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor


class HybridModelTrainer:
    """ハイブリッドシステム用モデル訓練"""

    def __init__(self):
        self.yahoo = YahooFinanceData()
        self.fred = EconomicIndicators()

        # デフォルト通貨ペア
        self.instrument = 'USD/JPY'

        # 保存ディレクトリ
        self.phase1_8_dir = Path('models/phase1_8')
        self.phase2_dir = Path('models/phase2')

        # ディレクトリ作成
        self.phase1_8_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("ハイブリッドシステム用モデル訓練・保存")
        logger.info("="*80)
        logger.info(f"Phase 1.8保存先: {self.phase1_8_dir}")
        logger.info(f"Phase 2保存先: {self.phase2_dir}")

    def run(self):
        """実行"""
        try:
            # 1. データ取得
            logger.info("\n1. データ取得中（10年分）...")
            data = self._get_data()

            if data.empty:
                logger.error("データ取得失敗")
                return False

            logger.info(f"✅ データ: {len(data)}日分")

            # 2. 特徴量生成
            logger.info("\n2. 特徴量生成中...")
            features_all, features_phase1 = self._generate_features(data)
            logger.info(f"✅ 特徴量（全体）: {features_all.shape}")
            logger.info(f"   特徴量（Phase 1.8用）: {features_phase1.shape}")

            # 3. Phase 1.8訓練・保存
            logger.info("\n3. Phase 1.8モデル訓練・保存（目標93.64%）...")
            phase1_results = self._train_and_save_phase1((features_all, features_phase1))

            # 4. Phase 2訓練・保存
            logger.info("\n4. Phase 2モデル訓練・保存（目標Sharpe 4+）...")
            phase2_results = self._train_and_save_phase2((features_all, features_phase1))

            # 5. 結果サマリー
            logger.info("\n5. 訓練結果サマリー")
            self._print_summary(phase1_results, phase2_results)

            return True

        except Exception as e:
            logger.error(f"エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _get_data(self):
        """データ取得（10年分）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3650)  # 10年

        data = self.yahoo.get_forex_data(
            pair='USD/JPY',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        return data

    def _generate_features(self, data):
        """特徴量生成（Phase 1.8 + Phase 2共通）"""
        df = data.copy()
        features = pd.DataFrame(index=df.index)

        # 基本価格
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']

        # 価格比率
        features['high_close_ratio'] = (df['high'] / df['close'] - 1) * 100
        features['low_close_ratio'] = (df['low'] / df['close'] - 1) * 100
        features['high_low_range'] = (df['high'] / df['low'] - 1) * 100

        # リターン
        for period in [1, 5, 10, 20]:
            features[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # SMA
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_vs_sma_{period}'] = ((df['close'] / sma) - 1) * 100

        # EMA
        for period in [12, 26]:
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema

        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26

        # ボラティリティ
        for period in [10, 20, 50]:
            features[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * 100

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()

        # ボリンジャーバンド
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_position'] = ((df['close'] - features['bb_lower']) /
                                   (features['bb_upper'] - features['bb_lower']) * 100)

        # ストキャスティクス
        for period in [14, 21]:
            lowest_low = df['low'].rolling(period).min()
            highest_high = df['high'].rolling(period).max()
            features[f'stoch_{period}'] = ((df['close'] - lowest_low) /
                                          (highest_high - lowest_low) * 100)

        # モメンタム
        for period in [10, 20]:
            features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # ROC
        for period in [10, 20]:
            features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                        df['close'].shift(period) * 100)

        # ラベル（Phase 1.8用: 閾値ベース方向）
        threshold = 0.5  # ±0.5%
        price_change = ((df['close'].shift(-1) / df['close']) - 1) * 100

        features['label_direction_raw'] = np.where(
            price_change > threshold, 1,
            np.where(price_change < -threshold, 0, -1)
        )

        # ラベル（Phase 2用: 実際のリターン）
        features['label_return'] = price_change

        # NaN除去
        features = features.dropna()

        # Phase 1.8用: 中立を除外
        features_phase1 = features[features['label_direction_raw'] != -1].copy()
        features_phase1['label_direction'] = features_phase1['label_direction_raw']

        return features, features_phase1

    def _train_and_save_phase1(self, data_tuple):
        """Phase 1.8訓練・保存"""
        features_all, features_phase1 = data_tuple

        logger.info(f"  サンプル数: {len(features_phase1)}件（±0.5%以上のみ）")

        # 特徴量とラベル
        feature_cols = [col for col in features_phase1.columns
                       if not col.startswith('label_')]

        X = features_phase1[feature_cols].values
        y = features_phase1['label_direction'].values

        # 分割（70/15/15）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, shuffle=False  # 0.15/0.85 ≈ 0.176
        )

        logger.info(f"  訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 時間重み
        sample_weights = np.array([0.95 ** i for i in range(len(X_train))])[::-1]

        # 5モデル訓練
        models = {}
        val_accuracies = {}

        # 1. GradientBoosting
        logger.info("  訓練中: GradientBoosting...")
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        gb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_acc = (gb.predict(X_val_scaled) == y_val).mean()
        models['gradient_boosting'] = gb
        val_accuracies['gradient_boosting'] = val_acc

        # 2. RandomForest
        logger.info("  訓練中: RandomForest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_acc = (rf.predict(X_val_scaled) == y_val).mean()
        models['random_forest'] = rf
        val_accuracies['random_forest'] = val_acc

        # 3. XGBoost
        logger.info("  訓練中: XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_acc = (xgb_model.predict(X_val_scaled) == y_val).mean()
        models['xgboost'] = xgb_model
        val_accuracies['xgboost'] = val_acc

        # 4. LightGBM
        logger.info("  訓練中: LightGBM...")
        lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
        lgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_acc = (lgb_model.predict(X_val_scaled) == y_val).mean()
        models['lightgbm'] = lgb_model
        val_accuracies['lightgbm'] = val_acc

        # 5. CatBoost
        logger.info("  訓練中: CatBoost...")
        cb = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=5, random_state=42, verbose=0)
        cb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        val_acc = (cb.predict(X_val_scaled) == y_val).mean()
        models['catboost'] = cb
        val_accuracies['catboost'] = val_acc

        # アンサンブル重み（検証精度ベース）
        total_acc = sum(val_accuracies.values())
        weights = {name: acc/total_acc for name, acc in val_accuracies.items()}

        logger.info("\n  モデル重み:")
        for name, weight in weights.items():
            logger.info(f"    {name}: {weight:.4f} (検証精度: {val_accuracies[name]*100:.2f}%)")

        # テストセット予測（信頼度フィルタリング）
        ensemble_probs = np.zeros((len(X_test_scaled), 2))
        for name, model in models.items():
            probs = model.predict_proba(X_test_scaled)
            ensemble_probs += probs * weights[name]

        # 信頼度フィルタ（0.65以上 or 0.35以下）
        confidence_threshold = 0.65
        max_probs = ensemble_probs.max(axis=1)
        confident_mask = max_probs >= confidence_threshold

        predictions = ensemble_probs.argmax(axis=1)
        confident_predictions = predictions[confident_mask]
        confident_labels = y_test[confident_mask]

        accuracy = (confident_predictions == confident_labels).mean() * 100
        coverage = confident_mask.sum() / len(y_test) * 100

        logger.info(f"\n✅ Phase 1.8結果:")
        logger.info(f"  方向性的中率: {accuracy:.2f}%")
        logger.info(f"  カバー率: {coverage:.2f}%")
        logger.info(f"  見送り: {(~confident_mask).sum()}件")

        # モデル保存
        logger.info(f"\n  モデル保存中: {self.phase1_8_dir}")

        save_data = {
            'models': models,
            'weights': weights,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'confidence_threshold': confidence_threshold,
            'metadata': {
                'accuracy': accuracy,
                'coverage': coverage,
                'train_samples': len(X_train),
                'val_accuracies': val_accuracies
            }
        }

        # 通貨ペア名を含むファイル名で保存
        pair_code = self.instrument.replace('/', '_')
        model_filename = f'{pair_code}_ensemble_models.pkl'
        joblib.dump(save_data, self.phase1_8_dir / model_filename)
        logger.info(f"  ✅ Phase 1.8モデル保存完了: {model_filename}")

        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'models': models,
            'weights': weights
        }

    def _train_and_save_phase2(self, data_tuple):
        """Phase 2訓練・保存"""
        features_all, _ = data_tuple

        logger.info(f"  サンプル数: {len(features_all)}件（全データ）")

        # 特徴量とラベル
        feature_cols = [col for col in features_all.columns
                       if not col.startswith('label_')]

        X = features_all[feature_cols].values
        y = features_all['label_return'].values

        # 分割（70/15/15）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, shuffle=False
        )

        logger.info(f"  訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # XGBoost訓練（Phase 2のベストモデル）
        logger.info("  訓練中: XGBoost（回帰）...")
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 予測
        y_pred = model.predict(X_test_scaled)

        # Phase 2評価
        positions = np.sign(y_pred)
        returns = positions * y_test

        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        cumulative_return = returns.sum()
        win_rate = (returns > 0).sum() / len(returns) * 100

        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0

        logger.info(f"\n✅ Phase 2結果:")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  累積リターン: {cumulative_return:.2f}%")
        logger.info(f"  勝率: {win_rate:.2f}%")
        logger.info(f"  PF: {profit_factor:.2f}")

        # モデル保存
        logger.info(f"\n  モデル保存中: {self.phase2_dir}")

        save_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'metadata': {
                'sharpe': sharpe,
                'cumulative_return': cumulative_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
        }

        # 通貨ペア名を含むファイル名で保存
        pair_code = self.instrument.replace('/', '_')
        model_filename = f'{pair_code}_xgboost_model.pkl'
        joblib.dump(save_data, self.phase2_dir / model_filename)
        logger.info(f"  ✅ Phase 2モデル保存完了: {model_filename}")

        return {
            'sharpe': sharpe,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def _print_summary(self, phase1, phase2):
        """結果サマリー"""
        logger.info("\n" + "="*80)
        logger.info("訓練完了サマリー")
        logger.info("="*80)

        logger.info("\nPhase 1.8（方向予測）:")
        logger.info(f"  ✅ 保存先: {self.phase1_8_dir}/ensemble_models.pkl")
        logger.info(f"  方向性的中率: {phase1['accuracy']:.2f}%")
        logger.info(f"  カバー率: {phase1['coverage']:.2f}%")

        logger.info("\nPhase 2（収益予測）:")
        logger.info(f"  ✅ 保存先: {self.phase2_dir}/xgboost_model.pkl")
        logger.info(f"  Sharpe Ratio: {phase2['sharpe']:.2f}")
        logger.info(f"  勝率: {phase2['win_rate']:.2f}%")

        logger.info("\n次のステップ:")
        logger.info("  1. ハイブリッド予測エンジン構築")
        logger.info("  2. バックテスト実施")
        logger.info("  3. 性能評価")

        logger.info("="*80)


if __name__ == "__main__":
    try:
        trainer = HybridModelTrainer()
        success = trainer.run()

        if success:
            logger.success("\n🚀 モデル訓練・保存完了！")
            logger.info("ハイブリッドシステムの準備が整いました")
            sys.exit(0)
        else:
            logger.error("\n❌ 訓練失敗")
            sys.exit(1)

    except Exception as e:
        logger.error(f"\nエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
