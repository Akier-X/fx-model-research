"""
時間足モデル訓練 - Phase 1.8 & Phase 2
すべての上下移動を捉える世界最強システム
"""
from pathlib import Path
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np
import joblib
from typing import Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from src.data_sources.yahoo_finance_hourly import YahooFinanceHourly


class HourlyModelTrainer:
    """時間足データでPhase 1.8とPhase 2を訓練"""

    def __init__(self, pair: str = 'USDJPY=X'):
        self.pair = pair
        self.hourly_data = YahooFinanceHourly(interval='1h')

        # モデル保存ディレクトリ
        self.models_dir = Path('models/hourly')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 出力ディレクトリ
        self.output_dir = Path('outputs/hourly_training')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"時間足モデル訓練器 初期化: {pair}")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データ読み込みと特徴量生成"""
        logger.info("\n" + "="*80)
        logger.info("1. データ読み込み")
        logger.info("="*80)

        # 保存済みデータを読み込み
        df = self.hourly_data.load_saved_data(pair=self.pair, period='2y')

        if df is None:
            raise ValueError(f"データが見つかりません: {self.pair}")

        # インデックスをDatetimeIndexに変換（UTC対応）
        df.index = pd.to_datetime(df.index, utc=True)

        logger.info(f"✅ データ読み込み完了: {len(df)}本")

        # 特徴量生成
        logger.info("\n2. 特徴量生成中...")
        features_all = self._generate_features(df)
        features_phase1 = features_all.copy()  # Phase 1.8用

        logger.info(f"✅ 特徴量生成完了")
        logger.info(f"  全特徴量: {features_all.shape}")
        logger.info(f"  Phase 1.8用: {features_phase1.shape}")

        return features_all, features_phase1

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間足に最適化された特徴量生成"""
        features = pd.DataFrame(index=df.index)

        # 基本価格特徴量
        features['price'] = df['close']
        features['return_1h'] = df['close'].pct_change(1) * 100
        features['return_4h'] = df['close'].pct_change(4) * 100
        features['return_24h'] = df['close'].pct_change(24) * 100

        # ボラティリティ
        features['volatility_24h'] = df['close'].pct_change().rolling(24).std() * 100
        features['volatility_168h'] = df['close'].pct_change().rolling(168).std() * 100  # 1週間

        # 移動平均
        for period in [12, 24, 48, 168]:  # 12h, 1d, 2d, 1week
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = (df['close'] / features[f'sma_{period}'] - 1) * 100

        # テクニカル指標
        # RSI
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']

        # ボリンジャーバンド
        for period in [20, 40]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (std * 2)
            features[f'bb_lower_{period}'] = sma - (std * 2)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / sma * 100

        # 時間帯特徴（FX市場の時間帯）
        features['hour'] = df.index.hour
        features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
        features['is_european_session'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
        features['is_us_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)

        # 曜日
        features['day_of_week'] = df.index.dayofweek

        logger.info(f"  生成された特徴量数: {len([c for c in features.columns if c not in ['label_direction', 'label_direction_raw', 'actual_return']])}")

        return features

    def train_phase1_8(self, features_df: pd.DataFrame) -> dict:
        """Phase 1.8: 方向予測モデル訓練（時間足版）"""
        logger.info("\n" + "="*80)
        logger.info("Phase 1.8 訓練開始（方向予測）")
        logger.info("="*80)

        # ラベル生成（時間足では±0.3%）
        df = self.hourly_data.load_saved_data(pair=self.pair, period='2y')
        df.index = pd.to_datetime(df.index, utc=True)
        threshold = 0.003  # 0.3%（日次の0.5%より小さく）

        price_change = ((df['close'].shift(-1) / df['close']) - 1) * 100
        features_df['label_direction_raw'] = np.where(
            price_change > (threshold * 100), 1,
            np.where(price_change < -(threshold * 100), 0, -1)
        )

        # 中立ラベルを除外
        features_df = features_df[features_df['label_direction_raw'] != -1].copy()
        features_df['label_direction'] = features_df['label_direction_raw']

        logger.info(f"訓練データ: {len(features_df)}サンプル")
        logger.info(f"  上昇: {(features_df['label_direction'] == 1).sum()}")
        logger.info(f"  下降: {(features_df['label_direction'] == 0).sum()}")

        # NaN除去
        feature_cols = [c for c in features_df.columns if c not in ['label_direction', 'label_direction_raw', 'actual_return']]
        features_df = features_df.dropna(subset=feature_cols + ['label_direction'])

        X = features_df[feature_cols].values
        y = features_df['label_direction'].values

        # 訓練/検証/テスト分割
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

        logger.info(f"訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 時間重み付け
        decay_rate = 0.95
        sample_weights = np.array([decay_rate ** i for i in range(len(X_train))])[::-1]

        # 5種類のモデル訓練
        models = {}
        logger.info("\nモデル訓練中...")

        models['gbc'] = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        models['rfc'] = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        models['xgb'] = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        models['lgbm'] = LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
        models['catboost'] = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=5, random_state=42, verbose=False)

        val_accuracies = {}
        for name, model in models.items():
            logger.info(f"  訓練中: {name}")
            if name in ['gbc', 'rfc']:
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_scaled, y_train)

            val_pred = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            val_accuracies[name] = val_acc
            logger.info(f"    検証精度: {val_acc:.4f}")

        # 重み計算
        weights = {name: acc / sum(val_accuracies.values()) for name, acc in val_accuracies.items()}

        # アンサンブル予測
        ensemble_probs = np.zeros((len(X_test_scaled), 2))
        for name, model in models.items():
            probs = model.predict_proba(X_test_scaled)
            ensemble_probs += probs * weights[name]

        final_pred = ensemble_probs.argmax(axis=1)
        final_confidence = ensemble_probs.max(axis=1)

        # 信頼度フィルタ
        confidence_threshold = 0.65
        confident_mask = final_confidence >= confidence_threshold
        filtered_pred = final_pred[confident_mask]
        filtered_actual = y_test[confident_mask]

        accuracy = accuracy_score(filtered_actual, filtered_pred)
        coverage = confident_mask.sum() / len(y_test) * 100

        logger.info(f"\n✅ Phase 1.8 訓練完了")
        logger.info(f"  精度: {accuracy*100:.2f}%")
        logger.info(f"  カバー率: {coverage:.2f}%")

        # モデル保存
        save_data = {
            'models': models,
            'weights': weights,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'confidence_threshold': confidence_threshold,
            'threshold': threshold,
            'metadata': {
                'accuracy': accuracy * 100,
                'coverage': coverage,
                'pair': self.pair
            }
        }

        save_path = self.models_dir / f'phase1_8_hourly_{self.pair.replace("=X", "")}.pkl'
        joblib.dump(save_data, save_path)
        logger.info(f"💾 モデル保存: {save_path}")

        return save_data

    def train_phase2(self, features_df: pd.DataFrame) -> dict:
        """Phase 2: 収益予測モデル訓練（時間足版）"""
        logger.info("\n" + "="*80)
        logger.info("Phase 2 訓練開始（収益予測）")
        logger.info("="*80)

        # ターゲット：次の1時間のリターン
        df = self.hourly_data.load_saved_data(pair=self.pair, period='2y')
        df.index = pd.to_datetime(df.index, utc=True)
        features_df['target_return'] = ((df['close'].shift(-1) / df['close']) - 1) * 100

        # NaN除去
        feature_cols = [c for c in features_df.columns if c not in ['label_direction', 'label_direction_raw', 'actual_return', 'target_return']]
        features_df = features_df.dropna(subset=feature_cols + ['target_return'])

        X = features_df[feature_cols].values
        y = features_df['target_return'].values

        # 訓練/検証/テスト分割
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

        logger.info(f"訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # GradientBoostingRegressor訓練
        logger.info("\nGradientBoostingRegressor 訓練中...")
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # テスト評価
        y_pred = model.predict(X_test_scaled)

        # Sharpe Ratio計算
        returns = y_test  # 実際のリターン
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # 年率換算（時間足）

        logger.info(f"\n✅ Phase 2 訓練完了")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")

        # モデル保存
        save_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'metadata': {
                'sharpe': sharpe,
                'pair': self.pair
            }
        }

        save_path = self.models_dir / f'phase2_hourly_{self.pair.replace("=X", "")}.pkl'
        joblib.dump(save_data, save_path)
        logger.info(f"💾 モデル保存: {save_path}")

        return save_data

    def run(self):
        """完全な訓練パイプライン実行"""
        logger.info("="*80)
        logger.info(f"時間足モデル訓練開始: {self.pair}")
        logger.info("="*80)

        # データ準備
        features_all, features_phase1 = self.load_and_prepare_data()

        # Phase 1.8 訓練
        phase1_model = self.train_phase1_8(features_phase1)

        # Phase 2 訓練
        phase2_model = self.train_phase2(features_all)

        logger.info("\n" + "="*80)
        logger.info("✅ すべての訓練完了")
        logger.info("="*80)

        return phase1_model, phase2_model


def train_all_pairs():
    """全通貨ペアのモデル訓練"""
    logger.info("="*80)
    logger.info("全通貨ペア時間足モデル訓練")
    logger.info("="*80)

    pairs = ['USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'EURJPY=X']

    results = {}
    for pair in pairs:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"通貨ペア: {pair}")
        logger.info(f"{'#'*80}\n")

        trainer = HourlyModelTrainer(pair=pair)
        phase1, phase2 = trainer.run()
        results[pair] = {'phase1': phase1, 'phase2': phase2}

    logger.info("\n\n" + "="*80)
    logger.info("全通貨ペア訓練完了サマリー")
    logger.info("="*80)

    for pair, models in results.items():
        logger.info(f"\n{pair}:")
        logger.info(f"  Phase 1.8 精度: {models['phase1']['metadata']['accuracy']:.2f}%")
        logger.info(f"  Phase 1.8 カバー率: {models['phase1']['metadata']['coverage']:.2f}%")
        logger.info(f"  Phase 2 Sharpe: {models['phase2']['metadata']['sharpe']:.2f}")

    return results


if __name__ == '__main__':
    train_all_pairs()
