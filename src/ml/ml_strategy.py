"""
機械学習ベースのトレード戦略
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import ta
from pathlib import Path
from loguru import logger

from ..strategies.base_strategy import BaseStrategy


class MLStrategy(BaseStrategy):
    """
    機械学習ベースのトレード戦略

    Random Forestを使用してトレードシグナルを予測
    """

    def __init__(
        self,
        model_path: str = "models/ml_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        n_estimators: int = 100,
        retrain: bool = False
    ):
        super().__init__(name="MLStrategy")
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.n_estimators = n_estimators

        if self.model_path.exists() and not retrain:
            self.load_model()
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を準備

        Args:
            data: OHLCV データ

        Returns:
            特徴量DataFrame
        """
        df = data.copy()

        # 移動平均線
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

        # EMA
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()

        # ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # ADX (トレンド強度)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])

        # 価格変化率
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        # ローソク足パターン
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # 欠損値を削除
        df = df.dropna()

        return df

    def create_labels(self, data: pd.DataFrame, future_periods: int = 5) -> pd.Series:
        """
        ラベルを作成 (将来価格の方向)

        Args:
            data: 価格データ
            future_periods: 何期先の価格を予測するか

        Returns:
            ラベル (1: 上昇, -1: 下落, 0: 横ばい)
        """
        future_price = data['close'].shift(-future_periods)
        current_price = data['close']

        price_change = (future_price - current_price) / current_price

        # 閾値: 0.5%以上の変化を有意とする
        threshold = 0.005

        labels = pd.Series(0, index=data.index)
        labels[price_change > threshold] = 1  # 買い
        labels[price_change < -threshold] = -1  # 売り

        return labels

    def train(self, data: pd.DataFrame, future_periods: int = 5):
        """
        モデルを訓練

        Args:
            data: OHLCV データ
            future_periods: 何期先を予測するか
        """
        logger.info("モデル訓練開始")

        # 特徴量準備
        df = self.prepare_features(data)
        labels = self.create_labels(data, future_periods)

        # データを揃える
        df = df[labels.index.isin(df.index)]
        labels = labels[labels.index.isin(df.index)]

        # 特徴量選択
        feature_columns = [
            'sma_10', 'sma_30', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_low', 'bb_width', 'atr',
            'stoch_k', 'stoch_d', 'adx',
            'price_change', 'volume_change',
            'body', 'upper_shadow', 'lower_shadow'
        ]

        X = df[feature_columns]
        y = labels

        # 欠損値を除去
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"訓練データサイズ: {len(X)}")
        logger.info(f"ラベル分布: {y.value_counts().to_dict()}")

        # スケーリング
        X_scaled = self.scaler.fit_transform(X)

        # 訓練
        self.model.fit(X_scaled, y)

        # 精度評価
        train_score = self.model.score(X_scaled, y)
        logger.info(f"訓練精度: {train_score:.4f}")

        # モデル保存
        self.save_model()

        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("特徴量重要度トップ10:")
        logger.info(f"\n{feature_importance.head(10)}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        トレードシグナルを生成

        Args:
            data: OHLCV データ

        Returns:
            シグナル付きDataFrame
        """
        df = self.prepare_features(data)

        feature_columns = [
            'sma_10', 'sma_30', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_low', 'bb_width', 'atr',
            'stoch_k', 'stoch_d', 'adx',
            'price_change', 'volume_change',
            'body', 'upper_shadow', 'lower_shadow'
        ]

        X = df[feature_columns]
        X_scaled = self.scaler.transform(X)

        # 予測
        predictions = self.model.predict(X_scaled)
        df['signal'] = predictions

        # 予測確率
        proba = self.model.predict_proba(X_scaled)
        df['confidence'] = proba.max(axis=1)

        # ポジション
        df['position'] = df['signal'].replace(0, pd.NA).ffill().fillna(0)

        return df

    def save_model(self):
        """モデルを保存"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"モデルを保存: {self.model_path}")

    def load_model(self):
        """モデルを読み込み"""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"モデルを読み込み: {self.model_path}")
