"""
Phase 2 クイック検証

世界最強を目指すため、実際の性能を素早く検証

簡易版:
- 短期データで動作（100日程度）
- 基本的なテクニカル指標のみ
- Phase 1.8との比較
"""

from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import sys

from src.data_sources.yahoo_finance import YahooFinanceData
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class QuickPhase2Validator:
    """クイック検証システム"""

    def __init__(self):
        self.yahoo = YahooFinanceData()
        logger.info("="*80)
        logger.info("Phase 2 クイック検証 - 世界最強への道")
        logger.info("="*80)

    def run(self):
        """実行"""
        # 1. データ取得
        logger.info("\n1. データ取得中...")
        data = self._get_data()

        if data.empty:
            logger.error("データ取得失敗")
            return False

        logger.info(f"✅ データ: {len(data)}日分")

        # 2. 特徴量生成
        logger.info("\n2. 特徴量生成中...")
        features_df = self._generate_features(data)

        logger.info(f"✅ 特徴量: {features_df.shape}")

        # 3. Phase 1.8モデル訓練（分類: 方向予測）
        logger.info("\n3. Phase 1.8モデル訓練（方向予測）...")
        phase1_results = self._train_phase1(features_df)

        # 4. Phase 2モデル訓練（回帰: 収益予測）
        logger.info("\n4. Phase 2モデル訓練（収益予測）...")
        phase2_results = self._train_phase2(features_df)

        # 5. 比較
        logger.info("\n5. Phase 1.8 vs Phase 2 比較")
        self._compare_results(phase1_results, phase2_results)

        return True

    def _get_data(self):
        """データ取得"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)  # 500日分

        data = self.yahoo.get_forex_data(
            pair='USD/JPY',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        return data

    def _generate_features(self, data):
        """特徴量生成（短期指標のみ）"""
        df = data.copy()
        features = pd.DataFrame(index=df.index)

        # 基本価格
        features['close'] = df['close']
        features['volume'] = df['volume']

        # リターン
        for period in [1, 5, 10, 20]:
            features[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # SMA（短期のみ）
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_vs_sma_{period}'] = ((df['close'] / sma) - 1) * 100

        # EMA
        for period in [12, 26]:
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26

        # ボラティリティ
        features['volatility_20d'] = df['close'].pct_change().rolling(20).std() * 100

        # ラベル
        # Phase 1.8: 方向（分類）
        features['label_direction'] = (df['close'].shift(-1) > df['close']).astype(int)

        # Phase 2: リターン（回帰）
        features['label_return'] = ((df['close'].shift(-1) / df['close']) - 1) * 100

        # NaN除去
        features = features.dropna()

        return features

    def _train_phase1(self, features_df):
        """Phase 1.8訓練（分類器）"""
        # 特徴量とラベル
        feature_cols = [col for col in features_df.columns
                       if not col.startswith('label_')]
        X = features_df[feature_cols].values
        y = features_df['label_direction'].values

        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 訓練
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        # 予測
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # 精度
        train_acc = (y_pred_train == y_train).mean() * 100
        test_acc = (y_pred_test == y_test).mean() * 100

        logger.info(f"  訓練精度: {train_acc:.2f}%")
        logger.info(f"  テスト精度: {test_acc:.2f}%")

        return {
            'accuracy': test_acc,
            'model_type': '分類器（方向予測）'
        }

    def _train_phase2(self, features_df):
        """Phase 2訓練（回帰器）"""
        # 特徴量とラベル
        feature_cols = [col for col in features_df.columns
                       if not col.startswith('label_')]
        X = features_df[feature_cols].values
        y = features_df['label_return'].values

        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 訓練
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        # 予測
        y_pred_test = model.predict(X_test_scaled)

        # Phase 2評価（収益ベース）
        positions = np.sign(y_pred_test)  # 予測方向に従ってポジション
        returns = positions * y_test  # 実際のリターン

        # Sharpe Ratio
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # 累積リターン
        cumulative_return = returns.sum()

        # 勝率
        win_rate = (returns > 0).sum() / len(returns) * 100

        # プロフィットファクター
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0

        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  累積リターン: {cumulative_return:.2f}%")
        logger.info(f"  勝率: {win_rate:.2f}%")
        logger.info(f"  プロフィットファクター: {profit_factor:.2f}")

        return {
            'sharpe': sharpe,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'model_type': '回帰器（収益予測）'
        }

    def _compare_results(self, phase1, phase2):
        """結果比較"""
        logger.info("\n" + "="*80)
        logger.info("Phase 1.8 vs Phase 2 - 実測比較")
        logger.info("="*80)
        logger.info(f"\nPhase 1.8（方向予測）:")
        logger.info(f"  モデル: {phase1['model_type']}")
        logger.info(f"  方向性的中率: {phase1['accuracy']:.2f}%")

        logger.info(f"\nPhase 2（収益予測）:")
        logger.info(f"  モデル: {phase2['model_type']}")
        logger.info(f"  Sharpe Ratio: {phase2['sharpe']:.2f}")
        logger.info(f"  累積リターン: {phase2['cumulative_return']:.2f}%")
        logger.info(f"  勝率: {phase2['win_rate']:.2f}%")
        logger.info(f"  プロフィットファクター: {phase2['profit_factor']:.2f}")

        logger.info("\n" + "="*80)
        logger.info("世界最強レベル判定")
        logger.info("="*80)

        # 判定基準
        world_class_criteria = {
            'Phase 1.8 精度 >= 80%': phase1['accuracy'] >= 80,
            'Phase 2 Sharpe >= 15': phase2['sharpe'] >= 15,
            'Phase 2 勝率 >= 70%': phase2['win_rate'] >= 70,
            'Phase 2 PF >= 1.5': phase2['profit_factor'] >= 1.5
        }

        passed = 0
        for criterion, result in world_class_criteria.items():
            status = "✅" if result else "⚠️"
            logger.info(f"  {status} {criterion}")
            if result:
                passed += 1

        logger.info(f"\n合格: {passed}/{len(world_class_criteria)}項目")

        if passed >= 3:
            logger.info("\n🎉 世界クラス水準達成！")
        elif passed >= 2:
            logger.info("\n⭐ 優秀！さらに改善可能")
        else:
            logger.info("\n⚠️  改善の余地あり")

        logger.info("\n次のステップ:")
        logger.info("  1. より長期データ（2500日）で完全訓練")
        logger.info("  2. ハイブリッドシステム（Phase 1.8 + Phase 2）")
        logger.info("  3. リアルタイム取引システム")
        logger.info("="*80)


if __name__ == "__main__":
    try:
        validator = QuickPhase2Validator()
        success = validator.run()

        if success:
            logger.info("\n🚀 検証完了！世界最強への道は続く...")
            sys.exit(0)
        else:
            logger.error("\n❌ 検証失敗")
            sys.exit(1)

    except Exception as e:
        logger.error(f"\nエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
