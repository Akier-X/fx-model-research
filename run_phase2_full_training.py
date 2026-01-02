"""
Phase 2 完全訓練 - 世界最強への挑戦

2500日データで完全訓練を実施

目標:
- Sharpe Ratio: 25-35
- 勝率: 85-90%
- プロフィットファクター: 2.5-3.5
- 月利: 15-25%
"""

from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from src.data_sources.yahoo_finance import YahooFinanceData
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost未インストール（スキップ）")


class Phase2FullTraining:
    """
    Phase 2 完全訓練システム

    世界最強を目指す完全実装
    """

    def __init__(self, lookback_days=2500):
        self.lookback_days = lookback_days
        self.yahoo = YahooFinanceData()
        self.output_dir = Path("models/phase2_full")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("Phase 2 完全訓練 - 世界最強への挑戦")
        logger.info("="*80)
        logger.info(f"訓練データ: {lookback_days}日分")

    def run(self):
        """完全訓練実行"""

        # 1. データ収集
        logger.info("\n" + "="*80)
        logger.info("1. データ収集（2500日）")
        logger.info("="*80)

        data = self._collect_data()

        if data.empty:
            logger.error("❌ データ収集失敗")
            return False

        logger.info(f"✅ データ収集完了: {len(data)}日分")

        # 2. 特徴量生成
        logger.info("\n" + "="*80)
        logger.info("2. 特徴量生成")
        logger.info("="*80)

        features_df = self._generate_features(data)

        logger.info(f"✅ 特徴量生成完了: {features_df.shape}")
        logger.info(f"   特徴量数: {features_df.shape[1] - 2}個")
        logger.info(f"   サンプル数: {len(features_df)}")

        # 3. Phase 1.8訓練（参考: 方向予測）
        logger.info("\n" + "="*80)
        logger.info("3. Phase 1.8モデル訓練（方向予測・参考用）")
        logger.info("="*80)

        phase1_results = self._train_phase1(features_df)

        # 4. Phase 2訓練（本命: 収益予測）
        logger.info("\n" + "="*80)
        logger.info("4. Phase 2モデル訓練（収益最適化）")
        logger.info("="*80)

        phase2_results = self._train_phase2_ensemble(features_df)

        # 5. バックテスト
        logger.info("\n" + "="*80)
        logger.info("5. バックテスト（実績シミュレーション）")
        logger.info("="*80)

        backtest_results = self._backtest(features_df, phase2_results)

        # 6. 総合評価
        logger.info("\n" + "="*80)
        logger.info("6. 総合評価・世界クラス判定")
        logger.info("="*80)

        self._final_evaluation(phase1_results, phase2_results, backtest_results)

        return True

    def _collect_data(self):
        """データ収集（2500日）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        logger.info(f"期間: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}")

        data = self.yahoo.get_forex_data(
            pair='USD/JPY',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        return data

    def _generate_features(self, data):
        """特徴量生成（拡張版）"""
        df = data.copy()
        features = pd.DataFrame(index=df.index)

        logger.info("特徴量生成中...")

        # 基本価格
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']

        # リターン（複数期間）
        for period in [1, 2, 3, 5, 10, 20, 30, 60]:
            features[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # SMA（複数期間）
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_vs_sma_{period}'] = ((df['close'] / sma) - 1) * 100

        # EMA
        for period in [9, 12, 21, 26, 50, 100]:
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_vs_ema_{period}'] = ((df['close'] / ema) - 1) * 100

        # RSI（複数期間）
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal

        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_middle_{period}'] = sma
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_width_{period}'] = ((sma + 2*std) - (sma - 2*std)) / sma
            features[f'bb_position_{period}'] = (df['close'] - (sma - 2*std)) / (4 * std)

        # ATR
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            features[f'atr_{period}'] = atr
            features[f'atr_ratio_{period}'] = (atr / df['close']) * 100

        # ボラティリティ
        for period in [5, 10, 20, 30, 60]:
            features[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * 100

        # モメンタム
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            features[f'momentum_pct_{period}'] = ((df['close'] / df['close'].shift(period)) - 1) * 100

        # ストキャスティクス
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            features[f'stoch_k_{period}'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

        # ラベル
        # Phase 1.8: 方向（閾値0.5%）
        price_change_pct = ((df['close'].shift(-1) / df['close']) - 1) * 100
        features['label_direction'] = (price_change_pct > 0.5).astype(int)

        # Phase 2: リターン
        features['label_return'] = price_change_pct

        # NaN除去
        features = features.dropna()

        logger.info(f"  生成特徴量: {features.shape[1] - 2}個")

        return features

    def _train_phase1(self, features_df):
        """Phase 1.8訓練（方向予測）"""
        feature_cols = [col for col in features_df.columns if not col.startswith('label_')]
        X = features_df[feature_cols].values
        y = features_df['label_direction'].values

        # 分割（時系列考慮）
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 訓練
        logger.info("訓練中（GradientBoostingClassifier）...")
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        # 評価
        train_acc = (model.predict(X_train_scaled) == y_train).mean() * 100
        val_acc = (model.predict(X_val_scaled) == y_val).mean() * 100
        test_acc = (model.predict(X_test_scaled) == y_test).mean() * 100

        logger.info(f"  訓練精度: {train_acc:.2f}%")
        logger.info(f"  検証精度: {val_acc:.2f}%")
        logger.info(f"  テスト精度: {test_acc:.2f}%")

        return {
            'model': model,
            'scaler': scaler,
            'test_accuracy': test_acc,
            'val_accuracy': val_acc
        }

    def _train_phase2_ensemble(self, features_df):
        """Phase 2アンサンブル訓練（収益予測）"""
        feature_cols = [col for col in features_df.columns if not col.startswith('label_')]
        X = features_df[feature_cols].values
        y = features_df['label_return'].values

        # 分割
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        logger.info(f"訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")

        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # モデル定義
        models = {}

        # 1. GradientBoosting
        logger.info("\n訓練中: GradientBoostingRegressor...")
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        models['gradient_boosting'].fit(X_train_scaled, y_train)

        # 2. RandomForest
        logger.info("訓練中: RandomForestRegressor...")
        models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        models['random_forest'].fit(X_train_scaled, y_train)

        # 3. XGBoost
        logger.info("訓練中: XGBoostRegressor...")
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        models['xgboost'].fit(X_train_scaled, y_train)

        # 4. LightGBM
        logger.info("訓練中: LightGBMRegressor...")
        models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        models['lightgbm'].fit(X_train_scaled, y_train)

        # 5. CatBoost (optional)
        if CATBOOST_AVAILABLE:
            logger.info("訓練中: CatBoostRegressor...")
            models['catboost'] = CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=5,
                random_state=42,
                verbose=False
            )
            models['catboost'].fit(X_train_scaled, y_train)

        # 各モデル評価
        results = {}

        for model_name, model in models.items():
            y_pred_test = model.predict(X_test_scaled)

            # Phase 2評価
            metrics = self._calculate_metrics(y_test, y_pred_test)

            results[model_name] = {
                'model': model,
                'metrics': metrics
            }

            logger.info(f"\n{model_name}:")
            logger.info(f"  Sharpe: {metrics['sharpe']:.2f}")
            logger.info(f"  累積リターン: {metrics['cumulative_return']:.2f}%")
            logger.info(f"  勝率: {metrics['win_rate']:.2f}%")
            logger.info(f"  PF: {metrics['profit_factor']:.2f}")

        # ベストモデル選択
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['sharpe'])

        logger.info(f"\n{'='*80}")
        logger.info(f"ベストモデル: {best_model_name}")
        logger.info(f"  Sharpe: {results[best_model_name]['metrics']['sharpe']:.2f}")
        logger.info(f"{'='*80}")

        return {
            'models': models,
            'results': results,
            'best_model': best_model_name,
            'scaler': scaler,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test
        }

    def _calculate_metrics(self, y_true, y_pred):
        """Phase 2評価指標計算"""
        # 予測方向にポジション
        positions = np.sign(y_pred)
        returns = positions * y_true

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

        # 最大ドローダウン
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = drawdown.max()

        return {
            'sharpe': sharpe,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'total_trades': len(returns),
            'winning_trades': (returns > 0).sum(),
            'losing_trades': (returns <= 0).sum()
        }

    def _backtest(self, features_df, phase2_results):
        """バックテスト"""
        initial_capital = 10000000  # 1000万円

        best_model = phase2_results['models'][phase2_results['best_model']]
        X_test = phase2_results['X_test_scaled']
        y_test = phase2_results['y_test']

        # 予測
        y_pred = best_model.predict(X_test)

        # トレード実行
        positions = np.sign(y_pred)
        returns = positions * y_test

        # 資産推移
        capital_series = [initial_capital]
        for ret in returns:
            capital_series.append(capital_series[-1] * (1 + ret/100))

        final_capital = capital_series[-1]
        total_return = ((final_capital / initial_capital) - 1) * 100

        # 月利換算
        days = len(returns)
        months = days / 20  # 20営業日/月
        monthly_return = (total_return / months) if months > 0 else 0

        logger.info(f"初期資金: ¥{initial_capital:,}")
        logger.info(f"最終資産: ¥{final_capital:,.0f}")
        logger.info(f"総リターン: {total_return:.2f}%")
        logger.info(f"月利（推定）: {monthly_return:.2f}%")
        logger.info(f"取引日数: {days}日（約{months:.1f}ヶ月）")

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'capital_series': capital_series
        }

    def _final_evaluation(self, phase1, phase2, backtest):
        """最終評価"""
        best_metrics = phase2['results'][phase2['best_model']]['metrics']

        logger.info("\n" + "="*80)
        logger.info("Phase 2 完全訓練 - 最終結果")
        logger.info("="*80)

        logger.info(f"\nPhase 1.8（方向予測）:")
        logger.info(f"  テスト精度: {phase1['test_accuracy']:.2f}%")

        logger.info(f"\nPhase 2（収益予測）:")
        logger.info(f"  ベストモデル: {phase2['best_model']}")
        logger.info(f"  Sharpe Ratio: {best_metrics['sharpe']:.2f}")
        logger.info(f"  累積リターン: {best_metrics['cumulative_return']:.2f}%")
        logger.info(f"  勝率: {best_metrics['win_rate']:.2f}%")
        logger.info(f"  プロフィットファクター: {best_metrics['profit_factor']:.2f}")
        logger.info(f"  最大DD: {best_metrics['max_drawdown']:.2f}%")

        logger.info(f"\nバックテスト:")
        logger.info(f"  総リターン: {backtest['total_return']:.2f}%")
        logger.info(f"  月利: {backtest['monthly_return']:.2f}%")
        logger.info(f"  最終資産: ¥{backtest['final_capital']:,.0f}")

        # 世界クラス判定
        logger.info(f"\n{'='*80}")
        logger.info("世界クラス判定")
        logger.info(f"{'='*80}")

        criteria = {
            'Phase 1.8精度 >= 80%': phase1['test_accuracy'] >= 80,
            'Phase 2 Sharpe >= 20': best_metrics['sharpe'] >= 20,
            'Phase 2 勝率 >= 75%': best_metrics['win_rate'] >= 75,
            'Phase 2 PF >= 2.0': best_metrics['profit_factor'] >= 2.0,
            '月利 >= 10%': backtest['monthly_return'] >= 10
        }

        passed = 0
        for criterion, result in criteria.items():
            status = "✅" if result else "⚠️"
            logger.info(f"  {status} {criterion}")
            if result:
                passed += 1

        logger.info(f"\n合格: {passed}/{len(criteria)}項目")

        if passed >= 4:
            logger.info("\n🎉🎉🎉 世界クラス達成！ 🎉🎉🎉")
        elif passed >= 3:
            logger.info("\n⭐ 優秀！あと少しで世界クラス")
        else:
            logger.info("\n💪 良好！さらなる改善で世界クラスへ")

        logger.info(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        trainer = Phase2FullTraining(lookback_days=2500)
        success = trainer.run()

        if success:
            logger.info("\n🚀 Phase 2 完全訓練完了！")
            logger.info("世界最強AIトレーダーへの道を前進！")
            sys.exit(0)
        else:
            logger.error("\n❌ 訓練失敗")
            sys.exit(1)

    except Exception as e:
        logger.error(f"\nエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
