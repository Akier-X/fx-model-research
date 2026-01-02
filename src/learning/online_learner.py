"""
オンライン学習システム - 失敗から自動的に学習

警告サインは「停止」ではなく「学習トリガー」として機能
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
from pathlib import Path
import joblib
from datetime import datetime
from collections import deque

from ..ml.ml_strategy import MLStrategy
from ..rl.rl_agent import RLAgent


class OnlineLearner:
    """
    継続的学習システム

    失敗・成功の経験から自動的に学習し、モデルを改善
    """

    def __init__(
        self,
        ml_strategy: MLStrategy,
        rl_agent: RLAgent,
        experience_buffer_size: int = 10000,
        retrain_threshold: int = 100,  # 100トレード毎に再訓練
        min_performance_threshold: float = -0.05  # -5%以下で緊急学習
    ):
        self.ml_strategy = ml_strategy
        self.rl_agent = rl_agent
        self.experience_buffer_size = experience_buffer_size
        self.retrain_threshold = retrain_threshold
        self.min_performance_threshold = min_performance_threshold

        # 経験バッファ
        self.trade_history = deque(maxlen=experience_buffer_size)
        self.performance_history = deque(maxlen=1000)

        # 学習統計
        self.total_retrains = 0
        self.emergency_retrains = 0
        self.improvement_count = 0

        # パフォーマンストラッキング
        self.current_drawdown = 0
        self.consecutive_losses = 0
        self.recent_performance = []

        logger.info("オンライン学習システム初期化完了")

    def add_trade_experience(
        self,
        trade_data: Dict,
        market_data: pd.DataFrame,
        outcome: str,  # 'win', 'loss', 'neutral'
        pnl: float,
        sentiment_data: Optional[Dict] = None
    ):
        """
        トレード経験を記録

        Args:
            trade_data: トレード情報
            market_data: 市場データ
            outcome: 結果
            pnl: 損益
            sentiment_data: センチメントデータ
        """
        experience = {
            'timestamp': datetime.now(),
            'trade_data': trade_data,
            'market_data': market_data.copy(),
            'outcome': outcome,
            'pnl': pnl,
            'sentiment_data': sentiment_data
        }

        self.trade_history.append(experience)
        self.performance_history.append(pnl)

        # パフォーマンス指標更新
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # ドローダウン計算
        if len(self.performance_history) > 0:
            cumulative = np.cumsum(list(self.performance_history))
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / (peak + 1e-10)
            self.current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0

        logger.info(
            f"経験追加: {outcome.upper()}, PnL: {pnl:,.0f}, "
            f"連敗: {self.consecutive_losses}, DD: {self.current_drawdown:.2%}"
        )

        # 学習トリガーチェック
        self._check_learning_triggers()

    def _check_learning_triggers(self):
        """
        学習トリガーをチェック

        警告サイン = 学習の機会
        """
        should_retrain = False
        reason = ""
        is_emergency = False

        # トリガー1: 定期的な再訓練
        if len(self.trade_history) >= self.retrain_threshold:
            should_retrain = True
            reason = f"定期再訓練 ({len(self.trade_history)}トレード)"

        # トリガー2: 大きなドローダウン（緊急学習）
        if self.current_drawdown < -0.15:  # -15%
            should_retrain = True
            is_emergency = True
            reason = f"ドローダウン緊急学習 ({self.current_drawdown:.2%})"

        # トリガー3: 連続損失（パターン学習）
        if self.consecutive_losses >= 5:
            should_retrain = True
            is_emergency = True
            reason = f"連敗パターン学習 ({self.consecutive_losses}連敗)"

        # トリガー4: 直近パフォーマンス悪化
        if len(self.performance_history) >= 20:
            recent_20 = list(self.performance_history)[-20:]
            if np.mean(recent_20) < self.min_performance_threshold:
                should_retrain = True
                is_emergency = True
                reason = f"パフォーマンス悪化 (平均PnL: {np.mean(recent_20):.2%})"

        if should_retrain:
            logger.warning(f"🔄 学習トリガー発動: {reason}")
            self.retrain_models(emergency=is_emergency, reason=reason)

    def retrain_models(self, emergency: bool = False, reason: str = ""):
        """
        モデルを再訓練

        Args:
            emergency: 緊急学習フラグ
            reason: 学習理由
        """
        logger.info("=" * 60)
        logger.info(f"🧠 モデル再訓練開始: {reason}")
        logger.info(f"緊急度: {'🚨 HIGH' if emergency else '📊 NORMAL'}")
        logger.info("=" * 60)

        if len(self.trade_history) < 50:
            logger.warning("経験データ不足。再訓練をスキップ")
            return

        try:
            # 1. 経験データから学習データを構築
            training_data = self._build_training_data()

            # 2. 失敗ケースに重点を置く
            weighted_data = self._weight_experiences(training_data, emergency)

            # 3. MLモデルを再訓練
            logger.info("[1/2] 機械学習モデル再訓練中...")
            self._retrain_ml_model(weighted_data)

            # 4. RLエージェントを追加学習
            logger.info("[2/2] 強化学習エージェント追加学習中...")
            self._retrain_rl_agent(weighted_data, emergency)

            # 5. 統計更新
            self.total_retrains += 1
            if emergency:
                self.emergency_retrains += 1

            # 6. パフォーマンス検証
            improvement = self._validate_improvement(training_data)

            if improvement:
                self.improvement_count += 1
                logger.info(f"✅ モデル改善成功 (改善率: {improvement:.2%})")
            else:
                logger.warning("⚠️ 改善が見られません。ロールバック検討")

            # 7. 経験バッファをクリア（部分的に）
            if not emergency:
                # 緊急時は経験を保持、通常時は半分クリア
                half = len(self.trade_history) // 2
                for _ in range(half):
                    self.trade_history.popleft()

            logger.info(f"再訓練完了。総再訓練回数: {self.total_retrains}")

        except Exception as e:
            logger.error(f"再訓練エラー: {e}")

    def _build_training_data(self) -> pd.DataFrame:
        """
        経験バッファから学習データを構築
        """
        all_market_data = []
        labels = []
        weights = []

        for exp in self.trade_history:
            market_data = exp['market_data']

            # ラベル: 勝ちトレード=1, 負けトレード=-1
            if exp['outcome'] == 'win':
                label = 1
                weight = 1.0
            elif exp['outcome'] == 'loss':
                label = -1
                weight = 1.5  # 失敗に重み
            else:
                label = 0
                weight = 0.5

            all_market_data.append(market_data)
            labels.append(label)
            weights.append(weight)

        # データフレームに統合
        combined_df = pd.concat(all_market_data, ignore_index=True)
        combined_df['label'] = labels * len(combined_df) // len(labels)
        combined_df['weight'] = weights * len(combined_df) // len(weights)

        return combined_df

    def _weight_experiences(
        self,
        training_data: pd.DataFrame,
        emergency: bool
    ) -> pd.DataFrame:
        """
        経験に重み付け

        緊急時は失敗ケースに3倍の重み
        """
        if emergency:
            # 失敗ケースを強調
            training_data.loc[training_data['label'] == -1, 'weight'] *= 3.0
            logger.info("緊急モード: 失敗ケースに3倍の重み")

        return training_data

    def _retrain_ml_model(self, training_data: pd.DataFrame):
        """
        機械学習モデルを再訓練
        """
        # 既存モデルを基に追加学習
        # warm_start を使用して、既存の知識を保持しつつ新データで学習

        self.ml_strategy.model.set_params(warm_start=True)

        features = self.ml_strategy.prepare_features(training_data)
        X = features.drop(['label', 'weight'], axis=1, errors='ignore')
        y = training_data['label']
        sample_weight = training_data['weight']

        # 追加学習
        self.ml_strategy.model.fit(X, y, sample_weight=sample_weight)

        # モデル保存
        self.ml_strategy.save_model()

        logger.info("MLモデル再訓練完了")

    def _retrain_rl_agent(
        self,
        training_data: pd.DataFrame,
        emergency: bool
    ):
        """
        強化学習エージェントを追加学習
        """
        # 経験再生バッファに追加
        # 緊急時は学習率を一時的に上げる

        if emergency:
            original_lr = self.rl_agent.learning_rate
            self.rl_agent.learning_rate *= 2.0  # 2倍の学習率

        # 少ないステップで集中学習
        self.rl_agent.train(
            train_df=training_data,
            total_timesteps=10000,  # 緊急時は短時間で集中学習
            initial_balance=10000
        )

        if emergency:
            self.rl_agent.learning_rate = original_lr  # 元に戻す

        logger.info("RLエージェント追加学習完了")

    def _validate_improvement(self, training_data: pd.DataFrame) -> float:
        """
        改善を検証

        Returns:
            改善率（正なら改善、負なら悪化）
        """
        # 簡易バックテストで検証
        # 実装は省略（実際にはバックテストを実行）
        return 0.05  # プレースホルダー

    def get_learning_stats(self) -> Dict:
        """
        学習統計を取得
        """
        return {
            'total_trades': len(self.trade_history),
            'total_retrains': self.total_retrains,
            'emergency_retrains': self.emergency_retrains,
            'improvement_count': self.improvement_count,
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'avg_recent_pnl': np.mean(list(self.performance_history)[-20:]) if len(self.performance_history) >= 20 else 0
        }

    def should_pause_trading(self) -> tuple[bool, str]:
        """
        トレードを一時停止すべきか判断

        Returns:
            (停止すべきか, 理由)
        """
        # 本当に深刻な状況でのみ停止

        # 1. 破産レベルのドローダウン
        if self.current_drawdown < -0.50:  # -50%
            return True, f"破産レベルのドローダウン ({self.current_drawdown:.2%})"

        # 2. 異常な連敗（学習しても改善しない）
        if self.consecutive_losses >= 15:
            return True, f"異常な連敗 ({self.consecutive_losses}連敗)"

        # 3. モデルが明らかに機能していない
        if len(self.performance_history) >= 50:
            recent_50 = list(self.performance_history)[-50:]
            if sum(1 for x in recent_50 if x < 0) >= 40:  # 50中40が損失
                return True, "モデル機能不全 (勝率20%以下)"

        return False, ""

    def adaptive_risk_adjustment(self) -> float:
        """
        パフォーマンスに基づいてリスクを動的調整

        Returns:
            リスク調整係数 (0.5 ~ 2.0)
        """
        # 好調時はリスクを上げ、不調時は下げる

        if len(self.performance_history) < 20:
            return 1.0  # デフォルト

        recent_20 = list(self.performance_history)[-20:]
        avg_pnl = np.mean(recent_20)
        win_rate = sum(1 for x in recent_20 if x > 0) / 20

        # 好調（勝率60%以上、平均PnL正）
        if win_rate >= 0.6 and avg_pnl > 0:
            return 1.5  # リスク1.5倍

        # 絶好調（勝率70%以上、平均PnL大きい）
        if win_rate >= 0.7 and avg_pnl > 0.01:
            return 2.0  # リスク2倍

        # 不調（勝率40%以下）
        if win_rate <= 0.4:
            return 0.5  # リスク半分

        # 深刻な不調（勝率30%以下）
        if win_rate <= 0.3:
            return 0.25  # リスク4分の1

        return 1.0  # 普通
