"""
FXトレーディング環境 (Gymnasium)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from loguru import logger


class TradingEnv(gym.Env):
    """
    FXトレーディングのための強化学習環境

    状態空間: 価格データ + テクニカル指標 + センチメント
    行動空間: 0=SELL, 1=HOLD, 2=BUY
    報酬: シャープレシオベースの報酬関数
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        transaction_cost: float = 0.0001,
        max_position: int = 1,
        lookback_window: int = 50,
        sentiment_data: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            df: OHLCV + テクニカル指標のDataFrame
            initial_balance: 初期残高
            transaction_cost: 取引コスト (0.01%)
            max_position: 最大ポジションサイズ
            lookback_window: 状態に含める過去データの期間
            sentiment_data: センチメントデータ (オプション)
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.sentiment_data = sentiment_data

        # 行動空間: 0=SELL, 1=HOLD, 2=BUY
        self.action_space = spaces.Discrete(3)

        # 状態空間: 価格データ + 指標 + ポジション情報
        # 各時刻の特徴量数を計算
        self.n_features = len(self._get_features(0))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window * self.n_features + 3,),  # +3 は残高、ポジション、未実現損益
            dtype=np.float32
        )

        # エピソード状態
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: ショート, 0: ノーポジション, 1: ロング
        self.entry_price = 0
        self.total_profit = 0
        self.trades_history = []
        self.balance_history = []

    def _get_features(self, step: int) -> np.ndarray:
        """
        指定したステップの特徴量を取得

        Args:
            step: ステップ番号

        Returns:
            特徴量ベクトル
        """
        row = self.df.iloc[step]

        features = [
            row.get('open', 0),
            row.get('high', 0),
            row.get('low', 0),
            row.get('close', 0),
            row.get('volume', 0),
            # テクニカル指標
            row.get('sma_10', 0),
            row.get('sma_30', 0),
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('macd_signal', 0),
            row.get('atr', 0),
        ]

        # センチメントデータがあれば追加
        if self.sentiment_data is not None and step < len(self.sentiment_data):
            sentiment_row = self.sentiment_data.iloc[step]
            features.append(sentiment_row.get('sentiment_score', 0))
            features.append(sentiment_row.get('positive_ratio', 0))
            features.append(sentiment_row.get('negative_ratio', 0))

        return np.array(features, dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        現在の状態（観測）を取得

        Returns:
            状態ベクトル
        """
        # 過去 lookback_window 期間の特徴量
        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step

        obs_features = []
        for i in range(start, end):
            obs_features.extend(self._get_features(i))

        # 不足分はゼロパディング
        if len(obs_features) < self.lookback_window * self.n_features:
            padding = [0] * (self.lookback_window * self.n_features - len(obs_features))
            obs_features = padding + obs_features

        # ポジション情報を追加
        obs_features.append(self.balance / self.initial_balance)  # 正規化された残高
        obs_features.append(float(self.position))  # -1, 0, 1
        obs_features.append(self._get_unrealized_pnl() / self.initial_balance)  # 正規化された未実現損益

        return np.array(obs_features, dtype=np.float32)

    def _get_unrealized_pnl(self) -> float:
        """未実現損益を計算"""
        if self.position == 0:
            return 0

        current_price = self.df.iloc[self.current_step]['close']

        if self.position > 0:  # ロング
            pnl = (current_price - self.entry_price) * abs(self.position) * 1000
        else:  # ショート
            pnl = (self.entry_price - current_price) * abs(self.position) * 1000

        return pnl

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        環境をリセット

        Returns:
            (初期状態, info)
        """
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trades_history = []
        self.balance_history = [self.balance]

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        1ステップ実行

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY

        Returns:
            (次の状態, 報酬, 終了フラグ, 切り詰めフラグ, info)
        """
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        trade_executed = False

        # アクションを実行
        if action == 2 and self.position <= 0:  # BUY
            # 既存のショートポジションをクローズ
            if self.position < 0:
                pnl = self._close_position(current_price)
                reward += pnl / self.initial_balance

            # ロングポジションを開く
            self.position = 1
            self.entry_price = current_price
            trade_executed = True

        elif action == 0 and self.position >= 0:  # SELL
            # 既存のロングポジションをクローズ
            if self.position > 0:
                pnl = self._close_position(current_price)
                reward += pnl / self.initial_balance

            # ショートポジションを開く
            self.position = -1
            self.entry_price = current_price
            trade_executed = True

        elif action == 1:  # HOLD
            # ポジションを保持
            # 未実現損益に基づく小さな報酬
            if self.position != 0:
                unrealized_pnl = self._get_unrealized_pnl()
                reward += unrealized_pnl / self.initial_balance * 0.1  # 小さな報酬

        # 取引コストを適用
        if trade_executed:
            cost = current_price * self.transaction_cost * 1000
            self.balance -= cost
            reward -= cost / self.initial_balance

        # 次のステップへ
        self.current_step += 1

        # 残高履歴を記録
        total_value = self.balance + self._get_unrealized_pnl()
        self.balance_history.append(total_value)

        # 終了判定
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.balance <= 0  # 破産

        # 最終ステップで未決済ポジションをクローズ
        if terminated and self.position != 0:
            pnl = self._close_position(current_price)
            reward += pnl / self.initial_balance

        # 情報
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'total_value': total_value,
            'return': (total_value - self.initial_balance) / self.initial_balance
        }

        next_obs = self._get_observation() if not (terminated or truncated) else self._get_observation()

        return next_obs, reward, terminated, truncated, info

    def _close_position(self, current_price: float) -> float:
        """
        ポジションをクローズ

        Args:
            current_price: 現在価格

        Returns:
            損益
        """
        if self.position == 0:
            return 0

        # 損益計算
        if self.position > 0:  # ロング
            pnl = (current_price - self.entry_price) * abs(self.position) * 1000
        else:  # ショート
            pnl = (self.entry_price - current_price) * abs(self.position) * 1000

        self.balance += pnl
        self.total_profit += pnl

        self.trades_history.append({
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position': self.position,
            'pnl': pnl,
            'step': self.current_step
        })

        self.position = 0
        self.entry_price = 0

        return pnl

    def render(self, mode: str = 'human'):
        """環境の状態を表示"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ¥{self.balance:,.0f}")
            print(f"Position: {self.position}")
            print(f"Total Profit: ¥{self.total_profit:,.0f}")

    def get_performance_metrics(self) -> Dict:
        """パフォーマンス指標を計算"""
        if len(self.balance_history) < 2:
            return {}

        returns = pd.Series(self.balance_history).pct_change().dropna()

        total_return = (self.balance_history[-1] - self.initial_balance) / self.initial_balance

        # シャープレシオ
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 最大ドローダウン
        peak = pd.Series(self.balance_history).expanding().max()
        drawdown = (pd.Series(self.balance_history) - peak) / peak * 100
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades_history),
            'final_balance': self.balance_history[-1]
        }
