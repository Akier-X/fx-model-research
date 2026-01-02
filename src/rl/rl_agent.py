"""
強化学習エージェント (PPO)
"""
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, Dict
import pandas as pd

from .trading_env import TradingEnv


class TrainingCallback(BaseCallback):
    """訓練中のコールバック"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """ロールアウト終了時"""
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            logger.info(f"Mean episode reward: {mean_reward:.2f}")


class RLAgent:
    """
    強化学習エージェント

    PPO (Proximal Policy Optimization) を使用
    """

    def __init__(
        self,
        algorithm: str = "PPO",
        model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        device: str = "auto"
    ):
        """
        Args:
            algorithm: "PPO" or "DQN"
            model_path: 保存済みモデルのパス
            learning_rate: 学習率
            n_steps: ステップ数 (PPO)
            batch_size: バッチサイズ
            gamma: 割引率
            device: "cpu", "cuda", or "auto"
        """
        self.algorithm = algorithm
        self.model_path = Path(model_path) if model_path else Path("models/rl_agent.zip")
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.model = None
        self.env = None

    def create_env(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> TradingEnv:
        """
        トレーディング環境を作成

        Args:
            df: 価格データ
            initial_balance: 初期残高
            sentiment_data: センチメントデータ

        Returns:
            環境
        """
        env = TradingEnv(
            df=df,
            initial_balance=initial_balance,
            sentiment_data=sentiment_data
        )
        return env

    def train(
        self,
        train_df: pd.DataFrame,
        total_timesteps: int = 100000,
        initial_balance: float = 10000,
        sentiment_data: Optional[pd.DataFrame] = None,
        save_freq: int = 10000
    ) -> Dict:
        """
        エージェントを訓練

        Args:
            train_df: 訓練データ
            total_timesteps: 総ステップ数
            initial_balance: 初期残高
            sentiment_data: センチメントデータ
            save_freq: モデル保存頻度

        Returns:
            訓練結果
        """
        logger.info(f"{self.algorithm} エージェントの訓練開始")
        logger.info(f"Total timesteps: {total_timesteps:,}")

        # 環境作成
        env = self.create_env(train_df, initial_balance, sentiment_data)
        vec_env = DummyVecEnv([lambda: env])

        # モデル作成
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=1,
                device=self.device,
                tensorboard_log="logs/tensorboard/"
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                vec_env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=1,
                device=self.device,
                tensorboard_log="logs/tensorboard/"
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # コールバック
        callback = TrainingCallback()

        # 訓練
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        # モデル保存
        self.save()

        logger.info("訓練完了")

        return {
            'algorithm': self.algorithm,
            'total_timesteps': total_timesteps,
            'model_path': str(self.model_path)
        }

    def evaluate(
        self,
        test_df: pd.DataFrame,
        initial_balance: float = 10000,
        sentiment_data: Optional[pd.DataFrame] = None,
        n_episodes: int = 10
    ) -> Dict:
        """
        エージェントを評価

        Args:
            test_df: テストデータ
            initial_balance: 初期残高
            sentiment_data: センチメントデータ
            n_episodes: エピソード数

        Returns:
            評価結果
        """
        if self.model is None:
            logger.error("モデルが読み込まれていません")
            return {}

        logger.info(f"エージェントの評価開始 ({n_episodes} episodes)")

        env = self.create_env(test_df, initial_balance, sentiment_data)

        episode_rewards = []
        episode_returns = []
        episode_sharpes = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

            # パフォーマンス指標
            metrics = env.get_performance_metrics()

            episode_rewards.append(episode_reward)
            episode_returns.append(metrics.get('total_return', 0))
            episode_sharpes.append(metrics.get('sharpe_ratio', 0))

            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Reward={episode_reward:.4f}, "
                f"Return={metrics.get('total_return', 0):.2f}%, "
                f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}"
            )

        results = {
            'mean_reward': np.mean(episode_rewards),
            'mean_return': np.mean(episode_returns),
            'mean_sharpe': np.mean(episode_sharpes),
            'std_reward': np.std(episode_rewards),
            'std_return': np.std(episode_returns),
            'episodes': n_episodes,
            'episode_rewards': episode_rewards,
            'episode_returns': episode_returns,
            'episode_sharpes': episode_sharpes
        }

        logger.info(f"平均リターン: {results['mean_return']:.2f}%")
        logger.info(f"平均シャープレシオ: {results['mean_sharpe']:.2f}")

        return results

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        行動を予測

        Args:
            observation: 観測
            deterministic: 決定的な行動を取るか

        Returns:
            行動 (0=SELL, 1=HOLD, 2=BUY)
        """
        if self.model is None:
            logger.error("モデルが読み込まれていません")
            return 1  # HOLD

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: Optional[str] = None):
        """モデルを保存"""
        if self.model is None:
            logger.error("モデルが存在しません")
            return

        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(save_path)
        logger.info(f"モデルを保存: {save_path}")

    def load(self, path: Optional[str] = None):
        """モデルを読み込み"""
        load_path = Path(path) if path else self.model_path

        if not load_path.exists():
            logger.error(f"モデルファイルが見つかりません: {load_path}")
            return

        if self.algorithm == "PPO":
            self.model = PPO.load(load_path, device=self.device)
        elif self.algorithm == "DQN":
            self.model = DQN.load(load_path, device=self.device)

        logger.info(f"モデルを読み込み: {load_path}")
