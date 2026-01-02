"""
バックテストエンジン
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
from ..strategies.base_strategy import BaseStrategy


class Backtester:
    """バックテストエンジン"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000,
        commission: float = 0.0001,  # 0.01%
        pip_value: float = 1000
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.pip_value = pip_value
        self.trades: List[Dict] = []

    def run(self, data: pd.DataFrame) -> Dict:
        """
        バックテストを実行

        Args:
            data: OHLCV データ

        Returns:
            パフォーマンス指標
        """
        df = self.strategy.generate_signals(data)

        capital = self.initial_capital
        position = 0
        entry_price = 0
        equity_curve = [capital]

        for i in range(1, len(df)):
            current_signal = df['signal'].iloc[i]
            current_price = df['close'].iloc[i]

            # エントリー
            if current_signal != 0 and position == 0:
                position = current_signal
                entry_price = current_price

                self.trades.append({
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None,
                    'return': None
                })

            # エグジット
            elif position != 0 and (
                current_signal == -position or  # 反対シグナル
                current_signal != 0  # 新しいシグナル
            ):
                # P&L計算
                if position > 0:  # ロングポジション
                    pnl = (current_price - entry_price) * self.pip_value
                else:  # ショートポジション
                    pnl = (entry_price - current_price) * self.pip_value

                # 手数料を差し引く
                pnl -= abs(entry_price * self.commission * self.pip_value * 2)

                capital += pnl
                equity_curve.append(capital)

                # トレード記録を更新
                if self.trades:
                    self.trades[-1].update({
                        'exit_time': df.index[i],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': (pnl / self.initial_capital) * 100
                    })

                position = 0
                entry_price = 0

                # 新しいポジションにエントリー
                if current_signal != 0:
                    position = current_signal
                    entry_price = current_price

                    self.trades.append({
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': None,
                        'return': None
                    })

        # パフォーマンス計算
        performance = self.calculate_performance(equity_curve, df)
        return performance

    def calculate_performance(
        self,
        equity_curve: List[float],
        data: pd.DataFrame
    ) -> Dict:
        """パフォーマンス指標を計算"""

        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        # 完了したトレードのみ
        completed_trades = [t for t in self.trades if t['pnl'] is not None]

        if not completed_trades:
            logger.warning("完了したトレードがありません")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        wins = [t for t in completed_trades if t['pnl'] > 0]
        losses = [t for t in completed_trades if t['pnl'] <= 0]

        total_return = ((equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100

        # 最大ドローダウン
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()

        # シャープレシオ (年率換算)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # プロフィットファクター
        gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # 期間計算
        start_date = data.index[0]
        end_date = data.index[-1]
        days = (end_date - start_date).days
        months = days / 30.0

        # 月次利回り
        monthly_return = (total_return / months) if months > 0 else 0

        performance = {
            'initial_capital': self.initial_capital,
            'final_capital': equity_curve[-1],
            'total_return': round(total_return, 2),
            'monthly_return': round(monthly_return, 2),
            'total_trades': len(completed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round((len(wins) / len(completed_trades)) * 100, 2) if completed_trades else 0,
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(np.mean([t['pnl'] for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t['pnl'] for t in losses]), 2) if losses else 0,
            'largest_win': max([t['pnl'] for t in wins]) if wins else 0,
            'largest_loss': min([t['pnl'] for t in losses]) if losses else 0,
            'test_period_days': days,
            'test_period_months': round(months, 1)
        }

        return performance

    def get_trades_dataframe(self) -> pd.DataFrame:
        """トレード履歴をDataFrameで取得"""
        return pd.DataFrame(self.trades)

    def print_summary(self, performance: Dict):
        """パフォーマンスサマリーを表示"""
        print("\n" + "="*60)
        print(f"バックテスト結果: {self.strategy.name}")
        print("="*60)
        print(f"テスト期間: {performance['test_period_days']}日 ({performance['test_period_months']}ヶ月)")
        print(f"初期資金: ¥{performance['initial_capital']:,.0f}")
        print(f"最終資金: ¥{performance['final_capital']:,.0f}")
        print(f"総利益率: {performance['total_return']:.2f}%")
        print(f"月次平均利益率: {performance['monthly_return']:.2f}%")
        print(f"\n取引統計:")
        print(f"  総取引数: {performance['total_trades']}")
        print(f"  勝ちトレード: {performance['winning_trades']}")
        print(f"  負けトレード: {performance['losing_trades']}")
        print(f"  勝率: {performance['win_rate']:.2f}%")
        print(f"\nリスク指標:")
        print(f"  最大ドローダウン: {performance['max_drawdown']:.2f}%")
        print(f"  シャープレシオ: {performance['sharpe_ratio']:.2f}")
        print(f"  プロフィットファクター: {performance['profit_factor']:.2f}")
        print(f"\n平均損益:")
        print(f"  平均利益: ¥{performance['avg_win']:,.0f}")
        print(f"  平均損失: ¥{performance['avg_loss']:,.0f}")
        print(f"  最大利益: ¥{performance['largest_win']:,.0f}")
        print(f"  最大損失: ¥{performance['largest_loss']:,.0f}")
        print("="*60 + "\n")
