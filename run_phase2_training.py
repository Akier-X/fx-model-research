"""
Phase 2 モデル訓練実行スクリプト

Phase 1.8 vs Phase 2の比較:

Phase 1.8（方向性予測）:
- 目標: 上がるか下がるかを当てる
- 精度: 93.64%
- 評価: 的中率

Phase 2（収益予測）:
- 目標: どれくらい利益が出るかを予測
- 精度: Sharpe Ratio最大化
- 評価: 実現PnL、リターン

このスクリプトでPhase 2モデルを訓練し、
実取引用の収益最適化モデルを構築します。
"""

from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

# Phase 2トレーナー
from src.phase2.phase2_model_trainer import Phase2ModelTrainer


def main():
    """
    Phase 2 モデル訓練メイン処理
    """
    logger.info("="*80)
    logger.info("Phase 2: 収益最適化モデル訓練")
    logger.info("="*80)
    logger.info("")
    logger.info("Phase 1.8との違い:")
    logger.info("  Phase 1.8: 方向性的中率 93.64% （上がるか下がるか）")
    logger.info("  Phase 2:   収益最大化（どれくらい利益が出るか）")
    logger.info("")
    logger.info("Phase 2の目標:")
    logger.info("  - Sharpe Ratio: 25-35")
    logger.info("  - 月利: 10-30%")
    logger.info("  - 最大DD: < 8%")
    logger.info("="*80)
    logger.info("")

    # 対象通貨ペア
    instruments = [
        'USD_JPY',
        # 'EUR_USD',  # 追加で訓練する場合
        # 'GBP_USD',
        # 'AUD_USD',
    ]

    # トレーナー初期化
    trainer = Phase2ModelTrainer(
        instruments=instruments,
        lookback_days=2500,  # 10年分データ
        output_dir=Path("models/phase2")
    )

    all_results = {}

    # 各通貨ペアでモデル訓練
    for instrument in instruments:
        logger.info(f"\n{'='*80}")
        logger.info(f"通貨ペア: {instrument}")
        logger.info(f"{'='*80}")

        try:
            # モデル訓練
            results = trainer.train_models(
                instrument=instrument,
                target_label='label_return_1d'  # 翌日リターン予測
            )

            all_results[instrument] = results

        except Exception as e:
            logger.error(f"訓練エラー ({instrument}): {e}")
            import traceback
            logger.error(traceback.format_exc())

    # 全体サマリー
    logger.info("\n" + "="*80)
    logger.info("Phase 2 訓練完了サマリー")
    logger.info("="*80)

    for instrument, results in all_results.items():
        logger.info(f"\n{instrument}:")

        # ベストモデル
        best_model = max(
            results.keys(),
            key=lambda x: results[x]['val']['sharpe']
        )

        best_result = results[best_model]

        logger.info(f"  ベストモデル: {best_model}")
        logger.info(f"  テスト結果:")
        logger.info(f"    Sharpe Ratio: {best_result['test']['sharpe']:.2f}")
        logger.info(f"    累積リターン: {best_result['test']['cumulative_return']:.2f}%")
        logger.info(f"    勝率: {best_result['test']['win_rate']:.2f}%")
        logger.info(f"    プロフィットファクター: {best_result['test']['profit_factor']:.2f}")
        logger.info(f"    総トレード数: {best_result['test']['total_trades']}")
        logger.info(f"    勝ちトレード: {best_result['test']['winning_trades']}")
        logger.info(f"    負けトレード: {best_result['test']['losing_trades']}")

        # 全モデルのSharpe比較
        logger.info(f"\n  全モデル Sharpe Ratio比較:")
        for model_name in sorted(results.keys(), key=lambda x: results[x]['test']['sharpe'], reverse=True):
            sharpe = results[model_name]['test']['sharpe']
            logger.info(f"    {model_name:20s}: {sharpe:6.2f}")

    logger.info("\n" + "="*80)
    logger.info("保存場所: models/phase2/")
    logger.info("="*80)

    # Phase 1.8 vs Phase 2 比較（概念的）
    logger.info("\n" + "="*80)
    logger.info("Phase 1.8 vs Phase 2 比較")
    logger.info("="*80)
    logger.info("")
    logger.info("Phase 1.8（方向性予測）:")
    logger.info("  - 方向性的中率: 93.64%")
    logger.info("  - 評価指標: 精度、F1スコア")
    logger.info("  - 目的: 上がるか下がるかを当てる")
    logger.info("")
    logger.info("Phase 2（収益予測）:")

    if all_results:
        # 最初の通貨ペアの結果を代表として表示
        first_instrument = list(all_results.keys())[0]
        first_results = all_results[first_instrument]

        best_model_name = max(
            first_results.keys(),
            key=lambda x: first_results[x]['test']['sharpe']
        )
        best = first_results[best_model_name]['test']

        logger.info(f"  - Sharpe Ratio: {best['sharpe']:.2f}")
        logger.info(f"  - 累積リターン: {best['cumulative_return']:.2f}%")
        logger.info(f"  - 勝率: {best['win_rate']:.2f}%")
        logger.info(f"  - 評価指標: Sharpe、リターン、PnL")
        logger.info(f"  - 目的: 利益を最大化する")

    logger.info("")
    logger.info("次のステップ:")
    logger.info("  1. Phase 2モデルをリアルタイム予測エンジンに統合")
    logger.info("  2. シグナル生成ロジック実装")
    logger.info("  3. 自動注文執行システム構築")
    logger.info("="*80)

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("\n🎉 Phase 2 訓練完了！")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
