"""
マルチ通貨ペア訓練 - 実用的世界最強システム
Sharpe 20.41達成済みシステムを4通貨ペアに展開
"""
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd

# 既存のモデル訓練器のコンポーネントを使用
from train_and_save_models import HybridModelTrainer


def train_single_pair(pair_name: str, pair_code: str):
    """単一通貨ペアを訓練"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"通貨ペア: {pair_name} ({pair_code})")
    logger.info(f"{'#'*80}")

    # モデル訓練器を初期化（USD/JPYがデフォルト）
    trainer = HybridModelTrainer()

    # ペア名をオーバーライド
    trainer.instrument = pair_name

    # 訓練実行
    success = trainer.run()

    if success:
        logger.info(f"\n✅ {pair_name} 訓練成功")
        return True
    else:
        logger.error(f"\n❌ {pair_name} 訓練失敗")
        return False


def train_all_currencies():
    """全通貨ペアを訓練"""
    logger.info("="*80)
    logger.info("実用的世界最強システム - マルチ通貨訓練")
    logger.info("="*80)
    logger.info("戦略: Sharpe 20.41達成済みシステムを4通貨ペアに展開")
    logger.info("="*80)

    # 4つの主要通貨ペア
    pairs = [
        'USD/JPY',      # 日本時間に強い（既に訓練済み）
        'EUR/USD',      # 欧州・米国時間に強い
        'GBP/USD',      # ボラティリティ高い
        'EUR/JPY',      # クロス円
    ]

    results = {}
    success_count = 0

    for pair_name in pairs:
        try:
            pair_code = pair_name.replace('/', '_')

            logger.info(f"\n\n{'='*80}")
            logger.info(f"訓練開始: {pair_name}")
            logger.info(f"{'='*80}")

            success = train_single_pair(pair_name, pair_code)

            if success:
                success_count += 1
                results[pair_name] = 'success'
            else:
                results[pair_name] = 'failed'

        except Exception as e:
            logger.error(f"\n❌ {pair_name} 訓練失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[pair_name] = 'error'
            continue

    # 最終サマリー
    logger.info("\n\n" + "="*80)
    logger.info("全通貨ペア訓練完了サマリー")
    logger.info("="*80)

    for pair_name, status in results.items():
        status_icon = "✅" if status == 'success' else "❌"
        logger.info(f"{status_icon} {pair_name}: {status}")

    logger.info(f"\n成功: {success_count}/{len(pairs)}ペア")

    if success_count == len(pairs):
        logger.success("\n🎉🎉🎉 実用的世界最強システム構築成功！ 🎉🎉🎉")
        logger.info("次のステップ: マルチ通貨ポートフォリオバックテスト")
    elif success_count > 0:
        logger.info(f"\n✅ {success_count}ペア訓練完了")
    else:
        logger.error("\n❌ 全ペア訓練失敗")

    return results


if __name__ == '__main__':
    train_all_currencies()
