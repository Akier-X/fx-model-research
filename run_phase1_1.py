"""
Phase 1.1 実行スクリプト

重み付きアンサンブルで方向性的中率 85% を目指す
"""
from src.model_builder.phase1_1_weighted_ensemble import Phase1_1_WeightedEnsemble
from loguru import logger
import sys

def main():
    """Phase 1.1 を実行"""
    logger.info("Phase 1.1: アンサンブル重み付け最適化 開始\n")

    # Phase 1.1 実行（2020年から150日訓練、150日予測）
    phase1_1 = Phase1_1_WeightedEnsemble(
        start_year=2020,
        train_days=150,
        predict_days=150,
        instrument="USD_JPY",
        top_features=30  # 上位30特徴量のみ使用
    )

    results = phase1_1.run()

    # 結果表示
    metrics = results['metrics']

    logger.info("\n" + "="*70)
    logger.info("Phase 1.1 結果サマリー")
    logger.info("="*70)
    logger.info(f"MAE:            {metrics['MAE']:.4f}")
    logger.info(f"RMSE:           {metrics['RMSE']:.4f}")
    logger.info(f"MAPE:           {metrics['MAPE']:.2f}%")
    logger.info(f"方向性的中率:   {metrics['Direction_Accuracy']:.2f}%")
    logger.info(f"R2スコア:       {metrics['R2_Score']:.4f}")
    logger.info("="*70)

    # 目標達成確認
    target_accuracy = 85.0
    actual_accuracy = metrics['Direction_Accuracy']

    if actual_accuracy >= target_accuracy:
        logger.success(f"\n✓ Phase 1.1 目標達成！ ({actual_accuracy:.2f}% >= {target_accuracy}%)")
        logger.info("→ Phase 1.2 へ進みます")
    else:
        logger.warning(f"\n✗ 目標未達成 ({actual_accuracy:.2f}% < {target_accuracy}%)")
        logger.info("→ パラメータ調整が必要かもしれません")

    logger.info(f"\nグラフ: {results['graph_path']}")

    # モデル重み表示
    logger.info("\nアンサンブル重み:")
    for model_name, weight in results['model_weights'].items():
        logger.info(f"  {model_name}: {weight:.4f}")

    logger.info("\n使用特徴量 Top 10:")
    for feat in results['selected_features'][:10]:
        logger.info(f"  - {feat}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
