"""
Phase 1.2 実行スクリプト

大量訓練データ(500日)で方向性的中率 90% を目指す
"""
from src.model_builder.phase1_2_massive_data import Phase1_2_MassiveData
from loguru import logger
import sys

def main():
    """Phase 1.2 を実行"""
    logger.info("Phase 1.2: 大量訓練データ版 (500日) 開始\n")

    # Phase 1.2 実行（2020年から500日訓練、150日予測）
    phase1_2 = Phase1_2_MassiveData(
        start_year=2020,
        train_days=500,  # ★150日 → 500日★
        predict_days=150,
        instrument="USD_JPY",
        top_features=40  # 上位40特徴量
    )

    results = phase1_2.run()

    # 結果表示
    metrics = results['metrics']

    logger.info("\n" + "="*70)
    logger.info("Phase 1.2 結果サマリー")
    logger.info("="*70)
    logger.info(f"MAE:            {metrics['MAE']:.4f}")
    logger.info(f"RMSE:           {metrics['RMSE']:.4f}")
    logger.info(f"MAPE:           {metrics['MAPE']:.2f}%")
    logger.info(f"方向性的中率:   {metrics['Direction_Accuracy']:.2f}%")
    logger.info(f"R2スコア:       {metrics['R2_Score']:.4f}")
    logger.info("="*70)

    # 前バージョンとの比較
    phase1_1_accuracy = 77.23
    improvement = metrics['Direction_Accuracy'] - phase1_1_accuracy

    logger.info(f"\nPhase 1.1との比較:")
    logger.info(f"  Phase 1.1: {phase1_1_accuracy}%")
    logger.info(f"  Phase 1.2: {metrics['Direction_Accuracy']:.2f}%")
    logger.info(f"  改善: {improvement:+.2f}%")

    # 目標達成確認
    target_accuracy = 90.0
    actual_accuracy = metrics['Direction_Accuracy']

    if actual_accuracy >= target_accuracy:
        logger.success(f"\n✓ Phase 1.2 目標達成！ ({actual_accuracy:.2f}% >= {target_accuracy}%)")
        logger.info("→ Phase 1.3 へ進みます")
    else:
        logger.warning(f"\n目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ さらなる改善が必要です")

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
