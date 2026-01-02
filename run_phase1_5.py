"""
Phase 1.5 実行スクリプト

★最重要フェーズ★
回帰 → 分類への戦略転換
方向性的中率 99% を目指す！
"""
from src.model_builder.phase1_5_direction_classifier import Phase1_5_DirectionClassifier
from loguru import logger
import sys

def main():
    """Phase 1.5 を実行"""
    logger.info("="*70)
    logger.info("Phase 1.5: 方向性予測特化モデル（分類） 開始")
    logger.info("="*70)
    logger.info("戦略転換: 回帰 → 分類")
    logger.info("目標: 方向性的中率 99%")
    logger.info("="*70 + "\n")

    # Phase 1.5 実行
    phase1_5 = Phase1_5_DirectionClassifier(
        start_year=2020,
        instrument="USD_JPY",
        top_features=50,  # 上位50特徴量
        lookahead_days=1  # 1日先の方向を予測
    )

    results = phase1_5.run()

    # 結果表示
    metrics = results['metrics']

    logger.info("\n" + "="*70)
    logger.info("Phase 1.5 最終結果")
    logger.info("="*70)
    logger.info(f"方向性的中率:     {metrics['Accuracy']:.2f}%")
    logger.info(f"上昇的中精度:     {metrics['Precision_Up']:.2f}%")
    logger.info(f"下降的中精度:     {metrics['Precision_Down']:.2f}%")
    logger.info(f"上昇再現率:       {metrics['Recall_Up']:.2f}%")
    logger.info(f"下降再現率:       {metrics['Recall_Down']:.2f}%")
    logger.info(f"F1スコア（上昇）: {metrics['F1_Up']:.2f}%")
    logger.info(f"F1スコア（下降）: {metrics['F1_Down']:.2f}%")
    logger.info("="*70)

    logger.info("\n混同行列:")
    logger.info(f"  正解下降・予測下降: {metrics['True_Negatives']}件")
    logger.info(f"  正解下降・予測上昇: {metrics['False_Positives']}件（誤り）")
    logger.info(f"  正解上昇・予測下降: {metrics['False_Negatives']}件（誤り）")
    logger.info(f"  正解上昇・予測上昇: {metrics['True_Positives']}件")

    # これまでのPhaseとの比較
    logger.info("\n" + "="*70)
    logger.info("Phase別 方向性的中率 比較")
    logger.info("="*70)
    logger.info("Phase 1.1 (重み付きアンサンブル):  77.23%")
    logger.info("Phase 1.2 (大量データ):            73.47%")
    logger.info(f"Phase 1.5 (分類モデル):            {metrics['Accuracy']:.2f}%")
    logger.info("="*70)

    improvement_from_11 = metrics['Accuracy'] - 77.23
    improvement_from_12 = metrics['Accuracy'] - 73.47

    logger.info(f"\nPhase 1.1からの改善: {improvement_from_11:+.2f}%")
    logger.info(f"Phase 1.2からの改善: {improvement_from_12:+.2f}%")

    # 目標達成確認
    target_accuracy = 99.0
    actual_accuracy = metrics['Accuracy']

    if actual_accuracy >= target_accuracy:
        logger.success(f"\n★★★ Phase 1 目標達成！ ★★★")
        logger.success(f"方向性的中率 {actual_accuracy:.2f}% >= {target_accuracy}%")
        logger.info("→ Phase 2 へ進む準備ができました")
    elif actual_accuracy >= 95.0:
        logger.info(f"\n素晴らしい結果です！ ({actual_accuracy:.2f}%)")
        logger.info(f"目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ Phase 1.6 で最終調整を行います")
    elif actual_accuracy >= 90.0:
        logger.info(f"\n良好な結果です！ ({actual_accuracy:.2f}%)")
        logger.info(f"目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ さらなるハイパーパラメータ調整が必要です")
    else:
        logger.warning(f"\n目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ 追加の改善策を検討します")

    logger.info(f"\nグラフ: {results['graph_path']}")

    # モデル重み表示
    logger.info("\nアンサンブル重み:")
    for model_name, weight in results['model_weights'].items():
        logger.info(f"  {model_name}: {weight:.4f}")

    logger.info("\n使用特徴量 Top 15:")
    for i, feat in enumerate(results['selected_features'][:15], 1):
        logger.info(f"  {i}. {feat}")

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
