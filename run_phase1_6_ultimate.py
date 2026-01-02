"""
Phase 1.6 Ultimate 実行スクリプト

Yahoo Finance 長期データ（3年+）で最強モデル構築
目標: 方向性的中率 90%+
"""
from src.model_builder.phase1_6_ultimate_longterm import Phase1_6_UltimateLongTerm
from loguru import logger
import sys

def main():
    """Phase 1.6 Ultimate を実行"""
    logger.info("="*70)
    logger.info("Phase 1.6: 究極の長期データモデル 開始")
    logger.info("="*70)
    logger.info("データソース: Yahoo Finance (3年+)")
    logger.info("目標: 方向性的中率 90%+")
    logger.info("="*70 + "\n")

    # Phase 1.6 実行
    phase1_6 = Phase1_6_UltimateLongTerm(
        years_back=3,          # 過去3年のデータ
        instrument="USD/JPY",
        top_features=60,       # 上位60特徴量
        lookahead_days=1       # 1日先予測
    )

    results = phase1_6.run()

    # 結果表示
    metrics = results['metrics']

    logger.info("\n" + "="*70)
    logger.info("Phase 1.6 最終結果")
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

    # 全Phase比較
    logger.info("\n" + "="*70)
    logger.info("全Phase 方向性的中率 比較")
    logger.info("="*70)
    logger.info("Phase 1.1 (重み付きアンサンブル):  77.23%")
    logger.info("Phase 1.2 (大量データ):            73.47%")
    logger.info("Phase 1.5 (分類モデル):            56.86%")
    logger.info(f"Phase 1.6 (究極長期データ):        {metrics['Accuracy']:.2f}%")
    logger.info("="*70)

    improvement_from_best = metrics['Accuracy'] - 77.23

    logger.info(f"\n最良Phase 1.1からの改善: {improvement_from_best:+.2f}%")

    # 目標達成確認
    target_accuracy = 90.0
    actual_accuracy = metrics['Accuracy']

    if actual_accuracy >= target_accuracy:
        logger.success(f"\n★★★ Phase 1 究極目標達成！ ★★★")
        logger.success(f"方向性的中率 {actual_accuracy:.2f}% >= {target_accuracy}%")
        logger.info("→ Phase 2 へ進む準備完了！")
    elif actual_accuracy >= 85.0:
        logger.info(f"\n優秀な結果です！ ({actual_accuracy:.2f}%)")
        logger.info(f"目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ 実用レベル達成")
    elif actual_accuracy >= 80.0:
        logger.info(f"\n良好な結果です！ ({actual_accuracy:.2f}%)")
        logger.info(f"目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ Phase 2でさらに改善可能")
    else:
        logger.warning(f"\n目標まであと {target_accuracy - actual_accuracy:.2f}%")
        logger.info("→ さらなる改善が必要")

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
