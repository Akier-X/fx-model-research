"""
Phase 1.8 Enhanced 実行スクリプト

3大改善策を統合した最強モデル:
1. 閾値ベースラベル (±0.5%以上の変動のみ)
2. 10年分データ (Yahoo Finance)
3. 信頼度フィルタリング (確率0.65以上)

目標: 方向性的中率 85-90%
"""
from src.model_builder.phase1_8_enhanced import Phase1_8_Enhanced
from loguru import logger
import sys

def main():
    """Phase 1.8 Enhanced を実行"""
    logger.info("=" * 70)
    logger.info("Phase 1.8 Enhanced: 最強モデル 開始")
    logger.info("=" * 70)
    logger.info("改善策:")
    logger.info("  1. 閾値ベースラベル (±0.5%以上)")
    logger.info("  2. 10年分データ (Yahoo Finance)")
    logger.info("  3. 信頼度フィルタリング (確率0.65以上)")
    logger.info("\n目標精度: 85-90%")
    logger.info("=" * 70 + "\n")

    # Phase 1.8 実行
    phase1_8 = Phase1_8_Enhanced(
        years_back=10,                    # 過去10年のデータ
        instrument="USD/JPY",
        top_features=60,                  # 上位60特徴量
        lookahead_days=1,                 # 1日先予測
        threshold=0.005,                  # ラベル閾値 ±0.5%
        confidence_threshold=0.65         # 信頼度閾値 0.65
    )

    results = phase1_8.run()

    # 結果表示
    metrics = results['metrics']

    logger.info("\n" + "=" * 70)
    logger.info("🎯 Phase 1.8 Enhanced 最終結果")
    logger.info("=" * 70)
    logger.info(f"方向性的中率:     {metrics['Accuracy']:.2f}%  ⭐⭐⭐")
    logger.info(f"カバー率:         {metrics['Coverage']:.2f}%")
    logger.info(f"見送り件数:       {metrics['Skipped']}件")
    logger.info("=" * 70)
    logger.info(f"上昇的中精度:     {metrics['Precision_Up']:.2f}%")
    logger.info(f"下降的中精度:     {metrics['Precision_Down']:.2f}%")
    logger.info(f"上昇再現率:       {metrics['Recall_Up']:.2f}%")
    logger.info(f"下降再現率:       {metrics['Recall_Down']:.2f}%")
    logger.info(f"F1スコア（上昇）: {metrics['F1_Up']:.2f}%")
    logger.info(f"F1スコア（下降）: {metrics['F1_Down']:.2f}%")
    logger.info("=" * 70)

    logger.info("\n混同行列:")
    logger.info(f"  正解下降・予測下降: {metrics['True_Negatives']}件")
    logger.info(f"  正解下降・予測上昇: {metrics['False_Positives']}件（誤り）")
    logger.info(f"  正解上昇・予測下降: {metrics['False_Negatives']}件（誤り）")
    logger.info(f"  正解上昇・予測上昇: {metrics['True_Positives']}件")

    # 全Phase比較
    logger.info("\n" + "=" * 70)
    logger.info("📊 全Phase 方向性的中率 比較")
    logger.info("=" * 70)
    logger.info("Phase 1.1 (重み付きアンサンブル):  77.23%")
    logger.info("Phase 1.2 (大量データ):            73.47%")
    logger.info("Phase 1.5 (分類モデル):            56.86%")
    logger.info("Phase 1.6 (究極長期データ):        79.34%")
    logger.info(f"Phase 1.8 (Enhanced 最強):        {metrics['Accuracy']:.2f}%  ⭐ NEW!")
    logger.info("=" * 70)

    # 改善幅
    improvement_from_16 = metrics['Accuracy'] - 79.34

    if improvement_from_16 > 0:
        logger.info(f"\n✅ Phase 1.6からの改善: +{improvement_from_16:.2f}%")
    else:
        logger.info(f"\n⚠️ Phase 1.6からの変化: {improvement_from_16:.2f}%")

    # 目標達成確認
    target_min = 85.0
    target_max = 90.0
    actual_accuracy = metrics['Accuracy']

    if actual_accuracy >= target_min and actual_accuracy <= target_max:
        logger.success(f"\n🎉🎉🎉 目標達成！ 🎉🎉🎉")
        logger.success(f"方向性的中率 {actual_accuracy:.2f}% (目標: {target_min}-{target_max}%)")
        logger.info("→ Phase 2 へ進む準備完了！")
    elif actual_accuracy > target_max:
        logger.success(f"\n⭐⭐⭐ 目標超過達成！ ⭐⭐⭐")
        logger.success(f"方向性的中率 {actual_accuracy:.2f}% > 目標{target_max}%")
        logger.info("→ 理論的上限に到達！Phase 2へ")
    elif actual_accuracy >= 80.0:
        logger.info(f"\n優秀な結果です！ ({actual_accuracy:.2f}%)")
        logger.info(f"目標まであと {target_min - actual_accuracy:.2f}%")
        logger.info("→ 実用レベル達成、Phase 2でさらに改善可能")
    else:
        logger.warning(f"\n目標まであと {target_min - actual_accuracy:.2f}%")
        logger.info("→ さらなる改善が必要")
        logger.info("→ docs/IMPROVEMENT_IMPLEMENTATION_GUIDE.md を参照")

    logger.info(f"\n📊 グラフ: {results['graph_path']}")

    # モデル重み表示
    logger.info("\n🔧 アンサンブル重み:")
    for model_name, weight in results['model_weights'].items():
        logger.info(f"  {model_name}: {weight:.4f}")

    # 使用特徴量 Top 15
    logger.info("\n📈 重要特徴量 Top 15:")
    for i, feat in enumerate(results['selected_features'][:15], 1):
        importance = results['feature_importances'][feat]
        logger.info(f"  {i:2d}. {feat:30s}: {importance:.4f}")

    # 改善提案
    logger.info("\n" + "=" * 70)
    logger.info("💡 さらなる改善のヒント")
    logger.info("=" * 70)

    if actual_accuracy < target_min:
        logger.info("1. ラベル閾値を調整してみる（0.007 = 0.7%に上げる）")
        logger.info("2. 信頼度閾値を調整してみる（0.70に上げる）")
        logger.info("3. COTレポートを追加する")
        logger.info("4. センチメント分析を追加する")
    else:
        logger.info("✅ 既に目標精度を達成！")
        logger.info("→ Phase 2でリスク管理と収益性最適化に注力")

    logger.info("\n詳細: docs/IMPROVEMENT_IMPLEMENTATION_GUIDE.md")
    logger.info("=" * 70)

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
