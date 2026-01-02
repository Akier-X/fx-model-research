"""
Phase 1.8 Enhanced - 実際の価格と予測の可視化

実際のUSD/JPY価格推移と予測結果を重ねて表示
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model_builder.phase1_8_enhanced import Phase1_8_Enhanced

print("=" * 70)
print("Phase 1.8 Enhanced - 価格予測グラフ作成")
print("=" * 70)

# Phase 1.8モデルの実行
print("\n[1/3] Phase 1.8 Enhancedモデルを実行中...")
phase1_8 = Phase1_8_Enhanced(
    years_back=10,
    instrument="USD/JPY",
    top_features=60,
    lookahead_days=1,
    threshold=0.005,
    confidence_threshold=0.65
)

results = phase1_8.run()

print("\n[2/3] テストデータと予測結果を取得中...")

# テストデータの取得
test_indices = results['test_indices']
predictions = results['predictions']
actual_labels = results['actual_labels']
probabilities = results['probabilities']

# 実際の価格データ（close price）を取得
actual_prices = phase1_8.price_data.loc[test_indices, 'close']

# 次の日の実際の価格も取得
next_day_prices = []
for idx in test_indices:
    idx_pos = phase1_8.price_data.index.get_loc(idx)
    if idx_pos + 1 < len(phase1_8.price_data):
        next_day_prices.append(phase1_8.price_data.iloc[idx_pos + 1]['close'])
    else:
        next_day_prices.append(np.nan)

next_day_prices = pd.Series(next_day_prices, index=test_indices)

# 予測結果の分類
correct_predictions = (predictions == actual_labels)
predicted_up = (predictions == 1)
predicted_down = (predictions == 0)
no_prediction = (predictions == -1)

print(f"\n予測サマリー:")
print(f"  総テストサンプル数: {len(predictions)}")
print(f"  上昇予測: {predicted_up.sum()}件")
print(f"  下降予測: {predicted_down.sum()}件")
print(f"  見送り: {no_prediction.sum()}件")
print(f"  正解数: {correct_predictions.sum()}件")
print(f"  精度: {100 * correct_predictions.sum() / (len(predictions) - no_prediction.sum()):.2f}%")

print("\n[3/3] グラフを作成中...")

# 出力ディレクトリ作成
output_dir = Path("outputs/phase1_8_enhanced")
output_dir.mkdir(parents=True, exist_ok=True)

# グラフ作成（全期間）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])

# 上段: 価格推移と予測ポイント
# 実際の価格（訓練データ全体）
train_end_idx = test_indices[0]
train_data = phase1_8.price_data.loc[:train_end_idx, 'close']

# 訓練データをプロット
ax1.plot(train_data.index, train_data.values, color='#2c3e50', linewidth=1.5,
         label='訓練データ（実際の価格）', alpha=0.7)

# テストデータの実際の価格をプロット
ax1.plot(actual_prices.index, actual_prices.values, color='#3498db', linewidth=2,
         label='テストデータ（実際の価格）', marker='o', markersize=3)

# 予測開始点を縦線で表示
ax1.axvline(x=test_indices[0], color='green', linestyle='--', linewidth=2,
            label='予測開始点', alpha=0.7)

# 予測結果をプロット
for idx, pred, actual, prob, next_price, current_price in zip(
    test_indices, predictions, actual_labels, probabilities,
    next_day_prices, actual_prices
):
    if pred == -1:  # 見送り
        continue

    is_correct = (pred == actual)

    # 予測方向に応じて矢印を描画
    if pred == 1:  # 上昇予測
        color = '#27ae60' if is_correct else '#e74c3c'  # 緑=正解, 赤=不正解
        arrow_y = current_price + (next_price - current_price) * 0.5 if not pd.isna(next_price) else current_price + 0.5
        ax1.annotate('', xy=(idx, arrow_y), xytext=(idx, current_price),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
        marker = '^'  # 上向き三角
    else:  # 下降予測
        color = '#27ae60' if is_correct else '#e74c3c'
        arrow_y = current_price - (current_price - next_price) * 0.5 if not pd.isna(next_price) else current_price - 0.5
        ax1.annotate('', xy=(idx, arrow_y), xytext=(idx, current_price),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
        marker = 'v'  # 下向き三角

    # 予測ポイントにマーカー
    ax1.scatter(idx, current_price, color=color, marker=marker, s=100,
               edgecolors='black', linewidths=1.5, alpha=0.8, zorder=5)

# 凡例用のダミープロット
ax1.scatter([], [], color='#27ae60', marker='^', s=100, edgecolors='black',
           linewidths=1.5, label='上昇予測（正解）')
ax1.scatter([], [], color='#e74c3c', marker='^', s=100, edgecolors='black',
           linewidths=1.5, label='上昇予測（不正解）')
ax1.scatter([], [], color='#27ae60', marker='v', s=100, edgecolors='black',
           linewidths=1.5, label='下降予測（正解）')
ax1.scatter([], [], color='#e74c3c', marker='v', s=100, edgecolors='black',
           linewidths=1.5, label='下降予測（不正解）')

ax1.set_xlabel('日付', fontsize=13, fontweight='bold')
ax1.set_ylabel('USD/JPY レート', fontsize=13, fontweight='bold')
ax1.set_title('Phase 1.8 Enhanced - 実際の価格推移と方向予測結果', fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=10, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

# 下段: 予測確率
prob_series = pd.Series(probabilities, index=test_indices)
colors_prob = ['#27ae60' if p > 0.65 else ('#e74c3c' if p < 0.35 else '#95a5a6')
               for p in probabilities]

ax2.bar(test_indices, probabilities, color=colors_prob, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.65, color='green', linestyle='--', linewidth=2, label='上昇予測閾値 (0.65)', alpha=0.7)
ax2.axhline(y=0.35, color='red', linestyle='--', linewidth=2, label='下降予測閾値 (0.35)', alpha=0.7)
ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='中立 (0.5)', alpha=0.5)

ax2.set_xlabel('日付', fontsize=13, fontweight='bold')
ax2.set_ylabel('上昇確率', fontsize=13, fontweight='bold')
ax2.set_title('予測確率分布', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "phase1_8_price_prediction_full.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] 全期間グラフ作成完了: {output_dir / 'phase1_8_price_prediction_full.png'}")

# 直近50サンプルの拡大グラフ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])

# 直近50サンプル
recent_indices = test_indices[-50:]
recent_prices = actual_prices.loc[recent_indices]
recent_predictions = predictions[-50:]
recent_actuals = actual_labels[-50:]
recent_probs = probabilities[-50:]
recent_next_prices = next_day_prices.loc[recent_indices]

# 少し前からのデータも表示（コンテキスト用）
context_start = max(0, len(actual_prices) - 60)
context_indices = test_indices[context_start:context_start+10]
context_prices = actual_prices.loc[context_indices]

ax1.plot(context_prices.index, context_prices.values, color='#95a5a6', linewidth=1.5,
         label='以前のテストデータ', alpha=0.5, linestyle='--')

# 直近データをプロット
ax1.plot(recent_prices.index, recent_prices.values, color='#3498db', linewidth=2.5,
         label='直近テストデータ（実際の価格）', marker='o', markersize=5)

# 予測結果をプロット
for idx, pred, actual, prob, next_price, current_price in zip(
    recent_indices, recent_predictions, recent_actuals, recent_probs,
    recent_next_prices, recent_prices
):
    if pred == -1:  # 見送り
        continue

    is_correct = (pred == actual)

    # 予測方向に応じて矢印を描画
    if pred == 1:  # 上昇予測
        color = '#27ae60' if is_correct else '#e74c3c'
        arrow_y = current_price + (next_price - current_price) * 0.5 if not pd.isna(next_price) else current_price + 0.5
        ax1.annotate('', xy=(idx, arrow_y), xytext=(idx, current_price),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.8))
        marker = '^'
    else:  # 下降予測
        color = '#27ae60' if is_correct else '#e74c3c'
        arrow_y = current_price - (current_price - next_price) * 0.5 if not pd.isna(next_price) else current_price - 0.5
        ax1.annotate('', xy=(idx, arrow_y), xytext=(idx, current_price),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.8))
        marker = 'v'

    # 予測ポイントにマーカーと確率を表示
    ax1.scatter(idx, current_price, color=color, marker=marker, s=150,
               edgecolors='black', linewidths=2, alpha=0.9, zorder=5)

    # 確率をテキストで表示
    ax1.text(idx, current_price, f'{prob:.2f}', fontsize=8, ha='center',
            va='bottom' if pred == 1 else 'top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

# 凡例
ax1.scatter([], [], color='#27ae60', marker='^', s=150, edgecolors='black',
           linewidths=2, label='上昇予測（正解）')
ax1.scatter([], [], color='#e74c3c', marker='^', s=150, edgecolors='black',
           linewidths=2, label='上昇予測（不正解）')
ax1.scatter([], [], color='#27ae60', marker='v', s=150, edgecolors='black',
           linewidths=2, label='下降予測（正解）')
ax1.scatter([], [], color='#e74c3c', marker='v', s=150, edgecolors='black',
           linewidths=2, label='下降予測（不正解）')

ax1.set_xlabel('日付', fontsize=13, fontweight='bold')
ax1.set_ylabel('USD/JPY レート', fontsize=13, fontweight='bold')
ax1.set_title('Phase 1.8 Enhanced - 直近50サンプルの詳細予測結果', fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

# 下段: 予測確率（直近50サンプル）
colors_prob = ['#27ae60' if p > 0.65 else ('#e74c3c' if p < 0.35 else '#95a5a6')
               for p in recent_probs]

ax2.bar(recent_indices, recent_probs, color=colors_prob, alpha=0.7, edgecolor='black', linewidth=1)
ax2.axhline(y=0.65, color='green', linestyle='--', linewidth=2, label='上昇予測閾値 (0.65)', alpha=0.7)
ax2.axhline(y=0.35, color='red', linestyle='--', linewidth=2, label='下降予測閾値 (0.35)', alpha=0.7)
ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='中立 (0.5)', alpha=0.5)

# 各バーに確率を表示
for idx, prob in zip(recent_indices, recent_probs):
    ax2.text(idx, prob + 0.02, f'{prob:.2f}', fontsize=8, ha='center',
            rotation=90, fontweight='bold')

ax2.set_xlabel('日付', fontsize=13, fontweight='bold')
ax2.set_ylabel('上昇確率', fontsize=13, fontweight='bold')
ax2.set_title('予測確率分布（直近50サンプル）', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "phase1_8_price_prediction_recent.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] 直近拡大グラフ作成完了: {output_dir / 'phase1_8_price_prediction_recent.png'}")

# 統計サマリー
print("\n" + "=" * 70)
print("Phase 1.8 Enhanced - 価格予測グラフ作成完了！")
print("=" * 70)
print(f"保存先: {output_dir}")
print("\n作成されたグラフ:")
print("  1. phase1_8_price_prediction_full.png   - 全期間の価格推移と予測")
print("  2. phase1_8_price_prediction_recent.png - 直近50サンプルの詳細")
print("=" * 70)
print("\n予測精度サマリー:")
print(f"  方向性的中率: {100 * correct_predictions.sum() / (len(predictions) - no_prediction.sum()):.2f}%")
print(f"  予測カバレッジ: {100 * (len(predictions) - no_prediction.sum()) / len(predictions):.2f}%")
print("=" * 70)
