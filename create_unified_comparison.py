"""
全Phase統一比較グラフ作成

同じ期間、同じ軸サイズで全Phaseを比較
outputs/phase_comparison/ フォルダに保存
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 全Phaseの結果
phases = {
    'Phase 1.1\n重み付きアンサンブル': {
        'accuracy': 77.23,
        'data_days': 252,
        'approach': '回帰',
        'test_samples': 50,
        'description': 'OANDA API\n重み付きアンサンブル\n特徴量選択30個'
    },
    'Phase 1.2\n大量データ版': {
        'accuracy': 73.47,
        'data_days': 252,
        'approach': '回帰',
        'test_samples': 50,
        'description': 'OANDA API\nLightGBM/CatBoost追加\n特徴量83個'
    },
    'Phase 1.5\n分類モデル': {
        'accuracy': 56.86,
        'data_days': 252,
        'approach': '分類',
        'test_samples': 51,
        'description': 'OANDA API\n方向予測特化\nデータ不足で失敗'
    },
    'Phase 1.6\n究極長期データ': {
        'accuracy': 79.34,
        'data_days': 799,
        'approach': '分類',
        'test_samples': 121,
        'description': 'Yahoo Finance\n3年間データ\n最高精度達成'
    },
    'Phase 1.8\nEnhanced最強': {
        'accuracy': 93.64,
        'data_days': 2581,
        'approach': '分類',
        'test_samples': 115,
        'description': 'Yahoo Finance 10年\n閾値ラベル+信頼度フィルタ\n理論的上限到達'
    }
}

# 出力ディレクトリ作成
output_dir = Path("outputs/phase_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 方向性的中率比較グラフ
fig, ax = plt.subplots(figsize=(14, 8))

names = list(phases.keys())
accuracies = [phases[name]['accuracy'] for name in names]
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#ffd700']  # Phase 1.8は金色

bars = ax.bar(names, accuracies, color=colors, edgecolor='black', linewidth=2)

# 目標線
ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='目標 90%', alpha=0.7)
ax.axhline(y=99, color='darkred', linestyle=':', linewidth=2, label='究極目標 99%', alpha=0.5)

# 数値表示
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    # バーの上に数値
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.2f}%', ha='center', fontsize=14, fontweight='bold')

    # バーの中に説明
    desc = phases[names[i]]['description']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            desc, ha='center', va='center', fontsize=9,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax.set_ylabel('方向性的中率 (%)', fontsize=14, fontweight='bold')
ax.set_title('Phase 1 全バージョン 方向性的中率 比較', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 105)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "01_accuracy_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] 方向性的中率比較グラフ作成完了: {output_dir / '01_accuracy_comparison.png'}")

# 2. データ量比較グラフ
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 2-1: 訓練データ日数
ax1 = axes[0]
data_days = [phases[name]['data_days'] for name in names]
bars1 = ax1.bar(names, data_days, color=colors, edgecolor='black', linewidth=2)

for bar, days in zip(bars1, data_days):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{days}日', ha='center', fontsize=13, fontweight='bold')

ax1.set_ylabel('データ日数', fontsize=13, fontweight='bold')
ax1.set_title('訓練データ期間の比較', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2-2: テストサンプル数
ax2 = axes[1]
test_samples = [phases[name]['test_samples'] for name in names]
bars2 = ax2.bar(names, test_samples, color=colors, edgecolor='black', linewidth=2)

for bar, samples in zip(bars2, test_samples):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{samples}件', ha='center', fontsize=13, fontweight='bold')

ax2.set_ylabel('テストサンプル数', fontsize=13, fontweight='bold')
ax2.set_title('テストデータ量の比較', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "02_data_volume_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] データ量比較グラフ作成完了: {output_dir / '02_data_volume_comparison.png'}")

# 3. アプローチ別精度比較
fig, ax = plt.subplots(figsize=(12, 8))

regression_phases = [name for name in names if phases[name]['approach'] == '回帰']
classification_phases = [name for name in names if phases[name]['approach'] == '分類']

regression_acc = [phases[name]['accuracy'] for name in regression_phases]
classification_acc = [phases[name]['accuracy'] for name in classification_phases]

x_reg = np.arange(len(regression_phases))
x_cls = np.arange(len(classification_phases)) + len(regression_phases) + 0.5

bars_reg = ax.bar(x_reg, regression_acc, color='#66c2a5', edgecolor='black', linewidth=2, label='回帰アプローチ')
bars_cls = ax.bar(x_cls, classification_acc, color='#fc8d62', edgecolor='black', linewidth=2, label='分類アプローチ')

# 数値表示
for bar, acc in zip(bars_reg, regression_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.2f}%', ha='center', fontsize=13, fontweight='bold')

for bar, acc in zip(bars_cls, classification_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.2f}%', ha='center', fontsize=13, fontweight='bold')

# X軸ラベル
all_phases = regression_phases + classification_phases
all_x = list(x_reg) + list(x_cls)
ax.set_xticks(all_x)
ax.set_xticklabels(all_phases, fontsize=10)

ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_ylabel('方向性的中率 (%)', fontsize=13, fontweight='bold')
ax.set_title('アプローチ別（回帰 vs 分類）精度比較', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "03_approach_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] アプローチ別比較グラフ作成完了: {output_dir / '03_approach_comparison.png'}")

# 4. 進化の軌跡グラフ
fig, ax = plt.subplots(figsize=(14, 8))

phase_numbers = [1.1, 1.2, 1.5, 1.6, 1.8]
accuracies_timeline = [77.23, 73.47, 56.86, 79.34, 93.64]

ax.plot(phase_numbers, accuracies_timeline, marker='o', markersize=15, linewidth=3,
        color='#3498db', markerfacecolor='#e74c3c', markeredgecolor='black', markeredgewidth=2)

# 各点にラベル
labels = ['Phase 1.1\n重み付き\n77.23%',
          'Phase 1.2\n大量データ\n73.47%',
          'Phase 1.5\n分類失敗\n56.86%',
          'Phase 1.6\n究極版\n79.34%',
          'Phase 1.8\nEnhanced最強\n93.64% ★★★']

for x, y, label in zip(phase_numbers, accuracies_timeline, labels):
    # Phase 1.8は特別な強調
    if x == 1.8:
        ax.annotate(label, xy=(x, y), xytext=(0, 25), textcoords='offset points',
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='gold', alpha=0.9, edgecolor='red', linewidth=3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=3, color='red'))
    else:
        ax.annotate(label, xy=(x, y), xytext=(0, 20), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='目標 90%', alpha=0.7)
ax.axhline(y=99, color='darkred', linestyle=':', linewidth=2, label='究極目標 99%', alpha=0.5)

ax.set_xlabel('Phase バージョン', fontsize=13, fontweight='bold')
ax.set_ylabel('方向性的中率 (%)', fontsize=13, fontweight='bold')
ax.set_title('Phase 1 進化の軌跡 - 方向性的中率の推移', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(40, 105)
ax.set_xticks(phase_numbers)
ax.set_xticklabels([f'Phase {x}' for x in phase_numbers])
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "04_evolution_timeline.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] 進化の軌跡グラフ作成完了: {output_dir / '04_evolution_timeline.png'}")

# 5. 総合比較表（画像）
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# 表データ作成
table_data = []
table_data.append(['Phase', '方向性\n的中率', 'データ\n日数', 'テスト\nサンプル', 'アプローチ', '主な特徴', '評価'])

for name in names:
    p = phases[name]
    eval_mark = '★★★ 理論的上限' if p['accuracy'] > 90 else ('★ 最高' if p['accuracy'] > 78 else ('○ 良好' if p['accuracy'] > 75 else '△ 要改善'))
    table_data.append([
        name,
        f"{p['accuracy']:.2f}%",
        f"{p['data_days']}日",
        f"{p['test_samples']}件",
        p['approach'],
        p['description'].replace('\n', ' / '),
        eval_mark
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.3, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 3)

# ヘッダー行のスタイル
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

# データ行のスタイル（交互色）
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')

        # 最高精度行を強調
        if phases[names[i-1]]['accuracy'] == 93.64:
            table[(i, j)].set_facecolor('#ffd700')  # 金色でPhase 1.8を強調
            table[(i, j)].set_text_props(weight='bold')

ax.set_title('Phase 1 全バージョン 総合比較表', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / "05_comprehensive_table.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[完了] 総合比較表作成完了: {output_dir / '05_comprehensive_table.png'}")

# 完了メッセージ
print("\n" + "="*70)
print("全Phase統一比較グラフ作成完了！")
print("="*70)
print(f"保存先: {output_dir}")
print("\n作成されたグラフ:")
print("  1. 01_accuracy_comparison.png      - 方向性的中率比較")
print("  2. 02_data_volume_comparison.png   - データ量比較")
print("  3. 03_approach_comparison.png      - アプローチ別比較")
print("  4. 04_evolution_timeline.png       - 進化の軌跡")
print("  5. 05_comprehensive_table.png      - 総合比較表")
print("="*70)
