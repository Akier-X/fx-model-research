"""
テスト結果ビューア

最新のペーパートレーディング結果を表示します。
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def find_latest_results():
    """最新の結果ファイルを探す"""
    output_dir = Path('outputs/paper_trading')

    if not output_dir.exists():
        print("❌ 結果ファイルが見つかりません")
        return None

    # サマリーファイルを探す
    summary_files = list(output_dir.glob('summary_*.json'))
    if not summary_files:
        print("❌ サマリーファイルが見つかりません")
        return None

    # 最新ファイル
    latest_summary = max(summary_files, key=os.path.getmtime)
    timestamp = latest_summary.stem.replace('summary_', '')

    return {
        'summary': latest_summary,
        'trades': output_dir / f'trades_{timestamp}.csv',
        'equity': output_dir / f'equity_{timestamp}.csv',
        'timestamp': timestamp
    }

def display_results(files):
    """結果を表示"""
    print("\n" + "=" * 80)
    print("📊 ペーパートレーディング結果")
    print("=" * 80)

    # サマリー読み込み
    with open(files['summary'], 'r', encoding='utf-8') as f:
        summary = json.load(f)

    print(f"\n⏰ テスト期間: {summary.get('start_time', 'N/A')} 〜 {summary.get('end_time', 'N/A')}")
    print(f"通貨ペア: {summary.get('pair', 'N/A')}")

    print("\n💰 資金状況:")
    print(f"  初期資金: ¥{summary.get('initial_capital', 0):,.0f}")
    print(f"  最終資金: ¥{summary.get('final_capital', 0):,.2f}")

    pnl = summary.get('total_pnl', 0)
    pnl_pct = summary.get('return_pct', 0)

    if pnl >= 0:
        print(f"  総損益: +¥{pnl:,.2f} (+{pnl_pct:.2f}%) ✅")
    else:
        print(f"  総損益: -¥{abs(pnl):,.2f} ({pnl_pct:.2f}%) ❌")

    print(f"\n📈 取引統計:")
    print(f"  総取引数: {summary.get('total_trades', 0)}回")
    print(f"  勝率: {summary.get('win_rate', 0):.2f}%")
    print(f"  平均利益: ¥{summary.get('avg_profit', 0):,.2f}")
    print(f"  最大利益: ¥{summary.get('max_profit', 0):,.2f}")
    print(f"  最大損失: ¥{summary.get('max_loss', 0):,.2f}")
    print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")
    print(f"  最大DD: {summary.get('max_drawdown_pct', 0):.2f}%")

    # 取引履歴
    if files['trades'].exists():
        trades_df = pd.read_csv(files['trades'])

        if len(trades_df) > 0:
            print(f"\n📋 最近の取引:")
            print("-" * 80)

            # 最新5件表示
            for idx, trade in trades_df.tail(5).iterrows():
                direction = "🟢 LONG" if trade.get('direction', '') == 'LONG' else "🔴 SHORT"
                pnl_trade = trade.get('pnl', 0)
                pnl_sign = "+" if pnl_trade >= 0 else ""

                print(f"{trade.get('entry_time', 'N/A')} | {direction} | "
                      f"エントリー: ¥{trade.get('entry_price', 0):.2f} | "
                      f"決済: ¥{trade.get('exit_price', 0):.2f} | "
                      f"損益: {pnl_sign}¥{pnl_trade:,.2f} ({pnl_sign}{trade.get('pnl_pct', 0):.2f}%)")

    # 評価額推移
    if files['equity'].exists():
        equity_df = pd.read_csv(files['equity'])

        if len(equity_df) > 0:
            print(f"\n📊 評価額推移:")
            print("-" * 80)

            for idx, row in equity_df.tail(10).iterrows():
                equity = row.get('equity', 0)
                change = equity - summary.get('initial_capital', 0)
                change_pct = (change / summary.get('initial_capital', 1)) * 100
                sign = "+" if change >= 0 else ""

                print(f"{row.get('timestamp', 'N/A')} | ¥{equity:,.2f} ({sign}{change_pct:.2f}%)")

    print("\n" + "=" * 80)
    print(f"📁 結果ファイル: {files['summary'].parent}")
    print("=" * 80 + "\n")

def main():
    files = find_latest_results()

    if files:
        display_results(files)
    else:
        print("\n💡 まだテストを実行していません。")
        print("以下のコマンドでテストを開始してください:")
        print("  python start_quick_test.py        # 1時間クイックテスト")
        print("  python start_1week_test.py        # 7日間フルテスト")

if __name__ == '__main__':
    main()
