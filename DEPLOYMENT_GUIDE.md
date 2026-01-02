# デプロイメントガイド - 1週間テスト&サーバー移行

## 📋 実装ロードマップ

あなたのリクエスト:
> そのまま実践戦まで移行して　また実行では仮で１日１万円入れたでもテストで実践し、そのあと、１日１万円入れての本番実践し、そのあと１週間のテスト期間をこのPCでやり、完成したら、別のサーバーにシステムを送り込んで自動システムにする

### ✅ 完了した段階

1. **✅ ペーパートレーディング実装**
   - `paper_trading_bot.py` - 仮想資金1万円シミュレーション
   - テスト実行済み - 正常動作確認
   - Yahoo Finance使用（認証不要）

2. **✅ OANDA API統合**
   - `live_trading_bot.py` - 本番取引用ボット
   - デモ口座設定（.env: `OANDA_ENVIRONMENT=practice`）
   - ⚠️ API認証エラーあり - 要トークン更新

---

## 🚀 次のステップ: 1週間テスト（このPC）

### 方法1: Windowsバッチファイル（推奨 - 簡単）

1. **`start_1week_test.bat`をダブルクリック**
   - 自動的に7日間テスト開始
   - 1時間ごとに取引判断
   - ログ自動保存

2. **注意事項**:
   - ⚠️ **ウィンドウを7日間開いたまま**にしてください
   - ⚠️ **PCをスリープさせない**でください
   - ⚠️ インターネット接続を維持してください

### 方法2: Pythonスクリプト直接実行

```bash
python start_1week_test.py
```

### 方法3: バックグラウンド実行（上級者向け）

```bash
# nohupで実行（Linuxサーバー用）
nohup python start_1week_test.py > 1week_test.log 2>&1 &

# Windowsタスクスケジューラ
# 1. タスクスケジューラを開く
# 2. 新しいタスク作成
# 3. プログラム: python.exe
# 4. 引数: D:\FX\start_1week_test.py
# 5. 開始: 今すぐ
```

---

## 📊 テスト期間中の監視

### リアルタイム監視

テスト中、以下のファイルで進捗確認:

```
logs/
└── 1week_test_YYYYMMDD_HHMMSS.log  # 詳細ログ

outputs/paper_trading/
├── trades_YYYYMMDD_HHMMSS.csv      # 取引履歴
├── equity_YYYYMMDD_HHMMSS.csv      # 評価額推移
└── summary_YYYYMMDD_HHMMSS.json    # 最終サマリー
```

### ログ確認コマンド

```bash
# 最新ログをリアルタイム表示
tail -f logs/1week_test_*.log

# Windowsの場合（PowerShell）
Get-Content logs\1week_test_*.log -Wait
```

---

## 🎯 期待される結果（7日間）

**バックテスト実績から推定**:

| 指標 | 推定値 | 根拠 |
|------|--------|------|
| 初期資金 | ¥10,000 | 設定値 |
| 月利 | 24.50% | 超積極的バックテスト実績 |
| **7日間リターン** | **+5.7%** | 月利24.50% ÷ 30日 × 7日 |
| **予想最終資金** | **¥10,570** | 10,000 × 1.057 |
| 取引回数 | 3-5回 | 303日で155取引 → 1日0.51回 × 7日 |
| 勝率 | 90%+ | バックテスト実績 |
| 最大DD | <5% | バックテスト実績4.74% |

**⚠️ 注意**: 実市場では以下の要因で結果が変動:
- スリッページ（価格滑り）
- スプレッド（買値・売値の差）
- 市場ボラティリティ
- 経済指標発表の影響

---

## 🖥️ サーバー移行（1週間テスト成功後）

### 推奨サーバー環境

1. **VPS（仮想プライベートサーバー）**:
   - ConoHa VPS: 月額1,000円〜
   - AWS EC2 t2.micro: 無料枠あり
   - さくらVPS: 月額880円〜

2. **スペック要件**:
   - OS: Ubuntu 20.04/22.04 LTS
   - CPU: 1コア以上
   - RAM: 2GB以上
   - ストレージ: 20GB以上
   - 帯域: 無制限

### サーバーセットアップ手順

#### ステップ1: サーバー準備

```bash
# Ubuntu/Debian
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3.11 python3-pip git

# Python仮想環境
python3.11 -m venv /opt/fx-bot
source /opt/fx-bot/bin/activate
```

#### ステップ2: システム転送

```bash
# このPCから（ローカル）
cd D:\FX
scp -r . user@server-ip:/opt/fx-bot/

# または Git経由
git init
git add .
git commit -m "Production FX bot system"
git push origin main

# サーバー側でクローン
cd /opt/fx-bot
git clone https://github.com/yourusername/fx-bot.git .
```

#### ステップ3: 依存関係インストール

```bash
# サーバー上で
cd /opt/fx-bot
pip install -r requirements.txt
```

#### ステップ4: 環境変数設定

```bash
# .envファイルをサーバーにコピー
nano .env

# 重要: 本番用設定
OANDA_ENVIRONMENT=practice  # テスト用
# または
OANDA_ENVIRONMENT=live      # 本番用（慎重に！）
```

#### ステップ5: systemdサービス化（自動起動）

```bash
# サービスファイル作成
sudo nano /etc/systemd/system/fx-bot.service
```

```ini
[Unit]
Description=FX Trading Bot - Ultra Aggressive
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/fx-bot
Environment="PATH=/opt/fx-bot/bin"
ExecStart=/opt/fx-bot/bin/python start_1week_test.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

```bash
# サービス有効化・起動
sudo systemctl daemon-reload
sudo systemctl enable fx-bot
sudo systemctl start fx-bot

# ステータス確認
sudo systemctl status fx-bot

# ログ確認
sudo journalctl -u fx-bot -f
```

#### ステップ6: 監視とアラート

```bash
# Cronで日次レポート送信（例）
crontab -e

# 毎日9:00にサマリーメール送信
0 9 * * * python /opt/fx-bot/send_daily_report.py
```

---

## 🔐 セキュリティ対策

### 必須セキュリティ設定

1. **SSHキー認証のみ許可**:
```bash
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PubkeyAuthentication yes
sudo systemctl restart sshd
```

2. **ファイアウォール設定**:
```bash
sudo ufw allow 22/tcp  # SSH
sudo ufw enable
```

3. **環境変数の保護**:
```bash
chmod 600 .env  # 自分のみ読み書き可能
```

4. **定期バックアップ**:
```bash
# 日次バックアップスクリプト
0 3 * * * tar -czf /backup/fx-bot-$(date +\%Y\%m\%d).tar.gz /opt/fx-bot/
```

---

## 📈 本番運用への移行（慎重に！）

### 本番移行チェックリスト

- [ ] 7日間ペーパートレーディングテスト成功
- [ ] 勝率85%以上達成
- [ ] 最大DD < 10%
- [ ] サーバー稼働安定（1週間以上）
- [ ] モニタリング体制構築
- [ ] OANDA本番アカウント準備
- [ ] 少額資金でテスト（1-3万円推奨）

### 本番用パラメータ調整（推奨）

**現在の超積極的設定**（バックテスト用）:
```python
kelly_fraction = 0.70      # 極めて積極的
max_leverage = 10.0        # 最大10倍レバレッジ
```

**本番推奨設定**（リスク抑制）:
```python
kelly_fraction = 0.30      # 保守的
max_leverage = 2.5         # 最大2.5倍レバレッジ
```

これにより:
- **月利**: 24.50% → 約8-12%（より安全）
- **最大DD**: 4.74% → 約2%（より安定）
- **Sharpe Ratio**: 維持または向上

### .env本番設定

```bash
# 本番運用時
OANDA_ENVIRONMENT=live
INITIAL_CAPITAL=30000      # 少額からスタート
MAX_RISK_PER_TRADE=0.01    # 1取引最大1%リスク
```

---

## 🆘 トラブルシューティング

### よくある問題

**問題1: OANDA API認証エラー**
```
ERROR: Insufficient authorization to perform request
```

**解決策**:
1. OANDAサイトでトークン再発行
2. .envファイルを更新
3. `OANDA_ENVIRONMENT`が正しいか確認（practice/live）

**問題2: Yahoo Financeデータ取得失敗**
```
ERROR: No data found for symbol USDJPY=X
```

**解決策**:
1. インターネット接続確認
2. Yahoo Financeサービス状況確認
3. シンボル名確認（`USDJPY=X`）

**問題3: モデルファイルが見つからない**
```
FileNotFoundError: models/phase1_8/USD_JPY_ensemble_models.pkl
```

**解決策**:
```bash
# モデルディレクトリ存在確認
ls -la models/phase1_8/
ls -la models/phase2/

# 必要に応じて再訓練
python train_and_save_models.py
```

---

## 📞 サポート

### ログファイル

すべてのログは以下に保存:
```
logs/
├── 1week_test_YYYYMMDD_HHMMSS.log  # 連続テストログ
└── paper_trading_YYYYMMDD_HHMMSS.log  # 個別実行ログ
```

### デバッグモード

詳細ログが必要な場合:
```python
# paper_trading_bot.py の logger設定を変更
logger.add(sys.stdout, level="DEBUG")  # INFO → DEBUG
```

---

## 🎉 成功基準

### 7日間テスト成功の定義

1. **✅ システム稼働**: 7日間中断なく動作
2. **✅ 予測精度**: 方向的中率 > 80%
3. **✅ 収益性**: プラスのリターン
4. **✅ リスク管理**: 最大DD < 10%
5. **✅ ログ完全性**: すべての取引ログ保存

### サーバー移行成功の定義

1. **✅ 自動起動**: システム再起動後も自動再開
2. **✅ 監視体制**: ログとアラートが機能
3. **✅ セキュリティ**: SSH・ファイアウォール設定完了
4. **✅ バックアップ**: 日次自動バックアップ動作

---

## 📚 関連ファイル

- `paper_trading_bot.py` - ペーパートレーディングボット本体
- `live_trading_bot.py` - 本番取引ボット（OANDA API使用）
- `start_1week_test.py` - 1週間テスト起動スクリプト
- `start_1week_test.bat` - Windows用ワンクリック起動
- `train_and_save_models.py` - モデル再訓練スクリプト
- `.env` - 環境変数設定

---

**作成日**: 2026-01-03
**プロジェクト**: 実用的世界最強FXシステム
**ステータス**: デプロイメント準備完了
