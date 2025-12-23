MTCargo_analysis
=================

簡単な説明: ND2 画像の前処理や解析を行うプロジェクトです。

🔧 Makefile (NAS 共有)
----------------------
このプロジェクトには `rsync` を使ってローカルと NAS を同期するための `Makefile` ターゲットがあります。

使い方の例:

- dry-run（差分確認）:

```bash
make dry-run NAS_USER=sasaki NAS_HOST=nas.local NAS_DIR=/mnt/nas/MTCargo_analysis
```

- push（ローカル → NAS）:

```bash
make push NAS_USER=sasaki NAS_HOST=nas.local NAS_DIR=/mnt/nas/MTCargo_analysis SSH_KEY=~/.ssh/id_rsa
```

- pull（NAS → ローカル）:

```bash
make pull NAS_USER=sasaki NAS_HOST=nas.local NAS_DIR=/mnt/nas/MTCargo_analysis
```

その他のターゲット:

- `status` : NAS 側のディレクトリ一覧を表示します（ssh を使用）
- `set-host` : 現在のホスト設定変数を表示します

変数 (Makefile の先頭で設定可能):

- `NAS_USER` (例: sasaki)
- `NAS_HOST` (例: nas.local)
- `NAS_DIR` (例: /mnt/nas/MTCargo_analysis)
- `SSH_PORT` (デフォルト: 22)
- `SSH_KEY` (例: ~/.ssh/id_rsa)
- `EXCLUDES` (同期から除外するパスのリスト)

前提:
- `rsync` と `ssh` が使えること
- macOS の場合は必要に応じて `ssh` のキーや接続許可を事前に設定しておくこと

安全上の注意:
- 重要なデータを上書きする可能性があるため、まず `dry-run` で動作を確認してください。

---

必要なら、sshfs を使った mount/unmount ターゲットの追加や README にスクリーンショットや検証手順を追記します。