NAS_DIR  ?= /Volumes/data/Sasaki/backup_git/MTCargo_analysis

.PHONY: save
save:
	@echo "🚀 プロジェクト全体の完全バックアップを開始します..."
	
	@echo "----------------------------------------"
	@echo "1. Git: ソースコードと履歴の保存"
	@echo "----------------------------------------"
	# 変更を全てステージング
	git add .
	# 日付入りで自動コミット (変更がない場合はエラーにせず通過させる '|| true')
	git commit -m "Auto-save: $$(date '+%Y-%m-%d %H:%M:%S')" || echo "⚠️ コミットする変更はありませんでした。"
	# GitHub (またはNASのGitリポジトリ) へ送信
	git push origin main
	
	@echo "----------------------------------------"
	@echo "2. NAS: データファイル(npy/mov)の同期"
	@echo "----------------------------------------"
	# 既存の sync タスクを呼び出す
	$(MAKE) sync
	
	@echo "✅ 全てのバックアップが完了しました！"
