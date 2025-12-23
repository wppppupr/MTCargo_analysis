NAS_DIR  ?= /Volumes/data/Sasaki/backup_git/MTCargo_analysis

.PHONY: sync
sync:
	@echo "🚚 NAS へデータファイル(.npy .npz .mov)を同期します -> $(NAS_DIR)"
	@if [ -z "$(NAS_DIR)" ]; then echo "⚠️ NAS_DIR が設定されていません。"; exit 1; fi
	@mkdir -p "$(NAS_DIR)"
	@rsync -av --prune-empty-dirs \
		--include='*/' \
		--include='*.npy' \
		--include='*.npz' \
		--include='*.mov' \
		--exclude='*' \
		. "$(NAS_DIR)/"
	@echo "✅ 同期完了: $(NAS_DIR)"


.PHONY: save
save:
	@echo "🚀 プロジェクト全体の完全バックアップを開始します..."
	
	@echo "----------------------------------------"
	@echo "1. Git: ソースコードと履歴の保存"
	@echo "----------------------------------------"
	@echo "変更を全てステージング..."
	@git add .
	@echo "変更の有無を確認します..."
	@if git diff --cached --quiet; then \
		echo "⚠️ 変更は検出されませんでした。コミットと push をスキップします。"; \
	else \
		echo "📝 変更をコミットします..."; \
		git commit -m "Auto-save: $$(date '+%Y-%m-%d %H:%M:%S')" || { echo "⚠️ コミットに失敗しました"; exit 1; }; \
		BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
		if [ "$(INTERACTIVE)" = "1" ]; then \
			if [ "$$BRANCH" = "main" ]; then \
				read -p "Push to origin $$BRANCH? [y/N] " RESP; \
				if [ "$$RESP" = "y" ] || [ "$$RESP" = "Y" ]; then git push origin $$BRANCH || echo "⚠️ git push でエラー"; else echo "⚠️ push をスキップしました"; fi; \
			else \
				echo "⚠️ 現在ブランチ: $$BRANCH (自動 push は行いません)"; \
			fi; \
		else \
			if [ "$(PUSH)" = "1" ]; then \
				if [ "$$BRANCH" = "main" ]; then git push origin $$BRANCH || echo "⚠️ git push でエラー"; else echo "⚠️ 現在ブランチ: $$BRANCH - push は main ブランチのみ行います。"; fi; \
			else \
				echo "⚠️ push はデフォルトで無効です。push するには 'make save PUSH=1' を使ってください。"; \
			fi; \
		fi; \
	fi
	
	@echo "----------------------------------------"
	@echo "2. NAS: データファイル(npy/mov)の同期"
	@echo "----------------------------------------"
	@if [ "$(SKIP_SYNC)" = "1" ]; then \
		echo "⚠️ sync をスキップしました (SKIP_SYNC=1)"; \
	else \
		$(MAKE) sync; \
	fi
	
	@echo "✅ 全てのバックアップが完了しました！"