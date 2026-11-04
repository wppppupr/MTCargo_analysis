#!/bin/bash
# ==============================================================================
# MTCargo_analysis: 夜間自律研究ワンライナー起動スクリプト (指示受付対応版)
# ==============================================================================

set -e

REPO_DIR="/home/sasaki/MTCargo_analysis"
cd "$REPO_DIR"

echo "=========================================================="
echo "🌙 夜間自律研究システム (Claude Code & Gemini 協調型)"
echo "=========================================================="
echo "・対象リポジトリ: $REPO_DIR"
echo "・トークン安全弁: 最大3イテレーション / 深夜4時自動停止"
echo "・要望メモファイル: OVERNIGHT_INSTRUCTION.md"
echo "・翌朝の確認先: MORNING_REPORT.md"
echo "=========================================================="

chmod +x "$REPO_DIR/run_overnight.py"

BG_MODE=false
INSTRUCTION=""
EXTRA_ARGS=()

# 引数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--bg)
            BG_MODE=true
            shift
            ;;
        -i|--instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# もし指示が引数で指定されておらず、かつフォアグラウンド実行の場合は対話で聞く
if [ -z "$INSTRUCTION" ] && [ "$BG_MODE" = false ]; then
    echo ""
    echo "💡 今夜 Claude & Gemini にやってほしい解析やコード改善はありますか？"
    echo "   （例: plot_velocity_decomposition.py のbeads3umを重点的に解析して）"
    echo "   ※何も入力せず Enter を押すと、OVERNIGHT_INSTRUCTION.md のメモ、またはおまかせモードで動作します。"
    echo -n "📝 リクエスト > "
    read -r USER_INPUT
    if [ -n "$USER_INPUT" ]; then
        INSTRUCTION="$USER_INPUT"
    fi
    echo ""
fi

# コマンド引数の組み立て
CMD_ARGS=()
if [ -n "$INSTRUCTION" ]; then
    CMD_ARGS+=("-i" "$INSTRUCTION")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [ "$BG_MODE" = true ]; then
    LOG_FILE="$REPO_DIR/overnight.log"
    echo "🚀 バックグラウンド (nohup) で起動します。"
    echo "ログ出力先: $LOG_FILE"
    nohup python3 "$REPO_DIR/run_overnight.py" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "プロセスID (PID): $PID で起動完了！"
    echo "ターミナルを閉じて研究室から帰宅して大丈夫です。お疲れ様でした！"
    echo "進行状況の確認コマンド: tail -f $LOG_FILE"
else
    echo "🚀 フォアグラウンドで起動します。（停止: Ctrl+C）"
    python3 "$REPO_DIR/run_overnight.py" "${CMD_ARGS[@]}"
fi
