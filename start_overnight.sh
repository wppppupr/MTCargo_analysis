#!/bin/bash
# ==============================================================================
# MTCargo_analysis: 夜間自律研究ワンライナー起動スクリプト (指示・回数指定対応版)
# ==============================================================================

set -e

REPO_DIR="/home/sasaki/MTCargo_analysis"
cd "$REPO_DIR"

chmod +x "$REPO_DIR/run_overnight.py"

BG_MODE=false
IS_DRY_RUN=false
INSTRUCTION=""
MAX_ITER=""
DEFAULT_ITER=5
EXTRA_ARGS=()

# ヘルプ表示関数
show_help() {
    echo "使用方法: ./start_overnight.sh [オプション]"
    echo ""
    echo "オプション:"
    echo "  -n, --iter, --max-iter <数値>  最大イテレーション数を指定 (デフォルト: $DEFAULT_ITER)"
    echo "  -i, --instruction '指示文'    今夜の解析・改善指示を直接指定"
    echo "  -b, --bg                      バックグラウンド(nohup)で実行"
    echo "  --dry-run                     接続テストと設定確認のみ実行（実際の変更なし）"
    echo "  -h, --help                    このヘルプを表示"
    echo ""
    echo "使用例:"
    echo "  ./start_overnight.sh                      # 対話形式で指示とイテレーション数を入力"
    echo "  ./start_overnight.sh -n 10                # 10イテレーションで実行"
    echo "  ./start_overnight.sh -b -n 8 -i '...'     # 指示と回数(8回)を指定してバックグラウンド実行"
    exit 0
}

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
        -n|--iter|--max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        --dry-run)
            IS_DRY_RUN=true
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# フォアグラウンド実行かつ端末対話可能、かつdry-runでない場合の対話入力
if [ "$BG_MODE" = false ] && [ "$IS_DRY_RUN" = false ] && [ -t 0 ]; then
    # 指示が未指定の場合に対話で確認
    if [ -z "$INSTRUCTION" ]; then
        echo ""
        echo "💡 今夜 Claude & Gemini にやってほしい解析やコード改善はありますか？"
        echo "   （例: plot_velocity_decomposition.py のbeads3umを重点的に解析して）"
        echo "   ※何も入力せず Enter を押すと、OVERNIGHT_INSTRUCTION.md のメモ、またはおまかせモードで動作します。"
        echo -n "📝 リクエスト > "
        read -r USER_INPUT
        if [ -n "$USER_INPUT" ]; then
            INSTRUCTION="$USER_INPUT"
        fi
    fi

    # イテレーション数が未指定の場合に対話で確認
    if [ -z "$MAX_ITER" ]; then
        echo ""
        echo "🔄 最大イテレーション数を指定してください（1サイクルあたり1つの解析・改善・考察を実施）"
        echo -n "🔢 回数 [Enterでデフォルト: $DEFAULT_ITER] > "
        read -r USER_ITER
        if [ -n "$USER_ITER" ]; then
            if [[ "$USER_ITER" =~ ^[0-9]+$ ]] && [ "$USER_ITER" -gt 0 ]; then
                MAX_ITER="$USER_ITER"
            else
                echo "⚠️ 正の整数ではないため、デフォルト値 ($DEFAULT_ITER) を使用します。"
                MAX_ITER="$DEFAULT_ITER"
            fi
        else
            MAX_ITER="$DEFAULT_ITER"
        fi
    fi
fi

# バックグラウンド実行時などで未指定の場合はデフォルト値を採用
if [ -z "$MAX_ITER" ]; then
    MAX_ITER="$DEFAULT_ITER"
fi

# ==============================================================================
# 事前ヘルスチェック（Antigravity / Gemini & Claude Code の認証・疎通確認）
# ==============================================================================
echo "🔍 外部LLMエージェントの疎通・認証確認中..."

# 1. Antigravity (agy / Gemini) の確認
AGY_BIN="/snap/antigravity-cli/24/bin/agy"
if [ ! -x "$AGY_BIN" ]; then
    AGY_BIN=$(which agy 2>/dev/null || echo "")
fi

if [ -n "$AGY_BIN" ]; then
    echo -n "  ✨ Antigravity (Gemini) 接続テスト... "
    # agy -p "ping" で疎通および認証チェック
    if AGY_OUT=$("$AGY_BIN" -p "ping" 2>&1); then
        echo "OK"
    else
        echo "⚠️ 認証待ちまたはエラー検出"
        echo "$AGY_OUT"
        if [ -t 0 ]; then
            echo ""
            echo "👉 バックグラウンド起動前に、対話モードで一度認証を完了してください:"
            "$AGY_BIN" -p "ping" || true
        fi
    fi
else
    echo "  ⚠️ agy コマンドが見つかりません。Geminiコメントはスキップされます。"
fi

# 2. Claude Code の確認
CLAUDE_BIN="/home/sasaki/.local/bin/claude"
if [ ! -x "$CLAUDE_BIN" ]; then
    CLAUDE_BIN=$(which claude 2>/dev/null || echo "")
fi

if [ -n "$CLAUDE_BIN" ]; then
    echo -n "  🤖 Claude Code 接続テスト... "
    if "$CLAUDE_BIN" --version >/dev/null 2>&1; then
        CLAUDE_VER=$("$CLAUDE_BIN" --version 2>/dev/null | head -n 1)
        echo "OK ($CLAUDE_VER)"
    else
        echo "OK (パス確認済み: $CLAUDE_BIN)"
    fi
fi
echo ""

echo "=========================================================="
echo "🌙 夜間自律研究システム (Claude Code & Gemini 協調型)"
echo "=========================================================="
echo "・対象リポジトリ: $REPO_DIR"
echo "・トークン安全弁: 最大${MAX_ITER}イテレーション / 深夜4時自動停止"
echo "・要望メモファイル: OVERNIGHT_INSTRUCTION.md"
echo "・翌朝の確認先: MORNING_REPORT.md"
echo "=========================================================="

# コマンド引数の組み立て
CMD_ARGS=("--max-iter" "$MAX_ITER")
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
