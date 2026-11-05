#!/usr/bin/env python3
"""
MTCargo_analysis: 夜間自律研究オーケストレーター (Overnight Research Orchestrator)

【最重要方針：トークン節約 & 翌朝の作業枠温存】
1. 最大イテレーション数を厳格に制限（デフォルト5回、--max-iter / -n で任意指定可能）。
2. 深夜（デフォルト午前4時）までに強制終了し、出社時にはClaudeのレートリミット枠（5時間枠）を満タン回復。
3. エラー修復は最大1回まで。失敗した場合は深追いせず翌朝の人間（あなた）の判断待ちに回す。
4. ユーザーからの解析要望（OVERNIGHT_INSTRUCTION.md または引数）を最優先で実行。
5. 生成されたプロット図（.png）を朝刊レポート（MORNING_REPORT.md）に自動埋め込み。
"""

import os
import sys
import re
import json
import time
import subprocess
import datetime
from pathlib import Path

CONFIG = {
    "MAX_ITERATIONS": 5,
    "STOP_HOUR": 4,             # 午前4時以降は新規イテレーションを開始しない
    "MAX_REPAIR_ATTEMPTS": 1,   # エラー自動修正は最大1回（トークン浪費防止）
    "COOLDOWN_SECONDS": 60,     # イテレーション間隔（秒）
    "WORKDIR": Path(__file__).resolve().parent,
    "REPORT_FILE": Path(__file__).resolve().parent / "MORNING_REPORT.md",
    "INSTRUCTION_FILE": Path(__file__).resolve().parent / "OVERNIGHT_INSTRUCTION.md",
    "CLAUDE_CMD": "/home/sasaki/.local/bin/claude",
    "AGY_CMD": "/snap/antigravity-cli/24/bin/agy",
    "COOLDOWN_HISTORY_LEN": 4   # 直近何回のイテレーションで触ったファイルを再選択禁止にするか
}


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def is_past_stop_time() -> bool:
    """設定した終了時刻（例: 朝4時）を過ぎているか判定"""
    now = datetime.datetime.now()
    if 0 <= now.hour < 12 and now.hour >= CONFIG["STOP_HOUR"]:
        return True
    return False


def run_cmd(cmd: list[str] | str, cwd: Path = None, timeout: int = 600) -> tuple[int, str]:
    """シェルコマンドを安全に実行"""
    if cwd is None:
        cwd = CONFIG["WORKDIR"]
    shell = isinstance(cmd, str)
    try:
        res = subprocess.run(
            cmd,
            cwd=cwd,
            shell=shell,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        output = (res.stdout or "") + (res.stderr or "")
        return res.returncode, output
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -2, str(e)


def call_claude_code(prompt: str, timeout: int = 400) -> tuple[int, str]:
    """
    Claude Code CLI を最小限のトークンで非対話実行
    --dangerously-skip-permissions で夜間の承認待ち停止を防止
    """
    cmd = [
        CONFIG["CLAUDE_CMD"],
        "-p", prompt,
        "--dangerously-skip-permissions"
    ]
    log(f"🤖 Claude Code 呼び出し中... (Prompt長: {len(prompt)}文字)")
    return run_cmd(cmd, timeout=timeout)


def call_gemini(prompt: str, timeout: int = 120) -> tuple[int, str]:
    """
    Antigravity CLI (Gemini) を呼び出して物理的考察を取得
    """
    cmd = [
        CONFIG["AGY_CMD"],
        "-p", prompt
    ]
    log(f"✨ Gemini 呼び出し中... (Prompt長: {len(prompt)}文字)")
    code, out = run_cmd(cmd, timeout=timeout)
    if code != 0 or "Waiting for authentication" in out:
        log(f"⚠️ Gemini CLI呼び出しスキップ (認証待ちまたはエラー code: {code})")
        return code, "⚠️ Gemini認証待ちのためスキップ（ターミナルで agy -p 'hi' で事前認証すると有効化されます）"
    return code, out


def get_git_diff_summary() -> str:
    """直近の未コミット差分をコンパクトに取得"""
    code, out = run_cmd(["git", "diff", "--stat"])
    if not out.strip():
        code, out = run_cmd(["git", "status", "-s"])
    lines = out.strip().splitlines()
    if len(lines) > 50:
        return "\n".join(lines[:50]) + f"\n... (他 {len(lines) - 50} 行省略)"
    return out


def get_user_instructions() -> list[str]:
    """
    OVERNIGHT_INSTRUCTION.md から未完了の要望（- [ ] ...）を抽出
    """
    inst_file = CONFIG["INSTRUCTION_FILE"]
    if not inst_file.exists():
        return []
    
    tasks = []
    with open(inst_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            # - [ ] で始まる行（中身が空でないもの）
            m = re.match(r"^-\s*\[\s*\]\s*(.+)$", line_str)
            if m:
                task_text = m.group(1).strip()
                if task_text:
                    tasks.append(task_text)
    return tasks


def mark_instruction_completed(task_text: str):
    """完了したタスクに [x] を付ける"""
    inst_file = CONFIG["INSTRUCTION_FILE"]
    if not inst_file.exists():
        return
    with open(inst_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # - [ ] task_text を - [x] task_text に置換
    # repl にバックスラッシュ（例: LaTeXの \xi）が含まれても置換エスケープエラーにならないよう関数を渡す
    pattern = re.compile(rf"^(-\s*\[\s*\]\s*{re.escape(task_text)})", re.MULTILINE)
    completed_time = datetime.datetime.now().strftime('%H:%M')
    replacement_str = f"- [x] {task_text} *(完了: {completed_time})*"
    new_content = pattern.sub(lambda _: replacement_str, content)
    
    with open(inst_file, "w", encoding="utf-8") as f:
        f.write(new_content)


def find_recent_figures(minutes_ago: int = 30) -> list[str]:
    """直近生成・更新された画像ファイル（figure/*.png等）を検出"""
    workdir = CONFIG["WORKDIR"]
    fig_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    recent_figs = []
    cutoff = time.time() - (minutes_ago * 60)

    for p in workdir.rglob("*"):
        if p.is_file() and p.suffix.lower() in fig_extensions:
            # .git やキャッシュ等は除外
            if ".git" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                if p.stat().st_mtime >= cutoff:
                    rel_path = p.relative_to(workdir)
                    recent_figs.append(str(rel_path))
            except Exception:
                pass
    return sorted(recent_figs)


def git_commit(iteration: int, message: str):
    """イテレーションごとの進捗を安全にGitコミット"""
    run_cmd(["git", "add", "-A"])
    commit_msg = f"overnight(iter {iteration}): {message}\n\nAutomated by Overnight Research Loop (Claude & Gemini)"
    run_cmd(["git", "commit", "-m", commit_msg])


def append_to_morning_report(entry: str):
    """MORNING_REPORT.md に追記"""
    report_path = CONFIG["REPORT_FILE"]
    header = ""
    if not report_path.exists():
        header = f"# 🌅 Overnight Research Report\n**開始日時**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n"
    
    with open(report_path, "a", encoding="utf-8") as f:
        if header:
            f.write(header)
        f.write(entry + "\n\n")


def get_candidate_scripts(cooldown_files: list[str]) -> list[str]:
    """
    自律改善モードで選定可能な主要解析スクリプト一覧を取得（クールダウン中のファイルを除外）
    """
    workdir = CONFIG["WORKDIR"]
    candidates = []
    # 主要な解析・プロットスクリプトのプレフィックス
    patterns = ["plot_*.py", "calc_*.py", "MSD.py", "MSAD.py", "contrast_analysis.py", 
                "displacement_analysis.py", "spatial_heterogeneity_analysis.py",
                "angular_correlation.py", "benchmark_velocity_methods.py"]
    
    cooldown_set = set(cooldown_files)
    for pat in patterns:
        for p in workdir.glob(pat):
            fname = p.name
            if fname not in cooldown_set and not fname.startswith("test_"):
                candidates.append(fname)
    return sorted(candidates)


def execute_cycle(iteration: int, custom_instruction: str = None, cooldown_files: list[str] = None) -> tuple[dict, list[str]]:
    """
    1イテレーションの実行サイクル
    """
    if cooldown_files is None:
        cooldown_files = []
    log(f"==========================================")
    log(f"🚀 Iteration {iteration} / {CONFIG['MAX_ITERATIONS']} 開始")
    log(f"==========================================")

    record = {
        "iteration": iteration,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "status": "in_progress",
        "instruction": custom_instruction or "自律改善モード（直近差分から推論）",
        "claude_notes": "",
        "gemini_notes": "",
        "generated_figures": [],
        "needs_human_decision": []
    }

    diff_summary = get_git_diff_summary()

    # 指示文の構築（ユーザー指示があれば最優先）
    if custom_instruction:
        task_section = f"""
【最優先：ユーザー（研究者）からの今夜のリクエスト】
「{custom_instruction}」
※この指示に直接従って、必要なコード修正または解析スクリプトの実行を行ってください。
"""
    else:
        cooldown_msg = f"※直近作業済みのため【再編集・微調整を禁止】するファイル: {', '.join(cooldown_files)}" if cooldown_files else ""
        task_section = f"""
【自律発展モード：今夜の研究テーマの深掘り】
{cooldown_msg}

【研究コンテキスト】
本リポジトリは、アクティブマター微小管流動場におけるカーゴ輸送効率 η、有効半径 Rc/ξ、流速相殺、および回転（自転・ロール）効果を解析しています。

【最重要ルール：無駄な解析の厳禁】
1. 研究者の確認負担を増やさないため、今夜の主眼（カーゴ結合効率、流動場相殺、回転効果、仮想トレーサー比較）と無関係なスクリプトをランダムに実行することは【厳禁】です。
2. 単なるコード整形、コメント追加、型アノテーションの追加のみのコミットも【厳禁】です。
3. もし、これ以上新規のプロット作成や検証を行う強い物理的理由・仮説がない場合は、無理に作業せず
   「【完了宣言】本夜の主課題は十分検証されたため自律終了します」
   とだけ報告してください。システムが自動で安全に終了します。
"""

    claude_prompt = f"""
あなたは研究室の夜間自律コーディングパートナーです。
対象リポジトリ: MTCargo_analysis
直近の作業状況・差分:
{diff_summary}
{task_section}

【厳守事項】
1. トークン節約のため、不要な全体探索は行わず、対象スクリプトを直接ピンポイントで実行・編集してください。
2. もし指示されたタスクが完了したら、その結果（生成された図や数値結果）を簡潔にまとめてください。
3. 報告には以下の3点を簡潔（3〜4行）に含めてください：
   - 行った変更または解析内容（物理的狙い）
   - 実行したコマンドとその結果
   - 生成された図やログファイル名（または終了宣言）
"""

    code, claude_output = call_claude_code(claude_prompt)
    record["claude_notes"] = claude_output[-500:] if len(claude_output) > 500 else claude_output

    # 自律完了・終了宣言の検出（無駄な解析防止）
    stop_phrases = ["【完了宣言】", "自律終了", "NO_FURTHER_ANALYSIS", "十分検証されたため終了"]
    record["should_stop"] = any(phrase in claude_output for phrase in stop_phrases)

    # 構文チェック
    check_code, modified_files = run_cmd(["git", "diff", "--name-only"])
    py_files = [f for f in modified_files.splitlines() if f.endswith(".py")]

    syntax_ok = True
    error_log = ""
    for py_file in py_files[:3]:
        ret, out = run_cmd(["python3", "-m", "py_compile", py_file])
        if ret != 0:
            syntax_ok = False
            error_log = f"Syntax error in {py_file}:\n{out[-300:]}"
            break

    # エラー自動修復（最大1回のみ）
    if not syntax_ok and error_log:
        log(f"⚠️ 構文エラーを検出。1回のみ自動修正を試みます...")
        repair_prompt = f"""
直前の作業で構文エラーが発生しました。
エラー内容:
{error_log}

ピンポイントで修正してください。他のファイルはいじらないでください。
"""
        call_claude_code(repair_prompt)
        
        syntax_ok = True
        for py_file in py_files[:3]:
            ret, _ = run_cmd(["python3", "-m", "py_compile", py_file])
            if ret != 0:
                syntax_ok = False
                record["needs_human_decision"].append(f"{py_file} の構文エラー修正が1回で完了しなかったためスキップしました。")
                break

    # 直近生成された図（.png）を収集
    record["generated_figures"] = find_recent_figures(minutes_ago=15)

    # Gemini による物理的考察（コンパクト）
    fig_mention = f"生成された図: {', '.join(record['generated_figures'])}" if record["generated_figures"] else "コード修正/解析"
    gemini_prompt = f"""
MTCargo_analysis 研究プロジェクトの進捗評価です。
実施内容:
{record['claude_notes'][-300:]}
状況: {fig_mention}

この結果がアクティブマター・微小管流動場やカーゴ輸送ダイナミクスにおいて持つ物理的意味、または翌朝人間が確認すべき注目点を2〜3文で簡潔にコメントしてください。
"""
    g_code, gemini_output = call_gemini(gemini_prompt)
    record["gemini_notes"] = gemini_output[-400:] if len(gemini_output) > 400 else gemini_output

    # コミット & レポート追記
    if syntax_ok and (py_files or record["generated_figures"]):
        commit_desc = custom_instruction[:30] if custom_instruction else f"update {', '.join(py_files)}"
        git_commit(iteration, commit_desc)
        record["status"] = "success"
        if custom_instruction:
            mark_instruction_completed(custom_instruction)
    else:
        record["status"] = "completed_without_code_changes"

    # レポート用Markdown作成
    report_entry = f"""### 🔹 Iteration {iteration} ({record['time']}) - 状態: `{record['status']}`
* **リクエスト**: {record['instruction']}
* **変更ファイル**: `{', '.join(py_files) if py_files else 'なし'}`
"""
    if record["generated_figures"]:
        report_entry += "* **📊 生成されたプロット**:\n"
        for fig in record["generated_figures"]:
            report_entry += f"  - ![{fig}]({fig})\n"

    report_entry += f"""* **Claudeの作業報告**:
> {record['claude_notes'][-250:].strip()}
* **Geminiの物理的コメント**:
> {record['gemini_notes'][-250:].strip()}
"""
    if record["needs_human_decision"]:
        report_entry += "* **⚠️ あなた（人間）に判断してほしいこと**:\n"
        for decision in record["needs_human_decision"]:
            report_entry += f"  - [ ] {decision}\n"

    append_to_morning_report(report_entry)
    log(f"✅ Iteration {iteration} 完了")
    return record, py_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTCargo_analysis 夜間自律研究オーケストレーター")
    parser.add_argument("--dry-run", action="store_true", help="動作確認のみ実行")
    parser.add_argument("-n", "--max-iter", type=int, default=CONFIG["MAX_ITERATIONS"], help="最大イテレーション数")
    parser.add_argument("-i", "--instruction", type=str, default=None, help="今夜の解析リクエスト（CLIから直接指定）")
    args = parser.parse_args()

    CONFIG["MAX_ITERATIONS"] = args.max_iter

    log("🌙 夜間自律研究ループを開始します。")
    log(f"・最大イテレーション数: {CONFIG['MAX_ITERATIONS']} 回（トークン保護）")
    log(f"・強制停止時刻: 午前 {CONFIG['STOP_HOUR']} 時（出社時の枠回復を保証）")
    log(f"・朝刊レポート: {CONFIG['REPORT_FILE']}")
    log(f"・微調整ループ防止（直近 {CONFIG['COOLDOWN_HISTORY_LEN']} 回作業ファイルの再選択禁止 & 成果物生成を義務化）")

    # ユーザーリクエストの読み込み
    instructions = []
    if args.instruction:
        instructions.append(args.instruction)
    else:
        instructions = get_user_instructions()

    if instructions:
        log(f"🎯 ユーザーからの解析リクエストを {len(instructions)} 件検出しました：")
        for idx, inst in enumerate(instructions, 1):
            log(f"   {idx}. {inst}")
    else:
        log("ℹ️ 個別の解析リクエストはありません。直近の作業差分をもとに自律改善を行います。")

    if args.dry_run:
        log("🔍 [Dry Run] 接続テストと設定確認のみを実行します...")
        diff = get_git_diff_summary()
        log(f"Git Diff 取得テスト: OK ({len(diff)} chars)")
        candidates = get_candidate_scripts([])
        log(f"解析スクリプト候補テスト: {len(candidates)} 件検出 (例: {', '.join(candidates[:5])})")
        figs = find_recent_figures(60)
        log(f"最近の図の検出テスト: {len(figs)} 件検出")
        log("Dry run 完了。正常に動作可能です。")
        return

    # 朝刊レポート初期化
    req_summary = "\n".join([f"- {t}" for t in instructions]) if instructions else "- 自律改善モード（おまかせ・微調整ループ防止）"
    append_to_morning_report(f"## 📋 運用開始\n- 開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- 今夜のリクエスト:\n{req_summary}")

    cooldown_history: list[str] = []
    has_initial_instructions = bool(instructions)

    for i in range(1, CONFIG["MAX_ITERATIONS"] + 1):
        if is_past_stop_time():
            log(f"⏰ 午前{CONFIG['STOP_HOUR']}時を迎えたため、出社時のトークン枠温存のため自動終了します。")
            append_to_morning_report(f"**⏰ 午前{CONFIG['STOP_HOUR']}時到達**: 翌朝の作業枠温存のため自動終了しました。")
            break

        # 指示リストがあれば順に消費、なければ自動改善
        current_inst = instructions.pop(0) if instructions else None
        record, modified_py = execute_cycle(i, current_inst, cooldown_files=cooldown_history[-CONFIG["COOLDOWN_HISTORY_LEN"]:])

        # 変更されたファイルをクールダウン履歴に追加
        for f in modified_py:
            if f not in cooldown_history:
                cooldown_history.append(f)

        # 1. Claudeからの自律終了宣言を検知した場合
        if record.get("should_stop", False):
            log("🛑 自律終了宣言を検出しました。無駄な解析の量産を防ぐためループを終了します。")
            append_to_morning_report("**🏁 自律終了**: 今夜の課題が十分検証されたため、無駄な解析の量産を防ぎ早期終了しました。")
            break

        # 2. ユーザー指示タスクがすべて完了した場合、無駄なループを回さず安全に早期終了
        if has_initial_instructions and not instructions:
            log("🎉 ユーザーからリクエストされた全タスクが完了しました！無駄な解析を避けるため早期終了します。")
            append_to_morning_report("**🎯 全リクエスト完了**: 指示されたタスクをすべて完了したため、朝の確認負担を減らすためここで安全に終了しました。")
            break

        if i < CONFIG["MAX_ITERATIONS"]:
            log(f"⏳ クールダウン中 ({CONFIG['COOLDOWN_SECONDS']}秒待機)...")
            time.sleep(CONFIG["COOLDOWN_SECONDS"])

    append_to_morning_report("## 🏁 全工程完了\nおはようございます！上記の変更点および生成された図をご確認ください。")
    log("🎉 本夜の全自律研究サイクルが安全に完了しました。お疲れ様でした！")


if __name__ == "__main__":
    main()
