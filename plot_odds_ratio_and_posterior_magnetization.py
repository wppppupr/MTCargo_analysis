#!/usr/bin/env python3
"""
plot_odds_ratio_and_posterior_magnetization.py

横軸を貨物領域における平均磁荷の絶対値 |M| とした以下の2つのグラフを作成するスクリプトです:
1. Odds Ratio (濃縮比 / 相対比率):
       \text{Odds Ratio} = \frac{ P(v > v_c \mid |M|) }{ P(v > v_c) } = \frac{ P(|M| \mid v > v_c) }{ P(|M|) }
   - 1.0 より大きい領域: 全体平均よりも高速走行が促進・濃縮されている領域
   - 1.0 より小さい領域: 高速走行が抑制されている領域

2. 条件付き磁荷分布 (事後分布):
       P(|M| \mid v > v_c) = \frac{ N(v > v_c,\, |M|) }{ N(v > v_c) }
   - 速度が v_c を超えている粒子に限定したとき、それらの粒子がどのような |M| の領域に存在しているかの確率分布
   - 全体背景分布 P(|M|) と比較可能

速度閾値:
    v_c = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] μm/s

出力ファイル:
1. figure/cargo_spin_velocity/odds_ratio_and_posterior_magnetization_2panel.png / .svg
   (左パネル: Odds Ratio, 右パネル: P(|M| | v > v_c))
2. figure/cargo_spin_velocity/odds_ratio_vs_magnetization_abs.png / .svg
   (Odds Ratio 単体図)
3. figure/cargo_spin_velocity/posterior_magnetization_vs_vc.png / .svg
   (条件付き磁荷分布 P(|M| | v > v_c) 単体図)
4. figure/cargo_spin_velocity/odds_ratio_per_condition.png / .svg
   (粒子径別 比較 4パネル図)
5. figure/cargo_spin_velocity/odds_ratio_and_posterior_summary.csv
   (集計サマリーCSV)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ビーズ基本情報
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "label": "0.63 μm", "marker": "^", "color": "#1f77b4"},
    {"name": "beads1um",  "diameter_um": 1.18, "label": "1.18 μm", "marker": "o", "color": "#ff7f0e"},
    {"name": "beads3um",  "diameter_um": 3.37, "label": "3.37 μm", "marker": "d", "color": "#2ca02c"},
]

DEFAULT_VC_LIST = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def apply_custom_style():
    style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
    if style_path.exists():
        try:
            plt.style.use(str(style_path))
        except Exception:
            pass
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'cm'


def load_points_data(csv_path: Path) -> pd.DataFrame:
    """cargo_spin_velocity_points.csv を読み込み前処理"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Points CSV not found: {csv_path}. Run plot_cargo_spin_velocity.py first.")

    df = pd.read_csv(csv_path)
    df['abs_m'] = df['m_ising'].abs()
    valid = np.isfinite(df['m_ising']) & np.isfinite(df['v_um_s']) & (df['abs_m'] <= 1.0)
    df_valid = df[valid].copy()
    print(f"Loaded {len(df_valid)} valid points from {csv_path}")
    return df_valid


def get_vc_colors(vc_list: List[float]) -> List[str]:
    """各 v_c に対応する明瞭なカラーパレット"""
    cmap = matplotlib.colormaps['plasma']
    n = len(vc_list)
    colors = [matplotlib.colors.to_hex(cmap(i / max(n - 0.5, 1))) for i in range(n)]
    return colors


def compute_odds_ratio_and_posterior(
    df: pd.DataFrame,
    vc_list: List[float],
    n_bins: int = 8,
    min_count: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    |M| のビンごとに、
    1. Odds Ratio = P(v > v_c | |M|) / P(v > v_c)
    2. P(|M| | v > v_c)
    3. 全体背景分布 P(|M|)
    を算出
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df_temp = df.copy()
    df_temp['bin_idx'] = pd.cut(df_temp['abs_m'], bins=bins, include_lowest=True, labels=False)

    n_total = len(df_temp)
    counts_m = df_temp.groupby('bin_idx', observed=False).size()
    p_m = counts_m / n_total
    p_m_err = np.sqrt(p_m * (1.0 - p_m) / n_total)

    # 背景分布レコード
    bg_records = []
    for bin_i in range(n_bins):
        m_low, m_high = bins[bin_i], bins[bin_i + 1]
        bin_subset = df_temp[df_temp['bin_idx'] == bin_i]
        n_bin = len(bin_subset)
        m_mean = float(bin_subset['abs_m'].mean()) if n_bin > 0 else (m_low + m_high) / 2.0
        m_median = float(bin_subset['abs_m'].median()) if n_bin > 0 else (m_low + m_high) / 2.0

        bg_records.append({
            "bin_idx": bin_i,
            "m_low": m_low,
            "m_high": m_high,
            "m_center": (m_low + m_high) / 2.0,
            "m_mean": m_mean,
            "m_median": m_median,
            "count_total_in_bin": n_bin,
            "p_m": p_m.get(bin_i, 0.0),
            "p_m_err": p_m_err.get(bin_i, 0.0),
        })
    df_bg = pd.DataFrame(bg_records)

    records = []
    for vc in vc_list:
        n_vc = (df_temp['v_um_s'] > vc).sum()
        p_vc_overall = n_vc / n_total if n_total > 0 else 0.0

        for bin_i in range(n_bins):
            m_low, m_high = bins[bin_i], bins[bin_i + 1]
            bin_subset = df_temp[df_temp['bin_idx'] == bin_i]
            n_bin = len(bin_subset)
            m_mean = float(bin_subset['abs_m'].mean()) if n_bin > 0 else (m_low + m_high) / 2.0
            m_median = float(bin_subset['abs_m'].median()) if n_bin > 0 else (m_low + m_high) / 2.0

            sub_vc = bin_subset[bin_subset['v_um_s'] > vc]
            k_vc = len(sub_vc)

            if n_bin >= min_count and n_vc > 0:
                # 1. P(v > vc | M)
                p_v_given_m = k_vc / n_bin
                p_v_given_m_err = np.sqrt(max(p_v_given_m * (1.0 - p_v_given_m) / n_bin, 1e-6 / n_bin))

                # 2. Odds ratio = P(v > vc | M) / P(v > vc)
                odds_ratio = p_v_given_m / p_vc_overall if p_vc_overall > 0 else np.nan
                # 誤差伝播 (二項比率)
                if odds_ratio > 0:
                    rel_err_sq = (p_v_given_m_err / p_v_given_m)**2 + ((1.0 - p_vc_overall) / (n_total * p_vc_overall))
                    odds_ratio_err = odds_ratio * np.sqrt(rel_err_sq)
                else:
                    odds_ratio_err = np.nan

                # 3. P(M | v > vc) = k_vc / n_vc
                p_m_given_vc = k_vc / n_vc
                p_m_given_vc_err = np.sqrt(max(p_m_given_vc * (1.0 - p_m_given_vc) / n_vc, 1e-6 / n_vc))
            else:
                p_v_given_m = np.nan
                p_v_given_m_err = np.nan
                odds_ratio = np.nan
                odds_ratio_err = np.nan
                p_m_given_vc = np.nan
                p_m_given_vc_err = np.nan

            records.append({
                "bin_idx": bin_i,
                "m_low": m_low,
                "m_high": m_high,
                "m_center": (m_low + m_high) / 2.0,
                "m_mean": m_mean,
                "m_median": m_median,
                "vc": vc,
                "n_total": n_total,
                "n_vc_total": n_vc,
                "p_vc_overall": p_vc_overall,
                "count_total_in_bin": n_bin,
                "count_vc_in_bin": k_vc,
                "p_v_given_m": p_v_given_m,
                "p_v_given_m_err": p_v_given_m_err,
                "odds_ratio": odds_ratio,
                "odds_ratio_err": odds_ratio_err,
                "p_m_given_vc": p_m_given_vc,
                "p_m_given_vc_err": p_m_given_vc_err,
                "p_m_background": p_m.get(bin_i, 0.0),
            })

    return pd.DataFrame(records), df_bg


def plot_odds_ratio_and_posterior_2panel(
    df_summary: pd.DataFrame,
    df_bg: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    title_suffix: str = "All Beads Pooled"
):
    """
    2パネル図:
    左パネル: Odds Ratio = P(v > v_c | |M|) / P(v > v_c)
    右パネル: P(|M| | v > v_c) (背景分布 P(|M|) も参照線として描画)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    # --- 左パネル: Odds Ratio ---
    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['odds_ratio'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        y = sub['odds_ratio']
        yerr = sub['odds_ratio_err']

        ax1.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

    # 基準線 (Odds ratio = 1.0)
    ax1.axhline(1.0, color='#666666', ls='--', lw=1.5, alpha=0.8, label='Baseline (= 1.0)')

    ax1.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12.5)
    ax1.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(v > v_c \mid |M|)}{P(v > v_c)}$", fontsize=13.5)
    ax1.set_title(r"Odds Ratio $\frac{P(v > v_c \mid |M|)}{P(v > v_c)}$", fontsize=13.5, fontweight='bold')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(0.0, 1.6)
    ax1.grid(True, which='major', ls=':', alpha=0.6)
    ax1.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    # --- 右パネル: P(|M| | v > v_c) ---
    # 背景分布 P(|M|)
    ax2.plot(
        df_bg['m_mean'], df_bg['p_m'],
        'k--', lw=2.0, marker='x', ms=7, label=r'Background $P(|M|)$'
    )

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['p_m_given_vc'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        y = sub['p_m_given_vc']
        yerr = sub['p_m_given_vc_err']

        ax2.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

    ax2.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12.5)
    ax2.set_ylabel(r"Conditional Probability $P(|M| \mid v > v_c)$", fontsize=12.5)
    ax2.set_title(r"Conditional Magnetization Distribution $P(|M| \mid v > v_c)$", fontsize=13.5, fontweight='bold')
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 0.9)
    ax2.grid(True, which='major', ls=':', alpha=0.6)
    ax2.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    plt.suptitle(
        rf"Cargo Motion Enrichment & Magnetization Distribution vs $|M|$ ({title_suffix})",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / "odds_ratio_and_posterior_magnetization_2panel.png"
    svg_path = output_dir / "odds_ratio_and_posterior_magnetization_2panel.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_single_odds_ratio(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    title_suffix: str = "All Beads Pooled"
):
    """Odds Ratio 単体図"""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['odds_ratio'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        y = sub['odds_ratio']
        yerr = sub['odds_ratio_err']

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

    ax.axhline(1.0, color='#666666', ls='--', lw=1.5, alpha=0.8, label='Baseline (= 1.0)')

    ax.set_xlabel(r"Magnetization Magnitude in Cargo Region $|M|$", fontsize=12.5)
    ax.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(v > v_c \mid |M|)}{P(v > v_c)}$", fontsize=13.5)
    ax.set_title(rf"Odds Ratio $\frac{{P(v > v_c \mid |M|)}}{{P(v > v_c)}}$ vs Magnetization Magnitude\n({title_suffix})", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.6)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    png_path = output_dir / "odds_ratio_vs_magnetization_abs.png"
    svg_path = output_dir / "odds_ratio_vs_magnetization_abs.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_single_posterior(
    df_summary: pd.DataFrame,
    df_bg: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    title_suffix: str = "All Beads Pooled"
):
    """P(|M| | v > v_c) 単体図"""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    # 背景分布
    ax.plot(
        df_bg['m_mean'], df_bg['p_m'],
        'k--', lw=2.2, marker='x', ms=7, label=r'Background $P(|M|)$'
    )

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['p_m_given_vc'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        y = sub['p_m_given_vc']
        yerr = sub['p_m_given_vc_err']

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

    ax.set_xlabel(r"Magnetization Magnitude in Cargo Region $|M|$", fontsize=12.5)
    ax.set_ylabel(r"Conditional Probability $P(|M| \mid v > v_c)$", fontsize=12.5)
    ax.set_title(rf"Conditional Distribution $P(|M| \mid v > v_c)$ vs Magnetization Magnitude\n({title_suffix})", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.9)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    png_path = output_dir / "posterior_magnetization_vs_vc.png"
    svg_path = output_dir / "posterior_magnetization_vs_vc.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_odds_ratio_per_condition(
    df_all: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    n_bins: int = 8
):
    """粒子径別 (0.63, 1.18, 3.37 um) および 全体プール の Odds Ratio 2x2 パネルプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    conditions = [
        {"key": "beads06um", "title": r"0.63 $\mu\mathrm{m}$ Beads"},
        {"key": "beads1um",  "title": r"1.18 $\mu\mathrm{m}$ Beads"},
        {"key": "beads3um",  "title": r"3.37 $\mu\mathrm{m}$ Beads"},
        {"key": "overall",   "title": "All Beads Pooled"},
    ]

    for idx_panel, cond in enumerate(conditions):
        ax = axes[idx_panel]
        key = cond["key"]
        title = cond["title"]

        if key == "overall":
            sub_df = df_all
        else:
            sub_df = df_all[df_all['bead_name'] == key]

        if sub_df.empty:
            ax.text(0.5, 0.5, f"No Data: {title}", ha='center', va='center', transform=ax.transAxes)
            continue

        df_summary, _ = compute_odds_ratio_and_posterior(sub_df, vc_list, n_bins=n_bins, min_count=3)

        for idx_vc, vc in enumerate(vc_list):
            sub = df_summary[df_summary['vc'] == vc].dropna(subset=['odds_ratio'])
            if sub.empty:
                continue

            color = colors[idx_vc]
            marker = markers[idx_vc % len(markers)]
            x = sub['m_mean']
            y = sub['odds_ratio']
            yerr = sub['odds_ratio_err']

            ax.errorbar(
                x, y, yerr=yerr,
                fmt=f'-{marker}', color=color, lw=1.6, ms=5.5, capsize=3.0, elinewidth=1.2,
                label=f'$v_c = {vc:g}$' if idx_panel == 0 else ""
            )

        ax.axhline(1.0, color='#666666', ls='--', lw=1.4, alpha=0.8)
        ax.set_title(f"{title} ($N={len(sub_df):,}$)", fontsize=13, fontweight='bold')
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.8)

        if idx_panel in [0, 2]:
            ax.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(v > v_c \mid |M|)}{P(v > v_c)}$", fontsize=12)
        if idx_panel in [2, 3]:
            ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12)

    axes[0].legend(title=r"Threshold $v_c$ [$\mu\mathrm{m/s}$]", fontsize=9, title_fontsize=10, loc='upper left', framealpha=0.9)

    plt.suptitle(
        r"Odds Ratio $\frac{P(v > v_c \mid |M|)}{P(v > v_c)}$ across Bead Diameters",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / "odds_ratio_per_condition.png"
    svg_path = output_dir / "odds_ratio_per_condition.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot odds ratio and conditional posterior magnetization distribution")
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity" / "cargo_spin_velocity_points.csv",
        help="Path to cargo_spin_velocity_points.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity",
        help="Output directory"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Number of bins for |M|"
    )
    parser.add_argument(
        "--vc",
        type=float,
        nargs="+",
        default=DEFAULT_VC_LIST,
        help="List of velocity thresholds v_c"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_custom_style()

    # 1. データの読み込み
    df = load_points_data(args.points_csv)

    # 2. 全体プールの計算
    df_summary_overall, df_bg_overall = compute_odds_ratio_and_posterior(
        df, vc_list=args.vc, n_bins=args.bins, min_count=5
    )
    df_summary_overall['condition'] = 'overall_pooled'

    # 粒子径別計算
    all_summaries = [df_summary_overall]
    for b in BEADS_INFO:
        b_name = b["name"]
        sub_df = df[df['bead_name'] == b_name]
        if not sub_df.empty:
            df_b, _ = compute_odds_ratio_and_posterior(sub_df, vc_list=args.vc, n_bins=args.bins, min_count=3)
            df_b['condition'] = b_name
            all_summaries.append(df_b)

    df_summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_csv_path = output_dir / "odds_ratio_and_posterior_summary.csv"
    df_summary_all.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV: {summary_csv_path}")

    # 3. プロット生成
    # (a) メイン2パネル図 (Odds Ratio & P(|M| | v > vc))
    plot_odds_ratio_and_posterior_2panel(df_summary_overall, df_bg_overall, args.vc, output_dir, title_suffix="All Beads Pooled")

    # (b) Odds Ratio 単体図
    plot_single_odds_ratio(df_summary_overall, args.vc, output_dir, title_suffix="All Beads Pooled")

    # (c) P(|M| | v > vc) 単体図
    plot_single_posterior(df_summary_overall, df_bg_overall, args.vc, output_dir, title_suffix="All Beads Pooled")

    # (d) 粒子径別 Odds Ratio 2x2 パネル図
    plot_odds_ratio_per_condition(df, args.vc, output_dir, n_bins=args.bins)

    print("\nOdds ratio and posterior magnetization plots created successfully!")


if __name__ == "__main__":
    main()
