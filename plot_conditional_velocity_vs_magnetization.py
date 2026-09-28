#!/usr/bin/env python3
"""
plot_conditional_velocity_vs_magnetization.py

貨物領域における平均磁荷の絶対値 |M| に対する
速度の条件付き期待値 E[v | |M|, v > v_c] を算出し、
各速度閾値 v_c = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] μm/s
についてプロットするスクリプトです。

出力ファイル:
1. figure/cargo_spin_velocity/conditional_velocity_vs_magnetization_abs.png / .svg
   (全体プール: 横軸 |M|, 縦軸 E[v | |M|, v > v_c], 各 v_c のライン)
2. figure/cargo_spin_velocity/conditional_velocity_vs_magnetization_abs_2panel.png / .svg
   (全体プール: 上段 E[v | |M|, v > v_c], 下段 サンプル数 N)
3. figure/cargo_spin_velocity/conditional_velocity_vs_magnetization_abs_per_condition.png / .svg
   (粒子径別 & 全体プール パネル比較)
4. figure/cargo_spin_velocity/conditional_velocity_vs_magnetization_abs_summary.csv
   (ビン集計データのサマリーCSV)
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
    # 磁荷 M の絶対値
    df['abs_m'] = df['m_ising'].abs()
    # 有効なデータのみフィルタ
    valid = np.isfinite(df['abs_m']) & np.isfinite(df['v_um_s']) & (df['abs_m'] >= 0.0) & (df['abs_m'] <= 1.0)
    df_valid = df[valid].copy()
    print(f"Loaded {len(df_valid)} valid points from {csv_path}")
    return df_valid


def compute_conditional_expectations(
    df: pd.DataFrame,
    vc_list: List[float],
    n_bins: int = 8,
    min_count: int = 3
) -> pd.DataFrame:
    """
    |M| のビンごとに、各 v_c に対する E[v | |M|, v > v_c] を算出
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df_temp = df.copy()
    df_temp['bin_idx'] = pd.cut(df_temp['abs_m'], bins=bins, include_lowest=True, labels=False)

    records = []

    for bin_i in range(n_bins):
        m_low, m_high = bins[bin_i], bins[bin_i + 1]
        bin_subset = df_temp[df_temp['bin_idx'] == bin_i]
        n_total_bin = len(bin_subset)
        m_mean = bin_subset['abs_m'].mean() if n_total_bin > 0 else (m_low + m_high) / 2.0
        m_median = bin_subset['abs_m'].median() if n_total_bin > 0 else (m_low + m_high) / 2.0

        for vc in vc_list:
            sub_vc = bin_subset[bin_subset['v_um_s'] > vc]
            count_vc = len(sub_vc)

            if count_vc >= min_count:
                v_mean = float(sub_vc['v_um_s'].mean())
                v_std = float(sub_vc['v_um_s'].std(ddof=1)) if count_vc > 1 else 0.0
                v_sem = float(v_std / np.sqrt(count_vc))
                v_median = float(sub_vc['v_um_s'].median())
            else:
                v_mean = np.nan
                v_std = np.nan
                v_sem = np.nan
                v_median = np.nan

            records.append({
                "bin_idx": bin_i,
                "m_low": m_low,
                "m_high": m_high,
                "m_center": (m_low + m_high) / 2.0,
                "m_mean": m_mean,
                "m_median": m_median,
                "vc": vc,
                "count_total_in_bin": n_total_bin,
                "count_vc": count_vc,
                "fraction_above_vc": count_vc / n_total_bin if n_total_bin > 0 else 0.0,
                "e_v": v_mean,
                "v_std": v_std,
                "v_sem": v_sem,
                "v_median": v_median,
            })

    return pd.DataFrame(records)


def get_vc_colors(vc_list: List[float]) -> List[str]:
    """各 v_c に対応する明瞭なカラーパレット"""
    cmap = matplotlib.colormaps['plasma']
    n = len(vc_list)
    colors = [matplotlib.colors.to_hex(cmap(i / max(n - 0.5, 1))) for i in range(n)]
    return colors


def plot_conditional_single(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    title_suffix: str = "All Beads Pooled"
):
    """単一パネルのメイン図: 横軸 |M|, 縦軸 E[v | |M|, v > v_c]"""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['e_v'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        y = sub['e_v']
        yerr = sub['v_sem']

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

    ax.set_xlabel(r"Magnetization Magnitude in Cargo Region $|M|$", fontsize=12.5)
    ax.set_ylabel(r"Conditional Expectation $\mathbb{E}[v \mid |M|, v > v_c]$ [$\mu\mathrm{m/s}$]", fontsize=12.5)
    ax.set_title(f"Conditional Velocity Expectation vs Magnetization Magnitude\n({title_suffix})", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(title=r"Threshold $v_c$", fontsize=10, title_fontsize=11, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    png_path = output_dir / "conditional_velocity_vs_magnetization_abs.png"
    svg_path = output_dir / "conditional_velocity_vs_magnetization_abs.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_conditional_2panel(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    title_suffix: str = "All Beads Pooled"
):
    """上段: E[v | |M|, v > v_c], 下段: 有効サンプル数 N(v > v_c)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['e_v'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        y = sub['e_v']
        yerr = sub['v_sem']
        counts = sub['count_vc']

        # 上段: E[v | |M|, v > v_c]
        ax1.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

        # 下段: カウント数
        ax2.plot(
            x, counts,
            f'-{marker}', color=color, lw=1.4, ms=5.0, alpha=0.85
        )

    ax1.set_ylabel(r"$\mathbb{E}[v \mid |M|, v > v_c]$ [$\mu\mathrm{m/s}$]", fontsize=12.5)
    ax1.set_title(f"Conditional Velocity vs Magnetization Magnitude\n({title_suffix})", fontsize=13.5, fontweight='bold')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, which='major', ls=':', alpha=0.6)
    ax1.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    ax2.set_xlabel(r"Magnetization Magnitude in Cargo Region $|M|$", fontsize=12.5)
    ax2.set_ylabel(r"Sample Count $N$", fontsize=11.5)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', ls=':', alpha=0.6)

    plt.tight_layout()
    png_path = output_dir / "conditional_velocity_vs_magnetization_abs_2panel.png"
    svg_path = output_dir / "conditional_velocity_vs_magnetization_abs_2panel.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_conditional_per_condition(
    df_all: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    n_bins: int = 8
):
    """粒子径別 (0.63, 1.18, 3.37 um) および 全体プール の 2x2 パネルプロット"""
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

        df_summary = compute_conditional_expectations(sub_df, vc_list, n_bins=n_bins, min_count=3)

        for idx_vc, vc in enumerate(vc_list):
            sub = df_summary[df_summary['vc'] == vc].dropna(subset=['e_v'])
            if sub.empty:
                continue

            color = colors[idx_vc]
            marker = markers[idx_vc % len(markers)]
            x = sub['m_mean']
            y = sub['e_v']
            yerr = sub['v_sem']

            ax.errorbar(
                x, y, yerr=yerr,
                fmt=f'-{marker}', color=color, lw=1.6, ms=5.5, capsize=3.0, elinewidth=1.2,
                label=f'$v_c = {vc:g}$' if idx_panel == 0 else ""
            )

        ax.set_title(f"{title} ($N={len(sub_df):,}$)", fontsize=13, fontweight='bold')
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0.0, top=0.7)

        if idx_panel in [0, 2]:
            ax.set_ylabel(r"$\mathbb{E}[v \mid |M|, v > v_c]$ [$\mu\mathrm{m/s}$]", fontsize=12)
        if idx_panel in [2, 3]:
            ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12)

    # 1つ目のパネルに凡例を配置
    axes[0].legend(title=r"Threshold $v_c$ [$\mu\mathrm{m/s}$]", fontsize=9, title_fontsize=10, loc='upper left', framealpha=0.9)

    plt.suptitle(
        r"Conditional Velocity Expectation $\mathbb{E}[v \mid |M|, v > v_c]$ across Bead Diameters",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / "conditional_velocity_vs_magnetization_abs_per_condition.png"
    svg_path = output_dir / "conditional_velocity_vs_magnetization_abs_per_condition.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot conditional velocity expectation vs magnetization magnitude")
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

    # 2. 全体プールの条件付き期待値計算
    df_summary_overall = compute_conditional_expectations(
        df,
        vc_list=args.vc,
        n_bins=args.bins,
        min_count=3
    )
    df_summary_overall['condition'] = 'overall_pooled'

    # 各粒子径別の条件付き期待値計算
    all_summaries = [df_summary_overall]
    for b in BEADS_INFO:
        b_name = b["name"]
        sub_df = df[df['bead_name'] == b_name]
        if not sub_df.empty:
            df_b = compute_conditional_expectations(sub_df, vc_list=args.vc, n_bins=args.bins, min_count=3)
            df_b['condition'] = b_name
            all_summaries.append(df_b)

    df_summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_csv_path = output_dir / "conditional_velocity_vs_magnetization_abs_summary.csv"
    df_summary_all.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV: {summary_csv_path}")

    # 3. プロット生成
    # (a) メイン単一パネル図
    plot_conditional_single(df_summary_overall, args.vc, output_dir, title_suffix="All Beads Pooled")

    # (b) 2パネル図（上段: E[v], 下段: サンプル数 N）
    plot_conditional_2panel(df_summary_overall, args.vc, output_dir, title_suffix="All Beads Pooled")

    # (c) 粒子径別 2x2 パネル図
    plot_conditional_per_condition(df, args.vc, output_dir, n_bins=args.bins)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
