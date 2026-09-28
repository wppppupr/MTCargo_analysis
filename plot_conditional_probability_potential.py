#!/usr/bin/env python3
"""
plot_conditional_probability_potential.py

貨物領域における平均磁荷 M およびその絶対値 |M| に対する
超過確率 P(v > v_c | M) および有効ポテンシャル U(M) = -log P(v > v_c | M)
を算出し、各速度閾値 v_c = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] μm/s
についてプロットするスクリプトです。

出力ファイル:
1. figure/cargo_spin_velocity/conditional_prob_potential_abs_2panel.png / .svg
   (絶対値 |M|: 左パネル P(v > v_c | |M|), 右パネル U(|M|) = -ln P(v > v_c | |M|))
2. figure/cargo_spin_velocity/conditional_probability_vs_magnetization_abs.png / .svg
   (絶対値 |M|: P(v > v_c | |M|) 単体図)
3. figure/cargo_spin_velocity/conditional_potential_vs_magnetization_abs.png / .svg
   (絶対値 |M|: U(|M|) = -ln P(v > v_c | |M|) 単体図)
4. figure/cargo_spin_velocity/conditional_prob_potential_signed_2panel.png / .svg
   (符号付き M: 左パネル P(v > v_c | M), 右パネル U(M) = -ln P(v > v_c | M))
5. figure/cargo_spin_velocity/conditional_potential_vs_magnetization_signed.png / .svg
   (符号付き M: U(M) = -ln P(v > v_c | M) 単体図)
6. figure/cargo_spin_velocity/conditional_prob_potential_per_condition.png / .svg
   (粒子径別 比較 4パネル図)
7. figure/cargo_spin_velocity/conditional_probability_potential_summary.csv
   (集計統計サマリーCSV)
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


def compute_prob_potential(
    df: pd.DataFrame,
    vc_list: List[float],
    col_name: str = 'abs_m',
    n_bins: int = 8,
    min_count: int = 5
) -> pd.DataFrame:
    """
    指定カラム（abs_m または m_ising）のビンごとに、
    P(v > v_c | M) および U(M) = -ln P(v > v_c | M) とその誤差を算出
    """
    if col_name == 'abs_m':
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bins = np.linspace(-1.0, 1.0, n_bins + 1)

    df_temp = df.copy()
    df_temp['bin_idx'] = pd.cut(df_temp[col_name], bins=bins, include_lowest=True, labels=False)

    records = []

    for bin_i in range(n_bins):
        m_low, m_high = bins[bin_i], bins[bin_i + 1]
        bin_subset = df_temp[df_temp['bin_idx'] == bin_i]
        n_total_bin = len(bin_subset)
        m_mean = float(bin_subset[col_name].mean()) if n_total_bin > 0 else (m_low + m_high) / 2.0
        m_median = float(bin_subset[col_name].median()) if n_total_bin > 0 else (m_low + m_high) / 2.0

        for vc in vc_list:
            sub_vc = bin_subset[bin_subset['v_um_s'] > vc]
            k = len(sub_vc)  # 超過数

            if n_total_bin >= min_count and k > 0:
                p = k / n_total_bin
                # 二項分布の標準誤差
                p_err = np.sqrt(max(p * (1.0 - p) / n_total_bin, 1e-6 / n_total_bin))
                # 有効ポテンシャル U = -ln(p)
                u = -np.log(p)
                # 誤差伝播: sigma_u = sigma_p / p = sqrt((1-p)/(n*p))
                u_err = p_err / p
            elif n_total_bin >= min_count and k == 0:
                p = 0.0
                p_err = np.nan
                u = np.nan
                u_err = np.nan
            else:
                p = np.nan
                p_err = np.nan
                u = np.nan
                u_err = np.nan

            records.append({
                "col_name": col_name,
                "bin_idx": bin_i,
                "m_low": m_low,
                "m_high": m_high,
                "m_center": (m_low + m_high) / 2.0,
                "m_mean": m_mean,
                "m_median": m_median,
                "vc": vc,
                "count_total_in_bin": n_total_bin,
                "count_vc": k,
                "prob": p,
                "prob_err": p_err,
                "u_m": u,
                "u_m_err": u_err,
            })

    return pd.DataFrame(records)


def plot_prob_potential_2panel(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    is_abs: bool = True,
    title_suffix: str = "All Beads Pooled"
):
    """
    2パネル図:
    左パネル: P(v > v_c | M)
    右パネル: U(M) = -ln P(v > v_c | M)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    var_symbol = r"|M|" if is_abs else r"M"
    var_label = r"Magnetization Magnitude $|M|$" if is_abs else r"Magnetization $M$"
    tag_str = "abs" if is_abs else "signed"

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['prob'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        p = sub['prob']
        p_err = sub['prob_err']
        u = sub['u_m']
        u_err = sub['u_m_err']

        # 左パネル: P(v > v_c | M)
        ax1.errorbar(
            x, p, yerr=p_err,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

        # 右パネル: U(M) = -ln P
        sub_u = sub.dropna(subset=['u_m'])
        if not sub_u.empty:
            ax2.errorbar(
                sub_u['m_mean'], sub_u['u_m'], yerr=sub_u['u_m_err'],
                fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
                label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
            )

    # 左パネル設定
    ax1.set_xlabel(var_label, fontsize=12.5)
    ax1.set_ylabel(rf"Exceedance Probability $P(v > v_c \mid {var_symbol})$", fontsize=12.5)
    ax1.set_title(rf"Conditional Probability $P(v > v_c \mid {var_symbol})$", fontsize=13.5, fontweight='bold')
    if is_abs:
        ax1.set_xlim(-0.02, 1.02)
    else:
        ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, which='major', ls=':', alpha=0.6)
    ax1.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left' if is_abs else 'upper center', framealpha=0.9)

    # 右パネル設定
    ax2.set_xlabel(var_label, fontsize=12.5)
    ax2.set_ylabel(rf"Effective Potential $U({var_symbol}) = -\ln P(v > v_c \mid {var_symbol})$", fontsize=12.5)
    ax2.set_title(rf"Effective Potential $U({var_symbol}) = -\ln P(v > v_c \mid {var_symbol})$", fontsize=13.5, fontweight='bold')
    if is_abs:
        ax2.set_xlim(-0.02, 1.02)
    else:
        ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(bottom=-0.1)
    ax2.grid(True, which='major', ls=':', alpha=0.6)
    ax2.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc='upper right' if is_abs else 'upper center', framealpha=0.9)

    plt.suptitle(
        rf"Cargo Velocity Exceedance Probability & Effective Potential ({title_suffix})",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / f"conditional_prob_potential_{tag_str}_2panel.png"
    svg_path = output_dir / f"conditional_prob_potential_{tag_str}_2panel.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_single_panel(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    plot_type: str = "probability",  # "probability" or "potential"
    is_abs: bool = True,
    title_suffix: str = "All Beads Pooled"
):
    """単一図プロット"""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    var_symbol = r"|M|" if is_abs else r"M"
    var_label = r"Magnetization Magnitude in Cargo Region $|M|$" if is_abs else r"Magnetization in Cargo Region $M$"
    tag_str = "abs" if is_abs else "signed"

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['prob'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]

        if plot_type == "probability":
            x = sub['m_mean']
            y = sub['prob']
            yerr = sub['prob_err']
            ylabel = rf"Exceedance Probability $P(v > v_c \mid {var_symbol})$"
            title = rf"Exceedance Probability $P(v > v_c \mid {var_symbol})$ vs Magnetization"
            ylim = (-0.02, 1.02)
            fname = f"conditional_probability_vs_magnetization_{tag_str}"
            loc = 'upper left' if is_abs else 'upper center'
        else:
            sub_u = sub.dropna(subset=['u_m'])
            if sub_u.empty:
                continue
            x = sub_u['m_mean']
            y = sub_u['u_m']
            yerr = sub_u['u_m_err']
            ylabel = rf"Effective Potential $U({var_symbol}) = -\ln P(v > v_c \mid {var_symbol})$"
            title = rf"Effective Potential $U({var_symbol}) = -\ln P(v > v_c \mid {var_symbol})$"
            ylim = (-0.1, max(y.max() * 1.15, 4.0))
            fname = f"conditional_potential_vs_magnetization_{tag_str}"
            loc = 'upper right' if is_abs else 'upper center'

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=f'$v_c = {vc:g}\\ \\mu\\mathrm{{m/s}}$'
        )

    ax.set_xlabel(var_label, fontsize=12.5)
    ax.set_ylabel(ylabel, fontsize=12.5)
    ax.set_title(f"{title}\n({title_suffix})", fontsize=13.5, fontweight='bold', pad=12)
    if is_abs:
        ax.set_xlim(-0.02, 1.02)
    else:
        ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(ylim)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(title=r"Threshold $v_c$", fontsize=9.5, title_fontsize=10.5, loc=loc, framealpha=0.9)

    plt.tight_layout()
    png_path = output_dir / f"{fname}.png"
    svg_path = output_dir / f"{fname}.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_per_condition_comparison(
    df_all: pd.DataFrame,
    vc_list: List[float],
    output_dir: Path,
    n_bins: int = 8
):
    """粒子径別 (0.63, 1.18, 3.37 um) および 全体プール の U(|M|) 2x2 パネルプロット"""
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

        df_summary = compute_prob_potential(sub_df, vc_list, col_name='abs_m', n_bins=n_bins, min_count=3)

        for idx_vc, vc in enumerate(vc_list):
            sub = df_summary[df_summary['vc'] == vc].dropna(subset=['u_m'])
            if sub.empty:
                continue

            color = colors[idx_vc]
            marker = markers[idx_vc % len(markers)]
            x = sub['m_mean']
            u = sub['u_m']
            u_err = sub['u_m_err']

            ax.errorbar(
                x, u, yerr=u_err,
                fmt=f'-{marker}', color=color, lw=1.6, ms=5.5, capsize=3.0, elinewidth=1.2,
                label=f'$v_c = {vc:g}$' if idx_panel == 0 else ""
            )

        ax.set_title(f"{title} ($N={len(sub_df):,}$)", fontsize=13, fontweight='bold')
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=-0.1, top=4.5)

        if idx_panel in [0, 2]:
            ax.set_ylabel(r"$U(|M|) = -\ln P(v > v_c \mid |M|)$", fontsize=12)
        if idx_panel in [2, 3]:
            ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12)

    axes[0].legend(title=r"Threshold $v_c$ [$\mu\mathrm{m/s}$]", fontsize=9, title_fontsize=10, loc='upper right', framealpha=0.9)

    plt.suptitle(
        r"Effective Potential $U(|M|) = -\ln P(v > v_c \mid |M|)$ across Bead Diameters",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / "conditional_prob_potential_per_condition.png"
    svg_path = output_dir / "conditional_prob_potential_per_condition.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot conditional probability P(v > vc | M) and effective potential U(M)")
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
        "--bins-abs",
        type=int,
        default=8,
        help="Number of bins for |M|"
    )
    parser.add_argument(
        "--bins-signed",
        type=int,
        default=10,
        help="Number of bins for signed M"
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
    # (a) 絶対値 |M|
    df_summary_abs_overall = compute_prob_potential(
        df, vc_list=args.vc, col_name='abs_m', n_bins=args.bins_abs, min_count=5
    )
    df_summary_abs_overall['condition'] = 'overall_pooled'

    # (b) 符号付き M
    df_summary_signed_overall = compute_prob_potential(
        df, vc_list=args.vc, col_name='m_ising', n_bins=args.bins_signed, min_count=5
    )
    df_summary_signed_overall['condition'] = 'overall_pooled'

    # 粒子径別計算
    all_summaries = [df_summary_abs_overall, df_summary_signed_overall]
    for b in BEADS_INFO:
        b_name = b["name"]
        sub_df = df[df['bead_name'] == b_name]
        if not sub_df.empty:
            df_b_abs = compute_prob_potential(sub_df, vc_list=args.vc, col_name='abs_m', n_bins=args.bins_abs, min_count=3)
            df_b_abs['condition'] = b_name
            df_b_signed = compute_prob_potential(sub_df, vc_list=args.vc, col_name='m_ising', n_bins=args.bins_signed, min_count=3)
            df_b_signed['condition'] = b_name
            all_summaries.extend([df_b_abs, df_b_signed])

    df_summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_csv_path = output_dir / "conditional_probability_potential_summary.csv"
    df_summary_all.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV: {summary_csv_path}")

    # 3. プロット生成
    # (a) 絶対値 |M| 2パネル (P(v > vc | |M|) & U(|M|))
    plot_prob_potential_2panel(df_summary_abs_overall, args.vc, output_dir, is_abs=True, title_suffix="All Beads Pooled")

    # (b) 絶対値 |M| 単体図
    plot_single_panel(df_summary_abs_overall, args.vc, output_dir, plot_type="probability", is_abs=True)
    plot_single_panel(df_summary_abs_overall, args.vc, output_dir, plot_type="potential", is_abs=True)

    # (c) 符号付き M 2パネル (P(v > vc | M) & U(M))
    plot_prob_potential_2panel(df_summary_signed_overall, args.vc, output_dir, is_abs=False, title_suffix="All Beads Pooled")

    # (d) 符号付き M 単体図 (U(M))
    plot_single_panel(df_summary_signed_overall, args.vc, output_dir, plot_type="potential", is_abs=False)

    # (e) 粒子径別 2x2 パネル図
    plot_per_condition_comparison(df, args.vc, output_dir, n_bins=args.bins_abs)

    print("\nAll conditional probability and potential plots created successfully!")


if __name__ == "__main__":
    main()
