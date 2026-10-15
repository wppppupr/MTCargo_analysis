#!/usr/bin/env python3
"""
plot_conditional_probability_potential.py

貨物領域における平均磁荷 M およびその絶対値 |M| に対する
無次元化超過確率 P(\tilde{v}_\parallel > \tilde{v}_c | M)
および有効ポテンシャル U(M) = -ln P(\tilde{v}_\parallel > \tilde{v}_c | M)
を算出し、各速度閾値 \tilde{v}_c = [0.1, 0.5, 0.8, 1.0]
についてプロットするスクリプトです。

出力ファイル（ローカルおよび root_dir / NAS-Ebanaru の両方に保存）:
1. figure/cargo_spin_velocity/conditional_prob_potential_per_condition.png / .svg
   (全6粒子径別 比較 2x3 パネル図: 0.63, 1.18, 3.37, 5.00, 7.24, 20.0 um)
2. figure/cargo_spin_velocity/conditional_prob_potential_small_beads_pooled_2panel.png / .svg
   (Small Beads Pooled (0.6, 1, 3 um): 左パネル P, 右パネル U(|M|))
3. figure/cargo_spin_velocity/conditional_potential_small_beads_pooled.png / .svg
   (Small Beads Pooled: U(|M|) 単体図)
4. figure/cargo_spin_velocity/conditional_prob_potential_large_beads_pooled_2panel.png / .svg
   (Large Beads Pooled (5, 7, 20 um): 左パネル P, 右パネル U(|M|))
5. figure/cargo_spin_velocity/conditional_potential_large_beads_pooled.png / .svg
   (Large Beads Pooled: U(|M|) 単体図)
6. figure/cargo_spin_velocity/conditional_potential_small_vs_large_beads_pooled_2panel.png / .svg
   (Small vs Large Beads Pooled: U(|M|) 2パネル比較図)
7. figure/cargo_spin_velocity/conditional_prob_potential_abs_2panel.png / .svg
   (All Beads Pooled: 左パネル P, 右パネル U(|M|))
8. figure/cargo_spin_velocity/conditional_potential_vs_magnetization_abs.png / .svg
   (All Beads Pooled: U(|M|) 単体図)
9. figure/cargo_spin_velocity/conditional_prob_potential_signed_2panel.png / .svg
   (符号付き M: 左パネル P, 右パネル U(M))
10. figure/cargo_spin_velocity/conditional_potential_vs_magnetization_signed.png / .svg
    (符号付き M: U(M) 単体図)
11. figure/cargo_spin_velocity/conditional_probability_potential_summary.csv
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

# ビーズ基本情報 (全粒子径)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "label": "0.63 μm", "marker": "^", "color": "#1f77b4"},
    {"name": "beads1um",  "diameter_um": 1.18, "label": "1.18 μm", "marker": "o", "color": "#ff7f0e"},
    {"name": "beads3um",  "diameter_um": 3.37, "label": "3.37 μm", "marker": "d", "color": "#2ca02c"},
    {"name": "beads5um",  "diameter_um": 5.00, "label": "5.00 μm", "marker": "p", "color": "#d62728"},
    {"name": "beads7um",  "diameter_um": 7.24, "label": "7.24 μm", "marker": "h", "color": "#9467bd"},
    {"name": "beads20um", "diameter_um": 20.0, "label": "20.0 μm", "marker": "s", "color": "#8c564b"},
]

DEFAULT_VC_LIST = [0.1, 0.5, 0.8, 1.0]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root() -> Optional[Path]:
    """存在するデータルートを返す"""
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um']:
                if (r / b).exists():
                    return r
    return None


def apply_custom_style():
    style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
    if style_path.exists():
        try:
            plt.style.use(str(style_path))
        except Exception:
            pass
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'cm'


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300):
    """指定されたすべての出力ディレクトリに png と svg を保存する"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / f"{basename}.png"
        svg_path = d / f"{basename}.svg"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {basename}.png / .svg -> {len(out_dirs)} dir(s)")


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]):
    """指定されたすべての出力ディレクトリに CSV を保存する"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / f"{basename}.csv"
        df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {basename}.csv -> {len(out_dirs)} dir(s)")


def load_points_data(csv_path: Path, vel_col: str = 'v_parallel_tilde') -> pd.DataFrame:
    """cargo_spin_velocity_points.csv を読み込み前処理"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Points CSV not found: {csv_path}. Run plot_cargo_spin_velocity.py first.")

    df = pd.read_csv(csv_path)
    df['abs_m'] = df['m_ising'].abs()

    if vel_col not in df.columns:
        if 'v_tilde' in df.columns:
            vel_col = 'v_tilde'
        elif 'v_um_s' in df.columns:
            vel_col = 'v_um_s'
    df['target_vel'] = df[vel_col]

    valid = np.isfinite(df['m_ising']) & np.isfinite(df['target_vel']) & (df['abs_m'] <= 1.0)
    df_valid = df[valid].copy()
    print(f"Loaded {len(df_valid)} valid points from {csv_path} (velocity column: {vel_col})")
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
    P(\tilde{v}_\parallel > \tilde{v}_c | M) および U(M) = -ln P(\tilde{v}_\parallel > \tilde{v}_c | M) とその誤差を算出
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
            sub_vc = bin_subset[bin_subset['target_vel'] > vc]
            k = len(sub_vc)

            if n_total_bin >= min_count and k > 0:
                p = k / n_total_bin
                p_err = np.sqrt(max(p * (1.0 - p) / n_total_bin, 1e-6 / n_total_bin))
                u = -np.log(p)
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
    out_dirs: List[Path],
    is_abs: bool = True,
    filename_prefix: str = "conditional_prob_potential_abs",
    title_suffix: str = "All Beads Pooled"
):
    """
    2パネル図:
    左パネル: P(\tilde{v}_\parallel > \tilde{v}_c | M)
    右パネル: U(M) = -ln P(\tilde{v}_\parallel > \tilde{v}_c | M)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    var_symbol = r"|M|" if is_abs else r"M"
    var_label = r"Magnetization Magnitude $|M|$" if is_abs else r"Magnetization $M$"

    for idx, vc in enumerate(vc_list):
        sub = df_summary[df_summary['vc'] == vc].dropna(subset=['prob'])
        if sub.empty:
            continue

        color = colors[idx]
        marker = markers[idx % len(markers)]
        x = sub['m_mean']
        p = sub['prob']
        p_err = sub['prob_err']

        # 左パネル: P(\tilde{v}_\parallel > \tilde{v}_c | M)
        ax1.errorbar(
            x, p, yerr=p_err,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=rf'$\tilde{{v}}_c = {vc:g}$'
        )

        # 右パネル: U(M) = -ln P
        sub_u = sub.dropna(subset=['u_m'])
        if not sub_u.empty:
            ax2.errorbar(
                sub_u['m_mean'], sub_u['u_m'], yerr=sub_u['u_m_err'],
                fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
                label=rf'$\tilde{{v}}_c = {vc:g}$'
            )

    ax1.set_xlabel(var_label, fontsize=12.5)
    ax1.set_ylabel(rf"Exceedance Probability $P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$", fontsize=12.5)
    ax1.set_title(rf"Conditional Probability $P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$", fontsize=13.5, fontweight='bold')
    if is_abs:
        ax1.set_xlim(-0.02, 1.02)
    else:
        ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, which='major', ls=':', alpha=0.6)
    ax1.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left' if is_abs else 'upper center', framealpha=0.9)

    ax2.set_xlabel(var_label, fontsize=12.5)
    ax2.set_ylabel(rf"Effective Potential $U({var_symbol}) = -\ln P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$", fontsize=12.5)
    ax2.set_title(rf"Effective Potential $U({var_symbol}) = -\ln P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$", fontsize=13.5, fontweight='bold')
    if is_abs:
        ax2.set_xlim(-0.02, 1.02)
    else:
        ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(bottom=-0.1)
    ax2.grid(True, which='major', ls=':', alpha=0.6)
    ax2.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper right' if is_abs else 'upper center', framealpha=0.9)

    plt.suptitle(
        rf"Cargo Velocity Exceedance Probability & Effective Potential ({title_suffix})",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure_to_all(fig, f"{filename_prefix}_2panel", out_dirs)


def plot_single_panel(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    out_dirs: List[Path],
    plot_type: str = "probability",  # "probability" or "potential"
    is_abs: bool = True,
    filename: Optional[str] = None,
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
            ylabel = rf"Exceedance Probability $P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$"
            title = rf"Exceedance Probability $P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$ vs Magnetization"
            ylim = (-0.02, 1.02)
            fname = filename if filename else f"conditional_probability_vs_magnetization_{tag_str}"
            loc = 'upper left' if is_abs else 'upper center'
        else:
            sub_u = sub.dropna(subset=['u_m'])
            if sub_u.empty:
                continue
            x = sub_u['m_mean']
            y = sub_u['u_m']
            yerr = sub_u['u_m_err']
            ylabel = rf"Effective Potential $U({var_symbol}) = -\ln P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$"
            title = rf"Effective Potential $U({var_symbol}) = -\ln P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid {var_symbol})$"
            ylim = (-0.1, max(y.max() * 1.15, 4.5))
            fname = filename if filename else f"conditional_potential_vs_magnetization_{tag_str}"
            loc = 'upper right' if is_abs else 'upper center'

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f'-{marker}', color=color, lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
            label=rf'$\tilde{{v}}_c = {vc:g}$'
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
    ax.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc=loc, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, fname, out_dirs)


def plot_per_condition_comparison(
    df_all: pd.DataFrame,
    vc_list: List[float],
    out_dirs: List[Path],
    n_bins: int = 8
):
    """全6粒子径別 (0.63, 1.18, 3.37, 5.00, 7.24, 20.0 um) の U(|M|) 2x3 パネルプロット（All Beads Pooled は除外）"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

    conditions = [
        {"key": "beads06um", "title": r"0.63 $\mu\mathrm{m}$ Beads"},
        {"key": "beads1um",  "title": r"1.18 $\mu\mathrm{m}$ Beads"},
        {"key": "beads3um",  "title": r"3.37 $\mu\mathrm{m}$ Beads"},
        {"key": "beads5um",  "title": r"5.00 $\mu\mathrm{m}$ Beads"},
        {"key": "beads7um",  "title": r"7.24 $\mu\mathrm{m}$ Beads"},
        {"key": "beads20um", "title": r"20.0 $\mu\mathrm{m}$ Beads"},
    ]

    for idx_panel, cond in enumerate(conditions):
        ax = axes[idx_panel]
        key = cond["key"]
        title = cond["title"]

        sub_df = df_all[df_all['bead_name'] == key]

        if sub_df.empty:
            ax.text(0.5, 0.5, f"No Data:\n{title}", ha='center', va='center', transform=ax.transAxes, fontsize=12)
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
                label=rf'$\tilde{{v}}_c = {vc:g}$' if idx_panel == 0 else ""
            )

        ax.set_title(f"{title} ($N={len(sub_df):,}$)", fontsize=12.5, fontweight='bold')
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=-0.1, top=4.5)

        if idx_panel in [0, 3]:
            ax.set_ylabel(r"$U(|M|) = -\ln P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)$", fontsize=11.5)
        if idx_panel >= 3:
            ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=11.5)

    axes[0].legend(title=r"Threshold $\tilde{v}_c$", fontsize=8.5, title_fontsize=9.5, loc='upper right', framealpha=0.9)

    plt.suptitle(
        r"Effective Potential $U(|M|) = -\ln P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)$ across All Bead Diameters ($0.63 - 20\ \mu\mathrm{m}$)",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_figure_to_all(fig, "conditional_prob_potential_per_condition", out_dirs)


def main():
    parser = argparse.ArgumentParser(description="Plot conditional probability P(v_tilde > vc | M) and effective potential U(M)")
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity" / "cargo_spin_velocity_points.csv",
        help="Path to cargo_spin_velocity_points.csv"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Root directory (NAS-Ebanaru). If specified or detected, figures and CSVs will also be saved to <root_dir>/figure/cargo_spin_velocity"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity",
        help="Local output directory"
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
        help="List of velocity thresholds v_tilde_c"
    )
    args = parser.parse_args()

    # 出力先ディレクトリ群の設定 (ローカル + root_dir/NAS)
    out_dirs = [args.output_dir]
    root_dir = args.root_dir if args.root_dir else find_default_root()
    if root_dir and root_dir.exists():
        nas_out_dir = root_dir / "figure" / "cargo_spin_velocity"
        if nas_out_dir not in out_dirs:
            out_dirs.append(nas_out_dir)

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("Output directories:")
    for d in out_dirs:
        print(f"  - {d}")

    apply_custom_style()

    # 1. データの読み込み
    df = load_points_data(args.points_csv)

    # 2. 全体プール (All Beads Pooled) の計算
    df_summary_abs_overall = compute_prob_potential(
        df, vc_list=args.vc, col_name='abs_m', n_bins=args.bins_abs, min_count=5
    )
    df_summary_abs_overall['condition'] = 'overall_pooled'

    df_summary_signed_overall = compute_prob_potential(
        df, vc_list=args.vc, col_name='m_ising', n_bins=args.bins_signed, min_count=5
    )
    df_summary_signed_overall['condition'] = 'overall_pooled'

    # 3. 小型粒子プール (Small Beads Pooled: 0.63, 1.18, 3.37 um) の計算
    small_df = df[df['bead_name'].isin(['beads06um', 'beads1um', 'beads3um'])].copy()
    df_summary_small_abs = compute_prob_potential(
        small_df, vc_list=args.vc, col_name='abs_m', n_bins=args.bins_abs, min_count=5
    )
    df_summary_small_abs['condition'] = 'small_beads_pooled'

    df_summary_small_signed = compute_prob_potential(
        small_df, vc_list=args.vc, col_name='m_ising', n_bins=args.bins_signed, min_count=5
    )
    df_summary_small_signed['condition'] = 'small_beads_pooled'

    # 4. 大型粒子プール (Large Beads Pooled: 5.00, 7.24, 20.0 um) の計算
    large_df = df[df['bead_name'].isin(['beads5um', 'beads7um', 'beads20um'])].copy()
    df_summary_large_abs = compute_prob_potential(
        large_df, vc_list=args.vc, col_name='abs_m', n_bins=args.bins_abs, min_count=5
    )
    df_summary_large_abs['condition'] = 'large_beads_pooled'

    df_summary_large_signed = compute_prob_potential(
        large_df, vc_list=args.vc, col_name='m_ising', n_bins=args.bins_signed, min_count=5
    )
    df_summary_large_signed['condition'] = 'large_beads_pooled'

    # 5. 各粒子径別計算
    all_summaries = [
        df_summary_abs_overall, df_summary_signed_overall,
        df_summary_small_abs, df_summary_small_signed,
        df_summary_large_abs, df_summary_large_signed
    ]
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
    save_csv_to_all(df_summary_all, "conditional_probability_potential_summary", out_dirs)

    # 6. プロット生成
    # (a) 全6粒子径別 U(|M|) 2x3 パネル図 (0.63, 1.18, 3.37, 5, 7, 20 um) ※All Pooledは非表示
    plot_per_condition_comparison(df, args.vc, out_dirs, n_bins=args.bins_abs)

    # (b) Small Beads Pooled (0.6, 1, 3 um) 2パネル図
    plot_prob_potential_2panel(
        df_summary_small_abs, args.vc, out_dirs, is_abs=True,
        filename_prefix="conditional_prob_potential_small_beads_pooled",
        title_suffix=rf"Small Beads Pooled ($0.63, 1.18, 3.37\ \mu\mathrm{{m}}$, $N={len(small_df):,}$)"
    )

    # (c) Small Beads Pooled U(|M|) 単体図
    plot_single_panel(
        df_summary_small_abs, args.vc, out_dirs, plot_type="potential", is_abs=True,
        filename="conditional_potential_small_beads_pooled",
        title_suffix=rf"Small Beads Pooled ($0.63, 1.18, 3.37\ \mu\mathrm{{m}}$, $N={len(small_df):,}$)"
    )

    # (d) Large Beads Pooled (5, 7, 20 um) 2パネル図
    plot_prob_potential_2panel(
        df_summary_large_abs, args.vc, out_dirs, is_abs=True,
        filename_prefix="conditional_prob_potential_large_beads_pooled",
        title_suffix=rf"Large Beads Pooled ($5.00, 7.24, 20.0\ \mu\mathrm{{m}}$, $N={len(large_df):,}$)"
    )

    # (e) Large Beads Pooled U(|M|) 単体図
    plot_single_panel(
        df_summary_large_abs, args.vc, out_dirs, plot_type="potential", is_abs=True,
        filename="conditional_potential_large_beads_pooled",
        title_suffix=rf"Large Beads Pooled ($5.00, 7.24, 20.0\ \mu\mathrm{{m}}$, $N={len(large_df):,}$)"
    )

    # (f) Small vs Large Beads Pooled U(|M|) 2パネル比較図
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    colors = get_vc_colors(args.vc)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']
    for idx, vc in enumerate(args.vc):
        sub_s = df_summary_small_abs[df_summary_small_abs['vc'] == vc].dropna(subset=['u_m'])
        if not sub_s.empty:
            ax_s.errorbar(
                sub_s['m_mean'], sub_s['u_m'], yerr=sub_s['u_m_err'],
                fmt=f'-{markers[idx % len(markers)]}', color=colors[idx], lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
                label=rf'$\tilde{{v}}_c = {vc:g}$'
            )
        sub_l = df_summary_large_abs[df_summary_large_abs['vc'] == vc].dropna(subset=['u_m'])
        if not sub_l.empty:
            ax_l.errorbar(
                sub_l['m_mean'], sub_l['u_m'], yerr=sub_l['u_m_err'],
                fmt=f'-{markers[idx % len(markers)]}', color=colors[idx], lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
                label=rf'$\tilde{{v}}_c = {vc:g}$'
            )

    for ax, title, n_pts in [(ax_s, "Small Beads Pooled (0.63, 1.18, 3.37 μm)", len(small_df)),
                             (ax_l, "Large Beads Pooled (5.00, 7.24, 20.0 μm)", len(large_df))]:
        ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12.5)
        ax.set_title(f"{title}\n($N={n_pts:,}$)", fontsize=12.5, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=-0.1, top=4.5)
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper right', framealpha=0.9)

    ax_s.set_ylabel(r"Effective Potential $U(|M|) = -\ln P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)$", fontsize=12.5)
    plt.suptitle(
        r"Effective Potential Comparison: Small vs Large Cargo Particles vs Magnetization Magnitude $|M|$",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "conditional_potential_small_vs_large_beads_pooled_2panel", out_dirs)

    # (g) All Beads Pooled 絶対値 |M| 2パネル (P(v_tilde > vc | |M|) & U(|M|))
    plot_prob_potential_2panel(
        df_summary_abs_overall, args.vc, out_dirs, is_abs=True,
        filename_prefix="conditional_prob_potential_abs",
        title_suffix=rf"All Beads Pooled ($0.63 - 20\ \mu\mathrm{{m}}$, $N={len(df):,}$)"
    )

    # (h) All Beads Pooled 単体図
    plot_single_panel(df_summary_abs_overall, args.vc, out_dirs, plot_type="probability", is_abs=True)
    plot_single_panel(df_summary_abs_overall, args.vc, out_dirs, plot_type="potential", is_abs=True)

    # (i) 符号付き M 2パネル (P(v_tilde > vc | M) & U(M))
    plot_prob_potential_2panel(
        df_summary_signed_overall, args.vc, out_dirs, is_abs=False,
        filename_prefix="conditional_prob_potential_signed",
        title_suffix=rf"All Beads Pooled ($0.63 - 20\ \mu\mathrm{{m}}$, $N={len(df):,}$)"
    )

    # (j) 符号付き M 単体図 (U(M))
    plot_single_panel(df_summary_signed_overall, args.vc, out_dirs, plot_type="potential", is_abs=False)

    print("\nAll conditional probability and potential plots created successfully in all directories!")


if __name__ == "__main__":
    main()
