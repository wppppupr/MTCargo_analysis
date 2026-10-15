#!/usr/bin/env python3
"""
plot_odds_ratio_and_posterior_magnetization.py

横軸を貨物領域における平均磁荷の絶対値 |M| とした以下のグラフを作成するスクリプトです:
1. Odds Ratio (濃縮比 / 相対比率):
       \\text{Odds Ratio} = \\frac{ P(\\tilde{v}_\\parallel > \\tilde{v}_c \\mid |M|) }{ P(\\tilde{v}_\\parallel > \\tilde{v}_c) } = \\frac{ P(|M| \\mid \\tilde{v}_\\parallel > \\tilde{v}_c) }{ P(|M|) }
   - 1.0 より大きい領域: 全体平均よりも高速走行が促進・濃縮されている領域
   - 1.0 より小さい領域: 高速走行が抑制されている領域

2. 条件付き磁荷分布 (事後分布):
       P(|M| \\mid \\tilde{v}_\\parallel > \\tilde{v}_c) = \\frac{ N(\\tilde{v}_\\parallel > \\tilde{v}_c,\\, |M|) }{ N(\\tilde{v}_\\parallel > \\tilde{v}_c) }
   - 速度が \\tilde{v}_c を超えている粒子に限定したときの、磁荷 |M| の存在確率分布

無次元速度閾値:
    \\tilde{v}_c = [0.1, 0.5, 0.8, 1.0]

出力ファイル（ローカルおよび root_dir / NAS-Ebanaru の両方に保存）:
1. figure/cargo_spin_velocity/odds_ratio_per_condition.png / .svg
   (全6粒子径 2x3 パネル図: 0.63, 1.18, 3.37, 5.00, 7.24, 20.0 um)
2. figure/cargo_spin_velocity/odds_ratio_and_posterior_small_beads_pooled_2panel.png / .svg
   (Small Beads Pooled (0.6, 1, 3 um): 左パネル Odds Ratio, 右パネル P(|M| | v_tilde > vc))
3. figure/cargo_spin_velocity/odds_ratio_small_beads_pooled.png / .svg
   (Small Beads Pooled: Odds Ratio 単体図)
4. figure/cargo_spin_velocity/posterior_magnetization_small_beads_pooled.png / .svg
   (Small Beads Pooled: P(|M| | v_tilde > vc) 単体図)
5. figure/cargo_spin_velocity/odds_ratio_and_posterior_large_beads_pooled_2panel.png / .svg
   (Large Beads Pooled (5, 7, 20 um): 左パネル Odds Ratio, 右パネル P(|M| | v_tilde > vc))
6. figure/cargo_spin_velocity/odds_ratio_large_beads_pooled.png / .svg
   (Large Beads Pooled: Odds Ratio 単体図)
7. figure/cargo_spin_velocity/posterior_magnetization_large_beads_pooled.png / .svg
   (Large Beads Pooled: P(|M| | v_tilde > vc) 単体図)
8. figure/cargo_spin_velocity/odds_ratio_small_vs_large_beads_pooled_2panel.png / .svg
   (Small vs Large Beads Pooled: Odds Ratio 2パネル比較図)
9. figure/cargo_spin_velocity/odds_ratio_and_posterior_magnetization_2panel.png / .svg
   (All Beads Pooled: 2パネル図)
10. figure/cargo_spin_velocity/odds_ratio_overlay_across_diameters.png / .svg
    (粒子径間 Odds Ratio 重ね合わせ図)
11. figure/cargo_spin_velocity/odds_ratio_and_posterior_summary.csv
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


def compute_odds_ratio_and_posterior(
    df: pd.DataFrame,
    vc_list: List[float],
    n_bins: int = 8,
    min_count: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    |M| のビンごとに、
    1. Odds Ratio = P(v_tilde > v_c | |M|) / P(v_tilde > v_c)
    2. P(|M| | v_tilde > v_c)
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
        n_vc = (df_temp['target_vel'] > vc).sum()
        p_vc_overall = n_vc / n_total if n_total > 0 else 0.0

        for bin_i in range(n_bins):
            m_low, m_high = bins[bin_i], bins[bin_i + 1]
            bin_subset = df_temp[df_temp['bin_idx'] == bin_i]
            n_bin = len(bin_subset)
            m_mean = float(bin_subset['abs_m'].mean()) if n_bin > 0 else (m_low + m_high) / 2.0
            m_median = float(bin_subset['abs_m'].median()) if n_bin > 0 else (m_low + m_high) / 2.0

            sub_vc = bin_subset[bin_subset['target_vel'] > vc]
            k_vc = len(sub_vc)

            if n_bin >= min_count and n_vc > 0:
                p_v_given_m = k_vc / n_bin
                p_v_given_m_err = np.sqrt(max(p_v_given_m * (1.0 - p_v_given_m) / n_bin, 1e-6 / n_bin))

                odds_ratio = p_v_given_m / p_vc_overall if p_vc_overall > 0 else np.nan
                if odds_ratio > 0:
                    rel_err_sq = (p_v_given_m_err / p_v_given_m)**2 + ((1.0 - p_vc_overall) / (n_total * p_vc_overall))
                    odds_ratio_err = odds_ratio * np.sqrt(rel_err_sq)
                else:
                    odds_ratio_err = np.nan

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
    out_dirs: List[Path],
    filename_prefix: str = "odds_ratio_and_posterior_magnetization",
    title_suffix: str = "All Beads Pooled"
):
    """
    2パネル図:
    左パネル: Odds Ratio = P(\tilde{v}_\parallel > \tilde{v}_c | |M|) / P(\tilde{v}_\parallel > \tilde{v}_c)
    右パネル: P(|M| | \tilde{v}_\parallel > \tilde{v}_c)
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
            label=rf'$\tilde{{v}}_c = {vc:g}$'
        )

    ax1.axhline(1.0, color='#666666', ls='--', lw=1.5, alpha=0.8, label='Baseline (= 1.0)')
    ax1.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12.5)
    ax1.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$", fontsize=13.5)
    ax1.set_title(r"Odds Ratio $\frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$", fontsize=13.5, fontweight='bold')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(0.0, 2.0)
    ax1.grid(True, which='major', ls=':', alpha=0.6)
    ax1.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    # --- 右パネル: P(|M| | \tilde{v}_\parallel > \tilde{v}_c) ---
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
            label=rf'$\tilde{{v}}_c = {vc:g}$'
        )

    ax2.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12.5)
    ax2.set_ylabel(r"Conditional Probability $P(|M| \mid \tilde{v}_\parallel > \tilde{v}_c)$", fontsize=12.5)
    ax2.set_title(r"Conditional Magnetization Distribution $P(|M| \mid \tilde{v}_\parallel > \tilde{v}_c)$", fontsize=13.5, fontweight='bold')
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 0.9)
    ax2.grid(True, which='major', ls=':', alpha=0.6)
    ax2.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    plt.suptitle(
        rf"Cargo Motion Enrichment & Magnetization Distribution vs $|M|$ ({title_suffix})",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure_to_all(fig, f"{filename_prefix}_2panel", out_dirs)


def plot_single_odds_ratio(
    df_summary: pd.DataFrame,
    vc_list: List[float],
    out_dirs: List[Path],
    filename: str = "odds_ratio_vs_magnetization_abs",
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
            label=rf'$\tilde{{v}}_c = {vc:g}$'
        )

    ax.axhline(1.0, color='#666666', ls='--', lw=1.5, alpha=0.8, label='Baseline (= 1.0)')
    ax.set_xlabel(r"Magnetization Magnitude in Cargo Region $|M|$", fontsize=12.5)
    ax.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$", fontsize=13.5)
    ax.set_title(rf"Odds Ratio $\frac{{P(\tilde{{v}}_\parallel > \tilde{{v}}_c \mid |M|)}}{{P(\tilde{{v}}_\parallel > \tilde{{v}}_c)}}$ vs Magnetization Magnitude\n({title_suffix})", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 2.0)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, filename, out_dirs)


def plot_single_posterior(
    df_summary: pd.DataFrame,
    df_bg: pd.DataFrame,
    vc_list: List[float],
    out_dirs: List[Path],
    filename: str = "posterior_magnetization_vs_vc",
    title_suffix: str = "All Beads Pooled"
):
    """P(|M| | \tilde{v}_\parallel > \tilde{v}_c) 単体図"""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = get_vc_colors(vc_list)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']

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
            label=rf'$\tilde{{v}}_c = {vc:g}$'
        )

    ax.set_xlabel(r"Magnetization Magnitude in Cargo Region $|M|$", fontsize=12.5)
    ax.set_ylabel(r"Conditional Probability $P(|M| \mid \tilde{v}_\parallel > \tilde{v}_c)$", fontsize=12.5)
    ax.set_title(rf"Conditional Distribution $P(|M| \mid \tilde{{v}}_\parallel > \tilde{{v}}_c)$ vs Magnetization Magnitude\n({title_suffix})", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.9)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, filename, out_dirs)


def plot_odds_ratio_per_condition(
    df_all: pd.DataFrame,
    vc_list: List[float],
    out_dirs: List[Path],
    n_bins: int = 8
):
    """全6粒子径別 (0.63, 1.18, 3.37, 5.00, 7.24, 20.0 um) の Odds Ratio 2x3 パネルプロット（All Beads Pooled は除外）"""
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
                label=rf'$\tilde{{v}}_c = {vc:g}$' if idx_panel == 0 else ""
            )

        ax.axhline(1.0, color='#666666', ls='--', lw=1.4, alpha=0.8)
        ax.set_title(f"{title} ($N={len(sub_df):,}$)", fontsize=13, fontweight='bold')
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 3.0)

        if idx_panel in [0, 3]:
            ax.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$", fontsize=11.5)
        if idx_panel >= 3:
            ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=11.5)

    axes[0].legend(title=r"Threshold $\tilde{v}_c$", fontsize=8.5, title_fontsize=9.5, loc='upper left', framealpha=0.9)

    plt.suptitle(
        r"Odds Ratio $\frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$ across All Bead Diameters ($0.63 - 20\ \mu\mathrm{m}$)",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_figure_to_all(fig, "odds_ratio_per_condition", out_dirs)


def plot_odds_ratio_overlay_across_diameters(
    df_all: pd.DataFrame,
    out_dirs: List[Path],
    target_vc_list: List[float] = [0.1, 0.5, 0.8, 1.0],
    n_bins: int = 8
):
    """代表的な速度閾値 (\tilde{v}_c = 0.1, 0.5, 0.8, 1.0) における粒子径間の Odds Ratio 重ね合わせ比較図"""
    fig, axes = plt.subplots(1, len(target_vc_list), figsize=(5.5 * len(target_vc_list), 5.2), sharey=True)
    if len(target_vc_list) == 1:
        axes = [axes]

    for idx_vc, vc in enumerate(target_vc_list):
        ax = axes[idx_vc]

        for b in BEADS_INFO:
            b_name = b["name"]
            sub_df = df_all[df_all['bead_name'] == b_name]
            if sub_df.empty:
                continue

            df_summary, _ = compute_odds_ratio_and_posterior(sub_df, [vc], n_bins=n_bins, min_count=3)
            sub = df_summary.dropna(subset=['odds_ratio'])
            if sub.empty:
                continue

            ax.errorbar(
                sub['m_mean'], sub['odds_ratio'], yerr=sub['odds_ratio_err'],
                fmt=f'-{b["marker"]}', color=b['color'], lw=1.8, ms=6.5, capsize=3.0, elinewidth=1.2,
                label=f'$d = {b["label"]}$'
            )

        # Small Beads Pooled 基準線
        small_df = df_all[df_all['bead_name'].isin(['beads06um', 'beads1um', 'beads3um'])]
        df_small, _ = compute_odds_ratio_and_posterior(small_df, [vc], n_bins=n_bins, min_count=5)
        sub_sm = df_small.dropna(subset=['odds_ratio'])
        if not sub_sm.empty:
            ax.plot(
                sub_sm['m_mean'], sub_sm['odds_ratio'],
                'k--', lw=2.2, label='Small Pooled (0.6-3um)'
            )

        ax.axhline(1.0, color='#888888', ls=':', lw=1.5, alpha=0.8)
        ax.set_title(rf"Threshold $\tilde{{v}}_c = {vc:g}$", fontsize=13, fontweight='bold')
        ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12)
        if idx_vc == 0:
            ax.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$", fontsize=12.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 3.0)
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=8.0, framealpha=0.9)

    plt.suptitle(
        r"Odds Ratio Comparison Across Bead Diameters ($0.63 - 20\ \mu\mathrm{m}$)",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure_to_all(fig, "odds_ratio_overlay_across_diameters", out_dirs)


def main():
    parser = argparse.ArgumentParser(description="Plot odds ratio and conditional posterior magnetization distribution")
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
    df_summary_overall, df_bg_overall = compute_odds_ratio_and_posterior(
        df, vc_list=args.vc, n_bins=args.bins, min_count=5
    )
    df_summary_overall['condition'] = 'overall_pooled'

    # 3. 小型粒子プール (Small Beads Pooled: 0.63, 1.18, 3.37 um) の計算
    small_df = df[df['bead_name'].isin(['beads06um', 'beads1um', 'beads3um'])].copy()
    df_summary_small, df_bg_small = compute_odds_ratio_and_posterior(
        small_df, vc_list=args.vc, n_bins=args.bins, min_count=5
    )
    df_summary_small['condition'] = 'small_beads_pooled'

    # 4. 大型粒子プール (Large Beads Pooled: 5.00, 7.24, 20.0 um) の計算
    large_df = df[df['bead_name'].isin(['beads5um', 'beads7um', 'beads20um'])].copy()
    df_summary_large, df_bg_large = compute_odds_ratio_and_posterior(
        large_df, vc_list=args.vc, n_bins=args.bins, min_count=5
    )
    df_summary_large['condition'] = 'large_beads_pooled'

    # 5. 各粒子径別計算
    all_summaries = [df_summary_overall, df_summary_small, df_summary_large]
    for b in BEADS_INFO:
        b_name = b["name"]
        sub_df = df[df['bead_name'] == b_name]
        if not sub_df.empty:
            df_b, _ = compute_odds_ratio_and_posterior(sub_df, vc_list=args.vc, n_bins=args.bins, min_count=3)
            df_b['condition'] = b_name
            all_summaries.append(df_b)

    df_summary_all = pd.concat(all_summaries, ignore_index=True)
    save_csv_to_all(df_summary_all, "odds_ratio_and_posterior_summary", out_dirs)

    # 6. プロット生成
    # (a) 全6粒子径別 Odds Ratio 2x3 パネル図 (0.63, 1.18, 3.37, 5, 7, 20 um) ※All Pooledは非表示
    plot_odds_ratio_per_condition(df, args.vc, out_dirs, n_bins=args.bins)

    # (b) Small Beads Pooled (0.6, 1, 3 um) 2パネル図
    plot_odds_ratio_and_posterior_2panel(
        df_summary_small, df_bg_small, args.vc, out_dirs,
        filename_prefix="odds_ratio_and_posterior_small_beads_pooled",
        title_suffix=rf"Small Beads Pooled ($0.63, 1.18, 3.37\ \mu\mathrm{{m}}$, $N={len(small_df):,}$)"
    )

    # (c) Small Beads Pooled Odds Ratio 単体図
    plot_single_odds_ratio(
        df_summary_small, args.vc, out_dirs,
        filename="odds_ratio_small_beads_pooled",
        title_suffix=rf"Small Beads Pooled ($0.63, 1.18, 3.37\ \mu\mathrm{{m}}$, $N={len(small_df):,}$)"
    )

    # (d) Small Beads Pooled P(|M| | v_tilde > vc) 単体図
    plot_single_posterior(
        df_summary_small, df_bg_small, args.vc, out_dirs,
        filename="posterior_magnetization_small_beads_pooled",
        title_suffix=rf"Small Beads Pooled ($0.63, 1.18, 3.37\ \mu\mathrm{{m}}$, $N={len(small_df):,}$)"
    )

    # (e) Large Beads Pooled (5, 7, 20 um) 2パネル図
    plot_odds_ratio_and_posterior_2panel(
        df_summary_large, df_bg_large, args.vc, out_dirs,
        filename_prefix="odds_ratio_and_posterior_large_beads_pooled",
        title_suffix=rf"Large Beads Pooled ($5.00, 7.24, 20.0\ \mu\mathrm{{m}}$, $N={len(large_df):,}$)"
    )

    # (f) Large Beads Pooled Odds Ratio 単体図
    plot_single_odds_ratio(
        df_summary_large, args.vc, out_dirs,
        filename="odds_ratio_large_beads_pooled",
        title_suffix=rf"Large Beads Pooled ($5.00, 7.24, 20.0\ \mu\mathrm{{m}}$, $N={len(large_df):,}$)"
    )

    # (g) Large Beads Pooled P(|M| | v_tilde > vc) 単体図
    plot_single_posterior(
        df_summary_large, df_bg_large, args.vc, out_dirs,
        filename="posterior_magnetization_large_beads_pooled",
        title_suffix=rf"Large Beads Pooled ($5.00, 7.24, 20.0\ \mu\mathrm{{m}}$, $N={len(large_df):,}$)"
    )

    # (h) Small vs Large Beads Pooled Odds Ratio 2パネル比較図
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    colors = get_vc_colors(args.vc)
    markers = ['o', 's', '^', 'd', 'v', 'p', 'h']
    for idx, vc in enumerate(args.vc):
        sub_s = df_summary_small[df_summary_small['vc'] == vc].dropna(subset=['odds_ratio'])
        if not sub_s.empty:
            ax_s.errorbar(
                sub_s['m_mean'], sub_s['odds_ratio'], yerr=sub_s['odds_ratio_err'],
                fmt=f'-{markers[idx % len(markers)]}', color=colors[idx], lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
                label=rf'$\tilde{{v}}_c = {vc:g}$'
            )
        sub_l = df_summary_large[df_summary_large['vc'] == vc].dropna(subset=['odds_ratio'])
        if not sub_l.empty:
            ax_l.errorbar(
                sub_l['m_mean'], sub_l['odds_ratio'], yerr=sub_l['odds_ratio_err'],
                fmt=f'-{markers[idx % len(markers)]}', color=colors[idx], lw=1.8, ms=6.5, capsize=3.5, elinewidth=1.3,
                label=rf'$\tilde{{v}}_c = {vc:g}$'
            )

    for ax, title, n_pts in [(ax_s, "Small Beads Pooled (0.63, 1.18, 3.37 μm)", len(small_df)),
                             (ax_l, "Large Beads Pooled (5.00, 7.24, 20.0 μm)", len(large_df))]:
        ax.axhline(1.0, color='#666666', ls='--', lw=1.5, alpha=0.8, label='Baseline (= 1.0)')
        ax.set_xlabel(r"Magnetization Magnitude $|M|$", fontsize=12.5)
        ax.set_title(f"{title}\n($N={n_pts:,}$)", fontsize=12.5, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 2.0)
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(title=r"Threshold $\tilde{v}_c$", fontsize=9.5, title_fontsize=10.5, loc='upper left', framealpha=0.9)

    ax_s.set_ylabel(r"$\mathrm{Odds\ Ratio} = \frac{P(\tilde{v}_\parallel > \tilde{v}_c \mid |M|)}{P(\tilde{v}_\parallel > \tilde{v}_c)}$", fontsize=13.5)
    plt.suptitle(
        r"Odds Ratio Comparison: Small vs Large Cargo Particles vs Magnetization Magnitude $|M|$",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "odds_ratio_small_vs_large_beads_pooled_2panel", out_dirs)

    # (i) All Beads Pooled 2パネル図
    plot_odds_ratio_and_posterior_2panel(
        df_summary_overall, df_bg_overall, args.vc, out_dirs,
        filename_prefix="odds_ratio_and_posterior_magnetization",
        title_suffix=rf"All Beads Pooled ($0.63 - 20\ \mu\mathrm{{m}}$, $N={len(df):,}$)"
    )

    # (j) 粒子径間 Odds Ratio 重ね合わせ図 (vc = 0.1, 0.5, 0.8, 1.0)
    plot_odds_ratio_overlay_across_diameters(df, out_dirs, target_vc_list=args.vc, n_bins=args.bins)

    print("\nOdds ratio and posterior magnetization plots created successfully in all directories!")


if __name__ == "__main__":
    main()
