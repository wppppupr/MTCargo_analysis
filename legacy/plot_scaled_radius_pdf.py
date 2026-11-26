#!/usr/bin/env python3
"""
plot_scaled_radius_pdf.py

貨物粒子（ビーズ）のスケール半径
    x = R_c / xi_{i,t}
（および相関長 xi_{i,t} [um]）の確率密度関数（PDF: Probability Density Function）を
貨物粒子の直径（0.63, 1.18, 3.37, 5.0, 7.24, 20.0 um）ごとに作成・比較するスクリプトです。

【主な出力図】
1. scaled_radius_pdf_grid_linear.png / .svg    : 各サイズごとの x の PDF Grid (Linear 横軸, 2x3 パネル)
2. scaled_radius_pdf_grid_logx.png / .svg      : 各サイズごとの ln(x) または log10(x) の PDF Grid (2x3 パネル)
3. scaled_radius_pdf_overlay_linear.png / .svg: 全サイズの x の PDF 重ね合わせ (Linear 横軸)
4. scaled_radius_pdf_overlay_logx.png / .svg  : 全サイズの x の PDF 重ね合わせ (Log 横軸)
5. scaled_radius_violin_and_stats.png / .svg  : 粒子径 d に対する x のバイオリン図 / Boxplot / Mean ± STD / Median ± IQR
6. correlation_length_xi_pdf_grid.png / .svg  : 元の相関長 xi [um] の PDF Grid (2x3 パネル)
7. correlation_length_xi_pdf_overlay.png / .svg: 相関長 xi [um] の PDF 重ね合わせ

【主な出力 CSV】
1. scaled_radius_pdf_summary.csv              : 各粒子径の統計量（Mean, STD, SEM, Median, IQR, Lognormal fit 等）
2. scaled_radius_all_points.csv               : 全点 (i, t) の瞬時 x = Rc / xi データ
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# プロジェクト設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイル適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        style_colors = [f"#{c}" if not c.startswith('#') else c for c in style_colors]
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']

# ビーズ基本情報 (全6サイズ)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "label": "0.63 μm", "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "label": "1.18 μm", "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "label": "3.37 μm", "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "label": "5.00 μm", "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "label": "7.24 μm", "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "label": "20.0 μm", "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def get_default_root_dir() -> Path:
    for p in POSSIBLE_ROOTS:
        if p.exists():
            return p
    return CURRENT_DIR


def save_figure_to_all(fig: plt.Figure, name_base: str, out_dirs: List[Path], dpi: int = 300):
    """PNG と SVG 形式で複数ディレクトリに図を保存"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / f"{name_base}.png"
        svg_path = d / f"{name_base}.svg"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {name_base}.png / .svg -> {len(out_dirs)} dir(s)")


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]):
    """複数ディレクトリに CSV を保存 (ロック対策リトライ付き)"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / f"{basename}.csv"
        for attempt in range(3):
            try:
                df.to_csv(csv_path, index=False)
                break
            except OSError:
                time.sleep(1.0)
    print(f"Saved CSV: {basename}.csv -> {len(out_dirs)} dir(s)")


# =============================================================================
# データ読み出し & 統計量計算
# =============================================================================

def load_all_scaled_radius_data(root_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    各ビーズサイズの xi_vs_velocity_{b_name}.csv から
    瞬時相関長 xi_{i,t} とスケール半径 x = Rc / xi_{i,t} を読み込み、集計する。
    """
    all_dfs = []
    stats_dict = {}

    for b in BEADS_INFO:
        b_name = b["name"]
        d_um = b["diameter_um"]
        r_um = b["radius_um"]
        label = b["label"]

        # CSV パス
        csv_path = CURRENT_DIR / "figure" / "xi_vs_velocity" / f"xi_vs_velocity_{b_name}.csv"
        if not csv_path.exists() and root_dir is not None:
            csv_path = root_dir / "figure" / "xi_vs_velocity" / f"xi_vs_velocity_{b_name}.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist. Skipping {b_name}.")
            continue

        df_raw = pd.read_csv(csv_path)
        valid = df_raw.dropna(subset=['xi_um']).copy()
        valid = valid[valid['xi_um'] > 0].copy()

        if len(valid) == 0:
            continue

        valid['bead_name'] = b_name
        valid['diameter_um'] = d_um
        valid['radius_um'] = r_um
        valid['label'] = label
        valid['rc_over_xi'] = r_um / valid['xi_um']
        valid['log10_rc_over_xi'] = np.log10(valid['rc_over_xi'])
        valid['ln_rc_over_xi'] = np.log(valid['rc_over_xi'])
        valid['log10_xi_um'] = np.log10(valid['xi_um'])

        all_dfs.append(valid)

        x_vals = valid['rc_over_xi'].values
        log10_x = valid['log10_rc_over_xi'].values
        xi_vals = valid['xi_um'].values
        n_pts = len(x_vals)

        # 統計量
        mean_x = float(np.mean(x_vals))
        std_x = float(np.std(x_vals, ddof=1)) if n_pts > 1 else 0.0
        sem_x = float(std_x / np.sqrt(n_pts)) if n_pts > 1 else 0.0
        median_x = float(np.median(x_vals))
        q25_x = float(np.percentile(x_vals, 25))
        q75_x = float(np.percentile(x_vals, 75))
        iqr_x = q75_x - q25_x
        skew_x = float(stats.skew(x_vals))
        kurt_x = float(stats.kurtosis(x_vals))

        # 対数統計量
        mean_log10_x = float(np.mean(log10_x))
        std_log10_x = float(np.std(log10_x, ddof=1)) if n_pts > 1 else 0.0
        geom_mean_x = float(10 ** mean_log10_x)
        geom_std_factor = float(10 ** std_log10_x)

        # xi 統計量
        mean_xi = float(np.mean(xi_vals))
        std_xi = float(np.std(xi_vals, ddof=1)) if n_pts > 1 else 0.0
        sem_xi = float(std_xi / np.sqrt(n_pts)) if n_pts > 1 else 0.0
        median_xi = float(np.median(xi_vals))
        q25_xi = float(np.percentile(xi_vals, 25))
        q75_xi = float(np.percentile(xi_vals, 75))

        stats_dict[b_name] = {
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": r_um,
            "label": label,
            "color": b["color"],
            "marker": b["marker"],
            "n_points": n_pts,
            "x_mean": mean_x,
            "x_std": std_x,
            "x_sem": sem_x,
            "x_median": median_x,
            "x_q25": q25_x,
            "x_q75": q75_x,
            "x_iqr": iqr_x,
            "x_skewness": skew_x,
            "x_kurtosis": kurt_x,
            "x_geom_mean": geom_mean_x,
            "x_geom_std_factor": geom_std_factor,
            "log10_x_mean": mean_log10_x,
            "log10_x_std": std_log10_x,
            "xi_mean": mean_xi,
            "xi_std": std_xi,
            "xi_sem": sem_xi,
            "xi_median": median_xi,
            "xi_q25": q25_xi,
            "xi_q75": q75_xi,
            "values_x": x_vals,
            "values_log10_x": log10_x,
            "values_xi": xi_vals,
        }

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return df_all, stats_dict


# =============================================================================
# 作図 1 & 2: 個別パネル Grid (Linear & Logx)
# =============================================================================

def plot_scaled_radius_pdf_grid(stats_dict: Dict[str, dict], out_dirs: List[Path], log_scale: bool = False):
    """2x3 パネルで各粒子径の x = Rc / xi の確率密度関数 PDF を描画"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes_flat = axes.flatten()

    for idx, b in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        b_name = b["name"]
        st = stats_dict.get(b_name)
        if st is None:
            ax.set_visible(False)
            continue

        c = st["color"]
        d = st["diameter_um"]
        n_pts = st["n_points"]

        if log_scale:
            vals = st["values_log10_x"]
            bins = np.linspace(vals.min() - 0.2, vals.max() + 0.2, 30)
            x_grid = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)

            # ヒストグラム
            counts, bin_edges, _ = ax.hist(
                vals, bins=bins, density=True, color=c, alpha=0.45,
                edgecolor='black', linewidth=0.8, label=f"Data ($N={n_pts}$)"
            )

            # KDE
            kde = stats.gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=c, lw=2.5, label="KDE")

            # 正規フィット
            mu, sigma = st["log10_x_mean"], st["log10_x_std"]
            norm_fit = stats.norm.pdf(x_grid, mu, sigma)
            ax.plot(x_grid, norm_fit, color='black', lw=1.8, ls='--', label=r"Gaussian $\mathcal{N}(\mu, \sigma^2)$")

            # 代表値の線
            ax.axvline(mu, color='black', lw=1.5, ls='-', label=f"Mean = {mu:.2f}")
            med_log = np.log10(st["x_median"])
            ax.axvline(med_log, color='blue', lw=1.5, ls=':', label=f"Median = {med_log:.2f}")

            ax.set_xlabel(r"$\log_{10}(R_c / \xi_{i,t})$", fontsize=13)
            ax.set_ylabel("Probability Density", fontsize=13)
            ax.set_title(f"$d = {st['label']}$ ($R_c = {st['radius_um']:.3f}$ μm)", fontsize=14, fontweight='bold')
            ax.grid(True, ls=':', alpha=0.5)
            ax.legend(fontsize=9, loc='upper left', framealpha=0.85)

        else:
            vals = st["values_x"]
            # 99.0% タイルで上限を設定して見やすく
            v_max = np.percentile(vals, 99.0) * 1.2
            bins = np.linspace(0, max(v_max, 0.5), 35)
            x_grid = np.linspace(0.001, max(v_max, 0.5), 400)

            # ヒストグラム
            ax.hist(
                vals, bins=bins, density=True, color=c, alpha=0.45,
                edgecolor='black', linewidth=0.8, label=f"Data ($N={n_pts}$)"
            )

            # KDE
            kde = stats.gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=c, lw=2.5, label="KDE")

            # 対数正規フィット
            shape, loc, scale_ln = stats.lognorm.fit(vals, floc=0)
            lognorm_fit = stats.lognorm.pdf(x_grid, shape, loc, scale_ln)
            ax.plot(x_grid, lognorm_fit, color='black', lw=1.8, ls='--', label="Lognormal Fit")

            # 平均・中央値の線
            ax.axvline(st["x_mean"], color='black', lw=1.5, ls='-', label=f"Mean = {st['x_mean']:.3f} ± {st['x_std']:.3f}")
            ax.axvline(st["x_median"], color='blue', lw=1.5, ls=':', label=f"Median = {st['x_median']:.3f}")

            ax.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13)
            ax.set_ylabel("Probability Density $P(x)$", fontsize=13)
            ax.set_title(f"$d = {st['label']}$ ($R_c = {st['radius_um']:.3f}$ μm)", fontsize=14, fontweight='bold')
            ax.set_xlim(0, max(v_max, 0.5))
            ax.grid(True, ls=':', alpha=0.5)
            ax.legend(fontsize=9, loc='upper right', framealpha=0.85)

    name_base = "scaled_radius_pdf_grid_logx" if log_scale else "scaled_radius_pdf_grid_linear"
    fig_title = r"Probability Density Function of Scaled Radius $\log_{10}(R_c / \xi_{i,t})$" if log_scale else r"Probability Density Function of Scaled Radius $x = R_c / \xi_{i,t}$"
    plt.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure_to_all(fig, name_base, out_dirs)


# =============================================================================
# 作図 3 & 4: 全サイズ重ね合わせ Overlay (Linear & Logx)
# =============================================================================

def plot_scaled_radius_pdf_overlay(stats_dict: Dict[str, dict], out_dirs: List[Path], log_scale: bool = False):
    """全サイズの x = Rc / xi の PDF (KDE) を同一軸上に重ね合わせて比較"""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    if log_scale:
        x_grid = np.linspace(-3.0, 1.5, 600)
        for b in BEADS_INFO:
            st = stats_dict.get(b["name"])
            if st is None:
                continue
            vals = st["values_log10_x"]
            kde = stats.gaussian_kde(vals)
            density = kde(x_grid)
            ax.plot(
                x_grid, density, color=st["color"], lw=2.5,
                label=f"$d = {st['label']}$ (Med = $10^{{{np.log10(st['x_median']):.2f}}}$)"
            )
            ax.fill_between(x_grid, density, color=st["color"], alpha=0.15)

        ax.set_xlabel(r"$\log_{10}(R_c / \xi_{i,t})$", fontsize=14)
        ax.set_ylabel("Probability Density $P(\\log_{10} x)$", fontsize=14)
        ax.set_title(r"Comparison of Scaled Radius PDF: $\log_{10}(R_c / \xi_{i,t})$", fontsize=15, fontweight='bold', pad=12)
        ax.set_xlim(-2.5, 1.0)
        name_base = "scaled_radius_pdf_overlay_logx"

    else:
        x_grid = np.linspace(0.001, 3.5, 700)
        for b in BEADS_INFO:
            st = stats_dict.get(b["name"])
            if st is None:
                continue
            vals = st["values_x"]
            kde = stats.gaussian_kde(vals)
            density = kde(x_grid)
            ax.plot(
                x_grid, density, color=st["color"], lw=2.5,
                label=f"$d = {st['label']}$ (Mean = {st['x_mean']:.2f}, Med = {st['x_median']:.2f})"
            )
            ax.fill_between(x_grid, density, color=st["color"], alpha=0.15)

        ax.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=14)
        ax.set_ylabel("Probability Density $P(x)$", fontsize=14)
        ax.set_title(r"Comparison of Scaled Radius PDF: $x = R_c / \xi_{i,t}$", fontsize=15, fontweight='bold', pad=12)
        ax.set_xlim(0, 3.0)
        name_base = "scaled_radius_pdf_overlay_linear"

    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(fontsize=10.5, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig, name_base, out_dirs)


# =============================================================================
# 作図 5: 粒子径 d に対する x のバイオリン図 & 統計量比較
# =============================================================================

def plot_scaled_radius_violin_and_stats(df_all: pd.DataFrame, stats_dict: Dict[str, dict], out_dirs: List[Path]):
    """
    粒子径 d に対する x = Rc / xi の分布推移
    (Left: Violin Plot on Log scale, Right: Mean ± STD vs Median & IQR)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # --- 左パネル: バイオリンプロット (log10 x) ---
    bead_names = [b["name"] for b in BEADS_INFO if b["name"] in stats_dict]
    data_list = [stats_dict[b]["values_log10_x"] for b in bead_names]
    positions = np.arange(1, len(bead_names) + 1)
    labels = [stats_dict[b]["label"] for b in bead_names]
    colors = [stats_dict[b]["color"] for b in bead_names]

    parts = ax1.violinplot(data_list, positions=positions, showmeans=True, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linestyle('--')
    parts['cmeans'].set_linewidth(1.8)
    parts['cmedians'].set_color('blue')
    parts['cmedians'].set_linestyle('-')
    parts['cmedians'].set_linewidth(2.0)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_xlabel("Particle Diameter $d$ [μm]", fontsize=13)
    ax1.set_ylabel(r"$\log_{10}(R_c / \xi_{i,t})$", fontsize=13)
    ax1.set_title(r"Distribution of $\log_{10}(R_c / \xi_{i,t})$ (Violin Plot)" + "\n(Dashed: Mean, Solid Blue: Median)", fontsize=13.5, fontweight='bold', pad=10)
    ax1.grid(True, ls=':', alpha=0.5)

    # --- 右パネル: Mean ± STD vs Median ± IQR vs Diameter ---
    d_vals = np.array([stats_dict[b]["diameter_um"] for b in bead_names])
    mean_vals = np.array([stats_dict[b]["x_mean"] for b in bead_names])
    std_vals = np.array([stats_dict[b]["x_std"] for b in bead_names])
    med_vals = np.array([stats_dict[b]["x_median"] for b in bead_names])
    q25_vals = np.array([stats_dict[b]["x_q25"] for b in bead_names])
    q75_vals = np.array([stats_dict[b]["x_q75"] for b in bead_names])

    # Mean ± STD
    ax2.errorbar(
        d_vals, mean_vals, yerr=std_vals,
        fmt='s-', color='black', ecolor='gray', elinewidth=1.8, capsize=5, capthick=1.5,
        markersize=8, markerfacecolor='black', markeredgecolor='black',
        label=r"Mean $\pm$ STD"
    )

    # Median & IQR
    yerr_iqr = [med_vals - q25_vals, q75_vals - med_vals]
    ax2.errorbar(
        d_vals * 1.05, med_vals, yerr=yerr_iqr,
        fmt='o--', color='#0077BB', ecolor='#0077BB', elinewidth=1.8, capsize=5, capthick=1.5,
        markersize=8, markerfacecolor='white', markeredgewidth=2.0, markeredgecolor='#0077BB',
        label=r"Median $[Q_1, Q_3]$ (IQR)"
    )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("Particle Diameter $d$ [μm]", fontsize=13)
    ax2.set_ylabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13)
    ax2.set_title("Scaled Radius Statistics vs Particle Diameter $d$", fontsize=13.5, fontweight='bold', pad=10)
    ax2.grid(True, which='both', ls=':', alpha=0.6)
    ax2.legend(fontsize=11, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, "scaled_radius_violin_and_stats", out_dirs)


# =============================================================================
# 作図 6 & 7: 相関長 xi [um] 自体の PDF Grid & Overlay
# =============================================================================

def plot_correlation_length_xi_pdf(stats_dict: Dict[str, dict], out_dirs: List[Path]):
    """相関長 xi [um] 自体の確率密度関数 Grid & Overlay"""
    # 1. Grid
    fig_grid, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes_flat = axes.flatten()

    for idx, b in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        b_name = b["name"]
        st = stats_dict.get(b_name)
        if st is None:
            ax.set_visible(False)
            continue

        c = st["color"]
        vals = st["values_xi"]
        n_pts = st["n_points"]

        bins = np.linspace(0, 45, 35)
        ax.hist(vals, bins=bins, density=True, color=c, alpha=0.45, edgecolor='black', linewidth=0.8, label=f"Data ($N={n_pts}$)")

        x_grid = np.linspace(0.1, 45, 300)
        kde = stats.gaussian_kde(vals)
        ax.plot(x_grid, kde(x_grid), color=c, lw=2.5, label="KDE")

        ax.axvline(st["xi_mean"], color='black', lw=1.5, ls='-', label=f"Mean = {st['xi_mean']:.2f} ± {st['xi_std']:.2f} μm")
        ax.axvline(st["xi_median"], color='blue', lw=1.5, ls=':', label=f"Median = {st['xi_median']:.2f} μm")

        ax.set_xlabel(r"Correlation Length $\xi_{i,t}$ [μm]", fontsize=13)
        ax.set_ylabel("Probability Density $P(\\xi)$", fontsize=13)
        ax.set_title(f"$d = {st['label']}$", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 45)
        ax.grid(True, ls=':', alpha=0.5)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.85)

    plt.suptitle(r"Probability Density Function of Correlation Length $\xi_{i,t}$ [μm]", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure_to_all(fig_grid, "correlation_length_xi_pdf_grid", out_dirs)

    # 2. Overlay
    fig_ov, ax_ov = plt.subplots(figsize=(9, 6.5))
    x_grid = np.linspace(0.1, 45, 500)
    for b in BEADS_INFO:
        st = stats_dict.get(b["name"])
        if st is None:
            continue
        vals = st["values_xi"]
        kde = stats.gaussian_kde(vals)
        density = kde(x_grid)
        ax_ov.plot(
            x_grid, density, color=st["color"], lw=2.5,
            label=f"$d = {st['label']}$ (Mean = {st['xi_mean']:.1f} μm, Med = {st['xi_median']:.1f} μm)"
        )
        ax_ov.fill_between(x_grid, density, color=st["color"], alpha=0.12)

    ax_ov.set_xlabel(r"Correlation Length $\xi_{i,t}$ [μm]", fontsize=14)
    ax_ov.set_ylabel("Probability Density $P(\\xi)$", fontsize=14)
    ax_ov.set_title(r"Comparison of Correlation Length PDF: $\xi_{i,t}$ [μm]", fontsize=15, fontweight='bold', pad=12)
    ax_ov.set_xlim(0, 45)
    ax_ov.grid(True, ls=':', alpha=0.6)
    ax_ov.legend(fontsize=10.5, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_ov, "correlation_length_xi_pdf_overlay", out_dirs)


# =============================================================================
# メイン処理
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot probability density function (PDF) of scaled radius x = Rc / xi_{i,t} across bead sizes."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=get_default_root_dir(),
        help="Root directory of data (default: detected automatically)"
    )
    args = parser.parse_args()

    # 出力先ディレクトリの設定
    out_dirs = [
        CURRENT_DIR / "figure" / "scaled_radius_pdf",
    ]
    if args.root_dir.exists() and args.root_dir != CURRENT_DIR:
        out_dirs.append(args.root_dir / "figure" / "scaled_radius_pdf")

    print(f"Output directories: {[str(d) for d in out_dirs]}")

    # 1. データ読み出し & 統計量計算
    print("\nLoading scaled radius x = Rc / xi_{i,t} and correlation length data...")
    df_all, stats_dict = load_all_scaled_radius_data(args.root_dir)

    if not stats_dict:
        print("Error: No valid data found. Exiting.")
        sys.exit(1)

    # 2. 作図実行
    print("\nGenerating Scaled Radius PDF plots...")
    # (1) x = Rc / xi PDF Grid (Linear)
    plot_scaled_radius_pdf_grid(stats_dict, out_dirs, log_scale=False)
    # (2) log10(x) PDF Grid (Log)
    plot_scaled_radius_pdf_grid(stats_dict, out_dirs, log_scale=True)
    # (3) x Overlay (Linear)
    plot_scaled_radius_pdf_overlay(stats_dict, out_dirs, log_scale=False)
    # (4) log10(x) Overlay (Log)
    plot_scaled_radius_pdf_overlay(stats_dict, out_dirs, log_scale=True)
    # (5) Violin plot & Statistics vs Diameter
    plot_scaled_radius_violin_and_stats(df_all, stats_dict, out_dirs)
    # (6) Correlation Length xi PDF (Grid & Overlay)
    plot_correlation_length_xi_pdf(stats_dict, out_dirs)

    # 3. サマリー CSV の作成・保存
    summary_records = []
    for b in BEADS_INFO:
        b_name = b["name"]
        st = stats_dict.get(b_name)
        if st is None:
            continue
        summary_records.append({
            "bead_name": b_name,
            "diameter_um": st["diameter_um"],
            "radius_um": st["radius_um"],
            "label": st["label"],
            "n_points": st["n_points"],
            "x_mean": st["x_mean"],
            "x_std": st["x_std"],
            "x_sem": st["x_sem"],
            "x_median": st["x_median"],
            "x_q25": st["x_q25"],
            "x_q75": st["x_q75"],
            "x_iqr": st["x_iqr"],
            "x_skewness": st["x_skewness"],
            "x_kurtosis": st["x_kurtosis"],
            "x_geom_mean": st["x_geom_mean"],
            "x_geom_std_factor": st["x_geom_std_factor"],
            "log10_x_mean": st["log10_x_mean"],
            "log10_x_std": st["log10_x_std"],
            "xi_mean_um": st["xi_mean"],
            "xi_std_um": st["xi_std"],
            "xi_sem_um": st["xi_sem"],
            "xi_median_um": st["xi_median"],
            "xi_q25_um": st["xi_q25"],
            "xi_q75_um": st["xi_q75"],
        })

    df_summary = pd.DataFrame(summary_records)
    save_csv_to_all(df_summary, "scaled_radius_pdf_summary", out_dirs)

    if not df_all.empty:
        save_csv_to_all(df_all, "scaled_radius_all_points", out_dirs)

    print("\n--- Scaled Radius x = Rc / xi_{i,t} Summary Table ---")
    print(df_summary.to_string(index=False))
    print("\nAll scaled radius PDF figures and summaries generated successfully!")


if __name__ == "__main__":
    main()
