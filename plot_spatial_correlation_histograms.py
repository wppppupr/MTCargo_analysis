#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spatial_correlation_histograms.py
======================================
指定された粒子間距離 r (例: r = 2, 16, 32 um) における
空間配向相関 (cos(theta) および 相対角度 Delta theta) のヒストグラム (PDF) を
全ビーズサイズおよびプールデータ、運動モード別に描画・保存するスクリプト。
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 親ディレクトリのパス設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import hmm_cargo as hc
from libs import spatial_correlation as sc

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]

# rごとのカラーパレット (距離が近い->遠いへのグラデーション)
R_COLORS = ['#d95f02', '#7570b3', '#1b9e77', '#e7298a', '#66a61e', '#e6ab02']


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


def load_all_bead_pairs(
    root_dir: Path,
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    全ビーズサイズについて HMM 特徴抽出および空間配向ペアデータを計算・収集する。
    """
    results_by_bead = {}
    all_pairs_list = []

    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        edirs = find_experiment_dirs(root_dir, bname)
        if not edirs:
            continue

        all_dfs = []
        particle_offset = 0
        for edir in edirs:
            tracks_csv = edir / "beads_tracks.csv"
            if not tracks_csv.exists():
                continue
            df = pd.read_csv(tracks_csv)
            if {'particle', 'frame', 'x', 'y'}.issubset(df.columns):
                df_copy = df[['particle', 'frame', 'x', 'y']].copy()
                df_copy['particle'] += particle_offset
                particle_offset += int(df_copy['particle'].max()) + 1
                df_copy['exp_dir'] = edir.name
                all_dfs.append(df_copy)

        if not all_dfs:
            continue

        df_combined = pd.concat(all_dfs, ignore_index=True)
        X, lengths, df_obs = hc.extract_hmm_features(
            df_combined,
            tau=tau,
            scale=scale,
            frame_interval=frame_interval,
            epsilon=epsilon,
        )

        # HMM フィッティング
        if len(X) >= 20:
            hmm_model = hc.CargoGaussianHMM(
                n_components=2,
                covariance_type="full",
                epsilon=epsilon,
                random_state=42,
            )
            try:
                hmm_model.fit(X, lengths=lengths)
                df_obs['pred_state'] = hmm_model.predict(X, lengths=lengths)
            except Exception:
                df_obs['pred_state'] = 0
        else:
            df_obs['pred_state'] = 0

        df_pairs = sc.compute_dataset_spatial_correlation(df_obs, normalize=True)
        if not df_pairs.empty:
            df_pairs['bead_name'] = bname
            df_pairs['diameter_um'] = dia
            # 相対角度 (度およびラジアン) も追加
            # correlation は cos(theta)
            cos_vals = np.clip(df_pairs['correlation'].to_numpy(), -1.0, 1.0)
            df_pairs['delta_theta_rad'] = np.arccos(cos_vals)
            df_pairs['delta_theta_deg'] = np.degrees(df_pairs['delta_theta_rad'])
            results_by_bead[bname] = df_pairs
            all_pairs_list.append(df_pairs)

    df_all_pooled = pd.concat(all_pairs_list, ignore_index=True) if all_pairs_list else pd.DataFrame()
    return results_by_bead, df_all_pooled


# =========================================================================
# プロット関数群
# =========================================================================

def plot_orientational_histograms_by_r_grid(
    results_by_bead: Dict[str, pd.DataFrame],
    df_all_pooled: pd.DataFrame,
    r_targets: List[float],
    dr: float,
    output_dir: Path,
    mode_filter: str = 'all',
    val_type: str = 'cos',  # 'cos' or 'angle'
):
    """
    ビーズサイズごとのサブプロット（+ 全体プール）で、指定された r における
    配向相関ヒストグラム (PDF) を重ね描きして比較する図。
    """
    n_beads = len(BEADS_INFO)
    # 2行4列 (6ビーズ + 1全プール + 1凡例/説明)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes = axes.flatten()

    is_cos = (val_type == 'cos')
    if is_cos:
        bins = np.linspace(-1.0, 1.0, 25)
        xlabel = r"Spatial Correlation $\cos(\theta_{ij}) = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j$"
        xlim = (-1.05, 1.05)
    else:
        bins = np.linspace(0, 180, 25)
        xlabel = r"Relative Direction $\Delta \theta_{ij}$ [deg]"
        xlim = (0, 180)

    # 1〜6: 各ビーズサイズ
    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ({bname})", fontsize=12, fontweight='bold')

        if bname not in results_by_bead:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        df_b = results_by_bead[bname]
        if mode_filter != 'all':
            df_b = df_b[df_b['mode_category'] == mode_filter]

        for r_idx, r_val in enumerate(r_targets):
            color = R_COLORS[r_idx % len(R_COLORS)]
            r_min = max(0, r_val - dr / 2)
            r_max = r_val + dr / 2
            sub = df_b[(df_b['distance_um'] >= r_min) & (df_b['distance_um'] < r_max)]

            if is_cos:
                vals = sub['correlation'].to_numpy()
            else:
                vals = sub['delta_theta_deg'].to_numpy()

            if len(vals) < 3:
                continue

            # PDF ヒストグラム
            counts, edges = np.histogram(vals, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(
                centers, counts,
                drawstyle='steps-mid',
                color=color,
                lw=2.0,
                label=f"$r \\approx {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals)}$)",
                alpha=0.9,
            )
            ax.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

        # ランダム配向の理論分布
        if is_cos:
            x_theory = np.linspace(-0.98, 0.98, 200)
            p_theory = 1.0 / (np.pi * np.sqrt(1.0 - x_theory**2))
            ax.plot(x_theory, p_theory, 'k--', lw=1.2, alpha=0.5, label="Isotropic (Random)" if idx == 0 else "")
        else:
            p_theory = np.ones(50) / 180.0
            ax.plot(np.linspace(0, 180, 50), p_theory, 'k--', lw=1.2, alpha=0.5, label="Isotropic (Random)" if idx == 0 else "")

        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability Density (PDF)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8, loc='upper center', framealpha=0.85)

    # 7番目: 全ビーズプール (Pooled All Beads)
    ax_pool = axes[6]
    ax_pool.set_title("All Cargo Beads Pooled", fontsize=12, fontweight='bold', color='#333333')
    df_p = df_all_pooled
    if mode_filter != 'all':
        df_p = df_p[df_p['mode_category'] == mode_filter]

    for r_idx, r_val in enumerate(r_targets):
        color = R_COLORS[r_idx % len(R_COLORS)]
        r_min = max(0, r_val - dr / 2)
        r_max = r_val + dr / 2
        sub = df_p[(df_p['distance_um'] >= r_min) & (df_p['distance_um'] < r_max)]

        if is_cos:
            vals = sub['correlation'].to_numpy()
        else:
            vals = sub['delta_theta_deg'].to_numpy()

        if len(vals) < 3:
            continue

        counts, edges = np.histogram(vals, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_c = np.mean(sub['correlation']) if is_cos else np.mean(sub['delta_theta_deg'])
        ax_pool.plot(
            centers, counts,
            drawstyle='steps-mid',
            color=color,
            lw=2.2,
            label=f"$r \\approx {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals)}$, $\\langle C \\rangle={mean_c:.2f}$)" if is_cos else f"$r \\approx {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals)}$)",
            alpha=0.95,
        )
        ax_pool.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

    if is_cos:
        x_theory = np.linspace(-0.98, 0.98, 200)
        p_theory = 1.0 / (np.pi * np.sqrt(1.0 - x_theory**2))
        ax_pool.plot(x_theory, p_theory, 'k--', lw=1.2, alpha=0.5, label="Isotropic (Random)")
    else:
        p_theory = np.ones(50) / 180.0
        ax_pool.plot(np.linspace(0, 180, 50), p_theory, 'k--', lw=1.2, alpha=0.5, label="Isotropic (Random)")

    ax_pool.set_xlim(xlim)
    ax_pool.set_ylim(bottom=0)
    ax_pool.set_xlabel(xlabel, fontsize=10)
    ax_pool.set_ylabel("Probability Density (PDF)", fontsize=10)
    ax_pool.grid(True, linestyle='--', alpha=0.4)
    ax_pool.legend(fontsize=8, loc='upper center', framealpha=0.85)

    # 8番目: サマリーパネル (平均相関 <C(r)> vs r)
    ax_sum = axes[7]
    ax_sum.set_title(r"Mean Correlation $\langle \cos\theta \rangle$ vs $r$", fontsize=12, fontweight='bold')
    r_dense = np.linspace(2, 60, 30)
    r_bin_w = 4.0

    for idx, binfo in enumerate(BEADS_INFO):
        bname = binfo['name']
        dia = binfo['diameter_um']
        col = binfo['color']
        m = binfo['marker']
        if bname not in results_by_bead:
            continue
        df_b = results_by_bead[bname]
        if mode_filter != 'all':
            df_b = df_b[df_b['mode_category'] == mode_filter]

        r_pts, c_pts = [], []
        for rc in r_dense:
            sub = df_b[(df_b['distance_um'] >= rc - r_bin_w/2) & (df_b['distance_um'] < rc + r_bin_w/2)]
            if len(sub) >= 5:
                r_pts.append(rc)
                c_pts.append(sub['correlation'].mean())

        if r_pts:
            ax_sum.plot(r_pts, c_pts, marker=m, color=col, lw=1.5, ms=4, label=f"{dia:.2f} $\\mu$m", alpha=0.85)

    ax_sum.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.6)
    for r_val in r_targets:
        ax_sum.axvline(r_val, color='gray', linestyle=':', lw=1.0, alpha=0.5)

    ax_sum.set_xlim(0, 60)
    ax_sum.set_ylim(-0.3, 0.6)
    ax_sum.set_xlabel(r"Interparticle Distance $r$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_sum.set_ylabel(r"Mean $\langle \cos\theta \rangle$", fontsize=10)
    ax_sum.grid(True, linestyle='--', alpha=0.4)
    ax_sum.legend(fontsize=8, loc='upper right', framealpha=0.85)

    mode_label = "All Pairs" if mode_filter == 'all' else mode_filter.replace('_', ' ').title()
    title_suffix = "Cosine Correlation" if is_cos else "Relative Angle"
    fig.suptitle(f"Spatial Orientational Correlation Histograms at $r = {', '.join([f'{r:.0f}' for r in r_targets])}\\,\\mu\\mathrm{{m}}$ ({mode_label})", fontsize=15, fontweight='bold', y=0.99)

    plt.tight_layout()
    prefix = f"spatial_corr_hist_grid_{val_type}_{mode_filter}"
    fig.savefig(output_dir / f"{prefix}.svg", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"{prefix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {prefix}.svg / .png")


def plot_focused_r_comparison(
    df_all_pooled: pd.DataFrame,
    r_targets: List[float],
    dr: float,
    output_dir: Path,
):
    """
    指定された r = 2, 16, 32 um 等のヒストグラムを洗練された1枚の図（2パネル: Cosine & Angle）
    にまとめて比較するフォーカスプロット。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # パネル1: cos(theta)
    bins_cos = np.linspace(-1.0, 1.0, 31)
    x_theory = np.linspace(-0.99, 0.99, 300)
    p_theory_cos = 1.0 / (np.pi * np.sqrt(1.0 - x_theory**2))

    # 理論等方分布
    ax1.plot(x_theory, p_theory_cos, 'k--', lw=1.5, alpha=0.6, label="Isotropic / Random (Theory)")

    for r_idx, r_val in enumerate(r_targets):
        color = R_COLORS[r_idx % len(R_COLORS)]
        r_min = max(0, r_val - dr / 2)
        r_max = r_val + dr / 2
        sub = df_all_pooled[(df_all_pooled['distance_um'] >= r_min) & (df_all_pooled['distance_um'] < r_max)]

        if len(sub) < 5:
            continue

        vals = sub['correlation'].to_numpy()
        counts, edges = np.histogram(vals, bins=bins_cos, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_c = np.mean(vals)
        sem_c = stats.sem(vals)

        ax1.plot(
            centers, counts,
            drawstyle='steps-mid',
            color=color,
            lw=2.5,
            label=f"$r = {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals)}$, $\\langle \\cos\\theta \\rangle = {mean_c:+.3f} \\pm {sem_c:.3f}$)",
            alpha=0.9,
        )
        ax1.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(0, 1.8)
    ax1.set_xlabel(r"Spatial Orientational Correlation $\cos(\theta_{ij}) = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j$", fontsize=11)
    ax1.set_ylabel("Probability Density Function (PDF)", fontsize=11)
    ax1.set_title("(a) Distribution of Dot Products $\\cos(\\theta_{ij})$", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=9, loc='upper center', framealpha=0.9)

    # パネル2: 相対角度 Delta theta [deg]
    bins_ang = np.linspace(0, 180, 25)
    p_theory_ang = np.ones(50) / 180.0
    ax2.plot(np.linspace(0, 180, 50), p_theory_ang, 'k--', lw=1.5, alpha=0.6, label="Isotropic / Random (Theory)")

    for r_idx, r_val in enumerate(r_targets):
        color = R_COLORS[r_idx % len(R_COLORS)]
        r_min = max(0, r_val - dr / 2)
        r_max = r_val + dr / 2
        sub = df_all_pooled[(df_all_pooled['distance_um'] >= r_min) & (df_all_pooled['distance_um'] < r_max)]

        if len(sub) < 5:
            continue

        vals = sub['delta_theta_deg'].to_numpy()
        counts, edges = np.histogram(vals, bins=bins_ang, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_ang = np.mean(vals)

        ax2.plot(
            centers, counts,
            drawstyle='steps-mid',
            color=color,
            lw=2.5,
            label=f"$r = {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals)}$, $\\langle \\Delta\\theta \\rangle = {mean_ang:.1f}^\\circ$)",
            alpha=0.9,
        )
        ax2.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, 0.015)
    ax2.set_xlabel(r"Relative Motion Angle $\Delta \theta_{ij}$ [deg]", fontsize=11)
    ax2.set_ylabel("Probability Density Function (PDF)", fontsize=11)
    ax2.set_title(r"(b) Distribution of Relative Angles $\Delta \theta_{ij}$", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.suptitle(f"Spatial Orientational Correlation Distributions at Varying Distances $r$ (Pooled Data, $\\Delta r = {dr:.0f}\\,\\mu\\mathrm{{m}}$)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_base = output_dir / "spatial_correlation_histograms_focused_r"
    fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}.svg / .png")


def plot_mode_dependent_r_comparison(
    df_all_pooled: pd.DataFrame,
    r_targets: List[float],
    dr: float,
    output_dir: Path,
):
    """
    運動モード（Run-Run, Tumble-Tumble, All）ごとに r = 2, 16, 32 um のヒストグラムを比較する3パネル図。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    mode_list = [
        ('run_run', 'Run - Run Pairs (Active High-Speed)'),
        ('tumble_tumble', 'Tumble - Tumble Pairs (Paused)'),
        ('all', 'All Pairs (Overall Population)'),
    ]

    bins_cos = np.linspace(-1.0, 1.0, 25)
    x_theory = np.linspace(-0.99, 0.99, 300)
    p_theory_cos = 1.0 / (np.pi * np.sqrt(1.0 - x_theory**2))

    for m_idx, (m_cat, m_title) in enumerate(mode_list):
        ax = axes[m_idx]
        ax.set_title(m_title, fontsize=12, fontweight='bold')
        ax.plot(x_theory, p_theory_cos, 'k--', lw=1.3, alpha=0.6, label="Isotropic (Random)")

        df_m = df_all_pooled if m_cat == 'all' else df_all_pooled[df_all_pooled['mode_category'] == m_cat]

        for r_idx, r_val in enumerate(r_targets):
            color = R_COLORS[r_idx % len(R_COLORS)]
            r_min = max(0, r_val - dr / 2)
            r_max = r_val + dr / 2
            sub = df_m[(df_m['distance_um'] >= r_min) & (df_m['distance_um'] < r_max)]

            if len(sub) < 3:
                continue

            vals = sub['correlation'].to_numpy()
            counts, edges = np.histogram(vals, bins=bins_cos, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mean_c = np.mean(vals)

            ax.plot(
                centers, counts,
                drawstyle='steps-mid',
                color=color,
                lw=2.2,
                label=f"$r \\approx {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals)}$, $\\langle C \\rangle = {mean_c:+.2f}$)",
                alpha=0.9,
            )
            ax.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0, 1.8)
        ax.set_xlabel(r"$\cos(\theta_{ij}) = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j$", fontsize=11)
        ax.set_ylabel("Probability Density (PDF)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8.5, loc='upper center', framealpha=0.9)

    plt.suptitle("Mode-Dependent Spatial Orientational Correlation Distributions across Distances $r$", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_base = output_dir / "spatial_correlation_histograms_by_mode"
    fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}.svg / .png")


def save_histogram_data_csv(
    df_all_pooled: pd.DataFrame,
    r_targets: List[float],
    dr: float,
    output_dir: Path,
):
    """
    プロットした各 r におけるヒストグラムの頻度・統計量サマリーを CSV に保存。
    """
    bins_cos = np.linspace(-1.0, 1.0, 21)
    bin_centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])

    records = []
    for r_val in r_targets:
        r_min = max(0, r_val - dr / 2)
        r_max = r_val + dr / 2
        sub = df_all_pooled[(df_all_pooled['distance_um'] >= r_min) & (df_all_pooled['distance_um'] < r_max)]

        if len(sub) == 0:
            continue

        vals = sub['correlation'].to_numpy()
        counts, _ = np.histogram(vals, bins=bins_cos, density=True)
        raw_counts, _ = np.histogram(vals, bins=bins_cos, density=False)

        for b_idx in range(len(bin_centers)):
            records.append({
                'r_target_um': r_val,
                'r_min_um': r_min,
                'r_max_um': r_max,
                'cos_bin_center': bin_centers[b_idx],
                'cos_bin_min': bins_cos[b_idx],
                'cos_bin_max': bins_cos[b_idx+1],
                'pdf_density': counts[b_idx],
                'raw_count': raw_counts[b_idx],
                'total_pairs_in_r': len(sub),
                'mean_correlation': np.mean(vals),
                'std_correlation': np.std(vals),
                'sem_correlation': stats.sem(vals) if len(vals) > 1 else np.nan,
            })

    df_csv = pd.DataFrame(records)
    csv_path = output_dir / "spatial_correlation_histogram_summary.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"  Saved histogram summary CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot spatial orientational correlation histograms at specified distances r (e.g. r=2, 16, 32 um)."
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figure/spatial_correlation', help="Output directory for figures.")
    parser.add_argument('--r_list', type=float, nargs='+', default=[2.0, 16.0, 32.0], help="List of target distances r in um.")
    parser.add_argument('--dr', type=float, default=4.0, help="Distance bin width dr in um around target r.")
    parser.add_argument('--tau', type=int, default=1, help="Lag time step for velocity calculation.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" Spatial Orientational Correlation Histogram Analysis")
    print("=================================================================")
    print(f"Data Root Directory: {root_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Target r distances:  {args.r_list} um (dr = {args.dr} um)")
    print("=================================================================\n")

    results_by_bead, df_all_pooled = load_all_bead_pairs(root_dir, tau=args.tau)

    if df_all_pooled.empty:
        print("[ERROR] No pair data could be extracted.")
        return

    print(f"\nTotal pairs collected across all beads: {len(df_all_pooled):,}")

    print("\n--- Generating Plots ---")
    # 1. フォーカスプロット (2パネル: cos & angle の r=2, 16, 32 um 比較)
    plot_focused_r_comparison(df_all_pooled, args.r_list, args.dr, output_dir)

    # 2. 粒子径別 8パネルグリッド (cos(theta) 版)
    plot_orientational_histograms_by_r_grid(results_by_bead, df_all_pooled, args.r_list, args.dr, output_dir, mode_filter='all', val_type='cos')

    # 3. 粒子径別 8パネルグリッド (相対角度 Delta theta [deg] 版)
    plot_orientational_histograms_by_r_grid(results_by_bead, df_all_pooled, args.r_list, args.dr, output_dir, mode_filter='all', val_type='angle')

    # 4. モード別 3パネル比較 (Run-Run, Tumble-Tumble, All)
    plot_mode_dependent_r_comparison(df_all_pooled, args.r_list, args.dr, output_dir)

    # 5. CSV データ出力
    save_histogram_data_csv(df_all_pooled, args.r_list, args.dr, output_dir)

    print("\n=================================================================")
    print(" All histogram plots and summaries generated successfully!")
    print("=================================================================")


if __name__ == "__main__":
    main()
