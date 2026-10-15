#!/usr/bin/env python3
"""
plot_diffusion_scaling_master_curve.py

巨視的拡散係数 D vs スケール半径 x = R_c / xi のスケーリングマスターカーブを作成するスクリプト。

縦軸: D_active または D_eff [um^2/s] (両対数プロット)
横軸: x = R_c / xi

重ねる理論線:
- 小粒子側 (RTP 予測): D_active(x) ~ (1/2) * v_0^2 * [g(x)]^2 * tau_p(x)
- 大粒子側 (動的イジング相殺予測): D_eff(x) ~ v_0^2 * tau_xi * x^-2

3層レイヤー構造:
Layer 1 (生データ): 半透明の小さなマーカー (alpha=0.35) 各粒子 i の D_i (横軸 Jitter 付き)
Layer 2 (統計代表値): 大きめの不透明マーカー + エラーバー (Mean +/- SEM)
Layer 3 (理論曲線): 小粒子側 RTP 予測線 (実線) および大粒子側 幾何相殺漸近線 (破線, x^-2)
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# パス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import fit_model as fm
from libs import displacement as dpm
from libs import cal_vel as cv

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
    except Exception:
        pass

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": "#1f77b4", "label": r"$2R_c = 0.63\,\mu\mathrm{m}\ (x = 0.11)$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": "#ff7f0e", "label": r"$2R_c = 1.18\,\mu\mathrm{m}\ (x = 0.21)$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": "#2ca02c", "label": r"$2R_c = 3.37\,\mu\mathrm{m}\ (x = 0.61)$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": "#d62728", "label": r"$2R_c = 5.00\,\mu\mathrm{m}\ (x = 0.90)$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": "#9467bd", "label": r"$2R_c = 7.24\,\mu\mathrm{m}\ (x = 1.30)$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": "#8c564b", "label": r"$2R_c = 20.0\,\mu\mathrm{m}\ (x = 3.60)$"},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]


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


def save_figure_to_all(fig, basename: str, out_dirs: list[Path]):
    """Save matplotlib Figure as both .svg and .png to all valid output directories."""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight', dpi=300)
            fig.savefig(d / f"{basename}.png", bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def plot_diffusion_scaling_curve(
    bead_records: list[dict],
    out_dirs: list[Path],
    v0: float = 0.207,
    xi: float = 2.7774,
    tau0: float = 14.00,
    tau_xi: float = 3.00,
    f_run_avg: float = 0.496,
):
    """
    3層レイヤー構造を持つ D vs x = R_c / xi のスケーリングマスターカーブを描画する。
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    rng = np.random.default_rng(42)

    # -------------------------------------------------------------
    # Layer 3: 理論曲線 (背景側 zorder=3, 4)
    # -------------------------------------------------------------
    x_dense = np.logspace(np.log10(0.04), np.log10(6.5), 500)

    # 2. 小粒子側 漸近線 (RTP 普遍スケーリング予測)
    g_x = fm.master_function(x_dense)
    tau_p_x = tau0 * np.exp(-4.0 * x_dense / 3.0)
    d_active_ideal = 0.5 * (v0 ** 2) * (g_x ** 2) * tau_p_x
    ax.plot(
        x_dense, d_active_ideal,
        color='#1f78b4', linestyle=':', linewidth=2.0, zorder=3,
        label=r'Small-cargo RTP asymptote: $D(x) \approx \frac{1}{2} v_0^2 [g(x)]^2 \tau_{\mathrm{p}}(x)$'
    )
    
    # -------------------------------------------------------------
    # Layer 1 (生データ) & Layer 2 (統計代表値) (前面 zorder=2, 5)
    # -------------------------------------------------------------
    has_l1_label = False
    
    for rec in bead_records:
        d_um = rec["diameter_um"]
        r_c = d_um / 2.0
        x_val = r_c / xi
        marker = rec["marker"]
        color = rec["color"]
        label = rec["label"]
        d_parts = rec["d_parts"]
        d_ens = rec["d_ens"]
        
        valid_p = np.isfinite(d_parts) & (d_parts > 0)
        d_parts_valid = d_parts[valid_p] if len(d_parts) > 0 else np.array([])
        
        # Layer 1: 生データ (各個別粒子 D_long,i, 水平 Jitter 付き)
        if len(d_parts_valid) > 0:
            jitter_offsets = rng.normal(loc=0.0, scale=0.022, size=len(d_parts_valid))
            x_jittered = x_val * (10.0 ** jitter_offsets)
            
            l1_label = r"Individual particle $D_{\mathrm{long}, i}$ ($\alpha=0.35$)" if not has_l1_label else None
            ax.scatter(
                x_jittered, d_parts_valid,
                s=38, color=color, alpha=0.35, edgecolors='none', zorder=2,
                label=l1_label
            )
            has_l1_label = True
            
            # Layer 2: 統計代表値 (Mean +/- SEM)
            mean_d = float(np.mean(d_parts_valid))
            sem_d = float(np.std(d_parts_valid) / np.sqrt(len(d_parts_valid))) if len(d_parts_valid) > 1 else 0.0
            
            plot_d_val = d_ens if (np.isfinite(d_ens) and d_ens > 0) else mean_d
            
            ax.errorbar(
                x_val, plot_d_val, yerr=sem_d,
                fmt=marker, color=color, ecolor='black', elinewidth=1.6, capsize=4.5, capthick=1.2,
                markersize=9.5, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
                label=label
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.06, 6.0)
    ax.set_ylim(2e-3, 0.8)
    
    ax.set_xlabel(r'Scaled Cargo Radius $x = R_c / \xi$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Long-time Diffusion Coefficient $D_{\mathrm{long}}\ [\mu\mathrm{m}^2/\mathrm{s}]$', fontsize=12, fontweight='bold')
    ax.set_title(r'Macroscopic Diffusion Coefficient $D_{\mathrm{long}}$ vs Scaled Radius $x = R_c / \xi$', fontsize=13, fontweight='bold', pad=12)
    
    # 理論パラメータ注釈ボックス
    param_str = (
        r"$\bf{Parameters:}$" + "\n"
        rf"$v_0 = {v0:.3f}\,\mu\mathrm{{m/s}}$" + "\n"
        rf"$\xi = {xi:.2f}\,\mu\mathrm{{m}}$" + "\n"
        rf"$\tau_0 = {tau0:.2f}\,\mathrm{{s}}$" + "\n"
        rf"$\tau_\xi = {tau_xi:.2f}\,\mathrm{{s}}$" + "\n"
        r"$D_{\mathrm{long}} \equiv \lim_{\Delta t \to \mathrm{large}} \frac{\langle \Delta r^2 \rangle}{4 \Delta t}$"
    )
    ax.text(
        0.04, 0.05, param_str,
        transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left',
        fontsize=9.2, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#bbbbbb'),
        zorder=6
    )
    
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    
    # 凡例
    ax.legend(
        loc='upper right', frameon=True, framealpha=0.92, fontsize=8.8,
        edgecolor='#cccccc', labelspacing=0.28
    )
    
    save_figure_to_all(fig, "diffusion_scaling_master_curve", out_dirs)
    save_figure_to_all(fig, "D_vs_scaled_radius_master_curve", out_dirs)
    #save_figure_to_all(fig, "D_long_scaling_master_curve", scaling_dirs if 'scaling_dirs' in locals() else out_dirs)
    plt.close(fig)
    print(f"Saved diffusion scaling master curve to {len(out_dirs)} output directories.")


def main():
    parser = argparse.ArgumentParser(description="Plot macroscopic diffusion coefficient D_long vs scaled radius x = R_c / xi")
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory of dataset")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for plots")
    parser.add_argument('--v0', type=float, default=0.207, help="MT flow speed v0 [um/s]")
    parser.add_argument('--xi', type=float, default=2.7774, help="Nematic correlation length xi [um]")
    parser.add_argument('--tau0', type=float, default=14.00, help="Zero-size orientation persistence time tau0 [s]")
    parser.add_argument('--tau_xi', type=float, default=3.00, help="Ising cancellation timescale tau_xi [s]")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir is not None else find_default_root()
    workspace_dir = CURRENT_DIR

    out_dirs = [
        workspace_dir / "figure" / "scaling",
        workspace_dir / "figure" / "msd",
        workspace_dir / "figure",
        root_dir / "figure" / "scaling",
        root_dir / "figure" / "msd",
        root_dir / "figure",
    ]
    if args.out_dir is not None:
        out_dirs.insert(0, Path(args.out_dir))

    # 各ビーズの個別粒子データのロード (モデルフリー D_long の抽出)
    from MSD import concat_pooled_particles_MSD, calc_pooled_MSD_stats, extract_long_time_diffusion

    intervals = [4.0] * 10
    bead_records = []

    for b in BEADS_INFO:
        b_name = b["name"]
        d_um = b["diameter_um"]
        folder = root_dir / b_name
        df_filt, df_pool, n_tot, n_filt = concat_pooled_particles_MSD(folder, intervals, scale=0.11, alpha_threshold=0.5)
        stats_df = calc_pooled_MSD_stats(df_filt, min_particles=2)
        
        # 個別粒子ごとの D_long,i を直接抽出
        d_long_parts = []
        for pid, grp in df_filt.groupby('unique_particle_id'):
            d_p = extract_long_time_diffusion(grp, min_t=100.0, max_t=300.0)
            if np.isfinite(d_p) and d_p > 0:
                d_long_parts.append(d_p)
                
        d_long_arr = np.array(d_long_parts, dtype=float)
        
        # アンサンブル平均 MSD に対する D_long,ens の直接抽出
        if not stats_df.empty:
            df_stats_ens = pd.DataFrame({
                'lag time': stats_df.index.values,
                'MSD': stats_df['mean'].values
            })
            d_long_ens = extract_long_time_diffusion(df_stats_ens, min_t=100.0, max_t=300.0)
        else:
            d_long_ens = np.nan
            
        bead_records.append({
            "name": b_name,
            "diameter_um": d_um,
            "radius_um": b["radius_um"],
            "marker": b["marker"],
            "color": b["color"],
            "label": b["label"],
            "d_parts": d_long_arr,
            "d_ens": d_long_ens
        })

    plot_diffusion_scaling_curve(
        bead_records=bead_records,
        out_dirs=out_dirs,
        v0=args.v0,
        xi=args.xi,
        tau0=args.tau0,
        tau_xi=args.tau_xi
    )


if __name__ == "__main__":
    main()

