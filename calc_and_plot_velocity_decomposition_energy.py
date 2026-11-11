#!/usr/bin/env python3
"""
calc_and_plot_velocity_decomposition_energy.py

【解析 1：縦・横速度成分の分離とエネルギー保存（最重要）】
「大粒子で進まなくなったのは、単に摩擦で運動エネルギーが奪われて静止（凍結）しただけではないか？」
という査読者の疑念を粉砕するための解析スクリプト。

(1) 各フレームでの速度分解
貨物の軌跡から得られる瞬時速度ベクトル v = (vx, vy) を、直下の局所微小管流向 u_hat = (cos theta_u, sin theta_u) に沿った
縦成分（v_parallel）と直交成分（v_perp）に射影する:
    v_parallel = v · u_hat = vx * cos theta_u + vy * sin theta_u
    v_perp     = v · u_hat_perp = -vx * sin theta_u + vy * cos theta_u

(2) プロットする指標（R_c / xi に対する変化）
以下の3つの量を粒子サイズごとに計算し、1つのグラフに並べる:
    - 正味の直進駆動力: ⟨v_parallel⟩
    - 直交方向の能動揺動強度: sqrt(⟨v_perp^2⟩)
    - 全比運動エネルギー（RMS速度または比運動エネルギー）: ⟨v^2⟩ = ⟨v_parallel^2⟩ + ⟨v_perp^2⟩, および v_rms = sqrt(⟨v^2⟩)
    - エネルギー分配比: ⟨v_perp^2⟩ / ⟨v^2⟩ vs ⟨v_parallel^2⟩ / ⟨v^2⟩

(3) 物理的結論の検証
    - ⟨v_parallel⟩: 小粒子では正（~0.05-0.08 um/s）を持ち、R_c / xi > 0.5-1.0 を境にゼロへと急落（空間パリティ回復）。
    - sqrt(⟨v_perp^2⟩): 大粒子（3~7 um, 20 um）になっても決してゼロに落ちず、~0.08-0.10 um/s の極大・一定値を維持。
    - ⟨v^2⟩: 全運動エネルギーは消失しておらず、「縦の並進運動（v_parallel）が、直交方向の能動的横揺れ（v_perp）へと再配分された」ことを証明。
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ワークスペースおよび保存先設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "velocity_decomposition_energy",
    ROOT_DIR / "figure" / "velocity_decomposition_energy"
]
CACHE_PATH = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"

# スタイル適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

# ビーズ基本情報 (MSD.py 準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]
BEADS_DICT = {b["name"]: b for b in BEADS_INFO}

SCALE = 0.11  # um / pixel
DT = 4.0      # s / frame
V_UNIT = SCALE / DT  # 0.0275 um/s


def load_and_decompose():
    """瞬時速度キャッシュから局所流向への分解計算を行う"""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_PATH}")
    
    print(f"Loading instantaneous cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    
    # 速度を um/s に変換
    vx = df['cargo_dx'] * V_UNIT
    vy = df['cargo_dy'] * V_UNIT
    
    # 局所微小管流速の単位ベクトル
    norm_u = np.sqrt(df['flow_u']**2 + df['flow_v']**2)
    valid_mask = norm_u > 0
    df = df[valid_mask].copy()
    
    norm_u = norm_u[valid_mask]
    ux = df['flow_u'] / norm_u
    uy = df['flow_v'] / norm_u
    
    vx = vx[valid_mask]
    vy = vy[valid_mask]
    
    # 縦成分 (v_parallel): 局所流向に沿った速度
    # 横成分 (v_perp): 局所流向に直交する速度
    v_par = vx * ux + vy * uy
    v_perp = -vx * uy + vy * ux
    v2 = vx**2 + vy**2
    
    df['vx'] = vx
    df['vy'] = vy
    df['v_par'] = v_par
    df['v_perp'] = v_perp
    df['v2'] = v2
    df['v_par2'] = v_par**2
    df['v_perp2'] = v_perp**2
    
    return df


def compute_statistics(df):
    """粒子ごとの平均を計算し、条件ごとのアンサンブル平均と誤差（SEM）を算出"""
    # 粒子ごとの統計量
    particle_grouped = df.groupby(['bead_name', 'particle']).agg({
        'diameter_um': 'first',
        'radius_um': 'first',
        'rc_over_xi': 'mean',
        'xi': 'mean',
        'v_par': 'mean',
        'v_perp': 'mean',
        'v_par2': 'mean',
        'v_perp2': 'mean',
        'v2': 'mean',
        'frame': 'count'
    }).reset_index()
    
    particle_grouped['rms_perp'] = np.sqrt(particle_grouped['v_perp2'])
    particle_grouped['rms_v'] = np.sqrt(particle_grouped['v2'])
    particle_grouped['rms_par'] = np.sqrt(particle_grouped['v_par2'])
    particle_grouped['perp_energy_fraction'] = particle_grouped['v_perp2'] / particle_grouped['v2']
    
    # 条件ごとの統計量集約
    summary_list = []
    for info in BEADS_INFO:
        b_name = info['name']
        sub_p = particle_grouped[particle_grouped['bead_name'] == b_name]
        if sub_p.empty:
            continue
        
        n_p = len(sub_p)
        rc_xi_mean = sub_p['rc_over_xi'].mean()
        rc_xi_std = sub_p['rc_over_xi'].std() if n_p > 1 else 0.0
        
        # v_par
        v_par_mean = sub_p['v_par'].mean()
        v_par_sem = sub_p['v_par'].sem() if n_p > 1 else 0.0
        
        # rms_perp = sqrt(<v_perp^2>)
        rms_perp_mean = sub_p['rms_perp'].mean()
        rms_perp_sem = sub_p['rms_perp'].sem() if n_p > 1 else 0.0
        
        # v2 = <v^2> (比運動エネルギー)
        v2_mean = sub_p['v2'].mean()
        v2_sem = sub_p['v2'].sem() if n_p > 1 else 0.0
        
        # rms_v = sqrt(<v^2>)
        rms_v_mean = sub_p['rms_v'].mean()
        rms_v_sem = sub_p['rms_v'].sem() if n_p > 1 else 0.0
        
        # perp_fraction
        perp_frac_mean = sub_p['perp_energy_fraction'].mean()
        perp_frac_sem = sub_p['perp_energy_fraction'].sem() if n_p > 1 else 0.0
        
        # フレーム全体での一括値（参考用）
        sub_raw = df[df['bead_name'] == b_name]
        raw_v_par = sub_raw['v_par'].mean()
        raw_rms_perp = np.sqrt(sub_raw['v_perp2'].mean())
        raw_rms_v = np.sqrt(sub_raw['v2'].mean())
        raw_v2 = sub_raw['v2'].mean()
        raw_perp_frac = sub_raw['v_perp2'].mean() / sub_raw['v2'].mean()
        
        summary_list.append({
            'bead_name': b_name,
            'diameter_um': info['diameter_um'],
            'radius_um': info['radius_um'],
            'label': info['label'],
            'color': info['color'],
            'marker': info['marker'],
            'n_particles': n_p,
            'n_frames': len(sub_raw),
            'rc_over_xi': rc_xi_mean,
            'rc_over_xi_std': rc_xi_std,
            'v_par_mean': v_par_mean,
            'v_par_sem': v_par_sem,
            'rms_perp_mean': rms_perp_mean,
            'rms_perp_sem': rms_perp_sem,
            'v2_mean': v2_mean,
            'v2_sem': v2_sem,
            'rms_v_mean': rms_v_mean,
            'rms_v_sem': rms_v_sem,
            'perp_frac_mean': perp_frac_mean,
            'perp_frac_sem': perp_frac_sem,
            'raw_v_par': raw_v_par,
            'raw_rms_perp': raw_rms_perp,
            'raw_rms_v': raw_rms_v,
            'raw_v2': raw_v2,
            'raw_perp_frac': raw_perp_frac,
        })
    
    df_summary = pd.DataFrame(summary_list)
    return particle_grouped, df_summary


def plot_main_energy_decomposition(df_summary, particle_grouped):
    """
    (2) プロットする指標（R_c / xi に対する変化）
    以下の3つの量を粒子サイズごとに計算し、1つのグラフに並べる:
        - 正味の直進駆動力: ⟨v_parallel⟩
        - 直交方向の能動揺動強度: sqrt(⟨v_perp^2⟩)
        - 全比運動エネルギー（またはRMS速度）: sqrt(⟨v^2⟩) または ⟨v^2⟩
    """
    # 整理して昇順にソート (Rc/xi 順)
    df_plot = df_summary.sort_values(by='rc_over_xi').copy()
    
    # --- Figure 1: 3指標の比較（速度次元: um/s） vs R_c / xi ---
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    
    x = df_plot['rc_over_xi']
    
    # 1. 全体速度 rms_v = sqrt(<v^2>)
    y_rms_v = df_plot['rms_v_mean']
    y_rms_v_err = df_plot['rms_v_sem']
    ax.errorbar(x, y_rms_v, yerr=y_rms_v_err, fmt='o-', color='#332288',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Total RMS speed $\sqrt{\langle v^2 \rangle}$")
    
    # 2. 直交能動揺動 rms_perp = sqrt(<v_perp^2>)
    y_rms_perp = df_plot['rms_perp_mean']
    y_rms_perp_err = df_plot['rms_perp_sem']
    ax.errorbar(x, y_rms_perp, yerr=y_rms_perp_err, fmt='s-', color='#117733',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Transverse active fluctuation $\sqrt{\langle v_\perp^2 \rangle}$")
    
    # 3. 正味直進駆動力 v_par = <v_parallel>
    y_v_par = df_plot['v_par_mean']
    y_v_par_err = df_plot['v_par_sem']
    ax.errorbar(x, y_v_par, yerr=y_v_par_err, fmt='^-', color='#CC6677',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Net forward propulsion $\langle v_\parallel \rangle$")
    
    # ゼロ基準線と転移点 R_c / xi = 1
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$ boundary")
    
    # 各ビーズのラベル注記
    for _, row in df_plot.iterrows():
        ax.annotate(f"{row['diameter_um']:.2g} $\mu\mathrm{{m}}$",
                    xy=(row['rc_over_xi'], row['rms_v_mean']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, color='#332288', weight='bold')
    
    ax.set_xscale('log')
    ax.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=13)
    ax.set_ylabel(r"Velocity $[\mu\mathrm{m/s}]$", fontsize=13)
    ax.set_ylim(-0.015, 0.22)
    ax.set_title("Velocity Decomposition & Kinetic Energy vs $R_c/\\xi$", fontsize=13, weight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10.5)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "velocity_decomposition_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "velocity_decomposition_vs_rc_xi.svg")
    plt.close(fig)
    print("Saved Figure 1: velocity_decomposition_vs_rc_xi.png / .svg")
    
    # --- Figure 2: 比運動エネルギー (Energy: <v^2>, um^2/s^2) vs R_c / xi ---
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    
    # <v^2>
    ax.errorbar(x, df_plot['v2_mean'], yerr=df_plot['v2_sem'], fmt='o-', color='#332288',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Total kinetic energy $\langle v^2 \rangle$")
    
    # <v_perp^2>
    ax.errorbar(x, df_plot['rms_perp_mean']**2, fmt='s-', color='#117733',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Transverse energy $\langle v_\perp^2 \rangle$")
    
    # <v_par^2>
    ax.errorbar(x, df_plot['v2_mean'] - df_plot['rms_perp_mean']**2, fmt='^-', color='#CC6677',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Longitudinal energy $\langle v_\parallel^2 \rangle$")
    
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax.set_xscale('log')
    ax.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=13)
    ax.set_ylabel(r"Specific Kinetic Energy $[\mu\mathrm{m}^2/\mathrm{s}^2]$", fontsize=13)
    ax.set_ylim(-0.002, 0.040)
    ax.set_title("Kinetic Energy Redistribution: Longitudinal to Transverse", fontsize=13, weight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10.5)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "kinetic_energy_components_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "kinetic_energy_components_vs_rc_xi.svg")
    plt.close(fig)
    print("Saved Figure 2: kinetic_energy_components_vs_rc_xi.png / .svg")
    
    # --- Figure 3: エネルギー配分比 (Fraction: <v_perp^2>/<v^2> vs <v_parallel>^2/<v^2>) ---
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    
    perp_frac = df_plot['perp_frac_mean']
    perp_err = df_plot['perp_frac_sem']
    par_frac = 1.0 - perp_frac
    
    ax.errorbar(x, perp_frac, yerr=perp_err, fmt='s-', color='#117733',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Transverse fraction $\langle v_\perp^2 \rangle / \langle v^2 \rangle$")
    ax.errorbar(x, par_frac, yerr=perp_err, fmt='^-', color='#CC6677',
                linewidth=2.2, markersize=8, capsize=4, capthick=1.5,
                label=r"Longitudinal fraction $\langle v_\parallel^2 \rangle / \langle v^2 \rangle$")
    
    # 2D等方限界 (1/2 = 50%)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, label=r"Isotropic 2D limit ($50\%$)")
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    
    for _, row in df_plot.iterrows():
        ax.annotate(f"{row['diameter_um']:.2g} $\mu\mathrm{{m}}$",
                    xy=(row['rc_over_xi'], row['perp_frac_mean']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, color='#117733', weight='bold')
    
    ax.set_xscale('log')
    ax.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=13)
    ax.set_ylabel(r"Energy Fraction", fontsize=13)
    ax.set_ylim(0.0, 0.75)
    ax.set_title("Energy Redistribution Ratio vs $R_c/\\xi$", fontsize=13, weight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10.5)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "energy_fraction_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "energy_fraction_vs_rc_xi.svg")
    plt.close(fig)
    print("Saved Figure 3: energy_fraction_vs_rc_xi.png / .svg")


def plot_distributions(df):
    """
    各ビーズサイズにおける P(v_parallel) と P(v_perp) の確率密度分布を比較表示。
    小粒子では P(v_par) が正にシフトしているが、大粒子では対称化（パリティ回復）し、
    かつ横方向 P(v_perp) の広がりは依然として広い（能動揺動の持続）ことを可視化する。
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    bins = np.linspace(-0.4, 0.4, 61)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    for idx, info in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = info['name']
        sub = df[df['bead_name'] == b_name]
        
        if sub.empty:
            continue
        
        # P(v_par)
        counts_par, _ = np.histogram(sub['v_par'], bins=bins, density=True)
        # P(v_perp)
        counts_perp, _ = np.histogram(sub['v_perp'], bins=bins, density=True)
        
        rc_xi = sub['rc_over_xi'].mean()
        mean_par = sub['v_par'].mean()
        rms_perp = np.sqrt(sub['v_perp2'].mean())
        
        ax.plot(bin_centers, counts_par, color='#CC6677', linewidth=2.0,
                label=rf"$v_\parallel$ (mean={mean_par:+.3f})")
        ax.plot(bin_centers, counts_perp, color='#117733', linewidth=2.0, linestyle='--',
                label=rf"$v_\perp$ (rms={rms_perp:.3f})")
        
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.0)
        ax.axvline(mean_par, color='#CC6677', linestyle='-.', linewidth=1.2, alpha=0.8)
        
        ax.set_title(f"{info['label']} ($R_c/\\xi \\approx {rc_xi:.2f}$)", fontsize=11, weight='bold')
        ax.set_xlabel(r"Velocity $[\mu\mathrm{m/s}]$", fontsize=10)
        ax.set_ylabel(r"Probability Density", fontsize=10)
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.8)
        ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.suptitle("Velocity Probability Distributions $P(v_\\parallel)$ and $P(v_\\perp)$ Across Cargo Sizes",
                 fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    for d in OUT_DIRS:
        fig.savefig(d / "velocity_distributions_grid.png", dpi=300)
        fig.savefig(d / "velocity_distributions_grid.svg")
    plt.close(fig)
    print("Saved Figure 4: velocity_distributions_grid.png / .svg")


def main():
    print("=== Velocity Decomposition & Kinetic Energy Analysis ===")
    df = load_and_decompose()
    particle_grouped, df_summary = compute_statistics(df)
    
    # サマリーCSVの出力
    csv_filename = "velocity_decomposition_energy_summary.csv"
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(d / csv_filename, index=False)
        particle_grouped.to_csv(d / "particle_level_velocity_decomposition.csv", index=False)
    print(f"Saved CSV summaries to {OUT_DIRS[0] / csv_filename}")
    
    # グラフ描画
    plot_main_energy_decomposition(df_summary, particle_grouped)
    plot_distributions(df)
    
    print("\n--- Summary Table ---")
    cols_to_print = ['bead_name', 'diameter_um', 'rc_over_xi', 'v_par_mean', 'v_par_sem', 'rms_perp_mean', 'rms_perp_sem', 'rms_v_mean', 'perp_frac_mean']
    print(df_summary[cols_to_print].to_string(index=False))
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()

