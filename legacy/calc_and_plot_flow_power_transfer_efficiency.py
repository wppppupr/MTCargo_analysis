#!/usr/bin/env python3
"""
calc_and_plot_flow_power_transfer_efficiency.py

【解析 3：流動伝達効率（Power Transfer Efficiency: eta_T）の定量化】
微小管流 u から貨物 v へと伝達される機械的仕事率（単位質量あたり）は W ~ v · u。
ここから、無次元の流動伝達効率 eta_T を定義して評価する:

    eta_T(R_c) = < v · u > / ( sqrt(< |v|^2 >) * sqrt(< |u|^2 >) )
                = < v · u > / ( v_rms * u_rms )

物理的意味:
    - 純粋なスレーブ輸送（完全同調）: eta_T = 1
    - 直交ケージング・相殺極限: eta_T = 0
    - 逆走抵抗: eta_T < 0

横軸 R_c / xi に対する eta_T のスケーリング曲線をプロットし、
「貨物が微小管流から推進パワーを吸い上げる効率」が、相関長 R_c / xi ~ 1 で
急峻にゼロへと遮断（Quenching）される様子を定量化・モデルフィッティングする。
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ワークスペースおよび保存先設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "flow_power_transfer_efficiency",
    ROOT_DIR / "figure" / "flow_power_transfer_efficiency"
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

SCALE = 0.11  # um / pixel
DT = 4.0      # s / frame
V_UNIT = SCALE / DT  # 0.0275 um/s


def load_data():
    """瞬時速度キャッシュをロードし、内積および二乗速度を計算"""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_PATH}")
    
    print(f"Loading instantaneous cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    
    df['vx'] = df['cargo_dx'] * V_UNIT
    df['vy'] = df['cargo_dy'] * V_UNIT
    df['ux'] = df['flow_u'] * V_UNIT
    df['uy'] = df['flow_v'] * V_UNIT
    
    # 機械的仕事率に比例する内積 v · u
    df['v_dot_u'] = df['vx'] * df['ux'] + df['vy'] * df['uy']
    df['v2'] = df['vx']**2 + df['vy']**2
    df['u2'] = df['ux']**2 + df['uy']**2
    
    # 局所流向への射影 v_parallel
    norm_u = np.sqrt(df['u2'])
    valid = norm_u > 0
    df = df[valid].copy()
    norm_u = norm_u[valid]
    df['v_par'] = (df['vx'] * df['ux'] + df['vy'] * df['uy']) / norm_u
    
    return df


def compute_efficiency_and_bootstrap(df, n_boot=2000, seed=42):
    """
    粒子レベルでの集計およびトラジェクトリ単位（クラスター）ブートストラップによる
    eta_T のアンサンブル値および 95% 信頼区間を算出
    """
    rng = np.random.default_rng(seed)
    summary_list = []
    particle_list = []
    
    for info in BEADS_INFO:
        b_name = info['name']
        sub = df[df['bead_name'] == b_name]
        if sub.empty:
            continue
        
        particles = sub['particle'].unique()
        n_p = len(particles)
        
        # 1. 粒子ごとの統計量
        p_agg = sub.groupby('particle').agg({
            'diameter_um': 'first',
            'radius_um': 'first',
            'rc_over_xi': 'mean',
            'xi': 'mean',
            'v_dot_u': 'mean',
            'v2': 'mean',
            'u2': 'mean',
            'frame': 'count'
        }).reset_index()
        p_agg['v_rms'] = np.sqrt(p_agg['v2'])
        p_agg['u_rms'] = np.sqrt(p_agg['u2'])
        p_agg['eta_T'] = p_agg['v_dot_u'] / (p_agg['v_rms'] * p_agg['u_rms'])
        p_agg['bead_name'] = b_name
        particle_list.append(p_agg)
        
        # 2. 条件全体の一括値 (Global Ensemble)
        mean_v_dot_u = sub['v_dot_u'].mean()
        v_rms_global = np.sqrt(sub['v2'].mean())
        u_rms_global = np.sqrt(sub['u2'].mean())
        eta_global = mean_v_dot_u / (v_rms_global * u_rms_global)
        
        # 3. トラジェクトリ単位のブートストラップ (Cluster Bootstrap)
        # 各粒子の集約値を利用して高速リサンプリング
        p_v_dot_u = p_agg['v_dot_u'].values
        p_v2 = p_agg['v2'].values
        p_u2 = p_agg['u2'].values
        p_weights = p_agg['frame'].values.astype(float)
        p_weights /= p_weights.sum()
        
        if n_p > 1:
            boot_etas = []
            for _ in range(n_boot):
                idx = rng.choice(n_p, size=n_p, replace=True)
                # フレーム数で重み付けしたアンサンブル平均
                w = p_agg['frame'].values[idx]
                w_tot = w.sum()
                b_v_dot_u = np.sum(p_v_dot_u[idx] * w) / w_tot
                b_v2 = np.sum(p_v2[idx] * w) / w_tot
                b_u2 = np.sum(p_u2[idx] * w) / w_tot
                b_eta = b_v_dot_u / (np.sqrt(b_v2) * np.sqrt(b_u2))
                boot_etas.append(b_eta)
            
            boot_mean = np.mean(boot_etas)
            boot_std = np.std(boot_etas)
            boot_ci_low = np.percentile(boot_etas, 2.5)
            boot_ci_high = np.percentile(boot_etas, 97.5)
        else:
            boot_mean = eta_global
            boot_std = 0.0
            boot_ci_low = eta_global
            boot_ci_high = eta_global
        
        # 粒子ごとの平均と標準誤差
        eta_p_mean = p_agg['eta_T'].mean()
        eta_p_sem = p_agg['eta_T'].sem() if n_p > 1 else 0.0
        
        summary_list.append({
            'bead_name': b_name,
            'diameter_um': info['diameter_um'],
            'radius_um': info['radius_um'],
            'label': info['label'],
            'color': info['color'],
            'marker': info['marker'],
            'n_particles': n_p,
            'n_frames': len(sub),
            'rc_over_xi': p_agg['rc_over_xi'].mean(),
            'rc_over_xi_std': p_agg['rc_over_xi'].std() if n_p > 1 else 0.0,
            'eta_T_global': eta_global,
            'boot_mean': boot_mean,
            'boot_std': boot_std,
            'boot_ci_low': boot_ci_low,
            'boot_ci_high': boot_ci_high,
            'eta_p_mean': eta_p_mean,
            'eta_p_sem': eta_p_sem,
            'v_rms': v_rms_global,
            'u_rms': u_rms_global,
            'mean_v_dot_u': mean_v_dot_u,
        })
    
    df_summary = pd.DataFrame(summary_list)
    df_particle = pd.concat(particle_list, ignore_index=True)
    return df_summary, df_particle


# 減衰モデルの定義
def sigmoid_quenching(x, eta_0, x_c, alpha):
    """シグモイド遮断モデル: eta(x) = eta_0 / (1 + (x / x_c)^alpha)"""
    return eta_0 / (1.0 + (x / x_c)**alpha)

def exponential_quenching(x, eta_0, lambda_c):
    """指数減衰モデル: eta(x) = eta_0 * exp(-x / lambda_c)"""
    return eta_0 * np.exp(-x / lambda_c)


def plot_efficiency_scaling(df_summary, df_particle):
    """
    eta_T vs R_c / xi のスケーリング曲線をプロットし、フィッティングを実施
    """
    df_plot = df_summary.sort_values(by='rc_over_xi').copy()
    
    x_data = df_plot['rc_over_xi'].values
    y_data = df_plot['eta_T_global'].values
    y_err = df_plot['boot_std'].values
    
    # --- フィッティング ---
    # シグモイド遮断モデル: eta_0 / (1 + (x/x_c)^alpha)
    popt_sig, _ = curve_fit(
        sigmoid_quenching, x_data, y_data,
        p0=[0.65, 0.4, 2.0],
        bounds=([0.4, 0.1, 0.5], [1.0, 2.0, 5.0]),
        maxfev=5000
    )
    # 指数減衰モデル
    popt_exp, _ = curve_fit(
        exponential_quenching, x_data, y_data,
        p0=[0.65, 0.5],
        bounds=([0.4, 0.1], [1.0, 3.0]),
        maxfev=5000
    )
    
    x_fit = np.logspace(np.log10(x_data.min() * 0.7), np.log10(x_data.max() * 1.4), 200)
    y_fit_sig = sigmoid_quenching(x_fit, *popt_sig)
    y_fit_exp = exponential_quenching(x_fit, *popt_exp)
    
    # --- Figure 1: メインスケーリング図 (Log-Linear) ---
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    # 粒子ごとのプロット（薄い散布図）
    for info in BEADS_INFO:
        b_name = info['name']
        sub_p = df_particle[df_particle['bead_name'] == b_name]
        ax.scatter(sub_p['rc_over_xi'], sub_p['eta_T'],
                   color=info['color'], marker=info['marker'],
                   alpha=0.35, s=40, edgecolors='none')
    
    # シグモイドフィット曲線
    label_sig = (rf"Quenching Model: $\eta_0 / [1 + (R_c/\xi_c)^\alpha]$" + "\n" +
                 rf"($\eta_0={popt_sig[0]:.2f},\ R_{{c,\mathrm{{crit}}}}/\xi={popt_sig[1]:.2f},\ \alpha={popt_sig[2]:.2f}$)")
    ax.plot(x_fit, y_fit_sig, color='#332288', linestyle='-', linewidth=2.4, label=label_sig)
    
    # 指数フィット曲線
    label_exp = rf"Exp decay: $\eta_0 e^{{-R_c/\lambda_c}}$ ($\lambda_c/\xi={popt_exp[1]:.2f}$)"
    ax.plot(x_fit, y_fit_exp, color='gray', linestyle='--', linewidth=1.8, alpha=0.8, label=label_exp)
    
    # 各ビーズのアンサンブル平均 + ブートストラップ誤差棒
    for _, row in df_plot.iterrows():
        yerr_bar = [[row['eta_T_global'] - row['boot_ci_low']], [row['boot_ci_high'] - row['eta_T_global']]]
        ax.errorbar(row['rc_over_xi'], row['eta_T_global'],
                    yerr=yerr_bar, fmt=row['marker'], color=row['color'],
                    markersize=10, markeredgewidth=1.8, markeredgecolor='black',
                    capsize=4.5, capthick=1.5, zorder=5)
        # ラベル注記
        ax.annotate(f"{row['diameter_um']:.2g} $\mu\mathrm{{m}}$\n($\eta_T={row['eta_T_global']:.2f}$)",
                    xy=(row['rc_over_xi'], row['eta_T_global']),
                    xytext=(0, 12 if row['rc_over_xi'] < 1.0 else 15), textcoords='offset points',
                    ha='center', fontsize=9.5, weight='bold', color=row['color'])
    
    # 基準線
    ax.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    ax.axhline(1.0, color='darkgray', linestyle=':', linewidth=1.2, alpha=0.8, label=r"Ideal Slave Limit ($\eta_T = 1$)")
    ax.axvline(1.0, color='crimson', linestyle='-.', linewidth=1.5, alpha=0.7, label=r"Correlation Boundary $R_c = \xi$")
    
    ax.set_xscale('log')
    ax.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=13)
    ax.set_ylabel(r"Flow Power Transfer Efficiency $\eta_T$", fontsize=13)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlim(0.04, 4.0)
    ax.set_title(r"Power Transfer Efficiency $\eta_T$ vs $R_c/\xi$ (Quenching Transition)",
                 fontsize=13, weight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9.5, frameon=True, framealpha=0.92)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "flow_power_transfer_efficiency_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "flow_power_transfer_efficiency_vs_rc_xi.svg")
    plt.close(fig)
    print("Saved Figure 1: flow_power_transfer_efficiency_vs_rc_xi.png / .svg")
    
    # --- Figure 2: Work rate <v · u> と速度成分の個別表示 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    
    # Work rate <v · u>
    ax1.plot(df_plot['rc_over_xi'], df_plot['mean_v_dot_u'], 'o-', color='#882255',
             linewidth=2.2, markersize=8)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax1.axvline(1.0, color='crimson', linestyle=':', linewidth=1.2, label=r"$R_c = \xi$")
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12)
    ax1.set_ylabel(r"Mechanical Work Rate $\langle \mathbf{v} \cdot \mathbf{u} \rangle\ [\mu\mathrm{m}^2/\mathrm{s}^2]$", fontsize=11)
    ax1.set_title(r"Specific Mechanical Work Rate $\langle \mathbf{v} \cdot \mathbf{u} \rangle$", fontsize=12, weight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=10)
    
    for _, row in df_plot.iterrows():
        ax1.annotate(f"{row['diameter_um']:.2g} $\mu\mathrm{{m}}$",
                     xy=(row['rc_over_xi'], row['mean_v_dot_u']),
                     xytext=(0, 7), textcoords='offset points',
                     ha='center', fontsize=9, color='#882255', weight='bold')
    
    # RMS Speeds v_rms and u_rms
    ax2.plot(df_plot['rc_over_xi'], df_plot['v_rms'], 's-', color='#332288',
             linewidth=2.0, markersize=7, label=r"Cargo $v_{\mathrm{rms}}$")
    ax2.plot(df_plot['rc_over_xi'], df_plot['u_rms'], '^-', color='#117733',
             linewidth=2.0, markersize=7, label=r"Flow $u_{\mathrm{rms}}$")
    ax2.axvline(1.0, color='crimson', linestyle=':', linewidth=1.2, label=r"$R_c = \xi$")
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12)
    ax2.set_ylabel(r"RMS Speed $[\mu\mathrm{m/s}]$", fontsize=12)
    ax2.set_title(r"Cargo and Flow RMS Speeds", fontsize=12, weight='bold')
    ax2.set_ylim(0.0, 0.22)
    ax2.grid(True, which='both', linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "work_rate_and_rms_speeds.png", dpi=300)
        fig.savefig(d / "work_rate_and_rms_speeds.svg")
    plt.close(fig)
    print("Saved Figure 2: work_rate_and_rms_speeds.png / .svg")
    
    return popt_sig, popt_exp


def main():
    print("=== Flow Power Transfer Efficiency (eta_T) Analysis ===")
    df = load_data()
    df_summary, df_particle = compute_efficiency_and_bootstrap(df)
    
    # フィッティングおよび作図
    popt_sig, popt_exp = plot_efficiency_scaling(df_summary, df_particle)
    
    # CSV 保存
    csv_filename = "flow_power_transfer_efficiency_summary.csv"
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(d / csv_filename, index=False)
        df_particle.to_csv(d / "particle_level_power_transfer_efficiency.csv", index=False)
    print(f"Saved summary CSV to {OUT_DIRS[0] / csv_filename}")
    
    print("\n--- Power Transfer Efficiency Summary Table ---")
    cols_to_print = ['bead_name', 'diameter_um', 'rc_over_xi', 'eta_T_global', 'boot_std', 'boot_ci_low', 'boot_ci_high', 'mean_v_dot_u']
    print(df_summary[cols_to_print].to_string(index=False))
    
    print("\n--- Fitting Parameters ---")
    print(f"Sigmoid Quenching: eta_0 = {popt_sig[0]:.3f}, R_c,crit / xi = {popt_sig[1]:.3f}, alpha = {popt_sig[2]:.3f}")
    print(f"Exponential Decay: eta_0 = {popt_exp[0]:.3f}, lambda_c / xi = {popt_exp[1]:.3f}")
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()

