#!/usr/bin/env python3
"""
calc_and_plot_velocity_potential_2d.py

【解析 2：速度空間の有効ポテンシャル V(v_parallel, v_perp) の 2D 等高線図 & 偶奇分解による対称性定量化】
配向角で行ったボルツマン反転を、速度空間 (v_parallel, v_perp) に対して直接実行する。

(1) 2次元確率密度の推定:
    各ビーズサイズについて、全軌跡ポイントの (v_parallel, v_perp) の 2D ガウス KDE
    から確率密度 P(v_parallel, v_perp) を推定。

(2) 速度空間ポテンシャルの反転:
    V_eff(v_parallel, v_perp) = - ln [ P(v_parallel, v_perp) / P_max ]

(3) 偶奇分解（Even-Odd Decomposition）による大域的対称性定量評価:
    ポテンシャルを縦速度 v_parallel に関して偶関数（対称な井戸）と奇関数（パリティを破る傾斜場）に分解:
        V_even(v_parallel) = [ V(v_parallel) + V(-v_parallel) ] / 2
        V_odd(v_parallel)  = [ V(v_parallel) - V(-v_parallel) ] / 2
    大域的奇成分ノルム（Global Odd-Component Integral）:
        A_odd = int_0^{v_max} |V_odd(v_parallel)| dv_parallel
    無次元化非対称寄与率（Asymmetry Ratio）:
        eta_asym = [ int_0^{v_max} |V_odd(v_parallel)| dv_parallel ] / [ int_0^{v_max} V_even(v_parallel) dv_parallel ]

(4) トポロジー・地形変化の可視化:
    - 2D コンター図（等高線 + カラーマップ）
    - 偶奇分解プロット: 各ビーズサイズの V(v_parallel), V_even, V_odd の 6パネル比較
    - 偶奇成分の大域的積分指標 A_odd および eta_asym のスケーリング曲線
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ワークスペースおよび保存先設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "velocity_potential_2d",
    ROOT_DIR / "figure" / "velocity_potential_2d"
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
V_MAX_INTEGRAL = 0.15  # um/s: 積分範囲 v_max


def load_and_decompose():
    """瞬時速度キャッシュから局所流向への分解計算を行う"""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_PATH}")
    
    print(f"Loading instantaneous cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    
    vx = df['cargo_dx'] * V_UNIT
    vy = df['cargo_dy'] * V_UNIT
    
    norm_u = np.sqrt(df['flow_u']**2 + df['flow_v']**2)
    valid_mask = norm_u > 0
    df = df[valid_mask].copy()
    
    norm_u = norm_u[valid_mask]
    ux = df['flow_u'] / norm_u
    uy = df['flow_v'] / norm_u
    
    vx = vx[valid_mask]
    vy = vy[valid_mask]
    
    # 縦成分 (v_parallel): 局所微小管流向に沿った速度
    # 直交成分 (v_perp): 局所微小管流向に垂直な速度
    df['v_par'] = vx * ux + vy * uy
    df['v_perp'] = -vx * uy + vy * ux
    df['v2'] = vx**2 + vy**2
    
    return df


def compute_effective_potentials(df, grid_limit=0.25, n_grid=101, v_max=V_MAX_INTEGRAL):
    """
    各ビーズサイズについて 2D ガウス KDE を計算し、
    V_eff(v_parallel, v_perp) = - ln( P / P_max ) を求める。
    さらに v_perp = 0 に沿ったスライスを抽出し、偶奇分解および積分ノルムを算出する。
    """
    grid_1d = np.linspace(-grid_limit, grid_limit, n_grid)
    V_PAR, V_PERP = np.meshgrid(grid_1d, grid_1d)
    pts = np.vstack([V_PAR.ravel(), V_PERP.ravel()])
    
    center_idx = n_grid // 2  # v = 0 のインデックス
    
    # 積分範囲のグリッド [0, v_max]
    v_mask = (grid_1d >= 0) & (grid_1d <= v_max)
    pos_v = grid_1d[v_mask]
    n_pts_pos = len(pos_v)
    
    results = {}
    summary_records = []
    
    for info in BEADS_INFO:
        b_name = info['name']
        sub = df[df['bead_name'] == b_name]
        if sub.empty:
            continue
        
        v_p = sub['v_par'].values
        v_o = sub['v_perp'].values
        rc_xi = sub['rc_over_xi'].mean()
        
        # 2D KDE 推定
        kde = gaussian_kde(np.vstack([v_p, v_o]), bw_method='silverman')
        P_2d = kde(pts).reshape(V_PAR.shape)
        
        p_max = P_2d.max()
        V_eff = -np.log(np.maximum(P_2d / p_max, 1e-12))
        
        # ポテンシャルの最小点 (最頻速度)
        min_idx = np.unravel_index(np.argmin(V_eff), V_eff.shape)
        min_v_par = grid_1d[min_idx[1]]
        min_v_perp = grid_1d[min_idx[0]]
        
        # v_perp = 0 に沿った縦方向スライス V(v_parallel, 0)
        slice_par = V_eff[center_idx, :]
        
        # 偶奇分解 (Even-Odd Decomposition)
        # 正方向 V(+v) と 負方向 V(-v) を抽出
        V_pos = slice_par[center_idx : center_idx + n_pts_pos]
        V_neg = slice_par[center_idx - n_pts_pos + 1 : center_idx + 1][::-1]
        
        V_even = 0.5 * (V_pos + V_neg)
        V_odd = 0.5 * (V_pos - V_neg)
        
        # 大域的奇成分ノルム A_odd = int_0^{v_max} |V_odd(v)| dv
        A_odd = float(trapezoid(np.abs(V_odd), pos_v))
        
        # 偶成分の積分 (井戸の全体ポテンシャル) I_even = int_0^{v_max} V_even(v) dv
        I_even = float(trapezoid(V_even, pos_v))
        
        # 無次元化非対称寄与率 eta_asym = A_odd / I_even
        eta_asym = float(A_odd / I_even if I_even > 0 else 0.0)
        
        # 局所非対称指標: ΔV_asym = V(-0.1, 0) - V(+0.1, 0)
        idx_pos = np.argmin(np.abs(grid_1d - 0.10))
        idx_neg = np.argmin(np.abs(grid_1d - (-0.10)))
        v_eff_pos = slice_par[idx_pos]
        v_eff_neg = slice_par[idx_neg]
        delta_v_asym = float(v_eff_neg - v_eff_pos)
        
        # 歪度 (Skewness)
        skew_par = float((sub['v_par'] - sub['v_par'].mean()).pow(3).mean() / (sub['v_par'].std()**3))
        skew_perp = float((sub['v_perp'] - sub['v_perp'].mean()).pow(3).mean() / (sub['v_perp'].std()**3))
        
        results[b_name] = {
            'info': info,
            'rc_over_xi': rc_xi,
            'V_PAR': V_PAR,
            'V_PERP': V_PERP,
            'grid_1d': grid_1d,
            'P_2d': P_2d,
            'V_eff': V_eff,
            'slice_par': slice_par,
            'pos_v': pos_v,
            'V_pos': V_pos,
            'V_neg': V_neg,
            'V_even': V_even,
            'V_odd': V_odd,
            'A_odd': A_odd,
            'I_even': I_even,
            'eta_asym': eta_asym,
            'min_v_par': min_v_par,
            'min_v_perp': min_v_perp,
            'delta_v_asym': delta_v_asym,
            'skew_par': skew_par,
            'skew_perp': skew_perp,
            'n_pts': len(sub),
        }
        
        summary_records.append({
            'bead_name': b_name,
            'diameter_um': info['diameter_um'],
            'radius_um': info['radius_um'],
            'rc_over_xi': rc_xi,
            'n_frames': len(sub),
            'v_max_integral_um_s': v_max,
            'A_odd_integral': A_odd,
            'I_even_integral': I_even,
            'eta_asym_ratio': eta_asym,
            'delta_v_asym_at_01um_s': delta_v_asym,
            'skewness_v_parallel': skew_par,
            'skewness_v_perp': skew_perp,
            'min_v_par': min_v_par,
            'min_v_perp': min_v_perp,
            'v_par_mean': sub['v_par'].mean(),
            'v_par_std': sub['v_par'].std(),
            'v_perp_std': sub['v_perp'].std(),
        })
    
    df_summary = pd.DataFrame(summary_records)
    return results, df_summary


def plot_even_odd_decomposition_grid(results):
    """
    全6サイズについて、V(v_parallel), V_even(v_parallel), V_odd(v_parallel) の
    偶奇分解を 2x3 グリッドで視覚化するプロット。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, info in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = info['name']
        data = results[b_name]
        
        pos_v = data['pos_v']
        V_pos = data['V_pos']
        V_neg = data['V_neg']
        V_even = data['V_even']
        V_odd = data['V_odd']
        rc_xi = data['rc_over_xi']
        
        # 偶成分 (対称な井戸)
        ax.plot(pos_v, V_even, color='#332288', linewidth=2.4, label=r"$\tilde{\mathcal{V}}_{\mathrm{even}}$ (Symmetric well)")
        # 奇成分 (パリティ破れ・傾斜場)
        ax.plot(pos_v, V_odd, color='#CC6677', linewidth=2.4, linestyle='--', label=r"$\tilde{\mathcal{V}}_{\mathrm{odd}}$ (Parity breaking)")
        # 奇成分の絶対値（積分対象）
        ax.fill_between(pos_v, 0, np.abs(V_odd), color='#CC6677', alpha=0.18,
                        label=rf"$\mathcal{{A}}_{{\mathrm{{odd}}}}={data['A_odd']:.3f}$")
        
        # 元の正方向・負方向の線（薄く表示）
        ax.plot(pos_v, V_pos, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, label=r"$\tilde{\mathcal{V}}(+v_\parallel)$")
        ax.plot(pos_v, V_neg, color='gray', linestyle='-.', linewidth=1.2, alpha=0.7, label=r"$\tilde{\mathcal{V}}(-v_\parallel)$")
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
        
        # タイトルおよび情報
        ax.set_title(rf"{info['label']}  ($R_c/\xi \approx {rc_xi:.2f}$)",
                     fontsize=12.5, weight='bold', pad=8)
        
        # 評価指標のバッジ
        ax.text(0.05, 0.93,
                rf"$\mathcal{{A}}_{{\mathrm{{odd}}}} = {data['A_odd']:.4f}$" + "\n" +
                rf"$\eta_{{\mathrm{{asym}}}} = {data['eta_asym']:.3f}$ ({data['eta_asym']*100:.1f}\%)",
                transform=ax.transAxes, fontsize=10.5, weight='bold',
                color='#332288', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#F0F0F0', edgecolor='gray', alpha=0.9))
        
        ax.set_xlim(0, V_MAX_INTEGRAL)
        ax.set_ylim(-1.0, 4.0)
        ax.set_xlabel(r"Velocity magnitude $v_\parallel\ [\mu\mathrm{m/s}]$", fontsize=11)
        ax.set_ylabel(r"Potential Component", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.45)
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8.8, framealpha=0.9)
    
    fig.suptitle(r"Even-Odd Decomposition of Velocity Potential $\tilde{\mathcal{V}}(v_\parallel)$: Parity Breaking to Restoration",
                 fontsize=14.5, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "velocity_potential_even_odd_decomposition_grid.png", dpi=300)
        fig.savefig(d / "velocity_potential_even_odd_decomposition_grid.svg")
    plt.close(fig)
    print("Saved Figure: velocity_potential_even_odd_decomposition_grid.png / .svg")


def plot_odd_integral_and_ratio_vs_rc_xi(df_summary):
    """
    奇成分大域ノルム A_odd および無次元化非対称寄与率 eta_asym の
    スケーリングプロット (vs R_c / xi)。
    """
    df_plot = df_summary.sort_values(by='rc_over_xi').copy()
    x = df_plot['rc_over_xi']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    
    # --- (A) 大域的奇成分ノルム A_odd ---
    ax1.plot(x, df_plot['A_odd_integral'], 'o-', color='#CC6677',
             linewidth=2.4, markersize=8.5, label=r"Odd norm $\mathcal{A}_{\mathrm{odd}} = \int | \tilde{\mathcal{V}}_{\mathrm{odd}} | dv$")
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax1.axvline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax1.set_ylabel(r"Odd-Component Norm $\mathcal{A}_{\mathrm{odd}}\ [\mu\mathrm{m/s}]$", fontsize=12)
    ax1.set_title(r"(A) Global Odd-Component Norm $\mathcal{A}_{\mathrm{odd}}$ vs $R_c/\xi$", fontsize=12.5, weight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.45)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    for _, row in df_plot.iterrows():
        ax1.annotate(f"{row['diameter_um']:.2g} $\mu\mathrm{{m}}$\n({row['A_odd_integral']:.3f})",
                     xy=(row['rc_over_xi'], row['A_odd_integral']),
                     xytext=(0, 9), textcoords='offset points',
                     ha='center', fontsize=9.5, color='#CC6677', weight='bold')
    
    # --- (B) 無次元化非対称寄与率 eta_asym ---
    ax2.plot(x, df_plot['eta_asym_ratio'] * 100, 's-', color='#332288',
             linewidth=2.4, markersize=8.5, label=r"Asymmetry contribution $\eta_{\mathrm{asym}} = \mathcal{A}_{\mathrm{odd}} / \mathcal{I}_{\mathrm{even}}$")
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax2.axvline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax2.set_ylabel(r"Asymmetry Contribution Ratio $\eta_{\mathrm{asym}}\ [\%]$", fontsize=12)
    ax2.set_title(r"(B) Dimensionless Asymmetry Ratio $\eta_{\mathrm{asym}}$ vs $R_c/\xi$", fontsize=12.5, weight='bold')
    ax2.set_ylim(-2, 35)
    ax2.grid(True, which='both', linestyle=':', alpha=0.45)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    for _, row in df_plot.iterrows():
        ax2.annotate(f"{row['diameter_um']:.2g} $\mu\mathrm{{m}}$\n({row['eta_asym_ratio']*100:.1f}%)",
                     xy=(row['rc_over_xi'], row['eta_asym_ratio'] * 100),
                     xytext=(0, 9), textcoords='offset points',
                     ha='center', fontsize=9.5, color='#332288', weight='bold')
    
    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "velocity_potential_even_odd_asymmetry_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "velocity_potential_even_odd_asymmetry_vs_rc_xi.svg")
    plt.close(fig)
    print("Saved Figure: velocity_potential_even_odd_asymmetry_vs_rc_xi.png / .svg")


def plot_2d_contour_grid(results):
    """2D 等高線図グリッドの作成"""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), sharex=True, sharey=True)
    axes = axes.flatten()
    
    contour_levels = np.linspace(0.0, 5.0, 11)
    last_cs = None
    
    for idx, info in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = info['name']
        data = results[b_name]
        
        V_PAR = data['V_PAR']
        V_PERP = data['V_PERP']
        V_eff = data['V_eff']
        rc_xi = data['rc_over_xi']
        
        cf = ax.contourf(V_PAR, V_PERP, V_eff, levels=contour_levels,
                         cmap='viridis_r', extend='max')
        cs = ax.contour(V_PAR, V_PERP, V_eff, levels=contour_levels,
                        colors='white', linewidths=0.7, alpha=0.6)
        last_cs = cf
        
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
        
        ax.plot(data['min_v_par'], data['min_v_perp'], marker='+', markersize=10,
                markeredgewidth=2.2, color='red', label='Potential Min')
        
        ax.set_title(rf"{info['label']}  ($R_c/\xi \approx {rc_xi:.2f}$)",
                     fontsize=13, weight='bold', pad=8)
        
        # 偶奇分解指標と非対称度をバッジで表示
        ax.text(0.05, 0.92,
                rf"$\mathcal{{A}}_{{\mathrm{{odd}}}} = {data['A_odd']:.3f}$" + "\n" +
                rf"$\eta_{{\mathrm{{asym}}}} = {data['eta_asym']*100:.1f}\%$" + "\n" +
                rf"$\mathrm{{Skew}}(v_\parallel) = {data['skew_par']:+.2f}$",
                transform=ax.transAxes, fontsize=9.5, weight='bold',
                color='white', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
        
        ax.set_xlim(-0.22, 0.22)
        ax.set_ylim(-0.22, 0.22)
        ax.set_aspect('equal')
        ax.set_xlabel(r"Longitudinal velocity $v_\parallel\ [\mu\mathrm{m/s}]$", fontsize=11)
        ax.set_ylabel(r"Transverse velocity $v_\perp\ [\mu\mathrm{m/s}]$", fontsize=11)
        ax.grid(False)
    
    # カラーバー
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_cs, cax=cbar_ax)
    cbar.set_label(r"Effective Potential $\tilde{\mathcal{V}}_{\mathrm{eff}}(v_\parallel, v_\perp) = -\ln(P/P_{\max})$",
                   fontsize=12, weight='bold')
    cbar.set_ticks(contour_levels[::2])
    
    fig.suptitle(r"Topological Evolution of 2D Velocity Potential $\tilde{\mathcal{V}}_{\mathrm{eff}}(v_\parallel, v_\perp)$ Across Cargo Sizes",
                 fontsize=15, weight='bold', y=0.97)
    
    plt.subplots_adjust(right=0.90, top=0.91, bottom=0.08, left=0.07, hspace=0.25, wspace=0.22)
    
    for d in OUT_DIRS:
        fig.savefig(d / "velocity_potential_2d_contour_grid.png", dpi=300)
        fig.savefig(d / "velocity_potential_2d_contour_grid.svg")
    plt.close(fig)
    print("Saved Figure: velocity_potential_2d_contour_grid.png / .svg")


def plot_1d_potential_slices(results):
    """ポテンシャルの断面プロット (v_perp = 0 および v_parallel = 0)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    
    for info in BEADS_INFO:
        b_name = info['name']
        data = results[b_name]
        grid_1d = data['grid_1d']
        V_eff = data['V_eff']
        center_idx = len(grid_1d) // 2
        
        slice_par = V_eff[center_idx, :]
        slice_perp = V_eff[:, center_idx]
        label = rf"{info['label']} ($R_c/\xi \approx {data['rc_over_xi']:.2f}$)"
        
        ax1.plot(grid_1d, slice_par, label=label, color=info['color'], linewidth=2.2)
        ax2.plot(grid_1d, slice_perp, label=label, color=info['color'], linewidth=2.2)
    
    ax1.axvline(0, color='gray', linestyle=':', linewidth=1.2)
    ax1.set_xlabel(r"Longitudinal velocity $v_\parallel\ [\mu\mathrm{m/s}]$", fontsize=12)
    ax1.set_ylabel(r"Effective Potential $\tilde{\mathcal{V}}_{\mathrm{eff}}(v_\parallel, v_\perp=0)$", fontsize=12)
    ax1.set_title("(A) Longitudinal Slice $v_\perp = 0$ (Parity Breaking & Restoration)", fontsize=12, weight='bold')
    ax1.set_xlim(-0.20, 0.20)
    ax1.set_ylim(-0.2, 5.0)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper center', fontsize=9.5, framealpha=0.85)
    
    ax2.axvline(0, color='gray', linestyle=':', linewidth=1.2)
    ax2.set_xlabel(r"Transverse velocity $v_\perp\ [\mu\mathrm{m/s}]$", fontsize=12)
    ax2.set_ylabel(r"Effective Potential $\tilde{\mathcal{V}}_{\mathrm{eff}}(v_\parallel=0, v_\perp)$", fontsize=12)
    ax2.set_title("(B) Transverse Slice $v_\parallel = 0$ (Robust Symmetric Well)", fontsize=12, weight='bold')
    ax2.set_xlim(-0.20, 0.20)
    ax2.set_ylim(-0.2, 5.0)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper center', fontsize=9.5, framealpha=0.85)
    
    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "velocity_potential_1d_slices.png", dpi=300)
        fig.savefig(d / "velocity_potential_1d_slices.svg")
    plt.close(fig)
    print("Saved Figure: velocity_potential_1d_slices.png / .svg")


def main():
    print("=== Velocity Space Effective Potential & Even-Odd Decomposition Analysis ===")
    df = load_and_decompose()
    results, df_summary = compute_effective_potentials(df)
    
    # CSV 保存
    csv_filename = "velocity_potential_2d_summary.csv"
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(d / csv_filename, index=False)
    print(f"Saved summary CSV to {OUT_DIRS[0] / csv_filename}")
    
    # プロット作成
    plot_2d_contour_grid(results)
    plot_1d_potential_slices(results)
    plot_even_odd_decomposition_grid(results)
    plot_odd_integral_and_ratio_vs_rc_xi(df_summary)
    
    print("\n--- Even-Odd Symmetry Decomposition Summary Table ---")
    cols_to_print = ['bead_name', 'diameter_um', 'rc_over_xi', 'A_odd_integral', 'I_even_integral', 'eta_asym_ratio', 'delta_v_asym_at_01um_s', 'skewness_v_parallel']
    print(df_summary[cols_to_print].to_string(index=False))
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
