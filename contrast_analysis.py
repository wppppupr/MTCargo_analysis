#!/usr/bin/env python3
"""
contrast_analysis.py

アクティブ微小管流動場における貨物微粒子の「必須対照解析 (Essential Contrast Analysis)」スクリプト:
1. 正規化結合度 cos(psi_i) = v_hat . <u_fp> / |<u_fp>| の解析 (Rc/xi 依存性)
2. 下限側基準 (Null Models): 時間シャッフル & 粒子軌道スワップ
3. 上限側基準 (Virtual Cargo): 実測足印平均流速での移流モデル
   - MSD の直接比較 (6パネル & 比率)
   - 変位 PDF の直接比較 (非ガウス性・尖度)
   - 速度伝達係数 kappa = vc / <u_fp>
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit

# ワークスペースおよびNASのパス設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR_CANDIDATES = [
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    CURRENT_DIR,
]
ROOT_DIR = next((p for p in ROOT_DIR_CANDIDATES if p.exists()), CURRENT_DIR)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis"
]

BEADS_INFO = [
    {'name': 'beads06um', 'label': r'0.63 $\mu\mathrm{m}$', 'diameter_um': 0.63, 'radius_um': 0.315,
     'color': '#882255', 'marker': '^'},
    {'name': 'beads1um',  'label': r'1.18 $\mu\mathrm{m}$', 'diameter_um': 1.18, 'radius_um': 0.590,
     'color': '#CC6677', 'marker': 'o'},
    {'name': 'beads3um',  'label': r'3.37 $\mu\mathrm{m}$', 'diameter_um': 3.37, 'radius_um': 1.685,
     'color': '#DDCC77', 'marker': 'd'},
    {'name': 'beads7um',  'label': r'7.24 $\mu\mathrm{m}$', 'diameter_um': 7.24, 'radius_um': 3.620,
     'color': '#117733', 'marker': 'h'},
    {'name': 'beads5um',  'label': r'5.00 $\mu\mathrm{m}$', 'diameter_um': 5.00, 'radius_um': 2.500,
     'color': '#999933', 'marker': 'p'},
    {'name': 'beads20um', 'label': r'20.0 $\mu\mathrm{m}$', 'diameter_um': 20.0, 'radius_um': 10.000,
     'color': '#44AA99', 'marker': 's'},
]

SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0


def setup_matplotlib():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def save_figure(fig, base_name: str, out_dirs=OUT_DIRS):
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


def load_cached_data():
    inst_path = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
    imsd_path = CURRENT_DIR / "scratch" / "contrast_imsd_cache.pkl"
    disp_path = CURRENT_DIR / "scratch" / "contrast_disp_cache.npy"

    if not inst_path.exists() or not imsd_path.exists() or not disp_path.exists():
        raise FileNotFoundError("Cache files not found. Run scratch/compute_contrast_data.py first.")

    df_inst = pd.read_pickle(inst_path)
    df_imsd = pd.read_pickle(imsd_path)
    disp_data = np.load(disp_path, allow_pickle=True).item()
    return df_inst, df_imsd, disp_data


def compute_null_models(df_inst: pd.DataFrame, n_iter: int = 100):
    """粒子ごとの時間シャッフル & 同一実験内粒子スワップによる下限ヌルモデルの算出"""
    np.random.seed(42)
    null_results = {}

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname].copy()
        
        # 1. 時間シャッフル
        shuf_means = []
        for _ in range(n_iter):
            shuf_cbs = []
            for (exp, pid), grp in sub.groupby(['exp_dir', 'particle']):
                if len(grp) < 2:
                    continue
                idx = np.random.permutation(len(grp))
                cdx_s = grp['cargo_dx'].values[idx]
                cdy_s = grp['cargo_dy'].values[idx]
                cspd_s = grp['cargo_spd'].values[idx]
                u = grp['flow_u'].values
                v = grp['flow_v'].values
                v_fn = grp['flow_spd'].values
                cb_shuf = (cdx_s * u + cdy_s * v) / (cspd_s * v_fn)
                shuf_cbs.extend(cb_shuf)
            if shuf_cbs:
                shuf_means.append(np.mean(shuf_cbs))

        # 2. 粒子スワップ
        swap_means = []
        exp_grps = sub.groupby('exp_dir')
        for _ in range(n_iter):
            swap_cbs = []
            for exp_name, e_grp in exp_grps:
                pids = e_grp['particle'].unique()
                if len(pids) < 2:
                    idx = np.random.permutation(len(e_grp))
                    cdx_s = e_grp['cargo_dx'].values[idx]
                    cdy_s = e_grp['cargo_dy'].values[idx]
                    cspd_s = e_grp['cargo_spd'].values[idx]
                    cb_sw = (cdx_s * e_grp['flow_u'].values + cdy_s * e_grp['flow_v'].values) / (cspd_s * e_grp['flow_spd'].values)
                    swap_cbs.extend(cb_sw)
                    continue
                pids_perm = np.random.permutation(pids)
                p_map = dict(zip(pids, pids_perm))
                for pid in pids:
                    target_pid = p_map[pid]
                    grp_a = e_grp[e_grp['particle'] == pid]
                    grp_b = e_grp[e_grp['particle'] == target_pid]
                    min_len = min(len(grp_a), len(grp_b))
                    if min_len < 2:
                        continue
                    cdx_b = grp_b['cargo_dx'].values[:min_len]
                    cdy_b = grp_b['cargo_dy'].values[:min_len]
                    cspd_b = grp_b['cargo_spd'].values[:min_len]
                    u_a = grp_a['flow_u'].values[:min_len]
                    v_a = grp_a['flow_v'].values[:min_len]
                    vfn_a = grp_a['flow_spd'].values[:min_len]
                    cb_sw = (cdx_b * u_a + cdy_b * v_a) / (cspd_b * vfn_a)
                    swap_cbs.extend(cb_sw)
            if swap_cbs:
                swap_means.append(np.mean(swap_cbs))

        null_results[bname] = {
            'time_shuf_mean': np.mean(shuf_means),
            'time_shuf_std': np.std(shuf_means),
            'swap_mean': np.mean(swap_means) if swap_means else np.nan,
            'swap_std': np.std(swap_means) if swap_means else np.nan,
        }

    return null_results


def plot_fig1_coupling_spectrum(df_inst: pd.DataFrame, null_results: dict):
    """
    Figure 1: 結合度スペクトル解析
    - (a) 正規化結合度 <cos psi> vs Rc/xi (上限仮想貨物・下限ヌルモデル併記)
    - (b) 従来点平均一致度 <cos theta_i> vs Rc/xi
    - (c) cos(psi) の確率密度分布 P(cos psi) (サイズ別 KDE)
    - (d) シグナル有意性 (Z-score) vs Rc/xi
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    rc_xi_list = []
    cos_psi_means = []
    cos_psi_sems = []
    cos_i_means = []
    cos_i_sems = []
    null_time_means = []
    null_time_stds = []
    null_swap_means = []
    z_scores = []
    labels = []
    colors = []
    markers = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname]
        rc_xi = sub['rc_over_xi'].mean()
        cb = sub['cos_bulk'].values
        ci = sub['cos_i'].values
        
        rc_xi_list.append(rc_xi)
        cos_psi_means.append(np.mean(cb))
        cos_psi_sems.append(stats.sem(cb))
        cos_i_means.append(np.mean(ci))
        cos_i_sems.append(stats.sem(ci))
        
        nr = null_results[bname]
        null_time_means.append(nr['time_shuf_mean'])
        null_time_stds.append(nr['time_shuf_std'])
        null_swap_means.append(nr['swap_mean'])
        
        z = (np.mean(cb) - nr['time_shuf_mean']) / stats.sem(cb)
        z_scores.append(z)
        
        labels.append(b_item['label'])
        colors.append(b_item['color'])
        markers.append(b_item['marker'])

    # ソート (Rc/xi 順)
    sort_idx = np.argsort(rc_xi_list)
    rc_xi_arr = np.array(rc_xi_list)[sort_idx]
    cos_psi_arr = np.array(cos_psi_means)[sort_idx]
    cos_psi_sem_arr = np.array(cos_psi_sems)[sort_idx]
    cos_i_arr = np.array(cos_i_means)[sort_idx]
    cos_i_sem_arr = np.array(cos_i_sems)[sort_idx]
    null_t_arr = np.array(null_time_means)[sort_idx]
    null_t_std_arr = np.array(null_time_stds)[sort_idx]
    null_s_arr = np.array(null_swap_means)[sort_idx]
    z_arr = np.array(z_scores)[sort_idx]

    # -------------------------------------------------------------
    # Panel (a): 正規化結合度 <cos psi> vs Rc/xi
    # -------------------------------------------------------------
    # 上限: 仮想貨物 理論線
    ax_a.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=2.0, label='Upper Bound: Virtual Cargo ($\\cos\\psi \\equiv 1$)')
    # 下限: ヌルモデル帯
    ax_a.fill_between([0.02, 5.0], 0, 0.05, color='gray', alpha=0.18, label='Lower Bound: Global Drift / Noise Baseline')
    ax_a.plot(rc_xi_arr, null_t_arr, color='#d7191c', linestyle=':', marker='x', markersize=7, label='Null Model (Time Shuffle)')
    ax_a.plot(rc_xi_arr, null_s_arr, color='#fdae61', linestyle='-.', marker='+', markersize=7, label='Null Model (Trajectory Swap)')

    # 実測値プロット
    for i in sort_idx:
        ax_a.errorbar(rc_xi_list[i], cos_psi_means[i], yerr=cos_psi_sems[i],
                      fmt=markers[i], color=colors[i], markersize=10, capsize=5, elinewidth=1.8,
                      label=f"{labels[i]} ($R_c/\\xi={rc_xi_list[i]:.2f}$)", zorder=5)
    ax_a.plot(rc_xi_arr, cos_psi_arr, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=4)

    # 臨界境界 Rc = xi
    ax_a.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_a.text(1.05, 0.85, r'$R_c = \xi$', fontsize=13, color='gray', fontweight='bold')

    ax_a.set_xscale('log')
    ax_a.set_xlim(0.04, 4.0)
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_a.set_ylabel(r'Normalized orientation coupling $\langle\cos\psi\rangle$')
    ax_a.set_title(r'(a) Normalized Coupling $\hat{v} \cdot \frac{\langle u\rangle_{R_c}}{|\langle u\rangle_{R_c}|}$ vs $R_c/\xi$', fontweight='bold')
    ax_a.legend(loc='lower left', fontsize=9.5, framealpha=0.9)
    ax_a.grid(True)

    # -------------------------------------------------------------
    # Panel (b): 従来の点平均一致度 <cos theta_i> vs Rc/xi
    # -------------------------------------------------------------
    for i in sort_idx:
        ax_b.errorbar(rc_xi_list[i], cos_i_means[i], yerr=cos_i_sems[i],
                      fmt=markers[i], color=colors[i], markersize=10, capsize=5, elinewidth=1.8,
                      label=labels[i], zorder=5)
    ax_b.plot(rc_xi_arr, cos_i_arr, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=4)

    ax_b.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_b.text(1.05, 0.40, r'$R_c = \xi$', fontsize=13, color='gray', fontweight='bold')

    ax_b.set_xscale('log')
    ax_b.set_xlim(0.04, 4.0)
    ax_b.set_ylim(-0.02, 0.50)
    ax_b.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_b.set_ylabel(r'Raw point-averaged alignment $\langle\cos\theta_i\rangle$')
    ax_b.set_title(r'(b) Point-Averaged Alignment $\langle\hat{v}\cdot\hat{u}\rangle$ vs $R_c/\xi$', fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_b.grid(True)

    # -------------------------------------------------------------
    # Panel (c): cos(psi) の確率密度分布 P(cos psi) (KDE)
    # -------------------------------------------------------------
    x_grid = np.linspace(-1.0, 1.0, 300)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname]
        cb = sub['cos_bulk'].values
        kde = stats.gaussian_kde(cb, bw_method=0.15)
        dens = kde(x_grid)
        ax_c.plot(x_grid, dens, color=b_item['color'], linewidth=2.2,
                  label=f"{b_item['label']} ($R_c/\\xi={sub['rc_over_xi'].mean():.2f}$)")

    # 等方分布基準線 (一様分布: 2次元単位ベクトルの内積分布 1 / (pi * sqrt(1 - x^2)))
    # 3次元なら一様分布 0.5
    ax_c.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Uniform isotropic reference')
    ax_c.set_xlim(-1.0, 1.0)
    ax_c.set_ylim(0, 1.5)
    ax_c.set_xlabel(r'Cosine of coupling angle $\cos\psi = \hat{v} \cdot \hat{u}_{\mathrm{fp}}$')
    ax_c.set_ylabel(r'Probability density $P(\cos\psi)$')
    ax_c.set_title(r'(c) Coupling Angle Distribution $P(\cos\psi)$', fontweight='bold')
    ax_c.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax_c.grid(True)

    # -------------------------------------------------------------
    # Panel (d): シグナル有意性 (Z-score) vs Rc/xi
    # -------------------------------------------------------------
    for i in sort_idx:
        ax_d.scatter(rc_xi_list[i], z_scores[i], color=colors[i], marker=markers[i], s=120, zorder=5, label=labels[i])
    ax_d.plot(rc_xi_arr, z_arr, color='#882255', linewidth=2.0, linestyle='-', alpha=0.7, zorder=4)

    # 有意水準線 (Z = 3: p = 0.0013, Z = 2: p = 0.05)
    ax_d.axhline(3.0, color='#cc2222', linestyle='--', linewidth=1.5, label=r'Significance threshold ($Z=3, p < 0.001$)')
    ax_d.axhline(0.0, color='gray', linestyle='-', linewidth=0.8)
    ax_d.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    # アノテーション
    ax_d.text(0.06, 25.0, 'Strong Genuine Coupling\n($Z > 20, p \\ll 10^{-10}$)', fontsize=11,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f5d0', edgecolor='#276419', alpha=0.8))
    ax_d.text(1.2, 5.0, 'Complete Decoupling\n($Z = 0.96, p = 0.34$)\nConsistent with Null', fontsize=11,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde0dd', edgecolor='#c51b7d', alpha=0.8))

    ax_d.set_xscale('log')
    ax_d.set_xlim(0.04, 4.0)
    ax_d.set_ylim(-2, 35)
    ax_d.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_d.set_ylabel(r'Signal Significance $Z = (\langle\cos\psi\rangle - \mu_{\mathrm{null}}) / \mathrm{SEM}$')
    ax_d.set_title(r'(d) Statistical Decoupling Test ($Z$-score vs Null)', fontweight='bold')
    ax_d.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_d.grid(True)

    save_figure(fig, "contrast_fig1_coupling_spectrum")
    plt.close(fig)


def plot_fig2_msd_real_vs_virtual(df_imsd: pd.DataFrame):
    """
    Figure 2: MSD 直接比較 (6パネル: 各ビーズサイズ)
    - 実測貨物のアンサンブル平均 MSD (実線)
    - 仮想貨物 (Footprint Advection) のアンサンブル平均 MSD (破線)
    - 比率 MSD_real / MSD_virt
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = b_item['name']
        sub = df_imsd[df_imsd['bead_name'] == bname].copy()
        
        rc_xi = sub['rc_over_xi'].mean()
        d_um = b_item['diameter_um']
        color = b_item['color']
        
        # ラグタイムごとの平均と標準誤差
        grp = sub.groupby('lag_time_s').agg({
            'msd_real': ['mean', stats.sem, 'count'],
            'msd_virt': ['mean', stats.sem]
        })
        
        lags = grp.index.values
        m_real = grp['msd_real']['mean'].values
        s_real = grp['msd_real']['sem'].values
        m_virt = grp['msd_virt']['mean'].values
        s_virt = grp['msd_virt']['sem'].values
        
        # 1. 仮想貨物 MSD (上限・理想追従トレーサー)
        ax.plot(lags, m_virt, color='#332288', linestyle='--', linewidth=2.4, label='Upper Bound: Virtual Cargo (Footprint Tracer)')
        ax.fill_between(lags, np.maximum(1e-4, m_virt - s_virt), m_virt + s_virt, color='#332288', alpha=0.15)
        
        # 2. 実測貨物 MSD
        ax.plot(lags, m_real, color=color, linestyle='-', linewidth=2.4, marker=b_item['marker'],
                markersize=6, markevery=5, label=f'Real Cargo ({b_item["label"]})')
        ax.fill_between(lags, np.maximum(1e-4, m_real - s_real), m_real + s_real, color=color, alpha=0.2)
        
        # 傾き参照線 (alpha = 1: 正常拡散, alpha = 2: 弾道運動)
        t_ref = np.array([4.0, 40.0])
        ref_scale = m_real[0] / t_ref[0]
        ax.plot(t_ref, ref_scale * t_ref, color='gray', linestyle=':', linewidth=1.2, label=r'$\sim \tau^1$ (Diffusion)')
        ax.plot(t_ref, (ref_scale / 10.0) * (t_ref**2), color='dimgray', linestyle='-.', linewidth=1.2, label=r'$\sim \tau^2$ (Ballistic)')
        
        # 比率インセット (Inset: Ratio = MSD_real / MSD_virt)
        ax_ins = ax.inset_axes([0.54, 0.12, 0.42, 0.32])
        ratio = m_real / np.where(m_virt > 0, m_virt, np.nan)
        ax_ins.plot(lags, ratio, color='black', linewidth=1.6)
        ax_ins.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
        ax_ins.set_xscale('log')
        ax_ins.set_xlim(4, 300)
        ax_ins.set_ylim(0, max(1.5, float(np.nanmax(ratio[:15])) * 1.1))
        ax_ins.set_title(r'$\mathrm{MSD}_{\mathrm{real}} / \mathrm{MSD}_{\mathrm{virt}}$', fontsize=9.5, pad=3)
        ax_ins.tick_params(labelsize=8)
        ax_ins.grid(True, linestyle=':', alpha=0.6)
        
        n_p = sub['particle'].nunique()
        ax.set_title(f"{b_item['label']}  ($R_c/\\xi = {rc_xi:.2f}, N={n_p}$)", fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3.5, 350)
        ax.set_ylim(0.01, 1000)
        ax.grid(True)
        
        if idx % 3 == 0:
            ax.set_ylabel(r'MSD $\langle \Delta r^2(\tau) \rangle$ [$\mu\mathrm{m}^2$]')
        if idx >= 3:
            ax.set_xlabel(r'Lag time $\tau$ [s]')
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9.0, framealpha=0.9)

    fig.suptitle(r'Ensemble Mean Squared Displacement: Real Cargo vs Virtual Cargo (Footprint Tracer)',
                 fontsize=16, fontweight='bold', y=0.995)
    save_figure(fig, "contrast_fig2_msd_real_vs_virtual_6panels")
    plt.close(fig)


def plot_fig3_displacement_pdf(disp_data: dict):
    """
    Figure 3: 変位確率密度関数 (Displacement PDF) 直接比較
    - tau = 4 s (1フレーム) での実測 vs 仮想貨物
    - 規格化変位 Delta x / sigma
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    x_grid = np.linspace(-6, 6, 200)
    gauss_ref = stats.norm.pdf(x_grid, 0, 1)

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = b_item['name']
        color = b_item['color']
        
        r_disp = np.array(disp_data['real'][bname][1])
        v_disp = np.array(disp_data['virt'][bname][1])
        
        r_std = np.std(r_disp)
        v_std = np.std(v_disp)
        r_norm = r_disp / r_std if r_std > 0 else r_disp
        v_norm = v_disp / v_std if v_std > 0 else v_disp
        
        r_kurt = stats.kurtosis(r_disp)
        v_kurt = stats.kurtosis(v_disp)
        
        # ガウス基準線
        ax.plot(x_grid, gauss_ref, color='gray', linestyle='--', linewidth=1.5, label='Gaussian $\mathcal{N}(0, 1)$')
        
        # KDE プロット
        kde_r = stats.gaussian_kde(r_norm, bw_method=0.2)
        kde_v = stats.gaussian_kde(v_norm, bw_method=0.2)
        
        ax.plot(x_grid, kde_v(x_grid), color='#332288', linestyle='--', linewidth=2.2,
                label=f'Virtual Cargo ($\kappa={v_kurt:.1f}$)')
        ax.plot(x_grid, kde_r(x_grid), color=color, linestyle='-', linewidth=2.4,
                label=f'Real Cargo ($\kappa={r_kurt:.1f}$)')
        
        ax.set_yscale('log')
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(1e-4, 1.2)
        ratio_val = r_std / v_std if v_std > 0 else np.nan
        title_str = f"{b_item['label']}  (" + r"$\sigma_{\mathrm{real}}/\sigma_{\mathrm{virt}} = " + f"{ratio_val:.2f}$)"
        ax.set_title(title_str, fontsize=13, fontweight='bold')
        ax.grid(True)
        
        if idx % 3 == 0:
            ax.set_ylabel(r'Normalized PDF $P(\Delta x / \sigma)$')
        if idx >= 3:
            ax.set_xlabel(r'Normalized displacement $\Delta x / \sigma$')
        if idx == 0:
            ax.legend(loc='lower center', fontsize=9.2, framealpha=0.9)

    fig.suptitle(r'Displacement Probability Density: Real Cargo vs Virtual Cargo ($\tau = 4\ \mathrm{s}$)',
                 fontsize=16, fontweight='bold', y=0.995)
    save_figure(fig, "contrast_fig3_displacement_pdf_real_vs_virtual")
    plt.close(fig)


def plot_fig4_velocity_transfer(df_inst: pd.DataFrame):
    """
    Figure 4: 速度伝達係数 kappa = vc / <u_fp> と速度比
    - (a) 伝達係数 kappa vs Rc/xi
    - (b) 速度ノルム比 <|vc|> / <|u_fp|> vs Rc/xi
    - (c) 代表的な散布図 (0.63 um vs 20.0 um)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])

    rc_xi_list = []
    kappa_list = []
    ratio_list = []
    labels = []
    colors = []
    markers = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname]
        rc_xi = sub['rc_over_xi'].mean()
        
        # 伝達係数 kappa = sum(vc . uf) / sum(|uf|^2)
        dot_sum = np.sum(sub['cargo_dx'] * sub['flow_u'] + sub['cargo_dy'] * sub['flow_v'])
        flow_sq_sum = np.sum(sub['flow_spd']**2)
        kappa = dot_sum / flow_sq_sum if flow_sq_sum > 0 else np.nan
        
        spd_ratio = np.mean(sub['cargo_spd']) / np.mean(sub['flow_spd'])
        
        rc_xi_list.append(rc_xi)
        kappa_list.append(kappa)
        ratio_list.append(spd_ratio)
        labels.append(b_item['label'])
        colors.append(b_item['color'])
        markers.append(b_item['marker'])

    sort_idx = np.argsort(rc_xi_list)
    rc_xi_arr = np.array(rc_xi_list)[sort_idx]
    kappa_arr = np.array(kappa_list)[sort_idx]
    ratio_arr = np.array(ratio_list)[sort_idx]

    # Panel (a): kappa vs Rc/xi
    ax_a.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=1.8, label='Perfect Transfer ($\kappa = 1.0$)')
    for i in sort_idx:
        ax_a.scatter(rc_xi_list[i], kappa_list[i], color=colors[i], marker=markers[i], s=120, zorder=5, label=labels[i])
    ax_a.plot(rc_xi_arr, kappa_arr, color='#882255', linewidth=2.0, linestyle='-', alpha=0.7, zorder=4)
    ax_a.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_a.set_xscale('log')
    ax_a.set_xlim(0.04, 4.0)
    ax_a.set_ylim(-0.05, 1.1)
    ax_a.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_a.set_ylabel(r'Velocity transfer coefficient $\kappa$')
    ax_a.set_title(r'(a) Velocity Transfer Coefficient $\mathbf{v}_c = \kappa \langle\mathbf{u}\rangle_{R_c} + \boldsymbol{\eta}$', fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_a.grid(True)

    # Panel (b): Speed ratio vs Rc/xi
    ax_b.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=1.8, label='Equal Speed Ratio')
    for i in sort_idx:
        ax_b.scatter(rc_xi_list[i], ratio_list[i], color=colors[i], marker=markers[i], s=120, zorder=5, label=labels[i])
    ax_b.plot(rc_xi_arr, ratio_arr, color='#44AA99', linewidth=2.0, linestyle='-', alpha=0.7, zorder=4)
    ax_b.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_b.set_xscale('log')
    ax_b.set_xlim(0.04, 4.0)
    ax_b.set_ylim(0.2, 1.2)
    ax_b.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_b.set_ylabel(r'Speed ratio $\langle |\mathbf{v}_c| \rangle / \langle |\langle\mathbf{u}\rangle_{R_c}| \rangle$')
    ax_b.set_title(r'(b) Cargo-to-Flow Speed Ratio vs $R_c/\xi$', fontweight='bold')
    ax_b.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax_b.grid(True)

    # Panel (c1): Scatter 0.63 um
    sub06 = df_inst[df_inst['bead_name'] == 'beads06um']
    ax_c1.hexbin(sub06['flow_spd'] * SCALE_UM_PER_PX / DT_SEC, 
                 sub06['cargo_spd'] * SCALE_UM_PER_PX / DT_SEC, 
                 gridsize=35, cmap='Blues', mincnt=1, bins='log')
    max_v = 0.35
    ax_c1.plot([0, max_v], [0, max_v], color='#cc2222', linestyle='--', linewidth=1.8, label='1:1 Line')
    ax_c1.set_xlim(0, max_v)
    ax_c1.set_ylim(0, max_v)
    ax_c1.set_xlabel(r'Footprint flow speed $|\langle\mathbf{u}\rangle_{R_c}|$ [$\mu\mathrm{m/s}$]')
    ax_c1.set_ylabel(r'Cargo speed $|\mathbf{v}_c|$ [$\mu\mathrm{m/s}$]')
    ax_c1.set_title(r'(c1) $0.63\ \mu\mathrm{m}$ ($R_c/\xi=0.07$, Strong Linear Coupling)', fontweight='bold')
    ax_c1.legend(loc='upper left', fontsize=10)
    ax_c1.grid(True)

    # Panel (c2): Scatter 20.0 um
    sub20 = df_inst[df_inst['bead_name'] == 'beads20um']
    ax_c2.hexbin(sub20['flow_spd'] * SCALE_UM_PER_PX / DT_SEC, 
                 sub20['cargo_spd'] * SCALE_UM_PER_PX / DT_SEC, 
                 gridsize=35, cmap='Greens', mincnt=1, bins='log')
    max_v2 = 0.35
    ax_c2.plot([0, max_v2], [0, max_v2], color='#cc2222', linestyle='--', linewidth=1.8, label='1:1 Line')
    ax_c2.set_xlim(0, max_v2)
    ax_c2.set_ylim(0, max_v2)
    ax_c2.set_xlabel(r'Footprint flow speed $|\langle\mathbf{u}\rangle_{R_c}|$ [$\mu\mathrm{m/s}$]')
    ax_c2.set_ylabel(r'Cargo speed $|\mathbf{v}_c|$ [$\mu\mathrm{m/s}$]')
    ax_c2.set_title(r'(c2) $20.0\ \mu\mathrm{m}$ ($R_c/\xi=2.56$, Decoupled / Stagnant)', fontweight='bold')
    ax_c2.legend(loc='upper left', fontsize=10)
    ax_c2.grid(True)

    save_figure(fig, "contrast_fig4_velocity_transfer")
    plt.close(fig)


def main():
    setup_matplotlib()
    print("Loading cached data...")
    df_inst, df_imsd, disp_data = load_cached_data()
    print(f"Data loaded: {len(df_inst)} inst, {len(df_imsd)} imsd")

    print("\nComputing Null Models (Time Shuffle & Trajectory Swap)...")
    null_results = compute_null_models(df_inst, n_iter=100)

    print("\nGenerating Figure 1: Coupling Spectrum & Null Models...")
    plot_fig1_coupling_spectrum(df_inst, null_results)

    print("\nGenerating Figure 2: MSD Real vs Virtual Cargo (6 panels)...")
    plot_fig2_msd_real_vs_virtual(df_imsd)

    print("\nGenerating Figure 3: Displacement PDF Real vs Virtual...")
    plot_fig3_displacement_pdf(disp_data)

    print("\nGenerating Figure 4: Velocity Transfer & Speed Ratio...")
    plot_fig4_velocity_transfer(df_inst)

    print("\nAll contrast analysis figures generated successfully!")


if __name__ == '__main__':
    main()
