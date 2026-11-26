#!/usr/bin/env python3
"""
plot_cargo_speed_vs_flow.py

「粒子の速さが足元の正味の流れで決まること」の包括的解析スクリプト:
【粒子個別フィット -> ブートストラップ平均（Bootstrap N=2000）法による再計算版】

1. Figure 7: 6パネル散布図 (Hexbin) + 個別粒子回帰線 + ブートストラップ平均回帰線 & 95% CI + 条件付きビン平均
2. Figure 8: マスター応答プロファイル & 応答感度 alpha (95% CI) と相関係数 r vs Rc/xi
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

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

SCALE = 0.11  # um/px
DT = 4.0      # s/frame
V_UNIT = SCALE / DT  # um/s


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
        'legend.fontsize': 10.0,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def save_fig(fig, base_name):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


def bootstrap_stats(data, n_boot=2000):
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    if n == 1:
        return arr[0], 0.0, arr[0], arr[0]
    
    boot_means = [np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    mean_val = np.mean(arr)
    sem_val = np.std(boot_means)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    return mean_val, sem_val, ci_low, ci_high


def main():
    setup_matplotlib()
    np.random.seed(42)

    df_inst = pd.read_pickle("scratch/contrast_inst_cache.pkl")
    print(f"Loaded {len(df_inst)} instantaneous records.")

    bins = np.linspace(0.0, 0.35, 15)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    particle_fits_by_bead = {}
    bootstrap_summary = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname].copy()
        d_um = b_item['diameter_um']
        rc_xi = sub['rc_over_xi'].mean()
        
        p_records = []
        for (exp, pid), grp in sub.groupby(['exp_dir', 'particle']):
            if len(grp) < 10:
                continue
            vc = grp['cargo_spd'].values * V_UNIT
            uf = grp['flow_spd'].values * V_UNIT
            
            reg = stats.linregress(uf, vc)
            r_val, _ = stats.pearsonr(uf, vc)
            ratio = np.mean(vc) / np.mean(uf) if np.mean(uf) > 0 else np.nan
            
            p_records.append({
                'exp_dir': exp,
                'particle': pid,
                'n_frames': len(grp),
                'slope': reg.slope,
                'intercept': reg.intercept,
                'r_value': r_val,
                'r_squared': reg.rvalue**2,
                'speed_ratio': ratio,
                'mean_vc': np.mean(vc),
                'mean_uf': np.mean(uf),
            })
            
        df_p = pd.DataFrame(p_records)
        particle_fits_by_bead[bname] = df_p
        
        sl_m, sl_sem, sl_l, sl_h = bootstrap_stats(df_p['slope'])
        in_m, in_sem, in_l, in_h = bootstrap_stats(df_p['intercept'])
        r_m, r_sem, r_l, r_h = bootstrap_stats(df_p['r_value'])
        rat_m, rat_sem, rat_l, rat_h = bootstrap_stats(df_p['speed_ratio'])
        
        bootstrap_summary.append({
            'bead_name': bname,
            'label': b_item['label'],
            'diameter_um': d_um,
            'rc_over_xi': rc_xi,
            'n_particles': len(df_p),
            'slope_mean': sl_m,
            'slope_sem': sl_sem,
            'slope_ci': (sl_l, sl_h),
            'intercept_mean': in_m,
            'intercept_sem': in_sem,
            'intercept_ci': (in_l, in_h),
            'r_mean': r_m,
            'r_sem': r_sem,
            'r_ci': (r_l, r_h),
            'ratio_mean': rat_m,
            'ratio_sem': rat_sem,
            'ratio_ci': (rat_l, rat_h),
            'slopes_all': df_p['slope'].values,
            'r_all': df_p['r_value'].values,
        })

    # -------------------------------------------------------------
    # Figure 7: 6パネル散布図 + 個別粒子回帰 + ブートストラップ平均線
    # -------------------------------------------------------------
    fig_6p, axes_6p = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes_6p = axes_6p.flatten()

    master_binned = {}

    for idx, (b_item, b_stat) in enumerate(zip(BEADS_INFO, bootstrap_summary)):
        ax = axes_6p[idx]
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname].copy()
        df_p = particle_fits_by_bead[bname]
        
        rc_xi = b_stat['rc_over_xi']
        color = b_item['color']
        
        vc = sub['cargo_spd'].values * V_UNIT
        uf = sub['flow_spd'].values * V_UNIT
        
        # 1. 2D 密度散布図 (Hexbin)
        ax.hexbin(uf, vc, gridsize=32, cmap='Purples', mincnt=1, bins='log', alpha=0.55)
        
        # 2. 個別粒子の回帰線 (薄い線)
        u_grid = np.linspace(0, 0.35, 100)
        for _, p_row in df_p.iterrows():
            ax.plot(u_grid, p_row['slope'] * u_grid + p_row['intercept'],
                    color='gray', alpha=0.35, linewidth=0.9, zorder=3)
            
        # 3. 条件付きビン平均
        bin_idx = np.digitize(uf, bins) - 1
        b_means, b_sems, b_cens = [], [], []
        for b_i in range(len(bin_centers)):
            mask = (bin_idx == b_i)
            if np.sum(mask) >= 15:
                b_means.append(np.mean(vc[mask]))
                b_sems.append(stats.sem(vc[mask]))
                b_cens.append(bin_centers[b_i])
                
        b_means = np.array(b_means)
        b_sems = np.array(b_sems)
        b_cens = np.array(b_cens)
        master_binned[bname] = (b_cens, b_means, b_sems)
        
        ax.errorbar(b_cens, b_means, yerr=b_sems, fmt=b_item['marker'], color=color,
                    markersize=8.5, capsize=3.5, elinewidth=1.8, markeredgecolor='black',
                    label=r'Binned Mean $\langle |v_c| \mid |\langle u\rangle| \rangle$', zorder=5)
        ax.plot(b_cens, b_means, color=color, linewidth=2.4, zorder=4)
        
        # 4. ブートストラップ平均回帰線 & 95% CI 帯
        sl_m = b_stat['slope_mean']
        sl_l, sl_h = b_stat['slope_ci']
        in_m = b_stat['intercept_mean']
        in_l, in_h = b_stat['intercept_ci']
        
        v_fit_mean = sl_m * u_grid + in_m
        v_fit_low = sl_l * u_grid + in_l
        v_fit_high = sl_h * u_grid + in_h
        
        ax.plot(u_grid, v_fit_mean, color='#d95f02', linestyle='-', linewidth=2.4,
                label=f'Bootstrap Mean Fit:\n$|v_c| = {sl_m:.2f}|u| + {in_m:.3f}$\n($r={b_stat["r_mean"]:.2f} \pm {b_stat["r_sem"]:.2f}$)', zorder=6)
        ax.fill_between(u_grid, np.maximum(0, v_fit_low), v_fit_high, color='#d95f02', alpha=0.18, zorder=5, label='95% Bootstrap CI')
        
        # 5. 1:1 完全追従線
        ax.plot([0, 0.4], [0, 0.4], color='dimgray', linestyle=':', linewidth=1.4, label='1:1 Line', zorder=2)
        
        n_p = b_stat['n_particles']
        ax.set_title(f"{b_item['label']}  ($R_c/\\xi = {rc_xi:.2f}, N={n_p}$ particles)", fontsize=13, fontweight='bold')
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0, 0.35)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=8.8, framealpha=0.92)
        
        if idx % 3 == 0:
            ax.set_ylabel(r'Cargo Speed $|v_c|$ [$\mu\mathrm{m/s}$]')
        if idx >= 3:
            ax.set_xlabel(r'Footprint Flow Speed $|\langle\mathbf{u}\rangle_{R_c}|$ [$\mu\mathrm{m/s}$]')

    fig_6p.suptitle(r'Cargo Speed vs Flow Speed: Particle-wise Fits & Bootstrap Ensemble Average ($N_{\mathrm{boot}}=2000$)',
                    fontsize=16, fontweight='bold', y=0.995)
    save_fig(fig_6p, "contrast_fig7_speed_vs_flow_6panels")
    plt.close(fig_6p)

    # -------------------------------------------------------------
    # Figure 8: マスター応答プロファイル & 応答感度サマリー (Bootstrap)
    # -------------------------------------------------------------
    fig_sum = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig_sum, wspace=0.28)

    ax_m1 = fig_sum.add_subplot(gs[0, 0])
    ax_m2 = fig_sum.add_subplot(gs[0, 1])

    # Panel 1: マスター条件付き平均速度プロファイル
    ax_m1.plot([0, 0.35], [0, 0.35], color='dimgray', linestyle=':', linewidth=1.5, label='1:1 Perfect Matching', zorder=1)
    for b_stat, b_item in zip(bootstrap_summary, BEADS_INFO):
        bname = b_item['name']
        bcens, bmeans, bsems = master_binned[bname]
        color = b_item['color']
        marker = b_item['marker']
        label = f"{b_item['label']} ($R_c/\\xi={b_stat['rc_over_xi']:.2f}, N={b_stat['n_particles']}$)"
        ax_m1.errorbar(bcens, bmeans, yerr=bsems, fmt=marker, color=color,
                       markersize=8, capsize=3, elinewidth=1.6, markeredgecolor='black', label=label, zorder=4)
        ax_m1.plot(bcens, bmeans, color=color, linewidth=2.2, zorder=3)

    ax_m1.set_xlim(0, 0.28)
    ax_m1.set_ylim(0, 0.22)
    ax_m1.set_xlabel(r'Footprint Flow Speed $|\langle\mathbf{u}\rangle_{R_c}|$ [$\mu\mathrm{m/s}$]')
    ax_m1.set_ylabel(r'Conditional Mean Cargo Speed $\langle |v_c| \rangle$ [$\mu\mathrm{m/s}$]')
    ax_m1.set_title(r'(a) Master Velocity Response Profiles $\langle |v_c| \mid |\langle u\rangle| \rangle$', fontweight='bold')
    ax_m1.legend(loc='upper left', fontsize=9.2, framealpha=0.92)
    ax_m1.grid(True)

    # Panel 2: ブートストラップ応答感度 alpha (95% CI) & 相関係数 r vs Rc/xi
    rc_xi_list = [r['rc_over_xi'] for r in bootstrap_summary]
    slopes = [r['slope_mean'] for r in bootstrap_summary]
    slopes_sem = [r['slope_sem'] for r in bootstrap_summary]
    slopes_ci = [r['slope_ci'] for r in bootstrap_summary]
    r_vals = [r['r_mean'] for r in bootstrap_summary]
    r_sems = [r['r_sem'] for r in bootstrap_summary]

    sort_idx = np.argsort(rc_xi_list)
    rc_arr = np.array(rc_xi_list)[sort_idx]
    slopes_arr = np.array(slopes)[sort_idx]
    slopes_sem_arr = np.array(slopes_sem)[sort_idx]
    r_arr = np.array(r_vals)[sort_idx]
    r_sem_arr = np.array(r_sems)[sort_idx]

    ax_m2_r = ax_m2.twinx()

    # 個別粒子の生スロープを散布 (背景)
    for b_stat, b_item in zip(bootstrap_summary, BEADS_INFO):
        rc_val = b_stat['rc_over_xi']
        for sl_p in b_stat['slopes_all']:
            ax_m2.scatter(rc_val, sl_p, color=b_item['color'], alpha=0.35, s=30, zorder=3)

    # Bootstrap 傾き alpha (左軸)
    eb1 = ax_m2.errorbar(rc_arr, slopes_arr, yerr=slopes_sem_arr, fmt='s-', color='#1f78b4',
                         linewidth=2.2, markersize=8, capsize=4, elinewidth=1.8, zorder=5,
                         label=r'Bootstrap Sensitivity $\alpha = \frac{d|v_c|}{d|u|}$ (Mean $\pm$ SEM)')

    # Bootstrap 相関係数 Pearson r (右軸)
    eb2 = ax_m2_r.errorbar(rc_arr, r_arr, yerr=r_sem_arr, fmt='o--', color='#e31a1c',
                           linewidth=2.2, markersize=8, capsize=4, elinewidth=1.8, zorder=5,
                           label=r'Bootstrap Correlation $r$ (Mean $\pm$ SEM)')

    ax_m2.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax_m2.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_m2.text(1.05, 0.65, r'$R_c = \xi$', fontsize=12, color='gray', fontweight='bold')

    ax_m2.set_xscale('log')
    ax_m2.set_xlim(0.04, 4.0)
    ax_m2.set_ylim(-0.35, 0.95)
    ax_m2_r.set_ylim(-0.45, 0.85)

    ax_m2.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_m2.set_ylabel(r'Velocity Sensitivity $\alpha = d|v_c| / d|\langle u\rangle|$', color='#1f78b4')
    ax_m2_r.set_ylabel(r'Correlation Coefficient $r$', color='#e31a1c')
    ax_m2.tick_params(axis='y', labelcolor='#1f78b4')
    ax_m2_r.tick_params(axis='y', labelcolor='#e31a1c')

    lines = [eb1, eb2]
    labels = [l.get_label() for l in lines]
    ax_m2.legend(lines, labels, loc='lower left', fontsize=9.5, framealpha=0.92)
    ax_m2.set_title(r'(b) Bootstrap Sensitivity & Correlation vs $R_c/\xi$', fontweight='bold')
    ax_m2.grid(True)

    save_fig(fig_sum, "contrast_fig8_speed_response_master_and_sensitivity")
    plt.close(fig_sum)

    df_boot_out = pd.DataFrame([{
        'bead_name': r['bead_name'],
        'diameter_um': r['diameter_um'],
        'rc_over_xi': r['rc_over_xi'],
        'n_particles': r['n_particles'],
        'slope_mean': r['slope_mean'],
        'slope_sem': r['slope_sem'],
        'slope_ci_low': r['slope_ci'][0],
        'slope_ci_high': r['slope_ci'][1],
        'intercept_mean': r['intercept_mean'],
        'intercept_sem': r['intercept_sem'],
        'intercept_ci_low': r['intercept_ci'][0],
        'intercept_ci_high': r['intercept_ci'][1],
        'r_mean': r['r_mean'],
        'r_sem': r['r_sem'],
        'r_ci_low': r['r_ci'][0],
        'r_ci_high': r['r_ci'][1],
        'ratio_mean': r['ratio_mean'],
        'ratio_sem': r['ratio_sem'],
    } for r in bootstrap_summary])
    df_boot_out.to_pickle("scratch/speed_vs_flow_bootstrap_summary.pkl")
    print("\nBootstrap Speed vs Flow Analysis complete!")


if __name__ == '__main__':
    main()
