#!/usr/bin/env python3
import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""
plot_alpha_vs_delta_u_prop.py

横軸に有効推進ポテンシャル井戸深さ Delta U_tilde_prop(R_c)
縦軸に無次元化 MSD (t_tilde >= 10) から得られた異常拡散指数 alpha(t_tilde >= 10)
をとったスケーリングプロットを作成する。
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO,
    setup_matplotlib, save_figure
)
from MSD import (
    load_strict_active_particles,
    load_time_varying_u_inf_and_xi,
    concat_pooled_particles_dimensionless_MSD,
)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "drive_decomposition",
    ROOT_DIR / "figure" / "drive_decomposition",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def fit_alpha_from_binned_curve(t_arr: np.ndarray, msd_arr: np.ndarray, weights: np.ndarray = None) -> float:
    mask = np.isfinite(t_arr) & np.isfinite(msd_arr) & (t_arr > 0) & (msd_arr > 0)
    if weights is not None:
        mask = mask & np.isfinite(weights) & (weights > 0)
    if np.sum(mask) < 2:
        return np.nan
    x = np.log10(t_arr[mask])
    y = np.log10(msd_arr[mask])
    if weights is not None:
        slope, _ = np.polyfit(x, y, 1, w=weights[mask])
    else:
        slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def bootstrap_ensemble_alpha_selected(
    df_dim_pool: pd.DataFrame,
    selected_t: np.ndarray,
    weights: np.ndarray,
    min_particles: int = 2,
    n_boot: int = 1000,
    seed: int = 42,
):
    """客観ウィンドウ (MSD_tilde >= 1 かつ N >= 0.5 N_max) 上で WLS の alpha と粒子単位ブートストラップ誤差を算出"""
    # selected_t との浮動小数点一致を保証するため最近傍ビンまたは isin を使用
    sub = df_dim_pool[df_dim_pool['t_tilde'].isin(selected_t)].copy()
    sub = sub.dropna(subset=['MSD_tilde'])
    sub = sub[sub['MSD_tilde'] > 0]
    if sub.empty:
        return np.nan, np.nan, np.nan, np.nan, 0

    pids = sub['unique_particle_id'].unique()
    n_p = len(pids)
    if n_p < min_particles:
        return np.nan, np.nan, np.nan, np.nan, n_p

    w_map = dict(zip(selected_t, weights))

    # 元のアンサンブル平均カーブ (WLS)
    grp = sub.groupby('t_tilde')['MSD_tilde'].agg(['mean', 'count']).reset_index()
    grp = grp[grp['count'] >= min_particles].sort_values('t_tilde')
    w_grp = np.array([w_map[t] for t in grp['t_tilde'].values], dtype=float)
    alpha_point = fit_alpha_from_binned_curve(grp['t_tilde'].values, grp['mean'].values, weights=w_grp)

    # ピボットテーブル (rows=particle, cols=t_tilde) で高速ブートストラップ
    piv = sub.pivot_table(index='unique_particle_id', columns='t_tilde', values='MSD_tilde', aggfunc='mean')
    t_cols = piv.columns.values.astype(float)
    w_cols = np.array([w_map[t] for t in t_cols], dtype=float)
    mat = piv.values  # shape (n_p, n_t)

    rng = np.random.default_rng(seed)
    boot_alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        sample = mat[idx, :]
        counts = np.sum(np.isfinite(sample), axis=0)
        valid_t = counts >= min_particles
        if np.sum(valid_t) < 2:
            continue
        means = np.nanmean(sample[:, valid_t], axis=0)
        a_b = fit_alpha_from_binned_curve(t_cols[valid_t], means, weights=w_cols[valid_t])
        if np.isfinite(a_b):
            boot_alphas.append(a_b)

    boot_arr = np.asarray(boot_alphas, dtype=float)
    if len(boot_arr) == 0:
        return alpha_point, np.nan, np.nan, np.nan, n_p

    sem = float(np.std(boot_arr, ddof=1))
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    return alpha_point, sem, ci_lo, ci_hi, n_p


def main():
    setup_matplotlib()
    print("=== Plotting alpha(t_tilde >= 10) vs Delta U_tilde_prop(R_c) ===")

    df_prop = pd.read_csv(CURRENT_DIR / "propulsion_potential_summary.csv")
    df_dim_sum = pd.read_csv(CURRENT_DIR / "figure" / "msd" / "msd_dimensionless_summary.csv")

    strict_set = load_strict_active_particles()
    flow_map = load_time_varying_u_inf_and_xi(CURRENT_DIR)

    dim_configs = {
        "beads06um": [4, 4, 4, 4],
        "beads1um":  [4, 4, 4, 4],
        "beads3um":  [4, 4, 4, 4],
        "beads5um":  [4, 4, 4, 4],
        "beads7um":  [4, 4, 4],
        "beads20um": [4, 4, 4],
    }

    rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        pr = df_prop[df_prop['bead_name'] == bname].iloc[0]

        # 客観ウィンドウ: MSD_tilde >= 1.0 (超ドメイン領域) かつ N >= 0.5 * N_max (生存バイアス回避)
        g_csv = df_dim_sum[df_dim_sum['bead_name'] == bname].sort_values('t_tilde')
        n_max = g_csv['n_active_particles'].max()
        sub_csv = g_csv[(g_csv['mean_msd_tilde'] >= 1.0) & (g_csv['n_active_particles'] >= 0.5 * n_max)].copy()
        sig_y = sub_csv['boot_sem_msd_tilde'].values / (sub_csv['mean_msd_tilde'].values * np.log(10.0))
        weights = 1.0 / np.maximum(sig_y, 1e-6)
        alpha_csv_wls = fit_alpha_from_binned_curve(sub_csv['t_tilde'].values, sub_csv['mean_msd_tilde'].values, weights=weights)

        # MSD.py と同一手順の粒子プールからブートストラップ CI を算出
        df_dim_pool = concat_pooled_particles_dimensionless_MSD(
            ROOT_DIR / bname,
            dim_configs[bname],
            flow_map=flow_map,
            scale=0.11,
            apply_cutoff=True,
            strict_particles=strict_set,
            t_tilde_bins=np.geomspace(0.05, 80.0, 46),
        )
        # 浮動小数点丸め誤差を吸収するため 10 桁丸めで照合
        df_dim_pool['t_key'] = df_dim_pool['t_tilde'].round(10)
        sel_keys = sub_csv['t_tilde'].round(10).values
        df_dim_pool_sel = df_dim_pool[df_dim_pool['t_key'].isin(sel_keys)].copy()
        selected_t = np.sort(df_dim_pool_sel['t_tilde'].unique())

        alpha_pt, alpha_sem, alpha_lo, alpha_hi, n_p = bootstrap_ensemble_alpha_selected(
            df_dim_pool_sel, selected_t=selected_t, weights=weights, min_particles=2, n_boot=1000, seed=42
        )

        rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'diameter_um': float(pr['radius_um']) * 2.0,
            'rc_over_xi': float(pr['rc_over_xi']),
            'delta_U_prop': float(pr['delta_U_prop']),
            'lambda_prop': float(pr['lambda_prop']),
            't_tilde_min': float(sub_csv['t_tilde'].min()),
            't_tilde_max': float(sub_csv['t_tilde'].max()),
            'n_bins': len(sub_csv),
            'alpha_wls': alpha_pt,
            'alpha_wls_csv': alpha_csv_wls,
            'alpha_sem': alpha_sem,
            'alpha_ci_lo': alpha_lo,
            'alpha_ci_hi': alpha_hi,
            'n_particles': n_p,
        })

    df_out = pd.DataFrame(rows).sort_values('delta_U_prop')
    df_out.to_csv(CURRENT_DIR / "alpha_superdomain_vs_delta_u_prop_summary.csv", index=False)
    print(df_out[['bead_name', 'diameter_um', 'delta_U_prop', 't_tilde_min', 't_tilde_max',
                  'n_bins', 'alpha_wls', 'alpha_wls_csv', 'alpha_sem', 'n_particles']].to_string(index=False))

    # 相関・線形回帰 (参考出力)
    x = df_out['delta_U_prop'].values
    y = df_out['alpha_wls'].values
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    rho_s, p_s = stats.spearmanr(x, y)
    print(f"\nLinear fit: alpha = {slope:.3f} * Delta_U_prop + {intercept:.3f}, R^2 = {r_val**2:.3f}, r = {r_val:.3f} (p = {p_val:.4f})")
    print(f"Spearman rho = {rho_s:.3f} (p = {p_s:.4f})")

    # プロット作成
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    x_min, x_max = -0.05, max(x) * 1.12
    y_min, y_max = 0.05, 2.38

    # 領域の背景シェーディング (U < 1: Diffusive, U > 1: Superdiffusive)
    ax.axvspan(x_min, 1.0, color='#e9eff6', alpha=0.65, zorder=0)
    ax.axvspan(1.0, x_max, color='#fcedea', alpha=0.65, zorder=0)

    # 基準線: alpha = 1.0 (通常拡散), alpha = 2.0 (弾道限界), Delta U_prop = 1 (転移境界)
    ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.3, zorder=1,
               label=r'Normal diffusion ($\alpha = 1$)')
    ax.axhline(2.0, color='#333333', linestyle='-.', linewidth=1.3, zorder=1,
               label=r'Ballistic limit ($\alpha = 2$)')
    ax.axvline(1.0, color='#444444', linestyle=':', linewidth=1.6, zorder=1,
               label=r'Pinning threshold ($\Delta\tilde{U}_{\mathrm{prop}} = 1$)')

    # 領域ラベル (上部に配置)
    ax.text(0.475, 2.21,
            'Diffusive regime\n' + r'($\Delta\tilde{U}_{\mathrm{prop}} < 1$)',
            ha='center', va='top', fontsize=11.0, fontweight='bold', color='#3b526b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b8c7d9', alpha=0.88),
            zorder=3)
    ax.text(0.5 * (1.0 + x_max), 2.21,
            'Superdiffusive regime\n' + r'($\Delta\tilde{U}_{\mathrm{prop}} > 1$)',
            ha='center', va='top', fontsize=11.0, fontweight='bold', color='#8c2d38',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e0b4b8', alpha=0.88),
            zorder=3)

    # 粒子サイズ別ラベルのオフセット微調整 (重複回避)
    offsets = {
        'beads06um': (-12, 8, 'right'),
        'beads1um':  (-12, -4, 'right'),
        'beads3um':  (-12, 8, 'right'),
        'beads7um':  (11, -8, 'left'),
        'beads5um':  (11, 8, 'left'),
        'beads20um': (11, 4, 'left'),
    }

    for b_item in BEADS_INFO:
        bname = b_item['name']
        r = df_out[df_out['bead_name'] == bname].iloc[0]
        yerr_lo = max(0.0, r['alpha_wls'] - r['alpha_ci_lo'])
        yerr_hi = max(0.0, r['alpha_ci_hi'] - r['alpha_wls'])
        ax.errorbar(
            r['delta_U_prop'], r['alpha_wls'],
            yerr=[[yerr_lo], [yerr_hi]],
            fmt=b_item['marker'], color=b_item['color'],
            markersize=9.5, markeredgecolor='#222222', markeredgewidth=0.9,
            ecolor=b_item['color'], elinewidth=1.8, capsize=4.0, alpha=0.92,
            zorder=4,
        )
        dx, dy, ha = offsets.get(bname, (8, 6, 'left'))
        ax.annotate(
            rf"{r['diameter_um']:.2f} $\mu$m" if r['diameter_um'] < 10 else rf"{r['diameter_um']:.1f} $\mu$m",
            xy=(r['delta_U_prop'], r['alpha_wls']),
            xytext=(dx, dy), textcoords='offset points',
            ha=ha, va='center',
            fontsize=10.5, fontweight='bold', color=b_item['color'],
            zorder=5,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'Effective propulsion well depth $\Delta \tilde{U}_{\mathrm{prop}}(R_c)$', fontsize=13)
    ax.set_ylabel(r'Super-domain diffusion exponent $\alpha_{\mathrm{WLS}}\,(\widetilde{\mathrm{MSD}} \geq 1,\ N \geq 0.5\,N_{\max})$', fontsize=12.0)
    ax.set_title(r'Super-Domain Diffusion Exponent $\alpha$ vs $\Delta \tilde{U}_{\mathrm{prop}}(R_c)$ (Error bars: 95% Bootstrap CI)',
                 fontsize=12.8, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.55)
    ax.legend(loc='lower right', fontsize=10.0, framealpha=0.94)
    fig.tight_layout()

    save_figure(fig, "fig_alpha_superdomain_vs_delta_u_prop", OUT_DIRS)
    plt.close(fig)
    print("Saved fig_alpha_superdomain_vs_delta_u_prop (.png and .svg)")

    # ------------------------------------------------------------------
    # Fig 2: 横軸 |lambda_tilde_prop(R_c)| (対数軸) vs alpha_WLS
    # ------------------------------------------------------------------
    lam_x = df_out['lambda_prop'].values
    r_log_lam, p_log_lam = stats.pearsonr(np.log10(lam_x), y)
    rho_lam, p_rho_lam = stats.spearmanr(lam_x, y)
    print(f"Correlation vs log10(|lambda_prop|): r = {r_log_lam:.3f} (p = {p_log_lam:.4f}), Spearman rho = {rho_lam:.3f} (p = {p_rho_lam:.4f})")

    fig2, ax2 = plt.subplots(figsize=(7.8, 6.0))
    lx_min, lx_max = 0.10, 65.0
    x_split = 10.0

    ax2.axvspan(lx_min, x_split, color='#e9eff6', alpha=0.65, zorder=0)
    ax2.axvspan(x_split, lx_max, color='#fcedea', alpha=0.65, zorder=0)

    ax2.axhline(1.0, color='#555555', linestyle='--', linewidth=1.3, zorder=1,
                label=r'Normal diffusion ($\alpha = 1$)')
    ax2.axhline(2.0, color='#333333', linestyle='-.', linewidth=1.3, zorder=1,
                label=r'Ballistic limit ($\alpha = 2$)')
    ax2.axvline(x_split, color='#444444', linestyle=':', linewidth=1.6, zorder=1,
                label=r'Stiffness jump ($|\tilde{\lambda}_{\mathrm{prop}}| \sim 10$)')

    ax2.text(np.sqrt(lx_min * x_split), 2.24,
             'Diffusive regime\n' + r'(Low stiffness: $|\tilde{\lambda}_{\mathrm{prop}}| < 10$)',
             ha='center', va='top', fontsize=10.8, fontweight='bold', color='#3b526b',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b8c7d9', alpha=0.88),
             zorder=3)
    ax2.text(np.sqrt(x_split * lx_max), 2.24,
             'Superdiffusive regime\n' + r'(Rail-locked: $|\tilde{\lambda}_{\mathrm{prop}}| > 10$)',
             ha='center', va='top', fontsize=10.8, fontweight='bold', color='#8c2d38',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e0b4b8', alpha=0.88),
             zorder=3)

    offsets_lam = {
        'beads06um': (10, 6, 'left'),
        'beads1um':  (-10, -4, 'right'),
        'beads3um':  (-12, 8, 'right'),
        'beads7um':  (11, -8, 'left'),
        'beads5um':  (11, 8, 'left'),
        'beads20um': (11, 4, 'left'),
    }

    for b_item in BEADS_INFO:
        bname = b_item['name']
        r = df_out[df_out['bead_name'] == bname].iloc[0]
        yerr_lo = max(0.0, r['alpha_wls'] - r['alpha_ci_lo'])
        yerr_hi = max(0.0, r['alpha_ci_hi'] - r['alpha_wls'])
        ax2.errorbar(
            r['lambda_prop'], r['alpha_wls'],
            yerr=[[yerr_lo], [yerr_hi]],
            fmt=b_item['marker'], color=b_item['color'],
            markersize=9.5, markeredgecolor='#222222', markeredgewidth=0.9,
            ecolor=b_item['color'], elinewidth=1.8, capsize=4.0, alpha=0.92,
            zorder=4,
        )
        dx, dy, ha = offsets_lam.get(bname, (8, 6, 'left'))
        ax2.annotate(
            rf"{r['diameter_um']:.2f} $\mu$m" if r['diameter_um'] < 10 else rf"{r['diameter_um']:.1f} $\mu$m",
            xy=(r['lambda_prop'], r['alpha_wls']),
            xytext=(dx, dy), textcoords='offset points',
            ha=ha, va='center',
            fontsize=10.5, fontweight='bold', color=b_item['color'],
            zorder=5,
        )

    ax2.set_xscale('log')
    ax2.set_xlim(lx_min, lx_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel(r'Propulsion stiffness $|\tilde{\lambda}_{\mathrm{prop}}(R_c)| \equiv \tilde{v}(R_c)\,\Phi(R_c)\,|\tilde{\lambda}_0|$', fontsize=12.8)
    ax2.set_ylabel(r'Super-domain diffusion exponent $\alpha_{\mathrm{WLS}}\,(\widetilde{\mathrm{MSD}} \geq 1,\ N \geq 0.5\,N_{\max})$', fontsize=12.0)
    ax2.set_title(r'Super-Domain Diffusion Exponent $\alpha$ vs $|\tilde{\lambda}_{\mathrm{prop}}(R_c)|$ (Error bars: 95% Bootstrap CI)',
                  fontsize=12.5, fontweight='bold', pad=10)
    ax2.grid(True, which='both', linestyle=':', alpha=0.55)
    ax2.legend(loc='lower right', fontsize=10.0, framealpha=0.94)
    fig2.tight_layout()

    save_figure(fig2, "fig_alpha_superdomain_vs_lambda_prop", OUT_DIRS)
    plt.close(fig2)
    print("Saved fig_alpha_superdomain_vs_lambda_prop (.png and .svg)")

    # ------------------------------------------------------------------
    # Fig 3: 横軸 R_c / xi (対数軸) vs alpha_WLS
    # ------------------------------------------------------------------
    rcxi_x = df_out['rc_over_xi'].values
    r_log_rcxi, p_log_rcxi = stats.pearsonr(np.log10(rcxi_x), y)
    rho_rcxi, p_rho_rcxi = stats.spearmanr(rcxi_x, y)
    print(f"Correlation vs log10(Rc/xi): r = {r_log_rcxi:.3f} (p = {p_log_rcxi:.4f}), Spearman rho = {rho_rcxi:.3f} (p = {p_rho_rcxi:.4f})")

    fig3, ax3 = plt.subplots(figsize=(7.8, 6.0))
    rx_min, rx_max = 0.045, 4.0
    rx_split = 0.28

    ax3.axvspan(rx_min, rx_split, color='#fcedea', alpha=0.65, zorder=0)
    ax3.axvspan(rx_split, rx_max, color='#e9eff6', alpha=0.65, zorder=0)

    ax3.axhline(1.0, color='#555555', linestyle='--', linewidth=1.3, zorder=1,
                label=r'Normal diffusion ($\alpha = 1$)')
    ax3.axhline(2.0, color='#333333', linestyle='-.', linewidth=1.3, zorder=1,
                label=r'Ballistic limit ($\alpha = 2$)')
    ax3.axvline(rx_split, color='#444444', linestyle=':', linewidth=1.6, zorder=1,
                label=r'Crossover scale ($R_c / \xi \approx 0.28$)')

    ax3.text(np.sqrt(rx_min * rx_split), 2.24,
             'Superdiffusive regime\n' + r'($R_c / \xi < 0.28$)',
             ha='center', va='top', fontsize=10.8, fontweight='bold', color='#8c2d38',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e0b4b8', alpha=0.88),
             zorder=3)
    ax3.text(np.sqrt(rx_split * rx_max), 2.24,
             'Diffusive regime\n' + r'($R_c / \xi > 0.28$)',
             ha='center', va='top', fontsize=10.8, fontweight='bold', color='#3b526b',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b8c7d9', alpha=0.88),
             zorder=3)

    offsets_rcxi = {
        'beads06um': (10, 6, 'left'),
        'beads1um':  (10, -4, 'left'),
        'beads3um':  (10, 6, 'left'),
        'beads7um':  (10, -10, 'left'),
        'beads5um':  (10, 8, 'left'),
        'beads20um': (-10, 10, 'right'),
    }

    for b_item in BEADS_INFO:
        bname = b_item['name']
        r = df_out[df_out['bead_name'] == bname].iloc[0]
        yerr_lo = max(0.0, r['alpha_wls'] - r['alpha_ci_lo'])
        yerr_hi = max(0.0, r['alpha_ci_hi'] - r['alpha_wls'])
        ax3.errorbar(
            r['rc_over_xi'], r['alpha_wls'],
            yerr=[[yerr_lo], [yerr_hi]],
            fmt=b_item['marker'], color=b_item['color'],
            markersize=9.5, markeredgecolor='#222222', markeredgewidth=0.9,
            ecolor=b_item['color'], elinewidth=1.8, capsize=4.0, alpha=0.92,
            zorder=4,
        )
        dx, dy, ha = offsets_rcxi.get(bname, (8, 6, 'left'))
        ax3.annotate(
            rf"{r['diameter_um']:.2f} $\mu$m" if r['diameter_um'] < 10 else rf"{r['diameter_um']:.1f} $\mu$m",
            xy=(r['rc_over_xi'], r['alpha_wls']),
            xytext=(dx, dy), textcoords='offset points',
            ha=ha, va='center',
            fontsize=10.5, fontweight='bold', color=b_item['color'],
            zorder=5,
        )

    ax3.set_xscale('log')
    ax3.set_xlim(rx_min, rx_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel(r'Dimensionless cargo radius $R_c / \xi$', fontsize=13)
    ax3.set_ylabel(r'Super-domain diffusion exponent $\alpha_{\mathrm{WLS}}\,(\widetilde{\mathrm{MSD}} \geq 1,\ N \geq 0.5\,N_{\max})$', fontsize=12.0)
    ax3.set_title(r'Super-Domain Diffusion Exponent $\alpha$ vs $R_c / \xi$ (Error bars: 95% Bootstrap CI)',
                  fontsize=12.8, fontweight='bold', pad=10)
    ax3.grid(True, which='both', linestyle=':', alpha=0.55)
    ax3.legend(loc='lower left', fontsize=10.0, framealpha=0.94)
    fig3.tight_layout()

    save_figure(fig3, "fig_alpha_superdomain_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig3)
    print("Saved fig_alpha_superdomain_vs_rc_over_xi (.png and .svg)")


if __name__ == "__main__":
    main()
