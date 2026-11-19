#!/usr/bin/env python3
"""
fit_model_b_pinning.py

モデル B（バルク整列 + 局所レール・ピンニング井戸）の非ビン化最尤推定 (MLE) フィット:
  U_eff(Theta_bulk) = -kappa_1 * cos(Theta_bulk) - U_pin * exp(-Theta_bulk^2 / (2 * sigma_pin^2))

- データセット: verify_cos_bulk_potential.py / analyze_drive_decomposition.py と同一 (19,899 frames, 71 cargos)
- 粒子単位のクラスターブートストラップ (B=500) で 95% 信頼区間を算出
- 純粋 von Mises (U_pin = 0) との AIC / BIC 比較を実施
- 出力先:
    figure/drive_decomposition/
    /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/drive_decomposition/
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import i0, i1

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths
)
from verify_coarse_grained_velocity_noise import load_exact_pipeline_frames
from plot_fig5_linear_stability import load_sigma_mt_by_bead

OUT_DIRS = [
    CURRENT_DIR / "figure" / "drive_decomposition",
    ROOT_DIR / "figure" / "drive_decomposition",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# 数値積分用の細かい等間隔グリッド [-pi, pi]
TH_GRID = np.linspace(-np.pi, np.pi, 601)
DTH = TH_GRID[1] - TH_GRID[0]
COS_GRID = np.cos(TH_GRID)
TH2_GRID = TH_GRID ** 2


def fit_von_mises_mle(cos_th: np.ndarray) -> float:
    r_bar = float(np.mean(cos_th))
    if not np.isfinite(r_bar) or r_bar <= 0.0:
        return 0.0
    if r_bar >= 0.9999:
        return 50.0
    return float(brentq(lambda k: (i1(k) / i0(k)) - r_bar, 0.0, 50.0))


def log_Z_model_b(k1: float, u_pin: float, sig: float) -> float:
    """規格化定数 Z = int_{-pi}^{pi} exp(k1*cos(th) + u_pin*exp(-th^2/(2*sig^2))) dth の対数"""
    exponent = k1 * COS_GRID + u_pin * np.exp(-0.5 * TH2_GRID / (sig ** 2))
    m = np.max(exponent)
    z_val = np.trapezoid(np.exp(exponent - m), dx=DTH)
    return float(m + np.log(z_val))


def fit_model_b_mle(cos_th: np.ndarray, th2: np.ndarray, fixed_sigma: float = None):
    """
    モデル B の最尤推定:
      log P(theta) = k1*cos(theta) + u_pin*exp(-theta^2/(2*sig^2)) - log(Z)
    """
    n = len(cos_th)
    if fixed_sigma is not None:
        sig = float(fixed_sigma)

        def nll(params):
            k1, u_pin = params
            lz = log_Z_model_b(k1, u_pin, sig)
            ll = np.sum(k1 * cos_th + u_pin * np.exp(-0.5 * th2 / (sig ** 2))) - n * lz
            return -float(ll)

        k_init = fit_von_mises_mle(cos_th)
        res = minimize(
            nll, x0=[max(0.05, k_init * 0.8), 0.3],
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            method='L-BFGS-B'
        )
        k1_hat, upin_hat = float(res.x[0]), float(res.x[1])
        return k1_hat, upin_hat, sig, -float(res.fun)
    else:
        def nll(params):
            k1, u_pin, sig = params
            lz = log_Z_model_b(k1, u_pin, sig)
            ll = np.sum(k1 * cos_th + u_pin * np.exp(-0.5 * th2 / (sig ** 2))) - n * lz
            return -float(ll)

        k_init = fit_von_mises_mle(cos_th)
        best_res = None
        for sig0 in [0.18, 0.24, 0.35]:
            res = minimize(
                nll, x0=[max(0.05, k_init * 0.8), 0.4, sig0],
                bounds=[(0.0, 10.0), (0.0, 10.0), (0.15, 0.50)],
                method='L-BFGS-B'
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        return float(best_res.x[0]), float(best_res.x[1]), float(best_res.x[2]), -float(best_res.fun)


def cluster_bootstrap_model_b(th: np.ndarray, cargo_ids: np.ndarray, fixed_sigma: float = 0.20, n_boot: int = 500, seed: int = 42):
    """粒子単位のクラスターブートストラップ (自由 sigma_pin と 固定 sigma_pin=sigma_MT)"""
    cos_th = np.cos(th)
    th2 = th ** 2
    uniq_c = np.unique(cargo_ids)
    n_c = len(uniq_c)

    k1_f, up_f, sig_f, ll_f = fit_model_b_mle(cos_th, th2, fixed_sigma=None)
    k1_fix, up_fix, _, ll_fix = fit_model_b_mle(cos_th, th2, fixed_sigma=fixed_sigma)

    # 粒子ごとの配列を事前にスライス
    c_cos = [cos_th[cargo_ids == cid] for cid in uniq_c]
    c_th2 = [th2[cargo_ids == cid] for cid in uniq_c]

    rng = np.random.default_rng(seed)
    b_k1_f, b_up_f, b_sig_f, b_lam_f, b_lpin_f = [], [], [], [], []
    b_k1_fix, b_up_fix, b_lam_fix, b_lpin_fix = [], [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n_c, size=n_c)
        cos_b = np.concatenate([c_cos[i] for i in idx])
        th2_b = np.concatenate([c_th2[i] for i in idx])

        k1_b, up_b, sig_b, _ = fit_model_b_mle(cos_b, th2_b, fixed_sigma=None)
        b_k1_f.append(k1_b)
        b_up_f.append(up_b)
        b_sig_f.append(sig_b)
        lp_b = up_b / (sig_b ** 2)
        b_lpin_f.append(lp_b)
        b_lam_f.append(k1_b + lp_b)

        k1_bx, up_bx, _, _ = fit_model_b_mle(cos_b, th2_b, fixed_sigma=fixed_sigma)
        b_k1_fix.append(k1_bx)
        b_up_fix.append(up_bx)
        lp_bx = up_bx / (fixed_sigma ** 2)
        b_lpin_fix.append(lp_bx)
        b_lam_fix.append(k1_bx + lp_bx)

    lpin_f = up_f / (sig_f ** 2)
    lam_f = k1_f + lpin_f
    lpin_fix = up_fix / (fixed_sigma ** 2)
    lam_fix = k1_fix + lpin_fix

    return {
        'k1_free': k1_f,
        'k1_free_lo': float(np.percentile(b_k1_f, 2.5)),
        'k1_free_hi': float(np.percentile(b_k1_f, 97.5)),
        'upin_free': up_f,
        'upin_free_lo': float(np.percentile(b_up_f, 2.5)),
        'upin_free_hi': float(np.percentile(b_up_f, 97.5)),
        'sig_free': sig_f,
        'sig_free_lo': float(np.percentile(b_sig_f, 2.5)),
        'sig_free_hi': float(np.percentile(b_sig_f, 97.5)),
        'lam_pin_free': lpin_f,
        'lam_pin_free_lo': float(np.percentile(b_lpin_f, 2.5)),
        'lam_pin_free_hi': float(np.percentile(b_lpin_f, 97.5)),
        'lam_free': lam_f,
        'lam_free_lo': float(np.percentile(b_lam_f, 2.5)),
        'lam_free_hi': float(np.percentile(b_lam_f, 97.5)),
        'll_free': ll_f,
        'k1_fix': k1_fix,
        'k1_fix_lo': float(np.percentile(b_k1_fix, 2.5)),
        'k1_fix_hi': float(np.percentile(b_k1_fix, 97.5)),
        'upin_fix': up_fix,
        'upin_fix_lo': float(np.percentile(b_up_fix, 2.5)),
        'upin_fix_hi': float(np.percentile(b_up_fix, 97.5)),
        'lam_pin_fix': lpin_fix,
        'lam_pin_fix_lo': float(np.percentile(b_lpin_fix, 2.5)),
        'lam_pin_fix_hi': float(np.percentile(b_lpin_fix, 97.5)),
        'lam_fix': lam_fix,
        'lam_fix_lo': float(np.percentile(b_lam_fix, 2.5)),
        'lam_fix_hi': float(np.percentile(b_lam_fix, 97.5)),
        'll_fix': ll_fix,
    }


def main():
    setup_matplotlib()
    print("=== Model B (Bulk Alignment + Local Rail Pinning Well) MLE Fit ===")

    sigma_mt_map = load_sigma_mt_by_bead(ROOT_DIR)
    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0

    df_frames = load_exact_pipeline_frames(ROOT_DIR, active_set, xi_dict)
    df_frames['xi_um'] = df_frames['exp_dir'].map(xi_dict).fillna(default_xi)
    df_frames['cargo_id'] = df_frames['exp_dir'].astype(str) + '_' + df_frames['particle'].astype(str)

    cross_val = df_frames['u'].values * df_frames['cdy_px'].values - df_frames['v'].values * df_frames['cdx_px'].values
    dot_val = df_frames['u'].values * df_frames['cdx_px'].values + df_frames['v'].values * df_frames['cdy_px'].values
    df_frames['theta_bulk'] = np.arctan2(cross_val, dot_val)

    # 粒子ごとの R_c / xi 集計
    radius_map = {b['name']: b['radius_um'] for b in BEADS_INFO}
    df_frames['radius_um'] = df_frames['bead_name'].map(radius_map)
    df_frames['rc_over_xi'] = df_frames['radius_um'] / df_frames['xi_um']

    # ビン設定 (小角度の尖りが見えるよう 31 ビン、中央ビンが 0 を中心にする)
    n_bins = 31
    th_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
    th_fine = np.linspace(-np.pi, np.pi, 400)

    records = []
    plot_cache = {}

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_frames[df_frames['bead_name'] == bname]
        th = sub['theta_bulk'].values
        cid = sub['cargo_id'].values
        n_f = len(th)
        p_rc = sub.drop_duplicates('cargo_id')
        rc_xi_m = float(p_rc['rc_over_xi'].mean())
        rc_xi_sem = float(p_rc['rc_over_xi'].sem()) if len(p_rc) > 1 else 0.0

        # Model A: pure von Mises
        k_vm = fit_von_mises_mle(np.cos(th))
        ll_vm = float(np.sum(k_vm * np.cos(th)) - n_f * np.log(2.0 * np.pi * i0(k_vm)))
        aic_vm = 2 * 1 - 2 * ll_vm
        bic_vm = 1 * np.log(n_f) - 2 * ll_vm

        # Model B: MLE + Cluster Bootstrap (B=500)
        sig_mt = sigma_mt_map[bname]['sigma_MT']
        s_global = sigma_mt_map[bname]['S_global']
        res_b = cluster_bootstrap_model_b(th, cid, fixed_sigma=sig_mt, n_boot=500, seed=42)
        aic_b_free = 2 * 3 - 2 * res_b['ll_free']
        bic_b_free = 3 * np.log(n_f) - 2 * res_b['ll_free']
        aic_b_fix = 2 * 2 - 2 * res_b['ll_fix']
        bic_b_fix = 2 * np.log(n_f) - 2 * res_b['ll_fix']

        d_aic_free = aic_vm - aic_b_free
        d_bic_free = bic_vm - bic_b_free
        d_aic_fix = aic_vm - aic_b_fix
        d_bic_fix = bic_vm - bic_b_fix

        # ヒストグラムとブートストラップ 95% CI (表示用)
        counts, _ = np.histogram(th, bins=th_edges, density=True)
        u_obs = np.where(counts > 0, -np.log(2.0 * np.pi * counts), np.nan)
        uniq_c = np.unique(cid)
        rng_h = np.random.default_rng(123)
        u_boots = np.full((500, n_bins), np.nan)
        c_th_list = [th[cid == c] for c in uniq_c]
        for b in range(500):
            idx = rng_h.integers(0, len(uniq_c), size=len(uniq_c))
            th_b = np.concatenate([c_th_list[i] for i in idx])
            cb, _ = np.histogram(th_b, bins=th_edges, density=True)
            u_boots[b] = np.where(cb > 0, -np.log(2.0 * np.pi * cb), np.nan)
        u_lo = np.nanpercentile(u_boots, 2.5, axis=0)
        u_hi = np.nanpercentile(u_boots, 97.5, axis=0)

        plot_cache[bname] = {
            'u_obs': u_obs, 'u_lo': u_lo, 'u_hi': u_hi,
            'k_vm': k_vm, 'd_bic_free': d_bic_free, 'd_aic_free': d_aic_free,
            'd_bic_fix': d_bic_fix, 'd_aic_fix': d_aic_fix,
            'sig_mt': sig_mt, 'S_global': s_global,
            'rc_xi': rc_xi_m, **res_b
        }

        records.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'rc_over_xi': rc_xi_m,
            'rc_over_xi_sem': rc_xi_sem,
            'n_frames': n_f,
            'n_cargos': len(uniq_c),
            'kappa_vm': k_vm,
            'kappa1_free': res_b['k1_free'],
            'kappa1_free_lo': res_b['k1_free_lo'],
            'kappa1_free_hi': res_b['k1_free_hi'],
            'upin_free': res_b['upin_free'],
            'upin_free_lo': res_b['upin_free_lo'],
            'upin_free_hi': res_b['upin_free_hi'],
            'sigma_pin_free': res_b['sig_free'],
            'sigma_pin_free_lo': res_b['sig_free_lo'],
            'sigma_pin_free_hi': res_b['sig_free_hi'],
            'lam_pin_free': res_b['lam_pin_free'],
            'lam_pin_free_lo': res_b['lam_pin_free_lo'],
            'lam_pin_free_hi': res_b['lam_pin_free_hi'],
            'lambda_tilde_free': res_b['lam_free'],
            'lambda_tilde_free_lo': res_b['lam_free_lo'],
            'lambda_tilde_free_hi': res_b['lam_free_hi'],
            'delta_AIC_free': d_aic_free,
            'delta_BIC_free': d_bic_free,
            'S_global': s_global,
            'sigma_MT': sig_mt,
            'kappa1_fix020': res_b['k1_fix'],
            'kappa1_fix020_lo': res_b['k1_fix_lo'],
            'kappa1_fix020_hi': res_b['k1_fix_hi'],
            'upin_fix020': res_b['upin_fix'],
            'upin_fix020_lo': res_b['upin_fix_lo'],
            'upin_fix020_hi': res_b['upin_fix_hi'],
            'lam_pin_fix020': res_b['lam_pin_fix'],
            'lam_pin_fix020_lo': res_b['lam_pin_fix_lo'],
            'lam_pin_fix020_hi': res_b['lam_pin_fix_hi'],
            'lambda_tilde_fix020': res_b['lam_fix'],
            'lambda_tilde_fix020_lo': res_b['lam_fix_lo'],
            'lambda_tilde_fix020_hi': res_b['lam_fix_hi'],
            'delta_AIC_fix020': d_aic_fix,
            'delta_BIC_fix020': d_bic_fix,
        })

    df_res = pd.DataFrame(records).sort_values('rc_over_xi')
    df_res.to_csv(CURRENT_DIR / "model_b_pinning_summary.csv", index=False)
    print("\n=== Model B Summary (Free sigma_pin in [0.15, 0.50], sorted by R_c/xi) ===")
    print(df_res[['bead_name', 'rc_over_xi', 'kappa_vm', 'kappa1_free',
                  'upin_free', 'upin_free_lo', 'upin_free_hi',
                  'sigma_pin_free', 'delta_AIC_free', 'delta_BIC_free']].to_string(index=False))
    print("\n=== Linear Stability Analysis at Theta_bulk = 0 (Free sigma_pin) ===")
    print(df_res[['bead_name', 'rc_over_xi', 'kappa1_free', 'lam_pin_free',
                  'lambda_tilde_free', 'lambda_tilde_free_lo', 'lambda_tilde_free_hi']].to_string(index=False))
    print("\n=== Linear Stability Analysis at Theta_bulk = 0 (Fixed sigma_pin = 0.20 rad) ===")
    print(df_res[['bead_name', 'rc_over_xi', 'kappa1_fix020', 'lam_pin_fix020',
                  'lambda_tilde_fix020', 'lambda_tilde_fix020_lo', 'lambda_tilde_fix020_hi']].to_string(index=False))

    # ------------------------------------------------------------------
    # Fig 1: 6パネル U_eff(Theta_bulk) + Model B フィット (バルク成分分解付き)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        pc = plot_cache[bname]

        valid = ~np.isnan(pc['u_obs'])
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.0, alpha=0.6)

        ax.plot(th_centers[valid], pc['u_obs'][valid], color=col, marker=b_item['marker'],
                markersize=4.5, linewidth=1.8, label=r'Observed $U_{\mathrm{eff}}(\Theta)$', zorder=4)
        ax.fill_between(th_centers[valid], pc['u_lo'][valid], pc['u_hi'][valid],
                        color=col, alpha=0.20, zorder=2)

        # Pure von Mises
        u_vm = np.log(i0(pc['k_vm'])) - pc['k_vm'] * np.cos(th_fine)
        ax.plot(th_fine, u_vm, color='#555555', linestyle='--', linewidth=1.6,
                label=rf'von Mises ($\kappa={pc["k_vm"]:.2f}$)', zorder=3)

        # Model B (fixed sigma_pin = sigma_MT from S_global)
        k1, up, sig = pc['k1_fix'], pc['upin_fix'], pc['sig_mt']
        lz_b = log_Z_model_b(k1, up, sig)
        u0_b = lz_b - np.log(2.0 * np.pi)
        u_b_tot = u0_b - k1 * np.cos(th_fine) - up * np.exp(-0.5 * (th_fine ** 2) / (sig ** 2))
        u_b_bulk = u0_b - k1 * np.cos(th_fine)

        ax.plot(th_fine, u_b_tot, color='#111111', linestyle='-', linewidth=2.2,
                label=rf'Model B ($\kappa_1={k1:.2f}, U_{{\mathrm{{pin}}}}={up:.2f}, \sigma_{{\mathrm{{MT}}}}={sig:.2f}$)', zorder=5)
        if up > 0.05:
            ax.plot(th_fine, u_b_bulk, color='#d62728', linestyle=':', linewidth=1.6,
                    label=r'Bulk term only ($-\kappa_1\cos\Theta$)', zorder=4)

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.6, 1.1)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_title(rf'{lbl}  ($R_c/\xi={pc["rc_xi"]:.2f}$, $\Delta\mathrm{{BIC}}={pc["d_bic_fix"]:+.1f}$)',
                     fontsize=12.5, fontweight='bold')
        if idx >= 3:
            ax.set_xlabel(r'Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ (rad)', fontsize=12)
        if idx % 3 == 0:
            ax.set_ylabel(r'Effective Potential $U_{\mathrm{eff}}(\Theta)$', fontsize=12.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=8.0, framealpha=0.92)

    fig.suptitle(r'Model B Fit: $U_{\mathrm{eff}}(\Theta_{\mathrm{bulk}}) = -\kappa_1\cos\Theta_{\mathrm{bulk}} - U_{\mathrm{pin}}\exp(-\Theta_{\mathrm{bulk}}^2 / 2\sigma_{\mathrm{pin}}^2)$',
                 fontsize=15.5, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_model_b_potential_6panels", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: kappa_1 と U_pin vs R_c / xi
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for b_item in BEADS_INFO:
        row = df_res[df_res['bead_name'] == b_item['name']].iloc[0]
        k1_err = [[row['kappa1_free'] - row['kappa1_free_lo']], [row['kappa1_free_hi'] - row['kappa1_free']]]
        up_err = [[row['upin_free'] - row['upin_free_lo']], [row['upin_free_hi'] - row['upin_free']]]

        ax1.errorbar(row['rc_over_xi'], row['kappa1_free'], yerr=k1_err,
                     xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
                     markersize=9.0, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
                     label=b_item['label'], zorder=4)
        ax2.errorbar(row['rc_over_xi'], row['upin_free'], yerr=up_err,
                     xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
                     markersize=9.0, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
                     label=b_item['label'], zorder=4)

    ax1.plot(df_res['rc_over_xi'], df_res['kappa1_free'], color='#222222', linewidth=1.6, alpha=0.7, zorder=2)
    ax1.plot(df_res['rc_over_xi'], df_res['kappa_vm'], color='#7f7f7f', linestyle='--', linewidth=1.5,
             marker='.', label=r'Pure von Mises $\kappa_{\mathrm{vM}}$', zorder=1)
    ax1.set_xscale('log')
    ax1.set_xlim(0.04, 3.2)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=13)
    ax1.set_ylabel(r'Bulk Alignment Strength $\kappa_1$', fontsize=13)
    ax1.set_title(r'(a) Bulk Alignment Term $\kappa_1$ vs $R_c/\xi$', fontsize=13.5, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.8, framealpha=0.92, ncol=2)

    ax2.axhline(0.0, color='#888888', linestyle='--', linewidth=1.0)
    ax2.plot(df_res['rc_over_xi'], df_res['upin_free'], color='#222222', linewidth=1.6, alpha=0.7, zorder=2)
    ax2.plot(df_res['rc_over_xi'], df_res['upin_fix020'], color='#d62728', linestyle=':', linewidth=1.6,
             marker='x', markersize=6, label=r'Fixed $\sigma_{\mathrm{pin}}=\sigma_{\mathrm{MT}}(S_{\mathrm{global}})$', zorder=3)
    ax2.set_xscale('log')
    ax2.set_xlim(0.04, 3.2)
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=13)
    ax2.set_ylabel(r'Pinning Well Depth $U_{\mathrm{pin}}$', fontsize=13)
    ax2.set_title(r'(b) Local Rail Pinning Well $U_{\mathrm{pin}}$ vs $R_c/\xi$', fontsize=13.5, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.8, framealpha=0.92, ncol=2)

    fig.tight_layout()
    save_figure(fig, "fig_model_b_params_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3: 線形安定性解析 |lambda_tilde| = kappa_1 + U_pin / sigma_pin^2 vs R_c/xi
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.6))

    for b_item in BEADS_INFO:
        row = df_res[df_res['bead_name'] == b_item['name']].iloc[0]
        lam_err = [[row['lambda_tilde_free'] - row['lambda_tilde_free_lo']],
                   [row['lambda_tilde_free_hi'] - row['lambda_tilde_free']]]
        ax1.errorbar(row['rc_over_xi'], row['lambda_tilde_free'], yerr=lam_err,
                     xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
                     markersize=9.0, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
                     label=b_item['label'], zorder=4)

    ax1.plot(df_res['rc_over_xi'], df_res['lambda_tilde_free'], color='#111111', linewidth=1.8,
             label=r'Total $|\tilde{\lambda}| = \kappa_1 + U_{\mathrm{pin}}/\sigma_{\mathrm{pin}}^2$', zorder=3)
    ax1.plot(df_res['rc_over_xi'], df_res['lambda_tilde_fix020'], color='#d62728', linestyle=':', linewidth=1.7,
             marker='x', markersize=6, label=r'Fixed $\sigma_{\mathrm{pin}}=\sigma_{\mathrm{MT}}(S_{\mathrm{global}})$', zorder=2)
    ax1.plot(df_res['rc_over_xi'], df_res['kappa1_free'], color='#1f77b4', linestyle='--', linewidth=1.6,
             marker='s', markersize=4.5, label=r'Bulk term only ($\kappa_1$)', zorder=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.04, 3.2)
    ax1.set_ylim(0.1, 60.0)
    ax1.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=13)
    ax1.set_ylabel(r'Restoring Stiffness $|\tilde{\lambda}| = |\lambda|/D_r^{\mathrm{eff}}$', fontsize=13)
    ax1.set_title(r'(a) Linear Stability Stiffness $|\tilde{\lambda}|$ at $\Theta_{\mathrm{bulk}}=0$', fontsize=13.5, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.92, ncol=2)

    # (b) 寄与分解バー (積み上げ棒グラフ: バルク項 kappa_1 vs ピンニング項 U_pin/sigma_pin^2)
    x_pos = np.arange(len(df_res))
    ax2.bar(x_pos, df_res['kappa1_free'], width=0.58, color='#4c72b0', edgecolor='k', linewidth=0.6,
            label=r'Bulk alignment $\kappa_1$')
    ax2.bar(x_pos, df_res['lam_pin_free'], bottom=df_res['kappa1_free'], width=0.58,
            color='#dd8452', edgecolor='k', linewidth=0.6,
            label=r'Rail pinning $U_{\mathrm{pin}} / \sigma_{\mathrm{pin}}^2$')
    for i, (_, row) in enumerate(df_res.iterrows()):
        tot = row['lambda_tilde_free']
        frac = 100.0 * row['lam_pin_free'] / tot if tot > 0 else 0.0
        ax2.text(i, tot + 0.5, f"{tot:.1f}\n({frac:.0f}%)", ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{r['label']}\n({r['rc_over_xi']:.2f})" for _, r in df_res.iterrows()], fontsize=9.0)
    ax2.set_ylim(0, max(df_res['lambda_tilde_free']) * 1.22)
    ax2.set_xlabel(r'Cargo Diameter ($R_c / \xi$)', fontsize=13)
    ax2.set_ylabel(r'Restoring Stiffness $|\tilde{\lambda}|$', fontsize=13)
    ax2.set_title(r'(b) Stiffness Decomposition: Bulk vs Pinning (% from Pinning)', fontsize=13.5, fontweight='bold')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "fig_model_b_linear_stability", OUT_DIRS)
    plt.close(fig)
    print("\nModel B fitting and linear stability analysis complete.")


if __name__ == "__main__":
    main()

