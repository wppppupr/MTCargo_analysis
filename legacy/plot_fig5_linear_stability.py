#!/usr/bin/env python3
"""
plot_fig5_linear_stability.py

系全体のネマチック秩序度 S_global (global_nematic_polar_order_summary.csv) から背景レールの角度分散
  sigma_MT = sqrt(-ln(S_global) / 2)
を見積もり、sigma_pin = sigma_MT に固定した上で周期核モデル B:
  U_tilde(Theta) = -kappa_1 * cos(Theta) - U_pin * exp((cos(Theta) - 1) / sigma_pin^2)
の (kappa_1, U_pin) を非ビン化最尤推定 (MLE) でフィットし、
線形安定性解析と 3 パネル Fig 5 を生成する。
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import i0, i1

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO, SCALE_UM_PER_PX,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths, load_all_data,
    find_experiment_dirs
)
from verify_coarse_grained_velocity_noise import load_exact_pipeline_frames

OUT_DIRS = [
    CURRENT_DIR / "figure" / "drive_decomposition",
    ROOT_DIR / "figure" / "drive_decomposition",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

TH_GRID = np.linspace(-np.pi, np.pi, 601)
DTH = TH_GRID[1] - TH_GRID[0]
COS_GRID = np.cos(TH_GRID)


def load_sigma_mt_by_bead(root_dir: Path) -> dict:
    """
    系全体のネマチック秩序度 S_global (global_nematic_polar_order_summary.csv) を取得し、
    sigma_MT = sqrt(-ln(S_global) / 2) を算出する。
    """
    global_csv = CURRENT_DIR / "figure" / "global_order" / "global_nematic_polar_order_summary.csv"
    df_glob = pd.read_csv(global_csv).set_index('bead_name')
    res = {}
    for b_item in BEADS_INFO:
        bname = b_item['name']
        if bname in df_glob.index:
            s_val = float(df_glob.loc[bname, 'nematic_mean'])
            n_exp = int(df_glob.loc[bname, 'n_nem'])
        else:
            s_val, n_exp = np.nan, 0
        sig_mt = float(np.sqrt(-np.log(s_val) / 2.0)) if (np.isfinite(s_val) and 0.0 < s_val < 1.0) else 0.20
        res[bname] = {'S_global': s_val, 'sigma_MT': sig_mt, 'n_exp': n_exp}
    return res


def fit_von_mises_mle(cos_th: np.ndarray) -> float:
    r_bar = float(np.mean(cos_th))
    if not np.isfinite(r_bar) or r_bar <= 0.0:
        return 0.0
    if r_bar >= 0.9999:
        return 50.0
    return float(brentq(lambda k: (i1(k) / i0(k)) - r_bar, 0.0, 50.0))


def log_Z_periodic(k1: float, u_pin: float, sig: float) -> float:
    exponent = k1 * COS_GRID + u_pin * np.exp((COS_GRID - 1.0) / (sig ** 2))
    m = np.max(exponent)
    z_val = np.trapezoid(np.exp(exponent - m), dx=DTH)
    return float(m + np.log(z_val))


def expected_cos_periodic(k1: float, u_pin: float, sig: float) -> float:
    exponent = k1 * COS_GRID + u_pin * np.exp((COS_GRID - 1.0) / (sig ** 2))
    m = np.max(exponent)
    w = np.exp(exponent - m)
    return float(np.trapezoid(COS_GRID * w, dx=DTH) / np.trapezoid(w, dx=DTH))


def fit_periodic_model_b_fixed_sigma(cos_th: np.ndarray, sig_fixed: float):
    """sigma_pin = sig_fixed に固定して (kappa_1, U_pin) を MLE で求める"""
    n = len(cos_th)
    kern = np.exp((cos_th - 1.0) / (sig_fixed ** 2))

    def nll(params):
        k1, u_pin = params
        lz = log_Z_periodic(k1, u_pin, sig_fixed)
        ll = np.sum(k1 * cos_th + u_pin * kern) - n * lz
        return -float(ll)

    k_init = fit_von_mises_mle(cos_th)
    res = minimize(
        nll, x0=[max(0.05, k_init * 0.7), 0.3],
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        method='L-BFGS-B'
    )
    return float(res.x[0]), float(res.x[1]), -float(res.fun)


def phase_velocity(theta: np.ndarray, k1: float, u_pin: float, sig: float) -> np.ndarray:
    return -np.sin(theta) * (k1 + (u_pin / (sig ** 2)) * np.exp((np.cos(theta) - 1.0) / (sig ** 2)))


def main():
    setup_matplotlib()
    print("=== Periodic Model B Fit with sigma_pin = sigma_MT (from S_global) ===")

    sigma_mt_map = load_sigma_mt_by_bead(ROOT_DIR)
    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0

    df_instant, _ = load_all_data(ROOT_DIR, active_set)
    df_frames = load_exact_pipeline_frames(ROOT_DIR, active_set, xi_dict)

    df = df_instant.merge(
        df_frames[['exp_dir', 'particle', 'frame', 'cdx_px', 'cdy_px', 'u', 'v']],
        on=['exp_dir', 'particle', 'frame'],
        how='inner',
    )
    df['xi_um'] = df['exp_dir'].map(xi_dict).fillna(default_xi)
    df['rc_over_xi'] = df['radius_um'] / df['xi_um']
    df['cargo_id'] = df['exp_dir'].astype(str) + '_' + df['particle'].astype(str)

    cross_val = df['u'].values * df['cdy_px'].values - df['v'].values * df['cdx_px'].values
    dot_val = df['u'].values * df['cdx_px'].values + df['v'].values * df['cdy_px'].values
    df['theta_bulk'] = np.arctan2(cross_val, dot_val)

    rng = np.random.default_rng(42)
    N_BOOT = 500
    rows = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df[df['bead_name'] == bname]
        th = sub['theta_bulk'].values
        cos_th = np.cos(th)
        cid = sub['cargo_id'].values
        n_f = len(th)

        p_rc = sub.drop_duplicates('cargo_id')
        rc_xi_m = float(p_rc['rc_over_xi'].mean())
        rc_xi_sem = float(p_rc['rc_over_xi'].sem()) if len(p_rc) > 1 else 0.0

        s_global = sigma_mt_map[bname]['S_global']
        sig_mt = sigma_mt_map[bname]['sigma_MT']

        # Fixed-sigma Periodic Model B MLE (2 params: kappa_1, U_pin)
        k1, upin, ll_b = fit_periodic_model_b_fixed_sigma(cos_th, sig_mt)
        k_vm = fit_von_mises_mle(cos_th)
        ll_vm = float(np.sum(k_vm * cos_th) - n_f * np.log(2.0 * np.pi * i0(k_vm)))
        d_aic = (2 * 1 - 2 * ll_vm) - (2 * 2 - 2 * ll_b)
        d_bic = (1 * np.log(n_f) - 2 * ll_vm) - (2 * np.log(n_f) - 2 * ll_b)

        lam0 = k1 + upin / (sig_mt ** 2)
        tau_rel = 1.0 / lam0 if lam0 > 0 else np.nan
        delta_u_tot = 2.0 * k1 + upin * (1.0 - np.exp(-2.0 / (sig_mt ** 2)))

        # Cluster bootstrap 95% CI
        uniq_c = np.unique(cid)
        n_c = len(uniq_c)
        c_cos = [cos_th[cid == c] for c in uniq_c]
        b_k1, b_up, b_lam, b_tau = [], [], [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_c, size=n_c)
            cb = np.concatenate([c_cos[i] for i in idx])
            k1_b, up_b, _ = fit_periodic_model_b_fixed_sigma(cb, sig_mt)
            l_b = k1_b + up_b / (sig_mt ** 2)
            b_k1.append(k1_b)
            b_up.append(up_b)
            b_lam.append(l_b)
            b_tau.append(1.0 / l_b if l_b > 0 else np.nan)

        phi_valid = sub['Phi_mag'].dropna().values
        phi_mean = float(np.mean(phi_valid)) if len(phi_valid) > 0 else np.nan
        cos_model = expected_cos_periodic(k1, upin, sig_mt)
        cos_obs = float(np.mean(cos_th))
        cos_i_obs = float(sub['cos_i'].mean())

        rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'rc_over_xi': rc_xi_m,
            'rc_over_xi_sem': rc_xi_sem,
            'S_global': s_global,
            'sigma_MT': sig_mt,
            'kappa_vm': k_vm,
            'kappa1': k1,
            'kappa1_lo': float(np.percentile(b_k1, 2.5)),
            'kappa1_hi': float(np.percentile(b_k1, 97.5)),
            'upin': upin,
            'upin_lo': float(np.percentile(b_up, 2.5)),
            'upin_hi': float(np.percentile(b_up, 97.5)),
            'delta_AIC': d_aic,
            'delta_BIC': d_bic,
            'lambda0_abs': lam0,
            'lambda0_lo': float(np.percentile(b_lam, 2.5)),
            'lambda0_hi': float(np.percentile(b_lam, 97.5)),
            'tau_relax': tau_rel,
            'tau_relax_lo': float(np.nanpercentile(b_tau, 2.5)),
            'tau_relax_hi': float(np.nanpercentile(b_tau, 97.5)),
            'delta_U_tot': delta_u_tot,
            'phi_mean': phi_mean,
            'cos_model': cos_model,
            'cos_obs': cos_obs,
            'phi_times_cos_model': phi_mean * cos_model,
            'cos_i_obs': cos_i_obs,
        })

    df_out = pd.DataFrame(rows).sort_values('rc_over_xi')
    df_out.to_csv(CURRENT_DIR / "fig5_linear_stability_summary.csv", index=False)
    print("\n=== Fixed sigma_pin = sigma_MT (from S_global) Summary (sorted by R_c/xi) ===")
    print(df_out[['bead_name', 'rc_over_xi', 'S_global', 'sigma_MT',
                  'kappa1', 'kappa1_lo', 'kappa1_hi',
                  'upin', 'upin_lo', 'upin_hi',
                  'delta_BIC', 'lambda0_abs', 'tau_relax']].to_string(index=False))

    # ------------------------------------------------------------------
    # 3-Panel Fig 5
    # ------------------------------------------------------------------
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18.5, 5.6))

    # Panel (A): Phase velocity field F(Theta_bulk)
    th_plot = np.linspace(-np.pi, np.pi, 400)
    axA.axhline(0.0, color='#666666', linestyle='-', linewidth=0.9)
    axA.axvline(0.0, color='#888888', linestyle=':', linewidth=1.1)

    rep_beads = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads20um']
    color_map = {b['name']: b['color'] for b in BEADS_INFO}
    label_map = {b['name']: b['label'] for b in BEADS_INFO}
    for bname in rep_beads:
        r = df_out[df_out['bead_name'] == bname].iloc[0]
        f_vals = phase_velocity(th_plot, r['kappa1'], r['upin'], r['sigma_MT'])
        axA.plot(th_plot, f_vals, color=color_map[bname], linewidth=2.2,
                 label=rf"{label_map[bname]} ($\sigma_{{\mathrm{{MT}}}}={r['sigma_MT']:.2f}, |\tilde{{\lambda}}_0|={r['lambda0_abs']:.1f}$)")

    axA.plot(0.0, 0.0, marker='o', markersize=8.5, color='#111111', zorder=6, label=r'Stable node ($\Theta^*=0$)')
    axA.plot([-np.pi, np.pi], [0.0, 0.0], marker='o', markersize=7.5, markerfacecolor='white',
             markeredgecolor='#111111', markeredgewidth=1.6, linestyle='none', zorder=6, label=r'Unstable ($\Theta^*=\pm\pi$)')
    axA.annotate('', xy=(0.15, 0.0), xytext=(1.1, 0.0),
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1.6))
    axA.annotate('', xy=(-0.15, 0.0), xytext=(-1.1, 0.0),
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1.6))

    axA.set_xlim(-np.pi, np.pi)
    axA.set_ylim(-4.2, 4.2)
    axA.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    axA.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    axA.set_xlabel(r'Bulk Alignment Angle $\Theta_{\mathrm{bulk}}$ (rad)', fontsize=12.5)
    axA.set_ylabel(r'Phase Velocity $F(\Theta_{\mathrm{bulk}}) = d\Theta_{\mathrm{bulk}}/d\tau$', fontsize=12.5)
    axA.set_title(r'(A) Phase Flow $F(\Theta_{\mathrm{bulk}})$ ($\sigma_{\mathrm{pin}}=\sigma_{\mathrm{MT}}$)', fontsize=12.5, fontweight='bold')
    axA.grid(True, linestyle=':', alpha=0.6)
    axA.legend(loc='upper right', fontsize=7.8, framealpha=0.92)

    # Panel (B): Restoring stiffness |lambda_tilde_0| and relaxation time tau_relax vs R_c/xi
    axB_r = axB.twinx()
    for b_item in BEADS_INFO:
        r = df_out[df_out['bead_name'] == b_item['name']].iloc[0]
        yerr = [[max(0.0, r['lambda0_abs'] - r['lambda0_lo'])],
                [max(0.0, r['lambda0_hi'] - r['lambda0_abs'])]]
        axB.errorbar(r['rc_over_xi'], r['lambda0_abs'], yerr=yerr, xerr=r['rc_over_xi_sem'],
                     fmt=b_item['marker'], color=b_item['color'], markersize=9.0,
                     capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
                     label=b_item['label'], zorder=5)

    axB.plot(df_out['rc_over_xi'], df_out['lambda0_abs'], color='#111111', linewidth=1.8,
             label=r'Stiffness $|\tilde{\lambda}_0|$ (left)', zorder=3)
    axB_r.plot(df_out['rc_over_xi'], df_out['tau_relax'], color='#d62728', linestyle='--',
               marker='x', markersize=6.5, linewidth=1.6,
               label=r'Relaxation time $\tilde{\tau}_{\mathrm{relax}}$ (right)', zorder=2)

    axB.set_xscale('log')
    axB.set_yscale('log')
    axB_r.set_yscale('log')
    axB.set_xlim(0.04, 3.2)
    axB.set_ylim(0.15, 60.0)
    axB_r.set_ylim(1.0 / 60.0, 1.0 / 0.15)
    axB.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=12.5)
    axB.set_ylabel(r'Restoring Stiffness $|\tilde{\lambda}_0| = \kappa_1 + U_{\mathrm{pin}}/\sigma_{\mathrm{MT}}^2$', fontsize=12.0)
    axB_r.set_ylabel(r'Relaxation Time $\tilde{\tau}_{\mathrm{relax}} = |\tilde{\lambda}_0|^{-1}$', fontsize=12.0, color='#d62728')
    axB.set_title(r'(B) Stiffness Collapse vs $R_c/\xi$', fontsize=13, fontweight='bold')
    axB.grid(True, which='both', linestyle=':', alpha=0.6)
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axB_r.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=7.8, framealpha=0.92, ncol=2)

    # Panel (C): Unified scaling Phi(R_c) * <cos Theta_bulk>_model vs observed
    axC.plot(df_out['rc_over_xi'], df_out['cos_obs'], color='#1f77b4', marker='o', markersize=7.0,
             linewidth=1.8, linestyle='-', label=r'Observed $\langle\cos\Theta_{\mathrm{bulk}}\rangle$')
    axC.plot(df_out['rc_over_xi'], df_out['cos_model'], color='#1f77b4', marker='o', markersize=7.0,
             markerfacecolor='white', linewidth=1.6, linestyle='--',
             label=r'Model B $\langle\cos\Theta_{\mathrm{bulk}}\rangle_{\mathrm{model}}$')

    axC.plot(df_out['rc_over_xi'], df_out['cos_i_obs'], color='#d62728', marker='s', markersize=7.0,
             linewidth=1.8, linestyle='-', label=r'Observed $\langle\cos\theta_i\rangle_{\mathrm{disk}} = \langle\Phi\cos\Theta_{\mathrm{bulk}}\rangle$')
    axC.plot(df_out['rc_over_xi'], df_out['phi_times_cos_model'], color='#d62728', marker='s', markersize=7.0,
             markerfacecolor='white', linewidth=1.6, linestyle='--',
             label=r'Predicted $\Phi(R_c)\langle\cos\Theta_{\mathrm{bulk}}\rangle_{\mathrm{model}}$')

    axC.set_xscale('log')
    axC.set_xlim(0.04, 3.2)
    axC.set_ylim(0.0, 0.45)
    axC.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=12.5)
    axC.set_ylabel(r'Mean Alignment / Propulsion Factor', fontsize=12.5)
    axC.set_title(r'(C) Unified Scaling: $\Phi(R_c)\langle\cos\Theta_{\mathrm{bulk}}\rangle$', fontsize=13, fontweight='bold')
    axC.grid(True, which='both', linestyle=':', alpha=0.6)
    axC.legend(loc='upper right', fontsize=8.0, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "fig5_linear_stability_3panels", OUT_DIRS)
    plt.close(fig)
    print("\nSaved fig5_linear_stability_3panels.png and .svg")


if __name__ == "__main__":
    main()
