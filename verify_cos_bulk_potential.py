#!/usr/bin/env python3
"""
verify_cos_bulk_potential.py

純粋な角度の余弦 c = cos(Theta_bulk) = (v_hat . u_bar_hat) のみを用いた
確率密度分布 P(cos Theta_bulk) および有効ポテンシャル U_eff(cos Theta_bulk) の解析スクリプト。

- verify_cos_decomposition.py と完全に同一のデータセット (19,899 frames, 71 active cargos) を使用
- Phi(R_c) の乗算を一切含まない純粋な角度余弦 c in [-1, 1] に対する単位円ヤコビアン補正
  U_eff(c) = - ln( sqrt(1 - c^2) * P(c) / P(0) )
- クラスターブートストラップ (B=1000, 粒子単位) による 95% 信頼区間算出
- 極性結合強度 A (U = -A arctanh(c) + 2 kappa_2 c^2) および線形結合 kappa_1 vs R_c / xi

出力先:
  - figure/cos_bulk_analysis/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/cos_bulk_analysis/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths, load_all_data
)
from verify_coarse_grained_velocity_noise import load_exact_pipeline_frames

TH_GRID_B = np.linspace(-np.pi, np.pi, 601)
DTH_B = TH_GRID_B[1] - TH_GRID_B[0]
COS_GRID_B = np.cos(TH_GRID_B)
TH2_GRID_B = TH_GRID_B ** 2


def log_Z_model_b(k1: float, u_pin: float, sig: float) -> float:
    """モデル B の規格化定数 Z の対数"""
    exponent = k1 * COS_GRID_B + u_pin * np.exp(-0.5 * TH2_GRID_B / (sig ** 2))
    m = np.max(exponent)
    z_val = np.trapezoid(np.exp(exponent - m), dx=DTH_B)
    return float(m + np.log(z_val))


def fit_model_b_mle(th: np.ndarray):
    """
    モデル B: U(Theta) = -kappa_1 * cos(Theta) - U_pin * exp(-Theta^2 / (2*sigma_pin^2))
    の非ビン化最尤推定 (MLE)
    """
    cos_th = np.cos(th)
    th2 = th ** 2
    n = len(th)

    def nll(params):
        k1, u_pin, sig = params
        lz = log_Z_model_b(k1, u_pin, sig)
        ll = np.sum(k1 * cos_th + u_pin * np.exp(-0.5 * th2 / (sig ** 2))) - n * lz
        return -float(ll)

    best_res = None
    for sig0 in [0.18, 0.24, 0.35]:
        res = minimize(
            nll, x0=[0.4, 0.4, sig0],
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.15, 0.50)],
            method='L-BFGS-B'
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    k1_h, up_h, sig_h = float(best_res.x[0]), float(best_res.x[1]), float(best_res.x[2])
    lz_h = log_Z_model_b(k1_h, up_h, sig_h)
    return k1_h, up_h, sig_h, lz_h

OUT_DIRS = [
    CURRENT_DIR / "figure" / "cos_bulk_analysis",
    ROOT_DIR / "figure" / "cos_bulk_analysis"
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def main():
    setup_matplotlib()
    print("=== Pure Angle Coupling cos(Theta_bulk) Distribution & Potential Analysis ===")
    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    df_instant, df_window = load_all_data(ROOT_DIR, active_set)
    print(f"Extracted {len(df_instant)} instantaneous points.")

    if df_instant.empty:
        print("Error: No data loaded.")
        return

    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0
    df_instant['xi_um'] = df_instant['exp_dir'].map(xi_dict).fillna(default_xi)
    df_instant['cargo_id'] = df_instant['exp_dir'].astype(str) + '_' + df_instant['particle'].astype(str)

    # ビーズごとの R_c / xi と Phi_mean の集計
    df_window['xi_um'] = df_window['exp_dir'].map(xi_dict).fillna(default_xi)
    df_window['rc_over_xi'] = df_window['radius_um'] / df_window['xi_um']
    rc_xi_records = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_w = df_window[df_window['bead_name'] == bname]
        if not sub_w.empty:
            p_sub = sub_w.drop_duplicates('particle')
            rx_m = float(p_sub['rc_over_xi'].mean())
            rx_s = float(p_sub['rc_over_xi'].sem()) if len(p_sub) > 1 else 0.0
            phi_m = float(p_sub['Phi_mean'].dropna().mean()) if not p_sub['Phi_mean'].dropna().empty else np.nan
        else:
            rx_m, rx_s, phi_m = np.nan, 0.0, np.nan
        rc_xi_records.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'rc_over_xi_mean': rx_m,
            'rc_over_xi_sem': rx_s,
            'Phi_mean': phi_m
        })
    df_rc_xi = pd.DataFrame(rc_xi_records)

    # -------------------------------------------------------------
    # 1. ヒストグラム P(cos Theta_bulk) と U_eff(cos Theta_bulk) の計算
    # -------------------------------------------------------------
    bins_cos = np.linspace(-1.0, 1.0, 29)  # 28 bins
    bin_centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])
    dc_bin = bin_centers[1] - bin_centers[0]
    idx_zero = int(np.argmin(np.abs(bin_centers)))
    dth_exact = np.arccos(bins_cos[:-1]) - np.arccos(bins_cos[1:])
    denom_jacob = dc_bin / dth_exact
    p_iso_k = dth_exact / (np.pi * dc_bin)

    B_BOOT = 1000
    rng_boot = np.random.default_rng(42)

    boot_dict = {}
    param_records = []

    def estimate_params(p_obs):
        p_neg = p_obs[::-1]
        p_even = 0.5 * (p_obs + p_neg)
        p_odd = 0.5 * (p_obs - p_neg)
        mask = (bin_centers > 0.05) & (bin_centers <= 0.85) & (p_even > 1e-4)
        if np.sum(mask) < 2:
            return np.nan, np.nan, np.nan
        c_m = bin_centers[mask]
        ratio_m = np.clip((p_odd / p_even)[mask], -0.999, 0.999)
        atanh_ratio = np.arctanh(ratio_m)

        # (a) 線形モデル: atanh(P_odd/P_even) = kappa_1 * c
        k1_est = max(0.0, float(np.sum(atanh_ratio * c_m) / np.sum(c_m**2)))

        # (b) arctanh モデル: atanh(P_odd/P_even) = A * arctanh(c)
        atanh_c = np.arctanh(c_m)
        A_est = max(0.0, float(np.sum(atanh_ratio * atanh_c) / np.sum(atanh_c**2)))

        # (c) kappa_2 推定
        y_vals = np.log(np.clip(denom_jacob[mask] * p_even[mask] / np.cosh(A_est * atanh_c), 1e-6, None))
        x_sq = c_m**2
        x_dev = x_sq - np.mean(x_sq)
        y_dev = y_vals - np.mean(y_vals)
        slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2) if np.sum(x_dev**2) > 0 else 0.0
        k2_est = max(0.0, float(-0.5 * slope))
        return A_est, k1_est, k2_est

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname]
        if sub.empty:
            continue

        cb_all = np.clip(sub['cos_bulk'].dropna().values, -1.0, 1.0)
        P_obs, _ = np.histogram(cb_all, bins=bins_cos, density=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            arg_u = P_obs / p_iso_k
            U_eff_obs = np.where(arg_u > 0, -np.log(arg_u), np.nan)

        A_obs, k1_obs, k2_obs = estimate_params(P_obs)
        p_diff_obs = float((np.sum(P_obs[bin_centers > 0]) - np.sum(P_obs[bin_centers < 0])) * dc_bin)

        # クラスターブートストラップ
        cargos_data = [np.clip(grp['cos_bulk'].dropna().values, -1.0, 1.0)
                       for _, grp in sub.groupby('cargo_id') if len(grp) > 0]
        n_cargos = len(cargos_data)

        boot_P = np.zeros((B_BOOT, len(bin_centers)))
        boot_U = np.zeros((B_BOOT, len(bin_centers)))
        boot_A, boot_k1, boot_k2, boot_pdiff = [], [], [], []

        if n_cargos >= 2:
            for b_idx in range(B_BOOT):
                sel = rng_boot.choice(n_cargos, size=n_cargos, replace=True)
                pts = np.concatenate([cargos_data[i] for i in sel])
                pb, _ = np.histogram(pts, bins=bins_cos, density=True)
                boot_P[b_idx] = pb

                with np.errstate(divide='ignore', invalid='ignore'):
                    au_b = pb / p_iso_k
                    boot_U[b_idx] = np.where(au_b > 0, -np.log(au_b), np.nan)

                ab, k1b, k2b = estimate_params(pb)
                boot_A.append(ab)
                boot_k1.append(k1b)
                boot_k2.append(k2b)
                boot_pdiff.append(float((np.sum(pb[bin_centers > 0]) - np.sum(pb[bin_centers < 0])) * dc_bin))

            ci_low_p = np.percentile(boot_P, 2.5, axis=0)
            ci_high_p = np.percentile(boot_P, 97.5, axis=0)
            ci_low_u = np.nanpercentile(boot_U, 2.5, axis=0)
            ci_high_u = np.nanpercentile(boot_U, 97.5, axis=0)
            A_low, A_high = np.nanpercentile(boot_A, [2.5, 97.5])
            k1_low, k1_high = np.nanpercentile(boot_k1, [2.5, 97.5])
            k2_low, k2_high = np.nanpercentile(boot_k2, [2.5, 97.5])
            pd_low, pd_high = np.nanpercentile(boot_pdiff, [2.5, 97.5])
        else:
            ci_low_p, ci_high_p = P_obs, P_obs
            ci_low_u, ci_high_u = U_eff_obs, U_eff_obs
            A_low, A_high = A_obs, A_obs
            k1_low, k1_high = k1_obs, k1_obs
            k2_low, k2_high = k2_obs, k2_obs
            pd_low, pd_high = p_diff_obs, p_diff_obs

        boot_dict[bname] = {
            'P_obs': P_obs, 'ci_low_p': ci_low_p, 'ci_high_p': ci_high_p,
            'U_eff_obs': U_eff_obs, 'ci_low_u': ci_low_u, 'ci_high_u': ci_high_u,
            'A': A_obs, 'kappa_1': k1_obs, 'kappa_2': k2_obs,
            'mean_val': float(np.mean(cb_all)),
            'c2_mean': float(np.mean(cb_all**2)),
            'n_cargos': n_cargos, 'n_points': len(cb_all),
        }

        row_rc = df_rc_xi[df_rc_xi['bead_name'] == bname]
        rx_m = float(row_rc['rc_over_xi_mean'].values[0]) if not row_rc.empty else np.nan
        rx_s = float(row_rc['rc_over_xi_sem'].values[0]) if not row_rc.empty else 0.0

        param_records.append({
            'bead_name': bname, 'label': b_item['label'],
            'color': b_item['color'], 'marker': b_item['marker'],
            'rc_over_xi_mean': rx_m, 'rc_over_xi_sem': rx_s,
            'mean_cos_bulk': float(np.mean(cb_all)),
            'mean_cos2_bulk': float(np.mean(cb_all**2)),
            'A': A_obs, 'A_ci_low': float(A_low), 'A_ci_high': float(A_high),
            'kappa_1': k1_obs, 'k1_ci_low': float(k1_low), 'k1_ci_high': float(k1_high),
            'kappa_2': k2_obs, 'k2_ci_low': float(k2_low), 'k2_ci_high': float(k2_high),
            'p_diff': p_diff_obs, 'pdiff_ci_low': float(pd_low), 'pdiff_ci_high': float(pd_high),
        })

    df_params = pd.DataFrame(param_records).sort_values('rc_over_xi_mean').reset_index(drop=True)
    for d in OUT_DIRS:
        df_params.to_csv(d / "cos_bulk_potential_params_summary.csv", index=False)
    print("\n=== cos(Theta_bulk) Parameters Summary ===")
    print(df_params[['bead_name', 'rc_over_xi_mean', 'mean_cos_bulk', 'mean_cos2_bulk', 'A', 'kappa_1', 'kappa_2']].to_string(index=False))

    # -------------------------------------------------------------
    # 2. 確率密度分布 P(cos Theta_bulk) (6パネル & 1パネル)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        ax.plot(bin_centers, res['P_obs'], color=col, linewidth=2.4, marker=b_item['marker'], markersize=5,
                label=rf"Observed ($N={res['n_points']}$, $N_{{\mathrm{{cargo}}}}={res['n_cargos']}$)", zorder=3)
        ax.fill_between(bin_centers, res['ci_low_p'], res['ci_high_p'], color=col, alpha=0.25,
                        label=r"95% CI (Cluster Bootstrap)", zorder=2)

        # 等方分布の参照線: P_iso(c) = 1 / (pi * sqrt(1 - c^2))
        c_iso = np.linspace(-0.96, 0.96, 200)
        p_iso = 1.0 / (np.pi * np.sqrt(1.0 - c_iso**2))
        ax.plot(c_iso, p_iso, color='#666666', linestyle=':', linewidth=1.4, label=r"Isotropic $1/(\pi\sqrt{1-c^2})$", zorder=1)

        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.0, zorder=1)
        ax.axvline(res['mean_val'], color=col, linestyle='--', linewidth=1.8,
                   label=rf"Mean $\cos\Theta_{{\mathrm{{bulk}}}} = {res['mean_val']:.2f}$", zorder=2)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0.0, 3.0)
        ax.set_title(rf"{lbl}  ($\langle \cos^2\Theta_{{\mathrm{{bulk}}}} \rangle = {res['c2_mean']:.2f}$)", fontsize=13.5, fontweight='bold')
        ax.set_xlabel(r"Bulk Angle Cosine $c = \cos\Theta_{\mathrm{bulk}}$", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\cos\Theta_{\mathrm{bulk}})$", fontsize=13.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Probability Density of Pure Bulk Alignment $P(\cos\Theta_{\mathrm{bulk}})$ (Cluster Bootstrap 95% CI)',
                 fontsize=16.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_bulk_probability_density_6panels", OUT_DIRS)
    plt.close(fig)

    # 1パネル版
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    c_iso = np.linspace(-0.96, 0.96, 200)
    ax.plot(c_iso, 1.0 / (np.pi * np.sqrt(1.0 - c_iso**2)), color='#555555', linestyle='--', linewidth=1.5,
            label=r"Isotropic 2D $1/(\pi\sqrt{1-c^2})$", zorder=1)
    ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
    for idx, b_item in enumerate(BEADS_INFO):
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        ax.plot(bin_centers, res['P_obs'], marker=b_item['marker'], markersize=6, color=col,
                linewidth=2.2, label=rf"{lbl} ($N={res['n_cargos']}$)", zorder=3 + idx)
        ax.fill_between(bin_centers, res['ci_low_p'], res['ci_high_p'], color=col, alpha=0.14, zorder=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(r"Bulk Angle Cosine $c = \cos\Theta_{\mathrm{bulk}}$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Probability Density $P(\cos\Theta_{\mathrm{bulk}})$", fontsize=14, labelpad=8)
    ax.set_title(r"Probability Density of Pure Bulk Alignment $P(\cos\Theta_{\mathrm{bulk}})$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "cos_bulk_probability_density_1panel", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. 有効ポテンシャル U_eff(cos Theta_bulk) (6パネル & 1パネル)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    c_fine = np.linspace(-0.94, 0.94, 300)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        u_obs = res['U_eff_obs']
        valid = ~np.isnan(u_obs)

        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)
        ax.plot(bin_centers[valid], u_obs[valid], color=col, linewidth=2.2,
                marker=b_item['marker'], markersize=6.0, label=r"Observed $U_{\mathrm{eff}}(\cos\Theta_{\mathrm{bulk}})$", zorder=4)
        ax.fill_between(bin_centers[valid], res['ci_low_u'][valid], res['ci_high_u'][valid],
                        color=col, alpha=0.22, label=r"95% CI (Bootstrap)", zorder=2)

        # arctanh フィット線: U = -A arctanh(c) + 2 kappa_2 c^2
        a_f, k2_f = res['A'], res['kappa_2']
        if np.isfinite(a_f) and np.isfinite(k2_f):
            u_fit = -a_f * np.arctanh(c_fine) + 2.0 * k2_f * (c_fine**2)
            ax.plot(c_fine, u_fit, color='#222222', linestyle='--', linewidth=2.0,
                    label=rf"Fit: $A={a_f:.2f}, \kappa_2={k2_f:.2f}$", zorder=5)

        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-2.5, 2.0)
        ax.set_title(rf"{lbl}  ($A={a_f:.2f}, \kappa_2={k2_f:.2f}$)", fontsize=13.5, fontweight='bold')
        ax.set_xlabel(r"Bulk Angle Cosine $c = \cos\Theta_{\mathrm{bulk}}$", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $U_{\mathrm{eff}}(c) / k_B T$", fontsize=13.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=9.0, framealpha=0.92)

    fig.suptitle(r'Effective Potential from Pure Bulk Alignment: $\tilde{U}_{\mathrm{eff}}(c) = -\ln(P(c)/P_{\mathrm{iso}}(c)) = -\ln(\pi\sqrt{1-c^2}\,P(c))$',
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_bulk_effective_potential_6panels", OUT_DIRS)
    plt.close(fig)

    # 1パネル版
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.axhline(0.0, color='#888888', linestyle='-', linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)
    for idx, b_item in enumerate(BEADS_INFO):
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        u_obs = res['U_eff_obs']
        valid = ~np.isnan(u_obs)
        ax.plot(bin_centers[valid], u_obs[valid], marker=b_item['marker'], markersize=5.5, color=col,
                linewidth=2.0, label=rf"{lbl} ($A={res['A']:.2f}, \kappa_2={res['kappa_2']:.2f}$)", zorder=3 + idx)
        ax.fill_between(bin_centers[valid], res['ci_low_u'][valid], res['ci_high_u'][valid], color=col, alpha=0.12, zorder=2)

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel(r"Bulk Angle Cosine $c = \cos\Theta_{\mathrm{bulk}}$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Effective Potential $U_{\mathrm{eff}}(c) / k_B T$", fontsize=14, labelpad=8)
    ax.set_title(r"Effective Potential Profiles of Pure Bulk Alignment $U_{\mathrm{eff}}(\cos\Theta_{\mathrm{bulk}})$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower left', fontsize=10.0, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "cos_bulk_effective_potential_1panel", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. 符号付き角度空間 Theta_bulk in [-pi, pi] での P(Theta_bulk) と U_eff(Theta_bulk)
    #    - Theta_bulk = arctan2(u * cdy - v * cdx, u * cdx + v * cdy) in [-pi, pi]
    #    - 0 が順流方向（中央）、+/- pi/2 が直交、+/- pi が逆流
    #    - 等方分布 = 1 / (2 * pi) の平坦な一様分布
    #    - U_eff(Theta) = -ln(P(Theta) / P(0)) (順流方向 Theta = 0 を基準 U=0)
    # -------------------------------------------------------------
    df_frames = load_exact_pipeline_frames(ROOT_DIR, active_set, xi_dict)
    cross_val = df_frames['u'].values * df_frames['cdy_px'].values - df_frames['v'].values * df_frames['cdx_px'].values
    dot_val = df_frames['u'].values * df_frames['cdx_px'].values + df_frames['v'].values * df_frames['cdy_px'].values
    df_frames['theta_signed'] = np.arctan2(cross_val, dot_val)

    bins_th = np.linspace(-np.pi, np.pi, 37)  # 36 bins (10 deg each)
    bin_centers_th = 0.5 * (bins_th[:-1] + bins_th[1:])
    idx_zero_th = int(np.argmin(np.abs(bin_centers_th)))

    th_boot_dict = {}
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_frames[df_frames['bead_name'] == bname]
        if sub.empty:
            continue
        th_all = sub['theta_signed'].dropna().values
        P_th_obs, _ = np.histogram(th_all, bins=bins_th, density=True)

        # 等方分布 P_iso = 1 / (2*pi) を基準として U_eff(Theta) = -ln(2*pi * P(Theta))
        # これにより中央 Theta=0 を含む全ビンで 95% CI が正しく表示される
        p_iso_val = 1.0 / (2.0 * np.pi)
        with np.errstate(divide='ignore', invalid='ignore'):
            arg_u = P_th_obs / p_iso_val
            U_th_obs = np.where(arg_u > 0, -np.log(arg_u), np.nan)

        cargos_th = [grp['theta_signed'].dropna().values for _, grp in sub.groupby('cargo_id') if len(grp) > 0]
        n_cargos = len(cargos_th)
        if n_cargos >= 2:
            b_P = np.zeros((B_BOOT, len(bin_centers_th)))
            b_U = np.zeros((B_BOOT, len(bin_centers_th)))
            for b_idx in range(B_BOOT):
                sel = rng_boot.choice(n_cargos, size=n_cargos, replace=True)
                pts = np.concatenate([cargos_th[i] for i in sel])
                pth_b, _ = np.histogram(pts, bins=bins_th, density=True)
                b_P[b_idx] = pth_b
                with np.errstate(divide='ignore', invalid='ignore'):
                    au_b = pth_b / p_iso_val
                    b_U[b_idx] = np.where(au_b > 0, -np.log(au_b), np.nan)
            ci_low_p = np.percentile(b_P, 2.5, axis=0)
            ci_high_p = np.percentile(b_P, 97.5, axis=0)
            ci_low_u = np.nanpercentile(b_U, 2.5, axis=0)
            ci_high_u = np.nanpercentile(b_U, 97.5, axis=0)
        else:
            ci_low_p, ci_high_p = P_th_obs, P_th_obs
            ci_low_u, ci_high_u = U_th_obs, U_th_obs

        # モデル B (バルク整列 + 局所レール・ピンニング井戸) の MLE フィット
        k1_b, upin_b, sig_b, lz_b = fit_model_b_mle(th_all)
        u0_b = lz_b - np.log(2.0 * np.pi)

        th_boot_dict[bname] = {
            'P_obs': P_th_obs, 'ci_low_p': ci_low_p, 'ci_high_p': ci_high_p,
            'U_obs': U_th_obs, 'ci_low_u': ci_low_u, 'ci_high_u': ci_high_u,
            'k1_b': k1_b, 'upin_b': upin_b, 'sig_b': sig_b, 'lz_b': lz_b, 'u0_b': u0_b,
            'n_cargos': n_cargos, 'n_points': len(th_all),
        }

    # (4a) P(Theta_bulk) 6パネル版 [-pi, pi]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in th_boot_dict:
            continue
        res = th_boot_dict[bname]
        ax.axhline(1.0 / (2.0 * np.pi), color='#666666', linestyle=':', linewidth=1.4, label=r"Isotropic $1/(2\pi)$", zorder=1)
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axvline(-np.pi / 2.0, color='#bbbbbb', linestyle=':', linewidth=0.9, zorder=1)
        ax.axvline(np.pi / 2.0, color='#bbbbbb', linestyle=':', linewidth=0.9, zorder=1)
        ax.plot(bin_centers_th, res['P_obs'], color=col, linewidth=2.4, marker=b_item['marker'], markersize=4.5,
                label=rf"Observed ($N={res['n_points']}$)", zorder=3)
        ax.fill_between(bin_centers_th, res['ci_low_p'], res['ci_high_p'], color=col, alpha=0.25,
                        label=r"95% CI (Bootstrap)", zorder=2)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0.0, 0.70)
        ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_title(rf"{lbl}", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ [rad]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\Theta_{\mathrm{bulk}})$ [$\mathrm{rad}^{-1}$]", fontsize=13.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)
    fig.suptitle(r'Probability Density of Signed Bulk Angle $P(\Theta_{\mathrm{bulk}})$ on $[-\pi, \pi]$',
                 fontsize=16.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "theta_bulk_probability_density_6panels", OUT_DIRS)
    plt.close(fig)

    # (4b) P(Theta_bulk) 1パネル版 [-pi, pi]
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.axhline(1.0 / (2.0 * np.pi), color='#555555', linestyle='--', linewidth=1.5, label=r"Isotropic $1/(2\pi)$", zorder=1)
    ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
    for idx, b_item in enumerate(BEADS_INFO):
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in th_boot_dict:
            continue
        res = th_boot_dict[bname]
        ax.plot(bin_centers_th, res['P_obs'], marker=b_item['marker'], markersize=5.5, color=col,
                linewidth=2.2, label=rf"{lbl} ($N={res['n_cargos']}$)", zorder=3 + idx)
        ax.fill_between(bin_centers_th, res['ci_low_p'], res['ci_high_p'], color=col, alpha=0.14, zorder=2)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0.0, 0.65)
    ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$ (Back)", r"$-\pi/2$ (Ortho)", r"$0$ (Forward)", r"$\pi/2$ (Ortho)", r"$\pi$ (Back)"])
    ax.set_xlabel(r"Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ [rad]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Probability Density $P(\Theta_{\mathrm{bulk}})$ [$\mathrm{rad}^{-1}$]", fontsize=14, labelpad=8)
    ax.set_title(r"Probability Density of Signed Bulk Angle $P(\Theta_{\mathrm{bulk}})$ on $[-\pi, \pi]$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10.2, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "theta_bulk_probability_density_1panel", OUT_DIRS)
    plt.close(fig)

    # (4c) U_eff(Theta_bulk) 6パネル版 [-pi, pi]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    th_fine = np.linspace(-np.pi, np.pi, 300)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in th_boot_dict:
            continue
        res = th_boot_dict[bname]
        u_obs = res['U_obs']
        valid = ~np.isnan(u_obs)
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)
        ax.plot(bin_centers_th[valid], u_obs[valid], color=col, linewidth=2.2,
                marker=b_item['marker'], markersize=5.0, label=r"Observed $U_{\mathrm{eff}}(\Theta_{\mathrm{bulk}})$", zorder=4)
        ax.fill_between(bin_centers_th[valid], res['ci_low_u'][valid], res['ci_high_u'][valid],
                        color=col, alpha=0.22, label=r"95% CI (Bootstrap)", zorder=2)

        k1_b, upin_b, sig_b, u0_b = res['k1_b'], res['upin_b'], res['sig_b'], res['u0_b']
        u_model_b = u0_b - k1_b * np.cos(th_fine) - upin_b * np.exp(-0.5 * (th_fine ** 2) / (sig_b ** 2))
        ax.plot(th_fine, u_model_b, color='#111111', linestyle='-', linewidth=2.2,
                label=rf"Model B ($\kappa_1={k1_b:.2f}, U_{{\mathrm{{pin}}}}={upin_b:.2f}, \sigma={sig_b:.2f}$)", zorder=5)
        if upin_b > 0.05:
            u_bulk_only = u0_b - k1_b * np.cos(th_fine)
            ax.plot(th_fine, u_bulk_only, color='#d62728', linestyle=':', linewidth=1.6,
                    label=r"Bulk term ($-\kappa_1\cos\Theta$)", zorder=4)

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.6, 1.4)
        ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_title(rf"{lbl}  ($\kappa_1={k1_b:.2f}, U_{{\mathrm{{pin}}}}={upin_b:.2f}$)", fontsize=13.5, fontweight='bold')
        ax.set_xlabel(r"Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ [rad]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $U_{\mathrm{eff}}(\Theta_{\mathrm{bulk}}) / k_B T$", fontsize=13.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower center', fontsize=8.5, framealpha=0.92)
    fig.suptitle(r'Effective Potential of Signed Bulk Angle: $U_{\mathrm{eff}}(\Theta_{\mathrm{bulk}}) = -\ln(2\pi P(\Theta_{\mathrm{bulk}}))$',
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "theta_bulk_effective_potential_6panels", OUT_DIRS)
    plt.close(fig)

    # (4d) U_eff(Theta_bulk) 1パネル版 [-pi, pi]
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.axhline(0.0, color='#888888', linestyle='-', linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)
    for idx, b_item in enumerate(BEADS_INFO):
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        if bname not in th_boot_dict:
            continue
        res = th_boot_dict[bname]
        u_obs = res['U_obs']
        valid = ~np.isnan(u_obs)
        ax.plot(bin_centers_th[valid], u_obs[valid], marker=b_item['marker'], markersize=5.0, color=col,
                linewidth=2.0, label=rf"{lbl} ($\kappa_1={res['k1_b']:.2f}, U_{{\mathrm{{pin}}}}={res['upin_b']:.2f}$)", zorder=3 + idx)
        ax.fill_between(bin_centers_th[valid], res['ci_low_u'][valid], res['ci_high_u'][valid], color=col, alpha=0.12, zorder=2)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.6, 1.4)
    ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$ (Back)", r"$-\pi/2$ (Ortho)", r"$0$ (Forward)", r"$\pi/2$ (Ortho)", r"$\pi$ (Back)"])
    ax.set_xlabel(r"Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ [rad]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Effective Potential $U_{\mathrm{eff}}(\Theta_{\mathrm{bulk}}) / k_B T$", fontsize=14, labelpad=8)
    ax.set_title(r"Effective Potential Profiles $U_{\mathrm{eff}}(\Theta_{\mathrm{bulk}}) = -\ln(2\pi P(\Theta_{\mathrm{bulk}}))$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower center', fontsize=9.8, ncol=2, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "theta_bulk_effective_potential_1panel", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 5. 小さい |Theta_bulk| 側の両対数プロット:
    #    べき則 P(|Theta|) ~ |Theta|^{-A} (arctanh由来) か
    #    滑らか・飽和 (von Mises) かを検証する
    # -------------------------------------------------------------
    bins_abs_log = np.logspace(np.log10(0.015), np.log10(np.pi), 22)
    centers_abs_log = np.sqrt(bins_abs_log[:-1] * bins_abs_log[1:])

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        sub = df_frames[df_frames['bead_name'] == bname]
        if sub.empty:
            continue
        abs_th = np.abs(sub['theta_signed'].dropna().values)
        p_log_obs, _ = np.histogram(abs_th, bins=bins_abs_log, density=True)

        # クラスターブートストラップ 95% CI
        cargos_abs = [np.abs(grp['theta_signed'].dropna().values) for _, grp in sub.groupby('cargo_id') if len(grp) > 0]
        n_c = len(cargos_abs)
        if n_c >= 2:
            b_pl = np.zeros((500, len(centers_abs_log)))
            for bi in range(500):
                sel = rng_boot.choice(n_c, size=n_c, replace=True)
                pts = np.concatenate([cargos_abs[i] for i in sel])
                b_pl[bi], _ = np.histogram(pts, bins=bins_abs_log, density=True)
            ci_l = np.percentile(b_pl, 2.5, axis=0)
            ci_h = np.percentile(b_pl, 97.5, axis=0)
        else:
            ci_l, ci_h = p_log_obs, p_log_obs

        valid = p_log_obs > 0
        ax.loglog(centers_abs_log[valid], p_log_obs[valid], color=col, marker=b_item['marker'],
                  markersize=5.5, linewidth=2.2, label=r"Observed $P(|\Theta_{\mathrm{bulk}}|)$", zorder=4)
        ax.fill_between(centers_abs_log[valid], np.maximum(ci_l[valid], 1e-4), np.maximum(ci_h[valid], 1e-4),
                        color=col, alpha=0.22, label=r"95% CI", zorder=2)

        # モデル B 曲線 (全体分布からフィットした kappa_1, U_pin, sigma_pin)
        if bname in th_boot_dict:
            rb = th_boot_dict[bname]
            k1_b, upin_b, sig_b, lz_b = rb['k1_b'], rb['upin_b'], rb['sig_b'], rb['lz_b']
            th_f = np.logspace(np.log10(0.015), np.log10(np.pi), 200)
            # 折り返した角度 [0, pi] の密度なので 2 / Z * exp(...)
            p_b_curve = 2.0 * np.exp(k1_b * np.cos(th_f) + upin_b * np.exp(-0.5 * (th_f ** 2) / (sig_b ** 2)) - lz_b)
            ax.loglog(th_f, p_b_curve, color='#222222', linestyle='--', linewidth=1.8,
                      label=rf"Model B ($\kappa_1={k1_b:.2f}, U_{{\mathrm{{pin}}}}={upin_b:.2f}$)", zorder=3)

        # べき則参照線 (arctanh モデルの A に相当する傾き -A)
        if bname in boot_dict and np.isfinite(boot_dict[bname]['A']):
            A_val = boot_dict[bname]['A']
            # th = 0.2 rad 付近の実測値に高さを合わせた参照線
            idx_ref = int(np.argmin(np.abs(centers_abs_log - 0.2)))
            if p_log_obs[idx_ref] > 0:
                th_p = np.logspace(np.log10(0.015), np.log10(0.8), 100)
                p_pow = p_log_obs[idx_ref] * (th_p / centers_abs_log[idx_ref])**(-A_val)
                ax.loglog(th_p, p_pow, color='crimson', linestyle=':', linewidth=2.0,
                          label=rf"Power law $\propto |\Theta|^{{-{A_val:.2f}}}$", zorder=5)

        ax.set_xlim(0.012, np.pi)
        ax.set_ylim(0.05, 5.0)
        ax.set_title(rf"{lbl}", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Absolute Bulk Angle $|\Theta_{\mathrm{bulk}}|$ [rad]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(|\Theta_{\mathrm{bulk}}|)$", fontsize=13.5)
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=8.5, framealpha=0.92)

    fig.suptitle(r'Small-Angle Log-Log Test: Power-Law Divergence ($|\Theta|^{-A}$) vs Model B',
                 fontsize=15.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "theta_bulk_small_angle_loglog_6panels", OUT_DIRS)
    plt.close(fig)


if __name__ == "__main__":
    main()

