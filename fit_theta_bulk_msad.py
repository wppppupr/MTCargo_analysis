#!/usr/bin/env python3
"""
fit_theta_bulk_msad.py

相対角 Theta_bulk(t) = angle(v_cargo) - angle(<u_disk>) のアンラップ MSAD を算出し、
1フレーム目 (Delta t = 4 s, m = 1) を除外した Delta t in [8, 36] s (m = 2..9) で
  MSAD_Theta(Delta t) = c_0 + 2 * D_r^eff * Delta t
を線形フィットして、真の有効回転拡散係数 D_r^eff と角度ノイズ分散 c_0 = 2 * sigma_phi^2 を抽出する。
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths
)
from verify_coarse_grained_velocity_noise import load_exact_pipeline_frames

OUT_DIRS = [
    CURRENT_DIR / "figure" / "drive_decomposition",
    ROOT_DIR / "figure" / "drive_decomposition",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

DT_FRAME = 4.0  # s/frame


def compute_unwrapped_msad_by_cargo(df_sub: pd.DataFrame, max_lag_frames: int = 15):
    """
    粒子ごとに連続フレーム区間 (df == 1) ごとに Theta_bulk を np.unwrap し、
    各ラグ m = 1..max_lag_frames の (dTheta)^2 の和・カウントを集計する。
    """
    lags = np.arange(1, max_lag_frames + 1)
    cargo_sums = []
    cargo_cnts = []

    for cid, grp in df_sub.groupby('cargo_id'):
        grp = grp.sort_values('frame')
        f_arr = grp['frame'].to_numpy(dtype=int)
        th_arr = grp['theta_bulk'].to_numpy(dtype=float)

        # 連続フレーム区間に分割
        breaks = np.where(np.diff(f_arr) != 1)[0] + 1
        segments = np.split(th_arr, breaks)

        sums = np.zeros(len(lags), dtype=float)
        cnts = np.zeros(len(lags), dtype=float)
        for seg in segments:
            if len(seg) < 2:
                continue
            seg_u = np.unwrap(seg)
            for idx_l, m in enumerate(lags):
                if len(seg_u) > m:
                    diff_sq = (seg_u[m:] - seg_u[:-m]) ** 2
                    sums[idx_l] += np.sum(diff_sq)
                    cnts[idx_l] += len(diff_sq)

        if np.sum(cnts) > 0:
            cargo_sums.append(sums)
            cargo_cnts.append(cnts)

    cargo_sums = np.array(cargo_sums)
    cargo_cnts = np.array(cargo_cnts)
    tot_cnts = np.sum(cargo_cnts, axis=0)
    msad_obs = np.where(tot_cnts > 0, np.sum(cargo_sums, axis=0) / tot_cnts, np.nan)

    # フィット範囲: m in [2, 9] -> Delta t in [8, 36] s
    fit_mask = (lags >= 2) & (lags <= 9) & np.isfinite(msad_obs)
    t_fit = lags[fit_mask] * DT_FRAME
    y_fit = msad_obs[fit_mask]
    slope, intercept = np.polyfit(t_fit, y_fit, 1)
    dr_obs = float(slope / 2.0)
    c0_obs = float(intercept)

    # 粒子単位クラスターブートストラップ 95% CI
    n_c = len(cargo_sums)
    rng = np.random.default_rng(42)
    n_boot = 500
    b_msad = np.full((n_boot, len(lags)), np.nan)
    b_dr = np.full(n_boot, np.nan)
    b_c0 = np.full(n_boot, np.nan)
    for b in range(n_boot):
        sel = rng.integers(0, n_c, size=n_c)
        sb = np.sum(cargo_sums[sel], axis=0)
        cb = np.sum(cargo_cnts[sel], axis=0)
        mb = np.where(cb > 0, sb / cb, np.nan)
        b_msad[b] = mb
        mf = (lags >= 2) & (lags <= 9) & np.isfinite(mb)
        if np.sum(mf) >= 3:
            sl, ic = np.polyfit(lags[mf] * DT_FRAME, mb[mf], 1)
            b_dr[b] = sl / 2.0
            b_c0[b] = ic

    return {
        'lags_frames': lags,
        'lags_sec': lags * DT_FRAME,
        'msad_obs': msad_obs,
        'msad_lo': np.nanpercentile(b_msad, 2.5, axis=0),
        'msad_hi': np.nanpercentile(b_msad, 97.5, axis=0),
        'dr_eff': dr_obs,
        'dr_lo': float(np.nanpercentile(b_dr, 2.5)),
        'dr_hi': float(np.nanpercentile(b_dr, 97.5)),
        'c0': c0_obs,
        'c0_lo': float(np.nanpercentile(b_c0, 2.5)),
        'c0_hi': float(np.nanpercentile(b_c0, 97.5)),
        'n_cargos': n_c,
    }


def main():
    setup_matplotlib()
    print("=== MSAD Fit of Relative Angle Theta_bulk (Delta t in [8, 36] s, m >= 2) ===")

    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0

    df_frames = load_exact_pipeline_frames(ROOT_DIR, active_set, xi_dict)
    df_frames['xi_um'] = df_frames['exp_dir'].map(xi_dict).fillna(default_xi)
    df_frames['cargo_id'] = df_frames['exp_dir'].astype(str) + '_' + df_frames['particle'].astype(str)
    radius_map = {b['name']: b['radius_um'] for b in BEADS_INFO}
    df_frames['rc_over_xi'] = df_frames['bead_name'].map(radius_map) / df_frames['xi_um']

    cross_val = df_frames['u'].values * df_frames['cdy_px'].values - df_frames['v'].values * df_frames['cdx_px'].values
    dot_val = df_frames['u'].values * df_frames['cdx_px'].values + df_frames['v'].values * df_frames['cdy_px'].values
    df_frames['theta_bulk'] = np.arctan2(cross_val, dot_val)

    rows = []
    cache = {}
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_frames[df_frames['bead_name'] == bname]
        rc_xi_m = float(sub.drop_duplicates('cargo_id')['rc_over_xi'].mean())
        rc_xi_sem = float(sub.drop_duplicates('cargo_id')['rc_over_xi'].sem())

        res = compute_unwrapped_msad_by_cargo(sub, max_lag_frames=15)
        sig_phi = float(np.sqrt(max(0.0, res['c0']) / 2.0))
        msad_4s_extrap = res['c0'] + 2.0 * res['dr_eff'] * 4.0
        excess_4s = float(res['msad_obs'][0] - msad_4s_extrap)

        cache[bname] = {'res': res, 'rc_over_xi': rc_xi_m, 'rc_over_xi_sem': rc_xi_sem, 'sig_phi': sig_phi}
        rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'rc_over_xi': rc_xi_m,
            'rc_over_xi_sem': rc_xi_sem,
            'n_cargos': res['n_cargos'],
            'Dr_eff_rad2_s': res['dr_eff'],
            'Dr_eff_lo': res['dr_lo'],
            'Dr_eff_hi': res['dr_hi'],
            'c0_rad2': res['c0'],
            'c0_lo': res['c0_lo'],
            'c0_hi': res['c0_hi'],
            'sigma_phi_rad': sig_phi,
            'MSAD_4s_obs': float(res['msad_obs'][0]),
            'MSAD_4s_extrap': msad_4s_extrap,
            'excess_4s_rad2': excess_4s,
        })

    df_out = pd.DataFrame(rows).sort_values('rc_over_xi')
    df_out.to_csv(CURRENT_DIR / "theta_bulk_msad_dr_summary.csv", index=False)
    print("\n=== Theta_bulk MSAD Linear Fit Summary (Delta t in [8, 36] s) ===")
    print(df_out[['bead_name', 'rc_over_xi', 'Dr_eff_rad2_s', 'Dr_eff_lo', 'Dr_eff_hi',
                  'c0_rad2', 'sigma_phi_rad', 'MSAD_4s_obs', 'MSAD_4s_extrap', 'excess_4s_rad2']].to_string(index=False))

    # ------------------------------------------------------------------
    # Fig 1: 6パネル MSAD_Theta(Delta t) と m>=2 直線フィット
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    t_line = np.linspace(0.0, 42.0, 100)
    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item['name'], b_item['color'], b_item['label']
        pc = cache[bname]
        r = pc['res']

        ax.fill_between(r['lags_sec'], r['msad_lo'], r['msad_hi'], color=col, alpha=0.20, label=r'95% CI (Bootstrap)')
        # m=1 (4s) は白抜き赤枠で区別
        ax.plot(r['lags_sec'][0], r['msad_obs'][0], marker='o', markersize=7.5,
                markerfacecolor='white', markeredgecolor='#d62728', markeredgewidth=2.0,
                linestyle='none', label=r'Excluded $m=1$ ($\Delta t=4\,$s)', zorder=6)
        # m >= 2
        ax.plot(r['lags_sec'][1:], r['msad_obs'][1:], marker=b_item['marker'], markersize=5.2,
                color=col, linestyle='none', label=r'Observed $\mathrm{MSAD}_\Theta$ ($m \geq 2$)', zorder=4)

        # 直線フィット c0 + 2*Dr*t
        y_line = r['c0'] + 2.0 * r['dr_eff'] * t_line
        ax.plot(t_line, y_line, color='#111111', linestyle='-', linewidth=2.0,
                label=rf'Fit $[8,36]\,$s: $D_r^{{\mathrm{{eff}}}}={r["dr_eff"]:.3f}$, $c_0={r["c0"]:.2f}$', zorder=5)
        ax.axvspan(8.0, 36.0, color='#eeeeee', alpha=0.45, zorder=1)

        ax.set_xlim(0.0, 62.0)
        ax.set_ylim(0.0, 22.0)
        ax.set_title(rf'{lbl}  ($R_c/\xi={pc["rc_over_xi"]:.2f}$, $\sigma_\phi={pc["sig_phi"]:.2f}\,$rad)',
                     fontsize=12.5, fontweight='bold')
        if idx >= 3:
            ax.set_xlabel(r'Lag time $\Delta t$ (s)', fontsize=12.5)
        if idx % 3 == 0:
            ax.set_ylabel(r'$\mathrm{MSAD}_\Theta(\Delta t)$ ($\mathrm{rad}^2$)', fontsize=12.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=8.0, framealpha=0.92)

    fig.suptitle(r'Relative Angle $\Theta_{\mathrm{bulk}}$ MSAD Fit ($m \geq 2$, $\Delta t \in [8, 36]\,$s): $\mathrm{MSAD}_\Theta = c_0 + 2 D_r^{\mathrm{eff}}\Delta t$',
                 fontsize=15.0, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_theta_bulk_msad_fit_6panels", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: D_r^eff と c0 (sigma_phi) vs R_c/xi
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for b_item in BEADS_INFO:
        row = df_out[df_out['bead_name'] == b_item['name']].iloc[0]
        dr_err = [[max(0.0, row['Dr_eff_rad2_s'] - row['Dr_eff_lo'])],
                  [max(0.0, row['Dr_eff_hi'] - row['Dr_eff_rad2_s'])]]
        c0_err = [[max(0.0, row['c0_rad2'] - row['c0_lo'])],
                  [max(0.0, row['c0_hi'] - row['c0_rad2'])]]
        ax1.errorbar(row['rc_over_xi'], row['Dr_eff_rad2_s'], yerr=dr_err,
                     xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
                     markersize=9.0, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
                     label=b_item['label'], zorder=4)
        ax2.errorbar(row['rc_over_xi'], row['c0_rad2'], yerr=c0_err,
                     xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
                     markersize=9.0, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
                     label=b_item['label'], zorder=4)

    ax1.plot(df_out['rc_over_xi'], df_out['Dr_eff_rad2_s'], color='#222222', linewidth=1.6, alpha=0.7, zorder=2)
    ax1.set_xscale('log')
    ax1.set_xlim(0.04, 3.2)
    ax1.set_ylim(0.0, 0.75)
    ax1.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=13)
    ax1.set_ylabel(r'Effective Rotational Diffusion $D_r^{\mathrm{eff}}$ ($\mathrm{rad}^2/\mathrm{s}$)', fontsize=12.5)
    ax1.set_title(r'(a) $D_r^{\mathrm{eff}}(R_c)$ from $\mathrm{MSAD}_\Theta$ ($8\text{--}36\,$s)', fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8.8, framealpha=0.92, ncol=2)

    ax2.plot(df_out['rc_over_xi'], df_out['c0_rad2'], color='#222222', linewidth=1.6, alpha=0.7, zorder=2)
    ax2.set_xscale('log')
    ax2.set_xlim(0.04, 3.2)
    ax2.axhline(0.0, color='#888888', linestyle='--', linewidth=1.0)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=13)
    ax2.set_ylabel(r'Noise Offset $c_0 = 2\sigma_\phi^2$ ($\mathrm{rad}^2$)', fontsize=12.5)
    ax2.set_title(r'(b) Angular Noise Offset $c_0$ vs $R_c/\xi$', fontsize=13, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=8.8, framealpha=0.92, ncol=2)

    fig.tight_layout()
    save_figure(fig, "fig_dr_eff_theta_bulk_vs_rc_xi", OUT_DIRS)
    plt.close(fig)
    print("Saved fig_theta_bulk_msad_fit_6panels and fig_dr_eff_theta_bulk_vs_rc_xi.")


if __name__ == "__main__":
    main()
