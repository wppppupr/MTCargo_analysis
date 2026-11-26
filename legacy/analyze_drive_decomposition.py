#!/usr/bin/env python3
"""
analyze_drive_decomposition.py

outline_new_story.md §4 (4.1 - 4.4) に基づく駆動分解・結合ポテンシャル解析:
  4.1 駆動の分解: <u_hat>_disk = Phi(R_c) * e_bulk, 角度 Theta_bulk in [-pi, pi]
  4.2 流れ側: 貨物位置での Phi(R_c) vs R_c/xi を背景の Phi(R_w/xi) および C(r) 予測と重ねる + P(Phi)
  4.3 無条件の U(Theta_bulk) と von Mises MLE kappa vs R_c/xi (95% CI)
      および Phi で条件付けた U(Theta | Phi) と kappa(Phi) の全サイズ重ね合わせ (95% CI)
  4.4 条件付きチェック: 高 Phi ビン [0.8, 1.0] でも小粒子 (0.63, 1.18 um) の Theta=0 尖りが残るか

出力先:
  - figure/drive_decomposition/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/drive_decomposition/
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.special import i0, i1
from scipy.optimize import brentq

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO, SCALE_UM_PER_PX,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths, load_all_data,
    calc_disk_polar_order, find_experiment_dirs
)
from verify_coarse_grained_velocity_noise import load_exact_pipeline_frames

OUT_DIRS = [
    CURRENT_DIR / "figure" / "drive_decomposition",
    ROOT_DIR / "figure" / "drive_decomposition",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def fit_von_mises_mle(theta_arr: np.ndarray) -> float:
    """
    von Mises 分布 P(theta) = exp(kappa * cos(theta)) / (2*pi*I0(kappa)) の最尤推定量 kappa を求める。
    I1(kappa) / I0(kappa) = <cos(theta)> を満たす kappa >= 0 を返す。
    """
    if len(theta_arr) == 0:
        return np.nan
    r_bar = float(np.mean(np.cos(theta_arr)))
    if not np.isfinite(r_bar) or r_bar <= 0.0:
        return 0.0
    if r_bar >= 0.9999:
        return 50.0
    try:
        return float(brentq(lambda k: (i1(k) / i0(k)) - r_bar, 0.0, 50.0))
    except Exception:
        return np.nan


def cluster_bootstrap_kappa(
    theta_arr: np.ndarray,
    cargo_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
):
    """粒子 (cargo_id) 単位のクラスターブートストラップで kappa_MLE の 95% CI を算出する。"""
    k_obs = fit_von_mises_mle(theta_arr)
    uniq_c = np.unique(cargo_ids)
    n_c = len(uniq_c)
    if n_c < 2:
        return k_obs, np.nan, np.nan, n_c

    c_cos_sums = np.array([np.sum(np.cos(theta_arr[cargo_ids == cid])) for cid in uniq_c], dtype=float)
    c_counts = np.array([np.sum(cargo_ids == cid) for cid in uniq_c], dtype=float)

    rng = np.random.default_rng(seed)
    k_boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_c, size=n_c)
        tot_n = np.sum(c_counts[idx])
        if tot_n <= 0:
            k_boots[b] = np.nan
            continue
        r_bar = np.sum(c_cos_sums[idx]) / tot_n
        if r_bar <= 0.0:
            k_boots[b] = 0.0
        elif r_bar >= 0.9999:
            k_boots[b] = 50.0
        else:
            try:
                k_boots[b] = brentq(lambda k: (i1(k) / i0(k)) - r_bar, 0.0, 50.0)
            except Exception:
                k_boots[b] = np.nan

    ci_lo = float(np.nanpercentile(k_boots, 2.5))
    ci_hi = float(np.nanpercentile(k_boots, 97.5))
    return k_obs, ci_lo, ci_hi, n_c


def cluster_bootstrap_mean(vals: np.ndarray, cargo_ids: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """粒子単位のクラスターブートストラップで平均値と 95% CI を算出する。"""
    m_obs = float(np.mean(vals)) if len(vals) > 0 else np.nan
    uniq_c = np.unique(cargo_ids)
    n_c = len(uniq_c)
    if n_c < 2:
        return m_obs, np.nan, np.nan, n_c

    c_sums = np.array([np.sum(vals[cargo_ids == cid]) for cid in uniq_c], dtype=float)
    c_counts = np.array([np.sum(cargo_ids == cid) for cid in uniq_c], dtype=float)
    rng = np.random.default_rng(seed)
    m_boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_c, size=n_c)
        m_boots[b] = np.sum(c_sums[idx]) / np.sum(c_counts[idx])
    return m_obs, float(np.percentile(m_boots, 2.5)), float(np.percentile(m_boots, 97.5)), n_c


def load_background_phi_curves(root_dir: Path, xi_dict: dict, default_xi: float) -> pd.DataFrame:
    """
    §2 の背景流れの局所ポーラー秩序 Phi(R_w / xi) を local_polar_bg.zarr (および local_polar_noCargo.zarr) から読み込む。
    """
    records = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        for exp_dir in find_experiment_dirs(root_dir, bname):
            exp_name = exp_dir.name
            xi_val = xi_dict.get(exp_name, default_xi)
            if not np.isfinite(xi_val) or xi_val <= 0:
                xi_val = default_xi
            bg_path = exp_dir / "local_polar_bg.zarr"
            if bg_path.exists():
                try:
                    ds = xr.open_zarr(bg_path, consolidated=False)
                    da = ds['polar_order']
                    w_dim = next((d for d in da.dims if 'window' in d or 'sigma' in d), None)
                    if w_dim is not None:
                        w_px = ds.coords[w_dim].values.astype(float)
                        phi_m = da.mean(dim=[d for d in da.dims if d != w_dim]).compute().values.astype(float)
                        r_um = w_px * SCALE_UM_PER_PX
                        valid = np.isfinite(phi_m) & (phi_m > 0.0) & (phi_m <= 1.05)
                        for rw_um, p_val in zip(r_um[valid], phi_m[valid]):
                            records.append({
                                'source': 'bg_with_cargo',
                                'bead_name': bname,
                                'exp_dir': exp_name,
                                'rw_over_xi': float(rw_um / xi_val),
                                'phi': float(np.clip(p_val, 0.0, 1.0)),
                            })
                except Exception:
                    pass

    # コントロール実験 (MT_only 等)
    for sub in sorted(root_dir.iterdir()):
        if not sub.is_dir():
            continue
        for no_path in sub.rglob("local_polar_noCargo.zarr"):
            exp_dir = no_path.parent
            exp_name = exp_dir.name
            xi_val = xi_dict.get(exp_name, default_xi)
            if not np.isfinite(xi_val) or xi_val <= 0:
                xi_val = default_xi
            try:
                ds = xr.open_zarr(no_path, consolidated=False)
                da = ds['polar_order']
                w_dim = next((d for d in da.dims if 'window' in d or 'sigma' in d), None)
                if w_dim is not None:
                    w_px = ds.coords[w_dim].values.astype(float)
                    phi_m = da.mean(dim=[d for d in da.dims if d != w_dim]).compute().values.astype(float)
                    r_um = w_px * SCALE_UM_PER_PX
                    valid = np.isfinite(phi_m) & (phi_m > 0.0) & (phi_m <= 1.05)
                    for rw_um, p_val in zip(r_um[valid], phi_m[valid]):
                        records.append({
                            'source': 'control_no_cargo',
                            'bead_name': 'control',
                            'exp_dir': exp_name,
                            'rw_over_xi': float(rw_um / xi_val),
                            'phi': float(np.clip(p_val, 0.0, 1.0)),
                        })
            except Exception:
                pass

    return pd.DataFrame(records)


def main():
    setup_matplotlib()
    print("=== Section 4 Analysis: Drive Decomposition & Phi-Conditioned Coupling ===")

    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0

    # 1. フレームデータのロード (欠損値を補完しない厳密な Phi_mag を df_instant から取得)
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

    # 符号付き角度 Theta_bulk in [-pi, pi]
    cross_val = df['u'].values * df['cdy_px'].values - df['v'].values * df['cdx_px'].values
    dot_val = df['u'].values * df['cdx_px'].values + df['v'].values * df['cdy_px'].values
    df['theta_bulk'] = np.arctan2(cross_val, dot_val)

    print(f"Total merged frames: {len(df)} (valid Phi_mag: {df['Phi_mag'].notna().sum()})")

    # ------------------------------------------------------------------
    # 4.2 貨物位置の Phi(R_c) vs 背景 Phi(R_w/xi) & P(Phi) 分布
    # ------------------------------------------------------------------
    df_bg = load_background_phi_curves(ROOT_DIR, xi_dict, default_xi)
    print(f"Loaded background Phi(R_w/xi) points: {len(df_bg)}")

    summary_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df[df['bead_name'] == bname]
        sub_phi = sub.dropna(subset=['Phi_mag'])

        # 粒子単位の平均 R_c/xi
        p_rc = sub.drop_duplicates('cargo_id')
        rc_xi_m = float(p_rc['rc_over_xi'].mean())
        rc_xi_sem = float(p_rc['rc_over_xi'].sem()) if len(p_rc) > 1 else 0.0

        # 無条件 kappa_MLE & CI
        k_all, k_lo, k_hi, n_c_all = cluster_bootstrap_kappa(
            sub['theta_bulk'].values, sub['cargo_id'].values, n_boot=1000, seed=42
        )

        # Phi(R_c) 平均 & CI (実測 Phi_mag のみ)
        phi_m, phi_lo, phi_hi, n_c_phi = cluster_bootstrap_mean(
            sub_phi['Phi_mag'].values, sub_phi['cargo_id'].values, n_boot=1000, seed=42
        )
        frac_lt_02 = float(np.mean(sub_phi['Phi_mag'].values < 0.2)) if len(sub_phi) > 0 else np.nan

        summary_rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'rc_over_xi': rc_xi_m,
            'rc_over_xi_sem': rc_xi_sem,
            'n_frames': len(sub),
            'n_cargos': n_c_all,
            'n_frames_valid_phi': len(sub_phi),
            'n_cargos_valid_phi': n_c_phi,
            'phi_mean': phi_m,
            'phi_ci_lo': phi_lo,
            'phi_ci_hi': phi_hi,
            'frac_phi_lt_02': frac_lt_02,
            'kappa_mle': k_all,
            'kappa_ci_lo': k_lo,
            'kappa_ci_hi': k_hi,
            'mean_cos_theta': float(np.mean(np.cos(sub['theta_bulk'].values))),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values('rc_over_xi')
    df_summary.to_csv(CURRENT_DIR / "rc_xi_phi_kappa_summary.csv", index=False)
    print("\n=== Section 4.2 & 4.3 Unconditioned Summary (sorted by R_c/xi) ===")
    print(df_summary[['bead_name', 'rc_over_xi', 'phi_mean', 'frac_phi_lt_02',
                      'kappa_mle', 'kappa_ci_lo', 'kappa_ci_hi', 'mean_cos_theta']].to_string(index=False))

    # --- Fig 4.2: (a) Phi(R_c) vs R_c/xi & 背景曲線, (b) P(Phi) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # (a) Phi(R_c) vs R_c/xi
    if not df_bg.empty:
        bins_rw = np.geomspace(0.03, 3.5, 22)
        for src, lbl_s, col_s, ls_s in [
            ('bg_with_cargo', r'Background w/ cargo ($\Phi(R_w/\xi)$)', '#444444', '-'),
            ('control_no_cargo', r'Control no cargo ($\Phi(R_w/\xi)$)', '#1f77b4', '-.'),
        ]:
            sub_s = df_bg[df_bg['source'] == src]
            if sub_s.empty:
                continue
            xs, ys, yerrs = [], [], []
            for j in range(len(bins_rw) - 1):
                m_j = (sub_s['rw_over_xi'] >= bins_rw[j]) & (sub_s['rw_over_xi'] < bins_rw[j + 1])
                if np.sum(m_j) >= 2:
                    xs.append(np.sqrt(bins_rw[j] * bins_rw[j + 1]))
                    ys.append(float(sub_s.loc[m_j, 'phi'].mean()))
                    yerrs.append(float(sub_s.loc[m_j, 'phi'].std()))
            if xs:
                xs, ys, yerrs = np.array(xs), np.array(ys), np.array(yerrs)
                ax1.plot(xs, ys, color=col_s, linestyle=ls_s, linewidth=2.0, label=lbl_s, zorder=2)
                ax1.fill_between(xs, ys - yerrs, ys + yerrs, color=col_s, alpha=0.14, zorder=1)

    # C(r) = exp(-r/xi) 理論予測
    rx_grid = np.geomspace(0.04, 3.0, 80)
    phi_theory = [calc_disk_polar_order(rx, 1.0) for rx in rx_grid]
    ax1.plot(rx_grid, phi_theory, color='#d62728', linestyle='--', linewidth=2.0,
             label=r'Prediction from $C(r)=e^{-r/\xi}$', zorder=3)

    # 貨物位置の実測 Phi(R_c)
    for b_item in BEADS_INFO:
        row = df_summary[df_summary['bead_name'] == b_item['name']].iloc[0]
        yerr = [[row['phi_mean'] - row['phi_ci_lo']], [row['phi_ci_hi'] - row['phi_mean']]]
        ax1.errorbar(
            row['rc_over_xi'], row['phi_mean'], yerr=yerr,
            xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
            markersize=9.5, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
            label=rf"{b_item['label']} ($N={int(row['n_cargos_valid_phi'])}$)", zorder=5,
        )

    ax1.set_xscale('log')
    ax1.set_xlim(0.04, 3.2)
    ax1.set_ylim(0.35, 1.02)
    ax1.set_xlabel(r'Dimensionless Radius $R_c / \xi$ (or $R_w / \xi$)', fontsize=13)
    ax1.set_ylabel(r'Local Polar Order $\Phi$', fontsize=13)
    ax1.set_title(r'(a) Cargo $\Phi(R_c)$ vs Background $\Phi(R_w/\xi)$', fontsize=13.5, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.92, ncol=1)

    # (b) P(Phi) per bead size
    phi_bins = np.linspace(0.0, 1.0, 21)
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    ax2.axvline(0.2, color='#555555', linestyle='--', linewidth=1.4, label=r'$\Phi = 0.2$ threshold')
    for b_item in BEADS_INFO:
        bname = b_item['name']
        vals = df.loc[(df['bead_name'] == bname) & df['Phi_mag'].notna(), 'Phi_mag'].values
        hist, _ = np.histogram(vals, bins=phi_bins, density=True)
        ax2.plot(phi_centers, hist, marker=b_item['marker'], markersize=5.0,
                 color=b_item['color'], linewidth=2.0, label=b_item['label'])

    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel(r'Instantaneous Local Polar Order $\Phi(R_c)$', fontsize=13)
    ax2.set_ylabel(r'Probability Density $P(\Phi)$', fontsize=13)
    ax2.set_title(r'(b) Distribution $P(\Phi)$ at Cargo Position', fontsize=13.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9.0, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "fig4_2_phi_rc_over_xi_and_dist", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4.3 (無条件) U(Theta_bulk) & kappa_MLE vs R_c/xi
    # ------------------------------------------------------------------
    n_th_bins = 24
    th_edges = np.linspace(-np.pi, np.pi, n_th_bins + 1)
    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
    th_fine = np.linspace(-np.pi, np.pi, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # (a) U(Theta) = -ln(2*pi*P(Theta))
    ax1.axhline(0.0, color='#888888', linestyle='--', linewidth=1.0)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df[df['bead_name'] == bname]
        th = sub['theta_bulk'].values
        cid = sub['cargo_id'].values
        counts, _ = np.histogram(th, bins=th_edges, density=True)
        u_obs = np.where(counts > 0, -np.log(2.0 * np.pi * counts), np.nan)
        k_mle = float(df_summary.loc[df_summary['bead_name'] == bname, 'kappa_mle'].values[0])
        ax1.plot(th_centers, u_obs, marker=b_item['marker'], markersize=4.5,
                 color=b_item['color'], linewidth=1.8,
                 label=rf"{b_item['label']} ($\kappa={k_mle:.2f}$)")
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax1.set_xlabel(r'Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ (rad)', fontsize=13)
    ax1.set_ylabel(r'Effective Potential $U(\Theta) = -\ln(2\pi P(\Theta))$', fontsize=13)
    ax1.set_title(r'(a) Unconditioned Orientation Potential $U(\Theta_{\mathrm{bulk}})$', fontsize=13.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper center', fontsize=8.8, framealpha=0.92, ncol=2)

    # (b) kappa_MLE and Phi(R_c) vs R_c/xi
    ax2_r = ax2.twinx()
    for b_item in BEADS_INFO:
        row = df_summary[df_summary['bead_name'] == b_item['name']].iloc[0]
        k_err = [[row['kappa_mle'] - row['kappa_ci_lo']], [row['kappa_ci_hi'] - row['kappa_mle']]]
        ax2.errorbar(
            row['rc_over_xi'], row['kappa_mle'], yerr=k_err,
            xerr=row['rc_over_xi_sem'], fmt=b_item['marker'], color=b_item['color'],
            markersize=9.0, capsize=4, elinewidth=1.8, markeredgecolor='k', markeredgewidth=0.7,
            label=rf"{b_item['label']}", zorder=5,
        )
    ax2.plot(df_summary['rc_over_xi'], df_summary['kappa_mle'], color='#222222',
             linestyle='-', linewidth=1.6, alpha=0.7, label=r'$\kappa_{\mathrm{MLE}}$', zorder=3)
    ax2_r.plot(df_summary['rc_over_xi'], df_summary['phi_mean'], color='#7f7f7f',
               linestyle='--', marker='s', markersize=5.5, linewidth=1.6, alpha=0.8,
               label=r'Mean $\Phi(R_c)$ (right)', zorder=2)

    ax2.set_xscale('log')
    ax2.set_xlim(0.04, 3.2)
    ax2.set_ylim(0.0, 1.05)
    ax2_r.set_ylim(0.4, 1.05)
    ax2.set_xlabel(r'Dimensionless Cargo Radius $R_c / \xi$', fontsize=13)
    ax2.set_ylabel(r'Coupling Strength $\kappa_{\mathrm{MLE}}$', fontsize=13)
    ax2_r.set_ylabel(r'Mean Local Polar Order $\Phi(R_c)$', fontsize=13, color='#555555')
    ax2.set_title(r'(b) Monotonic Crossover of $\kappa$ and $\Phi(R_c)$ vs $R_c/\xi$', fontsize=13.5, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=8.5, framealpha=0.92, ncol=2)

    fig.tight_layout()
    save_figure(fig, "fig4_3_unconditioned_kappa", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4.3 (核) Phi で条件付けた U(Theta | Phi) と kappa(Phi) の全サイズ重ね合わせ
    # ------------------------------------------------------------------
    phi_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    phi_bin_labels = ['[0.0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']

    df_valid_phi = df.dropna(subset=['Phi_mag']).copy()
    df_valid_phi['phi_bin'] = pd.cut(df_valid_phi['Phi_mag'], bins=phi_edges, labels=phi_bin_labels, right=False)

    binned_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_b = df_valid_phi[df_valid_phi['bead_name'] == bname]
        for p_lbl in phi_bin_labels:
            sub_bp = sub_b[sub_b['phi_bin'] == p_lbl]
            n_f = len(sub_bp)
            n_c = sub_bp['cargo_id'].nunique()
            if n_f < 20 or n_c < 2:
                continue
            phi_center_actual = float(sub_bp['Phi_mag'].mean())
            k_val, k_lo, k_hi, _ = cluster_bootstrap_kappa(
                sub_bp['theta_bulk'].values, sub_bp['cargo_id'].values, n_boot=1000, seed=42
            )
            binned_rows.append({
                'bead_name': bname,
                'label': b_item['label'],
                'radius_um': b_item['radius_um'],
                'phi_bin': p_lbl,
                'is_low_phi': (p_lbl == '[0.0, 0.2)'),
                'phi_mean_bin': phi_center_actual,
                'n_frames': n_f,
                'n_cargos': n_c,
                'kappa_mle': k_val,
                'kappa_ci_lo': k_lo,
                'kappa_ci_hi': k_hi,
                'mean_cos_theta': float(np.mean(np.cos(sub_bp['theta_bulk'].values))),
            })

    df_binned = pd.DataFrame(binned_rows)
    df_binned.to_csv(CURRENT_DIR / "kappa_vs_phi_binned.csv", index=False)
    print("\n=== Section 4.3 Phi-Conditioned kappa(Phi) Table ===")
    print(df_binned[['bead_name', 'phi_bin', 'phi_mean_bin', 'n_frames', 'n_cargos',
                     'kappa_mle', 'kappa_ci_lo', 'kappa_ci_hi']].to_string(index=False))

    # --- Fig 4.3 (core): kappa(Phi) vs Phi 全サイズ重ね合わせ ---
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ax.axvspan(0.0, 0.2, color='#cccccc', alpha=0.30, label=r'Low-$\Phi$ regime ($\Phi < 0.2$, large flow-angle noise)')
    ax.axvline(0.2, color='#666666', linestyle='--', linewidth=1.2)

    # 比例基準線 kappa = alpha * Phi (Phi >= 0.2 の全データに対する線形回帰、原点通過)
    df_fit = df_binned[~df_binned['is_low_phi']]
    if not df_fit.empty:
        slope = float(np.sum(df_fit['phi_mean_bin'] * df_fit['kappa_mle']) / np.sum(df_fit['phi_mean_bin']**2))
        phi_line = np.linspace(0.0, 1.0, 100)
        ax.plot(phi_line, slope * phi_line, color='#222222', linestyle=':', linewidth=2.0,
                label=rf'Proportional fit ($\kappa = {slope:.2f}\,\Phi$)', zorder=2)

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        sub_hi = df_binned[(df_binned['bead_name'] == bname) & (~df_binned['is_low_phi'])].sort_values('phi_mean_bin')
        sub_lo = df_binned[(df_binned['bead_name'] == bname) & (df_binned['is_low_phi'])]

        if not sub_hi.empty:
            yerr = np.vstack([
                sub_hi['kappa_mle'].values - sub_hi['kappa_ci_lo'].values,
                sub_hi['kappa_ci_hi'].values - sub_hi['kappa_mle'].values,
            ])
            ax.errorbar(
                sub_hi['phi_mean_bin'], sub_hi['kappa_mle'], yerr=yerr,
                marker=b_item['marker'], color=b_item['color'], markersize=8.5,
                linewidth=2.0, capsize=4, elinewidth=1.6, markeredgecolor='k', markeredgewidth=0.6,
                label=b_item['label'], zorder=4 + idx,
            )
        if not sub_lo.empty:
            yerr_l = np.vstack([
                sub_lo['kappa_mle'].values - sub_lo['kappa_ci_lo'].values,
                sub_lo['kappa_ci_hi'].values - sub_lo['kappa_mle'].values,
            ])
            ax.errorbar(
                sub_lo['phi_mean_bin'], sub_lo['kappa_mle'], yerr=yerr_l,
                marker=b_item['marker'], color=b_item['color'], markersize=7.5,
                markerfacecolor='white', capsize=3.5, elinewidth=1.3, alpha=0.75, zorder=3,
            )

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel(r'Local Polar Order $\Phi(R_c)$', fontsize=13.5)
    ax.set_ylabel(r'Orientational Coupling Strength $\kappa(\Phi)$', fontsize=13.5)
    ax.set_title(r'Conditioned Coupling Depth $\kappa(\Phi)$ Across All Bead Sizes (95% CI)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "fig4_3_kappa_vs_phi_overlay", OUT_DIRS)
    plt.close(fig)

    # --- Fig 4.3 / 4.4: Phi ビンごとの U(Theta | Phi) と高 Phi での小粒子尖りチェック ---
    use_bins = ['[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), sharex=True, sharey=True)
    for idx_b, p_lbl in enumerate(use_bins):
        ax = axes.flatten()[idx_b]
        ax.axhline(0.0, color='#888888', linestyle='--', linewidth=0.9)
        for b_item in BEADS_INFO:
            bname = b_item['name']
            sub_bp = df_valid_phi[(df_valid_phi['bead_name'] == bname) & (df_valid_phi['phi_bin'] == p_lbl)]
            if len(sub_bp) < 40 or sub_bp['cargo_id'].nunique() < 2:
                continue
            counts, _ = np.histogram(sub_bp['theta_bulk'].values, bins=th_edges, density=True)
            u_obs = np.where(counts > 0, -np.log(2.0 * np.pi * counts), np.nan)
            k_val = fit_von_mises_mle(sub_bp['theta_bulk'].values)
            ax.plot(th_centers, u_obs, marker=b_item['marker'], markersize=4.2,
                    color=b_item['color'], linewidth=1.7,
                    label=rf"{b_item['label']} ($n={len(sub_bp)}, \kappa={k_val:.2f}$)")

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.6, 1.2)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_title(rf'$\Phi \in {p_lbl}$', fontsize=13, fontweight='bold')
        if idx_b >= 2:
            ax.set_xlabel(r'Signed Bulk Angle $\Theta_{\mathrm{bulk}}$ (rad)', fontsize=12.5)
        if idx_b % 2 == 0:
            ax.set_ylabel(r'$U(\Theta \mid \Phi) = -\ln(2\pi P(\Theta \mid \Phi))$', fontsize=12.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper center', fontsize=8.3, framealpha=0.92, ncol=2)

    fig.suptitle(r'Phi-Conditioned Orientation Potential $U(\Theta_{\mathrm{bulk}} \mid \Phi)$',
                 fontsize=15.5, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig4_3_potential_conditioned_on_phi", OUT_DIRS)
    plt.close(fig)

    print("\nAll Section 4 figures and CSVs saved successfully.")


if __name__ == "__main__":
    main()

