#!/usr/bin/env python3
"""
plot_tau_oacf_theory.py

配向持続時間 (Orientation Persistence Time) tau_p = tau_OACF vs スケール半径 x = R_c / xi の
スケーリングマスターカーブおよび理論モデル可視化スクリプト。

理論式:
    tau_p(x) = tau_0 * exp(-4/3 * x) = tau_0 * exp(-4 * R_c / (3 * xi))
    1 / tau_p(x) = (1 / tau_0) * exp(4/3 * x)

各粒子 i の局所相関長 xi_i を angular_correlation_w.zarr から直接フィッティングし、
各粒子固有の無次元スケール半径 x_i = R_c / xi_i と配向相関時間 tau_{p, i} をプロット。
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import hmm_cargo as hc

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
    except Exception:
        pass

# 2状態粒子 (0.63, 1.18, 3.37 um)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": "#882255", "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": "#CC6677", "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": "#DDCC77", "label": r"$3.37\,\mu\mathrm{m}$"},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um']:
                if (r / b).exists():
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def exp_decay_model(r, xi, a=1.0):
    return a * np.exp(-r / xi)


def fit_local_xi(r_um, c_curve, min_r, max_r=25.0):
    """局所相関関数 C(r) から相関長 xi をフィッティング推定する。"""
    mask = (r_um >= min_r) & (r_um <= max_r) & np.isfinite(c_curve) & (c_curve > 0.05)
    if np.sum(mask) < 4:
        return np.nan
    r_fit = r_um[mask]
    c_fit = c_curve[mask]
    try:
        popt, _ = curve_fit(exp_decay_model, r_fit, c_fit, p0=[5.0, 1.0], bounds=([0.1, 0.3], [50.0, 1.5]), maxfev=400)
        return float(popt[0])
    except Exception:
        try:
            slope, _ = np.polyfit(r_fit, np.log(c_fit), 1)
            if slope < -0.01:
                xi_val = -1.0 / slope
                if 0.1 <= xi_val <= 50.0:
                    return float(xi_val)
        except Exception:
            pass
        return np.nan


def extract_particle_tau_and_xi(root_dir: Path) -> pd.DataFrame:
    """
    全粒子の個別 (x_i = R_c / xi_i, tau_{p, i}) ペアを直接抽出する。
    """
    records = []
    print("Extracting particle-level local correlation length xi_i and orientation time tau_p,i...")

    for b in BEADS_INFO:
        bname = b['name']
        dia = b['diameter_um']
        rc = b['radius_um']
        base = root_dir / bname
        edirs = [p for p in sorted(base.glob('*/*')) if p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / 'angular_correlation_w.zarr').exists()]
        if not edirs:
            edirs = [p for p in sorted(base.glob('*')) if p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / 'angular_correlation_w.zarr').exists()]

        if not edirs:
            continue

        # 2状態 HMM の学習 (Run / Tumble 分離)
        all_tracks = []
        p_offset = 0
        for edir in edirs:
            df_t = pd.read_csv(edir / 'beads_tracks.csv')
            df_t['particle'] = df_t['particle'] + p_offset
            p_offset += int(df_t['particle'].max()) + 1
            all_tracks.append(df_t)
        df_all_tracks = pd.concat(all_tracks, ignore_index=True)
        X, lengths, _ = hc.extract_hmm_features(df_all_tracks, tau=1, scale=0.11, frame_interval=4.0, epsilon=1e-3)
        hmm_model = hc.CargoGaussianHMM(n_components=2, covariance_type='full', epsilon=1e-3, random_state=42)
        hmm_model.fit(X, lengths=lengths)

        min_r = rc * 1.1

        for edir in edirs:
            fov = edir.name
            df_t = pd.read_csv(edir / 'beads_tracks.csv')
            X_fov, lengths_fov, df_obs_fov = hc.extract_hmm_features(df_t, tau=1, scale=0.11, frame_interval=4.0, epsilon=1e-3)
            if len(X_fov) > 0:
                states = hmm_model.predict(X_fov, lengths=lengths_fov)
                df_obs_fov['state'] = states
            else:
                df_obs_fov['state'] = 1

            # Zarr から空間相関プロファイルを読み込み
            ds_w = xr.open_zarr(str(edir / 'angular_correlation_w.zarr'), consolidated=False)
            r_um = ds_w.coords['distance'].values * 0.11
            frames_coord = ds_w.coords['frame'].values
            particles_coord = ds_w.coords['particle'].values
            arr_total = ds_w['angular_correlation'].values
            f_to_idx = {int(f_val): idx for idx, f_val in enumerate(frames_coord)}
            p_to_idx = {int(p_val): idx for idx, p_val in enumerate(particles_coord)}

            for pid, pgrp in df_obs_fov.groupby('particle'):
                if pid not in p_to_idx:
                    continue
                p_idx = p_to_idx[pid]
                valid_f_indices = [f_to_idx[int(f)] for f in pgrp['frame'].values if int(f) in f_to_idx]
                if len(valid_f_indices) < 3:
                    continue
                
                # 粒子 i の平均局所相関プロファイル C_i(r) から相関長 xi_i を決定
                c_p = np.nanmean(arr_total[:, valid_f_indices, p_idx], axis=1)
                xi_p = fit_local_xi(r_um, c_p, min_r, max_r=25.0)
                if not np.isfinite(xi_p) or xi_p <= 0.05:
                    continue

                # Run 状態（走行状態）における配向単位ベクトルの自己相関 (OACF)
                run_pgrp = pgrp[pgrp['state'] == 1].sort_values('frame')
                if len(run_pgrp) < 3:
                    # Run 点が少ない場合は全移動点を利用
                    run_pgrp = pgrp.sort_values('frame')
                
                dx = run_pgrp['dx_um'].values
                dy = run_pgrp['dy_um'].values
                sp = np.sqrt(dx**2 + dy**2)
                valid_sp = sp > 1e-6
                if np.sum(valid_sp) < 3:
                    continue
                ux = dx[valid_sp] / sp[valid_sp]
                uy = dy[valid_sp] / sp[valid_sp]
                
                # OACF の計算
                T = len(ux)
                max_m = min(15, T - 1)
                oacf_vals = []
                for m in range(max_m + 1):
                    if m == 0:
                        oacf_vals.append(1.0)
                    else:
                        dot = ux[m:] * ux[:-m] + uy[m:] * uy[:-m]
                        oacf_vals.append(np.mean(dot))
                
                # 1. 積分相関時間 tau_int (台形積分: 初回ゼロクロスまで)
                tau_int = 0.0
                dt = 4.0
                for m in range(len(oacf_vals) - 1):
                    c0 = oacf_vals[m]
                    c1 = oacf_vals[m+1]
                    if c1 <= 0:
                        frac = c0 / (c0 - c1 + 1e-12)
                        tau_int += 0.5 * c0 * (frac * dt)
                        break
                    else:
                        tau_int += 0.5 * (c0 + c1) * dt
                
                # 2. 指数フィッティング相関時間 tau_fit
                tau_fit = np.nan
                try:
                    t_fit_arr = np.arange(len(oacf_vals)) * dt
                    c_fit_arr = np.array(oacf_vals)
                    v_m = (c_fit_arr > 0.05) & np.isfinite(c_fit_arr)
                    if np.sum(v_m) >= 2:
                        popt_f, _ = curve_fit(lambda t, tau: np.exp(-t/tau), t_fit_arr[v_m], c_fit_arr[v_m], p0=[10.0], bounds=(0.5, 200.0))
                        tau_fit = float(popt_f[0])
                except Exception:
                    pass
                
                if tau_int <= 0:
                    continue
                
                x_p = rc / xi_p
                records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'radius_um': rc,
                    'fov': fov,
                    'particle': pid,
                    'xi_um': xi_p,
                    'x': x_p,
                    'tau_p_int': tau_int,
                    'tau_p_fit': tau_fit if np.isfinite(tau_fit) else tau_int,
                    'inv_tau_p_int': 1.0 / tau_int,
                    'n_points': len(run_pgrp),
                })

    df_res = pd.DataFrame(records)
    print(f"Extracted {len(df_res)} valid particle records with (x_i, tau_p,i).")
    return df_res


def save_figure_to_all(fig, basename: str, out_dirs: List[Path]):
    """全ての出力ディレクトリに PNG と SVG で保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.png", dpi=300, bbox_inches='tight')
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            print(f"Saved: {d / basename}.png")
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def main():
    root_dir = find_default_root()
    workspace_dir = CURRENT_DIR

    out_dirs = [
        workspace_dir / "figure" / "hmm_1d",
        workspace_dir / "figure" / "msd",
        workspace_dir / "figure" / "scaling",
        root_dir / "figure" / "hmm_1d",
        root_dir / "figure" / "msd",
        root_dir / "figure" / "scaling",
    ]

    # 個別粒子データの抽出
    df_parts = extract_particle_tau_and_xi(root_dir)

    # 理論パラメータ tau_0 の線形重み付きフィッティング (Linear Weighted Least Squares)
    # 理論式: tau_p(x) = tau_0 * exp(-4/3 * x)
    # 目的関数: min_{tau_0} \sum_i w_i [tau_{p, i} - tau_0 * exp(-4/3 * x_i)]^2
    # 重み w_i = n_points_i (各粒子のトラッキングフレーム点数。長い軌跡ほど高精度)
    if not df_parts.empty:
        x_arr = df_parts['x'].to_numpy(dtype=float)
        tau_arr = df_parts['tau_p_int'].to_numpy(dtype=float)
        n_weights = df_parts['n_points'].to_numpy(dtype=float) if 'n_points' in df_parts.columns else np.ones_like(x_arr)
        
        f_x = np.exp(-(4.0 / 3.0) * x_arr)
        
        # 線形重み付き最小二乗推定量
        tau0_fitted = float(np.sum(n_weights * tau_arr * f_x) / np.sum(n_weights * (f_x ** 2)))
        
        # フィッティング誤差 (標準誤差 SEM)
        residuals = tau_arr - tau0_fitted * f_x
        w_dof = max(1, len(x_arr) - 1)
        s_sq = np.sum(n_weights * (residuals ** 2)) / (np.sum(n_weights) * (w_dof / len(x_arr)))
        tau0_err = float(np.sqrt(s_sq / np.sum(n_weights * (f_x ** 2))))
        
        # 決定係数 R^2
        ss_res = np.sum(n_weights * (residuals ** 2))
        ss_tot = np.sum(n_weights * ((tau_arr - np.average(tau_arr, weights=n_weights)) ** 2))
        r2_fit = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan
    else:
        tau0_fitted = 14.00
        tau0_err = np.nan
        r2_fit = np.nan

    print(f"Linear weighted fit baseline persistence time: tau_0 = {tau0_fitted:.2f} ± {tau0_err:.2f} s (R^2 = {r2_fit:.3f})")

    # ビーズサイズごとの統計集計
    bead_stats = []
    for b in BEADS_INFO:
        sub = df_parts[df_parts['bead_name'] == b['name']]
        if len(sub) > 0:
            mean_x = float(sub['x'].mean())
            sem_x = float(sub['x'].sem()) if len(sub) > 1 else 0.0
            mean_tau_int = float(sub['tau_p_int'].mean())
            sem_tau_int = float(sub['tau_p_int'].sem()) if len(sub) > 1 else 0.0
            mean_inv_tau = float(sub['inv_tau_p_int'].mean())
            sem_inv_tau = float(sub['inv_tau_p_int'].sem()) if len(sub) > 1 else 0.0
            geom_tau = float(np.exp(np.mean(np.log(sub['tau_p_int']))))
            median_tau = float(sub['tau_p_int'].median())
            q25_tau = float(sub['tau_p_int'].quantile(0.25))
            q75_tau = float(sub['tau_p_int'].quantile(0.75))

            bead_stats.append({
                'bead_name': b['name'],
                'label': b['label'],
                'diameter_um': b['diameter_um'],
                'radius_um': b['radius_um'],
                'color': b['color'],
                'marker': b['marker'],
                'N': len(sub),
                'mean_x': mean_x,
                'sem_x': sem_x,
                'mean_tau_p': mean_tau_int,
                'sem_tau_p': sem_tau_int,
                'mean_inv_tau': mean_inv_tau,
                'sem_inv_tau': sem_inv_tau,
                'geom_tau_p': geom_tau,
                'median_tau_p': median_tau,
                'q25_tau_p': q25_tau,
                'q75_tau_p': q75_tau,
            })
    df_bead_stats = pd.DataFrame(bead_stats)

    # アンサンブル OACF サマリーの取得（参考比較用）
    acf_path = CURRENT_DIR / "figure" / "hmm_1d" / "hmm_autocorrelation_summary_k2.csv"
    df_acf = pd.read_csv(acf_path) if acf_path.exists() else pd.DataFrame()
    ens_oacf_map = {}
    if not df_acf.empty:
        sub_oacf = df_acf[(df_acf['state'] == 1) & (df_acf['mode'] == 'oacf')]
        for _, row in sub_oacf.iterrows():
            ens_oacf_map[row['diameter_um']] = {
                'tau_int': row.get('tau_int_zero_s', np.nan),
                'tau_fit': row.get('tau_corr_s', np.nan),
            }

    # CSV の保存
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df_parts.to_csv(d / "tau_p_particle_pairs.csv", index=False)
            df_bead_stats.to_csv(d / "tau_p_bead_summary.csv", index=False)
        except Exception:
            pass

    # 理論曲線
    x_dense = np.linspace(0.0, 1.2, 300)
    tau_p_theory = tau0_fitted * np.exp(-(4.0 / 3.0) * x_dense)
    inv_tau_theory = (1.0 / tau0_fitted) * np.exp((4.0 / 3.0) * x_dense)

    # =========================================================================
    # Figure 1: 2-Panel Comparison (Left: Rate 1/tau_p, Right: Time tau_p)
    # =========================================================================
    fig_2p, (ax_rate, ax_time) = plt.subplots(1, 2, figsize=(15.2, 6.2))

    # --- Panel (a): Orientation Relaxation Rate 1/tau_p vs x = R_c / xi ---
    label_rate_theory = r'Theory: $\frac{1}{\tau_{\mathrm{p}}(x)} = \frac{1}{\tau_0} \exp\left(\frac{4}{3}x\right)$' + f' ($\tau_0 = {tau0_fitted:.2f}\,\mathrm{{s}}$)'
    ax_rate.plot(
        x_dense, inv_tau_theory,
        color='#111111', linestyle='-', linewidth=2.8, zorder=3,
        label=label_rate_theory
    )

    # 個別粒子散布 (Layer 1)
    for b in BEADS_INFO:
        sub_p = df_parts[df_parts['bead_name'] == b['name']]
        if not sub_p.empty:
            ax_rate.scatter(
                sub_p['x'], sub_p['inv_tau_p_int'],
                color=b['color'], marker=b['marker'], s=42, alpha=0.35, edgecolors='none', zorder=2
            )

    # ビーズ平均代表点 (Layer 2: Mean ± SEM)
    for _, row in df_bead_stats.iterrows():
        ax_rate.errorbar(
            row['mean_x'], row['mean_inv_tau'],
            xerr=row['sem_x'], yerr=row['sem_inv_tau'],
            fmt=row['marker'], color=row['color'], ecolor='black', elinewidth=1.8,
            capsize=4.5, capthick=1.3, markersize=10.0, markeredgecolor='black', markeredgewidth=1.3,
            zorder=5, label=f"{row['label']} ($N={row['N']}$)"
        )

    ax_rate.set_xlim(-0.02, 0.90)
    ax_rate.set_ylim(0.0, 0.70)
    ax_rate.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_rate.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_rate.set_xlabel(r"Scaled Cargo Radius $x = R_c / \xi_i$", fontsize=12.5, fontweight='bold')
    ax_rate.set_ylabel(r"Orientation Relaxation Rate $\tau_{\mathrm{p}}^{-1}$ [$\mathrm{s}^{-1}$]", fontsize=12.5, fontweight='bold')
    ax_rate.set_title(r"$\mathbf{(a)}$ Relaxation Rate $\frac{1}{\tau_{\mathrm{p}}(x)} = \frac{1}{\tau_0} \exp\left(\frac{4}{3}x\right)$", fontsize=13.5, fontweight='bold')
    ax_rate.grid(True, which='both', linestyle='--', alpha=0.45)
    ax_rate.legend(fontsize=9.0, loc='upper left', frameon=True, framealpha=0.92)

    # --- Panel (b): Orientation Persistence Time tau_p vs x = R_c / xi (Semilog-y) ---
    label_time_theory = r'Theory: $\tau_{\mathrm{p}}(x) = \tau_0 \exp\left(-\frac{4}{3}x\right)$' + f' ($\tau_0 = {tau0_fitted:.2f}\,\mathrm{{s}}$)'
    ax_time.plot(
        x_dense, tau_p_theory,
        color='#111111', linestyle='-', linewidth=2.8, zorder=3,
        label=label_time_theory
    )

    # 個別粒子散布 (Layer 1)
    for b in BEADS_INFO:
        sub_p = df_parts[df_parts['bead_name'] == b['name']]
        if not sub_p.empty:
            ax_time.scatter(
                sub_p['x'], sub_p['tau_p_int'],
                color=b['color'], marker=b['marker'], s=42, alpha=0.35, edgecolors='none', zorder=2
            )

    # ビーズ平均代表点 (Layer 2: Mean ± SEM)
    for _, row in df_bead_stats.iterrows():
        ax_time.errorbar(
            row['mean_x'], row['mean_tau_p'],
            xerr=row['sem_x'], yerr=row['sem_tau_p'],
            fmt=row['marker'], color=row['color'], ecolor='black', elinewidth=1.8,
            capsize=4.5, capthick=1.3, markersize=10.0, markeredgecolor='black', markeredgewidth=1.3,
            zorder=5, label=f"{row['label']} ($N={row['N']}$)"
        )

    # フレーム間隔基準線 (4s)
    ax_time.hlines(y=4.0, xmin=-0.02, xmax=0.90, color='#666666', linestyle='--', linewidth=1.4, zorder=1)
    ax_time.text(0.55, 4.25, r"Frame interval $\Delta t = 4\,\mathrm{s}$", color='#555555', fontsize=9.0, fontweight='bold')

    err_str = f" \\pm {tau0_err:.2f}" if np.isfinite(tau0_err) else ""
    param_info = (
        r"$\mathbf{Scaling\ Law:}$" "\n"
        r"$\tau_{\mathrm{p}}(x) = \tau_0 \exp\left(-\frac{4}{3}x\right)$" "\n"
        r"$x_i = R_c / \xi_i\ (\mathrm{Per\ particle})$" "\n"
        f"Weighted fit $\\tau_0 = {tau0_fitted:.2f}{err_str}\\,\\mathrm{{s}}$\n"
        f"Total particles: $N = {len(df_parts)}$"
    )
    ax_time.text(
        0.04, 0.05, param_info,
        transform=ax_time.transAxes, verticalalignment='bottom',
        fontsize=9.2, bbox=dict(boxstyle='round,pad=0.45', facecolor='#ffffdd', alpha=0.92, edgecolor='#ddcc77')
    )

    ax_time.set_yscale('log')
    ax_time.set_xlim(-0.02, 0.90)
    ax_time.set_ylim(1.0, 50.0)
    ax_time.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_time.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_time.yaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 5, 10, 20, 50]))
    ax_time.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_time.set_xlabel(r"Scaled Cargo Radius $x = R_c / \xi_i$", fontsize=12.5, fontweight='bold')
    ax_time.set_ylabel(r"Orientation Persistence Time $\tau_{\mathrm{p}}$ [s]", fontsize=12.5, fontweight='bold')
    ax_time.set_title(r"$\mathbf{(b)}$ Orientation Persistence Time $\tau_{\mathrm{p}}$ vs $x = R_c / \xi_i$", fontsize=13.5, fontweight='bold')
    ax_time.grid(True, which='both', linestyle='--', alpha=0.45)
    ax_time.legend(fontsize=9.0, loc='upper right', frameon=True, framealpha=0.92)

    fig_2p.tight_layout()
    save_figure_to_all(fig_2p, "tau_p_theory_2panel", out_dirs)
    save_figure_to_all(fig_2p, "tau_oacf_theory_2panel", out_dirs)
    plt.close(fig_2p)

    # =========================================================================
    # Figure 2: Standalone Master Curve (Semilog-y)
    # =========================================================================
    fig_s, ax_s = plt.subplots(figsize=(8.5, 6.2))

    # 理論曲線
    label_single_theory = r'Theory Master Curve: $\tau_{\mathrm{p}}(x) = \tau_0 \exp\left(-\frac{4}{3}x\right)$' + f'\n  ($\\tau_0 = {tau0_fitted:.2f}\\,\\mathrm{{s}}$)'
    ax_s.plot(
        x_dense, tau_p_theory,
        color='#111111', linestyle='-', linewidth=3.0, zorder=3,
        label=label_single_theory
    )

    # 個別粒子散布 (Layer 1)
    for b in BEADS_INFO:
        sub_p = df_parts[df_parts['bead_name'] == b['name']]
        if not sub_p.empty:
            ax_s.scatter(
                sub_p['x'], sub_p['tau_p_int'],
                color=b['color'], marker=b['marker'], s=48, alpha=0.35, edgecolors='none', zorder=2
            )

    # ビーズ代表点 (Layer 2: Mean ± SEM)
    for _, row in df_bead_stats.iterrows():
        ax_s.errorbar(
            row['mean_x'], row['mean_tau_p'],
            xerr=row['sem_x'], yerr=row['sem_tau_p'],
            fmt=row['marker'], color=row['color'], ecolor='black', elinewidth=2.0,
            capsize=5.0, capthick=1.4, markersize=10.5, markeredgecolor='black', markeredgewidth=1.3,
            zorder=5, label=f"{row['label']} Mean $\\pm$ SEM ($N={row['N']}$)"
        )

    # フレーム間隔 (4s)
    ax_s.hlines(y=4.0, xmin=-0.02, xmax=1.20, color='#666666', linestyle='--', linewidth=1.4, zorder=1)
    ax_s.text(0.52, 4.3, r"Frame interval limit $\Delta t = 4\,\mathrm{s}$", color='#555555', fontsize=9.2, fontweight='bold')

    ax_s.text(
        0.04, 0.05, param_info,
        transform=ax_s.transAxes, verticalalignment='bottom',
        fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffdd', alpha=0.92, edgecolor='#ddcc77')
    )

    ax_s.set_yscale('log')
    ax_s.set_xlim(-0.02, 1.2)
    ax_s.set_ylim(1.0, 50.0)
    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_s.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_s.yaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 5, 10, 20, 50]))
    ax_s.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_s.set_xlabel(r"Scaled Cargo Radius $x = R_c / \xi_i$", fontsize=12.5, fontweight='bold')
    ax_s.set_ylabel(r"Orientation Persistence Time $\tau_{\mathrm{p}}$ [s]", fontsize=12.5, fontweight='bold')
    ax_s.set_title(r"Orientation Persistence Time $\tau_{\mathrm{p}}$ vs Scaled Radius $x = R_c / \xi_i$", fontsize=13.5, fontweight='bold', pad=10)
    ax_s.grid(True, which='both', linestyle='--', alpha=0.45)
    ax_s.legend(fontsize=9.0, loc='upper right', frameon=True, framealpha=0.92)

    fig_s.tight_layout()
    save_figure_to_all(fig_s, "tau_p_vs_scaled_radius", out_dirs)
    save_figure_to_all(fig_s, "tau_p_vs_diameter", out_dirs)
    save_figure_to_all(fig_s, "tau_oacf_vs_diameter", out_dirs)
    plt.close(fig_s)

    # =========================================================================
    # Figure 3: Log-Log Scale Master Curve
    # =========================================================================
    fig_ll, ax_ll = plt.subplots(figsize=(8.5, 6.2))
    ax_ll.plot(
        x_dense[x_dense > 0.02], tau_p_theory[x_dense > 0.02],
        color='#111111', linestyle='-', linewidth=3.0, zorder=3,
        label=rf'Theory Master Curve: $\tau_{{\mathrm{{p}}}}(x) = \tau_0 \exp\left(-\frac{{4}}{{3}}x\right)$'
    )

    for b in BEADS_INFO:
        sub_p = df_parts[df_parts['bead_name'] == b['name']]
        if not sub_p.empty:
            ax_ll.scatter(
                sub_p['x'], sub_p['tau_p_int'],
                color=b['color'], marker=b['marker'], s=45, alpha=0.35, edgecolors='none', zorder=2
            )

    for _, row in df_bead_stats.iterrows():
        ax_ll.errorbar(
            row['mean_x'], row['mean_tau_p'],
            xerr=row['sem_x'], yerr=row['sem_tau_p'],
            fmt=row['marker'], color=row['color'], ecolor='black', elinewidth=2.0,
            capsize=5.0, capthick=1.4, markersize=10.5, markeredgecolor='black', markeredgewidth=1.3,
            zorder=5, label=f"{row['label']} ($N={row['N']}$)"
        )

    ax_ll.set_xscale('log')
    ax_ll.set_yscale('log')
    ax_ll.set_xlim(0.04, 1.2)
    ax_ll.set_ylim(1.0, 50.0)
    ax_ll.yaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 5, 10, 20, 50]))
    ax_ll.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_ll.set_xlabel(r"Scaled Cargo Radius $x = R_c / \xi_i$", fontsize=12.5, fontweight='bold')
    ax_ll.set_ylabel(r"Orientation Persistence Time $\tau_{\mathrm{p}}$ [s]", fontsize=12.5, fontweight='bold')
    ax_ll.set_title(r"Log-Log Orientation Persistence Time $\tau_{\mathrm{p}}$ vs $x = R_c / \xi_i$", fontsize=13.5, fontweight='bold', pad=10)
    ax_ll.grid(True, which='both', linestyle='--', alpha=0.45)
    ax_ll.legend(fontsize=9.0, loc='lower left', frameon=True, framealpha=0.92)

    fig_ll.tight_layout()
    save_figure_to_all(fig_ll, "tau_p_scaling_master_curve_loglog", out_dirs)
    plt.close(fig_ll)

    print("\n[Done] Successfully generated all tau_p scaling master curve plots!")


if __name__ == "__main__":
    main()
