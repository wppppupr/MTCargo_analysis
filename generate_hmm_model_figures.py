#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_hmm_model_figures.py
=============================
0.6, 1, 3, 20 µm の粒子は HMM 2状態モデル (K=2: Tumble / Run)、
5, 7 µm の粒子は単一運動モード (K=1: Single State) として解析し、
figure/hmm_1d/ と同様の以下の5つのグラフを生成して figure/hmm_model/ に保存します：

1. hmm_trajectories_k2.svg (hmm_trajectories.svg)
2. hmm_emission_density_k2.svg (hmm_emission_density.svg)
3. hmm_turning_angle_polar_k2.svg (hmm_turning_angle_polar.svg)
4. hmm_state_msd_k2.svg (hmm_state_msd.svg)
5. hmm_autocorrelation_grid_k2.svg (hmm_autocorrelation_grid.svg)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from libs import hmm_cargo as hc

# ビーズ定義 (0.6, 1, 3, 20 um -> K=2; 5, 7 um -> K=1)
BEADS_CONFIG = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "k": 2},
    {"name": "beads1um", "diameter_um": 1.18, "marker": "o", "k": 2},
    {"name": "beads3um", "diameter_um": 3.37, "marker": "d", "k": 2},
    {"name": "beads5um", "diameter_um": 5.00, "marker": "p", "k": 1},
    {"name": "beads7um", "diameter_um": 7.24, "marker": "s", "k": 1},
    {"name": "beads20um", "diameter_um": 20.00, "marker": "h", "k": 2},
]

POSSIBLE_ROOTS = [
    Path("/Volumes/data/Sasaki/MTsingleBeads"),
    Path("/Volumes/data-1/Sasaki/MTsingleBeads"),
    Path("/Volumes/data/sasaki/MTsingleBeads"),
    Path("/Volumes/data-1/sasaki/MTsingleBeads"),
]

STATE_COLORS = {
    0: "#e6550d",  # Tumble / Pause / Single Mode (Orange)
    1: "#2ca02c",  # Run (Green)
}

STATE_NAMES_K2 = {
    0: "Tumble / Pause",
    1: "Run",
}

STATE_NAMES_K1 = {
    0: "Single Mode",
}


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


_TRACKS_CACHE = {}


def load_cached_tracks(csv_path: Path) -> Optional[pd.DataFrame]:
    str_path = str(csv_path)
    if str_path in _TRACKS_CACHE:
        return _TRACKS_CACHE[str_path]
    try:
        df = pd.read_csv(csv_path)
        if {'particle', 'frame', 'x', 'y'}.issubset(df.columns):
            _TRACKS_CACHE[str_path] = df[['particle', 'frame', 'x', 'y']].copy()
            return _TRACKS_CACHE[str_path]
    except Exception as e:
        print(f"[WARNING] Failed to load {csv_path}: {e}", flush=True)
    return None


def collect_bead_hmm_data(
    exp_dirs: List[Path],
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
):
    all_dfs = []
    particle_offset = 0

    for edir in exp_dirs:
        tracks_csv = edir / "beads_tracks.csv"
        if not tracks_csv.exists():
            continue
        df_tracks = load_cached_tracks(tracks_csv)
        if df_tracks is None or df_tracks.empty:
            continue

        df_copy = df_tracks.copy()
        if 'particle' not in df_copy.columns or 'frame' not in df_copy.columns:
            continue

        df_copy['particle'] = df_copy['particle'] + particle_offset
        particle_offset += int(df_copy['particle'].max()) + 1
        df_copy['exp_dir'] = edir.name
        all_dfs.append(df_copy)

    if not all_dfs:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_combined = pd.concat(all_dfs, ignore_index=True)
    X, lengths, df_obs = hc.extract_hmm_features(
        df_combined,
        tau=tau,
        scale=scale,
        frame_interval=frame_interval,
        epsilon=epsilon,
    )
    return X, lengths, df_obs


# =========================================================================
# 1. 軌跡セグメンテーション描画 (6パネル)
# =========================================================================
def plot_trajectories_hybrid(
    fitted_results: Dict[str, dict],
    output_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_CONFIG):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']
        k = binfo['k']

        if bname not in fitted_results:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)")
            continue

        res = fitted_results[bname]
        df_obs = res['df_obs']
        pred_states = res['pred_states']

        if df_obs.empty:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)")
            continue

        # 特定実験ディレクトリのトラックを優先して使用 (0.63 um -> beads06um001, 3.37 um -> beads3um003)
        target_exp = None
        if bname == 'beads06um':
            target_exp = '06um001'
        elif bname == 'beads3um':
            target_exp = '3um003'

        if target_exp and 'exp_dir' in df_obs.columns:
            sub_exp = df_obs[df_obs['exp_dir'].str.contains(target_exp, case=False, na=False)]
            if not sub_exp.empty:
                part_counts = sub_exp['particle'].value_counts()
            else:
                part_counts = df_obs['particle'].value_counts()
        else:
            part_counts = df_obs['particle'].value_counts()

        top_particles = part_counts.head(4).index.tolist()

        for pid in top_particles:
            p_mask = (df_obs['particle'] == pid).to_numpy()
            sub_p = df_obs[p_mask].sort_values(by='frame')
            p_states = pred_states[p_mask]

            x_pts = sub_p['x_um'].to_numpy()
            y_pts = sub_p['y_um'].to_numpy()
            frames = sub_p['frame'].to_numpy()

            if len(x_pts) < 3:
                continue

            for i in range(len(x_pts) - 1):
                if frames[i+1] != frames[i] + 1:
                    continue
                st = p_states[i]
                scolor = STATE_COLORS.get(st, f"C{st}")
                ax.plot(
                    [x_pts[i], x_pts[i+1]],
                    [y_pts[i], y_pts[i+1]],
                    color=scolor,
                    lw=2.2,
                    alpha=0.85,
                    solid_capstyle='round',
                    zorder=3,
                )
            ax.plot(x_pts[0], y_pts[0], marker='o', markersize=4.5, color='black', alpha=0.85, zorder=5)
            ax.plot(x_pts[-1], y_pts[-1], marker='s', markersize=4.0, color='black', alpha=0.85, zorder=5)

        mode_str = f" ($K={k}$)"
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}${mode_str}", fontsize=13, fontweight='bold', pad=8)
        ax.set_xlim(0, 285)
        ax.set_ylim(0, 240)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([0, 50, 100, 150, 200, 250])
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.35, color='gray')
        if idx >= 3:
            ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]", fontsize=11)

    legend_elements = [
        plt.Line2D([0], [0], color=STATE_COLORS[0], lw=2.5, label="Tumble / Pause (0.6, 1, 3, 20 $\\mu$m) / Single Mode (5, 7 $\\mu$m)"),
        plt.Line2D([0], [0], color=STATE_COLORS[1], lw=2.5, label="Run (0.6, 1, 3, 20 $\\mu$m)"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5, label="Start"),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=5, label="End"),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=True, fontsize=10.5)
    fig.suptitle("HMM Decoded Trajectory Segmentation (Representative Tracks)", fontsize=14, fontweight='bold', y=1.025)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


# =========================================================================
# 2. 1D 放出確率密度分布 (6パネル)
# =========================================================================
def plot_emission_density_hybrid(
    fitted_results: Dict[str, dict],
    output_path: Path,
    epsilon: float = 1e-3,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, binfo in enumerate(BEADS_CONFIG):
        ax = axes_flat[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']
        k = binfo['k']

        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        X = res['X']
        model = res['model']

        if len(X) == 0:
            ax.set_visible(False)
            continue

        log_v_vals = X[:, 0]
        pi_stat = model.get_stationary_distribution()

        # ヒストグラム
        counts, bin_edges, _ = ax.hist(
            log_v_vals,
            bins=40,
            density=True,
            color='#999999',
            alpha=0.35,
            edgecolor='#777777',
            label='Observed Data',
            zorder=1,
        )

        # ガウスフィッティング曲線
        x_grid = np.linspace(np.min(log_v_vals) - 0.5, np.max(log_v_vals) + 0.5, 300)
        total_pdf = np.zeros_like(x_grid)

        state_names = STATE_NAMES_K2 if k == 2 else STATE_NAMES_K1

        for s in range(k):
            mu = float(model.model.means_[s, 0])
            cov = model.model.covars_[s]
            var = float(cov[0, 0] if cov.ndim == 2 else (cov[0] if cov.ndim == 1 else cov))
            sigma = np.sqrt(max(var, 1e-8))
            weight = float(pi_stat[s])

            pdf_s = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
            weighted_pdf_s = weight * pdf_s
            total_pdf += weighted_pdf_s

            v_geom = float(np.exp(mu) - epsilon)
            if v_geom < 0:
                v_geom = 0.0

            s_lbl = state_names.get(s, f"State {s}")
            col = STATE_COLORS.get(s, f"C{s}")

            ax.plot(
                x_grid,
                weighted_pdf_s,
                color=col,
                lw=2.2,
                label=f"{s_lbl}: $v_{{\\mathrm{{geom}}}}={v_geom:.3f}\\,\\mu\\mathrm{{m/s}}$ ({weight*100:.1f}%)",
                zorder=3,
            )
            ax.fill_between(x_grid, 0, weighted_pdf_s, color=col, alpha=0.18, zorder=2)
            ax.axvline(mu, color=col, linestyle=':', lw=1.5, alpha=0.8, zorder=3)

        fit_label = r'Mixture Fit $\sum \pi_k \mathcal{N}_k$' if k > 1 else r'Gaussian Fit $\mathcal{N}$'
        ax.plot(x_grid, total_pdf, color='#111111', lw=1.8, linestyle='--', label=fit_label, zorder=4)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ($K={k}$, $N={len(X):,}$)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8.0, frameon=True, framealpha=0.92)

        if idx >= 3:
            ax.set_xlabel(f"$\\ln(v + \\epsilon)$  [$\\epsilon={epsilon}$]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel("Probability Density", fontsize=11)

    fig.suptitle(r"1D Speed Gaussian HMM Emission Distributions ($\ln(v+\epsilon)$)", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


# =========================================================================
# 3. 方向転換角極座標分布 (6パネル)
# =========================================================================
def plot_turning_angle_polar_hybrid(
    fitted_results: Dict[str, dict],
    output_path: Path,
    bins: int = 24,
):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_CONFIG):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']
        k = binfo['k']

        if bname not in fitted_results:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=11)
            ax.axis('off')
            continue

        res = fitted_results[bname]
        angles_dict = res.get('turning_angles', {})
        if not angles_dict:
            continue

        bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        state_names = STATE_NAMES_K2 if k == 2 else STATE_NAMES_K1

        for s in range(k):
            s_lbl = state_names.get(s, f"State {s}")
            scolor = STATE_COLORS.get(s, f"C{s}")
            arr_s = angles_dict.get(s, np.array([]))

            if len(arr_s) < 10:
                continue

            counts, _ = np.histogram(arr_s, bins=bin_edges, density=True)
            theta_closed = np.concatenate([centers, [centers[0]]])
            r_closed = np.concatenate([counts, [counts[0]]])

            ax.plot(theta_closed, r_closed, color=scolor, lw=2.0, label=f"{s_lbl} ($N={len(arr_s):,}$)", zorder=3)
            ax.fill(theta_closed, r_closed, color=scolor, alpha=0.18, zorder=2)

        # 一様分布円
        u_val = 1.0 / (2.0 * np.pi)
        th_circ = np.linspace(-np.pi, np.pi, 200)
        ax.plot(th_circ, np.full_like(th_circ, u_val), color='gray', linestyle=':', lw=1.2, label=r'Uniform $1/(2\pi)$', zorder=1)

        ax.set_theta_zero_location('E')
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ($K={k}$)", fontsize=11, fontweight='bold', pad=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.2), fontsize=7.5, frameon=True, framealpha=0.9)

    fig.suptitle(
        r"Polar Distributions of Turning Angles $P(\Delta\theta)$: Forward Persistence vs Isotropy",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


# =========================================================================
# 4. 状態別 MSD 曲線 (6パネル)
# =========================================================================
def plot_state_msd_hybrid(
    fitted_results: Dict[str, dict],
    output_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_CONFIG):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']
        k = binfo['k']

        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        df_msd = res['df_msd']
        df_fits = res['df_msd_fits']

        if df_msd.empty:
            ax.set_visible(False)
            continue

        state_names = STATE_NAMES_K2 if k == 2 else STATE_NAMES_K1

        for s in range(k):
            sub_m = df_msd[df_msd['state'] == s].sort_values(by='lag_time_s')
            if sub_m.empty:
                continue

            lbl = state_names.get(s, f"State {s}")
            col = STATE_COLORS.get(s, f"C{s}")
            mrk = 'o' if s == 0 else 's'

            fit_row = df_fits[df_fits['state'] == s]
            if not fit_row.empty and not np.isnan(fit_row.iloc[0]['alpha']):
                alpha_val = fit_row.iloc[0]['alpha']
                lbl_with_alpha = f"{lbl} ($\\alpha={alpha_val:.2f}$)"
            else:
                lbl_with_alpha = lbl

            valid_pts = sub_m[sub_m['msd_um2'] > 0]
            ax.errorbar(
                valid_pts['lag_time_s'],
                valid_pts['msd_um2'],
                yerr=valid_pts['msd_sem_um2'],
                marker=mrk,
                markersize=4,
                lw=1.8,
                color=col,
                capsize=2,
                label=lbl_with_alpha,
            )

            # フィット線
            if not fit_row.empty and not np.isnan(fit_row.iloc[0]['alpha']):
                alpha = fit_row.iloc[0]['alpha']
                D_app = fit_row.iloc[0]['D_apparent_um2_s']
                t_fit = np.logspace(np.log10(valid_pts['lag_time_s'].min()), np.log10(valid_pts['lag_time_s'].max()), 50)
                msd_fit = 4.0 * D_app * (t_fit ** alpha)
                ax.plot(t_fit, msd_fit, color=col, linestyle='--', lw=1.2, alpha=0.8)

        # All (全体のMSD)
        sub_all = df_msd[df_msd['state'] == -1].sort_values(by='lag_time_s')
        if not sub_all.empty:
            valid_all = sub_all[sub_all['msd_um2'] > 0]
            fit_all = df_fits[df_fits['state'] == -1]
            if not fit_all.empty and not np.isnan(fit_all.iloc[0]['alpha']):
                alpha_all = fit_all.iloc[0]['alpha']
                lbl_all = f"All ($\\alpha={alpha_all:.2f}$)"
            else:
                lbl_all = "All"
            ax.plot(valid_all['lag_time_s'], valid_all['msd_um2'], color='gray', linestyle=':', lw=1.5, label=lbl_all)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ($K={k}$)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8.5, frameon=True)

        if idx >= 3:
            ax.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"MSD $\langle \Delta r^2 \rangle$ [$\mu\mathrm{m}^2$]", fontsize=11)

    fig.suptitle("State-Dependent Mean Squared Displacement (MSD) of HMM Motion Modes", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


# =========================================================================
# 5. 状態別自己相関（VACF, OACF, SACF）グリッドプロット (3行6列)
# =========================================================================
def plot_autocorrelation_grid_hybrid(
    fitted_results: Dict[str, dict],
    output_path: Path,
    max_lag_s: float = 60.0,
):
    fig, axes = plt.subplots(3, 6, figsize=(22, 10.5), sharex=True)
    corr_types = [
        ('vacf', r'VACF $\langle \mathbf{v}(t)\cdot\mathbf{v}(t+\tau) \rangle / \langle v^2 \rangle$', 'Velocity Vector'),
        ('oacf', r'OACF $\langle \hat{\mathbf{e}}(t)\cdot\hat{\mathbf{e}}(t+\tau) \rangle$', 'Orientation Unit Vector'),
        ('sacf', r'SACF $\langle \delta v(t)\delta v(t+\tau) \rangle / \langle \delta v^2 \rangle$', 'Speed Fluctuation'),
    ]

    for col_idx, binfo in enumerate(BEADS_CONFIG):
        bname = binfo['name']
        dia = binfo['diameter_um']
        k = binfo['k']

        if bname not in fitted_results or 'autocorrelations' not in fitted_results[bname]:
            for row_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue

        ac_dict = fitted_results[bname]['autocorrelations']
        fits_dict = fitted_results[bname].get('autocorr_fits', {})
        state_names = STATE_NAMES_K2 if k == 2 else STATE_NAMES_K1

        for row_idx, (ctype, ctitle, clbl) in enumerate(corr_types):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color='gray', linestyle=':', lw=1.0, alpha=0.7)

            for s in range(k):
                if s not in ac_dict or ctype not in ac_dict[s]:
                    continue
                df_c = ac_dict[s][ctype]
                if df_c.empty:
                    continue

                sub_c = df_c[df_c['lag_time_s'] <= max_lag_s]
                s_lbl = state_names.get(s, f"State {s}")
                col = STATE_COLORS.get(s, f"C{s}")
                mrk = 'o' if s == 0 else 's'
                lsty = '--' if s == 0 else '-'

                fit_res = fits_dict.get(s, {}).get(ctype, {})
                tau_int = fit_res.get('tau_int_zero_s', np.nan)

                label_str = f"{s_lbl}"
                if not np.isnan(tau_int):
                    label_str += f" ($\\tau_{{\\mathrm{{int}}}}={tau_int:.1f}\\,\\mathrm{{s}}$)"

                # データ点 + 折れ線 + エラーバー
                if 'sem' in sub_c.columns and not sub_c['sem'].isna().all():
                    ax.errorbar(
                        sub_c['lag_time_s'], sub_c['corr'], yerr=sub_c['sem'],
                        fmt=mrk, color=col, ecolor=col, linestyle=lsty, lw=1.5,
                        markersize=4.5, capsize=2, label=label_str, zorder=3, alpha=0.9
                    )
                else:
                    ax.plot(
                        sub_c['lag_time_s'], sub_c['corr'],
                        marker=mrk, color=col, linestyle=lsty, lw=1.5,
                        markersize=4.5, label=label_str, zorder=3, alpha=0.9
                    )

            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim(-0.25, 1.05)

            if row_idx == 0:
                ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ($K={k}$)", fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(ctitle, fontsize=10, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel(r"Lag Time $\tau$ [s]", fontsize=11)

            ax.legend(loc='upper right', fontsize=7.2, frameon=True, framealpha=0.9)

    fig.suptitle(
        r"State-Dependent Autocorrelation Functions & Integral Correlation Times $\tau_{\mathrm{int}}$",
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


# =========================================================================
# Main Pipeline
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate HMM Hybrid Model Figures (0.6, 1, 3, 20 um: K=2; 5, 7 um: K=1)")
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing bead experiment folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: figure/hmm_model)")
    parser.add_argument("--tau", type=int, default=1, help="Lag time in frames (default: 1)")
    parser.add_argument("--scale", type=float, default=0.11, help="Spatial scale in um/pixel (default: 0.11)")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Epsilon for ln(v + epsilon) in um/s (default: 1e-3)")
    parser.add_argument("--min_dwell_frames", type=int, default=2, help="Glitch filter duration in frames (default: 2)")

    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "figure" / "hmm_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("       HMM Model Hybrid Figure Generation (K=2 for 0.6,1,3,20um; K=1 for 5,7um)")
    print("=================================================================")
    print(f"Root dir:         {root_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Lag tau:          {args.tau} ({args.tau * args.frame_interval:.1f} s)")
    print(f"Epsilon:          {args.epsilon} um/s")
    print(f"Glitch filter:    {args.min_dwell_frames} frames ({args.min_dwell_frames * args.frame_interval:.1f} s)")
    print("=================================================================\n")

    exp_dirs_by_bead = {}
    for binfo in BEADS_CONFIG:
        bname = binfo['name']
        edirs = find_experiment_dirs(root_dir, bname)
        exp_dirs_by_bead[bname] = edirs
        print(f"[{bname}] Found {len(edirs)} experiment directories.")

    fitted_results = {}
    all_summaries = []
    all_msd_curves = []
    all_msd_fits = []
    all_autocorr_records = []

    for binfo in BEADS_CONFIG:
        bname = binfo['name']
        dia = binfo['diameter_um']
        k = binfo['k']
        edirs = exp_dirs_by_bead.get(bname, [])

        if not edirs:
            print(f"[WARNING] No experiment directories for {bname}. Skipping.")
            continue

        print(f"\n--- Processing {bname} (d = {dia:.2f} um, K = {k}) ---")
        X, lengths, df_obs = collect_bead_hmm_data(
            edirs,
            tau=args.tau,
            scale=args.scale,
            frame_interval=args.frame_interval,
            epsilon=args.epsilon,
        )

        if len(X) < 30:
            print(f"  [WARNING] Insufficient data ({len(X)} points). Skipping.")
            continue

        print(f"  Extracted {len(X):,} observation points across {len(lengths):,} tracks.")

        # HMM フィッティング
        if k == 2:
            init_means = np.array([[-3.5], [0.0]])
            hmm_model = hc.CargoGaussianHMM(
                n_components=2,
                epsilon=args.epsilon,
                random_state=42,
                init_means=init_means,
            )
        else:
            hmm_model = hc.CargoGaussianHMM(
                n_components=1,
                epsilon=args.epsilon,
                random_state=42,
            )

        hmm_model.fit(X, lengths=lengths)
        raw_pred_states = hmm_model.predict(X, lengths=lengths)
        proba = hmm_model.predict_proba(X, lengths=lengths)

        if k == 2 and args.min_dwell_frames >= 2:
            pred_states = hc.filter_state_glitches(raw_pred_states, lengths, min_duration_frames=args.min_dwell_frames)
        else:
            pred_states = raw_pred_states

        df_obs['pred_state'] = pred_states

        # MSD
        df_msd, df_fits = hc.calc_state_dependent_msd(
            df_obs,
            max_tau=25,
            frame_interval=args.frame_interval,
            n_components=k,
            fit_min_tau=1,
            fit_max_tau=10,
        )
        df_msd['bead_name'] = bname
        df_msd['diameter_um'] = dia
        df_msd['k'] = k
        df_fits['bead_name'] = bname
        df_fits['diameter_um'] = dia
        df_fits['k'] = k
        all_msd_curves.append(df_msd)
        all_msd_fits.append(df_fits)

        # サマリー
        df_state_sum = hmm_model.get_state_summary(frame_interval=args.frame_interval)
        df_state_sum['bead_name'] = bname
        df_state_sum['diameter_um'] = dia
        df_state_sum['k'] = k
        df_state_sum['n_observations'] = len(X)
        df_state_sum['n_tracks'] = len(lengths)
        all_summaries.append(df_state_sum)

        # 方向転換角
        turning_angles = hc.calc_state_dependent_turning_angles(
            df_obs,
            n_components=k,
        )

        # 自己相関 (VACF, OACF, SACF)
        autocorr_data = hc.calc_state_dependent_autocorrelations(
            df_obs,
            n_components=k,
            frame_interval=args.frame_interval,
            max_lag_frames=25,
        )
        autocorr_fits = {}
        for s in range(k):
            autocorr_fits[s] = {}
            s_label = STATE_NAMES_K2.get(s, f"State {s}") if k == 2 else STATE_NAMES_K1.get(s, f"State {s}")
            for ctype in ['vacf', 'oacf', 'sacf']:
                if s in autocorr_data and ctype in autocorr_data[s]:
                    df_c = autocorr_data[s][ctype]
                    f_res = hc.fit_autocorrelation_exponential(df_c, max_lag_s=60.0)
                    autocorr_fits[s][ctype] = f_res
                    all_autocorr_records.append({
                        'bead_name': bname,
                        'diameter_um': dia,
                        'k': k,
                        'state': s,
                        'state_label': s_label,
                        'mode': ctype,
                        'tau_corr_s': f_res.get('tau_corr_s', np.nan),
                        'tau_err_s': f_res.get('tau_err_s', np.nan),
                        'tau_int_zero_s': f_res.get('tau_int_zero_s', np.nan),
                        'r_squared': f_res.get('r_squared', np.nan),
                        'count': f_res.get('count', 0),
                    })

        fitted_results[bname] = {
            'X': X,
            'lengths': lengths,
            'df_obs': df_obs,
            'model': hmm_model,
            'pred_states': pred_states,
            'proba': proba,
            'turning_angles': turning_angles,
            'autocorrelations': autocorr_data,
            'autocorr_fits': autocorr_fits,
            'summary': df_state_sum,
            'df_msd': df_msd,
            'df_msd_fits': df_fits,
        }

        print("  State summary:")
        for _, srow in df_state_sum.iterrows():
            print(
                f"    State {int(srow['state'])} ({srow['label']}): "
                f"v_geom={srow['mean_speed_geom_um_s']:.3f} um/s, "
                f"frac={srow['stationary_prob']*100:.1f}%"
            )

    print("\n=== Generating Figures in figure/hmm_model ===")

    # 1. 軌跡セグメンテーション描画 (6パネル)
    fig1_path_k2 = root_dir / output_dir / "hmm_trajectories_k2.svg"
    fig1_path = root_dir / output_dir / "hmm_trajectories.svg"
    plot_trajectories_hybrid(fitted_results, fig1_path_k2)
    plot_trajectories_hybrid(fitted_results, fig1_path)

    # 2. 1D 放出確率密度分布 (6パネル)
    fig2_path_k2 = root_dir / output_dir / "hmm_emission_density_k2.svg"
    fig2_path = root_dir / output_dir / "hmm_emission_density.svg"
    plot_emission_density_hybrid(fitted_results, fig2_path_k2, epsilon=args.epsilon)
    plot_emission_density_hybrid(fitted_results, fig2_path, epsilon=args.epsilon)

    # 3. 方向転換角極座標分布 (6パネル)
    fig3_path_k2 = root_dir / output_dir / "hmm_turning_angle_polar_k2.svg"
    fig3_path = root_dir / output_dir / "hmm_turning_angle_polar.svg"
    plot_turning_angle_polar_hybrid(fitted_results, fig3_path_k2)
    plot_turning_angle_polar_hybrid(fitted_results, fig3_path)

    # 4. 状態別 MSD 曲線 (6パネル)
    fig4_path_k2 = root_dir / output_dir / "hmm_state_msd_k2.svg"
    fig4_path = root_dir / output_dir / "hmm_state_msd.svg"
    plot_state_msd_hybrid(fitted_results, fig4_path_k2)
    plot_state_msd_hybrid(fitted_results, fig4_path)

    # 5. 状態別自己相関（VACF, OACF, SACF）グリッドプロット (3行6列)
    fig5_path_k2 = root_dir / output_dir / "hmm_autocorrelation_grid_k2.svg"
    fig5_path = root_dir / output_dir / "hmm_autocorrelation_grid.svg"
    plot_autocorrelation_grid_hybrid(fitted_results, fig5_path_k2)
    plot_autocorrelation_grid_hybrid(fitted_results, fig5_path)

    # CSVサマリー保存
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(output_dir / "hmm_state_parameters_summary.csv", index=False)
    if all_msd_fits:
        pd.concat(all_msd_fits, ignore_index=True).to_csv(output_dir / "hmm_state_msd_fits_summary.csv", index=False)
    if all_autocorr_records:
        pd.DataFrame(all_autocorr_records).to_csv(output_dir / "hmm_autocorrelation_summary.csv", index=False)

    print("\n[SUCCESS] All hybrid figures and summaries generated successfully in figure/hmm_model!")


if __name__ == "__main__":
    main()
