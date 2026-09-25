"""
plot_instantaneous_scaling_master_curve.py

各フレーム・各粒子の近傍における瞬時局所微小管相関長 xi_{i,t} を直接フィッティングして算出し、
各フレームの瞬時スケーリング変数 x_{i,t} = R_c / xi_{i,t} と瞬時速度 v_{i,t} を集約。
横軸 x を適切な幅のビン（Bin）に分割し、各ビン内での平均速度 <v> および平均の標準誤差 (SEM) を算出してプロットするスクリプト。

【対象】
- 2状態（Run / Tumble）を示す小粒子系（0.63 um, 1.18 um, 3.37 um）に特化。
- 5 um, 7 um (1-State) および Tumble のフィッティング曲線を削除し、Run のマスターカーブを中心にプロット。

【理論マスターカーブ】
  g(x) = (2 / x^2) * [1 - (1 + x) e^(-x)]
  <v_c>_Run = v_0 * g(x)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# プロジェクト設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import hmm_cargo as hc

# スタイル適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))

# データルート
POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]
root_dir = None
for r in POSSIBLE_ROOTS:
    if r.exists():
        root_dir = r
        break
if root_dir is None:
    raise FileNotFoundError("Data root directory not found.")

output_dir = root_dir / 'figure' / 'scaling'
output_dir.mkdir(parents=True, exist_ok=True)

# 2状態粒子（0.63, 1.18, 3.37 um）のみを対象とする
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": "#882255", "label": r"$0.63\,\mu\mathrm{m}$", "is_2state": True},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": "#CC6677", "label": r"$1.18\,\mu\mathrm{m}$", "is_2state": True},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": "#DDCC77", "label": r"$3.37\,\mu\mathrm{m}$", "is_2state": True},
]


def master_function(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    small_mask = (x < 1e-4)
    if np.any(small_mask):
        xs = x[small_mask]
        g[small_mask] = 1.0 - (2.0 / 3.0) * xs + (1.0 / 4.0) * (xs ** 2) - (1.0 / 15.0) * (xs ** 3)
    reg_mask = ~small_mask
    if np.any(reg_mask):
        xr_ = x[reg_mask]
        g[reg_mask] = (2.0 / (xr_ ** 2)) * (1.0 - (1.0 + xr_) * np.exp(-xr_))
    return g


def fit_instantaneous_xi(r_um, c_curve, min_r, max_r=25.0, min_corr_threshold=0.05):
    """
    局所空間配向相関 C(r) の縦軸の対数をとってフィッティングし、相関長 xi を推定する。

    指数減衰 C(r) = a * exp(-r / xi) の両辺の対数をとると
        ln C(r) = ln a - r / xi
    となるため、ln C(r) の r に対する線形回帰の傾きから xi = -1 / slope を得る。
    対数をとれない C(r) <= 0 の点や C(r) < min_corr_threshold の点は除外する。
    """
    r_arr = np.asarray(r_um, dtype=float)
    c_arr = np.asarray(c_curve, dtype=float)
    mask = (r_arr >= min_r) & (r_arr <= max_r) & np.isfinite(c_arr) & (c_arr >= min_corr_threshold)
    if np.sum(mask) < 4:
        return np.nan
    r_fit = r_arr[mask]
    c_fit = c_arr[mask]
    try:
        slope, _ = np.polyfit(r_fit, np.log(c_fit), 1)
    except Exception:
        return np.nan
    if not np.isfinite(slope) or slope >= -0.01:
        return np.nan
    xi_val = -1.0 / slope
    if 0.1 <= xi_val <= 50.0:
        return float(xi_val)
    return np.nan


def extract_instantaneous_frame_pairs():
    frame_records = []
    print("Extracting instantaneous local correlation length xi_{i,t} for 2-state beads (0.63, 1.18, 3.37 um)...")

    for b in BEADS_INFO:
        bname = b['name']
        dia = b['diameter_um']
        rc = b['radius_um']
        base = root_dir / bname
        
        edirs = [p for p in sorted(base.glob('*/*')) if p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / 'angular_correlation_w.zarr').exists()]
        if not edirs:
            edirs = [p for p in sorted(base.glob('*')) if p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / 'angular_correlation_w.zarr').exists()]

        # HMM学習（全軌跡統合）
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
            fov_name = edir.name
            df_t = pd.read_csv(edir / 'beads_tracks.csv')
            X_fov, lengths_fov, df_obs_fov = hc.extract_hmm_features(df_t, tau=1, scale=0.11, frame_interval=4.0, epsilon=1e-3)
            
            if len(X_fov) > 0:
                states = hmm_model.predict(X_fov, lengths=lengths_fov)
                df_obs_fov['state'] = states
            else:
                df_obs_fov['state'] = 1

            # Load Zarr
            ds_w = xr.open_zarr(str(edir / 'angular_correlation_w.zarr'), consolidated=False)
            r_um = ds_w.coords['distance'].values * 0.11
            frames_coord = ds_w.coords['frame'].values
            particles_coord = ds_w.coords['particle'].values
            arr_total = ds_w['angular_correlation'].values

            f_to_idx = {int(f_val): idx for idx, f_val in enumerate(frames_coord)}
            p_to_idx = {int(p_val): idx for idx, p_val in enumerate(particles_coord)}

            for _, row in df_obs_fov.iterrows():
                f_val = int(row['frame'])
                p_val = int(row['particle'])
                st = int(row['state'])
                v_val = float(row['v'])

                if f_val not in f_to_idx or p_val not in p_to_idx:
                    continue

                f_idx = f_to_idx[f_val]
                p_idx = p_to_idx[p_val]
                c_curve = arr_total[:, f_idx, p_idx]

                xi_local = fit_instantaneous_xi(r_um, c_curve, min_r, max_r=25.0)
                if not np.isfinite(xi_local) or xi_local <= 0.05:
                    continue

                x_local = rc / xi_local
                mode_str = 'Run' if (st == 1) else 'Tumble'

                frame_records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'radius_um': rc,
                    'fov_name': fov_name,
                    'particle': p_val,
                    'frame': f_val,
                    'mode': mode_str,
                    'is_2state': True,
                    'xi_local_um': xi_local,
                    'x': x_local,
                    'v': v_val,
                })

    df_inst = pd.DataFrame(frame_records)
    print(f"Extracted {len(df_inst)} valid instantaneous frame pairs.")

    # フィッティングによる Run の固有基準速度 v0 の決定
    run_valid = df_inst[df_inst['mode'] == 'Run']
    g_run = master_function(run_valid['x'].values)
    v0_run = np.sum(run_valid['v'].values * g_run) / np.sum(g_run ** 2)

    print(f"Fitted v0_run = {v0_run:.4f} um/s")

    # 規格化速度の付与（Run基準速度 v0_run で規格化）
    df_inst['v0'] = v0_run
    df_inst['v_norm'] = df_inst['v'] / v0_run

    # CSV保存
    csv_frames = output_dir / 'scaling_instantaneous_frames_data.csv'
    df_inst.to_csv(csv_frames, index=False)
    print(f"Saved instantaneous frame pairs CSV: {csv_frames}")

    return df_inst, v0_run


def compute_binned_statistics(df_data, value_col='v_norm', bin_edges=None):
    if bin_edges is None:
        x_min = df_data['x'].quantile(0.005)
        x_max = df_data['x'].quantile(0.995)
        bin_edges = np.linspace(x_min, x_max, 16)

    bin_records = []
    for idx in range(len(bin_edges) - 1):
        x_low = bin_edges[idx]
        x_high = bin_edges[idx + 1]
        
        if idx == len(bin_edges) - 2:
            sub = df_data[(df_data['x'] >= x_low) & (df_data['x'] <= x_high)]
        else:
            sub = df_data[(df_data['x'] >= x_low) & (df_data['x'] < x_high)]

        n_pts = len(sub)
        if n_pts < 5:
            continue

        x_mean = sub['x'].mean()
        x_center = 0.5 * (x_low + x_high)
        val_mean = sub[value_col].mean()
        val_std = sub[value_col].std()
        val_sem = sub[value_col].sem()
        val_median = sub[value_col].median()
        val_q25 = sub[value_col].quantile(0.25)
        val_q75 = sub[value_col].quantile(0.75)

        theory_g = master_function(x_mean)

        bin_records.append({
            'bin_idx': idx + 1,
            'x_bin_low': x_low,
            'x_bin_high': x_high,
            'x_bin_center': x_center,
            'x_mean': x_mean,
            'n_frames': n_pts,
            'val_mean': val_mean,
            'val_sem': val_sem,
            'val_std': val_std,
            'val_median': val_median,
            'val_q25': val_q25,
            'val_q75': val_q75,
            'theory_g': theory_g,
        })

    return pd.DataFrame(bin_records)


def make_instantaneous_plots(df_inst, v0_run):
    # ビン分割境界 (x in [0.01, 1.20])
    bin_edges = np.linspace(0.01, 1.15, 15)

    # モード別ビン (実速度 v および 規格化速度 v/v0)
    run_frames = df_inst[df_inst['mode'] == 'Run']
    tum_frames = df_inst[df_inst['mode'] == 'Tumble']

    df_bins_run = compute_binned_statistics(run_frames, value_col='v', bin_edges=bin_edges)
    df_bins_tum = compute_binned_statistics(tum_frames, value_col='v', bin_edges=bin_edges)

    df_bins_run_norm = compute_binned_statistics(run_frames, value_col='v_norm', bin_edges=bin_edges)
    df_bins_tum_norm = compute_binned_statistics(tum_frames, value_col='v_norm', bin_edges=bin_edges)

    # サマリー保存
    csv_bins_run = output_dir / 'scaling_instantaneous_binned_summary.csv'
    df_bins_run_norm.to_csv(csv_bins_run, index=False)
    print(f"Saved Instantaneous Binned Summary CSV: {csv_bins_run}")

    # 理論曲線（Run Master Curve のみ）
    x_theory = np.logspace(-2.5, 0.8, 500)
    g_theory = master_function(x_theory)
    v_theory_run = v0_run * g_theory

    # =========================================================================
    # Figure 1: 2-Panel Binned Master Curve Plot
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.2))

    # --- Panel (a): Real Velocity v [um/s] ---
    ax1.scatter(run_frames['x'], run_frames['v'], color='#d95f02', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)
    ax1.scatter(tum_frames['x'], tum_frames['v'], color='#7570b3', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)

    # Run 理論曲線のみプロット
    ax1.plot(x_theory, v_theory_run, color='#d95f02', lw=3.0, linestyle='-', zorder=4,
             label=r'Run Master Curve: $\langle v_c \rangle = v_{0} \cdot g(x)$ ($v_0 = ' + f'{v0_run:.3f}' + r'\,\mu\mathrm{m/s}$)')

    # Run / Tumble ビン点
    if not df_bins_run.empty:
        ax1.errorbar(
            df_bins_run['x_mean'], df_bins_run['val_mean'], yerr=df_bins_run['val_sem'],
            fmt='o-', color='#d95f02', ecolor='black', elinewidth=2.0, capsize=5.0, capthick=1.5,
            markersize=9.5, markeredgecolor='black', markeredgewidth=1.4, zorder=6,
            label=f'Run Binned Mean $\\pm$ SEM ($N={len(run_frames):,}$ frames)'
        )
    if not df_bins_tum.empty:
        ax1.errorbar(
            df_bins_tum['x_mean'], df_bins_tum['val_mean'], yerr=df_bins_tum['val_sem'],
            fmt='s--', color='#7570b3', ecolor='black', elinewidth=1.6, capsize=4.0, capthick=1.2,
            markersize=8.0, markeredgecolor='black', markeredgewidth=1.2, alpha=0.85, zorder=5,
            label=f'Tumble Binned Mean $\\pm$ SEM ($N={len(tum_frames):,}$ frames)'
        )

    ax1.set_xlabel(r"Instantaneous Scaled Radius $x_{i,t} = R_c / \xi_{i,t}$", fontsize=12.5, fontweight='bold')
    ax1.set_ylabel(r"Cargo Velocity $v$ [$\mu\mathrm{m/s}$]", fontsize=12.5, fontweight='bold')
    ax1.set_title(r"$\mathbf{(a)}$ Instantaneous Velocity vs $x_{i,t} = R_c / \xi_{i,t}$", fontsize=13.5, fontweight='bold')
    ax1.set_xlim(0.005, 1.20)
    ax1.set_ylim(-0.02, 0.40)
    ax1.grid(True, linestyle='--', alpha=0.45)
    ax1.legend(fontsize=9.0, loc='upper right', framealpha=0.92)

    # --- Panel (b): Universal Data Collapse (Normalized by v0_run) ---
    ax2.scatter(run_frames['x'], run_frames['v_norm'], color='#d95f02', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)
    ax2.scatter(tum_frames['x'], tum_frames['v_norm'], color='#7570b3', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)

    # 理論普遍マスターカーブ
    ax2.plot(x_theory, g_theory, color='#111111', lw=3.2, linestyle='-', zorder=4,
             label=r'Universal Master Curve: $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$')

    # Run ビン点（太い赤丸＋黒枠）
    if not df_bins_run_norm.empty:
        ax2.errorbar(
            df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], yerr=df_bins_run_norm['val_sem'],
            fmt='o-', color='#d95f02', mfc='#e41a1c', mec='black', mew=1.5,
            ecolor='black', elinewidth=2.2, capsize=5.5, capthick=1.8,
            markersize=10.0, zorder=7,
            label=f'Run Binned Mean $\\pm$ SEM ($N={len(run_frames):,}$ frames)'
        )
    if not df_bins_tum_norm.empty:
        ax2.errorbar(
            df_bins_tum_norm['x_mean'], df_bins_tum_norm['val_mean'], yerr=df_bins_tum_norm['val_sem'],
            fmt='s--', color='#7570b3', ecolor='#7570b3', elinewidth=1.4, capsize=4.0, capthick=1.2,
            markersize=7.5, markeredgecolor='black', markeredgewidth=1.0, alpha=0.85, zorder=5,
            label=f'Tumble Binned Mean ($v / v_{{0,\\mathrm{{run}}}}$)'
        )

    ax2.set_xlabel(r"Instantaneous Scaled Radius $x_{i,t} = R_c / \xi_{i,t}$", fontsize=12.5, fontweight='bold')
    ax2.set_ylabel(r"Normalized Velocity $\langle v / v_0 \rangle$", fontsize=12.5, fontweight='bold')
    ax2.set_title(r"$\mathbf{(b)}$ Universal Master Curve Data Collapse (Run Mode)", fontsize=13.5, fontweight='bold')
    ax2.set_xlim(0.005, 1.20)
    ax2.set_ylim(-0.05, 1.6)
    ax2.grid(True, linestyle='--', alpha=0.45)
    ax2.legend(fontsize=9.0, loc='upper right', framealpha=0.92)

    formula_text = (
        r"$\mathbf{Run\ Master\ Curve:}$" "\n"
        r"$\frac{\langle v_c \rangle_{\mathrm{Run}}}{v_0} = \frac{2}{x^2}\left[1 - (1+x)e^{-x}\right]$" "\n"
        r"$x_{i,t} = R_c / \xi_{i,t}\ (\mathrm{Instantaneous})$" "\n"
        f"Run frames: $N = {len(run_frames):,}$\n"
        r"$v_{0,\mathrm{run}} = " + f"{v0_run:.3f}" + r"\,\mu\mathrm{m/s}$"
    )
    ax2.text(0.04, 0.96, formula_text, transform=ax2.transAxes, fontsize=9.2,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffdd', alpha=0.92, edgecolor='#ddcc77'))

    plt.tight_layout()
    p1_png = output_dir / 'scaling_instantaneous_binned_master_curve_2panel.png'
    p1_svg = output_dir / 'scaling_instantaneous_binned_master_curve_2panel.svg'
    fig.savefig(p1_png, dpi=300, bbox_inches='tight')
    fig.savefig(p1_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p1_png} and {p1_svg}")

    # =========================================================================
    # Figure 2: Standalone Run Data Collapse Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    ax.scatter(run_frames['x'], run_frames['v_norm'], color='#d95f02', s=5, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)
    ax.scatter(tum_frames['x'], tum_frames['v_norm'], color='#7570b3', s=5, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)

    ax.plot(x_theory, g_theory, color='#111111', lw=3.4, linestyle='-', zorder=4,
            label=r'Theoretical Master Curve: $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$')

    ax.plot(x_theory[x_theory <= 0.8], np.exp(-2.0/3.0 * x_theory[x_theory <= 0.8]), color='#666666', lw=1.8, linestyle='--', zorder=4,
            label=r'Exponential Approx: $\exp\left(-\frac{2}{3}x\right)$')

    ax.errorbar(
        df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], yerr=df_bins_run_norm['val_sem'],
        fmt='o-', color='#d95f02', mfc='#e41a1c', mec='black', mew=1.6,
        ecolor='black', elinewidth=2.4, capsize=6.0, capthick=2.0,
        markersize=11.0, zorder=7,
        label=f'Run Binned Mean $\\pm$ SEM ($N={len(run_frames):,}$ frames)'
    )

    ax.errorbar(
        df_bins_tum_norm['x_mean'], df_bins_tum_norm['val_mean'], yerr=df_bins_tum_norm['val_sem'],
        fmt='s--', color='#7570b3', ecolor='#7570b3', elinewidth=1.4, capsize=4.0, capthick=1.2,
        markersize=7.5, markeredgecolor='black', markeredgewidth=1.0, alpha=0.85, zorder=5,
        label=f'Tumble Binned ($N={len(tum_frames):,}$ frames)'
    )

    ax.set_xlabel(r"Instantaneous Scaled Radius $x_{i,t} = R_c / \xi_{i,t}$", fontsize=13, fontweight='bold')
    ax.set_ylabel(r"Normalized Velocity $\langle v_c \rangle / v_0$", fontsize=13, fontweight='bold')
    ax.set_title(r"Run Transport Master Curve with Instantaneous Local $\xi_{i,t}$", fontsize=13.5, fontweight='bold')
    ax.set_xlim(0.005, 1.20)
    ax.set_ylim(-0.05, 1.5)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(fontsize=9.2, loc='upper right', framealpha=0.92)

    ax.text(0.04, 0.96, formula_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.55', facecolor='#ffffdd', alpha=0.95, edgecolor='#ddcc77'))

    plt.tight_layout()
    p2_png = output_dir / 'scaling_instantaneous_binned_datacollapse.png'
    p2_svg = output_dir / 'scaling_instantaneous_binned_datacollapse.svg'
    fig.savefig(p2_png, dpi=300, bbox_inches='tight')
    fig.savefig(p2_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p2_png} and {p2_svg}")

    # =========================================================================
    # Figure 3: Log-Log Scale Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 6.4))

    ax.scatter(run_frames['x'], run_frames['v_norm'], color='#d95f02', s=4, alpha=0.03, rasterized=True, edgecolors='none', zorder=1)

    ax.plot(x_theory, g_theory, color='#111111', lw=3.2, linestyle='-', zorder=4,
            label=r'Master Curve $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$')

    x_asymp_small = np.linspace(0.015, 0.3, 100)
    ax.plot(x_asymp_small, 1.0 - (2.0/3.0)*x_asymp_small, 'k:', lw=1.8, label=r'Small-$x$ limit: $1 - \frac{2}{3}x$')

    x_asymp_large = np.linspace(0.5, 1.5, 100)
    ax.plot(x_asymp_large, 2.0 / (x_asymp_large ** 2), 'k--', lw=1.8, label=r'Large-$x$ asymptote: $2/x^2$')

    ax.errorbar(
        df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], yerr=df_bins_run_norm['val_sem'],
        fmt='o-', color='#d95f02', mfc='#e41a1c', mec='black', mew=1.6,
        ecolor='black', elinewidth=2.4, capsize=6.0, capthick=2.0,
        markersize=10.0, zorder=7,
        label=f'Run Binned Mean $\\pm$ SEM ($N={len(run_frames):,}$)'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Instantaneous Scaled Radius $x_{i,t} = R_c / \xi_{i,t}$", fontsize=12.5, fontweight='bold')
    ax.set_ylabel(r"Normalized Velocity $\langle v_c \rangle / v_0$", fontsize=12.5, fontweight='bold')
    ax.set_title(r"Log-Log Run Master Curve ($0.63, 1.18, 3.37\,\mu\mathrm{m}$)", fontsize=13.5, fontweight='bold')
    ax.set_xlim(0.01, 1.5)
    ax.set_ylim(0.2, 1.8)
    ax.grid(True, which='both', linestyle='--', alpha=0.45)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.92)

    plt.tight_layout()
    p3_png = output_dir / 'scaling_instantaneous_binned_master_curve_loglog.png'
    p3_svg = output_dir / 'scaling_instantaneous_binned_master_curve_loglog.svg'
    fig.savefig(p3_png, dpi=300, bbox_inches='tight')
    fig.savefig(p3_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p3_png} and {p3_svg}")


def main():
    print("=== Extracting instantaneous local correlation lengths and speeds ===")
    df_inst, v0_run = extract_instantaneous_frame_pairs()

    print("=== Generating Instantaneous Binned Scaling Master Curve Plots ===")
    make_instantaneous_plots(df_inst, v0_run)
    print("All instantaneous binned master curve figures created successfully!")


if __name__ == '__main__':
    main()
