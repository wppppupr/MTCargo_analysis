"""
plot_binned_frame_scaling_master_curve.py

全粒子・全フレームのデータペア (x_{i,t}, v_{i,t}) を集約し、
横軸 x = R_c / xi を適切な幅のビン（Bin）に分割して、
各ビン内での平均速度 <v> および平均の標準誤差 (SEM) を算出してプロットするスクリプト。

【理論マスターカーブ】
  g(x) = (2 / x^2) * [1 - (1 + x) e^(-x)]
  <v_c> = v_0 * g(x)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
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

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": "#882255", "label": r"$0.63\,\mu\mathrm{m}$", "is_2state": True},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": "#CC6677", "label": r"$1.18\,\mu\mathrm{m}$", "is_2state": True},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": "#DDCC77", "label": r"$3.37\,\mu\mathrm{m}$", "is_2state": True},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": "#999933", "label": r"$5.00\,\mu\mathrm{m}$", "is_2state": False},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": "#117733", "label": r"$7.24\,\mu\mathrm{m}$", "is_2state": False},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": "#44AA99", "label": r"$20.0\,\mu\mathrm{m}$", "is_2state": False},
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


def extract_all_frame_pairs():
    # FOVサマリーから各FOVの局所相関長 xi を取得
    csv_fov = output_dir / 'scaling_fov_level_summary.csv'
    if not csv_fov.exists():
        raise FileNotFoundError(f"{csv_fov} not found. Run plot_fov_scaling_master_curve.py first.")
    
    df_fov_summary = pd.read_csv(csv_fov)
    fov_xi_map = {}
    for _, r in df_fov_summary.iterrows():
        fov_xi_map[(r['bead_name'], r['fov_name'])] = {
            'xi_run': r['xi_run_um'],
            'xi_tum': r['xi_tum_um'],
            'xi_all': r['xi_all_um'],
        }

    v0_run = df_fov_summary['v0_run'].iloc[0]
    v0_tum = df_fov_summary['v0_tum'].iloc[0]

    frame_records = []
    print("Collecting all frame pairs (x_{i,t}, v_{i,t})...")

    for b in BEADS_INFO:
        bname = b['name']
        dia = b['diameter_um']
        rc = b['radius_um']
        is_2st = b['is_2state']
        base = root_dir / bname
        
        edirs = [p for p in sorted(base.glob('*/*')) if p.is_dir() and (p / 'beads_tracks.csv').exists()]
        if not edirs:
            edirs = [p for p in sorted(base.glob('*')) if p.is_dir() and (p / 'beads_tracks.csv').exists()]

        # HMM学習（2状態のみ）
        all_tracks = []
        p_offset = 0
        for edir in edirs:
            df_t = pd.read_csv(edir / 'beads_tracks.csv')
            df_t['particle'] = df_t['particle'] + p_offset
            p_offset += int(df_t['particle'].max()) + 1
            all_tracks.append(df_t)
        df_all_tracks = pd.concat(all_tracks, ignore_index=True)

        hmm_model = None
        if is_2st:
            X, lengths, _ = hc.extract_hmm_features(df_all_tracks, tau=1, scale=0.11, frame_interval=4.0, epsilon=1e-3)
            hmm_model = hc.CargoGaussianHMM(n_components=2, covariance_type='full', epsilon=1e-3, random_state=42)
            hmm_model.fit(X, lengths=lengths)

        for edir in edirs:
            fov_name = edir.name
            xi_info = fov_xi_map.get((bname, fov_name), {})
            xi_run = xi_info.get('xi_run', np.nan)
            xi_tum = xi_info.get('xi_tum', np.nan)
            xi_all = xi_info.get('xi_all', np.nan)

            df_t = pd.read_csv(edir / 'beads_tracks.csv')
            X_fov, lengths_fov, df_obs_fov = hc.extract_hmm_features(df_t, tau=1, scale=0.11, frame_interval=4.0, epsilon=1e-3)
            
            if is_2st and hmm_model is not None and len(X_fov) > 0:
                states = hmm_model.predict(X_fov, lengths=lengths_fov)
                df_obs_fov['state'] = states
            else:
                df_obs_fov['state'] = 1

            for _, row in df_obs_fov.iterrows():
                st = row['state']
                v_val = row['v']

                if is_2st:
                    if st == 1 and np.isfinite(xi_run) and xi_run > 0:
                        x_val = rc / xi_run
                        v_norm = v_val / v0_run
                        mode_str = 'Run'
                        v0_used = v0_run
                    elif st == 0 and np.isfinite(xi_tum) and xi_tum > 0:
                        x_val = rc / xi_tum
                        v_norm = v_val / v0_tum
                        mode_str = 'Tumble'
                        v0_used = v0_tum
                    else:
                        continue
                else:
                    if np.isfinite(xi_all) and xi_all > 0:
                        x_val = rc / xi_all
                        v_norm = v_val / v0_run
                        mode_str = 'Single-state'
                        v0_used = v0_run
                    else:
                        continue

                frame_records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'radius_um': rc,
                    'fov_name': fov_name,
                    'particle': int(row['particle']),
                    'frame': int(row['frame']),
                    'mode': mode_str,
                    'is_2state': is_2st,
                    'v': v_val,
                    'v0': v0_used,
                    'v_norm': v_norm,
                    'x': x_val,
                })

    df_frames = pd.DataFrame(frame_records)
    print(f"Extracted {len(df_frames)} valid frame data pairs.")
    return df_frames, v0_run, v0_tum


def compute_binned_statistics(df_data, value_col='v_norm', n_bins=18, bin_edges=None):
    """
    x = R_c / xi をビン分割し、平均値、標準誤差 (SEM)、標準偏差、データ点数を算出
    """
    if bin_edges is None:
        x_min = df_data['x'].min()
        x_max = df_data['x'].max()
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    bin_records = []
    for idx in range(len(bin_edges) - 1):
        x_low = bin_edges[idx]
        x_high = bin_edges[idx + 1]
        
        # 最後のビンは右端を含む
        if idx == len(bin_edges) - 2:
            sub = df_data[(df_data['x'] >= x_low) & (df_data['x'] <= x_high)]
        else:
            sub = df_data[(df_data['x'] >= x_low) & (df_data['x'] < x_high)]

        n_pts = len(sub)
        if n_pts == 0:
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


def make_binned_plots(df_frames, v0_run, v0_tum):
    # 適切なビン境界の定義 (x in [0.04, 1.50])
    bin_edges = np.linspace(0.04, 1.45, 16)

    # 1. 全データ統合ビン（Universal Data Collapse）
    df_bins_all = compute_binned_statistics(df_frames, value_col='v_norm', bin_edges=bin_edges)
    
    # 2. モード別ビン (実速度 v および 規格化速度 v/v0)
    df_bins_run = compute_binned_statistics(df_frames[df_frames['mode'] == 'Run'], value_col='v', bin_edges=bin_edges)
    df_bins_tum = compute_binned_statistics(df_frames[df_frames['mode'] == 'Tumble'], value_col='v', bin_edges=bin_edges)
    df_bins_single = compute_binned_statistics(df_frames[df_frames['mode'] == 'Single-state'], value_col='v', bin_edges=bin_edges)

    df_bins_run_norm = compute_binned_statistics(df_frames[df_frames['mode'] == 'Run'], value_col='v_norm', bin_edges=bin_edges)
    df_bins_tum_norm = compute_binned_statistics(df_frames[df_frames['mode'] == 'Tumble'], value_col='v_norm', bin_edges=bin_edges)
    df_bins_single_norm = compute_binned_statistics(df_frames[df_frames['mode'] == 'Single-state'], value_col='v_norm', bin_edges=bin_edges)

    # サマリー保存
    csv_bins_all = output_dir / 'scaling_binned_master_curve_summary.csv'
    df_bins_all.to_csv(csv_bins_all, index=False)
    print(f"Saved Binned Summary CSV: {csv_bins_all}")

    # 理論曲線
    x_theory = np.logspace(-2.5, 1.2, 500)
    g_theory = master_function(x_theory)
    v_theory_run = v0_run * g_theory
    v_theory_tum = v0_tum * g_theory

    # =========================================================================
    # Figure 1: 2-Panel Binned Master Curve Plot
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.4))

    # --- Panel (a): Real Velocity v [um/s] Binned by Mode ---
    # 背景の生フレーム密度クラウド（薄いドット）
    run_frames = df_frames[df_frames['mode'] == 'Run']
    tum_frames = df_frames[df_frames['mode'] == 'Tumble']
    single_frames = df_frames[df_frames['mode'] == 'Single-state']

    ax1.scatter(run_frames['x'], run_frames['v'], color='#d95f02', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)
    ax1.scatter(tum_frames['x'], tum_frames['v'], color='#7570b3', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)
    ax1.scatter(single_frames['x'], single_frames['v'], color='#117733', s=4, alpha=0.04, rasterized=True, edgecolors='none', zorder=1)

    # 理論曲線
    ax1.plot(x_theory, v_theory_run, color='#d95f02', lw=2.8, linestyle='-', zorder=4,
             label=r'Run Master Curve: $v_{0,\mathrm{run}} \cdot g(x)$ ($v_0 = ' + f'{v0_run:.3f}' + r'\,\mu\mathrm{m/s}$)')
    ax1.plot(x_theory, v_theory_tum, color='#7570b3', lw=2.8, linestyle='--', zorder=4,
             label=r'Tumble Master Curve: $v_{0,\mathrm{tum}} \cdot g(x)$ ($v_0 = ' + f'{v0_tum:.3f}' + r'\,\mu\mathrm{m/s}$)')

    # ビン分割平均点（平均 ± SEM エラーバー）
    if not df_bins_run.empty:
        ax1.errorbar(
            df_bins_run['x_mean'], df_bins_run['val_mean'], yerr=df_bins_run['val_sem'],
            fmt='o-', color='#d95f02', ecolor='black', elinewidth=1.8, capsize=4.5, capthick=1.4,
            markersize=9.0, markeredgecolor='black', markeredgewidth=1.3, zorder=6,
            label=f'Run Binned Mean $\\pm$ SEM ($N={len(run_frames):,}$ frames)'
        )
    if not df_bins_tum.empty:
        ax1.errorbar(
            df_bins_tum['x_mean'], df_bins_tum['val_mean'], yerr=df_bins_tum['val_sem'],
            fmt='s-', color='#7570b3', ecolor='black', elinewidth=1.8, capsize=4.5, capthick=1.4,
            markersize=8.5, markeredgecolor='black', markeredgewidth=1.3, zorder=6,
            label=f'Tumble Binned Mean $\\pm$ SEM ($N={len(tum_frames):,}$ frames)'
        )
    if not df_bins_single.empty:
        ax1.errorbar(
            df_bins_single['x_mean'], df_bins_single['val_mean'], yerr=df_bins_single['val_sem'],
            fmt='^-', color='#117733', ecolor='black', elinewidth=2.0, capsize=5.0, capthick=1.5,
            markersize=9.5, markeredgecolor='black', markeredgewidth=1.4, zorder=6,
            label=f'1-State Binned Mean $\\pm$ SEM ($N={len(single_frames):,}$ frames)'
        )

    ax1.set_xlabel(r"Scaled Radius $x = R_c / \xi$", fontsize=12.5, fontweight='bold')
    ax1.set_ylabel(r"Cargo Velocity $v$ [$\mu\mathrm{m/s}$]", fontsize=12.5, fontweight='bold')
    ax1.set_title(r"$\mathbf{(a)}$ Binned Frame Velocity vs $x = R_c / \xi$", fontsize=13.5, fontweight='bold')
    ax1.set_xlim(0.015, 1.55)
    ax1.set_ylim(-0.02, 0.45)
    ax1.grid(True, linestyle='--', alpha=0.45)
    ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

    # --- Panel (b): Universal Data Collapse Binned across All Frames ---
    # 背景の規格化生フレーム密度クラウド
    ax2.scatter(run_frames['x'], run_frames['v_norm'], color='#d95f02', s=4, alpha=0.035, rasterized=True, edgecolors='none', zorder=1)
    ax2.scatter(tum_frames['x'], tum_frames['v_norm'], color='#7570b3', s=4, alpha=0.035, rasterized=True, edgecolors='none', zorder=1)
    ax2.scatter(single_frames['x'], single_frames['v_norm'], color='#117733', s=4, alpha=0.035, rasterized=True, edgecolors='none', zorder=1)

    # 理論普遍マスターカーブ
    ax2.plot(x_theory, g_theory, color='#111111', lw=3.2, linestyle='-', zorder=4,
             label=r'Universal Master Curve: $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$')

    # モード別ビン曲線（細線）
    if not df_bins_run_norm.empty:
        ax2.plot(df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], color='#d95f02', lw=1.6, linestyle=':', alpha=0.85, zorder=5)
    if not df_bins_tum_norm.empty:
        ax2.plot(df_bins_tum_norm['x_mean'], df_bins_tum_norm['val_mean'], color='#7570b3', lw=1.6, linestyle=':', alpha=0.85, zorder=5)
    if not df_bins_single_norm.empty:
        ax2.plot(df_bins_single_norm['x_mean'], df_bins_single_norm['val_mean'], color='#117733', lw=1.6, linestyle=':', alpha=0.85, zorder=5)

    # 全データ統合ビン（太い黒丸＋誤差棒）
    ax2.errorbar(
        df_bins_all['x_mean'], df_bins_all['val_mean'], yerr=df_bins_all['val_sem'],
        fmt='D-', color='#000000', mfc='#e41a1c', mec='black', mew=1.5,
        ecolor='black', elinewidth=2.2, capsize=5.5, capthick=1.8,
        markersize=10.0, zorder=7,
        label=f'All-Frames Binned Mean $\\pm$ SEM ($N={len(df_frames):,}$ total frames)'
    )

    # モード別ビン点
    ax2.errorbar(
        df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], yerr=df_bins_run_norm['val_sem'],
        fmt='o', color='#d95f02', ecolor='#d95f02', elinewidth=1.2, capsize=3.0, capthick=1.0,
        markersize=6.5, markeredgecolor='black', markeredgewidth=0.8, alpha=0.9, zorder=6,
        label='Run Binned'
    )
    ax2.errorbar(
        df_bins_tum_norm['x_mean'], df_bins_tum_norm['val_mean'], yerr=df_bins_tum_norm['val_sem'],
        fmt='s', color='#7570b3', ecolor='#7570b3', elinewidth=1.2, capsize=3.0, capthick=1.0,
        markersize=6.0, markeredgecolor='black', markeredgewidth=0.8, alpha=0.9, zorder=6,
        label='Tumble Binned'
    )
    ax2.errorbar(
        df_bins_single_norm['x_mean'], df_bins_single_norm['val_mean'], yerr=df_bins_single_norm['val_sem'],
        fmt='^', color='#117733', ecolor='#117733', elinewidth=1.2, capsize=3.0, capthick=1.0,
        markersize=7.0, markeredgecolor='black', markeredgewidth=0.8, alpha=0.9, zorder=6,
        label='1-State Binned'
    )

    ax2.set_xlabel(r"Scaled Radius $x = R_c / \xi$", fontsize=12.5, fontweight='bold')
    ax2.set_ylabel(r"Normalized Binned Velocity $\langle v / v_0 \rangle$", fontsize=12.5, fontweight='bold')
    ax2.set_title(r"$\mathbf{(b)}$ Universal Data Collapse across $34,783$ Frame Pairs", fontsize=13.5, fontweight='bold')
    ax2.set_xlim(0.015, 1.55)
    ax2.set_ylim(-0.05, 1.8)
    ax2.grid(True, linestyle='--', alpha=0.45)
    ax2.legend(fontsize=8.2, loc='upper right', framealpha=0.92, ncol=2)

    formula_text = (
        r"$\mathbf{Universal\ Master\ Curve:}$" "\n"
        r"$\frac{\langle v_c \rangle}{v_0} = \frac{2}{x^2}\left[1 - (1+x)e^{-x}\right]$" "\n"
        f"Total frames: $N = {len(df_frames):,}$\n"
        f"Number of Bins: $15$ bins\n"
        r"$v_{0,\mathrm{run}} = " + f"{v0_run:.3f}" + r"\,\mu\mathrm{m/s}$" "\n"
        r"$v_{0,\mathrm{tum}} = " + f"{v0_tum:.3f}" + r"\,\mu\mathrm{m/s}$"
    )
    ax2.text(0.04, 0.96, formula_text, transform=ax2.transAxes, fontsize=9.0,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffdd', alpha=0.92, edgecolor='#ddcc77'))

    plt.tight_layout()
    p1_png = output_dir / 'scaling_binned_frame_master_curve_2panel.png'
    p1_svg = output_dir / 'scaling_binned_frame_master_curve_2panel.svg'
    fig.savefig(p1_png, dpi=300, bbox_inches='tight')
    fig.savefig(p1_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p1_png} and {p1_svg}")

    # =========================================================================
    # Figure 2: Standalone Binned Data Collapse (Publication Focus)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.2, 7.0))

    # 背景クラウド
    ax.scatter(run_frames['x'], run_frames['v_norm'], color='#d95f02', s=5, alpha=0.035, rasterized=True, edgecolors='none', zorder=1)
    ax.scatter(tum_frames['x'], tum_frames['v_norm'], color='#7570b3', s=5, alpha=0.035, rasterized=True, edgecolors='none', zorder=1)
    ax.scatter(single_frames['x'], single_frames['v_norm'], color='#117733', s=5, alpha=0.035, rasterized=True, edgecolors='none', zorder=1)

    # 理論曲線
    ax.plot(x_theory, g_theory, color='#111111', lw=3.4, linestyle='-', zorder=4,
            label=r'Theoretical Master Curve: $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$')

    # Taylor近似
    ax.plot(x_theory[x_theory <= 0.8], np.exp(-2.0/3.0 * x_theory[x_theory <= 0.8]), color='#666666', lw=1.8, linestyle='--', zorder=4,
            label=r'Exponential Approximation: $\exp\left(-\frac{2}{3}x\right)$')

    # 全フレーム統合ビン
    ax.errorbar(
        df_bins_all['x_mean'], df_bins_all['val_mean'], yerr=df_bins_all['val_sem'],
        fmt='D-', color='#000000', mfc='#e41a1c', mec='black', mew=1.6,
        ecolor='black', elinewidth=2.4, capsize=6.0, capthick=2.0,
        markersize=11.0, zorder=7,
        label=f'All-Frames Binned Mean $\\pm$ SEM ($N={len(df_frames):,}$ frames, 15 bins)'
    )

    # モード別ビン点
    ax.errorbar(
        df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], yerr=df_bins_run_norm['val_sem'],
        fmt='o', color='#d95f02', ecolor='#d95f02', elinewidth=1.4, capsize=3.5, capthick=1.1,
        markersize=8.0, markeredgecolor='black', markeredgewidth=1.0, alpha=0.9, zorder=6,
        label=f'Run Binned ($N={len(run_frames):,}$)'
    )
    ax.errorbar(
        df_bins_tum_norm['x_mean'], df_bins_tum_norm['val_mean'], yerr=df_bins_tum_norm['val_sem'],
        fmt='s', color='#7570b3', ecolor='#7570b3', elinewidth=1.4, capsize=3.5, capthick=1.1,
        markersize=7.5, markeredgecolor='black', markeredgewidth=1.0, alpha=0.9, zorder=6,
        label=f'Tumble Binned ($N={len(tum_frames):,}$)'
    )
    ax.errorbar(
        df_bins_single_norm['x_mean'], df_bins_single_norm['val_mean'], yerr=df_bins_single_norm['val_sem'],
        fmt='^', color='#117733', ecolor='#117733', elinewidth=1.4, capsize=3.5, capthick=1.1,
        markersize=8.5, markeredgecolor='black', markeredgewidth=1.0, alpha=0.9, zorder=6,
        label=f'1-State Binned ($N={len(single_frames):,}$)'
    )

    ax.set_xlabel(r"Scaled Cargo Radius $x = R_c / \xi$", fontsize=13, fontweight='bold')
    ax.set_ylabel(r"Normalized Velocity $\langle v_c \rangle / v_0$", fontsize=13, fontweight='bold')
    ax.set_title(r"Binned Universal Master Curve Collapse across All Frame Pairs", fontsize=14, fontweight='bold')
    ax.set_xlim(0.015, 1.55)
    ax.set_ylim(-0.05, 1.6)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(fontsize=9.0, loc='upper right', framealpha=0.92, ncol=2)

    ax.text(0.04, 0.96, formula_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.55', facecolor='#ffffdd', alpha=0.95, edgecolor='#ddcc77'))

    plt.tight_layout()
    p2_png = output_dir / 'scaling_binned_frame_master_curve_datacollapse.png'
    p2_svg = output_dir / 'scaling_binned_frame_master_curve_datacollapse.svg'
    fig.savefig(p2_png, dpi=300, bbox_inches='tight')
    fig.savefig(p2_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p2_png} and {p2_svg}")

    # =========================================================================
    # Figure 3: Log-Log Scale Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 6.4))

    ax.scatter(run_frames['x'], run_frames['v_norm'], color='#d95f02', s=4, alpha=0.03, rasterized=True, edgecolors='none', zorder=1)
    ax.scatter(tum_frames['x'], tum_frames['v_norm'], color='#7570b3', s=4, alpha=0.03, rasterized=True, edgecolors='none', zorder=1)
    ax.scatter(single_frames['x'], single_frames['v_norm'], color='#117733', s=4, alpha=0.03, rasterized=True, edgecolors='none', zorder=1)

    ax.plot(x_theory, g_theory, color='#111111', lw=3.2, linestyle='-', zorder=4,
            label=r'Master Curve $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$')

    x_asymp_small = np.linspace(0.02, 0.3, 100)
    ax.plot(x_asymp_small, 1.0 - (2.0/3.0)*x_asymp_small, 'k:', lw=1.8, label=r'Small-$x$ limit: $1 - \frac{2}{3}x$')

    x_asymp_large = np.linspace(0.6, 2.5, 100)
    ax.plot(x_asymp_large, 2.0 / (x_asymp_large ** 2), 'k--', lw=1.8, label=r'Large-$x$ asymptote: $2/x^2$')

    # 全フレーム統合ビン
    ax.errorbar(
        df_bins_all['x_mean'], df_bins_all['val_mean'], yerr=df_bins_all['val_sem'],
        fmt='D-', color='#000000', mfc='#e41a1c', mec='black', mew=1.6,
        ecolor='black', elinewidth=2.4, capsize=6.0, capthick=2.0,
        markersize=11.0, zorder=7,
        label=f'All-Frames Binned Mean $\\pm$ SEM ($N={len(df_frames):,}$)'
    )

    ax.errorbar(
        df_bins_run_norm['x_mean'], df_bins_run_norm['val_mean'], yerr=df_bins_run_norm['val_sem'],
        fmt='o', color='#d95f02', ecolor='#d95f02', elinewidth=1.4, capsize=3.5, capthick=1.1,
        markersize=7.5, markeredgecolor='black', markeredgewidth=1.0, alpha=0.9, zorder=6,
        label='Run Binned'
    )
    ax.errorbar(
        df_bins_tum_norm['x_mean'], df_bins_tum_norm['val_mean'], yerr=df_bins_tum_norm['val_sem'],
        fmt='s', color='#7570b3', ecolor='#7570b3', elinewidth=1.4, capsize=3.5, capthick=1.1,
        markersize=7.0, markeredgecolor='black', markeredgewidth=1.0, alpha=0.9, zorder=6,
        label='Tumble Binned'
    )
    ax.errorbar(
        df_bins_single_norm['x_mean'], df_bins_single_norm['val_mean'], yerr=df_bins_single_norm['val_sem'],
        fmt='^', color='#117733', ecolor='#117733', elinewidth=1.4, capsize=3.5, capthick=1.1,
        markersize=8.0, markeredgecolor='black', markeredgewidth=1.0, alpha=0.9, zorder=6,
        label='1-State Binned'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Scaled Radius $x = R_c / \xi$", fontsize=12.5, fontweight='bold')
    ax.set_ylabel(r"Normalized Velocity $\langle v_c \rangle / v_0$", fontsize=12.5, fontweight='bold')
    ax.set_title(r"Log-Log Binned Universal Master Curve", fontsize=13.5, fontweight='bold')
    ax.set_xlim(0.03, 2.0)
    ax.set_ylim(0.1, 2.0)
    ax.grid(True, which='both', linestyle='--', alpha=0.45)
    ax.legend(fontsize=8.2, loc='lower left', framealpha=0.92, ncol=2)

    plt.tight_layout()
    p3_png = output_dir / 'scaling_binned_frame_master_curve_loglog.png'
    p3_svg = output_dir / 'scaling_binned_frame_master_curve_loglog.svg'
    fig.savefig(p3_png, dpi=300, bbox_inches='tight')
    fig.savefig(p3_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p3_png} and {p3_svg}")


def main():
    print("=== Extracting all frame pairs ===")
    df_frames, v0_run, v0_tum = extract_all_frame_pairs()

    print("=== Generating Binned Scaling Master Curve Plots ===")
    make_binned_plots(df_frames, v0_run, v0_tum)
    print("All binned master curve figures created successfully!")


if __name__ == '__main__':
    main()
