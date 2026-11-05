#!/usr/bin/env python3
"""
calc_and_plot_cargo_flow_profile.py

貨物とくっついている微小管は遅くなることを実証するための解析スクリプト:
貨物まわりの速さの動径分布 |u|(r) を計算・可視化する。

2通りの横軸定義でグラフを出力:
1. 貨物の縁からの距離 r = dist - R_c (および r/xi)
2. 貨物中心からの距離 dist (および dist/xi)  ※ -R_c を無くしたもの
いずれも貨物直下 (dist <= R_c) は完全にマスクされ、dist = R_c からプロットが始まります。
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ワークスペースおよび保存先設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "flow_profile_around_cargo",
    ROOT_DIR / "figure" / "flow_profile_around_cargo"
]
CACHE_DIR = CURRENT_DIR / "scratch"

# スタイル適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

# ビーズ基本情報 (MSD.py 準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$", "rc_px": 3.0},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$", "rc_px": 6.0},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$", "rc_px": 15.0},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$", "rc_px": 25.0},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$", "rc_px": 35.0},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$", "rc_px": 95.0},
]

# 実験対象ディレクトリ
TARGET_DIRS = [
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um001_crop",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um_crop",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um_crop"
]

SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0
V_UNIT = SCALE_UM_PER_PX / DT_SEC # um/s

def setup_matplotlib():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10.0,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })

def save_fig(fig, base_name):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")

def load_xi_dict():
    candidates = [
        CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        ROOT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
    ]
    xi_map = {}
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    exp_name = str(r['exp_dir']).strip()
                    xi_val = float(r['xi_um'])
                    xi_map[exp_name] = xi_val
                print(f"Loaded {len(xi_map)} xi values from {p.name}")
                break
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
    return xi_map

def process_single_experiment(exp_dir_str: str, b_info: dict, max_r_edge_um: float = 35.0, bin_width_um: float = 0.50, max_samples: int = 25):
    """
    1実験ディレクトリについて、貨物縁からの動径速度プロファイル |u|(r) を計算。
    遠方流速 u_inf は、十分遠方（r >= 20 um）の漸近プラトー平均として算出。
    """
    exp_dir = Path(exp_dir_str)
    h5_path = exp_dir / "GFP_flows.h5"
    csv_path = exp_dir / "beads_tracks.csv"
    
    if not h5_path.exists() or not csv_path.exists():
        print(f"Skipping {exp_dir.name}: missing files.")
        return None
        
    df_tracks = pd.read_csv(csv_path)
    if 'particle' not in df_tracks.columns:
        df_tracks['particle'] = 0
        
    Rc_px = b_info['rc_px']
    max_r_edge_px = max_r_edge_um / SCALE_UM_PER_PX
    bin_width_px = bin_width_um / SCALE_UM_PER_PX
    
    bins_px = np.arange(0, max_r_edge_px + bin_width_px, bin_width_px)
    n_bins = len(bins_px) - 1
    
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    with h5py.File(str(h5_path), 'r') as hf:
        dataset_key = list(hf.keys())[0]
        flows = hf[dataset_key]
        num_frames = flows.shape[0]
        H, W = flows.shape[2], flows.shape[3]
        
        available_frames = [f for f in sorted(df_tracks['frame'].unique()) if f < num_frames]
        n_samples = min(len(available_frames), max_samples)
        sample_indices = np.linspace(0, len(available_frames) - 1, n_samples, dtype=int)
        sampled_frames = [available_frames[i] for i in sample_indices]
        
        for fr in sampled_frames:
            flow_fr = flows[fr]
            u = flow_fr[0].astype(np.float32)
            v = flow_fr[1].astype(np.float32)
            speed = np.sqrt(u**2 + v**2)
            
            sub_tracks = df_tracks[df_tracks['frame'] == fr]
            for _, row in sub_tracks.iterrows():
                bx, by = row['x'], row['y']
                max_dist = Rc_px + max_r_edge_px
                
                x_min = max(0, int(np.floor(bx - max_dist)))
                x_max = min(W, int(np.ceil(bx + max_dist)) + 1)
                y_min = max(0, int(np.floor(by - max_dist)))
                y_max = min(H, int(np.ceil(by + max_dist)) + 1)
                
                sub_spd = speed[y_min:y_max, x_min:x_max]
                y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
                dist = np.sqrt((x_idx - bx)**2 + (y_idx - by)**2)
                
                # 貨物の縁からの距離 r = dist - Rc_px
                # 貨物真下 (dist <= Rc_px, r <= 0) は完全にマスク
                r_edge = dist - Rc_px
                valid = (r_edge >= 0) & (r_edge < max_r_edge_px)
                
                if not np.any(valid):
                    continue
                    
                b_idx = np.digitize(r_edge[valid], bins_px) - 1
                b_c = np.bincount(b_idx, minlength=n_bins)
                b_s = np.bincount(b_idx, weights=sub_spd[valid], minlength=n_bins)
                
                bin_counts += b_c
                bin_sums += b_s
                
    mean_speed_px = bin_sums / np.maximum(bin_counts, 1)
    mean_speed_um_s = mean_speed_px * V_UNIT
    
    bin_centers_px = 0.5 * (bins_px[:-1] + bins_px[1:])
    bin_centers_um = bin_centers_px * SCALE_UM_PER_PX
    
    # 遠方漸近流速 u_inf の定義:
    # 貨物の影響が十分に消滅した遠方領域 (r >= 20 um) におけるプロファイル平均
    far_mask = bin_centers_um >= 20.0
    if np.sum(bin_counts[far_mask]) > 100:
        u_inf = np.average(mean_speed_um_s[far_mask], weights=bin_counts[far_mask])
    else:
        u_inf = np.mean(mean_speed_um_s[int(n_bins * 0.8):])
        
    norm_speed = mean_speed_um_s / u_inf if (u_inf > 0 and np.isfinite(u_inf)) else np.zeros_like(mean_speed_um_s)
    
    return {
        'exp_dir': exp_dir_str,
        'bead_name': b_info['name'],
        'diameter_um': b_info['diameter_um'],
        'radius_um': b_info['radius_um'],
        'r_edge_um': bin_centers_um,
        'dist_um': bin_centers_um + b_info['radius_um'], # 貨物中心からの距離 dist = r + R_c
        'speed_um_s': mean_speed_um_s,
        'u_inf': u_inf,
        'norm_speed': norm_speed,
        'bin_counts': bin_counts
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate flow speed profile |u|(r) around cargo.")
    parser.add_argument('--max_samples', type=int, default=25, help="Max frame samples per experiment (default 25)")
    parser.add_argument('--max_r', type=float, default=35.0, help="Max distance from cargo edge in um (default 35.0)")
    parser.add_argument('--bin_w', type=float, default=0.50, help="Radial bin width in um (default 0.50)")
    parser.add_argument('--force_recompute', action='store_true', help="Force recomputation ignoring cache")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "cargo_flow_profile_extended_cache.pkl"

    if cache_file.exists() and not args.force_recompute:
        print(f"Loading extended cached flow profiles from {cache_file}...")
        df_all = pd.read_pickle(cache_file)
        if 'dist_um' not in df_all.columns:
            df_all['dist_um'] = df_all.apply(lambda row: row['r_edge_um'] + row['radius_um'], axis=1)
    else:
        print(f"Computing extended flow profiles (0 to {args.max_r} um) across {len(TARGET_DIRS)} experiments...")
        t0 = time.time()
        results = []
        for exp_idx, exp_dir in enumerate(TARGET_DIRS):
            b_name = None
            for b in BEADS_INFO:
                if f"/{b['name']}/" in exp_dir:
                    b_name = b['name']
                    b_info = b
                    break
            if not b_name:
                continue
                
            print(f"[{exp_idx+1}/{len(TARGET_DIRS)}] Processing {b_name}: {Path(exp_dir).name}...")
            t_exp = time.time()
            res = process_single_experiment(exp_dir, b_info, max_r_edge_um=args.max_r, bin_width_um=args.bin_w, max_samples=args.max_samples)
            if res is not None:
                results.append(res)
                print(f"    Done in {time.time() - t_exp:.1f} s. u_inf = {res['u_inf']:.3f} um/s, edge |u|/u_inf = {res['norm_speed'][0]:.3f}")
                
        df_all = pd.DataFrame(results)
        df_all.to_pickle(cache_file)
        print(f"All experiments processed in {time.time() - t0:.1f} s. Cached to {cache_file}.")

    # xi の読み込み
    xi_map = load_xi_dict()

    setup_matplotlib()

    # =========================================================================
    # 1. 縁からの距離 r = dist - R_c ベースの集計
    # =========================================================================
    grouped_profiles_r = {}
    r_coords = None
    
    for b in BEADS_INFO:
        b_name = b['name']
        sub = df_all[df_all['bead_name'] == b_name]
        if len(sub) == 0:
            continue
            
        r_coords = sub.iloc[0]['r_edge_um']
        
        norm_matrix = np.vstack(sub['norm_speed'].values)
        speed_matrix = np.vstack(sub['speed_um_s'].values)
        
        mean_norm = np.mean(norm_matrix, axis=0)
        std_norm = np.std(norm_matrix, axis=0)
        sem_norm = std_norm / np.sqrt(len(sub)) if len(sub) > 1 else np.zeros_like(std_norm)
        
        mean_speed = np.mean(speed_matrix, axis=0)
        std_speed = np.std(speed_matrix, axis=0)
        sem_speed = std_speed / np.sqrt(len(sub)) if len(sub) > 1 else np.zeros_like(std_speed)
        
        # 中心からの距離 dist = r + R_c
        dist_coords = r_coords + b['radius_um']
        
        grouped_profiles_r[b_name] = {
            'info': b,
            'r_coords': r_coords,
            'dist_coords': dist_coords,
            'mean_norm': mean_norm,
            'sem_norm': sem_norm,
            'mean_speed': mean_speed,
            'sem_speed': sem_speed,
            'n_exps': len(sub),
            'raw_norm': norm_matrix
        }

    # ==========================================
    # Figure 1: 全ビーズ径重ね描きプロファイル (|u|(r) / u_inf vs r [um])
    # ==========================================
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name not in grouped_profiles_r:
            continue
        p = grouped_profiles_r[b_name]
        lbl = f"{b['label']} (N={p['n_exps']})"
        ax.plot(r_coords, p['mean_norm'], color=b['color'], marker=b['marker'], 
                markevery=max(1, len(r_coords)//14), markersize=6.0, label=lbl, lw=2.2)
        ax.fill_between(r_coords, p['mean_norm'] - p['sem_norm'], p['mean_norm'] + p['sem_norm'],
                        color=b['color'], alpha=0.15)
        
    ax.axhline(1.0, color='gray', linestyle=':', lw=1.5, label=r'Bulk flow $u_{\infty}$')
    ax.axvspan(20.0, args.max_r, color='lightgray', alpha=0.25, label=r'Far-field reference region ($r \geq 20\,\mu\mathrm{m}$)')
    ax.set_xlabel(r'Distance from cargo edge $r \equiv \mathrm{dist} - R_c$ [$\mu\mathrm{m}$]')
    ax.set_ylabel(r'Normalized flow speed $|\mathbf{u}|(r) \,/\, u_{\infty}$ [-]')
    ax.set_title(r'Microtubule Flow Speed around Cargo (Edge Distance $r$)')
    ax.set_xlim(0, args.max_r)
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True)
    save_fig(fig, "cargo_flow_speed_profile_all_beads")
    plt.close(fig)

    # ==========================================
    # Figure 2: 6パネル個別プロファイル (|u|(r) / u_inf vs r [um])
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, b in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = b['name']
        if b_name not in grouped_profiles_r:
            ax.set_visible(False)
            continue
        p = grouped_profiles_r[b_name]
        for exp_curve in p['raw_norm']:
            ax.plot(r_coords, exp_curve, color=b['color'], alpha=0.35, lw=1.2)
        ax.plot(r_coords, p['mean_norm'], color=b['color'], marker=b['marker'],
                markevery=max(1, len(r_coords)//14), markersize=6.0, lw=2.5, label='Mean ± SEM')
        ax.fill_between(r_coords, p['mean_norm'] - p['sem_norm'], p['mean_norm'] + p['sem_norm'],
                        color=b['color'], alpha=0.25)
        ax.axhline(1.0, color='gray', linestyle=':', lw=1.2, label=r'$u_{\infty}$')
        ax.axvspan(20.0, args.max_r, color='lightgray', alpha=0.2, label=r'Ref region ($r\geq 20\,\mu\mathrm{m}$)')
        v_edge = p['mean_norm'][0]
        reduction_pct = (1.0 - v_edge) * 100
        ax.text(0.05, 0.15, f"Edge speed: {v_edge:.2f} $u_\\infty$\nSlowdown: -{reduction_pct:.1f}%",
                transform=ax.transAxes, fontsize=10.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
        ax.set_title(f"{b['label']} (N={p['n_exps']} exps)", fontsize=13, fontweight='bold')
        ax.grid(True)
        if idx in [0, 3]:
            ax.set_ylabel(r'$|\mathbf{u}|(r) \,/\, u_{\infty}$ [-]')
        if idx >= 3:
            ax.set_xlabel(r'$r \equiv \mathrm{dist} - R_c$ [$\mu\mathrm{m}$]')
        ax.set_xlim(0, args.max_r)
        ax.set_ylim(0.40, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9.0)
    plt.suptitle(r'Radial Profile of Flow Speed $|\mathbf{u}|(r) \,/\, u_{\infty}$ across Cargo Sizes', fontsize=15, y=0.98)
    plt.tight_layout()
    save_fig(fig, "cargo_flow_speed_profile_6panels")
    plt.close(fig)

    # =========================================================================
    # 2. 横軸を相関長 \xi で割った無次元プロファイル (r / \xi = 0 〜 3.5)
    # =========================================================================
    max_scaled_r = 3.5
    scaled_r_grid = np.linspace(0.04, max_scaled_r, 60)
    scaled_grouped_profiles_r = {}
    
    for b in BEADS_INFO:
        b_name = b['name']
        sub = df_all[df_all['bead_name'] == b_name]
        if len(sub) == 0:
            continue
        interp_curves = []
        for _, row in sub.iterrows():
            exp_n = Path(row['exp_dir']).name
            xi_val = xi_map.get(exp_n, np.nan)
            if not np.isfinite(xi_val) or xi_val <= 0:
                continue
            r_um = row['r_edge_um']
            r_over_xi = r_um / xi_val
            norm_spd = row['norm_speed']
            f_interp = interp1d(r_over_xi, norm_spd, bounds_error=False, fill_value=(norm_spd[0], norm_spd[-1]))
            interp_curves.append(f_interp(scaled_r_grid))
            
        if len(interp_curves) > 0:
            matrix_scaled = np.vstack(interp_curves)
            mean_sc = np.mean(matrix_scaled, axis=0)
            std_sc = np.std(matrix_scaled, axis=0)
            sem_sc = std_sc / np.sqrt(len(interp_curves)) if len(interp_curves) > 1 else np.zeros_like(std_sc)
            scaled_grouped_profiles_r[b_name] = {
                'info': b,
                'mean': mean_sc,
                'sem': sem_sc,
                'n_exps': len(interp_curves),
                'raw': matrix_scaled
            }

    # Figure 4: r/xi 全ビーズ重ね描き図
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name not in scaled_grouped_profiles_r:
            continue
        p = scaled_grouped_profiles_r[b_name]
        lbl = f"{b['label']} (N={p['n_exps']})"
        ax.plot(scaled_r_grid, p['mean'], color=b['color'], marker=b['marker'],
                markevery=max(1, len(scaled_r_grid)//14), markersize=6.0, label=lbl, lw=2.2)
        ax.fill_between(scaled_r_grid, p['mean'] - p['sem'], p['mean'] + p['sem'],
                        color=b['color'], alpha=0.15)
    ax.axhline(1.0, color='gray', linestyle=':', lw=1.5, label=r'Bulk flow $u_{\infty}$')
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, lw=1.2, label=r'$r = \xi$')
    ax.axvspan(2.5, max_scaled_r, color='lightgray', alpha=0.25, label=r'Far-field plateau ($r \geq 2.5\xi$)')
    ax.set_xlabel(r'Scaled distance from cargo edge $r \,/\, \xi \equiv (\mathrm{dist} - R_c) \,/\, \xi$ [-]')
    ax.set_ylabel(r'Normalized flow speed $|\mathbf{u}|(r) \,/\, u_{\infty}$ [-]')
    ax.set_title(r'Convergence of Scaled Flow Speed Profile to Far Field $u_{\infty}$')
    ax.set_xlim(0, max_scaled_r)
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True)
    save_fig(fig, "cargo_flow_speed_profile_scaled_xi_all_beads")
    plt.close(fig)

    # Figure 5: r/xi 6パネル個別図
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, b in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = b['name']
        if b_name not in scaled_grouped_profiles_r:
            ax.set_visible(False)
            continue
        p = scaled_grouped_profiles_r[b_name]
        for exp_curve in p['raw']:
            ax.plot(scaled_r_grid, exp_curve, color=b['color'], alpha=0.35, lw=1.2)
        ax.plot(scaled_r_grid, p['mean'], color=b['color'], marker=b['marker'],
                markevery=max(1, len(scaled_r_grid)//14), markersize=6.0, lw=2.5, label='Mean ± SEM')
        ax.fill_between(scaled_r_grid, p['mean'] - p['sem'], p['mean'] + p['sem'],
                        color=b['color'], alpha=0.25)
        ax.axhline(1.0, color='gray', linestyle=':', lw=1.2, label=r'$u_{\infty}$')
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, lw=1.2, label=r'$r = \xi$')
        ax.axvspan(2.5, max_scaled_r, color='lightgray', alpha=0.2, label=r'Plateau ($r \geq 2.5\xi$)')
        v_edge = p['mean'][0]
        v_3xi = p['mean'][-1]
        reduction_pct = (1.0 - v_edge) * 100
        ax.text(0.05, 0.15, f"Edge speed: {v_edge:.2f} $u_\\infty$\nSlowdown: -{reduction_pct:.1f}%\n$r=3.5\\xi$: {v_3xi:.3f} $u_\\infty$",
                transform=ax.transAxes, fontsize=10.0,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
        ax.set_title(f"{b['label']} (N={p['n_exps']} exps)", fontsize=13, fontweight='bold')
        ax.grid(True)
        if idx in [0, 3]:
            ax.set_ylabel(r'$|\mathbf{u}|(r) \,/\, u_{\infty}$ [-]')
        if idx >= 3:
            ax.set_xlabel(r'Scaled distance $r \,/\, \xi$ [-]')
        ax.set_xlim(0, max_scaled_r)
        ax.set_ylim(0.40, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9.0)
    plt.suptitle(r'Convergence of Radial Flow Speed Profile $|\mathbf{u}|(r/\xi) \,/\, u_{\infty}$ to $1.0$', fontsize=15, y=0.98)
    plt.tight_layout()
    save_fig(fig, "cargo_flow_speed_profile_scaled_xi_6panels")
    plt.close(fig)

    # =========================================================================
    # 3. 【新機能】横軸を「中心からの距離 dist」にしたプロファイル (-R_c を無くしたもの)
    #    各ビーズで dist < R_c はマスクされているため、曲線は dist = R_c から始まります。
    # =========================================================================
    max_dist_um = 35.0 + max([b['radius_um'] for b in BEADS_INFO]) # ~45 um

    # Figure 6: 全ビーズ重ね描き図 (横軸: dist [um])
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name not in grouped_profiles_r:
            continue
        p = grouped_profiles_r[b_name]
        d_coords = p['dist_coords']
        lbl = f"{b['label']} ($R_c={b['radius_um']:.2f}\\,\\mu\\mathrm{{m}}$)"
        ax.plot(d_coords, p['mean_norm'], color=b['color'], marker=b['marker'],
                markevery=max(1, len(d_coords)//14), markersize=6.0, label=lbl, lw=2.2)
        ax.fill_between(d_coords, p['mean_norm'] - p['sem_norm'], p['mean_norm'] + p['sem_norm'],
                        color=b['color'], alpha=0.15)
        # 貨物半径 R_c の位置に縦マーカー
        ax.scatter([b['radius_um']], [p['mean_norm'][0]], color=b['color'], s=50, zorder=5)
        
    ax.axhline(1.0, color='gray', linestyle=':', lw=1.5, label=r'Bulk flow $u_{\infty}$')
    ax.set_xlabel(r'Distance from cargo center $\mathrm{dist}$ [$\mu\mathrm{m}$]')
    ax.set_ylabel(r'Normalized flow speed $|\mathbf{u}|(\mathrm{dist}) \,/\, u_{\infty}$ [-]')
    ax.set_title(r'Microtubule Flow Speed from Cargo Center $\mathrm{dist}$')
    ax.set_xlim(0, 40.0)
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True)
    save_fig(fig, "cargo_flow_speed_profile_dist_center_all_beads")
    plt.close(fig)

    # Figure 7: 6パネル個別図 (横軸: dist [um])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, b in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = b['name']
        if b_name not in grouped_profiles_r:
            ax.set_visible(False)
            continue
        p = grouped_profiles_r[b_name]
        d_coords = p['dist_coords']
        Rc = b['radius_um']
        
        # 貨物直下のマスク領域 (0 <= dist <= Rc) を薄いハッチングで可視化
        ax.axvspan(0, Rc, color='salmon', alpha=0.25, hatch='//', label=r'Masked cargo ($r \leq 0$)')
        ax.axvline(Rc, color=b['color'], linestyle='--', lw=1.5, label=f'$R_c = {Rc:.2f}\\,\\mu\\mathrm{{m}}$')
        
        for exp_curve in p['raw_norm']:
            ax.plot(d_coords, exp_curve, color=b['color'], alpha=0.35, lw=1.2)
        ax.plot(d_coords, p['mean_norm'], color=b['color'], marker=b['marker'],
                markevery=max(1, len(d_coords)//14), markersize=6.0, lw=2.5, label='Mean ± SEM')
        ax.fill_between(d_coords, p['mean_norm'] - p['sem_norm'], p['mean_norm'] + p['sem_norm'],
                        color=b['color'], alpha=0.25)
        ax.axhline(1.0, color='gray', linestyle=':', lw=1.2, label=r'$u_{\infty}$')
        
        v_edge = p['mean_norm'][0]
        reduction_pct = (1.0 - v_edge) * 100
        ax.text(0.40, 0.15, f"Edge ($R_c$): {v_edge:.2f} $u_\\infty$\nSlowdown: -{reduction_pct:.1f}%",
                transform=ax.transAxes, fontsize=10.0,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
        
        ax.set_title(f"{b['label']} ($R_c={Rc:.2f}\\,\\mu\\mathrm{{m}}$)", fontsize=13, fontweight='bold')
        ax.grid(True)
        if idx in [0, 3]:
            ax.set_ylabel(r'$|\mathbf{u}|(\mathrm{dist}) \,/\, u_{\infty}$ [-]')
        if idx >= 3:
            ax.set_xlabel(r'Distance from cargo center $\mathrm{dist}$ [$\mu\mathrm{m}$]')
        ax.set_xlim(0, 40.0)
        ax.set_ylim(0.40, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8.5)
            
    plt.suptitle(r'Flow Speed Profile from Cargo Center $\mathrm{dist}$ (Masked region $\mathrm{dist} \leq R_c$ shaded)', fontsize=15, y=0.98)
    plt.tight_layout()
    save_fig(fig, "cargo_flow_speed_profile_dist_center_6panels")
    plt.close(fig)

    # =========================================================================
    # 4. 【新機能】横軸を「中心からの無次元距離 dist / \xi」にしたプロファイル
    # =========================================================================
    max_scaled_dist = 5.0
    scaled_dist_grid = np.linspace(0.04, max_scaled_dist, 80)
    scaled_grouped_profiles_dist = {}
    
    for b in BEADS_INFO:
        b_name = b['name']
        sub = df_all[df_all['bead_name'] == b_name]
        if len(sub) == 0:
            continue
        interp_curves = []
        for _, row in sub.iterrows():
            exp_n = Path(row['exp_dir']).name
            xi_val = xi_map.get(exp_n, np.nan)
            if not np.isfinite(xi_val) or xi_val <= 0:
                continue
            dist_um = row['dist_um']
            dist_over_xi = dist_um / xi_val
            norm_spd = row['norm_speed']
            # dist/xi < Rc/xi の領域はマスクされているため、補間時に NaN を与える
            f_interp = interp1d(dist_over_xi, norm_spd, bounds_error=False, fill_value=np.nan)
            interp_curves.append(f_interp(scaled_dist_grid))
            
        if len(interp_curves) > 0:
            matrix_scaled = np.vstack(interp_curves)
            mean_sc = np.nanmean(matrix_scaled, axis=0)
            std_sc = np.nanstd(matrix_scaled, axis=0)
            valid_counts = np.sum(np.isfinite(matrix_scaled), axis=0)
            sem_sc = np.where(valid_counts > 1, std_sc / np.sqrt(valid_counts), np.zeros_like(std_sc))
            
            scaled_grouped_profiles_dist[b_name] = {
                'info': b,
                'mean': mean_sc,
                'sem': sem_sc,
                'n_exps': len(interp_curves),
                'raw': matrix_scaled
            }

    # Figure 8: dist/xi 全ビーズ重ね描き図
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name not in scaled_grouped_profiles_dist:
            continue
        p = scaled_grouped_profiles_dist[b_name]
        lbl = f"{b['label']} (N={p['n_exps']})"
        valid = np.isfinite(p['mean'])
        ax.plot(scaled_dist_grid[valid], p['mean'][valid], color=b['color'], marker=b['marker'],
                markevery=max(1, np.sum(valid)//14), markersize=6.0, label=lbl, lw=2.2)
        ax.fill_between(scaled_dist_grid[valid], (p['mean'] - p['sem'])[valid], (p['mean'] + p['sem'])[valid],
                        color=b['color'], alpha=0.15)
        
    ax.axhline(1.0, color='gray', linestyle=':', lw=1.5, label=r'Bulk flow $u_{\infty}$')
    ax.set_xlabel(r'Scaled distance from center $\mathrm{dist} \,/\, \xi$ [-]')
    ax.set_ylabel(r'Normalized flow speed $|\mathbf{u}|(\mathrm{dist}) \,/\, u_{\infty}$ [-]')
    ax.set_title(r'Scaled Flow Speed Profile from Cargo Center $\mathrm{dist} \,/\, \xi$')
    ax.set_xlim(0, max_scaled_dist)
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True)
    save_fig(fig, "cargo_flow_speed_profile_scaled_xi_dist_center_all_beads")
    plt.close(fig)

    # Figure 9: dist/xi 6パネル個別図
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, b in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = b['name']
        if b_name not in scaled_grouped_profiles_dist:
            ax.set_visible(False)
            continue
        p = scaled_grouped_profiles_dist[b_name]
        valid = np.isfinite(p['mean'])
        
        # 実効的な平均 Rc/xi
        sub = df_all[df_all['bead_name'] == b_name]
        mean_xi = np.mean([xi_map.get(Path(row['exp_dir']).name, np.nan) for _, row in sub.iterrows()])
        mean_rc_over_xi = b['radius_um'] / mean_xi if mean_xi > 0 else np.nan
        
        if np.isfinite(mean_rc_over_xi):
            ax.axvspan(0, mean_rc_over_xi, color='salmon', alpha=0.25, hatch='//', label=r'Masked ($R_c/\xi$)')
            ax.axvline(mean_rc_over_xi, color=b['color'], linestyle='--', lw=1.5, label=f'$R_c/\\xi \\approx {mean_rc_over_xi:.2f}$')
            
        for exp_curve in p['raw']:
            v_exp = np.isfinite(exp_curve)
            ax.plot(scaled_dist_grid[v_exp], exp_curve[v_exp], color=b['color'], alpha=0.35, lw=1.2)
            
        ax.plot(scaled_dist_grid[valid], p['mean'][valid], color=b['color'], marker=b['marker'],
                markevery=max(1, np.sum(valid)//14), markersize=6.0, lw=2.5, label='Mean ± SEM')
        ax.fill_between(scaled_dist_grid[valid], (p['mean'] - p['sem'])[valid], (p['mean'] + p['sem'])[valid],
                        color=b['color'], alpha=0.25)
        ax.axhline(1.0, color='gray', linestyle=':', lw=1.2, label=r'$u_{\infty}$')
        
        ax.set_title(f"{b['label']} (N={p['n_exps']} exps)", fontsize=13, fontweight='bold')
        ax.grid(True)
        if idx in [0, 3]:
            ax.set_ylabel(r'$|\mathbf{u}|(\mathrm{dist}) \,/\, u_{\infty}$ [-]')
        if idx >= 3:
            ax.set_xlabel(r'Scaled distance $\mathrm{dist} \,/\, \xi$ [-]')
        ax.set_xlim(0, max_scaled_dist)
        ax.set_ylim(0.40, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8.5)
            
    plt.suptitle(r'Scaled Flow Speed Profile from Cargo Center $\mathrm{dist}/\xi$ across Cargo Sizes', fontsize=15, y=0.98)
    plt.tight_layout()
    save_fig(fig, "cargo_flow_speed_profile_scaled_xi_dist_center_6panels")
    plt.close(fig)

    # ==========================================
    # Figure 3: 貨物直近 (r -> 0) の速度低下とビーズ径の関係
    # ==========================================
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    diameters = []
    edge_ratios = []
    sem_ratios = []
    
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name not in grouped_profiles_r:
            continue
        p = grouped_profiles_r[b_name]
        diameters.append(b['diameter_um'])
        edge_ratios.append(p['mean_norm'][0])
        sem_ratios.append(p['sem_norm'][0])
        
    ax.errorbar(diameters, edge_ratios, yerr=sem_ratios, fmt='o-', color='#332288',
                ecolor='#332288', elinewidth=1.8, capsize=4, markersize=8, lw=2.0)
    ax.axhline(1.0, color='gray', linestyle=':', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel(r'Cargo diameter $2 R_c$ [$\mu\mathrm{m}$]')
    ax.set_ylabel(r'Flow speed at cargo edge $|\mathbf{u}|(r\to 0) \,/\, u_{\infty}$ [-]')
    ax.set_title(r'Direct Evidence: Slowing of Contacting Microtubules')
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, which='both')
    
    for d, er in zip(diameters, edge_ratios):
        ax.annotate(f"{er:.2f}", (d, er), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9.5, fontweight='bold')
        
    save_fig(fig, "cargo_edge_speed_ratio_vs_diameter")
    plt.close(fig)

    # Summary CSV の出力
    summary_rows = []
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name in grouped_profiles_r:
            p = grouped_profiles_r[b_name]
            p_sc = scaled_grouped_profiles_r.get(b_name, None)
            v_35xi = p_sc['mean'][-1] if p_sc is not None else np.nan
            summary_rows.append({
                'bead_name': b_name,
                'diameter_um': b['diameter_um'],
                'radius_um': b['radius_um'],
                'n_exps': p['n_exps'],
                'edge_norm_speed': p['mean_norm'][0],
                'edge_norm_sem': p['sem_norm'][0],
                'edge_slowdown_pct': (1.0 - p['mean_norm'][0]) * 100,
                'norm_speed_at_3.5xi': v_35xi,
            })
    df_summary = pd.DataFrame(summary_rows)
    for out_dir in OUT_DIRS:
        df_summary.to_csv(out_dir / "cargo_flow_profile_summary.csv", index=False)
    print("\n=== Analysis Summary ===")
    print(df_summary.to_string(index=False))
    print("\nAll figures (both r = dist - Rc and dist from center) generated successfully!")

if __name__ == "__main__":
    main()
