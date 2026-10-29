#!/usr/bin/env python3
"""
verify_cos_decomposition.py

各点方向一致度の平均 <cos theta_i> と、
重心流速との方向一致度 cos Theta_bulk、局所ポーラーオーダー Phi(R) の関係:
    <cos theta_i> ≈ Phi(R) * cos Theta_bulk
が実験データ上で成り立つかを包括的に検証するスクリプト。
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from scipy.special import i0, i1
from scipy.optimize import curve_fit
from scipy.stats import skew

# ワークスペースおよびNASのパス設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR_CANDIDATES = [
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    CURRENT_DIR,
]
ROOT_DIR = next((p for p in ROOT_DIR_CANDIDATES if p.exists()), CURRENT_DIR)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "cos_decomposition",
    ROOT_DIR / "figure" / "cos_decomposition"
]

# ビーズ定義 (Tol's Muted パレット)
BEADS_INFO = [
    {'name': 'beads06um', 'label': r'0.63 $\mu\mathrm{m}$', 'diameter_um': 0.63, 'radius_um': 0.315,
     'color': '#882255', 'marker': '^'},
    {'name': 'beads1um',  'label': r'1.18 $\mu\mathrm{m}$', 'diameter_um': 1.18, 'radius_um': 0.590,
     'color': '#CC6677', 'marker': 'o'},
    {'name': 'beads3um',  'label': r'3.37 $\mu\mathrm{m}$', 'diameter_um': 3.37, 'radius_um': 1.685,
     'color': '#DDCC77', 'marker': 'd'},
    {'name': 'beads5um',  'label': r'5.00 $\mu\mathrm{m}$', 'diameter_um': 5.00, 'radius_um': 2.500,
     'color': '#999933', 'marker': 'p'},
    {'name': 'beads7um',  'label': r'7.24 $\mu\mathrm{m}$', 'diameter_um': 7.24, 'radius_um': 3.620,
     'color': '#117733', 'marker': 'h'},
    {'name': 'beads20um', 'label': r'20.0 $\mu\mathrm{m}$', 'diameter_um': 20.0, 'radius_um': 10.000,
     'color': '#44AA99', 'marker': 's'},
]

SCALE_UM_PER_PX = 0.11


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
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'lines.linewidth': 1.8,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def calc_disk_polar_order(R_um: float, xi_um: float) -> float:
    """
    空間配向自己相関長 xi を持つ微小管流動場における、
    半径 R の円形窓（ディスク領域）での真の局所ポーラーオーダー Phi_true(R) を
    統計力学的自己畳み込み積分により厳密に計算する。
    """
    if not np.isfinite(xi_um) or xi_um <= 0:
        xi_um = 6.0
    if R_um <= 1e-4:
        return 1.0

    r = np.linspace(1e-5, 2.0 * R_um, 200)
    s = r / (2.0 * R_um)
    geom = (2.0 / np.pi) * (np.arccos(s) - s * np.sqrt(np.maximum(0.0, 1.0 - s**2)))
    c_r = np.exp(-r / xi_um)
    integrand = (2.0 * np.pi * r / (np.pi * R_um**2)) * geom * c_r
    val = np.trapezoid(integrand, r)
    return float(np.sqrt(np.clip(val, 0.0, 1.0)))


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    def _has_interaction(p: Path) -> bool:
        return (p / "beads_flow_interaction.csv").exists() or (p / "beads_flow_interaction_gaussian.csv").exists()

    for p in sorted(base.glob("*/*")):
        if p.is_dir() and _has_interaction(p):
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and _has_interaction(p):
                exp_dirs.append(p)
    return exp_dirs


def load_active_particles(workspace_dir: Path) -> set:
    candidates = [
        workspace_dir / "figure" / "chi_vs_radius" / "cos_vs_radius_particle_summary.csv",
        ROOT_DIR / "figure" / "chi_vs_radius" / "cos_vs_radius_particle_summary.csv",
    ]
    active_set = set()
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                for _, row in df.iterrows():
                    exp = str(row['exp_dir']).strip()
                    pid = int(row['particle'])
                    active_set.add((exp, pid))
                print(f"Loaded {len(active_set)} active particles from {p.name}")
                return active_set
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
    return active_set


def load_experiment_correlation_lengths(root_dir: Path, workspace_dir: Path) -> Dict[str, float]:
    candidates = [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
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
                break
            except Exception:
                pass
    return xi_map


def save_figure(fig, base_name: str, out_dirs: List[Path]):
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


def load_all_data(root_dir: Path, active_set: set):
    """
    全ビーズサイズ・全実験のインタラクションデータと軌跡データを統合抽出する。
    beads_flow_interaction.csv を最優先で採用し、存在しない場合は beads_flow_interaction_gaussian.csv を採用する。
    """
    instant_list = []
    window_list = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        target_rad_px = r_um / SCALE_UM_PER_PX

        exp_dirs = find_experiment_dirs(root_dir, bname)
        print(f"Processing {bname} ({len(exp_dirs)} experiments)...")

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            csv_std = exp_dir / "beads_flow_interaction.csv"
            csv_gau = exp_dir / "beads_flow_interaction_gaussian.csv"
            tracks_p = exp_dir / "beads_tracks.csv"

            if csv_std.exists():
                csv_p = csv_std
                rad_col = "radius"
                source_type = "tophat"
            elif csv_gau.exists():
                csv_p = csv_gau
                rad_col = "sigma"
                source_type = "gaussian"
            else:
                continue

            if not tracks_p.exists():
                continue

            try:
                df_flow = pd.read_csv(csv_p)
                tdf = pd.read_csv(tracks_p)
            except Exception as e:
                print(f"  Error loading {exp_name}: {e}")
                continue

            # 貨物速度マップ:
            # 1. beads_flow_interaction.csv 自身から真の貨物速度ベクトル (bdx, bdy) を復元
            #    (dot_product = bdx * u + bdy * v。これにより tracks.csv の粒子ID再割り当て齟齬を完全に解消)
            cargo_velocity_map = {}
            for (frm, pid), grp_fp in df_flow.groupby(['frame', 'particle']):
                grp_v = grp_fp[(grp_fp['mean_flow_u'] != 0) | (grp_fp['mean_flow_v'] != 0)]
                if len(grp_v) >= 2:
                    u_vals = grp_v['mean_flow_u'].values
                    v_vals = grp_v['mean_flow_v'].values
                    d_vals = grp_v['dot_product'].values
                    A = np.column_stack([u_vals, v_vals])
                    res, _, _, _ = np.linalg.lstsq(A, d_vals, rcond=None)
                    bdx, bdy = float(res[0]), float(res[1])
                    b_spd = float(np.hypot(bdx, bdy))
                    if b_spd > 0:
                        cargo_velocity_map[(int(pid), int(frm))] = (bdx, bdy, b_spd)

            # 2. フォールバック用の beads_tracks.csv 速度マップ
            cargo_speed_map = {}
            for pid, grp in tdf.groupby('particle'):
                grp_s = grp.sort_values('frame')
                fs = grp_s['frame'].values
                xs = grp_s['x'].values
                ys = grp_s['y'].values
                for i in range(len(fs) - 1):
                    if fs[i+1] == fs[i] + 1:
                        dx = xs[i+1] - xs[i]
                        dy = ys[i+1] - ys[i]
                        spd = np.hypot(dx, dy)
                        cargo_speed_map[(int(pid), int(fs[i]))] = (dx, dy, spd)

            def get_cargo_vel(pid: int, frm: int):
                k = (pid, frm)
                if k in cargo_velocity_map:
                    return cargo_velocity_map[k]
                return cargo_speed_map.get(k, None)

            # local_polar_w.zarr からの局所ポーラーオーダーのロード
            zarr_p = exp_dir / "local_polar_w.zarr"
            if not zarr_p.exists():
                zarr_p = exp_dir / "local_polar_w_gaussian.zarr"

            polar_cache = {}
            if zarr_p.exists():
                try:
                    ds_polar = xr.open_zarr(zarr_p, consolidated=False)
                    if 'polar_order' in ds_polar:
                        da_polar = ds_polar['polar_order']
                        w_dim = next((d for d in da_polar.dims if 'window' in d or 'sigma' in d), None)
                        if w_dim is not None:
                            z_windows = da_polar[w_dim].values
                            z_frames = da_polar['frame'].values
                            z_particles = da_polar['particle'].values
                            for p_val in z_particles:
                                p_da = da_polar.sel(particle=p_val)
                                w_axis = list(p_da.dims).index(w_dim)
                                arr = p_da.values
                                if w_axis == 1:
                                    arr = arr.T  # shape: (n_windows, n_frames)
                                f_map = {int(f): i for i, f in enumerate(z_frames)}
                                w_map = {float(w): i for i, w in enumerate(z_windows)}
                                polar_cache[int(p_val)] = {
                                    'arr': arr,
                                    'f_map': f_map,
                                    'w_map': w_map,
                                    'windows': z_windows
                                }
                except Exception as e:
                    print(f"  Warning: failed to load zarr for {exp_name}: {e}")

            df_flow['rad_px'] = df_flow[rad_col].astype(float)
            # 有効な窓サイズ（少なくとも1つのフレームで微小管流速が存在する）
            # beads_flow_interaction.csv ではビーズ半径以下の窓で流速が 0 になっているためくり抜きを除外
            valid_mask = (df_flow['mean_flow_u'] != 0) | (df_flow['mean_flow_v'] != 0) | (df_flow['cos_sim'] != 0)
            valid_rads = np.sort(df_flow.loc[valid_mask, 'rad_px'].unique())
            if len(valid_rads) == 0:
                continue

            # 粒子直下窓 (最寄りの有効な rad_px)
            best_idx = int(np.argmin(np.abs(valid_rads - target_rad_px)))
            best_rad = float(valid_rads[best_idx])

            # 1. 粒子直下窓での瞬時値レコード
            sub_best = df_flow[(df_flow['rad_px'] == best_rad) & valid_mask]
            for _, r in sub_best.iterrows():
                pid = int(r['particle'])
                frm = int(r['frame'])
                if active_set and (exp_name, pid) not in active_set:
                    continue

                c_vel = get_cargo_vel(pid, frm)
                if c_vel is None or c_vel[2] <= 0:
                    continue
                cdx, cdy, cspd = c_vel

                u = float(r['mean_flow_u'])
                v = float(r['mean_flow_v'])
                v_flow_norm = np.hypot(u, v)
                if v_flow_norm <= 0:
                    continue

                dot_p = float(r['dot_product'])
                cos_i = float(r['cos_sim']) # 各点方向一致度の平均 <cos theta_i>

                # 重心流速方向一致度 cos Theta_bulk
                cos_bulk = (cdx * u + cdy * v) / (cspd * v_flow_norm)

                # local_polar_w.zarr からの瞬時ポーラーオーダー取得
                phi_inst = np.nan
                if pid in polar_cache:
                    pc = polar_cache[pid]
                    f_idx = pc['f_map'].get(frm)
                    if f_idx is not None:
                        # 1. best_rad に一致する窓
                        if best_rad in pc['w_map']:
                            w_idx = pc['w_map'][best_rad]
                            val = pc['arr'][w_idx, f_idx]
                            if np.isfinite(val) and 0.0 <= val <= 1.5:
                                phi_inst = float(np.clip(val, 0.0, 1.0))
                        # 2. もし nan や inf の場合、best_rad 以上の最初の有効窓を探す
                        if np.isnan(phi_inst):
                            for w_cand in sorted(pc['windows']):
                                if w_cand >= best_rad:
                                    w_idx = pc['w_map'][float(w_cand)]
                                    val = pc['arr'][w_idx, f_idx]
                                    if np.isfinite(val) and 0.0 <= val <= 1.5:
                                        phi_inst = float(np.clip(val, 0.0, 1.0))
                                        break

                instant_list.append({
                    'bead_name': bname,
                    'diameter_um': d_um,
                    'radius_um': r_um,
                    'exp_dir': exp_name,
                    'particle': pid,
                    'frame': frm,
                    'radius_px': best_rad,
                    'window_radius_um': best_rad * SCALE_UM_PER_PX,
                    'cos_i': cos_i,
                    'cos_bulk': cos_bulk,
                    'Phi_mag': phi_inst,
                    'v_flow_norm': v_flow_norm,
                    'v_cargo_norm': cspd,
                    'dot_product': dot_p,
                    'source_type': source_type,
                })

            # 2. 全窓サイズでの平均値レコード (窓サイズ依存性)
            for rad in valid_rads:
                sub_rad = df_flow[(df_flow['rad_px'] == rad) & valid_mask]
                for pid, grp in sub_rad.groupby('particle'):
                    pid = int(pid)
                    if active_set and (exp_name, pid) not in active_set:
                        continue

                    # 粒子内の有効フレームについて計算
                    ci_vals = []
                    cb_vals = []
                    vf_vals = []
                    for _, r in grp.iterrows():
                        frm = int(r['frame'])
                        c_vel = get_cargo_vel(pid, frm)
                        if c_vel is None or c_vel[2] <= 0:
                            continue
                        cdx, cdy, cspd = c_vel
                        u = float(r['mean_flow_u'])
                        v = float(r['mean_flow_v'])
                        vn = np.hypot(u, v)
                        if vn <= 0:
                            continue
                        cb = (cdx * u + cdy * v) / (cspd * vn)
                        ci = float(r['cos_sim'])
                        ci_vals.append(ci)
                        cb_vals.append(cb)
                        vf_vals.append(vn)

                    if len(ci_vals) >= 10:
                        # local_polar_w.zarr からの窓ごとのフレーム平均ポーラーオーダー取得
                        phi_window = np.nan
                        if pid in polar_cache:
                            pc = polar_cache[pid]
                            cand_ws = pc['windows']
                            w_closest = cand_ws[int(np.argmin(np.abs(cand_ws - rad)))]
                            w_idx = pc['w_map'][float(w_closest)]
                            row_vals = pc['arr'][w_idx, :]
                            valid_vals = row_vals[np.isfinite(row_vals) & (row_vals >= 0.0) & (row_vals <= 1.5)]
                            if len(valid_vals) > 0:
                                phi_window = float(np.clip(np.mean(valid_vals), 0.0, 1.0))

                        window_list.append({
                            'bead_name': bname,
                            'diameter_um': d_um,
                            'radius_um': r_um,
                            'exp_dir': exp_name,
                            'particle': pid,
                            'radius_px': float(rad),
                            'window_radius_um': float(rad) * SCALE_UM_PER_PX,
                            'cos_i_mean': float(np.mean(ci_vals)),
                            'cos_bulk_mean': float(np.mean(cb_vals)),
                            'Phi_mean': phi_window,
                            'v_flow_norm_mean': float(np.mean(vf_vals)),
                            'n_frames': len(ci_vals),
                            'source_type': source_type,
                        })

    df_instant = pd.DataFrame(instant_list)
    df_window = pd.DataFrame(window_list)
    return df_instant, df_window


def analyze_and_plot(df_instant: pd.DataFrame, df_window: pd.DataFrame, xi_dict: Dict[str, float]):
    """
    数理関係 <cos theta_i> ≈ Phi(R) * cos Theta_bulk の検証と可視化。
    真の局所ポーラーオーダー Phi_true(R) は実験計測値 local_polar_w.zarr を直接採用。
    """
    setup_matplotlib()

    # 全体の代表相関長（フォールバック用）
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0

    # 瞬時値データ: 実験計測値 local_polar_w.zarr の Phi_mag を最優先で使用
    df_instant['xi_um'] = df_instant['exp_dir'].map(xi_dict).fillna(default_xi)
    nan_mask_inst = df_instant['Phi_mag'].isna()
    if nan_mask_inst.any():
        df_instant.loc[nan_mask_inst, 'Phi_mag'] = [
            calc_disk_polar_order(rw, xi)
            for rw, xi in zip(df_instant.loc[nan_mask_inst, 'window_radius_um'], df_instant.loc[nan_mask_inst, 'xi_um'])
        ]
    df_instant['pred_cos_i'] = df_instant['Phi_mag'] * df_instant['cos_bulk']

    # 窓データ: 実験計測値 local_polar_w.zarr の Phi_mean を最優先で使用
    df_window['xi_um'] = df_window['exp_dir'].map(xi_dict).fillna(default_xi)
    if 'Phi_mag' not in df_window.columns:
        df_window['Phi_mag'] = df_window['Phi_mean']
    nan_mask_win = df_window['Phi_mag'].isna()
    if nan_mask_win.any():
        df_window.loc[nan_mask_win, 'Phi_mag'] = [
            calc_disk_polar_order(rw, xi)
            for rw, xi in zip(df_window.loc[nan_mask_win, 'window_radius_um'], df_window.loc[nan_mask_win, 'xi_um'])
        ]
    df_window['pred_cos_i'] = df_window['Phi_mag'] * df_window['cos_bulk_mean']

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 図1: 瞬時値散布図: <cos theta_i> vs cos Theta_bulk, Phi(R_c), Phi*cos_bulk (ビーズ直下窓)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.2))

    # Panel A: cos_i vs cos_bulk
    ax = axes[0]
    ax.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax.axvline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax.plot([-1, 1], [-1, 1], color='#333333', linestyle='--', linewidth=1.5, label='1:1 line ($y = x$)')

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname]
        if sub.empty:
            continue
        corr = sub['cos_i'].corr(sub['cos_bulk'])
        ax.scatter(sub['cos_bulk'], sub['cos_i'], color=b_item['color'], marker=b_item['marker'],
                   alpha=0.25, s=18, label=f"{b_item['label']} ($r={corr:.3f}$)")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Bulk Directional Coupling $\cos \Theta_{\mathrm{bulk}} \equiv \frac{\mathbf{v}_{\mathrm{cargo}} \cdot \bar{\mathbf{v}}_{\mathrm{MT}}}{|\mathbf{v}_{\mathrm{cargo}}| |\bar{\mathbf{v}}_{\mathrm{MT}}|}$")
    ax.set_ylabel(r"Pixel-Averaged Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$")
    ax.set_title(r"$\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ vs $\cos \Theta_{\mathrm{bulk}}$ ($R = R_c$)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax.grid(True)

    # Panel B: cos_i vs Phi(R_c) (直接相関散布図)
    ax1 = axes[1]
    ax1.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax1.axvline(0, color='#888888', linestyle=':', linewidth=1.2)

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname].dropna(subset=['Phi_mag', 'cos_i'])
        if sub.empty:
            continue
        corr = sub['cos_i'].corr(sub['Phi_mag'])
        ax1.scatter(sub['Phi_mag'], sub['cos_i'], color=b_item['color'], marker=b_item['marker'],
                    alpha=0.25, s=18, label=f"{b_item['label']} ($r={corr:.3f}$)")

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xlabel(r"Local Polar Order $\Phi(R_c)$ (from local_polar_w.zarr)")
    ax1.set_ylabel(r"Pixel-Averaged Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$")
    ax1.set_title(r"Direct Correlation: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ vs $\Phi(R_c)$", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax1.grid(True)

    # Panel C: cos_i vs (Phi * cos_bulk)
    ax2 = axes[2]
    ax2.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax2.axvline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax2.plot([-1, 1], [-1, 1], color='#333333', linestyle='--', linewidth=1.5, label='1:1 line ($y = x$)')

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname].dropna(subset=['pred_cos_i', 'cos_i'])
        if sub.empty:
            continue
        corr = sub['cos_i'].corr(sub['pred_cos_i'])
        ax2.scatter(sub['pred_cos_i'], sub['cos_i'], color=b_item['color'], marker=b_item['marker'],
                    alpha=0.25, s=18, label=f"{b_item['label']} ($r={corr:.3f}$)")

    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel(r"Product: $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$")
    ax2.set_ylabel(r"Pixel-Averaged Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$")
    ax2.set_title(r"Verification: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ vs $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax2.grid(True)

    fig.tight_layout()
    save_figure(fig, "cos_instantaneous_decomposition_scatter_3panels", OUT_DIRS)
    save_figure(fig, "cos_instantaneous_decomposition_scatter", OUT_DIRS)
    plt.close(fig)

    # 単独の「<cos theta_i> vs Phi(R_c)」直接相関散布図 (瞬時値 & 粒子平均)
    fig_single, axes_s = plt.subplots(1, 2, figsize=(15.0, 6.5))

    # Panel A: 瞬時値散布図
    ax_sa = axes_s[0]
    ax_sa.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax_sa.axvline(0, color='#888888', linestyle=':', linewidth=1.2)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname].dropna(subset=['Phi_mag', 'cos_i'])
        if sub.empty:
            continue
        corr = sub['cos_i'].corr(sub['Phi_mag'])
        ax_sa.scatter(sub['Phi_mag'], sub['cos_i'], color=b_item['color'], marker=b_item['marker'],
                      alpha=0.25, s=20, label=f"{b_item['label']} ($r={corr:.3f}$)")
    ax_sa.set_xlim(-0.05, 1.05)
    ax_sa.set_ylim(-1.05, 1.05)
    ax_sa.set_xlabel(r"Local Polar Order $\Phi(R_c)$")
    ax_sa.set_ylabel(r"Pixel-Averaged Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$")
    ax_sa.set_title(r"Instantaneous: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ vs $\Phi(R_c)$", fontsize=14, fontweight='bold')
    ax_sa.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_sa.grid(True)

    # Panel B: 粒子平均散布図
    ax_sb = axes_s[1]
    ax_sb.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax_sb.axvline(0, color='#888888', linestyle=':', linewidth=1.2)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname].dropna(subset=['Phi_mag', 'cos_i'])
        if sub.empty:
            continue
        sub = sub.copy()
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)
        p_agg = sub.groupby('unique_pid').agg({'cos_i': 'mean', 'Phi_mag': 'mean'})
        corr = p_agg['cos_i'].corr(p_agg['Phi_mag']) if len(p_agg) > 1 else np.nan
        corr_str = f"$r={corr:.3f}$" if np.isfinite(corr) else "N/A"
        ax_sb.scatter(p_agg['Phi_mag'], p_agg['cos_i'], color=b_item['color'], marker=b_item['marker'],
                      alpha=0.85, s=55, edgecolor='black', linewidth=0.6,
                      label=f"{b_item['label']} ({corr_str}, $N={len(p_agg)}$)")
    ax_sb.set_xlim(-0.05, 1.05)
    ax_sb.set_ylim(-0.4, 1.05)
    ax_sb.set_xlabel(r"Track-Averaged Local Polar Order $\langle \Phi(R_c) \rangle_{\mathrm{track}}$")
    ax_sb.set_ylabel(r"Track-Averaged Coupling $\langle \mathbf{\hat{v}} \cdot \mathbf{\hat{u}} \rangle_{\mathrm{track}}$")
    ax_sb.set_title(r"Particle-Averaged: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ vs $\Phi(R_c)$", fontsize=14, fontweight='bold')
    ax_sb.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_sb.grid(True)

    fig_single.tight_layout()
    save_figure(fig_single, "cos_i_vs_phi_scatter", OUT_DIRS)
    plt.close(fig_single)

    # -------------------------------------------------------------
    # 図2: 貨物半径 R_c における 3 指標の比較サマリー
    # -------------------------------------------------------------
    summary_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue

        sub_inst = sub_inst.copy()
        sub_inst['unique_pid'] = sub_inst['exp_dir'].astype(str) + '_' + sub_inst['particle'].astype(str)

        p_agg = sub_inst.groupby('unique_pid').agg({
            'cos_i': 'mean',
            'cos_bulk': 'mean',
            'Phi_mag': 'mean',
            'pred_cos_i': 'mean'
        })

        n_p = len(p_agg)
        cos_i_vals = p_agg['cos_i'].values
        m_cos_i = float(np.mean(cos_i_vals))
        phi_vals = p_agg['Phi_mag'].values
        m_phi = float(np.mean(phi_vals))
        rng_b = np.random.default_rng(42)
        if n_p >= 2:
            bs_m = [np.mean(rng_b.choice(cos_i_vals, size=n_p, replace=True)) for _ in range(1000)]
            ci_low = float(np.percentile(bs_m, 2.5))
            ci_high = float(np.percentile(bs_m, 97.5))
            cos_i_yerr_low = m_cos_i - ci_low
            cos_i_yerr_high = ci_high - m_cos_i

            bs_phi = [np.mean(rng_b.choice(phi_vals, size=n_p, replace=True)) for _ in range(1000)]
            ci_phi_low = float(np.percentile(bs_phi, 2.5))
            ci_phi_high = float(np.percentile(bs_phi, 97.5))
            phi_yerr_low = m_phi - ci_phi_low
            phi_yerr_high = ci_phi_high - m_phi
        else:
            cos_i_yerr_low = 0.0
            cos_i_yerr_high = 0.0
            phi_yerr_low = 0.0
            phi_yerr_high = 0.0

        summary_rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'n_particles': n_p,
            'cos_i_mean': m_cos_i,
            'cos_i_sem': float(p_agg['cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_yerr_low': cos_i_yerr_low,
            'cos_i_yerr_high': cos_i_yerr_high,
            'cos_bulk_mean': float(p_agg['cos_bulk'].mean()),
            'cos_bulk_sem': float(p_agg['cos_bulk'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_mean': m_phi,
            'Phi_sem': float(p_agg['Phi_mag'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_yerr_low': phi_yerr_low,
            'Phi_yerr_high': phi_yerr_high,
            'pred_mean': float(p_agg['pred_cos_i'].mean()),
            'pred_sem': float(p_agg['pred_cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'ratio_obs_to_bulk': float(p_agg['cos_i'].mean() / p_agg['cos_bulk'].mean()) if p_agg['cos_bulk'].mean() != 0 else np.nan,
        })

    df_summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    r_vals = df_summary['radius_um'].values
    ax.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax.axhline(1, color='#333333', linestyle='--', linewidth=1.4)

    # cos_bulk (重心流速結合)
    ax.errorbar(r_vals, df_summary['cos_bulk_mean'], yerr=df_summary['cos_bulk_sem'],
                fmt='s--', color='#4477AA', markersize=8, capsize=4, linewidth=2.0,
                label=r"Bulk alignment $\cos \Theta_{\mathrm{bulk}} \equiv \frac{\mathbf{v}_{\mathrm{cargo}} \cdot \bar{\mathbf{v}}_{\mathrm{MT}}}{|\mathbf{v}_{\mathrm{cargo}}| |\bar{\mathbf{v}}_{\mathrm{MT}}|}$")

    # Local polar order Phi(R_c) (粒子間ブートストラップ非対称エラーバー)
    phi_yerr_summary = [df_summary['Phi_yerr_low'].values, df_summary['Phi_yerr_high'].values]
    ax.errorbar(r_vals, df_summary['Phi_mean'], yerr=phi_yerr_summary,
                fmt='^-.', color='#EE6677', markersize=8, capsize=4, linewidth=2.0,
                label=r"Local polar order $\Phi(R_c)$ (95% CI)")

    # Product Phi * cos_bulk (理論積)
    ax.errorbar(r_vals * 1.04, df_summary['pred_mean'], yerr=df_summary['pred_sem'],
                fmt='d:', color='#AA3377', markersize=8, capsize=4, linewidth=2.0,
                label=r"Decomposition $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$")

    # Actual <cos theta_i> (各点平均: 粒子間ブートストラップ非対称エラーバー)
    cos_i_yerr = [df_summary['cos_i_yerr_low'].values, df_summary['cos_i_yerr_high'].values]
    ax.errorbar(r_vals, df_summary['cos_i_mean'], yerr=cos_i_yerr,
                fmt='o-', color='#228833', markersize=9, capsize=5, linewidth=2.4,
                label=r"Observed $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.2, 15.0)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"Cargo Radius $R_c$ [$\mu\mathrm{m}$]")
    ax.set_ylabel("Coupling / Order Parameter [-]")
    ax.set_title(r"Decomposition of Directional Coupling: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}} \approx \Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$",
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
    ax.grid(True)

    fig.tight_layout()
    save_figure(fig, "cos_decomposition_vs_radius", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 図3: 窓サイズ依存性の 6 パネル分解プロット
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    window_summary_list = []

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        sub = sub.copy()
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'cos_bulk_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'Phi_mag': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'pred_cos_i': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'exp_dir': 'nunique'
        }).reset_index()

        r_w = agg['window_radius_um'].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values
        cb_m = agg[('cos_bulk_mean', 'mean')].values
        cb_s = agg[('cos_bulk_mean', '<lambda_0>')].values
        phi_m = agg[('Phi_mag', 'mean')].values
        phi_s = agg[('Phi_mag', '<lambda_0>')].values
        pred_m = agg[('pred_cos_i', 'mean')].values
        pred_s = agg[('pred_cos_i', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        n_e = agg[('exp_dir', 'nunique')].max()

        for i_rw in range(len(r_w)):
            window_summary_list.append({
                'bead_name': bname,
                'window_radius_um': r_w[i_rw],
                'cos_i_mean': ci_m[i_rw],
                'cos_i_sem': ci_s[i_rw],
                'cos_bulk_mean': cb_m[i_rw],
                'cos_bulk_sem': cb_s[i_rw],
                'Phi_mean': phi_m[i_rw],
                'Phi_sem': phi_s[i_rw],
                'pred_mean': pred_m[i_rw],
                'pred_sem': pred_s[i_rw],
            })

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)
        ax.axvline(r_c, color='#cc2222', linestyle=':', linewidth=1.5, zorder=2, label=f'Cargo $R_c = {r_c:.2f}\,\mu\mathrm{{m}}$')

        # 実験相関長 xi
        exp_names = sub['exp_dir'].unique()
        xis = [xi_dict[e] for e in exp_names if e in xi_dict and not np.isnan(xi_dict[e])]
        if xis:
            xi_mean = float(np.mean(xis))
            ax.axvline(xi_mean, color='#2b5c8f', linestyle='--', linewidth=1.6, zorder=2,
                       label=rf'Corr. length $\xi = {xi_mean:.2f}\,\mu\mathrm{{m}}$')

        # 1. 局所ポーラーオーダー Phi(R)
        ax.plot(r_w, phi_m, linestyle='-.', color='#EE6677', linewidth=2.0, label=r'$\Phi(R)$ (from local_polar_w.zarr)')
        ax.fill_between(r_w, phi_m - phi_s, phi_m + phi_s, color='#EE6677', alpha=0.15)

        # 2. 重心流速方向一致度 cos_bulk(R)
        ax.plot(r_w, cb_m, linestyle='--', color='#4477AA', linewidth=2.0, label=r'$\cos \Theta_{\mathrm{bulk}}(R)$')
        ax.fill_between(r_w, cb_m - cb_s, cb_m + cb_s, color='#4477AA', alpha=0.15)

        # 3. 理論積 Phi * cos_bulk
        ax.plot(r_w, pred_m, linestyle=':', color='#AA3377', linewidth=2.2, label=r'$\Phi(R) \times \cos \Theta_{\mathrm{bulk}}(R)$')

        # 4. 観測各点平均 <cos theta_i>(R)
        ax.plot(r_w, ci_m, marker=b_item['marker'], markersize=6, color=color, linewidth=2.4,
                label=rf'$\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}}(R)$ (Observed)')
        ax.fill_between(r_w, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        ax.set_xscale('log')
        ax.set_xlim(0.4, 120.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(f"{lbl}  ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$, $N={n_p}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel("Coupling / Order Parameter", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Window Dependence Decomposition: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}(R) \approx \Phi(R) \times \cos \Theta_{\mathrm{bulk}}(R)$',
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_decomposition_6panels", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 図4: 貨物半径の無次元化 R_c / \xi における指標プロット (cos_decomposition_vs_rc_over_xi)
    # -------------------------------------------------------------
    particle_inst_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue
        sub_inst = sub_inst.copy()
        sub_inst['unique_pid'] = sub_inst['exp_dir'].astype(str) + '_' + sub_inst['particle'].astype(str)

        p_agg = sub_inst.groupby('unique_pid').agg({
            'cos_i': 'mean',
            'cos_bulk': 'mean',
            'Phi_mag': 'mean',
            'pred_cos_i': 'mean',
            'radius_um': 'first',
            'xi_um': 'first',
            'bead_name': 'first',
        })
        p_agg['rc_over_xi'] = p_agg['radius_um'] / p_agg['xi_um']
        particle_inst_rows.append(p_agg)

    df_p_inst = pd.concat(particle_inst_rows, ignore_index=True)

    rc_xi_summary = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_p = df_p_inst[df_p_inst['bead_name'] == bname]
        if sub_p.empty:
            continue
        n_p = len(sub_p)
        cos_i_vals = sub_p['cos_i'].values
        m_cos_i = float(np.mean(cos_i_vals))
        phi_vals = sub_p['Phi_mag'].values
        m_phi = float(np.mean(phi_vals))
        rng_b = np.random.default_rng(42)
        if n_p >= 2:
            bs_m = [np.mean(rng_b.choice(cos_i_vals, size=n_p, replace=True)) for _ in range(1000)]
            ci_low = float(np.percentile(bs_m, 2.5))
            ci_high = float(np.percentile(bs_m, 97.5))
            cos_i_yerr_low = m_cos_i - ci_low
            cos_i_yerr_high = ci_high - m_cos_i

            bs_phi = [np.mean(rng_b.choice(phi_vals, size=n_p, replace=True)) for _ in range(1000)]
            ci_phi_low = float(np.percentile(bs_phi, 2.5))
            ci_phi_high = float(np.percentile(bs_phi, 97.5))
            phi_yerr_low = m_phi - ci_phi_low
            phi_yerr_high = ci_phi_high - m_phi
        else:
            cos_i_yerr_low = 0.0
            cos_i_yerr_high = 0.0
            phi_yerr_low = 0.0
            phi_yerr_high = 0.0

        rc_xi_summary.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'radius_um': b_item['radius_um'],
            'n_particles': n_p,
            'rc_over_xi_mean': float(sub_p['rc_over_xi'].mean()),
            'rc_over_xi_sem': float(sub_p['rc_over_xi'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_mean': m_cos_i,
            'cos_i_sem': float(sub_p['cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_yerr_low': cos_i_yerr_low,
            'cos_i_yerr_high': cos_i_yerr_high,
            'cos_bulk_mean': float(sub_p['cos_bulk'].mean()),
            'cos_bulk_sem': float(sub_p['cos_bulk'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_mean': m_phi,
            'Phi_sem': float(sub_p['Phi_mag'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_yerr_low': phi_yerr_low,
            'Phi_yerr_high': phi_yerr_high,
            'pred_mean': float(sub_p['pred_cos_i'].mean()),
            'pred_sem': float(sub_p['pred_cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
        })
    df_rc_xi = pd.DataFrame(rc_xi_summary)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax.axhline(1, color='#333333', linestyle='--', linewidth=1.4)
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, label=r'Crossover $R_c / \xi = 1$')

    # 各粒子の個別データ（薄くプロット）
    for b_item in BEADS_INFO:
        sub_p = df_p_inst[df_p_inst['bead_name'] == b_item['name']]
        if not sub_p.empty:
            ax.scatter(sub_p['rc_over_xi'], sub_p['cos_i'], color=b_item['color'], marker=b_item['marker'],
                       alpha=0.18, s=28, zorder=2)

    rc_xi_vals = df_rc_xi['rc_over_xi_mean'].values

    # cos_bulk (重心流速結合)
    ax.errorbar(rc_xi_vals, df_rc_xi['cos_bulk_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['cos_bulk_sem'],
                fmt='s--', color='#4477AA', markersize=8, capsize=4, linewidth=2.0, zorder=4,
                label=r"Bulk alignment $\cos \Theta_{\mathrm{bulk}}$")

    # Local polar order Phi(R_c) (粒子間ブートストラップ非対称エラーバー)
    phi_yerr_rc = [df_rc_xi['Phi_yerr_low'].values, df_rc_xi['Phi_yerr_high'].values]
    ax.errorbar(rc_xi_vals, df_rc_xi['Phi_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=phi_yerr_rc,
                fmt='^-.', color='#EE6677', markersize=8, capsize=4, linewidth=2.0, zorder=4,
                label=r"Local polar order $\Phi(R_c)$ (95% CI)")

    # Product Phi * cos_bulk (理論積)
    ax.errorbar(rc_xi_vals * 1.03, df_rc_xi['pred_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['pred_sem'],
                fmt='d:', color='#AA3377', markersize=8, capsize=4, linewidth=2.0, zorder=5,
                label=r"Decomposition $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$")

    # Actual <cos theta_i> (各点平均: 粒子間ブートストラップ非対称エラーバー)
    cos_i_yerr_rc = [df_rc_xi['cos_i_yerr_low'].values, df_rc_xi['cos_i_yerr_high'].values]
    ax.errorbar(rc_xi_vals, df_rc_xi['cos_i_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=cos_i_yerr_rc,
                fmt='o-', color='#228833', markersize=9, capsize=5, linewidth=2.4, zorder=6,
                label=r"Observed $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]")
    ax.set_ylabel("Coupling / Order Parameter [-]")
    ax.set_title(r"Decomposition vs Normalized Cargo Radius: $R_c / \xi$",
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
    ax.grid(True)

    fig.tight_layout()
    save_figure(fig, "cos_decomposition_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 図5: 窓サイズ依存性の 6 パネル分解プロット (横軸 R / \xi)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        sub = sub.copy()
        sub['r_over_xi'] = sub['window_radius_um'] / sub['xi_um']
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'r_over_xi': 'mean',
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'cos_bulk_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'Phi_mag': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'pred_cos_i': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'exp_dir': 'nunique',
            'xi_um': 'mean',
        }).reset_index()

        r_norm = agg[('r_over_xi', 'mean')].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values
        cb_m = agg[('cos_bulk_mean', 'mean')].values
        cb_s = agg[('cos_bulk_mean', '<lambda_0>')].values
        phi_m = agg[('Phi_mag', 'mean')].values
        phi_s = agg[('Phi_mag', '<lambda_0>')].values
        pred_m = agg[('pred_cos_i', 'mean')].values
        pred_s = agg[('pred_cos_i', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        xi_avg = agg[('xi_um', 'mean')].mean()
        rc_over_xi_avg = r_c / xi_avg if xi_avg > 0 else np.nan

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)
        
        ax.axvline(1.0, color='#2b5c8f', linestyle='--', linewidth=1.6, zorder=2,
                   label=r'Correlation length $R / \xi = 1$')
        ax.axvline(rc_over_xi_avg, color='#cc2222', linestyle=':', linewidth=1.5, zorder=2,
                   label=rf'Cargo $R_c / \xi = {rc_over_xi_avg:.2f}$')

        # 1. 局所ポーラーオーダー Phi(R)
        ax.plot(r_norm, phi_m, linestyle='-.', color='#EE6677', linewidth=2.0, label=r'$\Phi(R)$')
        ax.fill_between(r_norm, phi_m - phi_s, phi_m + phi_s, color='#EE6677', alpha=0.15)

        # 2. 重心流速方向一致度 cos_bulk(R)
        ax.plot(r_norm, cb_m, linestyle='--', color='#4477AA', linewidth=2.0, label=r'$\cos \Theta_{\mathrm{bulk}}(R)$')
        ax.fill_between(r_norm, cb_m - cb_s, cb_m + cb_s, color='#4477AA', alpha=0.15)

        # 3. 理論積 Phi * cos_bulk
        ax.plot(r_norm, pred_m, linestyle=':', color='#AA3377', linewidth=2.2, label=r'$\Phi(R) \times \cos \Theta_{\mathrm{bulk}}(R)$')

        # 4. 観測各点平均 <cos theta_i>(R)
        ax.plot(r_norm, ci_m, marker=b_item['marker'], markersize=6, color=color, linewidth=2.4,
                label=rf'$\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}}(R)$')
        ax.fill_between(r_norm, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        ax.set_xscale('log')
        ax.set_xlim(0.08, 25.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(rf"{lbl}  ($R_c/\langle\xi\rangle = {rc_over_xi_avg:.2f}$, $N={n_p}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Normalized Window Radius $R / \xi$ [-]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel("Coupling / Order Parameter", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Window Dependence Decomposition vs Normalized Radius $R / \xi$',
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_decomposition_6panels_normalized_xi", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 図6: 全ビーズ重ね合わせ Master Curve プロット (横軸 R / \xi)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.5))

    # Panel A: 観測各点結合 <cos theta_i>(R) の R/xi 依存性重ね合わせ
    ax = axes[0]
    ax.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax.axhline(1, color='#333333', linestyle='--', linewidth=1.4)
    ax.axvline(1.0, color='#2b5c8f', linestyle='--', linewidth=1.6, label=r'Correlation length $R / \xi = 1$')

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            continue
        sub = sub.copy()
        sub['r_over_xi'] = sub['window_radius_um'] / sub['xi_um']
        agg = sub.groupby('window_radius_um').agg({
            'r_over_xi': 'mean',
            'cos_i_mean': 'mean'
        }).reset_index()
        ax.plot(agg['r_over_xi'], agg['cos_i_mean'], marker=b_item['marker'], markersize=6,
                color=b_item['color'], linewidth=2.0, label=b_item['label'])

    ax.set_xscale('log')
    ax.set_xlim(0.08, 25.0)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel(r"Normalized Window Radius $R / \xi$ [-]")
    ax.set_ylabel(r"Observed Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}(R)$")
    ax.set_title(r"Observed Coupling vs $R / \xi$ (All Cargo Sizes)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True)

    # Panel B: 局所ポーラーオーダー Phi(R) の R/xi 依存性重ね合わせ
    ax2 = axes[1]
    ax2.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax2.axhline(1, color='#333333', linestyle='--', linewidth=1.4)
    ax2.axvline(1.0, color='#2b5c8f', linestyle='--', linewidth=1.6, label=r'Correlation length $R / \xi = 1$')

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            continue
        sub = sub.copy()
        sub['r_over_xi'] = sub['window_radius_um'] / sub['xi_um']
        agg = sub.groupby('window_radius_um').agg({
            'r_over_xi': 'mean',
            'Phi_mag': 'mean'
        }).reset_index()
        ax2.plot(agg['r_over_xi'], agg['Phi_mag'], marker=b_item['marker'], markersize=6,
                 color=b_item['color'], linewidth=2.0, label=b_item['label'])

    ax2.set_xscale('log')
    ax2.set_xlim(0.08, 25.0)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel(r"Normalized Window Radius $R / \xi$ [-]")
    ax2.set_ylabel(r"Local Polar Order $\Phi(R)$")
    ax2.set_title(r"Local Polar Order vs $R / \xi$ (Scaling Collapse)", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True)

    fig.tight_layout()
    save_figure(fig, "cos_decomposition_master_curve_r_over_xi", OUT_DIRS)
    plt.close(fig)


    # -------------------------------------------------------------
    # 図7: 窓サイズ依存性の 6 パネル分解プロット only <cos \theta_i>
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    window_summary_list = []

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        sub = sub.copy()
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'pred_cos_i': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'exp_dir': 'nunique'
        }).reset_index()

        r_w = agg['window_radius_um'].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values
        pred_m = agg[('pred_cos_i', 'mean')].values
        pred_s = agg[('pred_cos_i', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        n_e = agg[('exp_dir', 'nunique')].max()

        for i_rw in range(len(r_w)):
            window_summary_list.append({
                'bead_name': bname,
                'window_radius_um': r_w[i_rw],
                'cos_i_mean': ci_m[i_rw],
                'cos_i_sem': ci_s[i_rw],
                'pred_mean': pred_m[i_rw],
                'pred_sem': pred_s[i_rw],
            })

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)
        ax.axvline(r_c, color='#cc2222', linestyle=':', linewidth=1.5, zorder=2, label=f'Cargo $R_c = {r_c:.2f}\,\mu\mathrm{{m}}$')

        # 実験相関長 xi
        exp_names = sub['exp_dir'].unique()
        xis = [xi_dict[e] for e in exp_names if e in xi_dict and not np.isnan(xi_dict[e])]
        if xis:
            xi_mean = float(np.mean(xis))
            ax.axvline(xi_mean, color='#2b5c8f', linestyle='--', linewidth=1.6, zorder=2,
                       label=rf'Corr. length $\xi = {xi_mean:.2f}\,\mu\mathrm{{m}}$')

        # 観測各点平均 <cos theta_i>(R)
        ax.plot(r_w, ci_m, marker=b_item['marker'], markersize=6, color=color, linewidth=2.4,
                label=rf'$\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}}(R)$')
        ax.fill_between(r_w, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        #ax.set_xscale('log')
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(f"{lbl}  ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$, $N={n_p}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}\left(R\right)$", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Window Dependence: $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}(R)$',
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_decomposition_6panels_raw", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 図8: 窓サイズ依存性の 1 パネルプロット only <cos \theta_i>
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    window_summary_list = []

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            continue

        sub = sub.copy()
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'pred_cos_i': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'exp_dir': 'nunique'
        }).reset_index()

        r_w = agg['window_radius_um'].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        n_e = agg[('exp_dir', 'nunique')].max()

        for i_rw in range(len(r_w)):
            window_summary_list.append({
                'bead_name': bname,
                'window_radius_um': r_w[i_rw],
                'cos_i_mean': ci_m[i_rw],
                'cos_i_sem': ci_s[i_rw],
            })
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        sub = sub.copy()
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'pred_cos_i': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'exp_dir': 'nunique'
        }).reset_index()

        r_w = agg['window_radius_um'].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values
        pred_m = agg[('pred_cos_i', 'mean')].values
        pred_s = agg[('pred_cos_i', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        n_e = agg[('exp_dir', 'nunique')].max()

        for i_rw in range(len(r_w)):
            window_summary_list.append({
                'bead_name': bname,
                'window_radius_um': r_w[i_rw],
                'cos_i_mean': ci_m[i_rw],
                'cos_i_sem': ci_s[i_rw],
                'pred_mean': pred_m[i_rw],
                'pred_sem': pred_s[i_rw],
            })

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)

        # 観測各点平均 <cos theta_i>(R)
        ax.plot(r_w, ci_m, marker=b_item['marker'], markersize=6, color=color, linewidth=2.4,
                label=rf'$\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}}(R)$')
        ax.fill_between(r_w, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        #ax.set_xscale('log')
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}\left(R\right)$", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_decomposition_raw", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 図9: 貨物半径の無次元化 R_c / \xi における指標プロット (cos_decomposition_vs_rc_over_xi)
    # -------------------------------------------------------------
    particle_inst_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue
        sub_inst = sub_inst.copy()
        sub_inst['unique_pid'] = sub_inst['exp_dir'].astype(str) + '_' + sub_inst['particle'].astype(str)

        p_agg = sub_inst.groupby('unique_pid').agg({
            'cos_i': 'mean',
            'Phi_mag': 'mean',
            'pred_cos_i': 'mean',
            'radius_um': 'first',
            'xi_um': 'first',
            'bead_name': 'first',
        })
        p_agg['rc_over_xi'] = p_agg['radius_um'] / p_agg['xi_um']
        particle_inst_rows.append(p_agg)

    df_p_inst = pd.concat(particle_inst_rows, ignore_index=True)

    rc_xi_summary = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_p = df_p_inst[df_p_inst['bead_name'] == bname]
        if sub_p.empty:
            continue
        n_p = len(sub_p)
        cos_i_vals = sub_p['cos_i'].values
        m_cos_i = float(np.mean(cos_i_vals))
        phi_vals = sub_p['Phi_mag'].values
        m_phi = float(np.mean(phi_vals))
        rng_b = np.random.default_rng(42)
        if n_p >= 2:
            bs_m = [np.mean(rng_b.choice(cos_i_vals, size=n_p, replace=True)) for _ in range(1000)]
            ci_low = float(np.percentile(bs_m, 2.5))
            ci_high = float(np.percentile(bs_m, 97.5))
            cos_i_yerr_low = m_cos_i - ci_low
            cos_i_yerr_high = ci_high - m_cos_i

            bs_phi = [np.mean(rng_b.choice(phi_vals, size=n_p, replace=True)) for _ in range(1000)]
            ci_phi_low = float(np.percentile(bs_phi, 2.5))
            ci_phi_high = float(np.percentile(bs_phi, 97.5))
            phi_yerr_low = m_phi - ci_phi_low
            phi_yerr_high = ci_phi_high - m_phi
        else:
            cos_i_yerr_low = 0.0
            cos_i_yerr_high = 0.0
            phi_yerr_low = 0.0
            phi_yerr_high = 0.0

        rc_xi_summary.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'radius_um': b_item['radius_um'],
            'n_particles': n_p,
            'rc_over_xi_mean': float(sub_p['rc_over_xi'].mean()),
            'rc_over_xi_sem': float(sub_p['rc_over_xi'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_mean': m_cos_i,
            'cos_i_sem': float(sub_p['cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_yerr_low': cos_i_yerr_low,
            'cos_i_yerr_high': cos_i_yerr_high,
            'Phi_mean': m_phi,
            'Phi_sem': float(sub_p['Phi_mag'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_yerr_low': phi_yerr_low,
            'Phi_yerr_high': phi_yerr_high,
            'pred_mean': float(sub_p['pred_cos_i'].mean()),
            'pred_sem': float(sub_p['pred_cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
        })
    df_rc_xi = pd.DataFrame(rc_xi_summary)
    if not df_rc_xi.empty:
        df_rc_xi = df_rc_xi.sort_values('rc_over_xi_mean').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax2 = ax.twinx()

    ax.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    ax.axhline(1, color='#333333', linestyle='--', linewidth=1.4)
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, label=r'Crossover $R_c / \xi = 1$')

    # 各粒子の個別データ（薄くプロット）
    for b_item in BEADS_INFO:
        sub_p = df_p_inst[df_p_inst['bead_name'] == b_item['name']]
        if not sub_p.empty:
            ax.scatter(sub_p['rc_over_xi'], sub_p['cos_i'], color=b_item['color'], marker=b_item['marker'],
                       alpha=0.18, s=28, zorder=2)

    rc_xi_vals = df_rc_xi['rc_over_xi_mean'].values

    # Local polar order Phi(R_c) (粒子間ブートストラップ非対称エラーバー)
    phi_yerr_raw = [df_rc_xi['Phi_yerr_low'].values, df_rc_xi['Phi_yerr_high'].values]
    ax2.errorbar(rc_xi_vals, df_rc_xi['Phi_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=phi_yerr_raw,
                fmt='^-.', color='#EE6677', markersize=8, capsize=4, linewidth=2.0, zorder=4,
                label=r"Local polar order $\Phi(R_c)$ (95% CI)")

    # Actual <cos theta_i> (各点平均: 粒子間ブートストラップ非対称エラーバー)
    cos_i_yerr_raw = [df_rc_xi['cos_i_yerr_low'].values, df_rc_xi['cos_i_yerr_high'].values]
    ax.errorbar(rc_xi_vals, df_rc_xi['cos_i_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=cos_i_yerr_raw,
                fmt='o-', color='#1f77b4', markersize=9, capsize=5, linewidth=2.4, zorder=6,
                label=r"Observed $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ (95% CI)")

    #ax.set_xscale('log')
    ax.set_xlim(0.0, 3.5)
    ax.set_ylim(-0.2, 1.05)
    ax2.set_ylim(-0.2, 1.05)
    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]")
    ax.set_ylabel(r"$\langle \mathbf{\hat{v}} \cdot \mathbf{\hat{u}} \rangle\left(\frac{R_c}{\xi}\right)$", color='#1f77b4')
    ax2.set_ylabel(r"Local polar order $\Phi\left(\frac{R_c}{\xi}\right)$", color='#EE6677')
    ax.tick_params(axis='y', colors='#1f77b4', which='both')
    ax2.tick_params(axis='y', colors='#EE6677', which='both')
    ax.spines['left'].set_color('#1f77b4')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_color('#EE6677')
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # 凡例統合
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10.0, framealpha=0.92)
    ax.grid(True)

    fig.tight_layout()
    save_figure(fig, "cos_decomposition_vs_rc_over_xi_raw", OUT_DIRS)
    plt.close(fig)


    # -------------------------------------------------------------
    # 図10: 窓サイズ依存性の 6 パネル分解プロット (横軸 R / \xi)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        sub = sub.copy()
        sub['r_over_xi'] = sub['window_radius_um'] / sub['xi_um']
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'r_over_xi': 'mean',
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'cos_bulk_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'Phi_mag': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'pred_cos_i': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'exp_dir': 'nunique',
            'xi_um': 'mean',
        }).reset_index()

        r_norm = agg[('r_over_xi', 'mean')].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values
        pred_m = agg[('pred_cos_i', 'mean')].values
        pred_s = agg[('pred_cos_i', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        xi_avg = agg[('xi_um', 'mean')].mean()
        rc_over_xi_avg = r_c / xi_avg if xi_avg > 0 else np.nan

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)
        
        ax.axvline(1.0, color='#2b5c8f', linestyle='--', linewidth=1.6, zorder=2,
                   label=r'Correlation length $R / \xi = 1$')
        ax.axvline(rc_over_xi_avg, color='#cc2222', linestyle=':', linewidth=1.5, zorder=2,
                   label=rf'Cargo $R_c / \xi = {rc_over_xi_avg:.2f}$')

        # 1. 観測各点平均 <cos theta_i>(R)
        ax.plot(r_norm, ci_m, marker=b_item['marker'], markersize=6, color=color, linewidth=2.4,
                label=rf'$\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}}(R/\xi)$')
        ax.fill_between(r_norm, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        ax.set_xscale('log')
        ax.set_xlim(0.08, 25.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(rf"{lbl}  ($R_c/\langle\xi\rangle = {rc_over_xi_avg:.2f}$, $N={n_p}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Normalized Window Radius $R / \xi$ [-]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Window Dependence Decomposition vs Normalized Radius $R / \xi$',
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_decomposition_6panels_cos_theta_i_only", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 7. 全ビーズサイズ統合 1 パネルプロット: <cos theta_i>(R/xi)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
    ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)
    ax.axvline(1.0, color='#2b5c8f', linestyle='--', linewidth=1.8, zorder=2,
               label=r'Correlation length $R / \xi = 1$')

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        r_c = b_item['radius_um']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_window[df_window['bead_name'] == bname]
        if sub.empty:
            continue

        sub = sub.copy()
        sub['r_over_xi'] = sub['window_radius_um'] / sub['xi_um']
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)

        agg = sub.groupby('window_radius_um').agg({
            'r_over_xi': 'mean',
            'cos_i_mean': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.0],
            'unique_pid': 'nunique',
            'xi_um': 'mean',
        }).reset_index()

        r_norm = agg[('r_over_xi', 'mean')].values
        ci_m = agg[('cos_i_mean', 'mean')].values
        ci_s = agg[('cos_i_mean', '<lambda_0>')].values

        n_p = agg[('unique_pid', 'nunique')].max()
        xi_avg = agg[('xi_um', 'mean')].mean()
        rc_over_xi_avg = r_c / xi_avg if xi_avg > 0 else np.nan

        line_lbl = rf"{lbl} ($R_c/\langle\xi\rangle = {rc_over_xi_avg:.2f}$, $N={n_p}$)"
        ax.plot(r_norm, ci_m, marker=b_item['marker'], markersize=7, color=color, linewidth=2.5,
                label=line_lbl, zorder=3 + idx)
        ax.fill_between(r_norm, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.15, zorder=2)

    ax.set_xscale('log')
    ax.set_xlim(0.08, 25.0)
    ax.set_ylim(-0.15, 0.65)
    ax.set_xlabel(r"Normalized Window Radius $R / \xi$ [-]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Pixel-Averaged Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}(R/\xi)$", fontsize=14, labelpad=8)
    ax.set_title(r"Cargo-Flow Coupling Decay $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ vs $R / \xi$ (All Cargo Sizes)",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8, zorder=0)
    ax.legend(loc='upper right', fontsize=10.0, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "cos_decomposition_1panel_cos_theta_i_only", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 8. 方向一致度 cos theta_i の確率密度ヒストグラム P(cos theta_i)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    bins_cos = np.linspace(-1.0, 1.0, 29) # 28ビン (-1から1)
    cos_grid = np.linspace(-0.9999, 0.9999, 400)

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        color = b_item['color']
        lbl = b_item['label']

        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        cos_vals = sub_inst['cos_i'].dropna().values
        phi_mean_val = float(df_rc_xi[df_rc_xi['bead_name'] == bname]['Phi_mean'].values[0]) if not df_rc_xi.empty else np.nan
        ci_mean_val = float(np.mean(cos_vals)) if len(cos_vals) > 0 else np.nan

        # 実測ヒストグラム (確率密度 normalized)
        ax.hist(cos_vals, bins=bins_cos, density=True, color=color, alpha=0.45,
                edgecolor=color, linewidth=1.2, label=rf"Observed ($N = {len(cos_vals)}$)")

        # 代表粒子平均値の縦線
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)
        ax.axvline(ci_mean_val, color=color, linestyle='-', linewidth=2.0,
                   label=rf"Mean $\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}} = {ci_mean_val:.2f}$")

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0.0, 4.0)
        ax.set_title(rf"{lbl}  ($\Phi = {phi_mean_val:.2f}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9.2, framealpha=0.92)

    fig.suptitle(r'Probability Density of Directional Coupling $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$',
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_i_probability_density_histograms", OUT_DIRS)
    plt.close(fig)

    # 全粒子統合の 1 パネル版ヒストグラム (KDE / ヒストグラム重ね合わせ)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        color = b_item['color']
        lbl = b_item['label']

        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue

        cos_vals = sub_inst['cos_i'].dropna().values
        counts, bin_edges = np.histogram(cos_vals, bins=bins_cos, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(bin_centers, counts, marker=b_item['marker'], markersize=6, color=color,
                linewidth=2.2, label=lbl)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 4.0)
    ax.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Probability Density $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$", fontsize=14, labelpad=8)
    ax.set_title(r"Probability Density of Cargo-Flow Coupling $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ (All Cargo Sizes)",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "cos_i_probability_density_1panel", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 9. 貨物粒子単位の歪度 \gamma_{1, i} およびその平均 <\gamma_1> の算出と可視化
    #    1. 貨物 i の全ピクセルデータ（各フレーム）から通常の平均 mu_i, 標準偏差 sigma_i, 歪度 \gamma_{1, i} を計算
    #    2. 各サイズごとに全貨物平均 <\gamma_1> = (1 / N_cargo) * sum(\gamma_{1, i}) を算出
    #    3. エラーバーは貨物間のばらつき（\gamma_{1, i} のブートストラップ信頼区間 / SEM）から決定
    # -------------------------------------------------------------
    df_instant['cargo_id'] = df_instant['exp_dir'].astype(str) + '_' + df_instant['particle'].astype(str)

    cargo_stat_records = []
    skew_records = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue

        row_rc = df_rc_xi[df_rc_xi['bead_name'] == bname]
        phi_mean_val = float(row_rc['Phi_mean'].values[0]) if not row_rc.empty else np.nan
        rc_xi_m = float(row_rc['rc_over_xi_mean'].values[0]) if not row_rc.empty else np.nan
        rc_xi_s = float(row_rc['rc_over_xi_sem'].values[0]) if not row_rc.empty else 0.0

        # 各貨物粒子ごとの集計
        cargo_skews = []
        cargo_means = []
        cargo_stds = []
        cargo_m2_centrals = []
        cargo_m2_raws = []
        cargo_m3_centrals = []
        cargo_m3_raws = []
        cargo_m5_centrals = []
        cargo_m5_raws = []

        for cid, grp in sub_inst.groupby('cargo_id'):
            c_vals = grp['cos_i'].dropna().values
            n_c_pts = len(c_vals)
            if n_c_pts < 3:
                continue

            mu_i = float(np.mean(c_vals))
            sigma_i = float(np.std(c_vals, ddof=1)) if n_c_pts > 1 else 0.0
            gamma1_i = float(skew(c_vals, bias=False))

            # 2次中心モーメント mu_{2, i} = Var(X) = 1/(n-1) * sum((x-mu)^2)
            mu2_central_i = float(np.var(c_vals, ddof=1)) if n_c_pts > 1 else 0.0
            # 2次原点モーメント m_{2, i} = E[X^2]
            m2_raw_i = float(np.mean(c_vals**2))

            # 3次中心モーメント mu_{3, i} = E[(X - mu)^3] (不偏推定量: n / ((n-1)(n-2)) * sum((x-mu)^3))
            mu3_central_i = float(np.sum((c_vals - mu_i)**3) * n_c_pts / ((n_c_pts - 1) * (n_c_pts - 2))) if n_c_pts > 2 else float(np.mean((c_vals - mu_i)**3))
            # 3次原点モーメント m_{3, i} = E[X^3]
            m3_raw_i = float(np.mean(c_vals**3))

            # 5次中心モーメント mu_{5, i} = E[(X - mu)^5]
            mu5_central_i = float(np.mean((c_vals - mu_i)**5))
            # 5次原点モーメント m_{5, i} = E[X^5]
            m5_raw_i = float(np.mean(c_vals**5))

            if np.isfinite(gamma1_i):
                cargo_skews.append(gamma1_i)
                cargo_means.append(mu_i)
                cargo_stds.append(sigma_i)
                cargo_m2_centrals.append(mu2_central_i)
                cargo_m2_raws.append(m2_raw_i)
                cargo_m3_centrals.append(mu3_central_i)
                cargo_m3_raws.append(m3_raw_i)
                cargo_m5_centrals.append(mu5_central_i)
                cargo_m5_raws.append(m5_raw_i)

                cargo_stat_records.append({
                    'bead_name': bname,
                    'label': b_item['label'],
                    'cargo_id': cid,
                    'n_frames': n_c_pts,
                    'mu_i': mu_i,
                    'sigma_i': sigma_i,
                    'gamma1_i': gamma1_i,
                    'mu2_central_i': mu2_central_i,
                    'm2_raw_i': m2_raw_i,
                    'mu3_central_i': mu3_central_i,
                    'm3_raw_i': m3_raw_i,
                    'mu5_central_i': mu5_central_i,
                    'm5_raw_i': m5_raw_i,
                })

        cargo_skews = np.array(cargo_skews)
        cargo_m2_centrals = np.array(cargo_m2_centrals)
        cargo_m2_raws = np.array(cargo_m2_raws)
        cargo_m3_centrals = np.array(cargo_m3_centrals)
        cargo_m3_raws = np.array(cargo_m3_raws)
        cargo_m5_centrals = np.array(cargo_m5_centrals)
        cargo_m5_raws = np.array(cargo_m5_raws)
        n_cargos = len(cargo_skews)
        if n_cargos == 0:
            continue

        # 2. 全貨物での平均 <\gamma_1> = 1/N_cargo * sum(\gamma_{1, i})
        mean_skew = float(np.mean(cargo_skews))
        std_skew = float(np.std(cargo_skews, ddof=1)) if n_cargos > 1 else 0.0
        sem_skew = std_skew / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 2次中心モーメント平均 <\mu_2> (分散)
        mean_m2_central = float(np.mean(cargo_m2_centrals))
        std_m2_central = float(np.std(cargo_m2_centrals, ddof=1)) if n_cargos > 1 else 0.0
        sem_m2_central = std_m2_central / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 2次原点モーメント平均 <m_2> = <(\hat{v}\cdot\hat{u})^2>
        mean_m2_raw = float(np.mean(cargo_m2_raws))
        std_m2_raw = float(np.std(cargo_m2_raws, ddof=1)) if n_cargos > 1 else 0.0
        sem_m2_raw = std_m2_raw / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 3次中心モーメント平均 <\mu_3>
        mean_m3_central = float(np.mean(cargo_m3_centrals))
        std_m3_central = float(np.std(cargo_m3_centrals, ddof=1)) if n_cargos > 1 else 0.0
        sem_m3_central = std_m3_central / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 3次原点モーメント平均 <m_3> = <(\hat{v}\cdot\hat{u})^3>
        mean_m3_raw = float(np.mean(cargo_m3_raws))
        std_m3_raw = float(np.std(cargo_m3_raws, ddof=1)) if n_cargos > 1 else 0.0
        sem_m3_raw = std_m3_raw / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 5次中心モーメント平均 <\mu_5>
        mean_m5_central = float(np.mean(cargo_m5_centrals))
        std_m5_central = float(np.std(cargo_m5_centrals, ddof=1)) if n_cargos > 1 else 0.0
        sem_m5_central = std_m5_central / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 5次原点モーメント平均 <m_5> = <(\hat{v}\cdot\hat{u})^5>
        mean_m5_raw = float(np.mean(cargo_m5_raws))
        std_m5_raw = float(np.std(cargo_m5_raws, ddof=1)) if n_cargos > 1 else 0.0
        sem_m5_raw = std_m5_raw / np.sqrt(n_cargos) if n_cargos > 0 else 0.0

        # 3. 貨物間のばらつきからブートストラップ法（N=1000）で信頼区間（95% CI）および SEM を決定
        rng = np.random.default_rng(42)
        if n_cargos >= 2:
            def calc_bs_ci(arr):
                bs_vals = [np.mean(rng.choice(arr, size=n_cargos, replace=True)) for _ in range(1000)]
                c_low = float(np.percentile(bs_vals, 2.5))
                c_high = float(np.percentile(bs_vals, 97.5))
                b_sem = float(np.std(bs_vals, ddof=1))
                return c_low, c_high, b_sem

            ci_low, ci_high, bs_sem = calc_bs_ci(cargo_skews)
            ci_m2_c_low, ci_m2_c_high, bs_m2_c_sem = calc_bs_ci(cargo_m2_centrals)
            ci_m2_r_low, ci_m2_r_high, bs_m2_r_sem = calc_bs_ci(cargo_m2_raws)
            ci_m3_c_low, ci_m3_c_high, bs_m3_c_sem = calc_bs_ci(cargo_m3_centrals)
            ci_m3_r_low, ci_m3_r_high, bs_m3_r_sem = calc_bs_ci(cargo_m3_raws)
            ci_m5_c_low, ci_m5_c_high, bs_m5_c_sem = calc_bs_ci(cargo_m5_centrals)
            ci_m5_r_low, ci_m5_r_high, bs_m5_r_sem = calc_bs_ci(cargo_m5_raws)
        else:
            ci_low, ci_high, bs_sem = mean_skew, mean_skew, 0.0
            ci_m2_c_low, ci_m2_c_high, bs_m2_c_sem = mean_m2_central, mean_m2_central, 0.0
            ci_m2_r_low, ci_m2_r_high, bs_m2_r_sem = mean_m2_raw, mean_m2_raw, 0.0
            ci_m3_c_low, ci_m3_c_high, bs_m3_c_sem = mean_m3_central, mean_m3_central, 0.0
            ci_m3_r_low, ci_m3_r_high, bs_m3_r_sem = mean_m3_raw, mean_m3_raw, 0.0
            ci_m5_c_low, ci_m5_c_high, bs_m5_c_sem = mean_m5_central, mean_m5_central, 0.0
            ci_m5_r_low, ci_m5_r_high, bs_m5_r_sem = mean_m5_raw, mean_m5_raw, 0.0

        # エラーバーの非対称幅（yerr 用）
        yerr_low = mean_skew - ci_low
        yerr_high = ci_high - mean_skew

        yerr_m2_c_low = mean_m2_central - ci_m2_c_low
        yerr_m2_c_high = ci_m2_c_high - mean_m2_central
        yerr_m2_r_low = mean_m2_raw - ci_m2_r_low
        yerr_m2_r_high = ci_m2_r_high - mean_m2_raw

        yerr_m3_c_low = mean_m3_central - ci_m3_c_low
        yerr_m3_c_high = ci_m3_c_high - mean_m3_central
        yerr_m3_r_low = mean_m3_raw - ci_m3_r_low
        yerr_m3_r_high = ci_m3_r_high - mean_m3_raw

        yerr_m5_c_low = mean_m5_central - ci_m5_c_low
        yerr_m5_c_high = ci_m5_c_high - mean_m5_central
        yerr_m5_r_low = mean_m5_raw - ci_m5_r_low
        yerr_m5_r_high = ci_m5_r_high - mean_m5_raw

        skew_records.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'radius_um': b_item['radius_um'],
            'n_cargos': n_cargos,
            'rc_over_xi_mean': rc_xi_m,
            'rc_over_xi_sem': rc_xi_s,
            'mean_skew': mean_skew,
            'std_skew': std_skew,
            'sem_skew': sem_skew,
            'bs_sem': bs_sem,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'yerr_low': yerr_low,
            'yerr_high': yerr_high,
            # 2次モーメント
            'mean_m2_central': mean_m2_central,
            'std_m2_central': std_m2_central,
            'sem_m2_central': sem_m2_central,
            'bs_m2_c_sem': bs_m2_c_sem,
            'ci_m2_c_low': ci_m2_c_low,
            'ci_m2_c_high': ci_m2_c_high,
            'yerr_m2_c_low': yerr_m2_c_low,
            'yerr_m2_c_high': yerr_m2_c_high,
            'mean_m2_raw': mean_m2_raw,
            'std_m2_raw': std_m2_raw,
            'sem_m2_raw': sem_m2_raw,
            'bs_m2_r_sem': bs_m2_r_sem,
            'ci_m2_r_low': ci_m2_r_low,
            'ci_m2_r_high': ci_m2_r_high,
            'yerr_m2_r_low': yerr_m2_r_low,
            'yerr_m2_r_high': yerr_m2_r_high,
            # 3次モーメント
            'mean_m3_central': mean_m3_central,
            'std_m3_central': std_m3_central,
            'sem_m3_central': sem_m3_central,
            'bs_m3_c_sem': bs_m3_c_sem,
            'ci_m3_c_low': ci_m3_c_low,
            'ci_m3_c_high': ci_m3_c_high,
            'yerr_m3_c_low': yerr_m3_c_low,
            'yerr_m3_c_high': yerr_m3_c_high,
            'mean_m3_raw': mean_m3_raw,
            'std_m3_raw': std_m3_raw,
            'sem_m3_raw': sem_m3_raw,
            'bs_m3_r_sem': bs_m3_r_sem,
            'ci_m3_r_low': ci_m3_r_low,
            'ci_m3_r_high': ci_m3_r_high,
            'yerr_m3_r_low': yerr_m3_r_low,
            'yerr_m3_r_high': yerr_m3_r_high,
            # 5次モーメント
            'mean_m5_central': mean_m5_central,
            'std_m5_central': std_m5_central,
            'sem_m5_central': sem_m5_central,
            'bs_m5_c_sem': bs_m5_c_sem,
            'ci_m5_c_low': ci_m5_c_low,
            'ci_m5_c_high': ci_m5_c_high,
            'yerr_m5_c_low': yerr_m5_c_low,
            'yerr_m5_c_high': yerr_m5_c_high,
            'mean_m5_raw': mean_m5_raw,
            'std_m5_raw': std_m5_raw,
            'sem_m5_raw': sem_m5_raw,
            'bs_m5_r_sem': bs_m5_r_sem,
            'ci_m5_r_low': ci_m5_r_low,
            'ci_m5_r_high': ci_m5_r_high,
            'yerr_m5_r_low': yerr_m5_r_low,
            'yerr_m5_r_high': yerr_m5_r_high,
        })

    df_cargo_stats = pd.DataFrame(cargo_stat_records)
    df_skew = pd.DataFrame(skew_records)
    if not df_skew.empty:
        df_skew = df_skew.sort_values('rc_over_xi_mean').reset_index(drop=True)

    # グラフ描画: <\gamma_1> vs R_c / \xi
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r'Symmetric ($\gamma_1 = 0$)')
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, zorder=1,
               label=r'Crossover $R_c / \xi = 1$')

    # 各貨物のプロット (半透明散布図で分布・ばらつきを可視化)
    for _, r_sk in df_skew.iterrows():
        bname = r_sk['bead_name']
        sub_c = df_cargo_stats[df_cargo_stats['bead_name'] == bname]
        rc_val = r_sk['rc_over_xi_mean']
        # 貨物ごとの \gamma_{1, i} をプロット (ジッターをわずかに加えて見やすくする)
        jitter = np.linspace(0.95, 1.05, len(sub_c)) if len(sub_c) > 1 else np.array([1.0])
        ax.scatter(rc_val * jitter, sub_c['gamma1_i'], color=r_sk['color'], marker=r_sk['marker'],
                   alpha=0.35, s=28, zorder=2)

    # 貨物間ブートストラップ信頼区間によるエラーバー付き平均 <\gamma_1>
    for _, r_sk in df_skew.iterrows():
        yerr_bar = [[r_sk['yerr_low']], [r_sk['yerr_high']]]
        ax.errorbar(r_sk['rc_over_xi_mean'], r_sk['mean_skew'],
                    xerr=r_sk['rc_over_xi_sem'], yerr=yerr_bar,
                    fmt=r_sk['marker'], color=r_sk['color'], markersize=9, capsize=5,
                    linewidth=2.2, zorder=4)

    ax.plot(df_skew['rc_over_xi_mean'], df_skew['mean_skew'], color='#2b5c8f', linestyle='-',
            linewidth=2.4, zorder=3,
            label=r"Per-Cargo Ensemble Mean $\langle \gamma_1 \rangle$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-2.2, 0.8)

    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Cargo Skewness $\langle \gamma_1 \rangle = \frac{1}{N_{\mathrm{cargo}}} \sum \gamma_{1, i}$ [-]", fontsize=14, labelpad=8)
    ax.set_title(r"Ensemble Cargo Skewness $\langle \gamma_1 \rangle$ vs $R_c / \xi$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "cos_i_skewness_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 10. 3次中心モーメント <\mu_3> = <(\hat{v}\cdot\hat{u} - \mu)^3> vs R_c / \xi
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r'Symmetric ($\mu_3 = 0$)')
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, zorder=1,
               label=r'Crossover $R_c / \xi = 1$')

    # 各貨物のプロット (半透明散布図で分布・ばらつきを可視化)
    for _, r_sk in df_skew.iterrows():
        bname = r_sk['bead_name']
        sub_c = df_cargo_stats[df_cargo_stats['bead_name'] == bname]
        rc_val = r_sk['rc_over_xi_mean']
        jitter = np.linspace(0.95, 1.05, len(sub_c)) if len(sub_c) > 1 else np.array([1.0])
        ax.scatter(rc_val * jitter, sub_c['mu3_central_i'], color=r_sk['color'], marker=r_sk['marker'],
                   alpha=0.35, s=28, zorder=2)

    # 貨物間ブートストラップ信頼区間によるエラーバー付き平均 <\mu_3>
    for _, r_sk in df_skew.iterrows():
        yerr_bar = [[r_sk['yerr_m3_c_low']], [r_sk['yerr_m3_c_high']]]
        ax.errorbar(r_sk['rc_over_xi_mean'], r_sk['mean_m3_central'],
                    xerr=r_sk['rc_over_xi_sem'], yerr=yerr_bar,
                    fmt=r_sk['marker'], color=r_sk['color'], markersize=9, capsize=5,
                    linewidth=2.2, zorder=4)

    ax.plot(df_skew['rc_over_xi_mean'], df_skew['mean_m3_central'], color='#2b5c8f', linestyle='-',
            linewidth=2.4, zorder=3,
            label=r"Per-Cargo Ensemble Mean $\langle \mu_3 \rangle$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-0.45, 0.15)

    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"3rd Central Moment $\langle \mu_3 \rangle = \frac{1}{N_{\mathrm{cargo}}} \sum \mu_{3, i}$ [-]", fontsize=14, labelpad=8)
    ax.set_title(r"Ensemble Cargo 3rd Central Moment $\langle \mu_3 \rangle$ vs $R_c / \xi$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "cos_i_third_central_moment_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 11. 3次原点モーメント <m_3> = <(\hat{v}\cdot\hat{u})^3> vs R_c / \xi
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r'Zero line ($m_3 = 0$)')
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, zorder=1,
               label=r'Crossover $R_c / \xi = 1$')

    # 各貨物のプロット (半透明散布図で分布・ばらつきを可視化)
    for _, r_sk in df_skew.iterrows():
        bname = r_sk['bead_name']
        sub_c = df_cargo_stats[df_cargo_stats['bead_name'] == bname]
        rc_val = r_sk['rc_over_xi_mean']
        jitter = np.linspace(0.95, 1.05, len(sub_c)) if len(sub_c) > 1 else np.array([1.0])
        ax.scatter(rc_val * jitter, sub_c['m3_raw_i'], color=r_sk['color'], marker=r_sk['marker'],
                   alpha=0.35, s=28, zorder=2)

    # 貨物間ブートストラップ信頼区間によるエラーバー付き平均 <m_3>
    for _, r_sk in df_skew.iterrows():
        yerr_bar = [[r_sk['yerr_m3_r_low']], [r_sk['yerr_m3_r_high']]]
        ax.errorbar(r_sk['rc_over_xi_mean'], r_sk['mean_m3_raw'],
                    xerr=r_sk['rc_over_xi_sem'], yerr=yerr_bar,
                    fmt=r_sk['marker'], color=r_sk['color'], markersize=9, capsize=5,
                    linewidth=2.2, zorder=4)

    ax.plot(df_skew['rc_over_xi_mean'], df_skew['mean_m3_raw'], color='#2b5c8f', linestyle='-',
            linewidth=2.4, zorder=3,
            label=r"Per-Cargo Ensemble Mean $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^3 \rangle$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-0.2, 1.0)

    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"3rd Raw Moment $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^3 \rangle$ [-]", fontsize=14, labelpad=8)
    ax.set_title(r"Ensemble Cargo 3rd Moment $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^3 \rangle$ vs $R_c / \xi$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "cos_i_third_moment_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 13. 2次原点モーメント <m_2> = <(\hat{v}\cdot\hat{u})^2> vs R_c / \xi
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axhline(0.5, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r'Isotropic 2D ($\langle\cos^2\rangle = 0.5$)')
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, zorder=1,
               label=r'Crossover $R_c / \xi = 1$')

    for _, r_sk in df_skew.iterrows():
        bname = r_sk['bead_name']
        sub_c = df_cargo_stats[df_cargo_stats['bead_name'] == bname]
        rc_val = r_sk['rc_over_xi_mean']
        jitter = np.linspace(0.95, 1.05, len(sub_c)) if len(sub_c) > 1 else np.array([1.0])
        ax.scatter(rc_val * jitter, sub_c['m2_raw_i'], color=r_sk['color'], marker=r_sk['marker'],
                   alpha=0.35, s=28, zorder=2)

    for _, r_sk in df_skew.iterrows():
        yerr_bar = [[r_sk['yerr_m2_r_low']], [r_sk['yerr_m2_r_high']]]
        ax.errorbar(r_sk['rc_over_xi_mean'], r_sk['mean_m2_raw'],
                    xerr=r_sk['rc_over_xi_sem'], yerr=yerr_bar,
                    fmt=r_sk['marker'], color=r_sk['color'], markersize=9, capsize=5,
                    linewidth=2.2, zorder=4)

    ax.plot(df_skew['rc_over_xi_mean'], df_skew['mean_m2_raw'], color='#2b5c8f', linestyle='-',
            linewidth=2.4, zorder=3,
            label=r"Per-Cargo Ensemble Mean $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^2 \rangle$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"2nd Raw Moment $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^2 \rangle$ [-]", fontsize=14, labelpad=8)
    ax.set_title(r"Ensemble Cargo 2nd Moment $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^2 \rangle$ vs $R_c / \xi$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "cos_i_second_moment_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 15. 5次原点モーメント <m_5> = <(\hat{v}\cdot\hat{u})^5> vs R_c / \xi
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r'Zero line ($m_5 = 0$)')
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.6, zorder=1,
               label=r'Crossover $R_c / \xi = 1$')

    for _, r_sk in df_skew.iterrows():
        bname = r_sk['bead_name']
        sub_c = df_cargo_stats[df_cargo_stats['bead_name'] == bname]
        rc_val = r_sk['rc_over_xi_mean']
        jitter = np.linspace(0.95, 1.05, len(sub_c)) if len(sub_c) > 1 else np.array([1.0])
        ax.scatter(rc_val * jitter, sub_c['m5_raw_i'], color=r_sk['color'], marker=r_sk['marker'],
                   alpha=0.35, s=28, zorder=2)

    for _, r_sk in df_skew.iterrows():
        yerr_bar = [[r_sk['yerr_m5_r_low']], [r_sk['yerr_m5_r_high']]]
        ax.errorbar(r_sk['rc_over_xi_mean'], r_sk['mean_m5_raw'],
                    xerr=r_sk['rc_over_xi_sem'], yerr=yerr_bar,
                    fmt=r_sk['marker'], color=r_sk['color'], markersize=9, capsize=5,
                    linewidth=2.2, zorder=4)

    ax.plot(df_skew['rc_over_xi_mean'], df_skew['mean_m5_raw'], color='#2b5c8f', linestyle='-',
            linewidth=2.4, zorder=3,
            label=r"Per-Cargo Ensemble Mean $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^5 \rangle$ (95% CI)")

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
    ax.set_ylabel(r"5th Raw Moment $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^5 \rangle$ [-]", fontsize=14, labelpad=8)
    ax.set_title(r"Ensemble Cargo 5th Moment $\langle (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^5 \rangle$ vs $R_c / \xi$",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "cos_i_fifth_moment_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig)

    # CSV 保存
    df_window_summary = pd.DataFrame(window_summary_list)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_dir / "cos_decomposition_summary_by_radius.csv", index=False)
        df_window_summary.to_csv(out_dir / "cos_decomposition_summary_by_window.csv", index=False)
        df_skew.to_csv(out_dir / "cos_i_skewness_summary.csv", index=False)
        df_cargo_stats.to_csv(out_dir / "cos_i_cargo_individual_stats.csv", index=False)
    print("Saved decomposition and skewness summary CSVs.")

    return df_summary


def main():
    print("=== Directional Coupling Decomposition Analysis ===")
    setup_matplotlib()

    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    print(f"Loaded {len(xi_dict)} correlation lengths xi.")

    active_set = load_active_particles(CURRENT_DIR)
    print(f"Loaded {len(active_set)} active particles.")

    df_instant, df_window = load_all_data(ROOT_DIR, active_set)
    print(f"\nExtracted {len(df_instant)} instantaneous points, {len(df_window)} window aggregation points.")

    df_summary = analyze_and_plot(df_instant, df_window, xi_dict)

    print("\n=== Directional Coupling Decomposition Summary Table ===")
    cols = ['bead_name', 'radius_um', 'n_particles', 'cos_i_mean', 'cos_bulk_mean', 'Phi_mean', 'pred_mean', 'ratio_obs_to_bulk']
    print(df_summary[cols].to_string(index=False))


if __name__ == '__main__':
    main()
