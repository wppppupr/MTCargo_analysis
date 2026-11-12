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
from scipy.optimize import curve_fit, minimize
from scipy.stats import skew

# ワークスペースおよびNASのパス設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR_CANDIDATES = [
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    CURRENT_DIR,
]
ROOT_DIR = next((p for p in ROOT_DIR_CANDIDATES if p.exists()), CURRENT_DIR)

try:
    from libs.experiment_cutoff import get_cutoff_frame, filter_by_cutoff
except ImportError:
    from experiment_cutoff import get_cutoff_frame, filter_by_cutoff

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

            # カットオフ適用 (退色・フロー暴走領域の除外)
            cutoff = get_cutoff_frame(exp_name)
            if cutoff is not None:
                df_flow = df_flow[df_flow['frame'] <= cutoff]
                tdf = tdf[tdf['frame'] <= cutoff]

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
    #    および 軌道単位ブートストラップ（Cluster Bootstrap, B=1000）による95%信頼区間評価
    # -------------------------------------------------------------
    df_instant['cargo_id'] = df_instant['exp_dir'].astype(str) + '_' + df_instant['particle'].astype(str)

    bins_cos = np.linspace(-1.0, 1.0, 29) # 28ビン (-1から1)
    bin_centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])
    B_BOOT = 1000
    rng_boot = np.random.default_rng(42)

    # 各ビーズごとに代表値ヒストグラムおよびクラスターブートストラップ分布を計算
    # boot_results[bname] = { 'obs_p', 'ci_low_p', 'ci_high_p',
    #                         'obs_peven', 'ci_low_peven', 'ci_high_peven',
    #                         'obs_podd', 'ci_low_podd', 'ci_high_podd',
    #                         'n_cargos', 'n_points', 'mean_val' }
    boot_hist_dict = {}

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue

        cos_vals_all = sub_inst['cos_i'].dropna().values
        if len(cos_vals_all) == 0:
            continue

        # 代表値の作成（全データ点をプールして正規化）
        counts_obs, _ = np.histogram(cos_vals_all, bins=bins_cos, density=True)
        P_obs = counts_obs
        P_obs_neg = counts_obs[::-1]
        P_even_obs = 0.5 * (P_obs + P_obs_neg)
        P_odd_obs = 0.5 * (P_obs - P_obs_neg)

        # 軌道単位のデータ分割
        cargos_data = []
        for _, grp in sub_inst.groupby('cargo_id'):
            c_vals = grp['cos_i'].dropna().values
            if len(c_vals) > 0:
                cargos_data.append(c_vals)

        n_cargos = len(cargos_data)
        boot_P = np.zeros((B_BOOT, len(bin_centers)))
        boot_Peven = np.zeros((B_BOOT, len(bin_centers)))
        boot_Podd = np.zeros((B_BOOT, len(bin_centers)))
        boot_Ueff = np.zeros((B_BOOT, len(bin_centers)))

        # ヤコビアン補正ボルツマン反転による有効ポテンシャル U_eff(c):
        # U_eff(c) = - ln( sqrt(1 - c^2) * P(c) / P(0) )
        # c = 0 に最も近いビンインデックス (c=0でU_eff(0)=0)
        idx_zero = int(np.argmin(np.abs(bin_centers)))
        denom_jacob = np.sqrt(np.clip(1.0 - bin_centers**2, 1e-6, 1.0))

        # 実測代表値の U_eff(c)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_zero_obs = P_obs[idx_zero] if P_obs[idx_zero] > 0 else np.nan
            arg_u_obs = denom_jacob * (P_obs / p_zero_obs)
            U_eff_obs = np.where(arg_u_obs > 0, -np.log(arg_u_obs), np.nan)

        # 偶奇分解:
        # U_odd(c) = (U_eff(c) - U_eff(-c)) / 2 = - \kappa_1 c
        # U_even(c) = (U_eff(c) + U_eff(-c)) / 2 = 2 \kappa_2 c^2
        U_eff_obs_neg = U_eff_obs[::-1]
        U_odd_obs = 0.5 * (U_eff_obs - U_eff_obs_neg)
        U_even_obs = 0.5 * (U_eff_obs + U_eff_obs_neg)

        # -------------------------------------------------------------
        # 方式2: P_odd / P_even 比率法による極性結合 \kappa_1 と ネマティック結合 \kappa_2 の安定推定
        # 1. P_odd(c) / P_even(c) = tanh(\kappa_1 c)  (ヤコビアン・ガウス減衰・規格化定数がすべて相殺)
        # 2. ln[ sqrt(1 - c^2) * P_even(c) / cosh(\kappa_1 c) ] = const - 2 * \kappa_2 * c^2  (\kappa_2 >= 0)
        # -------------------------------------------------------------
        dc_bin = bin_centers[1] - bin_centers[0]
        pos_mask_c = bin_centers > 0
        neg_mask_c = bin_centers < 0

        # P(c > 0) - P(c < 0) の実測値
        p_pos_obs = float(np.sum(P_obs[pos_mask_c]) * dc_bin)
        p_neg_obs = float(np.sum(P_obs[neg_mask_c]) * dc_bin)
        p_diff_obs = p_pos_obs - p_neg_obs

        def estimate_k1_k2(p_e, p_o):
            mask = (bin_centers > 0.05) & (bin_centers <= 0.85) & (p_e > 1e-4)
            if np.sum(mask) < 2:
                return np.nan, np.nan
            c_m = bin_centers[mask]
            ratio_m = np.clip((p_o / p_e)[mask], -0.999, 0.999)

            # 1. \kappa_1 の推定: arctanh(ratio) = \kappa_1 * c
            atanh_vals = np.arctanh(ratio_m)
            k1_est = float(np.sum(atanh_vals * c_m) / np.sum(c_m**2))
            k1_est = max(0.0, k1_est)

            # 2. \kappa_2 の推定:
            y_vals = np.log(np.clip(denom_jacob[mask] * p_e[mask] / np.cosh(k1_est * c_m), 1e-6, None))
            x_sq = c_m**2
            x_dev = x_sq - np.mean(x_sq)
            y_dev = y_vals - np.mean(y_vals)
            slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2) if np.sum(x_dev**2) > 0 else 0.0
            k2_est = max(0.0, float(-0.5 * slope))

            return k1_est, k2_est

        k1_direct, k2_direct = estimate_k1_k2(P_even_obs, P_odd_obs)

        # \sigma_c = 1 / sqrt(4 * \kappa_2) (ただし \kappa_2 <= 1e-4 の場合は \infty)
        if not np.isnan(k2_direct) and k2_direct > 1e-4:
            sig_c_direct = 1.0 / np.sqrt(4.0 * k2_direct)
        else:
            sig_c_direct = 20.0 # 実質的に無限大 (減衰なし)

        if n_cargos >= 2:
            boot_k1_arr = np.zeros(B_BOOT)
            boot_k2_arr = np.zeros(B_BOOT)
            boot_pdiff_arr = np.zeros(B_BOOT)

            for b_idx in range(B_BOOT):
                sel_indices = rng_boot.choice(n_cargos, size=n_cargos, replace=True)
                sample_pts = np.concatenate([cargos_data[i] for i in sel_indices])
                c_boot, _ = np.histogram(sample_pts, bins=bins_cos, density=True)
                c_boot_neg = c_boot[::-1]
                boot_P[b_idx] = c_boot
                pe_b = 0.5 * (c_boot + c_boot_neg)
                po_b = 0.5 * (c_boot - c_boot_neg)
                boot_Peven[b_idx] = pe_b
                boot_Podd[b_idx] = po_b

                # P(c > 0) - P(c < 0)
                boot_pdiff_arr[b_idx] = (np.sum(c_boot[pos_mask_c]) - np.sum(c_boot[neg_mask_c])) * dc_bin

                with np.errstate(divide='ignore', invalid='ignore'):
                    p_zero_b = c_boot[idx_zero] if c_boot[idx_zero] > 0 else np.nan
                    arg_u_b = denom_jacob * (c_boot / p_zero_b)
                    u_b = np.where(arg_u_b > 0, -np.log(arg_u_b), np.nan)
                boot_Ueff[b_idx] = u_b

                k1_b, k2_b = estimate_k1_k2(pe_b, po_b)
                boot_k1_arr[b_idx] = k1_b
                boot_k2_arr[b_idx] = k2_b

            ci_low_p = np.percentile(boot_P, 2.5, axis=0)
            ci_high_p = np.percentile(boot_P, 97.5, axis=0)

            ci_low_peven = np.percentile(boot_Peven, 2.5, axis=0)
            ci_high_peven = np.percentile(boot_Peven, 97.5, axis=0)

            ci_low_podd = np.percentile(boot_Podd, 2.5, axis=0)
            ci_high_podd = np.percentile(boot_Podd, 97.5, axis=0)

            ci_low_ueff = np.nanpercentile(boot_Ueff, 2.5, axis=0)
            ci_high_ueff = np.nanpercentile(boot_Ueff, 97.5, axis=0)

            k1_ci_low = float(np.nanpercentile(boot_k1_arr, 2.5))
            k1_ci_high = float(np.nanpercentile(boot_k1_arr, 97.5))
            k2_ci_low = float(np.nanpercentile(boot_k2_arr, 2.5))
            k2_ci_high = float(np.nanpercentile(boot_k2_arr, 97.5))

            pdiff_ci_low = float(np.nanpercentile(boot_pdiff_arr, 2.5))
            pdiff_ci_high = float(np.nanpercentile(boot_pdiff_arr, 97.5))
        else:
            ci_low_p, ci_high_p = P_obs, P_obs
            ci_low_peven, ci_high_peven = P_even_obs, P_even_obs
            ci_low_podd, ci_high_podd = P_odd_obs, P_odd_obs
            ci_low_ueff, ci_high_ueff = U_eff_obs, U_eff_obs
            k1_ci_low, k1_ci_high = k1_direct, k1_direct
            k2_ci_low, k2_ci_high = k2_direct, k2_direct
            pdiff_ci_low, pdiff_ci_high = p_diff_obs, p_diff_obs

        boot_hist_dict[bname] = {
            'P_obs': P_obs,
            'ci_low_p': ci_low_p,
            'ci_high_p': ci_high_p,
            'P_even_obs': P_even_obs,
            'ci_low_peven': ci_low_peven,
            'ci_high_peven': ci_high_peven,
            'P_odd_obs': P_odd_obs,
            'ci_low_podd': ci_low_podd,
            'ci_high_podd': ci_high_podd,
            'U_eff_obs': U_eff_obs,
            'ci_low_ueff': ci_low_ueff,
            'ci_high_ueff': ci_high_ueff,
            'U_odd_obs': U_odd_obs,
            'U_even_obs': U_even_obs,
            'kappa_1_fit': k1_direct,
            'kappa_2_fit': k2_direct,
            'sigma_c_fit': sig_c_direct,
            'k1_ci_low': k1_ci_low,
            'k1_ci_high': k1_ci_high,
            'k2_ci_low': k2_ci_low,
            'k2_ci_high': k2_ci_high,
            'p_diff_obs': p_diff_obs,
            'pdiff_ci_low': pdiff_ci_low,
            'pdiff_ci_high': pdiff_ci_high,
            'n_cargos': n_cargos,
            'n_points': len(cos_vals_all),
            'mean_val': float(np.mean(cos_vals_all)),
        }

    # (1) 6 パネル版 P(\hat{v}\cdot\hat{u}) ヒストグラム（クラスターブートストラップ 95% 信頼区間帯付き）
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        color = b_item['color']
        lbl = b_item['label']

        if bname not in boot_hist_dict:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        res = boot_hist_dict[bname]
        phi_mean_val = float(df_rc_xi[df_rc_xi['bead_name'] == bname]['Phi_mean'].values[0]) if not df_rc_xi.empty else np.nan
        ci_mean_val = res['mean_val']

        # 実測ヒストグラム（ステップ線＋マーカー）
        ax.plot(bin_centers, res['P_obs'], color=color, linewidth=2.4, marker=b_item['marker'], markersize=5,
                label=rf"Observed ($N = {res['n_points']}$, $N_{{\mathrm{{cargo}}}} = {res['n_cargos']}$)", zorder=3)
        # クラスターブートストラップ 95% 信頼区間帯
        ax.fill_between(bin_centers, res['ci_low_p'], res['ci_high_p'], color=color, alpha=0.25,
                        label=r"95% CI (Cluster Bootstrap)", zorder=2)

        # 代表粒子平均値の縦線
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axvline(ci_mean_val, color=color, linestyle='--', linewidth=1.8,
                   label=rf"Mean $\mathbf{{\hat{{v}}}} \cdot \mathbf{{\hat{{u}}}} = {ci_mean_val:.2f}$", zorder=2)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0.0, 4.0)
        ax.set_title(rf"{lbl}  ($\Phi = {phi_mean_val:.2f}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9.5, framealpha=0.92)

    fig.suptitle(r'Probability Density of Directional Coupling $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ (Cluster Bootstrap 95% CI)',
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_i_probability_density_histograms", OUT_DIRS)
    plt.close(fig)

    # (2) 全粒子統合の 1 パネル版ヒストグラム（代表曲線＋信頼区間帯）
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        color = b_item['color']
        lbl = b_item['label']

        if bname not in boot_hist_dict:
            continue

        res = boot_hist_dict[bname]
        ax.plot(bin_centers, res['P_obs'], marker=b_item['marker'], markersize=6, color=color,
                linewidth=2.2, label=rf"{lbl} ($N={res['n_cargos']}$)", zorder=3 + idx)
        ax.fill_between(bin_centers, res['ci_low_p'], res['ci_high_p'], color=color, alpha=0.15, zorder=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 4.0)
    ax.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Probability Density $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$", fontsize=14, labelpad=8)
    ax.set_title(r"Probability Density of Cargo-Flow Coupling $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ (Cluster Bootstrap 95% CI)",
                 fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "cos_i_probability_density_1panel", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 8b. 偶関数成分 P_even と 奇関数成分 P_odd の分解と可視化
    #     P_even(x) = (P(x) + P(-x)) / 2
    #     P_odd(x)  = (P(x) - P(-x)) / 2  (x = \hat{v} \cdot \hat{u})
    #     信頼区間も同様に軌道単位ブートストラップ試行ごとに算出
    # -------------------------------------------------------------
    # (1) 6パネル版: 各粒子サイズごとに P, P_even, P_odd を比較
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    peven_podd_records = []

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        color = b_item['color']
        lbl = b_item['label']

        if bname not in boot_hist_dict:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        res = boot_hist_dict[bname]

        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)

        # 実測 P, 偶関数成分 P_even, 奇関数成分 P_odd のヒストグラム点・線・信頼区間帯をプロット
        # 1. 全体 P(c)
        ax.plot(bin_centers, res['P_obs'], color='#555555', linestyle=':', linewidth=1.6,
                marker='.', markersize=6.0, label=r"$P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ (Observed)", zorder=4)
        ax.fill_between(bin_centers, res['ci_low_p'], res['ci_high_p'], color='#888888', alpha=0.15, zorder=1)

        # 2. 偶関数成分 P_even(c)
        ax.plot(bin_centers, res['P_even_obs'], color='#1b7837', linestyle='-', linewidth=2.0,
                marker='o', markersize=5.5, label=r"$P_{\mathrm{even}} = \frac{P(x) + P(-x)}{2}$", zorder=5)
        ax.fill_between(bin_centers, res['ci_low_peven'], res['ci_high_peven'], color='#1b7837', alpha=0.20, zorder=2)

        # 3. 奇関数成分 P_odd(c)
        ax.plot(bin_centers, res['P_odd_obs'], color='#d95f02', linestyle='--', linewidth=2.0,
                marker='s', markersize=5.0, label=r"$P_{\mathrm{odd}} = \frac{P(x) - P(-x)}{2}$", zorder=5)
        ax.fill_between(bin_centers, res['ci_low_podd'], res['ci_high_podd'], color='#d95f02', alpha=0.20, zorder=2)

        for bc, px, pe, po, px_l, px_h, pe_l, pe_h, po_l, po_h in zip(
                bin_centers, res['P_obs'], res['P_even_obs'], res['P_odd_obs'],
                res['ci_low_p'], res['ci_high_p'],
                res['ci_low_peven'], res['ci_high_peven'],
                res['ci_low_podd'], res['ci_high_podd']):
            peven_podd_records.append({
                'bead_name': bname,
                'label': lbl,
                'cos_val': bc,
                'P': px,
                'P_ci_low': px_l,
                'P_ci_high': px_h,
                'P_even': pe,
                'P_even_ci_low': pe_l,
                'P_even_ci_high': pe_h,
                'P_odd': po,
                'P_odd_ci_low': po_l,
                'P_odd_ci_high': po_h,
                'kappa_1_fit': res.get('kappa_1_fit', np.nan),
                'sigma_c_fit': res.get('sigma_c_fit', np.nan),
                'N_fit': res.get('N_fit', np.nan),
            })

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-2.0, 3.0)
        ax.set_title(rf"{lbl} ($N_{{\mathrm{{cargo}}}} = {res['n_cargos']}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density / Component", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9.2, framealpha=0.92)

    fig.suptitle(r'Decomposition into Even and Odd Components: $P(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}) = P_{\mathrm{even}} + P_{\mathrm{odd}}$ (95% CI)',
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos_i_peven_podd_histograms_6panels", OUT_DIRS)
    plt.close(fig)

    # (2) 全サイズ比較用の 1 パネル版 (P_even と P_odd を並列して 1 画面で比較)
    fig, (ax_even, ax_odd) = plt.subplots(1, 2, figsize=(16.0, 6.5), sharey=True)

    ax_even.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)
    ax_odd.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
    ax_odd.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        color = b_item['color']
        lbl = b_item['label']

        if bname not in boot_hist_dict:
            continue

        res = boot_hist_dict[bname]

        ax_even.plot(bin_centers, res['P_even_obs'], marker=b_item['marker'], markersize=6,
                     color=color, linewidth=2.2, label=lbl, zorder=3 + idx)
        ax_even.fill_between(bin_centers, res['ci_low_peven'], res['ci_high_peven'],
                             color=color, alpha=0.15, zorder=2)

        ax_odd.plot(bin_centers, res['P_odd_obs'], marker=b_item['marker'], markersize=6,
                    color=color, linewidth=2.2, label=lbl, zorder=3 + idx)
        ax_odd.fill_between(bin_centers, res['ci_low_podd'], res['ci_high_podd'],
                            color=color, alpha=0.15, zorder=2)

    ax_even.set_xlim(-1.05, 1.05)
    ax_even.set_ylim(-1.5, 3.8)
    ax_even.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=14, labelpad=8)
    ax_even.set_ylabel(r"Even Component $P_{\mathrm{even}}(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$", fontsize=14, labelpad=8)
    ax_even.set_title(r"Even Component $P_{\mathrm{even}} = \frac{P(x) + P(-x)}{2}$ (Cluster Bootstrap 95% CI)",
                      fontsize=14, fontweight='bold', pad=10)
    ax_even.grid(True, linestyle=':', alpha=0.7)
    ax_even.legend(loc='upper right', fontsize=10.0, framealpha=0.92)

    ax_odd.set_xlim(-1.05, 1.05)
    ax_odd.set_xlabel(r"Directional Coupling $\mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=14, labelpad=8)
    ax_odd.set_ylabel(r"Odd Component $P_{\mathrm{odd}}(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$", fontsize=14, labelpad=8)
    ax_odd.set_title(r"Odd Component $P_{\mathrm{odd}} = \frac{P(x) - P(-x)}{2}$ (Cluster Bootstrap 95% CI)",
                     fontsize=14, fontweight='bold', pad=10)
    ax_odd.grid(True, linestyle=':', alpha=0.7)
    ax_odd.legend(loc='upper left', fontsize=10.0, framealpha=0.92)

    fig.suptitle(r'Comparison of Even and Odd Components across All Cargo Sizes',
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "cos_i_peven_podd_comparison_1panel", OUT_DIRS)
    plt.close(fig)

    # -------------------------------------------------------------
    # 8c. フィッティングパラメータ \kappa_1 vs R_c / \xi および \sigma_c vs R_c / \xi の個別グラフ
    # -------------------------------------------------------------
    fit_param_records = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        if bname not in boot_hist_dict:
            continue
        res = boot_hist_dict[bname]
        row_rc = df_rc_xi[df_rc_xi['bead_name'] == bname]
        rc_xi_m = float(row_rc['rc_over_xi_mean'].values[0]) if not row_rc.empty else np.nan
        rc_xi_s = float(row_rc['rc_over_xi_sem'].values[0]) if not row_rc.empty else 0.0

        fit_param_records.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'rc_over_xi_mean': rc_xi_m,
            'rc_over_xi_sem': rc_xi_s,
            'kappa_1': res.get('kappa_1_fit', np.nan),
            'kappa_2': res.get('kappa_2_fit', np.nan),
            'sigma_c': res.get('sigma_c_fit', np.nan),
            'k1_ci_low': res.get('k1_ci_low', np.nan),
            'k1_ci_high': res.get('k1_ci_high', np.nan),
            'k2_ci_low': res.get('k2_ci_low', np.nan),
            'k2_ci_high': res.get('k2_ci_high', np.nan),
            'p_diff_obs': res.get('p_diff_obs', np.nan),
            'pdiff_ci_low': res.get('pdiff_ci_low', np.nan),
            'pdiff_ci_high': res.get('pdiff_ci_high', np.nan),
        })

    df_fit_params = pd.DataFrame(fit_param_records)
    if not df_fit_params.empty:
        df_fit_params = df_fit_params.sort_values('rc_over_xi_mean').reset_index(drop=True)

        # ---------------------------------------------------------
        # グラフ 1: \kappa_1 vs R_c / \xi
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1,
                   label=r'Crossover $R_c / \xi = 1$')

        # トレンド線
        ax.plot(df_fit_params['rc_over_xi_mean'], df_fit_params['kappa_1'],
                color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

        # 個別ビーズマーカー
        for _, r_p in df_fit_params.iterrows():
            ax.scatter(r_p['rc_over_xi_mean'], r_p['kappa_1'],
                       color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                       linewidths=1.2, zorder=5, label=r_p['label'])

        ax.set_xscale('log')
        ax.set_xlim(0.04, 3.5)
        ax.set_ylim(-0.05, 1.65)
        ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
        ax.set_ylabel(r"Polar Coupling Parameter $\kappa_1$ [-]", fontsize=14, labelpad=8)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
        ax.legend(loc='upper right', fontsize=11.0, framealpha=0.92)
        #ax.set_title(r"Polar Coupling Parameter $\kappa_1$ vs $R_c / \xi$",
        #             fontsize=15, fontweight='bold', pad=12)

        fig.tight_layout()
        save_figure(fig, "cos_i_fitted_param_kappa1_vs_rc_over_xi", OUT_DIRS)
        plt.close(fig)

        # ---------------------------------------------------------
        # グラフ 2: \sigma_c vs R_c / \xi
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1,
                   label=r'Crossover $R_c / \xi = 1$')

        # 有限値のプロットとトレンド線
        finite_mask = df_fit_params['sigma_c'] < 10.0
        df_finite = df_fit_params[finite_mask]
        if not df_finite.empty:
            ax.plot(df_finite['rc_over_xi_mean'], df_finite['sigma_c'],
                    color='#333333', linestyle='--', linewidth=2.0, alpha=0.7, zorder=2)

        # 軸上限（有限値の可視範囲）
        y_max = 2.6
        ax.set_ylim(0.0, y_max)

        for _, r_p in df_fit_params.iterrows():
            s_val = r_p['sigma_c']
            if s_val >= 10.0:
                # 枠外へ外れる表現: 枠の上端線上に上向き三角マーカーを配置し、上向き矢印と \sigma_c -> \infty を描画
                ax.scatter(r_p['rc_over_xi_mean'], y_max,
                           color=r_p['color'], marker='^', s=120, edgecolors='black',
                           linewidths=1.3, zorder=6, clip_on=False,
                           label=f"{r_p['label']} ($\sigma_c \\to \\infty$)")
                # 上向きの破線または矢印＋テキスト
                ax.annotate(r"$\sigma_c \to \infty$" + f"\n({r_p['label']})",
                            xy=(r_p['rc_over_xi_mean'], y_max),
                            xytext=(0, 14), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10.5, color=r_p['color'],
                            fontweight='bold',
                            arrowprops=dict(arrowstyle="->", color=r_p['color'], lw=1.5))
            else:
                ax.scatter(r_p['rc_over_xi_mean'], s_val,
                           color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                           linewidths=1.2, zorder=5, label=r_p['label'])

        ax.set_xscale('log')
        ax.set_xlim(0.04, 3.5)
        ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
        ax.set_ylabel(r"Gaussian Width Parameter $\sigma_c$ [-]", fontsize=14, labelpad=8)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
        ax.legend(loc='lower left', fontsize=10.0, framealpha=0.92)
        #ax.set_title(r"Gaussian Width Parameter $\sigma_c$ vs $R_c / \xi$",
        #             fontsize=15, fontweight='bold', pad=28)

        fig.tight_layout()
        save_figure(fig, "cos_i_fitted_param_sigmac_vs_rc_over_xi", OUT_DIRS)
        plt.close(fig)

        # ---------------------------------------------------------
        # グラフ 3: 比率 K_1 / K_2 = \kappa_1 / \kappa_2 vs R_c / \xi
        # (ポテンシャルの偶奇分解 U_odd / U_even より直接読み取った極性結合 K_1 vs ネマティック結合 K_2 の優位性比較)
        # ---------------------------------------------------------
        df_fit_params['k1_over_k2'] = df_fit_params['kappa_1'] / df_fit_params['kappa_2']

        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1,
                   label=r'Crossover $R_c / \xi = 1$')

        # 基準線 K_1 / K_2 = 1 (極性とネマティックの強さが拮抗)
        ax.axhline(1.0, color='darkred', linestyle=':', linewidth=1.5, zorder=1,
                   label=r'Equipartition $K_1 / K_2 = 1$')

        # 有限値のトレンド線
        finite_mask_ratio = (df_fit_params['kappa_2'] > 0.05) & (~np.isnan(df_fit_params['k1_over_k2']))
        df_finite_ratio = df_fit_params[finite_mask_ratio]
        if not df_finite_ratio.empty:
            ax.plot(df_finite_ratio['rc_over_xi_mean'], df_finite_ratio['k1_over_k2'],
                    color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

        # 有限値の最大値に基づくプロット上限設定
        y_max_ratio = 7.0
        ax.set_ylim(0.0, y_max_ratio)

        for _, r_p in df_fit_params.iterrows():
            k2_val = r_p['kappa_2']
            ratio_val = r_p['k1_over_k2']
            if k2_val <= 0.05 or np.isnan(ratio_val):
                # \kappa_2 \approx 0 のため K_1 / K_2 -> \infty (完全な極性支配)
                ax.scatter(r_p['rc_over_xi_mean'], y_max_ratio,
                           color=r_p['color'], marker='^', s=120, edgecolors='black',
                           linewidths=1.3, zorder=6, clip_on=False,
                           label=f"{r_p['label']} ($K_1/K_2 \\to \\infty$)")
                ax.annotate(r"$K_1 / K_2 \to \infty$" + f"\n({r_p['label']})",
                            xy=(r_p['rc_over_xi_mean'], y_max_ratio),
                            xytext=(0, 14), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10.5, color=r_p['color'],
                            fontweight='bold',
                            arrowprops=dict(arrowstyle="->", color=r_p['color'], lw=1.5))
            else:
                ax.scatter(r_p['rc_over_xi_mean'], ratio_val,
                           color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                           linewidths=1.2, zorder=5, label=f"{r_p['label']} ({ratio_val:.2f})")

        ax.set_xscale('log')
        ax.set_xlim(0.04, 3.5)
        ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
        ax.set_ylabel(r"Coupling Ratio $K_1 / K_2 = \kappa_1 / \kappa_2$ [-]", fontsize=14, labelpad=8)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)

        # 領域のテキスト注記
        ax.text(0.05, 0.2, r"Nematic Dominant ($K_2 > K_1$)", fontsize=11, color='gray', style='italic')
        ax.text(0.05, 1.2, r"Polar Dominant ($K_1 > K_2$)", fontsize=11, color='gray', style='italic')

        #ax.legend(loc='lower left', fontsize=10.0, framealpha=0.92)

        fig.tight_layout()
        save_figure(fig, "cos_i_fitted_param_k1_over_k2_vs_rc_over_xi", OUT_DIRS)
        plt.close(fig)

        # ---------------------------------------------------------
        # グラフ 4: 異方性（順行-逆行の確率差） P(c > 0) - P(c < 0) vs R_c / \xi
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1,
                   label=r'Crossover $R_c / \xi = 1$')
        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)

        # トレンド線
        ax.plot(df_fit_params['rc_over_xi_mean'], df_fit_params['p_diff_obs'],
                color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

        # クラスターブートストラップ 95% 信頼区間の縦エラーバー & 横軸 SEM エラーバー付きプロット
        for _, r_p in df_fit_params.iterrows():
            x_val = r_p['rc_over_xi_mean']
            x_sem = r_p['rc_over_xi_sem']
            xerr = [x_sem] if (not np.isnan(x_sem) and x_sem > 0) else None

            y_val = r_p['p_diff_obs']
            y_low = r_p['pdiff_ci_low']
            y_high = r_p['pdiff_ci_high']
            yerr = [[y_val - y_low], [y_high - y_val]] if (not np.isnan(y_low) and not np.isnan(y_high)) else None

            # エラーバー（縦・横）
            if (yerr is not None) or (xerr is not None):
                ax.errorbar(x_val, y_val, xerr=xerr, yerr=yerr,
                            fmt='none', ecolor=r_p['color'], elinewidth=1.6, capsize=4.0,
                            capthick=1.2, zorder=4)

            ax.scatter(x_val, y_val,
                       color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                       linewidths=1.2, zorder=5, label=f"{r_p['label']} ({y_val:.2f})")

        ax.set_xscale('log')
        ax.set_xlim(0.04, 3.5)
        ax.set_ylim(-0.05, 0.55)
        ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
        ax.set_ylabel(r"$P(c > 0) - P(c < 0)$ [-]", fontsize=14, labelpad=8)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
        ax.legend(loc='upper right', fontsize=10.5, framealpha=0.92)

        fig.tight_layout()
        save_figure(fig, "cos_i_p_diff_pos_neg_vs_rc_over_xi", OUT_DIRS)
        plt.close(fig)

        # ---------------------------------------------------------
        # 8d. ヤコビアン補正ボルツマン反転による有効配向ポテンシャル \tilde{U}_eff(c) の推定
        #     \tilde{U}_eff(c) = - ln[ \sqrt{1 - c^2} * P(c) / P(0) ]
        #     理論モデル曲線: \tilde{U}_theo(c) = - \kappa_1 * c + c^2 / (2 \sigma_c^2)
        # ---------------------------------------------------------
        # (1) 6パネル版: 各粒子サイズごとの有効ポテンシャル実測値 (95% CI) と理論フィッティング曲線
        fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        c_fine_u = np.linspace(-0.95, 0.95, 400)

        for idx, b_item in enumerate(BEADS_INFO):
            ax = axes_flat[idx]
            bname = b_item['name']
            color = b_item['color']
            lbl = b_item['label']

            if bname not in boot_hist_dict:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue

            res = boot_hist_dict[bname]
            u_obs = res['U_eff_obs']
            ci_low_u = res['ci_low_ueff']
            ci_high_u = res['ci_high_ueff']

            ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
            ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)

            # 実測プロット (ビン点 + 95% CI帯)
            valid_mask = ~np.isnan(u_obs)
            ax.plot(bin_centers[valid_mask], u_obs[valid_mask], color=color, linewidth=2.2,
                    marker=b_item['marker'], markersize=6.0,
                    label=rf"Inverted $\tilde{{U}}_{{\mathrm{{eff}}}}(c)$ ($N={res['n_cargos']}$)", zorder=4)
            ax.fill_between(bin_centers[valid_mask], ci_low_u[valid_mask], ci_high_u[valid_mask],
                            color=color, alpha=0.22, label=r"95% CI (Cluster Bootstrap)", zorder=2)

            # 理論ポテンシャル（直接読み取りパラメータに基づく）:
            # \tilde{U}_theo(c) = - \kappa_1 c + 2 \kappa_2 c^2
            k1_fit = res.get('kappa_1_fit', np.nan)
            k2_fit = res.get('kappa_2_fit', np.nan)
            sig_fit = res.get('sigma_c_fit', np.nan)
            if not np.isnan(k1_fit) and not np.isnan(k2_fit):
                u_theo = - k1_fit * c_fine_u + 2.0 * k2_fit * (c_fine_u**2)
                ax.plot(c_fine_u, u_theo, color='#222222', linestyle='--', linewidth=2.0, alpha=0.9,
                        label=rf"Direct read: $-\kappa_1 c + 2\kappa_2 c^2$ ($\kappa_1={k1_fit:.2f}, \kappa_2={k2_fit:.2f}$)", zorder=5)

            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-2.5, 3.5)
            ax.set_title(rf"{lbl}", fontsize=14, fontweight='bold')
            ax.set_xlabel(r"Directional Coupling $c = \mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=13)
            if idx % 3 == 0:
                ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$ [$\zeta_r D_r^{\mathrm{eff}}$]", fontsize=13)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper center', fontsize=9.2, framealpha=0.92)

        fig.suptitle(r'Effective Orientation Potential $\tilde{U}_{\mathrm{eff}}(c) = -\ln\left[\sqrt{1-c^2} \frac{P(c)}{P(0)}\right]$ (Jacobian-Corrected Boltzmann Inversion)',
                     fontsize=16, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_figure(fig, "cos_i_effective_potential_6panels", OUT_DIRS)
        plt.close(fig)

        # (2) 全粒子統合 1パネル版: 有効ポテンシャルの比較
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
        ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)

        for idx, b_item in enumerate(BEADS_INFO):
            bname = b_item['name']
            if bname not in boot_hist_dict:
                continue
            res = boot_hist_dict[bname]
            u_obs = res['U_eff_obs']
            ci_low_u = res['ci_low_ueff']
            ci_high_u = res['ci_high_ueff']

            valid_mask = ~np.isnan(u_obs)
            ax.plot(bin_centers[valid_mask], u_obs[valid_mask], color=b_item['color'], linewidth=2.2,
                    marker=b_item['marker'], markersize=5.5, label=rf"{b_item['label']}", zorder=3 + idx)
            ax.fill_between(bin_centers[valid_mask], ci_low_u[valid_mask], ci_high_u[valid_mask],
                            color=b_item['color'], alpha=0.12, zorder=2)

        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-2.5, 3.5)
        ax.set_xlabel(r"Directional Coupling $c = \mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=14, labelpad=8)
        ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$ [$\zeta_r D_r^{\mathrm{eff}}$]", fontsize=14, labelpad=8)
        ax.set_title(r"Effective Orientation Potential $\tilde{U}_{\mathrm{eff}}(c)$ across Cargo Sizes",
                     fontsize=15, fontweight='bold', pad=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper center', fontsize=10.5, framealpha=0.92)
        fig.tight_layout()
        save_figure(fig, "cos_i_effective_potential_1panel", OUT_DIRS)
        plt.close(fig)

        # ---------------------------------------------------------
        # 8e. ポテンシャル形状 \tilde{U}(c) = - A * arctanh(c) + 2 * \kappa_2 * c^2 によるフィッティンググラフ（別図）
        #     （順行方向への吸引とネマティック調和閉じ込めの組み合わせ）
        # ---------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        c_fine_atanh = np.linspace(-0.95, 0.95, 400)

        for idx, b_item in enumerate(BEADS_INFO):
            ax = axes_flat[idx]
            bname = b_item['name']
            color = b_item['color']
            lbl = b_item['label']

            if bname not in boot_hist_dict:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue

            res = boot_hist_dict[bname]
            u_obs = res['U_eff_obs']
            ci_low_u = res['ci_low_ueff']
            ci_high_u = res['ci_high_ueff']

            ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.7)
            ax.axvline(0.0, color='#888888', linestyle=':', linewidth=1.2)

            # 実測ポテンシャル（ビン点 + 95% CI帯）
            valid_mask = ~np.isnan(u_obs)
            ax.plot(bin_centers[valid_mask], u_obs[valid_mask], color=color, linewidth=2.2,
                    marker=b_item['marker'], markersize=6.0,
                    label=rf"Inverted $\tilde{{U}}_{{\mathrm{{eff}}}}(c)$ ($N={res['n_cargos']}$)", zorder=4)
            ax.fill_between(bin_centers[valid_mask], ci_low_u[valid_mask], ci_high_u[valid_mask],
                            color=color, alpha=0.22, label=r"95% CI (Cluster Bootstrap)", zorder=2)

            # フィッティング: \tilde{U}(c) = - A * arctanh(c) + 2 * \kappa_2 * c^2
            # 偶奇成分の分離推定（比率法）により、ノイズによる人工的な極小・極大（偽の山谷）の発生を防止
            # 1. arctanh(P_odd / P_even) = A * arctanh(c) から極性強度 A を決定
            # 2. ln[ sqrt(1 - c^2) * P_even / cosh(A * arctanh(c)) ] = const - 2 * \kappa_2 * c^2 から \kappa_2 を決定
            pe_obs = res['P_even_obs']
            po_obs = res['P_odd_obs']
            mask_ratio = (bin_centers > 0.05) & (bin_centers <= 0.85) & (pe_obs > 1e-4)
            if np.sum(mask_ratio) >= 2:
                c_m = bin_centers[mask_ratio]
                ratio_m = np.clip((po_obs / pe_obs)[mask_ratio], -0.999, 0.999)
                atanh_ratio = np.arctanh(ratio_m)
                atanh_c = np.arctanh(c_m)
                A_fit = float(np.sum(atanh_ratio * atanh_c) / np.sum(atanh_c**2))
                A_fit = max(0.0, A_fit)

                y_vals = np.log(np.clip(denom_jacob[mask_ratio] * pe_obs[mask_ratio] / np.cosh(A_fit * atanh_c), 1e-6, None))
                x_sq = c_m**2
                x_dev = x_sq - np.mean(x_sq)
                y_dev = y_vals - np.mean(y_vals)
                slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2) if np.sum(x_dev**2) > 0 else 0.0
                k2_atanh_fit = max(0.0, float(-0.5 * slope))

                u_theo_atanh = - A_fit * np.arctanh(c_fine_atanh) + 2.0 * k2_atanh_fit * (c_fine_atanh**2)
                ax.plot(c_fine_atanh, u_theo_atanh, color='#222222', linestyle='--', linewidth=2.0, alpha=0.9,
                        label=rf"Fit: $-A\operatorname{{arctanh}}(c) + 2\kappa_2 c^2$" + f"\n($A={A_fit:.2f}, \kappa_2={k2_atanh_fit:.2f}$)", zorder=5)

            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-2.5, 3.5)
            ax.set_title(rf"{lbl}", fontsize=14, fontweight='bold')
            ax.set_xlabel(r"Directional Coupling $c = \mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$", fontsize=13)
            if idx % 3 == 0:
                ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$ [$\zeta_r D_r^{\mathrm{eff}}$]", fontsize=13)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper center', fontsize=9.2, framealpha=0.92)

        fig.suptitle(r'Effective Potential with $\operatorname{arctanh}(c)$ Fitting: $\tilde{U}(c) \approx - A \operatorname{arctanh}(c) + 2 \kappa_2 c^2$',
                     fontsize=16, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_figure(fig, "cos_i_effective_potential_arctanh_fit_6panels", OUT_DIRS)
        plt.close(fig)

        # ---------------------------------------------------------
        # 8f. 比率 A / \kappa_2 vs R_c / \xi のグラフ
        #     - beads20um (20 \mu m) は A -> 0, \kappa_2 -> 0 の極限として 1.0 に収束したとしてプロット
        #     - 20 \mu m は参考値として色抜き（白抜きマーカー、facecolor='none' または 'white'）
        # ---------------------------------------------------------
        atanh_ratio_records = []
        for b_item in BEADS_INFO:
            bname = b_item['name']
            if bname not in boot_hist_dict:
                continue
            res = boot_hist_dict[bname]
            u_obs = res['U_eff_obs']
            row_rc = df_rc_xi[df_rc_xi['bead_name'] == bname]
            rc_xi_m = float(row_rc['rc_over_xi_mean'].values[0]) if not row_rc.empty else np.nan
            rc_xi_s = float(row_rc['rc_over_xi_sem'].values[0]) if not row_rc.empty else 0.0

            pe_obs = res['P_even_obs']
            po_obs = res['P_odd_obs']
            mask_ratio = (bin_centers > 0.05) & (bin_centers <= 0.85) & (pe_obs > 1e-4)

            A_val, k2_val = np.nan, np.nan
            if np.sum(mask_ratio) >= 2:
                c_m = bin_centers[mask_ratio]
                ratio_m = np.clip((po_obs / pe_obs)[mask_ratio], -0.999, 0.999)
                atanh_ratio = np.arctanh(ratio_m)
                atanh_c = np.arctanh(c_m)
                A_val = float(np.sum(atanh_ratio * atanh_c) / np.sum(atanh_c**2))
                A_val = max(0.0, A_val)

                y_vals = np.log(np.clip(denom_jacob[mask_ratio] * pe_obs[mask_ratio] / np.cosh(A_val * atanh_c), 1e-6, None))
                x_sq = c_m**2
                x_dev = x_sq - np.mean(x_sq)
                y_dev = y_vals - np.mean(y_vals)
                slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2) if np.sum(x_dev**2) > 0 else 0.0
                k2_val = max(0.0, float(-0.5 * slope))

            atanh_ratio_records.append({
                'bead_name': bname,
                'label': b_item['label'],
                'color': b_item['color'],
                'marker': b_item['marker'],
                'rc_over_xi_mean': rc_xi_m,
                'rc_over_xi_sem': rc_xi_s,
                'A': A_val,
                'kappa_2': k2_val,
            })

        df_atanh_ratios = pd.DataFrame(atanh_ratio_records)
        if not df_atanh_ratios.empty:
            df_atanh_ratios = df_atanh_ratios.sort_values('rc_over_xi_mean').reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(8.0, 6.2))
            ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1,
                       label=r'Crossover $R_c / \xi = 1$')
            ax.axhline(1.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1,
                       label=r'Equipartition $A / \kappa_2 = 1$')

            # サドルノード分岐線: \Lambda = A / (4 * \kappa_2) = 2 / (3 * \sqrt{3}) \approx 0.385
            # すなわち A / \kappa_2 = 4 * \Lambda_c = 8 / (3 * \sqrt{3}) \approx 1.540
            lam_crit = 2.0 / (3.0 * np.sqrt(3.0))
            ratio_sn = 4.0 * lam_crit
            ax.axhline(ratio_sn, color='darkred', linestyle='-.', linewidth=1.6, zorder=1,
                       label=rf'Saddle-Node Bifurcation $A / \kappa_2 \approx {ratio_sn:.2f}$ ($\Lambda = 0.385$)')

            y_max_a_ratio = 5.0
            ax.set_ylim(0.0, y_max_a_ratio)

            # トレンド線（有限値の通常データ点 beads1um, beads3um, beads7um, beads5um）
            trend_mask = (df_atanh_ratios['bead_name'] != 'beads06um') & (df_atanh_ratios['bead_name'] != 'beads20um')
            df_trend = df_atanh_ratios[trend_mask].copy()
            df_trend['ratio'] = df_trend['A'] / df_trend['kappa_2']
            ax.plot(df_trend['rc_over_xi_mean'], df_trend['ratio'],
                    color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

            for _, r_p in df_atanh_ratios.iterrows():
                bname = r_p['bead_name']
                x_val = r_p['rc_over_xi_mean']
                x_sem = r_p['rc_over_xi_sem']
                xerr = [x_sem] if (not np.isnan(x_sem) and x_sem > 0) else None

                if bname == 'beads06um':
                    # \kappa_2 \to 0 のため A / \kappa_2 \to \infty
                    ax.scatter(x_val, y_max_a_ratio,
                               color=r_p['color'], marker='^', s=120, edgecolors='black',
                               linewidths=1.3, zorder=6, clip_on=False,
                               label=f"{r_p['label']} ($A/\\kappa_2 \\to \\infty$)")
                    ax.annotate(r"$A / \kappa_2 \to \infty$" + f"\n({r_p['label']})",
                                xy=(x_val, y_max_a_ratio),
                                xytext=(0, 14), textcoords="offset points",
                                ha='center', va='bottom', fontsize=10.5, color=r_p['color'],
                                fontweight='bold',
                                arrowprops=dict(arrowstyle="->", color=r_p['color'], lw=1.5))
                    if xerr is not None:
                        ax.errorbar(x_val, y_max_a_ratio, xerr=xerr, fmt='none',
                                    ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)

                elif bname == 'beads20um':
                    # 20 µm: A=0, \kappa_2=0, \Lambda=0 の極限、参考値として色抜き（白抜きマーカー）
                    y_val = 0.0
                    ax.scatter(x_val, y_val,
                               facecolors='white', edgecolors=r_p['color'], marker=r_p['marker'],
                               s=110, linewidths=2.0, linestyle='--', zorder=6,
                               label=f"{r_p['label']} (limit $\\to 0$, ref)")
                    ax.annotate(r"$A, \kappa_2 \to 0$ ($\to 0$, ref)",
                                xy=(x_val, y_val), xytext=(0, 12), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9.5, color='#555555',
                                fontweight='bold')
                    if xerr is not None:
                        ax.errorbar(x_val, y_val, xerr=xerr, fmt='none',
                                    ecolor='#888888', elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)

                else:
                    # beads1um, beads3um, beads7um, beads5um
                    y_val = r_p['A'] / r_p['kappa_2']
                    if xerr is not None:
                        ax.errorbar(x_val, y_val, xerr=xerr, fmt='none',
                                    ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)
                    ax.scatter(x_val, y_val,
                               color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                               linewidths=1.2, zorder=5, label=f"{r_p['label']} ({y_val:.2f})")

            ax.set_xscale('log')
            ax.set_xlim(0.04, 3.5)
            ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
            ax.set_ylabel(r"Potential Ratio $A / \kappa_2$ [-]", fontsize=14, labelpad=8)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)

            # 領域の注記
            ax.text(0.05, 0.25, r"Well Stabilized ($\Lambda < 0.385$)", fontsize=10.5, color='darkred', style='italic')
            ax.text(0.05, 1.65, r"No Minimum ($\Lambda > 0.385$)", fontsize=10.5, color='darkblue', style='italic')

            ax.legend(loc='lower left', fontsize=9.5, framealpha=0.92)

            fig.tight_layout()
            save_figure(fig, "cos_i_fitted_param_atanh_A_over_k2_vs_rc_over_xi", OUT_DIRS)
            plt.close(fig)

            # ---------------------------------------------------------
            # 8g. サドルノード分岐パラメータ \Lambda = A / (4 * \kappa_2) vs R_c / \xi の直接プロット
            # ---------------------------------------------------------
            fig_lam, ax_lam = plt.subplots(figsize=(8.0, 6.2))
            ax_lam.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1,
                           label=r'Crossover $R_c / \xi = 1$')
            ax_lam.axhline(lam_crit, color='darkred', linestyle='-.', linewidth=1.8, zorder=1,
                           label=rf'Saddle-Node Critical $\Lambda_c \approx {lam_crit:.3f}$')

            y_max_lam = 1.5
            ax_lam.set_ylim(0.0, y_max_lam)

            # トレンド線
            df_trend_lam = df_atanh_ratios[trend_mask].copy()
            df_trend_lam['lambda'] = df_trend_lam['A'] / (4.0 * df_trend_lam['kappa_2'])
            ax_lam.plot(df_trend_lam['rc_over_xi_mean'], df_trend_lam['lambda'],
                        color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

            for _, r_p in df_atanh_ratios.iterrows():
                bname = r_p['bead_name']
                x_val = r_p['rc_over_xi_mean']
                x_sem = r_p['rc_over_xi_sem']
                xerr = [x_sem] if (not np.isnan(x_sem) and x_sem > 0) else None

                if bname == 'beads06um':
                    ax_lam.scatter(x_val, y_max_lam,
                                   color=r_p['color'], marker='^', s=120, edgecolors='black',
                                   linewidths=1.3, zorder=6, clip_on=False,
                                   label=f"{r_p['label']} ($\\Lambda \\to \\infty$)")
                    ax_lam.annotate(r"$\Lambda \to \infty$" + f"\n({r_p['label']})",
                                    xy=(x_val, y_max_lam),
                                    xytext=(0, 14), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=10.5, color=r_p['color'],
                                    fontweight='bold',
                                    arrowprops=dict(arrowstyle="->", color=r_p['color'], lw=1.5))
                    if xerr is not None:
                        ax_lam.errorbar(x_val, y_max_lam, xerr=xerr, fmt='none',
                                        ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)

                elif bname == 'beads20um':
                    y_val = 0.0 # Lambda -> 0 (ref)
                    ax_lam.scatter(x_val, y_val,
                                   facecolors='white', edgecolors=r_p['color'], marker=r_p['marker'],
                                   s=110, linewidths=2.0, linestyle='--', zorder=6,
                                   label=f"{r_p['label']} (limit $\\to 0$, ref)")
                    ax_lam.annotate(r"$\Lambda \to 0$ (ref)",
                                    xy=(x_val, y_val), xytext=(0, 12), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9.5, color='#555555',
                                    fontweight='bold')
                    if xerr is not None:
                        ax_lam.errorbar(x_val, y_val, xerr=xerr, fmt='none',
                                        ecolor='#888888', elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)

                else:
                    y_val = r_p['A'] / (4.0 * r_p['kappa_2'])
                    if xerr is not None:
                        ax_lam.errorbar(x_val, y_val, xerr=xerr, fmt='none',
                                        ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)
                    ax_lam.scatter(x_val, y_val,
                                   color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                                   linewidths=1.2, zorder=5, label=f"{r_p['label']} ({y_val:.2f})")

            ax_lam.set_xscale('log')
            ax_lam.set_xlim(0.04, 3.5)
            ax_lam.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=14, labelpad=8)
            ax_lam.set_ylabel(r"Bifurcation Parameter $\Lambda = \frac{A}{4\kappa_2}$ [-]", fontsize=14, labelpad=8)
            ax_lam.tick_params(axis='both', labelsize=12)
            ax_lam.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)

            ax_lam.text(0.05, 0.10, r"Orthogonal Well Stabilized ($\Lambda < 0.385$)", fontsize=10.5, color='darkred', style='italic')
            ax_lam.text(0.05, 0.45, r"No Orthogonal Minimum ($\Lambda > 0.385$)", fontsize=10.5, color='darkblue', style='italic')

            ax_lam.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

            fig_lam.tight_layout()
            save_figure(fig_lam, "cos_i_fitted_param_bifurcation_lambda_vs_rc_over_xi", OUT_DIRS)
            plt.close(fig_lam)

            # ---------------------------------------------------------
            # 8h. 過減衰配向ダイナミクスの線形安定性解析（3 パネル統合図）
            #     Panel (A): サドルノード分岐図 (Bifurcation Diagram: c* vs \Lambda)
            #     Panel (B): 位相フロー場 (Phase Flow: F(c) = \Lambda/(1-c^2) - c)
            #     Panel (C): 線形緩和率 / 臨界減速 (Linear Relaxation Rate: \lambda/(4\kappa_2) vs \Lambda)
            # ---------------------------------------------------------
            fig_bif, axes_bif = plt.subplots(1, 3, figsize=(18.0, 5.5))
            ax_a, ax_b, ax_c = axes_bif

            # --- 理論曲線の計算 ---
            lam_c = 2.0 / (3.0 * np.sqrt(3.0)) # \approx 0.3849
            lam_arr = np.linspace(0.0001, lam_c, 400)
            arg_sin = np.clip(3.0 * np.sqrt(3.0) / 2.0 * lam_arr, -1.0, 1.0)

            # 安定枝 (c_min: 0 <= c* < 1/sqrt(3))
            psi_min = (1.0 / 3.0) * np.arcsin(arg_sin)
            c_star_stable = (2.0 / np.sqrt(3.0)) * np.sin(psi_min)

            # 不安定枝 (c_max: 1/sqrt(3) < c* < 1)
            psi_max = (np.pi / 3.0) - (1.0 / 3.0) * np.arcsin(arg_sin)
            c_star_unstable = (2.0 / np.sqrt(3.0)) * np.sin(psi_max)

            # 線形化固有値 \lambda / (4 \kappa_2) = - (1 - 3 c*^2) / (1 - c*^2)
            eig_stable = - (1.0 - 3.0 * c_star_stable**2) / (1.0 - c_star_stable**2)
            eig_unstable = - (1.0 - 3.0 * c_star_unstable**2) / (1.0 - c_star_unstable**2)

            # ==========================================
            # Panel (A): サドルノード分岐図
            # ==========================================
            ax_a.plot(lam_arr, c_star_stable, color='#1b7837', linestyle='-', linewidth=2.5,
                      label=r'Stable Node Branch $c_{\min}$ ($\lambda < 0$)', zorder=3)
            ax_a.plot(lam_arr, c_star_unstable, color='#d95f02', linestyle='--', linewidth=2.2,
                      label=r'Unstable Saddle Branch $c_{\max}$ ($\lambda > 0$)', zorder=3)

            # サドルノード臨界点
            c_crit = 1.0 / np.sqrt(3.0)
            ax_a.scatter(lam_c, c_crit, color='crimson', marker='*', s=180, edgecolors='black',
                         linewidths=1.2, zorder=6, label=rf'SN Bifurcation Point ($\Lambda_c \approx {lam_c:.3f}$)')

            # 領域の背景色分け（弾道輸送相 vs ケージング相）
            ax_a.axvspan(0.0, lam_c, color='#1b7837', alpha=0.07, zorder=1)
            ax_a.axvspan(lam_c, 1.25, color='#7570b3', alpha=0.08, zorder=1)
            ax_a.axvline(lam_c, color='crimson', linestyle=':', linewidth=1.5, zorder=2)

            ax_a.text(0.12, 0.90, "Diffusive Phase\n(Stable Well)", fontsize=11, color='#1b7837',
                      fontweight='bold', ha='center', va='center')
            ax_a.text(0.75, 0.90, "Ballistic Phase\n(No Fixed Points)", fontsize=11, color='#7570b3',
                      fontweight='bold', ha='center', va='center')

            # 実験データのプロット
            for _, r_p in df_atanh_ratios.iterrows():
                bname = r_p['bead_name']
                a_val = r_p['A']
                k2_val = r_p['kappa_2']
                lbl = r_p['label']
                col = r_p['color']
                mrk = r_p['marker']

                if k2_val > 0.001:
                    lam_exp = a_val / (4.0 * k2_val)
                    if lam_exp <= lam_c:
                        # 安定枝上の理論固定点に対応
                        arg_e = np.clip(3.0 * np.sqrt(3.0) / 2.0 * lam_exp, -1.0, 1.0)
                        c_e = float((2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_e)))
                        if bname == 'beads20um':
                            ax_a.scatter(0.0, 0.0, facecolors='white', edgecolors=col, marker=mrk,
                                         s=110, linewidths=2.0, linestyle='--', zorder=5, label=f"{lbl} ($\Lambda=0$, ref)")
                        else:
                            ax_a.scatter(lam_exp, c_e, color=col, marker=mrk, s=100, edgecolors='black',
                                         linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda={lam_exp:.2f}$)")
                    else:
                        # 弾道輸送領域 (\Lambda > \Lambda_c)
                        # 右端枠外に矢印プロット
                        ax_a.scatter(min(lam_exp, 1.15), 0.95, color=col, marker='>', s=110, edgecolors='black',
                                     linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda={lam_exp:.2f} > \Lambda_c$)")
                else:
                    # \Lambda -> \infty
                    ax_a.scatter(1.15, 0.95, color=col, marker='>', s=110, edgecolors='black',
                                 linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda \\to \\infty$)")

            ax_a.set_xlim(-0.02, 1.25)
            ax_a.set_ylim(-0.05, 1.05)
            ax_a.set_xlabel(r"Control Parameter $\Lambda = \frac{\alpha}{4\kappa_2}$ [-]", fontsize=13, labelpad=8)
            ax_a.set_ylabel(r"Fixed Point $c^* = (\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})^*$ [-]", fontsize=13, labelpad=8)
            ax_a.set_title(r"(A) Saddle-Node Bifurcation Diagram", fontsize=14, fontweight='bold', pad=10)
            ax_a.grid(True, linestyle=':', alpha=0.7)
            ax_a.legend(loc='lower right', fontsize=8.8, framealpha=0.92)

            # ==========================================
            # Panel (B): 位相フロー場とベクトル場
            # ==========================================
            c_flow = np.linspace(-0.95, 0.95, 500)
            denom_flow = 1.0 - c_flow**2
            ax_b.axhline(0.0, color='#666666', linestyle='-', linewidth=1.2, zorder=2)
            ax_b.axvline(0.0, color='#aaaaaa', linestyle=':', linewidth=1.0, zorder=1)

            # 代表的な \Lambda のフロー曲線
            flow_configs = [
                (0.0, '#2b83ba', '-', r'$\Lambda = 0.0$ (Symmetric)'),
                (0.2, '#1b7837', '-', r'$\Lambda = 0.2 < \Lambda_c$ (Diffusive)'),
                (lam_c, '#fdae61', '-.', rf'$\Lambda = \Lambda_c \approx {lam_c:.3f}$ (Critical)'),
                (0.6, '#7570b3', '--', r'$\Lambda = 0.6 > \Lambda_c$ (Ballistic)'),
            ]

            for l_val, c_color, l_style, l_label in flow_configs:
                f_val = (l_val / denom_flow) - c_flow
                ax_b.plot(c_flow, f_val, color=c_color, linestyle=l_style, linewidth=2.0, label=l_label, zorder=3)

                # 固定点のマーカー
                if l_val <= lam_c:
                    arg_l = np.clip(3.0 * np.sqrt(3.0) / 2.0 * l_val, -1.0, 1.0)
                    cs_stab = (2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_l))
                    ax_b.scatter(cs_stab, 0.0, color=c_color, marker='o', s=60, edgecolors='black', zorder=5)

                    if l_val > 0.001 and l_val < lam_c:
                        cs_unstab = (2.0 / np.sqrt(3.0)) * np.sin((np.pi / 3.0) - (1.0 / 3.0) * np.arcsin(arg_l))
                        ax_b.scatter(cs_unstab, 0.0, facecolors='none', edgecolors=c_color, marker='o', s=60, linewidths=1.5, zorder=5)

            # フロー方向の矢印 (\Lambda = 0.2 のケージング領域を代表としてゼロ線 y=0 付近に配置)
            # 安定固定点 c_s \approx 0.21, 不安定固定点 c_u \approx 0.88
            # 区間 1 (c < 0.21): F > 0 -> 右向き (->)
            # 区間 2 (0.21 < c < 0.88): F < 0 -> 左向き (<-) (安定ノードへ引き戻される復元フロー)
            # 区間 3 (c > 0.88): F > 0 -> 右向き (->) (境界 c=1 への脱出フロー)
            y_arr = -0.1 # ゼロ線上にフローの矢印を描画
            # 区間 1: c < 0.21 (右向き)
            ax_b.annotate('', xy=(-0.35, y_arr), xytext=(-0.55, y_arr),
                          arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
            ax_b.annotate('', xy=(0.05, y_arr), xytext=(-0.15, y_arr),
                          arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
            # 区間 2: 0.21 < c < 0.88 (左向き)
            ax_b.annotate('', xy=(0.35, y_arr), xytext=(0.55, y_arr),
                          arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
            ax_b.annotate('', xy=(0.65, y_arr), xytext=(0.80, y_arr),
                          arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
            # 区間 3: c > 0.88 (右向き)
            ax_b.annotate('', xy=(0.94, y_arr), xytext=(0.90, y_arr),
                          arrowprops=dict(arrowstyle="->", color='#d95f02', lw=2.0))

            ax_b.set_xlim(-0.95, 0.95)
            ax_b.set_ylim(-1.5, 2.5)
            ax_b.set_xlabel(r"Orientation State $c = \mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ [-]", fontsize=13, labelpad=8)
            ax_b.set_ylabel(r"Effective Phase Velocity $F(c; \Lambda) = \frac{1}{4\kappa_2D_r^\mathrm{eff}} \dot{c}$ [-]", fontsize=13, labelpad=8)
            ax_b.set_title(r"(B) Phase Flow Field $F(c; \Lambda)$", fontsize=14, fontweight='bold', pad=10)
            ax_b.grid(True, linestyle=':', alpha=0.7)
            ax_b.legend(loc='upper left', fontsize=9.2, framealpha=0.92)

            # ==========================================
            # Panel (C): 線形緩和率と臨界減速
            # ==========================================
            ax_c.plot(lam_arr, eig_stable, color='#1b7837', linestyle='-', linewidth=2.5,
                      label=r'Stable Node $\lambda_{\min} / (4\kappa_2D_r^\mathrm{eff}) \leq 0$', zorder=3)
            ax_c.axhline(0.0, color='gray', linestyle=':', linewidth=1.2, zorder=2)
            ax_c.axvline(lam_c, color='crimson', linestyle=':', linewidth=1.5, zorder=2)

            # 臨界減速の注記
            ax_c.scatter(lam_c, 0.0, color='crimson', marker='*', s=180, edgecolors='black',
                         linewidths=1.2, zorder=6, label=rf'Critical Slowing Down ($\lambda \to 0$)')
            ax_c.annotate(r"Critical Slowing Down" + "\n" + r"($\lambda \to 0$ as $\Lambda \to \Lambda_c$)",
                          xy=(lam_c, 0.0), xytext=(lam_c - 0.15, -0.35),
                          ha='center', va='top', fontsize=9.5, color='crimson', fontweight='bold',
                          arrowprops=dict(arrowstyle="->", color='crimson', lw=1.5))

            # 実験データの固有値プロット
            for _, r_p in df_atanh_ratios.iterrows():
                bname = r_p['bead_name']
                a_val = r_p['A']
                k2_val = r_p['kappa_2']
                lbl = r_p['label']
                col = r_p['color']
                mrk = r_p['marker']

                if k2_val > 0.001:
                    lam_exp = a_val / (4.0 * k2_val)
                    if lam_exp <= lam_c:
                        arg_e = np.clip(3.0 * np.sqrt(3.0) / 2.0 * lam_exp, -1.0, 1.0)
                        c_e = float((2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_e)))
                        eig_e = - (1.0 - 3.0 * c_e**2) / (1.0 - c_e**2)
                        if bname == 'beads20um':
                            ax_c.scatter(0.0, -1.0, facecolors='white',
                                         edgecolors=col, marker=mrk, s=110, linewidths=2.0, linestyle='--',
                                         zorder=5, label=f"{lbl} ($\Lambda=0$, ref)")
                        else:
                            ax_c.scatter(lam_exp, eig_e, color=col, marker=mrk, s=100, edgecolors='black',
                                         linewidths=1.2, zorder=5, label=f"{lbl} ($\lambda={eig_e:.2f}$)")

            ax_c.set_xlim(-0.02, 0.45)
            ax_c.set_ylim(-1.05, 0.15)
            ax_c.set_xlabel(r"Control Parameter $\Lambda = \frac{\alpha}{4\kappa_2}$ [-]", fontsize=13, labelpad=8)
            ax_c.set_ylabel(r"Dimensionless Relaxation Rate $\lambda / (4\kappa_2D_r^\mathrm{eff})$ [-]", fontsize=13, labelpad=8)
            ax_c.set_title(r"(C) Linear Relaxation Rate & Critical Slowing", fontsize=14, fontweight='bold', pad=10)
            ax_c.grid(True, linestyle=':', alpha=0.7)
            ax_c.legend(loc='lower left', fontsize=9.0, framealpha=0.92)

            fig_bif.suptitle(r'Nonlinear Orientation Dynamics & Saddle-Node Bifurcation Analysis',
                             fontsize=16, y=0.99)
            fig_bif.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(fig_bif, "cos_i_saddle_node_bifurcation_phase_dynamics_3panels", OUT_DIRS)
            plt.close(fig_bif)

            # ---------------------------------------------------------
            # 8i. 1次+2次打ち切りモデル vs ハイブリッドモデルの分岐トポロジー直接比較
            #     - 打ち切りモデル: \tilde{U}(c) = -\kappa_1 c + 2\kappa_2 c^2 -> c* = \Lambda (分岐なし・非有界)
            #     - ハイブリッドモデル: \tilde{U}(c) = -\alpha \operatorname{arctanh}(c) + 2\kappa_2 c^2
            #                         -> c*(1 - c*^2) = \Lambda (サドルノード分岐 \Lambda_c \approx 0.385, 有界 |c|<1)
            # ---------------------------------------------------------
            fig_comp, axes_comp = plt.subplots(1, 3, figsize=(18.0, 5.5))
            ax_cp1, ax_cp2, ax_cp3 = axes_comp

            # Panel 1: 分岐ダイアグラムの直接比較 (c* vs \Lambda)
            lam_full = np.linspace(0.0, 1.2, 500)
            # 打ち切りモデル: c* = \Lambda (線形、分岐なし、物理的境界 c=1 を突き破る)
            ax_cp1.plot(lam_full, lam_full, color='#444444', linestyle=':', linewidth=2.2,
                        label=r'Truncated Linear: $c^* = \Lambda$ (No Bifurcation, Unbounded)', zorder=2)
            # 物理的上限 c = 1
            ax_cp1.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8,
                           label=r'Physical Boundary $c = 1$')

            # ハイブリッドモデル: 安定枝 & 不安定枝
            ax_cp1.plot(lam_arr, c_star_stable, color='#1b7837', linestyle='-', linewidth=2.5,
                        label=r'Hybrid: Stable Node $c_{\min}^*$ ($\Lambda \leq \Lambda_c$)', zorder=3)
            ax_cp1.plot(lam_arr, c_star_unstable, color='#d95f02', linestyle='--', linewidth=2.2,
                        label=r'Hybrid: Unstable Saddle $c_{\max}^*$', zorder=3)
            ax_cp1.scatter(lam_c, c_crit, color='crimson', marker='*', s=160, edgecolors='black',
                           linewidths=1.2, zorder=6, label=rf'Saddle-Node Point ($\Lambda_c \approx {lam_c:.3f}$)')

            ax_cp1.axvline(lam_c, color='crimson', linestyle=':', linewidth=1.5, zorder=1)
            ax_cp1.axvspan(lam_c, 1.25, color='#7570b3', alpha=0.08, zorder=1)
            ax_cp1.text(0.75, 0.45, "Ballistic Phase\n(No Fixed Points\nin Hybrid Model)",
                        fontsize=10.5, color='#7570b3', fontweight='bold', ha='center', va='center')

            # 実験データの配置
            for _, r_p in df_atanh_ratios.iterrows():
                bname = r_p['bead_name']
                a_val = r_p['A']
                k2_val = r_p['kappa_2']
                lbl = r_p['label']
                col = r_p['color']
                mrk = r_p['marker']
                if k2_val > 0.001:
                    lam_exp = a_val / (4.0 * k2_val)
                    if lam_exp <= lam_c:
                        arg_e = np.clip(3.0 * np.sqrt(3.0) / 2.0 * lam_exp, -1.0, 1.0)
                        c_e = float((2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_e)))
                        if bname == 'beads20um':
                            ax_cp1.scatter(0.0, 0.0, facecolors='white', edgecolors=col, marker=mrk,
                                           s=110, linewidths=2.0, linestyle='--', zorder=5, label=f"{lbl} ($\Lambda=0$, ref)")
                        else:
                            ax_cp1.scatter(lam_exp, c_e, color=col, marker=mrk, s=100, edgecolors='black',
                                           linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda={lam_exp:.2f}$)")
                    else:
                        ax_cp1.scatter(min(lam_exp, 1.15), 0.95, color=col, marker='>', s=110, edgecolors='black',
                                       linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda={lam_exp:.2f} > \Lambda_c$)")
                else:
                    ax_cp1.scatter(1.15, 0.95, color=col, marker='>', s=110, edgecolors='black',
                                   linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda \\to \\infty$)")

            ax_cp1.set_xlim(-0.02, 1.2)
            ax_cp1.set_ylim(-0.05, 1.25)
            ax_cp1.set_xlabel(r"Control Parameter $\Lambda = \frac{\alpha}{4\kappa_2}$ [-]", fontsize=13, labelpad=8)
            ax_cp1.set_ylabel(r"Fixed Point $c^*$ [-]", fontsize=13, labelpad=8)
            ax_cp1.set_title(r"(A) Bifurcation Topology Comparison", fontsize=14, fontweight='bold', pad=10)
            ax_cp1.grid(True, linestyle=':', alpha=0.7)
            ax_cp1.legend(loc='upper left', fontsize=8.2, framealpha=0.92)

            # Panel 2: 有効ポテンシャル形状の比較 (\Lambda = 0.2 の場合)
            c_pot = np.linspace(-0.95, 0.95, 400)
            lam_sample = 0.2
            # 規格化ポテンシャル (4 \kappa_2 で除算した形状)
            # 打ち切り: \tilde{U}_trunc / (4\kappa_2) = - \Lambda * c + 0.5 * c^2
            # ハイブリッド: \tilde{U}_hybrid / (4\kappa_2) = - \Lambda * arctanh(c) + 0.5 * c^2
            u_trunc_sample = - lam_sample * c_pot + 0.5 * (c_pot**2)
            u_hyb_sample = - lam_sample * np.arctanh(c_pot) + 0.5 * (c_pot**2)

            ax_cp2.plot(c_pot, u_trunc_sample, color='#444444', linestyle=':', linewidth=2.2,
                        label=r'Truncated: $-\Lambda c + \frac{1}{2} c^2$ (Parabolic Well)', zorder=2)
            ax_cp2.plot(c_pot, u_hyb_sample, color='#1b7837', linestyle='-', linewidth=2.5,
                        label=r'Hybrid: $-\Lambda \operatorname{arctanh}(c) + \frac{1}{2} c^2$', zorder=3)

            # 極小・極大のマーカー
            c_min_trunc = lam_sample
            u_min_trunc = - lam_sample * c_min_trunc + 0.5 * (c_min_trunc**2)
            ax_cp2.scatter(c_min_trunc, u_min_trunc, color='#444444', marker='s', s=80, edgecolors='black',
                           zorder=5, label=rf'Truncated Min ($c^*={c_min_trunc:.2f}$)')

            arg_s = 3.0 * np.sqrt(3.0) / 2.0 * lam_sample
            c_min_h = (2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_s))
            u_min_h = - lam_sample * np.arctanh(c_min_h) + 0.5 * (c_min_h**2)
            ax_cp2.scatter(c_min_h, u_min_h, color='#1b7837', marker='o', s=80, edgecolors='black',
                           zorder=5, label=r'Hybrid Stable Well ($c_{\mathrm{min}}^*=' + f'{c_min_h:.2f}' + r'$)')

            c_max_h = (2.0 / np.sqrt(3.0)) * np.sin((np.pi / 3.0) - (1.0 / 3.0) * np.arcsin(arg_s))
            u_max_h = - lam_sample * np.arctanh(c_max_h) + 0.5 * (c_max_h**2)
            ax_cp2.scatter(c_max_h, u_max_h, color='#d95f02', marker='^', s=80, edgecolors='black',
                           zorder=5, label=r'Hybrid Barrier Peak ($c_{\mathrm{max}}^*=' + f'{c_max_h:.2f}' + r'$)')

            ax_cp2.set_xlim(-1.0, 1.0)
            ax_cp2.set_ylim(-0.45, 0.75)
            ax_cp2.set_xlabel(r"Orientation State $c = \mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ [-]", fontsize=13, labelpad=8)
            ax_cp2.set_ylabel(r"Reduced Potential $\tilde{U}(c) / (4\kappa_2)$ [-]", fontsize=13, labelpad=8)
            ax_cp2.set_title(rf"(B) Potential Landscape at $\Lambda = {lam_sample}$", fontsize=14, fontweight='bold', pad=10)
            ax_cp2.grid(True, linestyle=':', alpha=0.7)
            ax_cp2.legend(loc='upper center', fontsize=8.8, framealpha=0.92)

            # Panel 3: 位相速度 F(c) の比較 (\Lambda = 0.2 & \Lambda = 0.5)
            denom_cp = 1.0 - c_pot**2
            ax_cp3.axhline(0.0, color='#666666', linestyle='-', linewidth=1.2, zorder=2)
            # \Lambda = 0.2
            f_trunc_02 = lam_sample - c_pot
            f_hyb_02 = (lam_sample / denom_cp) - c_pot
            ax_cp3.plot(c_pot, f_trunc_02, color='#444444', linestyle=':', linewidth=2.0,
                        label=r'Truncated ($\Lambda = 0.2$): $\Lambda - c$', zorder=2)
            ax_cp3.plot(c_pot, f_hyb_02, color='#1b7837', linestyle='-', linewidth=2.2,
                        label=r'Hybrid ($\Lambda = 0.2$): $\frac{\Lambda}{1-c^2} - c$', zorder=3)

            # \Lambda = 0.5 > \Lambda_c (弾道輸送領域)
            lam_high = 0.5
            f_trunc_05 = lam_high - c_pot
            f_hyb_05 = (lam_high / denom_cp) - c_pot
            ax_cp3.plot(c_pot, f_trunc_05, color='#888888', linestyle=':', linewidth=1.8,
                        label=r'Truncated ($\Lambda = 0.5$): Fixed point at $c=0.5$', zorder=2)
            ax_cp3.plot(c_pot, f_hyb_05, color='#7570b3', linestyle='--', linewidth=2.2,
                        label=r'Hybrid ($\Lambda = 0.5$): No Fixed Point (All $\dot{c} > 0$)', zorder=3)

            ax_cp3.set_xlim(-0.95, 0.95)
            ax_cp3.set_ylim(-1.5, 2.5)
            ax_cp3.set_xlabel(r"Orientation State $c = \mathbf{\hat{v}} \cdot \mathbf{\hat{u}}$ [-]", fontsize=13, labelpad=8)
            ax_cp3.set_ylabel(r"Phase Velocity $F(c; \Lambda) = \frac{1}{4\kappa_2D_r^\mathrm{eff}} \dot{c}$ [-]", fontsize=13, labelpad=8)
            ax_cp3.set_title(r"(C) Phase Velocity: Diffusive vs Ballistic Escape", fontsize=14, fontweight='bold', pad=10)
            ax_cp3.grid(True, linestyle=':', alpha=0.7)
            ax_cp3.legend(loc='upper left', fontsize=8.5, framealpha=0.92)

            fig_comp.suptitle(r'Bifurcation Topology: Truncated Linear-Quadratic vs Hybrid Arctanh Model',
                              fontsize=16, y=0.99)
            fig_comp.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(fig_comp, "cos_i_model_topology_comparison_truncated_vs_hybrid_3panels", OUT_DIRS)
            plt.close(fig_comp)

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
    df_peven_podd = pd.DataFrame(peven_podd_records)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_dir / "cos_decomposition_summary_by_radius.csv", index=False)
        df_window_summary.to_csv(out_dir / "cos_decomposition_summary_by_window.csv", index=False)
        df_skew.to_csv(out_dir / "cos_i_skewness_summary.csv", index=False)
        df_cargo_stats.to_csv(out_dir / "cos_i_cargo_individual_stats.csv", index=False)
        if not df_peven_podd.empty:
            df_peven_podd.to_csv(out_dir / "cos_i_peven_podd_summary.csv", index=False)
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
