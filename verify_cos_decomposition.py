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
    ax.set_ylabel(r"Pixel-Averaged Coupling $\langle \cos \theta_i \rangle$")
    ax.set_title(r"$\langle \cos \theta_i \rangle$ vs $\cos \Theta_{\mathrm{bulk}}$ ($R = R_c$)", fontsize=14, fontweight='bold')
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
    ax1.set_ylabel(r"Pixel-Averaged Coupling $\langle \cos \theta_i \rangle$")
    ax1.set_title(r"Direct Correlation: $\langle \cos \theta_i \rangle$ vs $\Phi(R_c)$", fontsize=14, fontweight='bold')
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
    ax2.set_ylabel(r"Pixel-Averaged Coupling $\langle \cos \theta_i \rangle$")
    ax2.set_title(r"Verification: $\langle \cos \theta_i \rangle$ vs $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$", fontsize=14, fontweight='bold')
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
    ax_sa.set_ylabel(r"Pixel-Averaged Coupling $\langle \cos \theta_i \rangle$")
    ax_sa.set_title(r"Instantaneous: $\langle \cos \theta_i \rangle$ vs $\Phi(R_c)$", fontsize=14, fontweight='bold')
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
    ax_sb.set_ylabel(r"Track-Averaged Coupling $\langle \langle \cos \theta_i \rangle \rangle_{\mathrm{track}}$")
    ax_sb.set_title(r"Particle-Averaged: $\langle \cos \theta_i \rangle$ vs $\Phi(R_c)$", fontsize=14, fontweight='bold')
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
        summary_rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'n_particles': n_p,
            'cos_i_mean': float(p_agg['cos_i'].mean()),
            'cos_i_sem': float(p_agg['cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_bulk_mean': float(p_agg['cos_bulk'].mean()),
            'cos_bulk_sem': float(p_agg['cos_bulk'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_mean': float(p_agg['Phi_mag'].mean()),
            'Phi_sem': float(p_agg['Phi_mag'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
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

    # Local polar order Phi(R_c)
    ax.errorbar(r_vals, df_summary['Phi_mean'], yerr=df_summary['Phi_sem'],
                fmt='^-.', color='#EE6677', markersize=8, capsize=4, linewidth=2.0,
                label=r"Local polar order $\Phi(R_c)$ (from local_polar_w.zarr)")

    # Product Phi * cos_bulk (理論積)
    ax.errorbar(r_vals * 1.04, df_summary['pred_mean'], yerr=df_summary['pred_sem'],
                fmt='d:', color='#AA3377', markersize=8, capsize=4, linewidth=2.0,
                label=r"Decomposition $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$")

    # Actual <cos theta_i> (各点平均)
    ax.errorbar(r_vals, df_summary['cos_i_mean'], yerr=df_summary['cos_i_sem'],
                fmt='o-', color='#228833', markersize=9, capsize=5, linewidth=2.4,
                label=r"Observed pixel-mean $\langle \cos \theta_i \rangle$")

    ax.set_xscale('log')
    ax.set_xlim(0.2, 15.0)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"Cargo Radius $R_c$ [$\mu\mathrm{m}$]")
    ax.set_ylabel("Coupling / Order Parameter [-]")
    ax.set_title(r"Decomposition of Directional Coupling: $\langle \cos \theta_i \rangle \approx \Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$",
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
                label=rf'$\langle \cos \theta_i \rangle(R)$ (Observed)')
        ax.fill_between(r_w, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        ax.set_xscale('log')
        ax.set_xlim(0.4, 120.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(f"{lbl}  ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$, $N={n_p}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel("Coupling / Order Parameter", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Window Dependence Decomposition: $\langle \cos \theta_i \rangle(R) \approx \Phi(R) \times \cos \Theta_{\mathrm{bulk}}(R)$',
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
        rc_xi_summary.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'radius_um': b_item['radius_um'],
            'n_particles': n_p,
            'rc_over_xi_mean': float(sub_p['rc_over_xi'].mean()),
            'rc_over_xi_sem': float(sub_p['rc_over_xi'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_mean': float(sub_p['cos_i'].mean()),
            'cos_i_sem': float(sub_p['cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_bulk_mean': float(sub_p['cos_bulk'].mean()),
            'cos_bulk_sem': float(sub_p['cos_bulk'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_mean': float(sub_p['Phi_mag'].mean()),
            'Phi_sem': float(sub_p['Phi_mag'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
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

    # Local polar order Phi(R_c)
    ax.errorbar(rc_xi_vals, df_rc_xi['Phi_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['Phi_sem'],
                fmt='^-.', color='#EE6677', markersize=8, capsize=4, linewidth=2.0, zorder=4,
                label=r"Local polar order $\Phi(R_c)$")

    # Product Phi * cos_bulk (理論積)
    ax.errorbar(rc_xi_vals * 1.03, df_rc_xi['pred_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['pred_sem'],
                fmt='d:', color='#AA3377', markersize=8, capsize=4, linewidth=2.0, zorder=5,
                label=r"Decomposition $\Phi(R_c) \times \cos \Theta_{\mathrm{bulk}}$")

    # Actual <cos theta_i> (各点平均)
    ax.errorbar(rc_xi_vals, df_rc_xi['cos_i_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['cos_i_sem'],
                fmt='o-', color='#228833', markersize=9, capsize=5, linewidth=2.4, zorder=6,
                label=r"Observed pixel-mean $\langle \cos \theta_i \rangle$")

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
                label=rf'$\langle \cos \theta_i \rangle(R)$')
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
    ax.set_ylabel(r"Observed Coupling $\langle \cos \theta_i \rangle(R)$")
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
                label=rf'$\langle \cos \theta_i \rangle(R)$')
        ax.fill_between(r_w, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        #ax.set_xscale('log')
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(f"{lbl}  ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$, $N={n_p}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\langle \cos \theta_i \rangle\left(R\right)$", fontsize=14)
        ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92)

    fig.suptitle(r'Window Dependence: $\langle \cos \theta_i \rangle(R)$',
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
                label=rf'$\langle \cos \theta_i \rangle(R)$')
        ax.fill_between(r_w, ci_m - ci_s, ci_m + ci_s, color=color, alpha=0.2)

        #ax.set_xscale('log')
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(-0.25, 1.05)
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\langle \cos \theta_i \rangle\left(R\right)$", fontsize=14)
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
        rc_xi_summary.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'radius_um': b_item['radius_um'],
            'n_particles': n_p,
            'rc_over_xi_mean': float(sub_p['rc_over_xi'].mean()),
            'rc_over_xi_sem': float(sub_p['rc_over_xi'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_i_mean': float(sub_p['cos_i'].mean()),
            'cos_i_sem': float(sub_p['cos_i'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'Phi_mean': float(sub_p['Phi_mag'].mean()),
            'Phi_sem': float(sub_p['Phi_mag'].std() / np.sqrt(n_p)) if n_p > 1 else 0.0,
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

    # Local polar order Phi(R_c)
    ax.errorbar(rc_xi_vals, df_rc_xi['Phi_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['Phi_sem'],
                fmt='^-.', color='#EE6677', markersize=8, capsize=4, linewidth=2.0, zorder=4,
                label=r"Local polar order $\Phi(R_c)$")

    # Actual <cos theta_i> (各点平均)
    ax.errorbar(rc_xi_vals, df_rc_xi['cos_i_mean'],
                xerr=df_rc_xi['rc_over_xi_sem'], yerr=df_rc_xi['cos_i_sem'],
                fmt='o-', color='#228833', markersize=9, capsize=5, linewidth=2.4, zorder=6,
                label=r"Observed pixel-mean $\langle \cos \theta_i \rangle$")

    #ax.set_xscale('log')
    ax.set_xlim(0.04, 3.5)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]")
    ax.set_ylabel("Coupling / Order Parameter [-]")
    ax.set_title(r"Decomposition vs Normalized Cargo Radius: $R_c / \xi$",
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
    ax.grid(True)

    fig.tight_layout()
    save_figure(fig, "cos_decomposition_vs_rc_over_xi_raw", OUT_DIRS)
    plt.close(fig)


    # CSV 保存
    df_window_summary = pd.DataFrame(window_summary_list)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_dir / "cos_decomposition_summary_by_radius.csv", index=False)
        df_window_summary.to_csv(out_dir / "cos_decomposition_summary_by_window.csv", index=False)
    print("Saved decomposition summary CSVs.")

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
