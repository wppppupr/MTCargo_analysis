#!/usr/bin/env python3
"""
verify_theta_decomposition.py

角度変数 theta = arctan2(sin, cos) に基づく配向ダイナミクス解析スクリプト:
1. 角度 theta の厳密導出:
   c = cos_sim, s = sqrt(1 - c^2), theta = arctan2(s, c)  (theta in [0, pi])
2. 確率密度分布 P(theta):
   - クラスターブートストラップ (95% CI)
   - 6パネル & 1パネル統合図
   - ※ 偶奇分解は行わない
3. 有効ポテンシャル U_eff(theta) のボルツマン反転:
   U_eff(theta) = - ln( P(theta) / P(pi/2) )
4. theta 表現におけるハイブリッドポテンシャルフィッティング:
   U_model(theta) = A * ln(tan(theta / 2)) + 2 * kappa_2 * cos^2(theta)
   (arctanh(cos theta) = -ln(tan(theta/2)) による厳密等価変換)
5. 抽出パラメータ A, kappa_2, Lambda = A / (4 * kappa_2) vs R_c / xi
   - 0.63 um: 生のフィッティング値を素直にプロット
   - 20.0 um: A=0, kappa_2=0, Lambda=0 の参考値（白抜きマーカー）
6. theta 空間における過減衰配向ダイナミクスの線形安定性解析（3 パネル図）:
   - Panel (A): サドルノード分岐図 (theta* vs Lambda)
   - Panel (B): 位相フロー場 F_theta(theta; Lambda) = d(theta)/dt / (2*kappa_2)
   - Panel (C): 線形緩和率 / 臨界減速 lambda_theta / (4*kappa_2) vs Lambda
7. モデルトポロジー比較図（打ち切り線形 vs ハイブリッド）

出力先:
  - figure/theta_decomposition/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/theta_decomposition/
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
from scipy.optimize import curve_fit

# ワークスペースおよびNASのパス設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR_CANDIDATES = [
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    CURRENT_DIR,
]
ROOT_DIR = next((p for p in ROOT_DIR_CANDIDATES if p.exists()), CURRENT_DIR)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "theta_decomposition",
    ROOT_DIR / "figure" / "theta_decomposition"
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

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


def save_figure(fig, base_name: str, out_dirs: List[Path]):
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            png_p = d / f"{base_name}.png"
            svg_p = d / f"{base_name}.svg"
            fig.savefig(png_p, dpi=300, bbox_inches='tight')
            fig.savefig(svg_p, format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"Warning: failed to save to {d}: {e}")
    print(f"Saved figure: {base_name}.png and .svg")


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
                print(f"Loaded {len(xi_map)} correlation lengths xi from {p.name}")
                return xi_map
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
    return xi_map


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


def load_all_data(root_dir: Path, active_set: set):
    instant_list = []
    window_list = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        exp_dirs = find_experiment_dirs(root_dir, bname)
        print(f"Processing {bname} ({len(exp_dirs)} experiments)...")

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            inter_p = exp_dir / "beads_flow_interaction.csv"
            source_type = "box"
            if not inter_p.exists():
                inter_p = exp_dir / "beads_flow_interaction_gaussian.csv"
                source_type = "gaussian"
            if not inter_p.exists():
                continue

            try:
                df_flow = pd.read_csv(inter_p)
            except Exception:
                continue

            rad_col = 'radius' if 'radius' in df_flow.columns else 'rad'
            if rad_col not in df_flow.columns:
                continue

            # トラック座標とカーゴ速度のロード
            tracks_p = exp_dir / "beads_tracks.csv"
            if not tracks_p.exists():
                tracks_p = exp_dir / "beads_tracks_raw.csv"
            if not tracks_p.exists():
                continue

            try:
                tdf = pd.read_csv(tracks_p)
            except Exception:
                continue

            cargo_velocity_map = {}
            if 'vx' in tdf.columns and 'vy' in tdf.columns:
                for _, r in tdf.iterrows():
                    vx = float(r['vx'])
                    vy = float(r['vy'])
                    spd = np.hypot(vx, vy)
                    cargo_velocity_map[(int(r['particle']), int(r['frame']))] = (vx, vy, spd)

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
                                    arr = arr.T
                                f_map = {int(f): i for i, f in enumerate(z_frames)}
                                w_map = {float(w): i for i, w in enumerate(z_windows)}
                                polar_cache[int(p_val)] = {
                                    'arr': arr,
                                    'f_map': f_map,
                                    'w_map': w_map,
                                    'windows': z_windows
                                }
                except Exception:
                    pass

            df_flow['rad_px'] = df_flow[rad_col].astype(float)
            valid_mask = (df_flow['mean_flow_u'] != 0) | (df_flow['mean_flow_v'] != 0) | (df_flow['cos_sim'] != 0)
            valid_rads = np.sort(df_flow.loc[valid_mask, 'rad_px'].unique())
            if len(valid_rads) == 0:
                continue

            cargo_rad_px = r_um / SCALE_UM_PER_PX
            cand_rads = valid_rads[valid_rads >= cargo_rad_px * 0.8]
            if len(cand_rads) > 0:
                best_rad = float(cand_rads[int(np.argmin(np.abs(cand_rads - cargo_rad_px)))])
            else:
                best_rad = float(valid_rads[-1])

            sub_best = df_flow[(df_flow['rad_px'] == best_rad) & valid_mask]
            for _, r in sub_best.iterrows():
                pid = int(r['particle'])
                if active_set and (exp_name, pid) not in active_set:
                    continue
                frm = int(r['frame'])
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
                cos_i = float(r['cos_sim'])
                cos_bulk = (cdx * u + cdy * v) / (cspd * v_flow_norm)

                phi_inst = np.nan
                if pid in polar_cache:
                    pc = polar_cache[pid]
                    f_idx = pc['f_map'].get(frm)
                    if f_idx is not None:
                        if best_rad in pc['w_map']:
                            w_idx = pc['w_map'][best_rad]
                            val = pc['arr'][w_idx, f_idx]
                            if np.isfinite(val) and 0.0 <= val <= 1.5:
                                phi_inst = float(np.clip(val, 0.0, 1.0))
                        if np.isnan(phi_inst):
                            for w_cand in sorted(pc['windows']):
                                if w_cand >= best_rad:
                                    w_idx = pc['w_map'][float(w_cand)]
                                    val = pc['arr'][w_idx, f_idx]
                                    if np.isfinite(val) and 0.0 <= val <= 1.5:
                                        phi_inst = float(np.clip(val, 0.0, 1.0))
                                        break

                # 角度 theta の厳密計算: arctan2(sin, cos)
                cos_clip = np.clip(cos_i, -1.0, 1.0)
                sin_i = np.sqrt(np.clip(1.0 - cos_clip**2, 0.0, 1.0))
                theta_val = float(np.arctan2(sin_i, cos_clip))

                instant_list.append({
                    'bead_name': bname,
                    'diameter_um': d_um,
                    'radius_um': r_um,
                    'exp_dir': exp_name,
                    'particle': pid,
                    'frame': frm,
                    'radius_px': best_rad,
                    'window_radius_um': best_rad * SCALE_UM_PER_PX,
                    'cos_i': cos_clip,
                    'theta': theta_val,
                    'cos_bulk': cos_bulk,
                    'Phi_mag': phi_inst,
                    'v_flow_norm': v_flow_norm,
                    'v_cargo_norm': cspd,
                    'dot_product': dot_p,
                    'source_type': source_type,
                })

            for rad in valid_rads:
                sub_rad = df_flow[(df_flow['rad_px'] == rad) & valid_mask]
                for pid, grp in sub_rad.groupby('particle'):
                    pid = int(pid)
                    if active_set and (exp_name, pid) not in active_set:
                        continue
                    ci_vals, cb_vals, vf_vals = [], [], []
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


def main():
    setup_matplotlib()
    print("=== Directional Coupling Theta Decomposition Analysis ===")
    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    df_instant, df_window = load_all_data(ROOT_DIR, active_set)
    print(f"Extracted {len(df_instant)} instantaneous points, {len(df_window)} window aggregation points.")

    if df_instant.empty:
        print("Error: No instantaneous points collected. Exiting.")
        return

    # 全体の代表相関長（フォールバック用）
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0
    df_instant['xi_um'] = df_instant['exp_dir'].map(xi_dict).fillna(default_xi)
    df_instant['cargo_id'] = df_instant['exp_dir'].astype(str) + '_' + df_instant['particle'].astype(str)

    # ビーズごとの R_c / xi 集計
    df_window['xi_um'] = df_window['exp_dir'].map(xi_dict).fillna(default_xi)
    df_window['rc_over_xi'] = df_window['radius_um'] / df_window['xi_um']

    rc_xi_records = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_w = df_window[df_window['bead_name'] == bname]
        if not sub_w.empty:
            p_sub = sub_w.drop_duplicates('particle')
            rx_m = float(p_sub['rc_over_xi'].mean())
            rx_s = float(p_sub['rc_over_xi'].sem()) if len(p_sub) > 1 else 0.0
            phi_m = float(p_sub['Phi_mean'].dropna().mean()) if not p_sub['Phi_mean'].dropna().empty else np.nan
        else:
            rx_m, rx_s, phi_m = np.nan, 0.0, np.nan
        rc_xi_records.append({
            'bead_name': bname,
            'label': b_item['label'],
            'radius_um': b_item['radius_um'],
            'rc_over_xi_mean': rx_m,
            'rc_over_xi_sem': rx_s,
            'Phi_mean': phi_m
        })
    df_rc_xi = pd.DataFrame(rc_xi_records)

    # -------------------------------------------------------------
    # 1. 確率密度分布 P(theta) の計算（クラスターブートストラップ 95% 信頼区間）
    #    theta in [0, pi] を 28 ビンに分割
    # -------------------------------------------------------------
    N_BINS_TH = 28
    bins_th = np.linspace(0.0, np.pi, N_BINS_TH + 1)
    bin_centers_th = 0.5 * (bins_th[:-1] + bins_th[1:])
    idx_half_pi = int(np.argmin(np.abs(bin_centers_th - np.pi / 2.0)))

    B_BOOT = 1000
    rng_boot = np.random.default_rng(42)

    boot_hist_dict = {}
    fit_records = []

    def model_U_theta(th, A, k2):
        # U_model(theta) = A * ln(tan(theta/2)) + 2 * kappa_2 * cos^2(theta)
        th_clip = np.clip(th, 1e-4, np.pi - 1e-4)
        return A * np.log(np.tan(th_clip / 2.0)) + 2.0 * k2 * (np.cos(th_clip)**2)

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub_inst = df_instant[df_instant['bead_name'] == bname]
        if sub_inst.empty:
            continue

        th_vals_all = sub_inst['theta'].dropna().values
        if len(th_vals_all) == 0:
            continue

        counts_obs, _ = np.histogram(th_vals_all, bins=bins_th, density=True)
        P_obs = counts_obs

        # 有効ポテンシャル U_eff(theta) = - ln( P(theta) / P(pi/2) )
        p_ref_obs = P_obs[idx_half_pi] if P_obs[idx_half_pi] > 0 else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            arg_u = P_obs / p_ref_obs
            U_eff_obs = np.where(arg_u > 0, -np.log(arg_u), np.nan)

        # 軌道単位の分割
        cargos_data = []
        for _, grp in sub_inst.groupby('cargo_id'):
            t_vals = grp['theta'].dropna().values
            if len(t_vals) > 0:
                cargos_data.append(t_vals)
        n_cargos = len(cargos_data)

        boot_P = np.zeros((B_BOOT, len(bin_centers_th)))
        boot_Ueff = np.zeros((B_BOOT, len(bin_centers_th)))
        boot_A_arr = np.zeros(B_BOOT)
        boot_k2_arr = np.zeros(B_BOOT)

        # 代表フィッティング
        mask_fit = (P_obs > 1e-4) & (bin_centers_th > 0.15) & (bin_centers_th < np.pi - 0.15)
        A_fit, k2_fit = np.nan, np.nan
        if np.sum(mask_fit) >= 3:
            try:
                popt, _ = curve_fit(
                    model_U_theta,
                    bin_centers_th[mask_fit],
                    U_eff_obs[mask_fit],
                    p0=[0.35, 0.8],
                    bounds=([0.0, 0.0], [5.0, 5.0]),
                    maxfev=5000
                )
                A_fit, k2_fit = float(popt[0]), float(popt[1])
            except Exception as e:
                print(f"Fit failed for {bname}: {e}")

        # ブートストラップ試行
        if n_cargos >= 2:
            for b_idx in range(B_BOOT):
                sel_indices = rng_boot.choice(n_cargos, size=n_cargos, replace=True)
                sample_pts = np.concatenate([cargos_data[i] for i in sel_indices])
                th_boot, _ = np.histogram(sample_pts, bins=bins_th, density=True)
                boot_P[b_idx] = th_boot

                p_ref_b = th_boot[idx_half_pi] if th_boot[idx_half_pi] > 0 else np.nan
                with np.errstate(divide='ignore', invalid='ignore'):
                    arg_ub = th_boot / p_ref_b
                    u_b = np.where(arg_ub > 0, -np.log(arg_ub), np.nan)
                boot_Ueff[b_idx] = u_b

                mask_b = (th_boot > 1e-4) & (bin_centers_th > 0.15) & (bin_centers_th < np.pi - 0.15)
                if np.sum(mask_b) >= 3:
                    try:
                        popt_b, _ = curve_fit(
                            model_U_theta,
                            bin_centers_th[mask_b],
                            u_b[mask_b],
                            p0=[A_fit if not np.isnan(A_fit) else 0.35, k2_fit if not np.isnan(k2_fit) else 0.8],
                            bounds=([0.0, 0.0], [5.0, 5.0]),
                            maxfev=3000
                        )
                        boot_A_arr[b_idx] = float(popt_b[0])
                        boot_k2_arr[b_idx] = float(popt_b[1])
                    except Exception:
                        boot_A_arr[b_idx] = np.nan
                        boot_k2_arr[b_idx] = np.nan
                else:
                    boot_A_arr[b_idx] = np.nan
                    boot_k2_arr[b_idx] = np.nan

            ci_low_p = np.percentile(boot_P, 2.5, axis=0)
            ci_high_p = np.percentile(boot_P, 97.5, axis=0)
            ci_low_ueff = np.nanpercentile(boot_Ueff, 2.5, axis=0)
            ci_high_ueff = np.nanpercentile(boot_Ueff, 97.5, axis=0)
            A_ci_low = float(np.nanpercentile(boot_A_arr, 2.5))
            A_ci_high = float(np.nanpercentile(boot_A_arr, 97.5))
            k2_ci_low = float(np.nanpercentile(boot_k2_arr, 2.5))
            k2_ci_high = float(np.nanpercentile(boot_k2_arr, 97.5))
        else:
            ci_low_p, ci_high_p = P_obs, P_obs
            ci_low_ueff, ci_high_ueff = U_eff_obs, U_eff_obs
            A_ci_low, A_ci_high = A_fit, A_fit
            k2_ci_low, k2_ci_high = k2_fit, k2_fit

        boot_hist_dict[bname] = {
            'P_obs': P_obs,
            'ci_low_p': ci_low_p,
            'ci_high_p': ci_high_p,
            'U_eff_obs': U_eff_obs,
            'ci_low_ueff': ci_low_ueff,
            'ci_high_ueff': ci_high_ueff,
            'A_fit': A_fit,
            'kappa_2_fit': k2_fit,
            'A_ci_low': A_ci_low,
            'A_ci_high': A_ci_high,
            'k2_ci_low': k2_ci_low,
            'k2_ci_high': k2_ci_high,
            'n_cargos': n_cargos,
            'n_points': len(th_vals_all),
            'mean_theta': float(np.mean(th_vals_all)),
        }

        row_rc = df_rc_xi[df_rc_xi['bead_name'] == bname]
        rc_m = float(row_rc['rc_over_xi_mean'].values[0]) if not row_rc.empty else np.nan
        rc_s = float(row_rc['rc_over_xi_sem'].values[0]) if not row_rc.empty else 0.0

        fit_records.append({
            'bead_name': bname,
            'label': b_item['label'],
            'color': b_item['color'],
            'marker': b_item['marker'],
            'rc_over_xi_mean': rc_m,
            'rc_over_xi_sem': rc_s,
            'A': A_fit,
            'A_ci_low': A_ci_low,
            'A_ci_high': A_ci_high,
            'kappa_2': k2_fit,
            'k2_ci_low': k2_ci_low,
            'k2_ci_high': k2_ci_high,
        })

    df_theta_params = pd.DataFrame(fit_records).sort_values('rc_over_xi_mean').reset_index(drop=True)

    # -------------------------------------------------------------
    # 2. 確率密度分布 P(theta) の描画（6パネル & 1パネル統合図）
    # -------------------------------------------------------------
    fig_p6, axes_p6 = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_p6_flat = axes_p6.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_p6_flat[idx]
        bname = b_item['name']
        col = b_item['color']
        lbl = b_item['label']

        if bname not in boot_hist_dict:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        res = boot_hist_dict[bname]
        phi_mean_val = float(df_rc_xi[df_rc_xi['bead_name'] == bname]['Phi_mean'].values[0]) if not df_rc_xi.empty else np.nan
        mean_th = res['mean_theta']

        ax.plot(bin_centers_th, res['P_obs'], color=col, linewidth=2.4, marker=b_item['marker'], markersize=5,
                label=rf"Observed ($N = {res['n_points']}$, $N_{{\mathrm{{cargo}}}} = {res['n_cargos']}$)", zorder=3)
        ax.fill_between(bin_centers_th, res['ci_low_p'], res['ci_high_p'], color=col, alpha=0.25,
                        label=r"95% CI (Cluster Bootstrap)", zorder=2)

        ax.axvline(np.pi / 2.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r"Orthogonal $\pi/2$")
        ax.axvline(mean_th, color=col, linestyle='--', linewidth=1.8,
                   label=rf"Mean $\theta = {mean_th:.2f}$ rad ({np.degrees(mean_th):.1f}$^\circ$)", zorder=2)

        ax.set_xlim(0.0, np.pi)
        ax.set_ylim(0.0, 1.4)
        ax.set_xticks([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_title(rf"{lbl}  ($\Phi = {phi_mean_val:.2f}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Orientation Angle $\theta = \arccos(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ [rad]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\theta)$ [$\mathrm{rad}^{-1}$]", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9.2, framealpha=0.92)

    fig_p6.suptitle(r'Probability Density of Cargo-Flow Angle $P(\theta)$ (Cluster Bootstrap 95% CI)',
                    fontsize=17, y=0.99)
    fig_p6.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig_p6, "theta_probability_density_histograms_6panels", OUT_DIRS)
    plt.close(fig_p6)

    # 1パネル統合図
    fig_p1, ax_p1 = plt.subplots(figsize=(10.0, 7.0))
    ax_p1.axvline(np.pi / 2.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1, label=r"Orthogonal $\pi/2$")

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        col = b_item['color']
        lbl = b_item['label']
        if bname not in boot_hist_dict:
            continue
        res = boot_hist_dict[bname]
        ax_p1.plot(bin_centers_th, res['P_obs'], marker=b_item['marker'], markersize=6, color=col,
                   linewidth=2.2, label=rf"{lbl} ($N={res['n_cargos']}$)", zorder=3 + idx)
        ax_p1.fill_between(bin_centers_th, res['ci_low_p'], res['ci_high_p'], color=col, alpha=0.15, zorder=2)

    ax_p1.set_xlim(0.0, np.pi)
    ax_p1.set_ylim(0.0, 1.4)
    ax_p1.set_xticks([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax_p1.set_xticklabels([r"$0$ (Forward)", r"$\pi/4$", r"$\pi/2$ (Orthogonal)", r"$3\pi/4$", r"$\pi$ (Backward)"])
    ax_p1.set_xlabel(r"Orientation Angle $\theta = \arccos(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ [rad]", fontsize=14, labelpad=8)
    ax_p1.set_ylabel(r"Probability Density $P(\theta)$ [$\mathrm{rad}^{-1}$]", fontsize=14, labelpad=8)
    ax_p1.set_title(r"Probability Density of Cargo-Flow Coupling $P(\theta)$ (Cluster Bootstrap 95% CI)",
                    fontsize=15, fontweight='bold', pad=12)
    ax_p1.grid(True, linestyle=':', alpha=0.7)
    ax_p1.legend(loc='upper right', fontsize=10.5, framealpha=0.92)
    fig_p1.tight_layout()
    save_figure(fig_p1, "theta_probability_density_1panel", OUT_DIRS)
    plt.close(fig_p1)

    # -------------------------------------------------------------
    # 3. 有効ポテンシャル U_eff(theta) とフィッティング曲線の描画
    #    (1) 6パネル版 (観測値 + 95% CI + フィット曲線)
    #    (2) 1パネル統合版
    # -------------------------------------------------------------
    fig_u6, axes_u6 = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    axes_u6_flat = axes_u6.flatten()
    th_fine = np.linspace(0.05, np.pi - 0.05, 300)

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_u6_flat[idx]
        bname = b_item['name']
        col = b_item['color']
        lbl = b_item['label']

        if bname not in boot_hist_dict:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        res = boot_hist_dict[bname]
        a_f = res['A_fit']
        k2_f = res['kappa_2_fit']

        ax.scatter(bin_centers_th, res['U_eff_obs'], color=col, marker=b_item['marker'], s=45,
                   edgecolors='black', linewidths=1.0, zorder=4, label=r"Observed $U_{\mathrm{eff}}(\theta)$")
        ax.fill_between(bin_centers_th, res['ci_low_ueff'], res['ci_high_ueff'], color=col, alpha=0.25,
                        label=r"95% CI (Bootstrap)", zorder=2)

        if not np.isnan(a_f) and not np.isnan(k2_f):
            u_curve = model_U_theta(th_fine, a_f, k2_f)
            ax.plot(th_fine, u_curve, color='#222222', linestyle='-', linewidth=2.2, zorder=5,
                    label=rf"Fit: $A={a_f:.2f}, \kappa_2={k2_f:.2f}$")

        ax.axhline(0.0, color='gray', linestyle=':', linewidth=1.0, zorder=1)
        ax.axvline(np.pi / 2.0, color='gray', linestyle=':', linewidth=1.0, zorder=1)

        ax.set_xlim(0.0, np.pi)
        ax.set_ylim(-2.0, 4.0)
        ax.set_xticks([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_title(rf"{lbl}  ($A={a_f:.2f}, \kappa_2={k2_f:.2f}$)", fontsize=13.5, fontweight='bold')
        ax.set_xlabel(r"Orientation Angle $\theta$ [rad]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $U_{\mathrm{eff}}(\theta) / k_B T$", fontsize=13.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9.2, framealpha=0.92)

    fig_u6.suptitle(r'Effective Orientation Potential $U_{\mathrm{eff}}(\theta) = -\ln(P(\theta)/P(\pi/2))$ & Model Fit',
                    fontsize=16, y=0.99)
    fig_u6.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig_u6, "theta_effective_potential_fit_6panels", OUT_DIRS)
    plt.close(fig_u6)

    # 1パネル統合版
    fig_u1, ax_u1 = plt.subplots(figsize=(10.0, 7.0))
    ax_u1.axhline(0.0, color='gray', linestyle=':', linewidth=1.0, zorder=1)
    ax_u1.axvline(np.pi / 2.0, color='gray', linestyle=':', linewidth=1.0, zorder=1)

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        col = b_item['color']
        lbl = b_item['label']
        if bname not in boot_hist_dict:
            continue
        res = boot_hist_dict[bname]
        a_f = res['A_fit']
        k2_f = res['kappa_2_fit']

        ax_u1.plot(bin_centers_th, res['U_eff_obs'], marker=b_item['marker'], markersize=5, color=col,
                   linewidth=1.8, label=rf"{lbl} ($A={a_f:.2f}, \kappa_2={k2_f:.2f}$)", zorder=3 + idx)
        ax_u1.fill_between(bin_centers_th, res['ci_low_ueff'], res['ci_high_ueff'], color=col, alpha=0.12, zorder=2)

    ax_u1.set_xlim(0.0, np.pi)
    ax_u1.set_ylim(-2.0, 3.5)
    ax_u1.set_xticks([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax_u1.set_xticklabels([r"$0$ (Forward)", r"$\pi/4$", r"$\pi/2$ (Orthogonal)", r"$3\pi/4$", r"$\pi$ (Backward)"])
    ax_u1.set_xlabel(r"Orientation Angle $\theta = \arccos(\mathbf{\hat{v}} \cdot \mathbf{\hat{u}})$ [rad]", fontsize=14, labelpad=8)
    ax_u1.set_ylabel(r"Effective Potential $U_{\mathrm{eff}}(\theta) / k_B T$", fontsize=14, labelpad=8)
    ax_u1.set_title(r"Effective Potential Profiles Across Bead Sizes $U_{\mathrm{eff}}(\theta)$",
                    fontsize=15, fontweight='bold', pad=12)
    ax_u1.grid(True, linestyle=':', alpha=0.7)
    ax_u1.legend(loc='upper right', fontsize=10.0, framealpha=0.92)
    fig_u1.tight_layout()
    save_figure(fig_u1, "theta_effective_potential_1panel", OUT_DIRS)
    plt.close(fig_u1)

    # -------------------------------------------------------------
    # 4. パラメータ A, kappa_2, Lambda vs R_c / xi のプロット
    #    - 0.63 um: フィッティングで得られた生の A, kappa_2 を素直にプロット
    #    - 20.0 um: A=0, kappa_2=0, Lambda=0 の参考値（白抜きマーカー）
    # -------------------------------------------------------------
    # (1) 2パネル統合プロット (A & kappa_2)
    fig_p2, (ax_a, ax_k) = plt.subplots(1, 2, figsize=(15.0, 5.8))
    df_trend = df_theta_params[df_theta_params['bead_name'] != 'beads20um'].copy()

    # パネル (A): A vs Rc / xi
    ax_a.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1, label=r'Crossover $R_c / \xi = 1$')
    ax_a.plot(df_trend['rc_over_xi_mean'], df_trend['A'], color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

    for _, r_p in df_theta_params.iterrows():
        bname = r_p['bead_name']
        x_val = r_p['rc_over_xi_mean']
        x_sem = r_p['rc_over_xi_sem']
        xerr = [x_sem] if (not np.isnan(x_sem) and x_sem > 0) else None

        if bname == 'beads20um':
            y_val = 0.0
            ax_a.scatter(x_val, y_val, facecolors='white', edgecolors=r_p['color'], marker=r_p['marker'],
                         s=110, linewidths=2.0, linestyle='--', zorder=6, label=f"{r_p['label']} ($A \\to 0$, ref)")
            ax_a.annotate(r"$A \to 0$ (ref)", xy=(x_val, y_val), xytext=(0, 12), textcoords="offset points",
                          ha='center', va='bottom', fontsize=9.5, color='#555555', fontweight='bold')
            if xerr is not None:
                ax_a.errorbar(x_val, y_val, xerr=xerr, fmt='none', ecolor='#888888', elinewidth=1.6, capsize=4.0, zorder=4)
        else:
            y_val = r_p['A']
            y_err_low = max(0.0, y_val - r_p['A_ci_low'])
            y_err_high = max(0.0, r_p['A_ci_high'] - y_val)
            yerr = [[y_err_low], [y_err_high]] if np.isfinite(y_err_low) and np.isfinite(y_err_high) else None

            ax_a.errorbar(x_val, y_val, xerr=xerr, yerr=yerr, fmt='none',
                          ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)
            ax_a.scatter(x_val, y_val, color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                         linewidths=1.2, zorder=5, label=f"{r_p['label']} ({y_val:.2f})")

    ax_a.set_xscale('log')
    ax_a.set_xlim(0.04, 3.5)
    ax_a.set_ylim(-0.05, 0.8)
    ax_a.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=13, labelpad=8)
    ax_a.set_ylabel(r"Parity-Breaking Amplitude $A$ [-]", fontsize=13, labelpad=8)
    ax_a.set_title(r"(A) Parity-Breaking Torque Amplitude $A$", fontsize=14, fontweight='bold', pad=10)
    ax_a.tick_params(axis='both', labelsize=11)
    ax_a.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax_a.legend(loc='lower left', fontsize=9.2, framealpha=0.92)

    # パネル (B): kappa_2 vs Rc / xi
    ax_k.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1, label=r'Crossover $R_c / \xi = 1$')
    ax_k.plot(df_trend['rc_over_xi_mean'], df_trend['kappa_2'], color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

    for _, r_p in df_theta_params.iterrows():
        bname = r_p['bead_name']
        x_val = r_p['rc_over_xi_mean']
        x_sem = r_p['rc_over_xi_sem']
        xerr = [x_sem] if (not np.isnan(x_sem) and x_sem > 0) else None

        if bname == 'beads20um':
            y_val = 0.0
            ax_k.scatter(x_val, y_val, facecolors='white', edgecolors=r_p['color'], marker=r_p['marker'],
                         s=110, linewidths=2.0, linestyle='--', zorder=6, label=f"{r_p['label']} ($\\kappa_2 \\to 0$, ref)")
            ax_k.annotate(r"$\kappa_2 \to 0$ (ref)", xy=(x_val, y_val), xytext=(0, 12), textcoords="offset points",
                          ha='center', va='bottom', fontsize=9.5, color='#555555', fontweight='bold')
            if xerr is not None:
                ax_k.errorbar(x_val, y_val, xerr=xerr, fmt='none', ecolor='#888888', elinewidth=1.6, capsize=4.0, zorder=4)
        else:
            y_val = r_p['kappa_2']
            y_err_low = max(0.0, y_val - r_p['k2_ci_low'])
            y_err_high = max(0.0, r_p['k2_ci_high'] - y_val)
            yerr = [[y_err_low], [y_err_high]] if np.isfinite(y_err_low) and np.isfinite(y_err_high) else None

            ax_k.errorbar(x_val, y_val, xerr=xerr, yerr=yerr, fmt='none',
                          ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, capthick=1.2, zorder=4)
            ax_k.scatter(x_val, y_val, color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
                         linewidths=1.2, zorder=5, label=f"{r_p['label']} ({y_val:.2f})")

    ax_k.set_xscale('log')
    ax_k.set_xlim(0.04, 3.5)
    ax_k.set_ylim(-0.1, 1.8)
    ax_k.set_xlabel(r"Normalized Cargo Radius $R_c / \xi$ [-]", fontsize=13, labelpad=8)
    ax_k.set_ylabel(r"Dynamic Caging Strength $\kappa_2$ [-]", fontsize=13, labelpad=8)
    ax_k.set_title(r"(B) Dynamic Caging Strength $\kappa_2$", fontsize=14, fontweight='bold', pad=10)
    ax_k.tick_params(axis='both', labelsize=11)
    ax_k.grid(True, which='both', linestyle=':', color='#dddddd', alpha=0.8)
    ax_k.legend(loc='upper left', fontsize=9.2, framealpha=0.92)

    fig_p2.tight_layout()
    save_figure(fig_p2, "theta_fitted_param_A_and_kappa2_vs_rc_over_xi_2panels", OUT_DIRS)
    plt.close(fig_p2)

    # (2) 分岐パラメータ Lambda = A / (4 * kappa_2) vs R_c / xi のプロット
    lam_crit = 2.0 / (3.0 * np.sqrt(3.0)) # approx 0.3849
    fig_lam, ax_lam = plt.subplots(figsize=(8.0, 6.2))
    ax_lam.axvline(1.0, color='#666666', linestyle='--', linewidth=1.5, zorder=1, label=r'Crossover $R_c / \xi = 1$')
    ax_lam.axhline(lam_crit, color='darkred', linestyle='-.', linewidth=1.8, zorder=1,
                   label=rf'Saddle-Node Critical $\Lambda_c \approx {lam_crit:.3f}$')

    y_max_lam = 1.0
    ax_lam.set_ylim(0.0, y_max_lam)

    # トレンド線
    df_trend_lam = df_theta_params[(df_theta_params['bead_name'] != 'beads06um') & (df_theta_params['bead_name'] != 'beads20um')].copy()
    df_trend_lam['lambda'] = df_trend_lam['A'] / (4.0 * df_trend_lam['kappa_2'])
    ax_lam.plot(df_trend_lam['rc_over_xi_mean'], df_trend_lam['lambda'],
                color='#333333', linestyle='-', linewidth=2.0, alpha=0.7, zorder=2)

    for _, r_p in df_theta_params.iterrows():
        bname = r_p['bead_name']
        x_val = r_p['rc_over_xi_mean']
        x_sem = r_p['rc_over_xi_sem']
        xerr = [x_sem] if (not np.isnan(x_sem) and x_sem > 0) else None

        if bname == 'beads06um':
            ax_lam.scatter(x_val, y_max_lam, color=r_p['color'], marker='^', s=120, edgecolors='black',
                           linewidths=1.3, zorder=6, clip_on=False, label=f"{r_p['label']} ($\\Lambda \\to \\infty$)")
            ax_lam.annotate(r"$\Lambda \to \infty$" + f"\n({r_p['label']})",
                            xy=(x_val, y_max_lam), xytext=(0, 14), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10.5, color=r_p['color'], fontweight='bold',
                            arrowprops=dict(arrowstyle="->", color=r_p['color'], lw=1.5))
            if xerr is not None:
                ax_lam.errorbar(x_val, y_max_lam, xerr=xerr, fmt='none', ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, zorder=4)

        elif bname == 'beads20um':
            y_val = 0.0
            ax_lam.scatter(x_val, y_val, facecolors='white', edgecolors=r_p['color'], marker=r_p['marker'],
                           s=110, linewidths=2.0, linestyle='--', zorder=6, label=f"{r_p['label']} (limit $\\to 0$, ref)")
            ax_lam.annotate(r"$\Lambda \to 0$ (ref)", xy=(x_val, y_val), xytext=(0, 12), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9.5, color='#555555', fontweight='bold')
            if xerr is not None:
                ax_lam.errorbar(x_val, y_val, xerr=xerr, fmt='none', ecolor='#888888', elinewidth=1.6, capsize=4.0, zorder=4)

        else:
            y_val = r_p['A'] / (4.0 * r_p['kappa_2'])
            if xerr is not None:
                ax_lam.errorbar(x_val, y_val, xerr=xerr, fmt='none', ecolor=r_p['color'], elinewidth=1.6, capsize=4.0, zorder=4)
            ax_lam.scatter(x_val, y_val, color=r_p['color'], marker=r_p['marker'], s=100, edgecolors='black',
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
    save_figure(fig_lam, "theta_fitted_param_bifurcation_lambda_vs_rc_over_xi", OUT_DIRS)
    plt.close(fig_lam)

    # -------------------------------------------------------------
    # 5. theta 空間における線形安定性解析（3 パネル統合図）
    #    Panel (A): サドルノード分岐図 (theta* vs Lambda)
    #    Panel (B): 位相フロー場 F_theta(theta; Lambda)
    #    Panel (C): 線形緩和率 / 臨界減速 lambda_theta / (4*kappa_2) vs Lambda
    # -------------------------------------------------------------
    fig_bif, (ax_bifa, ax_bifb, ax_bifc) = plt.subplots(1, 3, figsize=(18.0, 5.5))

    lam_arr = np.linspace(0.0, lam_crit, 300)
    # 安定固定点・不安定固定点
    arg_lam = 3.0 * np.sqrt(3.0) / 2.0 * lam_arr
    c_star_stable = (2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_lam))
    c_star_unstable = (2.0 / np.sqrt(3.0)) * np.sin((np.pi / 3.0) - (1.0 / 3.0) * np.arcsin(arg_lam))

    # theta* = arctan2(sqrt(1-c*^2), c*)
    th_star_stable = np.arctan2(np.sqrt(np.clip(1.0 - c_star_stable**2, 0.0, 1.0)), c_star_stable)
    th_star_unstable = np.arctan2(np.sqrt(np.clip(1.0 - c_star_unstable**2, 0.0, 1.0)), c_star_unstable)

    # 固有値 lambda_theta / (4*kappa_2) = - (1 - 3*c*^2)/(1 - c*^2)
    denom_eig_s = np.clip(1.0 - c_star_stable**2, 1e-6, None)
    eig_stable = - (1.0 - 3.0 * c_star_stable**2) / denom_eig_s

    denom_eig_u = np.clip(1.0 - c_star_unstable**2, 1e-6, None)
    eig_unstable = - (1.0 - 3.0 * c_star_unstable**2) / denom_eig_u

    th_crit = np.arctan2(np.sqrt(1.0 - (1.0/3.0)), 1.0/np.sqrt(3.0)) # approx 0.9553 rad (54.7 deg)

    # ==========================================
    # Panel (A): サドルノード分岐図 (theta* vs Lambda)
    # ==========================================
    ax_bifa.plot(lam_arr, th_star_stable, color='#1b7837', linestyle='-', linewidth=2.5,
                 label=r'Stable Node $\theta_{\min}^*$ (Dynamic Cage)', zorder=3)
    ax_bifa.plot(lam_arr, th_star_unstable, color='#d95f02', linestyle='--', linewidth=2.2,
                 label=r'Unstable Saddle $\theta_{\max}^*$ (Barrier Peak)', zorder=3)
    ax_bifa.scatter(lam_crit, th_crit, color='crimson', marker='*', s=160, edgecolors='black',
                    linewidths=1.2, zorder=6, label=rf'Saddle-Node ($\Lambda_c \approx {lam_crit:.3f}$)')

    ax_bifa.axvline(lam_crit, color='crimson', linestyle=':', linewidth=1.5, zorder=1)
    ax_bifa.axvspan(lam_crit, 1.25, color='#7570b3', alpha=0.08, zorder=1)
    ax_bifa.text(0.75, 0.8, "Ballistic Phase\n(No Fixed Points,\nUnbounded Drift)",
                 fontsize=10.5, color='#7570b3', fontweight='bold', ha='center', va='center')

    # 実験データの配置
    for _, r_p in df_theta_params.iterrows():
        bname = r_p['bead_name']
        a_val = r_p['A']
        k2_val = r_p['kappa_2']
        lbl = r_p['label']
        col = r_p['color']
        mrk = r_p['marker']

        if bname == 'beads20um':
            ax_bifa.scatter(0.0, np.pi/2.0, facecolors='white', edgecolors=col, marker=mrk,
                            s=110, linewidths=2.0, linestyle='--', zorder=5, label=f"{lbl} ($\Lambda=0$, ref)")
        elif k2_val > 0.001:
            lam_exp = a_val / (4.0 * k2_val)
            if lam_exp <= lam_crit:
                arg_e = np.clip(3.0 * np.sqrt(3.0) / 2.0 * lam_exp, -1.0, 1.0)
                c_e = float((2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_e)))
                th_e = float(np.arctan2(np.sqrt(np.clip(1.0 - c_e**2, 0.0, 1.0)), c_e))
                ax_bifa.scatter(lam_exp, th_e, color=col, marker=mrk, s=100, edgecolors='black',
                                linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda={lam_exp:.2f}$)")
            else:
                ax_bifa.scatter(min(lam_exp, 1.15), 0.1, color=col, marker='>', s=110, edgecolors='black',
                                linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda={lam_exp:.2f} > \Lambda_c$)")
        else:
            ax_bifa.scatter(1.15, 0.1, color=col, marker='>', s=110, edgecolors='black',
                            linewidths=1.2, zorder=5, label=f"{lbl} ($\Lambda \\to \\infty$)")

    ax_bifa.set_xlim(-0.02, 1.25)
    ax_bifa.set_ylim(-0.05, 1.7)
    ax_bifa.set_yticks([0.0, np.pi/4, np.pi/2])
    ax_bifa.set_yticklabels([r"$0$ (Forward)", r"$\pi/4$", r"$\pi/2$ (Orthogonal)"])
    ax_bifa.set_xlabel(r"Control Parameter $\Lambda = \frac{A}{4\kappa_2}$ [-]", fontsize=13, labelpad=8)
    ax_bifa.set_ylabel(r"Fixed Point Angle $\theta^*$ [rad]", fontsize=13, labelpad=8)
    ax_bifa.set_title(r"(A) Saddle-Node Bifurcation Diagram in $\theta$", fontsize=14, fontweight='bold', pad=10)
    ax_bifa.grid(True, linestyle=':', alpha=0.7)
    ax_bifa.legend(loc='lower right', fontsize=8.8, framealpha=0.92)

    # ==========================================
    # Panel (B): 位相フロー場 F_theta(theta; Lambda)
    # ==========================================
    th_flow = np.linspace(0.05, np.pi - 0.05, 400)
    sin_flow = np.sin(th_flow)
    cos_flow = np.cos(th_flow)

    flow_configs = [
        (0.0, '#2b83ba', '-', r'$\Lambda = 0.0$ (Symmetric)'),
        (0.2, '#1b7837', '-', r'$\Lambda = 0.2 < \Lambda_c$ (Caging)'),
        (lam_crit, '#fdae61', '-.', rf'$\Lambda = \Lambda_c \approx {lam_crit:.3f}$ (Critical)'),
        (0.6, '#7570b3', '--', r'$\Lambda = 0.6 > \Lambda_c$ (Ballistic)'),
    ]

    ax_bifb.axhline(0.0, color='#666666', linestyle='-', linewidth=1.2, zorder=2)
    ax_bifb.axvline(np.pi / 2.0, color='#aaaaaa', linestyle=':', linewidth=1.0, zorder=1)

    for l_val, c_color, l_style, l_label in flow_configs:
        # F_theta(theta; Lambda) = - [ (2*Lambda / sin(theta)) - 2*sin(2*theta) ]
        f_theta = - ((2.0 * l_val / sin_flow) - 2.0 * np.sin(2.0 * th_flow))
        ax_bifb.plot(th_flow, f_theta, color=c_color, linestyle=l_style, linewidth=2.0, label=l_label, zorder=3)

        if l_val <= lam_crit:
            arg_l = np.clip(3.0 * np.sqrt(3.0) / 2.0 * l_val, -1.0, 1.0)
            c_s = (2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_l))
            th_s = np.arctan2(np.sqrt(np.clip(1.0 - c_s**2, 0.0, 1.0)), c_s)
            ax_bifb.scatter(th_s, 0.0, color=c_color, marker='o', s=60, edgecolors='black', zorder=5)

            if l_val > 0.001 and l_val < lam_crit:
                c_u = (2.0 / np.sqrt(3.0)) * np.sin((np.pi / 3.0) - (1.0 / 3.0) * np.arcsin(arg_l))
                th_u = np.arctan2(np.sqrt(np.clip(1.0 - c_u**2, 0.0, 1.0)), c_u)
                ax_bifb.scatter(th_u, 0.0, facecolors='none', edgecolors=c_color, marker='o', s=60, linewidths=1.5, zorder=5)

    # 矢印プロット (Lambda = 0.2 の代表フロー)
    # 安定固定点 th_s approx 1.360 rad, 不安定固定点 th_u approx 0.497 rad
    # th < 0.497: F < 0 (左向き <-、順流 theta=0 へ脱出)
    # 0.497 < th < 1.360: F > 0 (右向き ->、安定井戸 th_s へ引き戻される)
    # th > 1.360: F < 0 (左向き <-、安定井戸 th_s へ引き戻される)
    y_arr = 0.0
    # 区間 1: th < 0.497 (左向き)
    ax_bifb.annotate('', xy=(0.20, y_arr), xytext=(0.40, y_arr),
                     arrowprops=dict(arrowstyle="->", color='#d95f02', lw=2.0))
    # 区間 2: 0.497 < th < 1.360 (右向き)
    ax_bifb.annotate('', xy=(0.85, y_arr), xytext=(0.65, y_arr),
                     arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
    ax_bifb.annotate('', xy=(1.25, y_arr), xytext=(1.05, y_arr),
                     arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
    # 区間 3: th > 1.360 (左向き)
    ax_bifb.annotate('', xy=(1.45, y_arr), xytext=(1.65, y_arr),
                     arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))
    ax_bifb.annotate('', xy=(2.0, y_arr), xytext=(2.3, y_arr),
                     arrowprops=dict(arrowstyle="->", color='#1b7837', lw=2.0))

    ax_bifb.set_xlim(0.0, np.pi)
    ax_bifb.set_ylim(-3.5, 3.5)
    ax_bifb.set_xticks([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax_bifb.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax_bifb.set_xlabel(r"Orientation Angle $\theta$ [rad]", fontsize=13, labelpad=8)
    ax_bifb.set_ylabel(r"Phase Velocity $F_\theta(\theta; \Lambda) = \frac{\dot{\theta}}{2\kappa_2}$ [-]", fontsize=13, labelpad=8)
    ax_bifb.set_title(r"(B) Angular Phase Flow Field $F_\theta(\theta; \Lambda)$", fontsize=14, fontweight='bold', pad=10)
    ax_bifb.grid(True, linestyle=':', alpha=0.7)
    ax_bifb.legend(loc='lower left', fontsize=9.2, framealpha=0.92)

    # ==========================================
    # Panel (C): 線形緩和率と臨界減速
    # ==========================================
    ax_bifc.plot(lam_arr, eig_stable, color='#1b7837', linestyle='-', linewidth=2.5,
                 label=r'Stable Node $\lambda_\theta / (4\kappa_2) \leq 0$', zorder=3)
    ax_bifc.axhline(0.0, color='gray', linestyle=':', linewidth=1.2, zorder=2)
    ax_bifc.axvline(lam_crit, color='crimson', linestyle=':', linewidth=1.5, zorder=2)

    ax_bifc.scatter(lam_crit, 0.0, color='crimson', marker='*', s=180, edgecolors='black',
                    linewidths=1.2, zorder=6, label=rf'Critical Slowing Down ($\lambda \to 0$)')
    ax_bifc.annotate(r"Critical Slowing Down" + "\n" + r"($\lambda \to 0$ as $\Lambda \to \Lambda_c$)",
                     xy=(lam_crit, 0.0), xytext=(lam_crit - 0.15, -0.35),
                     ha='center', va='top', fontsize=9.5, color='crimson', fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='crimson', lw=1.5))

    for _, r_p in df_theta_params.iterrows():
        bname = r_p['bead_name']
        a_val = r_p['A']
        k2_val = r_p['kappa_2']
        lbl = r_p['label']
        col = r_p['color']
        mrk = r_p['marker']

        if bname == 'beads20um':
            ax_bifc.scatter(0.0, -1.0, facecolors='white', edgecolors=col, marker=mrk,
                            s=110, linewidths=2.0, linestyle='--', zorder=5, label=f"{lbl} ($\Lambda=0$, ref)")
        elif k2_val > 0.001:
            lam_exp = a_val / (4.0 * k2_val)
            if lam_exp <= lam_crit:
                arg_e = np.clip(3.0 * np.sqrt(3.0) / 2.0 * lam_exp, -1.0, 1.0)
                c_e = float((2.0 / np.sqrt(3.0)) * np.sin((1.0 / 3.0) * np.arcsin(arg_e)))
                eig_e = - (1.0 - 3.0 * c_e**2) / (1.0 - c_e**2)
                ax_bifc.scatter(lam_exp, eig_e, color=col, marker=mrk, s=100, edgecolors='black',
                                linewidths=1.2, zorder=5, label=f"{lbl} ($\lambda={eig_e:.2f}$)")

    ax_bifc.set_xlim(-0.02, 0.45)
    ax_bifc.set_ylim(-1.05, 0.15)
    ax_bifc.set_xlabel(r"Control Parameter $\Lambda = \frac{A}{4\kappa_2}$ [-]", fontsize=13, labelpad=8)
    ax_bifc.set_ylabel(r"Dimensionless Relaxation Rate $\lambda / (4\kappa_2)$ [-]", fontsize=13, labelpad=8)
    ax_bifc.set_title(r"(C) Linear Relaxation Rate & Critical Slowing", fontsize=14, fontweight='bold', pad=10)
    ax_bifc.grid(True, linestyle=':', alpha=0.7)
    ax_bifc.legend(loc='lower left', fontsize=9.0, framealpha=0.92)

    fig_bif.suptitle(r'Nonlinear Orientation Dynamics in $\theta$ & Saddle-Node Bifurcation',
                     fontsize=16, y=0.99)
    fig_bif.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig_bif, "theta_saddle_node_bifurcation_phase_dynamics_3panels", OUT_DIRS)
    plt.close(fig_bif)

    # -------------------------------------------------------------
    # 6. CSV サマリー保存
    # -------------------------------------------------------------
    for d in OUT_DIRS:
        df_theta_params.to_csv(d / "theta_fitted_params_summary.csv", index=False)
    print("Saved theta fitted parameters summary CSV.")


if __name__ == "__main__":
    main()
