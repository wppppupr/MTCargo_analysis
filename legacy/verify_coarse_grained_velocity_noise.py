#!/usr/bin/env python3
"""
verify_coarse_grained_velocity_noise.py

Claude/査読コメントへの厳密な再検証スクリプト:
1. 本論 (verify_cos_decomposition.py) と完全に同一のパイプライン
   (cutoff 適用, active_set, 同一 best_rad, lstsq 復元変位ベクトル, sqrt(1-c^2) ヤコビアン,
    P_odd/P_even 比率法) を使用し、n=1 で本論の数値が厳密に再現されることを確認した上で、
   粗視化速度 v_n(t) = [r(t+n*dt) - r(t)] / (n*dt) (n = 1, 2, 3, 5) について:
   - P_n(c) および A, kappa_2, Lambda をクラスターブートストラップ (95% CI) 付きで算出
     (kappa_2 = 0 に張り付いた場合は Lambda を下限扱い)
   - 重ならない区間 (non-overlapping intervals) またはラグ >= n*dt での自己相関関数 C(tau) を比較し、
     高周波ノイズ除去の効果を検証する
2. 連続する1フレーム変位の共分散 <dx_k dx_{k+1}> (および <dy_k dy_{k+1}>) から
   位置決め誤差の下限 sigma >= sqrt(max(0, -<dx_k dx_{k+1}>)) を算出し、
   1フレーム変位のRMSと定量比較する。
3. SI (Supporting Information) 用の図およびサマリー CSV を出力する。
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO, SCALE_UM_PER_PX,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths, find_experiment_dirs,
    calc_disk_polar_order
)

try:
    from libs.experiment_cutoff import get_cutoff_frame
except ImportError:
    from experiment_cutoff import get_cutoff_frame

OUT_DIRS = [
    CURRENT_DIR / "figure" / "cos_decomposition",
    ROOT_DIR / "figure" / "cos_decomposition"
]

N_STEPS_LIST = [1, 2, 3, 5]
DT_SEC = 4.0  # 1 frame = 4.0 s


def load_exact_pipeline_frames(root_dir: Path, active_set: set, xi_dict: Dict[str, float]) -> pd.DataFrame:
    """
    verify_cos_decomposition.py の load_all_data と完全に同一のデータ抽出ロジックで、
    各フレームの真の1フレーム変位 (cdx, cdy), 周囲フロー (u, v), cos_i (n=1), cos_bulk (n=1),
    および有効な局所ポーラーオーダー Phi_eff を返す。
    """
    valid_xis = [v for v in xi_dict.values() if np.isfinite(v) and v > 0]
    default_xi = float(np.mean(valid_xis)) if valid_xis else 6.0

    records = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        r_um = b_item['radius_um']
        target_rad_px = r_um / SCALE_UM_PER_PX
        exp_dirs = find_experiment_dirs(root_dir, bname)

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            csv_std = exp_dir / "beads_flow_interaction.csv"
            csv_gau = exp_dir / "beads_flow_interaction_gaussian.csv"
            tracks_p = exp_dir / "beads_tracks.csv"

            if csv_std.exists():
                csv_p, rad_col = csv_std, "radius"
            elif csv_gau.exists():
                csv_p, rad_col = csv_gau, "sigma"
            else:
                continue
            if not tracks_p.exists():
                continue

            try:
                df_flow = pd.read_csv(csv_p)
                tdf = pd.read_csv(tracks_p)
            except Exception:
                continue

            cutoff = get_cutoff_frame(exp_name)
            if cutoff is not None:
                df_flow = df_flow[df_flow['frame'] <= cutoff]
                tdf = tdf[tdf['frame'] <= cutoff]

            # 1. beads_flow_interaction.csv から lstsq で真の変位 (bdx, bdy) を復元
            cargo_velocity_map = {}
            for (frm, pid), grp_fp in df_flow.groupby(['frame', 'particle']):
                grp_v = grp_fp[(grp_fp['mean_flow_u'] != 0) | (grp_fp['mean_flow_v'] != 0)]
                if len(grp_v) >= 2:
                    u_vals = grp_v['mean_flow_u'].values
                    v_vals = grp_v['mean_flow_v'].values
                    d_vals = grp_v['dot_product'].values
                    A_mat = np.column_stack([u_vals, v_vals])
                    res, _, _, _ = np.linalg.lstsq(A_mat, d_vals, rcond=None)
                    bdx, bdy = float(res[0]), float(res[1])
                    b_spd = float(np.hypot(bdx, bdy))
                    if b_spd > 0:
                        cargo_velocity_map[(int(pid), int(frm))] = (bdx, bdy, b_spd)

            # 2. フォールバック用の tracks.csv 速度マップ
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

            # 3. local_polar_w.zarr のロード
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
                            for p_val in da_polar['particle'].values:
                                p_da = da_polar.sel(particle=p_val)
                                arr = p_da.values
                                if list(p_da.dims).index(w_dim) == 1:
                                    arr = arr.T
                                polar_cache[int(p_val)] = {
                                    'arr': arr,
                                    'f_map': {int(f): i for i, f in enumerate(z_frames)},
                                    'w_map': {float(w): i for i, w in enumerate(z_windows)},
                                    'windows': z_windows
                                }
                except Exception:
                    pass

            df_flow['rad_px'] = df_flow[rad_col].astype(float)
            valid_mask = (df_flow['mean_flow_u'] != 0) | (df_flow['mean_flow_v'] != 0) | (df_flow['cos_sim'] != 0)
            valid_rads = np.sort(df_flow.loc[valid_mask, 'rad_px'].unique())
            if len(valid_rads) == 0:
                continue

            best_idx = int(np.argmin(np.abs(valid_rads - target_rad_px)))
            best_rad = float(valid_rads[best_idx])
            xi_val = xi_dict.get(exp_name, default_xi)
            phi_disk_fallback = calc_disk_polar_order(best_rad * SCALE_UM_PER_PX, xi_val)

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

                cos_i = float(r['cos_sim'])
                cos_bulk = (cdx * u + cdy * v) / (cspd * v_flow_norm)

                phi_inst = np.nan
                if pid in polar_cache:
                    pc = polar_cache[pid]
                    f_idx = pc['f_map'].get(frm)
                    if f_idx is not None:
                        if best_rad in pc['w_map']:
                            val = pc['arr'][pc['w_map'][best_rad], f_idx]
                            if np.isfinite(val) and 0.0 <= val <= 1.5:
                                phi_inst = float(np.clip(val, 0.0, 1.0))
                        if np.isnan(phi_inst):
                            for w_cand in sorted(pc['windows']):
                                if w_cand >= best_rad:
                                    val = pc['arr'][pc['w_map'][float(w_cand)], f_idx]
                                    if np.isfinite(val) and 0.0 <= val <= 1.5:
                                        phi_inst = float(np.clip(val, 0.0, 1.0))
                                        break
                if np.isnan(phi_inst):
                    phi_inst = phi_disk_fallback

                records.append({
                    'bead_name': bname,
                    'exp_dir': exp_name,
                    'particle': pid,
                    'cargo_id': f"{exp_name}_{pid}",
                    'frame': frm,
                    'cdx_px': cdx,
                    'cdy_px': cdy,
                    'u': u,
                    'v': v,
                    'cos_i_n1': cos_i,
                    'cos_bulk_n1': cos_bulk,
                    'Phi_inst': phi_inst,
                })

    return pd.DataFrame(records)


def build_coarse_grained_dataset(df_frames: pd.DataFrame, n_step: int) -> pd.DataFrame:
    """
    n_step == 1 の場合は本論と 100% 同一の cos_i_n1 を使用。
    n_step > 1 の場合は、連続フレーム t, t+1, ..., t+n_step-1 の 1フレーム変位を積算して
    nフレーム変位 (sum cdx, sum cdy) を構成し、
    cos_i_n(t) = Phi_eff(t) * cos_bulk_n(t) を算出する。
    (Phi_eff は |cos_bulk_n1| > 0.1 のときは cos_i_n1 / cos_bulk_n1 との整合性、または Phi_inst を使用)
    """
    if n_step == 1:
        df_out = df_frames[['bead_name', 'cargo_id', 'frame']].copy()
        df_out['c_val'] = df_frames['cos_i_n1'].values
        return df_out

    rows = []
    for cid, grp in df_frames.groupby('cargo_id'):
        grp = grp.sort_values('frame')
        frames = grp['frame'].values
        dx = grp['cdx_px'].values
        dy = grp['cdy_px'].values
        u = grp['u'].values
        v = grp['v'].values
        ci1 = grp['cos_i_n1'].values
        cb1 = grp['cos_bulk_n1'].values
        phi = grp['Phi_inst'].values
        bname = grp['bead_name'].iloc[0]

        f_to_idx = {int(f): idx for idx, f in enumerate(frames)}
        for idx, f in enumerate(frames):
            # 連続する n_step フレームが揃っているか確認
            indices = [f_to_idx.get(int(f + k)) for k in range(n_step)]
            if any(ix is None for ix in indices):
                continue
            dx_n = sum(dx[ix] for ix in indices)
            dy_n = sum(dy[ix] for ix in indices)
            spd_n = np.hypot(dx_n, dy_n)
            unorm = np.hypot(u[idx], v[idx])
            if spd_n <= 1e-8 or unorm <= 1e-8:
                continue
            cb_n = (dx_n * u[idx] + dy_n * v[idx]) / (spd_n * unorm)

            # Phi_eff の取得: n=1 で cos_i_n1 = Phi_eff * cos_bulk_n1 が成り立つ場合はその比を優先
            if abs(cb1[idx]) > 0.15:
                phi_eff = np.clip(ci1[idx] / cb1[idx], 0.0, 1.0)
            else:
                phi_eff = phi[idx]

            c_n = float(np.clip(phi_eff * cb_n, -1.0, 1.0))
            rows.append({
                'bead_name': bname,
                'cargo_id': cid,
                'frame': int(f),
                'c_val': c_n,
            })
    return pd.DataFrame(rows)


def fit_atanh_params(c_vals: np.ndarray) -> Tuple[float, float, float]:
    """
    verify_cos_decomposition.py と完全に同一の推定法:
    - 28 ビン (-1 to 1)
    - 0.05 < c <= 0.85, P_even > 1e-4
    - ヤコビアン sqrt(1 - c^2)
    """
    bins_cos = np.linspace(-1.0, 1.0, 29)
    bin_centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])
    denom_jacob = np.sqrt(np.clip(1.0 - bin_centers**2, 1e-6, 1.0))

    P_obs, _ = np.histogram(c_vals, bins=bins_cos, density=True)
    P_neg = P_obs[::-1]
    pe = 0.5 * (P_obs + P_neg)
    po = 0.5 * (P_obs - P_neg)

    mask = (bin_centers > 0.05) & (bin_centers <= 0.85) & (pe > 1e-4)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan

    c_m = bin_centers[mask]
    ratio_m = np.clip((po / pe)[mask], -0.999, 0.999)
    atanh_ratio = np.arctanh(ratio_m)
    atanh_c = np.arctanh(c_m)
    A_val = max(0.0, float(np.sum(atanh_ratio * atanh_c) / np.sum(atanh_c**2)))

    y_vals = np.log(np.clip(denom_jacob[mask] * pe[mask] / np.cosh(A_val * atanh_c), 1e-6, None))
    x_sq = c_m**2
    x_dev = x_sq - np.mean(x_sq)
    y_dev = y_vals - np.mean(y_vals)
    slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2) if np.sum(x_dev**2) > 0 else 0.0
    k2_val = max(0.0, float(-0.5 * slope))
    lam_val = A_val / (4.0 * k2_val) if k2_val > 1e-4 else np.inf
    return A_val, k2_val, lam_val


def bootstrap_atanh_params(df_c: pd.DataFrame, B: int = 500) -> Dict:
    """
    粒子 (cargo_id) 単位のクラスターブートストラップによる A, kappa_2, Lambda の 95% CI 算出。
    kappa_2 == 0 に張り付いた試行は Lambda の下限算出に配慮する。
    """
    c_all = df_c['c_val'].dropna().values
    A_obs, k2_obs, lam_obs = fit_atanh_params(c_all)

    cargos_data = [grp['c_val'].dropna().values for _, grp in df_c.groupby('cargo_id') if len(grp) > 0]
    n_cargos = len(cargos_data)
    if n_cargos < 2:
        return {
            'A': A_obs, 'A_low': A_obs, 'A_high': A_obs,
            'k2': k2_obs, 'k2_low': k2_obs, 'k2_high': k2_obs,
            'lam': lam_obs, 'lam_low': lam_obs, 'lam_high': lam_obs,
            'is_lower_bound': np.isinf(lam_obs),
        }

    rng = np.random.default_rng(42)
    b_A, b_k2, b_lam = [], [], []
    for _ in range(B):
        sel = rng.choice(n_cargos, size=n_cargos, replace=True)
        pts = np.concatenate([cargos_data[i] for i in sel])
        a_b, k2_b, lam_b = fit_atanh_params(pts)
        if np.isfinite(a_b):
            b_A.append(a_b)
        if np.isfinite(k2_b):
            b_k2.append(k2_b)
        b_lam.append(lam_b)

    A_low, A_high = np.percentile(b_A, [2.5, 97.5]) if b_A else (A_obs, A_obs)
    k2_low, k2_high = np.percentile(b_k2, [2.5, 97.5]) if b_k2 else (k2_obs, k2_obs)

    # kappa_2 == 0 に張り付いた試行 (lam = inf) が多い場合、または観測値が inf の場合は下限扱い
    b_lam_arr = np.array(b_lam)
    finite_lams = b_lam_arr[np.isfinite(b_lam_arr)]
    if np.isinf(lam_obs) or np.mean(np.isinf(b_lam_arr)) > 0.05:
        lam_low = float(np.percentile(b_lam_arr, 2.5)) if np.isfinite(np.percentile(b_lam_arr, 2.5)) else (float(np.min(finite_lams)) if len(finite_lams) > 0 else np.inf)
        lam_high = np.inf
        is_lb = True
    else:
        lam_low, lam_high = np.percentile(finite_lams, [2.5, 97.5])
        is_lb = False

    return {
        'A': A_obs, 'A_low': float(A_low), 'A_high': float(A_high),
        'k2': k2_obs, 'k2_low': float(k2_low), 'k2_high': float(k2_high),
        'lam': lam_obs, 'lam_low': float(lam_low), 'lam_high': float(lam_high),
        'is_lower_bound': is_lb,
    }


def compute_nonoverlapping_autocorr(df_c: pd.DataFrame, n_step: int, max_lag_blocks: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    重ならない区間 (frame % n_step == offset ごとにブロック分割) で速度をサンプリングし、
    ラグ k * (n_step * dt) における自己相関関数 C(tau) を計算する。
    これにより隣接ブロック間で変位の共有が一切起きない。
    """
    lags_s = np.arange(0, max_lag_blocks + 1) * (n_step * DT_SEC)
    acf_sum = np.zeros(len(lags_s))
    acf_cnt = np.zeros(len(lags_s))

    for _, grp in df_c.groupby('cargo_id'):
        grp = grp.sort_values('frame')
        frames = grp['frame'].values
        vals = grp['c_val'].values
        if len(vals) < 10:
            continue
        mean_c = np.mean(vals)
        var_c = np.mean((vals - mean_c)**2)
        if var_c <= 1e-8:
            continue
        f_map = {int(f): (vals[i] - mean_c) for i, f in enumerate(frames)}

        # 重ならないラグ: k * n_step フレーム (tau = k * n_step * DT_SEC)
        for k_idx in range(len(lags_s)):
            lag_frm = k_idx * n_step
            prods = []
            for f in frames:
                if (f + lag_frm) in f_map:
                    prods.append(f_map[f] * f_map[f + lag_frm])
            if len(prods) >= 5:
                acf_sum[k_idx] += np.mean(prods) / var_c
                acf_cnt[k_idx] += 1

    acf_mean = np.where(acf_cnt > 0, acf_sum / acf_cnt, np.nan)
    return lags_s, acf_mean


def estimate_localization_error_covariance(df_frames: pd.DataFrame) -> pd.DataFrame:
    """
    連続する1フレーム変位の共分散 <dx_k dx_{k+1}> (および <dy_k dy_{k+1}>) を計算し、
    位置決め誤差の下限:
      sigma^2 >= - <dx_k dx_{k+1}>
    を求める。また 1フレーム変位の RMS と比較する。
    """
    rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_frames[df_frames['bead_name'] == bname]
        cov_x_list, cov_y_list = [], []
        disp_sq_list = []

        for _, grp in sub.groupby('cargo_id'):
            grp = grp.sort_values('frame')
            fs = grp['frame'].values
            dx = grp['cdx_px'].values * SCALE_UM_PER_PX
            dy = grp['cdy_px'].values * SCALE_UM_PER_PX
            disp_sq_list.extend((dx**2 + dy**2).tolist())
            f_map = {int(f): (dx[i], dy[i]) for i, f in enumerate(fs)}
            for f in fs:
                if (f + 1) in f_map:
                    dx0, dy0 = f_map[f]
                    dx1, dy1 = f_map[f + 1]
                    cov_x_list.append(dx0 * dx1)
                    cov_y_list.append(dy0 * dy1)

        cov_x = float(np.mean(cov_x_list)) if cov_x_list else np.nan
        cov_y = float(np.mean(cov_y_list)) if cov_y_list else np.nan
        cov_iso = 0.5 * (cov_x + cov_y)
        sigma_lb = float(np.sqrt(max(0.0, -cov_iso)))
        rms_disp = float(np.sqrt(np.mean(disp_sq_list))) if disp_sq_list else np.nan
        rows.append({
            'bead_name': bname,
            'label': b_item['label'],
            'n_pairs': len(cov_x_list),
            'cov_dx_um2': cov_x,
            'cov_dy_um2': cov_y,
            'cov_iso_um2': cov_iso,
            'sigma_lower_bound_um': sigma_lb,
            'rms_1frame_disp_um': rms_disp,
            'sigma_lb_over_disp': sigma_lb / rms_disp if rms_disp > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    setup_matplotlib()
    print("=== Rigorous Coarse-Grained Velocity & Localization Noise Analysis ===")
    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)
    df_frames = load_exact_pipeline_frames(ROOT_DIR, active_set, xi_dict)
    print(f"Loaded {len(df_frames)} frames (exact match to verify_cos_decomposition.py).")

    # 1. 位置決め誤差の下限見積もり
    df_sigma = estimate_localization_error_covariance(df_frames)
    print("\n=== Localization Error Lower Bound from Displacement Covariance <dx_k dx_{k+1}> ===")
    print(df_sigma.to_string(index=False))
    for d in OUT_DIRS:
        df_sigma.to_csv(d / "si_localization_error_covariance.csv", index=False)

    # 2. n = 1, 2, 3, 5 に対する P_n(c), A, kappa_2, Lambda (95% CI) と非重複自己相関
    summary_rows = []
    acf_results = {}  # (bname, n_step) -> (lags_s, acf_mean)

    for n_step in N_STEPS_LIST:
        df_c = build_coarse_grained_dataset(df_frames, n_step)
        print(f"\n--- n_step = {n_step} (dt = {n_step*DT_SEC:.1f} s), total points = {len(df_c)} ---")
        for b_item in BEADS_INFO:
            bname = b_item['name']
            sub = df_c[df_c['bead_name'] == bname]
            if sub.empty:
                continue
            stats = bootstrap_atanh_params(sub, B=500)
            lags_s, acf_m = compute_nonoverlapping_autocorr(sub, n_step, max_lag_blocks=10)
            acf_results[(bname, n_step)] = (lags_s, acf_m)

            # tau = 20 s (最小共通ラグ付近) での自己相関値などを記録
            c_tau1 = float(acf_m[1]) if len(acf_m) > 1 else np.nan
            summary_rows.append({
                'bead_name': bname,
                'label': b_item['label'],
                'n_step': n_step,
                'dt_s': n_step * DT_SEC,
                'n_points': len(sub),
                'A': stats['A'],
                'A_ci_low': stats['A_low'],
                'A_ci_high': stats['A_high'],
                'kappa_2': stats['k2'],
                'k2_ci_low': stats['k2_low'],
                'k2_ci_high': stats['k2_high'],
                'Lambda': stats['lam'],
                'Lambda_ci_low': stats['lam_low'],
                'Lambda_ci_high': stats['lam_high'],
                'Lambda_is_lower_bound': stats['is_lower_bound'],
                'C_first_nonoverlap_lag': c_tau1,
            })

    df_sum = pd.DataFrame(summary_rows)
    for d in OUT_DIRS:
        df_sum.to_csv(d / "si_coarse_grained_velocity_stability_summary.csv", index=False)
    print("\n=== Coarse-Grained Parameters Summary (with 95% CI) ===")
    print(df_sum[['bead_name', 'n_step', 'n_points', 'A', 'kappa_2', 'Lambda', 'Lambda_ci_low', 'Lambda_is_lower_bound']].to_string(index=False))

    # ---------------------------------------------------------
    # SI 用 4パネル統合図の作成
    # Panel (A): 非重複自己相関 C(tau) の比較 (代表例: 3.37 um & 7.24 um での n=1 vs n=5, または第1ラグ値)
    # Panel (B): Parity-Breaking Amplitude A vs n_step (95% CI付き)
    # Panel (C): Dynamic Caging Strength kappa_2 vs n_step (95% CI付き)
    # Panel (D): Bifurcation Parameter Lambda vs n_step (95% CI付き, 下限矢印表示)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    ax_acf, ax_a, ax_k, ax_l = axes.flatten()

    # Panel (A): C(tau) at non-overlapping lags for representative sizes (1.18 um, 3.37 um, 7.24 um)
    rep_beads = ['beads1um', 'beads3um', 'beads7um']
    styles_n = {1: (':', 'o', 0.6), 3: ('--', 's', 0.8), 5: ('-', '^', 1.0)}
    for b_item in BEADS_INFO:
        bname = b_item['name']
        if bname not in rep_beads:
            continue
        col = b_item['color']
        lbl = b_item['label']
        for n_s in [1, 5]:
            ls, mrk, alp = styles_n[n_s]
            lags_s, acf_m = acf_results[(bname, n_s)]
            valid = (lags_s <= 60.0) & np.isfinite(acf_m)
            ax_acf.plot(lags_s[valid], acf_m[valid], color=col, linestyle=ls, marker=mrk,
                        markersize=5.5, linewidth=1.8, alpha=alp,
                        label=f"{lbl} ($n={n_s}$)")

    ax_acf.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)
    ax_acf.set_title(r"(A) Non-Overlapping Autocorrelation $C(\tau)$ ($\tau \geq n\Delta t$)", fontsize=13.5, fontweight='bold')
    ax_acf.set_xlabel(r"Lag Time $\tau$ [s]", fontsize=12.5)
    ax_acf.set_ylabel(r"Autocorrelation $C(\tau)$ [-]", fontsize=12.5)
    ax_acf.set_xlim(-2, 62)
    ax_acf.set_ylim(-0.15, 1.05)
    ax_acf.grid(True, linestyle=':', alpha=0.7)
    ax_acf.legend(loc='upper right', fontsize=8.8, ncol=2, framealpha=0.92)

    # Panel (B), (C), (D): A, kappa_2, Lambda vs n_step (95% CI付き)
    x_offsets = np.linspace(-0.12, 0.12, len(BEADS_INFO))
    for idx_b, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        col = b_item['color']
        mrk = b_item['marker']
        lbl = b_item['label']
        sub = df_sum[df_sum['bead_name'] == bname].sort_values('n_step')
        if sub.empty:
            continue
        ns = sub['n_step'].values + x_offsets[idx_b]

        # A
        yerr_a = [np.maximum(0.0, sub['A'].values - sub['A_ci_low'].values),
                  np.maximum(0.0, sub['A_ci_high'].values - sub['A'].values)]
        ax_a.errorbar(ns, sub['A'].values, yerr=yerr_a, color=col, marker=mrk, markersize=7,
                      linewidth=1.8, capsize=3.5, label=lbl)

        # kappa_2
        yerr_k = [np.maximum(0.0, sub['kappa_2'].values - sub['k2_ci_low'].values),
                  np.maximum(0.0, sub['k2_ci_high'].values - sub['kappa_2'].values)]
        ax_k.errorbar(ns, sub['kappa_2'].values, yerr=yerr_k, color=col, marker=mrk, markersize=7,
                      linewidth=1.8, capsize=3.5, label=lbl)

        # Lambda
        for _, r_l in sub.iterrows():
            x_pt = r_l['n_step'] + x_offsets[idx_b]
            if r_l['Lambda_is_lower_bound']:
                # 下限 (Lambda_ci_low から上向き矢印)
                lb_val = min(r_l['Lambda_ci_low'], 1.8) if np.isfinite(r_l['Lambda_ci_low']) else 1.8
                ax_l.scatter(x_pt, lb_val, color=col, marker='^', s=65, edgecolors='black', zorder=5)
                ax_l.annotate('', xy=(x_pt, lb_val + 0.25), xytext=(x_pt, lb_val),
                              arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
            else:
                val_l = r_l['Lambda']
                err_low = max(0.0, val_l - r_l['Lambda_ci_low'])
                err_high = max(0.0, r_l['Lambda_ci_high'] - val_l)
                ax_l.errorbar(x_pt, val_l, yerr=[[err_low], [err_high]], color=col, marker=mrk,
                              markersize=7, capsize=3.5, zorder=5)
        # 有限値点を線で繋ぐ
        fin_mask = ~sub['Lambda_is_lower_bound'].values
        if np.sum(fin_mask) >= 2:
            ax_l.plot(ns[fin_mask], sub['Lambda'].values[fin_mask], color=col, linewidth=1.6, alpha=0.8, label=lbl)
        else:
            ax_l.plot([], [], color=col, marker=mrk, linewidth=1.6, label=f"{lbl} (lower bound)")

    ax_a.set_title(r"(B) Parity-Breaking Amplitude $A$ (95% CI)", fontsize=13.5, fontweight='bold')
    ax_a.set_xlabel(r"Coarse-Graining Step $n$ ($\Delta t_n = n \times 4\,\mathrm{s}$)", fontsize=12.5)
    ax_a.set_ylabel(r"Amplitude $A$ [-]", fontsize=12.5)
    ax_a.set_xticks(N_STEPS_LIST)
    ax_a.set_ylim(0.0, 0.9)
    ax_a.grid(True, linestyle=':', alpha=0.7)
    ax_a.legend(loc='upper right', fontsize=8.8, ncol=2, framealpha=0.92)

    ax_k.set_title(r"(C) Dynamic Caging Strength $\kappa_2$ (95% CI)", fontsize=13.5, fontweight='bold')
    ax_k.set_xlabel(r"Coarse-Graining Step $n$ ($\Delta t_n = n \times 4\,\mathrm{s}$)", fontsize=12.5)
    ax_k.set_ylabel(r"Caging Strength $\kappa_2$ [-]", fontsize=12.5)
    ax_k.set_xticks(N_STEPS_LIST)
    ax_k.set_ylim(-0.05, 1.8)
    ax_k.grid(True, linestyle=':', alpha=0.7)

    lam_c = 2.0 / (3.0 * np.sqrt(3.0))
    ax_l.axhline(lam_c, color='darkred', linestyle='-.', linewidth=1.8, label=rf'Critical $\Lambda_c \approx {lam_c:.3f}$')
    ax_l.set_title(r"(D) Bifurcation Parameter $\Lambda = A/(4\kappa_2)$ (95% CI)", fontsize=13.5, fontweight='bold')
    ax_l.set_xlabel(r"Coarse-Graining Step $n$ ($\Delta t_n = n \times 4\,\mathrm{s}$)", fontsize=12.5)
    ax_l.set_ylabel(r"Bifurcation Parameter $\Lambda$ [-]", fontsize=12.5)
    ax_l.set_xticks(N_STEPS_LIST)
    ax_l.set_ylim(0.0, 2.3)
    ax_l.grid(True, linestyle=':', alpha=0.7)
    ax_l.legend(loc='upper right', fontsize=8.5, ncol=2, framealpha=0.92)

    fig.suptitle(r"SI Robustness Check: Velocity Coarse-Graining ($n = 1, 2, 3, 5$) on Exact Main-Text Pipeline",
                 fontsize=15.0, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "si_coarse_grained_velocity_noise_robustness", OUT_DIRS)
    plt.close(fig)


if __name__ == "__main__":
    main()
