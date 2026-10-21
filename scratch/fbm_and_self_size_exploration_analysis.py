#!/usr/bin/env python3
"""
fbm_and_self_size_exploration_analysis.py

各カーゴ微粒子がフラクショナルブラウニアン運動（fBm）に従うかどうかの厳密な診断テスト
（増分ガウス性、非ガウス性パラメータ、増分自己相関関数とfBm理論値の比較）、
およびfBmモデルによるMSDフィッティングと自己サイズ探索時間（半径 Rc / 直径 2Rc）
の算出・集計・可視化スクリプト。
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.interpolate import interp1d

# プロジェクトルートとライブラリ
workspace_dir = Path(__file__).parent.parent.resolve()
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

import MSD
from libs import cal_vel as cv
from libs import displacement as dpm

# スタイルの適用
style_path = workspace_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
try:
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
except Exception:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "intervals": [4, 4, 4, 4], "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "intervals": [4, 4, 4, 4], "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "intervals": [4, 4, 4, 4], "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "intervals": [4, 4, 4, 4], "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "intervals": [4, 4, 4],    "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "intervals": [4, 4, 4],    "marker": "s", "color": style_colors[5]},
]

def autocorr(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    xc = x - np.mean(x)
    var = np.var(xc)
    if var == 0 or len(xc) <= max_lag:
        return np.full(max_lag, np.nan)
    n = len(xc)
    res = []
    for l in range(1, max_lag + 1):
        if n - l > 0:
            res.append(np.mean(xc[:n-l] * xc[l:]) / var)
        else:
            res.append(np.nan)
    return np.array(res)

def calc_empirical_tau_self(dt_arr: np.ndarray, msd_arr: np.ndarray, target_msd: float) -> float:
    """実測MSDから target_msd と交差するラグタイムを対数線形補間により求める"""
    valid = (dt_arr > 0) & (msd_arr > 0)
    dt_v = dt_arr[valid]
    msd_v = msd_arr[valid]
    if len(dt_v) < 2:
        return np.nan
    
    # 昇順ソート
    order = np.argsort(dt_v)
    dt_v = dt_v[order]
    msd_v = msd_v[order]
    
    # 探索範囲内にあるか
    if target_msd < msd_v[0]:
        # 外挿（最初の2点から）
        log_dt = np.log10(dt_v[:2])
        log_m = np.log10(msd_v[:2])
        slope = (log_m[1] - log_m[0]) / (log_dt[1] - log_dt[0])
        if slope > 0:
            log_tau = log_dt[0] + (np.log10(target_msd) - log_m[0]) / slope
            return float(10.0 ** log_tau)
        return np.nan
    elif target_msd > msd_v[-1]:
        # 外挿（最後の2点から）
        log_dt = np.log10(dt_v[-2:])
        log_m = np.log10(msd_v[-2:])
        slope = (log_m[1] - log_m[0]) / (log_dt[1] - log_dt[0])
        if slope > 0:
            log_tau = log_dt[1] + (np.log10(target_msd) - log_m[1]) / slope
            return float(10.0 ** log_tau)
        return np.nan
    else:
        # 内挿（交差区間を見つける）
        idx = np.where(msd_v >= target_msd)[0][0]
        if idx == 0:
            return float(dt_v[0])
        log_dt0, log_dt1 = np.log10(dt_v[idx-1]), np.log10(dt_v[idx])
        log_m0, log_m1 = np.log10(msd_v[idx-1]), np.log10(msd_v[idx])
        if log_m1 == log_m0:
            return float(dt_v[idx-1])
        log_tau = log_dt0 + (np.log10(target_msd) - log_m0) * (log_dt1 - log_dt0) / (log_m1 - log_m0)
        return float(10.0 ** log_tau)

def main():
    root_dir = MSD.find_default_root()
    out_dir = workspace_dir / "figure" / "msd"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data root: {root_dir}")
    print("Beginning Fractional Brownian Motion & Self-size exploration analysis across all particles...")
    
    particle_records = []
    
    # 各条件のループ
    for b_info in BEADS_INFO:
        b_name = b_info["name"]
        d_um = b_info["diameter_um"]
        r_c = b_info["radius_um"]
        intervals = b_info["intervals"]
        color = b_info["color"]
        marker = b_info["marker"]
        
        folder = root_dir / b_name
        file_list = sorted(glob.glob(str(folder / "*" / "*" / "beads_tracks.csv")))
        
        # MSD プール取得 (MSD.pyと同じフィルタリング alpha > 0.5)
        pool_df, raw_df, n_tot, n_filt = MSD.concat_pooled_particles_MSD(folder, intervals, scale=0.11, alpha_threshold=0.5)
        
        # トラックファイルのキャッシュ
        tracks_by_vid = {}
        for vid_idx, fpath in enumerate(file_list):
            interval = intervals[vid_idx] if vid_idx < len(intervals) else intervals[-1]
            t_df = cv.cal(pd.read_csv(fpath), scale=0.11, frame_interval=interval)
            tracks_by_vid[vid_idx] = t_df
            
        # 各個別粒子についての解析
        for pid, grp in pool_df.groupby('unique_particle_id'):
            v_idx = int(grp['video_idx'].iloc[0])
            orig_p_num = int(grp['particle'].iloc[0])
            
            # --- 1. fBm Power-Law フィッティング (MSD = K_alpha * t^alpha) ---
            # 適合ラグ時間: 4s ~ 300s (十分なサンプル数がある区間)
            mask_fit = (grp['lag time'] >= 4.0) & (grp['lag time'] <= 300.0) & (grp['MSD'] > 0)
            if np.sum(mask_fit) < 3:
                mask_fit = (grp['MSD'] > 0)
            if np.sum(mask_fit) < 3:
                continue
                
            t_pts = grp.loc[mask_fit, 'lag time'].values
            msd_pts = grp.loc[mask_fit, 'MSD'].values
            log_t = np.log10(t_pts)
            log_m = np.log10(msd_pts)
            
            slope, intercept = np.polyfit(log_t, log_m, 1)
            alpha = float(slope)
            Hurst = alpha / 2.0
            log_K = float(intercept)
            K_alpha = 10.0 ** log_K
            D_alpha = K_alpha / 4.0  # 2D generalized diffusion coeff
            
            # 適合度 R^2
            fitted_log_m = intercept + slope * log_t
            ss_tot = np.sum((log_m - np.mean(log_m))**2)
            ss_res = np.sum((log_m - fitted_log_m)**2)
            r2_fbm = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            # --- 2. 自己サイズ探索時間の計算 ---
            # (A) fBm モデル理論値
            # MSD(tau) = K_alpha * tau^alpha = R_c^2 or (2R_c)^2
            if alpha > 0 and K_alpha > 0:
                tau_self_rc_fbm = float((r_c**2 / K_alpha) ** (1.0 / alpha))
                tau_self_2rc_fbm = float(((2.0 * r_c)**2 / K_alpha) ** (1.0 / alpha))
            else:
                tau_self_rc_fbm = np.nan
                tau_self_2rc_fbm = np.nan
                
            # (B) 実測MSDデータからの直接内挿値
            all_dt = grp['lag time'].values
            all_msd = grp['MSD'].values
            tau_self_rc_emp = calc_empirical_tau_self(all_dt, all_msd, target_msd=r_c**2)
            tau_self_2rc_emp = calc_empirical_tau_self(all_dt, all_msd, target_msd=(2.0 * r_c)**2)
            
            # --- 3. 軌道データからの増分統計と fBm 適合性診断 ---
            kurt_x = np.nan
            kurt_y = np.nan
            beta_ng = np.nan
            shapiro_p = np.nan
            rho_emp_1 = np.nan
            rho_emp_2 = np.nan
            rho_fbm_1 = np.nan
            rho_fbm_2 = np.nan
            rho_mse_1to5 = np.nan
            
            if v_idx in tracks_by_vid:
                trk_df = tracks_by_vid[v_idx]
                p_trk = trk_df[trk_df['particle'] == orig_p_num].sort_values('frame')
                if len(p_trk) >= 10:
                    dx = np.diff(p_trk['x'].values)
                    dy = np.diff(p_trk['y'].values)
                    dx_c = dx - np.mean(dx)
                    dy_c = dy - np.mean(dy)
                    
                    # 尖度 (Kurtosis)
                    kurt_x = float(stats.kurtosis(dx, fisher=True))
                    kurt_y = float(stats.kurtosis(dy, fisher=True))
                    
                    # 2D 非ガウス性パラメータ beta = <dr^4> / (2 <dr^2>^2) - 1 in 2D
                    dr2 = dx_c**2 + dy_c**2
                    dr4 = dr2**2
                    mean_dr2 = np.mean(dr2)
                    if mean_dr2 > 0:
                        # 2Dガウス過程なら <dr^4> = 2 <dr^2>^2
                        beta_ng = float(np.mean(dr4) / (2.0 * (mean_dr2**2)) - 1.0)
                        
                    # Shapiro-Wilk 正規性検定 (サンプル数 <= 5000)
                    sample_dx = dx[:min(len(dx), 500)]
                    if len(sample_dx) >= 8:
                        shapiro_p = float(stats.shapiro(sample_dx)[1])
                        
                    # 増分自己相関関数 rho(n)
                    ac_x = autocorr(dx, max_lag=5)
                    ac_y = autocorr(dy, max_lag=5)
                    rho_emp = 0.5 * (ac_x + ac_y)
                    rho_emp_1 = float(rho_emp[0]) if len(rho_emp) > 0 else np.nan
                    rho_emp_2 = float(rho_emp[1]) if len(rho_emp) > 1 else np.nan
                    
                    # fBm 理論自己相関: rho_fbm(n) = 0.5 * [(n+1)^alpha - 2 n^alpha + (n-1)^alpha]
                    n_arr = np.arange(1, 6)
                    rho_fbm = 0.5 * ((n_arr + 1.0)**alpha - 2.0 * (n_arr**alpha) + (n_arr - 1.0)**alpha)
                    rho_fbm_1 = float(rho_fbm[0])
                    rho_fbm_2 = float(rho_fbm[1])
                    
                    valid_ac = ~np.isnan(rho_emp)
                    if np.sum(valid_ac) >= 3:
                        rho_mse_1to5 = float(np.mean((rho_emp[valid_ac] - rho_fbm[valid_ac])**2))
            
            particle_records.append({
                "unique_particle_id": pid,
                "bead_name": b_name,
                "diameter_um": d_um,
                "radius_um": r_c,
                "video_idx": v_idx,
                "particle_num": orig_p_num,
                "alpha": alpha,
                "Hurst_exponent_H": Hurst,
                "K_alpha_um2_s_alpha": K_alpha,
                "D_alpha_um2_s_alpha": D_alpha,
                "r2_fbm_fit": r2_fbm,
                "tau_self_Rc_fbm_s": tau_self_rc_fbm,
                "tau_self_2Rc_fbm_s": tau_self_2rc_fbm,
                "tau_self_Rc_emp_s": tau_self_rc_emp,
                "tau_self_2Rc_emp_s": tau_self_2rc_emp,
                "ratio_fbm_to_emp_Rc": (tau_self_rc_fbm / tau_self_rc_emp) if (tau_self_rc_emp > 0) else np.nan,
                "excess_kurtosis_x": kurt_x,
                "excess_kurtosis_y": kurt_y,
                "mean_excess_kurtosis": 0.5 * (kurt_x + kurt_y) if not np.isnan(kurt_x) else np.nan,
                "non_gaussianity_beta": beta_ng,
                "shapiro_normality_pval": shapiro_p,
                "rho_emp_lag1": rho_emp_1,
                "rho_fbm_lag1": rho_fbm_1,
                "rho_emp_lag2": rho_emp_2,
                "rho_fbm_lag2": rho_fbm_2,
                "rho_mse_1to5": rho_mse_1to5,
                "color": color,
                "marker": marker
            })
            
    df_part = pd.DataFrame(particle_records)
    df_part.to_csv(out_dir / "fbm_and_self_size_exploration_particles.csv", index=False)
    print(f"Saved particle records (N={len(df_part)}) to figure/msd/fbm_and_self_size_exploration_particles.csv")
    
    # --- 条件別サマリー統計の作成 ---
    summary_rows = []
    for b_info in BEADS_INFO:
        b_name = b_info["name"]
        d_um = b_info["diameter_um"]
        r_c = b_info["radius_um"]
        
        sub = df_part[df_part['bead_name'] == b_name]
        n_p = len(sub)
        if n_p == 0:
            continue
            
        def get_stat(col):
            vals = sub[col].dropna().values
            if len(vals) == 0:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            mean_v = float(np.mean(vals))
            sem_v = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            med_v = float(np.median(vals))
            q25_v = float(np.percentile(vals, 25))
            q75_v = float(np.percentile(vals, 75))
            return mean_v, sem_v, med_v, q25_v, q75_v
            
        a_mean, a_sem, a_med, a_q25, a_q75 = get_stat('alpha')
        H_mean, H_sem, H_med, H_q25, H_q75 = get_stat('Hurst_exponent_H')
        k_mean, k_sem, k_med, k_q25, k_q75 = get_stat('K_alpha_um2_s_alpha')
        r2_mean, r2_sem, r2_med, r2_q25, r2_q75 = get_stat('r2_fbm_fit')
        
        t_rc_mean, t_rc_sem, t_rc_med, t_rc_q25, t_rc_q75 = get_stat('tau_self_Rc_fbm_s')
        t_2rc_mean, t_2rc_sem, t_2rc_med, t_2rc_q25, t_2rc_q75 = get_stat('tau_self_2Rc_fbm_s')
        t_rc_emp_mean, t_rc_emp_sem, t_rc_emp_med, t_rc_emp_q25, t_rc_emp_q75 = get_stat('tau_self_Rc_emp_s')
        t_2rc_emp_mean, t_2rc_emp_sem, t_2rc_emp_med, t_2rc_emp_q25, t_2rc_emp_q75 = get_stat('tau_self_2Rc_emp_s')
        
        kurt_mean, kurt_sem, kurt_med, _, _ = get_stat('mean_excess_kurtosis')
        beta_mean, beta_sem, beta_med, _, _ = get_stat('non_gaussianity_beta')
        rho1_mean, rho1_sem, rho1_med, _, _ = get_stat('rho_emp_lag1')
        rho1_fbm_mean, _, rho1_fbm_med, _, _ = get_stat('rho_fbm_lag1')
        
        # ガウス性検定の棄却率 (p < 0.05 の割合)
        shapiro_vals = sub['shapiro_normality_pval'].dropna().values
        non_gaussian_ratio = float(np.mean(shapiro_vals < 0.05)) if len(shapiro_vals) > 0 else np.nan
        
        summary_rows.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": r_c,
            "n_particles": n_p,
            "alpha_mean": a_mean, "alpha_sem": a_sem, "alpha_median": a_med, "alpha_q25": a_q25, "alpha_q75": a_q75,
            "Hurst_H_mean": H_mean, "Hurst_H_median": H_med,
            "K_alpha_mean": k_mean, "K_alpha_sem": k_sem, "K_alpha_median": k_med,
            "r2_fbm_mean": r2_mean, "r2_fbm_median": r2_med,
            "tau_self_Rc_fbm_mean_s": t_rc_mean, "tau_self_Rc_fbm_sem_s": t_rc_sem, "tau_self_Rc_fbm_median_s": t_rc_med, "tau_self_Rc_fbm_q25_s": t_rc_q25, "tau_self_Rc_fbm_q75_s": t_rc_q75,
            "tau_self_2Rc_fbm_mean_s": t_2rc_mean, "tau_self_2Rc_fbm_sem_s": t_2rc_sem, "tau_self_2Rc_fbm_median_s": t_2rc_med, "tau_self_2Rc_fbm_q25_s": t_2rc_q25, "tau_self_2Rc_fbm_q75_s": t_2rc_q75,
            "tau_self_Rc_emp_mean_s": t_rc_emp_mean, "tau_self_Rc_emp_sem_s": t_rc_emp_sem, "tau_self_Rc_emp_median_s": t_rc_emp_med,
            "tau_self_2Rc_emp_mean_s": t_2rc_emp_mean, "tau_self_2Rc_emp_sem_s": t_2rc_emp_sem, "tau_self_2Rc_emp_median_s": t_2rc_emp_med,
            "excess_kurtosis_mean": kurt_mean, "excess_kurtosis_median": kurt_med,
            "non_gaussianity_beta_mean": beta_mean, "non_gaussianity_beta_median": beta_med,
            "non_gaussian_fraction_p005": non_gaussian_ratio,
            "rho_emp_lag1_mean": rho1_mean, "rho_emp_lag1_median": rho1_med,
            "rho_fbm_lag1_mean": rho1_fbm_mean, "rho_fbm_lag1_median": rho1_fbm_med,
        })
        
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "fbm_and_self_size_exploration_summary.csv", index=False)
    print(f"Saved condition summary to figure/msd/fbm_and_self_size_exploration_summary.csv")
    
    # ------------------------------------------------------------------
    # 図 1: fBm 適合性診断 4 パネル図
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 10))
    ax1_a, ax1_b = axes1[0, 0], axes1[0, 1]
    ax1_c, ax1_d = axes1[1, 0], axes1[1, 1]
    
    # (a) 異常拡散指数 alpha (Hurst H) vs 粒子径
    for b_info in BEADS_INFO:
        sub = df_part[df_part['bead_name'] == b_info['name']]
        # 個別点
        jit = np.random.normal(1.0, 0.04, size=len(sub))
        ax1_a.scatter(sub['diameter_um'] * jit, sub['alpha'], color=b_info['color'], marker=b_info['marker'], alpha=0.6, s=45)
    
    # アンサンブル平均・エラーバー
    ax1_a.errorbar(
        df_sum['diameter_um'], df_sum['alpha_median'],
        yerr=[df_sum['alpha_median'] - df_sum['alpha_q25'], df_sum['alpha_q75'] - df_sum['alpha_median']],
        fmt='s-', color='black', lw=2.0, capsize=4.5, markersize=7, label=r'Median $\pm$ IQR'
    )
    ax1_a.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label=r'Normal Brownian ($\alpha = 1, H = 0.5$)')
    ax1_a.axhline(2.0, color='gray', linestyle=':', alpha=0.7, label=r'Ballistic ($\alpha = 2, H = 1.0$)')
    ax1_a.set_xscale('log')
    ax1_a.set_ylim(0.4, 2.2)
    ax1_a.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax1_a.set_ylabel(r'Anomalous Exponent $\alpha$ ($2H$)', fontsize=11, fontweight='bold')
    ax1_a.set_title(r'(a) Fitted Exponent $\alpha$ (Sub vs Superdiffusion)', fontsize=12, fontweight='bold')
    ax1_a.grid(True, which='both', ls='--', alpha=0.3)
    ax1_a.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    
    # (b) 非ガウス性パラメータ beta vs 粒子径
    for b_info in BEADS_INFO:
        sub = df_part[df_part['bead_name'] == b_info['name']]
        jit = np.random.normal(1.0, 0.04, size=len(sub))
        ax1_b.scatter(sub['diameter_um'] * jit, sub['non_gaussianity_beta'], color=b_info['color'], marker=b_info['marker'], alpha=0.6, s=45)
    ax1_b.errorbar(
        df_sum['diameter_um'], df_sum['non_gaussianity_beta_median'],
        fmt='D-', color='black', lw=2.0, capsize=4.5, markersize=7, label=r'Median Non-Gaussianity $\beta$'
    )
    ax1_b.axhline(0.0, color='red', linestyle='--', lw=1.8, label=r'Pure fBm / Gaussian ($\beta = 0$)')
    ax1_b.set_xscale('log')
    ax1_b.set_yscale('symlog', linthresh=0.1)
    ax1_b.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax1_b.set_ylabel(r'Non-Gaussianity Parameter $\beta = \frac{\langle \Delta r^4 \rangle}{2\langle \Delta r^2 \rangle^2} - 1$', fontsize=10.5, fontweight='bold')
    ax1_b.set_title(r'(b) Increments Gaussianity Test ($\beta \gg 0$: Non-Gaussian)', fontsize=12, fontweight='bold')
    ax1_b.grid(True, which='both', ls='--', alpha=0.3)
    ax1_b.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    
    # (c) 増分自己相関 rho(1): 実測 vs fBm理論値
    ax1_c.plot(df_sum['diameter_um'], df_sum['rho_emp_lag1_median'], 'o-', color='#1f77b4', lw=2.2, markersize=8, label=r'Empirical $\rho_{\mathrm{emp}}(\Delta t = 4\,\mathrm{s})$')
    ax1_c.plot(df_sum['diameter_um'], df_sum['rho_fbm_lag1_median'], 's--', color='#d62728', lw=2.0, markersize=7.5, label=r'fBm Theoretical $\rho_{\mathrm{fBm}}(\alpha)$')
    ax1_c.axhline(0.0, color='gray', linestyle=':', alpha=0.7)
    ax1_c.set_xscale('log')
    ax1_c.set_ylim(-0.3, 0.7)
    ax1_c.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax1_c.set_ylabel(r'Increment Correlation $\rho(\Delta t = 4\,\mathrm{s})$', fontsize=11, fontweight='bold')
    ax1_c.set_title(r'(c) Velocity Autocorrelation: Empirical vs Pure fBm', fontsize=12, fontweight='bold')
    ax1_c.grid(True, which='both', ls='--', alpha=0.3)
    ax1_c.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    
    # (d) fBm べき乗則 MSD 決定係数 R^2
    for b_info in BEADS_INFO:
        sub = df_part[df_part['bead_name'] == b_info['name']]
        jit = np.random.normal(1.0, 0.04, size=len(sub))
        ax1_d.scatter(sub['diameter_um'] * jit, sub['r2_fbm_fit'], color=b_info['color'], marker=b_info['marker'], alpha=0.6, s=45)
    ax1_d.errorbar(
        df_sum['diameter_um'], df_sum['r2_fbm_median'],
        fmt='o-', color='black', lw=2.0, capsize=4.5, markersize=7, label=r'Median $R^2$'
    )
    ax1_d.axhline(0.95, color='orange', linestyle=':', lw=1.5, label=r'$R^2 = 0.95$')
    ax1_d.set_xscale('log')
    ax1_d.set_ylim(0.7, 1.02)
    ax1_d.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax1_d.set_ylabel(r'Power-Law Goodness of Fit $R^2$', fontsize=11, fontweight='bold')
    ax1_d.set_title(r'(d) Linearity of $\ln\mathrm{MSD}$ vs $\ln\Delta t$', fontsize=12, fontweight='bold')
    ax1_d.grid(True, which='both', ls='--', alpha=0.3)
    ax1_d.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    
    fig1.suptitle('Fractional Brownian Motion (fBm) Diagnostic Evaluation for Cargo Beads', fontsize=14, fontweight='bold', y=0.99)
    fig1.tight_layout()
    fig1.savefig(out_dir / "fbm_diagnostic_evaluation_4panel.png", dpi=300, bbox_inches='tight')
    fig1.savefig(out_dir / "fbm_diagnostic_evaluation_4panel.svg", bbox_inches='tight')
    plt.close(fig1)
    print("Saved Figure 1: fbm_diagnostic_evaluation_4panel (.png & .svg)")
    
    # ------------------------------------------------------------------
    # 図 2: 自己サイズ探索時間 tau_self vs 粒子サイズ (2パネル)
    # パネル (a): 半径探索時間 tau_self,Rc vs 半径 Rc
    # パネル (b): 直径探索時間 tau_self,2Rc vs 直径 2Rc
    # ------------------------------------------------------------------
    fig2, (ax2_a, ax2_b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- パネル (a): 半径探索時間 tau_self(R_c) ---
    for b_info in BEADS_INFO:
        sub = df_part[df_part['bead_name'] == b_info['name']]
        # 個別点 (fBm推計値)
        ax2_a.scatter(
            sub['radius_um'], sub['tau_self_Rc_fbm_s'],
            color=b_info['color'], marker=b_info['marker'], alpha=0.55, s=50,
            label=rf"${b_info['diameter_um']:.2f}\,\mu\mathrm{{m}}$ ($N={len(sub)}$)"
        )
    # アンサンブル中央値 (fBm)
    ax2_a.errorbar(
        df_sum['radius_um'], df_sum['tau_self_Rc_fbm_median_s'],
        yerr=[df_sum['tau_self_Rc_fbm_median_s'] - df_sum['tau_self_Rc_fbm_q25_s'],
              df_sum['tau_self_Rc_fbm_q75_s'] - df_sum['tau_self_Rc_fbm_median_s']],
        fmt='o-', color='#111111', lw=2.2, capsize=5, markersize=8, zorder=5,
        label=r'fBm Fit: $\tau_{\mathrm{self}, R_c} = (R_c^2 / K_\alpha)^{1/\alpha}$'
    )
    # 実測直接補間値 (Empirical interpolation)
    ax2_a.plot(
        df_sum['radius_um'], df_sum['tau_self_Rc_emp_median_s'],
        's--', color='#d62728', lw=1.8, markersize=7.5, zorder=4,
        label=r'Empirical MSD Cross: $\mathrm{MSD}(\tau) = R_c^2$'
    )
    
    ax2_a.set_xscale('log')
    ax2_a.set_yscale('log')
    ax2_a.set_xlim(0.2, 15.0)
    ax2_a.set_ylim(0.05, 500.0)
    ax2_a.xaxis.set_major_locator(ticker.FixedLocator([0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]))
    ax2_a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax2_a.set_xlabel(r'Cargo Radius $R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax2_a.set_ylabel(r'Self-Radius Exploration Time $\tau_{\mathrm{self}, R_c}$ [s]', fontsize=12, fontweight='bold')
    ax2_a.set_title(r'(a) Self-Radius Exploration Time ($\sqrt{\Delta r^2} = R_c$)', fontsize=12.5, fontweight='bold')
    ax2_a.grid(True, which='both', ls='--', alpha=0.35)
    ax2_a.legend(fontsize=8, loc='upper left', framealpha=0.92)
    
    # --- パネル (b): 直径探索時間 tau_self(2R_c) ---
    for b_info in BEADS_INFO:
        sub = df_part[df_part['bead_name'] == b_info['name']]
        ax2_b.scatter(
            sub['diameter_um'], sub['tau_self_2Rc_fbm_s'],
            color=b_info['color'], marker=b_info['marker'], alpha=0.55, s=50,
            label=rf"${b_info['diameter_um']:.2f}\,\mu\mathrm{{m}}$ ($N={len(sub)}$)"
        )
    ax2_b.errorbar(
        df_sum['diameter_um'], df_sum['tau_self_2Rc_fbm_median_s'],
        yerr=[df_sum['tau_self_2Rc_fbm_median_s'] - df_sum['tau_self_2Rc_fbm_q25_s'],
              df_sum['tau_self_2Rc_fbm_q75_s'] - df_sum['tau_self_2Rc_fbm_median_s']],
        fmt='o-', color='#111111', lw=2.2, capsize=5, markersize=8, zorder=5,
        label=r'fBm Fit: $\tau_{\mathrm{self}, 2R_c} = (4 R_c^2 / K_\alpha)^{1/\alpha}$'
    )
    ax2_b.plot(
        df_sum['diameter_um'], df_sum['tau_self_2Rc_emp_median_s'],
        's--', color='#d62728', lw=1.8, markersize=7.5, zorder=4,
        label=r'Empirical MSD Cross: $\mathrm{MSD}(\tau) = (2R_c)^2$'
    )
    
    ax2_b.set_xscale('log')
    ax2_b.set_yscale('log')
    ax2_b.set_xlim(0.4, 28.0)
    ax2_b.set_ylim(0.2, 2000.0)
    ax2_b.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    ax2_b.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax2_b.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax2_b.set_ylabel(r'Self-Diameter Exploration Time $\tau_{\mathrm{self}, 2R_c}$ [s]', fontsize=12, fontweight='bold')
    ax2_b.set_title(r'(b) Self-Diameter Exploration Time ($\sqrt{\Delta r^2} = 2R_c$)', fontsize=12.5, fontweight='bold')
    ax2_b.grid(True, which='both', ls='--', alpha=0.35)
    ax2_b.legend(fontsize=8, loc='upper left', framealpha=0.92)
    
    fig2.suptitle(r'Cargo Self-Size Exploration Times vs Particle Dimensions', fontsize=14, fontweight='bold', y=0.98)
    fig2.tight_layout()
    fig2.savefig(out_dir / "self_size_exploration_time_vs_cargo_size_2panel.png", dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / "self_size_exploration_time_vs_cargo_size_2panel.svg", bbox_inches='tight')
    plt.close(fig2)
    print("Saved Figure 2: self_size_exploration_time_vs_cargo_size_2panel (.png & .svg)")
    
    # ------------------------------------------------------------------
    # 図 3: 各ビーズ条件の代表的 iMSD と fBm フィッティング曲線 (6パネル)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes3 = axes3.flatten()
    
    for idx, b_info in enumerate(BEADS_INFO):
        ax = axes3[idx]
        b_name = b_info["name"]
        d_um = b_info["diameter_um"]
        r_c = b_info["radius_um"]
        color = b_info["color"]
        
        folder = root_dir / b_name
        pool_df, _, _, _ = MSD.concat_pooled_particles_MSD(folder, b_info["intervals"], scale=0.11, alpha_threshold=0.5)
        
        # 個別粒子の iMSD を薄くプロット
        sub_p = df_part[df_part['bead_name'] == b_name]
        
        for pid, grp in pool_df.groupby('unique_particle_id'):
            ax.plot(grp['lag time'], grp['MSD'], color=color, alpha=0.25, lw=0.9)
            
        # 代表的な中央値粒子 (Median alpha 粒子) のフィッティングを表示
        if not sub_p.empty:
            med_a = sub_p['alpha'].median()
            best_idx = (sub_p['alpha'] - med_a).abs().argsort().iloc[0]
            rep_row = sub_p.iloc[best_idx]
            rep_pid = rep_row['unique_particle_id']
            rep_grp = pool_df[pool_df['unique_particle_id'] == rep_pid]
            
            ax.plot(rep_grp['lag time'], rep_grp['MSD'], 'o', color=color, markersize=5, label=f"Rep. Particle ({rep_pid})")
            
            t_fit = np.logspace(np.log10(4.0), np.log10(300.0), 100)
            msd_fit = rep_row['K_alpha_um2_s_alpha'] * (t_fit ** rep_row['alpha'])
            ax.plot(t_fit, msd_fit, '-', color='black', lw=2.0,
                    label=rf"fBm Fit: $\alpha={rep_row['alpha']:.2f}$, $R^2={rep_row['r2_fbm_fit']:.3f}$")
            
            # 自己サイズ探索時間のライン
            tau_rc = rep_row['tau_self_Rc_fbm_s']
            tau_2rc = rep_row['tau_self_2Rc_fbm_s']
            
            # 水平線: R_c^2 と (2R_c)^2
            ax.axhline(r_c**2, color='gray', linestyle=':', alpha=0.7)
            ax.axhline((2.0 * r_c)**2, color='gray', linestyle='--', alpha=0.7)
            
            if 4.0 <= tau_rc <= 1000.0:
                ax.axvline(tau_rc, color='tab:blue', linestyle=':', alpha=0.8,
                           label=rf"$\tau_{{R_c}} = {tau_rc:.1f}\,\mathrm{{s}}$")
            if 4.0 <= tau_2rc <= 1000.0:
                ax.axvline(tau_2rc, color='tab:red', linestyle='--', alpha=0.8,
                           label=rf"$\tau_{{2R_c}} = {tau_2rc:.1f}\,\mathrm{{s}}$")
                
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3.0, 1000.0)
        ax.set_ylim(0.01, 5000.0)
        ax.set_title(rf"${d_um:.2f}\,\mu\mathrm{{m}}$ ({b_name}, $N={len(sub_p)}$)", fontsize=11, fontweight='bold')
        ax.set_xlabel(r'Lag time $\Delta t$ [s]', fontsize=10)
        ax.set_ylabel(r'iMSD [$\mu\mathrm{m}^2$]', fontsize=10)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
        
    fig3.suptitle(r'Individual Particles iMSD & Fractional Brownian Motion Fits', fontsize=14, fontweight='bold', y=0.99)
    fig3.tight_layout()
    fig3.savefig(out_dir / "individual_imsd_and_fbm_fits_6panel.png", dpi=300, bbox_inches='tight')
    fig3.savefig(out_dir / "individual_imsd_and_fbm_fits_6panel.svg", bbox_inches='tight')
    plt.close(fig3)
    print("Saved Figure 3: individual_imsd_and_fbm_fits_6panel (.png & .svg)")
    
    print("\n--- Summary Table ---")
    cols_disp = ['bead_name', 'diameter_um', 'n_particles', 'alpha_median', 'non_gaussianity_beta_median', 'tau_self_Rc_fbm_median_s', 'tau_self_2Rc_fbm_median_s']
    print(df_sum[cols_disp].to_string(index=False))

if __name__ == '__main__':
    main()
