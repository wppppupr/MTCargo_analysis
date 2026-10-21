#!/usr/bin/env python3
"""
benchmark_relaxation_time_methods.py

MSDの弾道領域(Δt^2)から拡散領域(Δt^1)へのクロスオーバー時間（緩和時間 τ）を
高精度かつ滑らかに求めるための複数の手法をベンチマーク比較するスクリプト。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from pathlib import Path

# 1. 各種フィッティング関数の定義
def log_pade_msd(log_t, log_v0_sq, log_tau):
    """有理関数（パデ近似）: MSD(t) = v0^2 * t^2 / (1 + t / tau)"""
    t = 10.0 ** log_t
    tau = 10.0 ** log_tau
    v0_sq = 10.0 ** log_v0_sq
    msd = v0_sq * (t ** 2) / (1.0 + t / tau)
    return np.log10(np.maximum(msd, 1e-12))

def log_abp_msd(log_t, log_Deff, log_tau):
    """ABP / OU 過程 (Furthの公式): MSD(t) = 4 * Deff * [t - tau * (1 - exp(-t / tau))]"""
    t = 10.0 ** log_t
    tau = 10.0 ** log_tau
    Deff = 10.0 ** log_Deff
    # t / tau
    u = t / tau
    # 数値安定化: uが小さいときはテーラー展開
    # t - tau*(1 - exp(-u)) = tau * [u - 1 + exp(-u)]
    small_u = u < 1e-4
    active_term = np.zeros_like(u)
    active_term[small_u] = tau * (0.5 * u[small_u]**2 - (u[small_u]**3)/6.0)
    active_term[~small_u] = t[~small_u] - tau * (1.0 - np.exp(-u[~small_u]))
    
    msd = 4.0 * Deff * active_term
    return np.log10(np.maximum(msd, 1e-12))

def log_sbpl_msd(log_t, log_A, log_tau, s):
    """一般化滑らか遷移モデル (Smooth Broken Power Law):
       MSD(t) = A * t^2 * [1 + (t/tau)^s]^(-1/s)
    """
    t = 10.0 ** log_t
    tau = 10.0 ** log_tau
    A = 10.0 ** log_A
    s = np.clip(s, 0.2, 5.0)
    ratio = (t / tau) ** s
    log_msd = np.log10(A) + 2.0 * log_t - (1.0 / s) * np.log10(1.0 + ratio)
    return log_msd

def run_benchmark():
    df = pd.read_csv('figure/msd/msd_pooled_mean_iqr_summary.csv')
    beads = df['bead_name'].unique()
    
    results = []
    
    for b in beads:
        sub = df[df['bead_name'] == b].sort_values('lag_time_s')
        d_um = sub['diameter_um'].iloc[0]
        
        # 信頼できるラグタイム範囲（active particles >= 2, かつ最大1000sまで）
        sub = sub[(sub['lag_time_s'] <= 1000) & (sub['n_active_particles'] >= 2) & (sub['mean_msd_um2'] > 0)]
        t = sub['lag_time_s'].values
        msd = sub['mean_msd_um2'].values
        log_t = np.log10(t)
        log_msd = np.log10(msd)
        
        # --- 手法 0: 生微分 (Raw Gradient) ---
        alpha_raw = np.gradient(log_msd, log_t)
        # alpha_raw が 1.5 を横切る点（複数ある場合は最初と最後）
        cross_raw_idx = np.where(np.diff(np.sign(alpha_raw - 1.5)))[0]
        if len(cross_raw_idx) > 0:
            tau_raw_first = t[cross_raw_idx[0]]
            tau_raw_last = t[cross_raw_idx[-1]]
            n_crossings = len(cross_raw_idx)
        else:
            tau_raw_first = tau_raw_last = np.nan
            n_crossings = 0
            
        # --- 手法 1: Padé 有理関数モデル ---
        # 初期値推定: v0^2 ~ msd[0]/t[0]^2, tau ~ 50s
        p0_pade = [np.log10(msd[0] / (t[0]**2)), np.log10(50.0)]
        try:
            popt_pade, pcov_pade = curve_fit(log_pade_msd, log_t, log_msd, p0=p0_pade, maxfev=5000)
            tau_pade = 10.0 ** popt_pade[1]
            tau_pade_err = np.log(10.0) * tau_pade * np.sqrt(pcov_pade[1, 1]) if pcov_pade[1, 1] > 0 else np.nan
            fit_msd_pade = 10.0 ** log_pade_msd(log_t, *popt_pade)
            r2_pade = 1.0 - np.sum((log_msd - np.log10(fit_msd_pade))**2) / np.sum((log_msd - np.mean(log_msd))**2)
        except Exception as e:
            tau_pade = tau_pade_err = r2_pade = np.nan
            
        # --- 手法 2: ABP / OU (Furth) モデル ---
        # Deff = v0^2 * tau / 2 -> msd_inf ~ 4 * Deff * t
        p0_abp = [np.log10(msd[-1] / (4.0 * t[-1])), np.log10(30.0)]
        try:
            popt_abp, pcov_abp = curve_fit(log_abp_msd, log_t, log_msd, p0=p0_abp, maxfev=5000)
            tau_abp = 10.0 ** popt_abp[1]
            tau_abp_err = np.log(10.0) * tau_abp * np.sqrt(pcov_abp[1, 1]) if pcov_abp[1, 1] > 0 else np.nan
            fit_msd_abp = 10.0 ** log_abp_msd(log_t, *popt_abp)
            r2_abp = 1.0 - np.sum((log_msd - np.log10(fit_msd_abp))**2) / np.sum((log_msd - np.mean(log_msd))**2)
            # ABP で alpha = 1.5 となる時間は t_crossover = 2.149 * tau_abp
            tau_abp_crossover = 2.14914 * tau_abp
        except Exception as e:
            tau_abp = tau_abp_err = r2_abp = tau_abp_crossover = np.nan

        # --- 手法 3: 一般化滑らか遷移モデル (SBPL) ---
        p0_sbpl = [np.log10(msd[0] / (t[0]**2)), np.log10(50.0), 1.0]
        bounds_sbpl = ([-np.inf, -1.0, 0.2], [np.inf, 4.0, 5.0])
        try:
            popt_sbpl, pcov_sbpl = curve_fit(log_sbpl_msd, log_t, log_msd, p0=p0_sbpl, bounds=bounds_sbpl, maxfev=5000)
            tau_sbpl = 10.0 ** popt_sbpl[1]
            s_sbpl = popt_sbpl[2]
            tau_sbpl_err = np.log(10.0) * tau_sbpl * np.sqrt(pcov_sbpl[1, 1]) if pcov_sbpl[1, 1] > 0 else np.nan
            fit_msd_sbpl = 10.0 ** log_sbpl_msd(log_t, *popt_sbpl)
            r2_sbpl = 1.0 - np.sum((log_msd - np.log10(fit_msd_sbpl))**2) / np.sum((log_msd - np.mean(log_msd))**2)
        except Exception as e:
            tau_sbpl = s_sbpl = tau_sbpl_err = r2_sbpl = np.nan

        # --- 手法 4: 平滑化スプライン (UnivariateSpline) ---
        # s はデータ点数 * ノイズ分散の目安
        try:
            # 適切な平滑化パラメータ s_spline
            spline = UnivariateSpline(log_t, log_msd, s=0.005 * len(log_t), k=3)
            spline_deriv = spline.derivative()
            alpha_spline = spline_deriv(log_t)
            
            # alpha_spline(log_t) = 1.5 を解く
            def root_func(lx):
                return spline_deriv(lx) - 1.5
                
            # 探索範囲
            if np.min(alpha_spline) <= 1.5 <= np.max(alpha_spline):
                sol = root_scalar(root_func, bracket=[log_t[0], log_t[-1]], method='brentq')
                tau_spline = 10.0 ** sol.root
            else:
                tau_spline = np.nan
        except Exception as e:
            tau_spline = np.nan

        # --- 手法 5: Savitzky-Golay フィルタ平滑化微分 ---
        try:
            # log_t は非等間隔なので均等グリッドに対数補間してからSavgol適用
            log_t_grid = np.linspace(log_t[0], log_t[-1], 200)
            log_msd_interp = np.interp(log_t_grid, log_t, log_msd)
            delta_log_t = log_t_grid[1] - log_t_grid[0]
            # 窓幅 31, 多項式 3, 1階微分
            alpha_sg = savgol_filter(log_msd_interp, window_length=31, polyorder=3, deriv=1, delta=delta_log_t)
            if np.min(alpha_sg) <= 1.5 <= np.max(alpha_sg):
                idx_cross = np.where(np.diff(np.sign(alpha_sg - 1.5)))[0]
                if len(idx_cross) > 0:
                    tau_savgol = 10.0 ** log_t_grid[idx_cross[0]]
                else:
                    tau_savgol = np.nan
            else:
                tau_savgol = np.nan
        except Exception as e:
            tau_savgol = np.nan

        results.append({
            'bead_name': b,
            'diameter_um': d_um,
            'n_crossings_raw': n_crossings,
            'tau_raw_first': tau_raw_first,
            'tau_raw_last': tau_raw_last,
            'tau_pade': tau_pade,
            'tau_pade_err': tau_pade_err,
            'r2_pade': r2_pade,
            'tau_abp': tau_abp,
            'tau_abp_crossover': tau_abp_crossover,
            'r2_abp': r2_abp,
            'tau_sbpl': tau_sbpl,
            's_sbpl': s_sbpl,
            'r2_sbpl': r2_sbpl,
            'tau_spline': tau_spline,
            'tau_savgol': tau_savgol,
        })
        
    df_res = pd.DataFrame(results)
    print("=== Benchmark Results ===")
    cols_display = ['bead_name', 'diameter_um', 'n_crossings_raw', 'tau_pade', 'tau_abp_crossover', 'tau_sbpl', 'tau_spline', 'tau_savgol', 'r2_pade', 'r2_abp']
    print(df_res[cols_display].to_string(index=False))
    return df_res

if __name__ == '__main__':
    run_benchmark()
