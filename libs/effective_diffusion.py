"""
libs/effective_diffusion.py

速度自己相関関数（VACF）のGreen-Kubo積分および
HMMパラメータに基づくRun-and-Tumble Particle (RTP) 理論モデルから
有効拡散係数 D_eff を計算・比較・可視化するモジュールです。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from libs import vacf
from libs import vacf_analysis as va

# スタイルの適用
style_path = Path(__file__).parent / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "label": "0.63 μm", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "label": "1.18 μm", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "label": "3.37 μm", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "label": "5.00 μm", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "label": "7.24 μm", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "label": "20.0 μm", "color": style_colors[5]},
]


def integrate_vacf_green_kubo(
    evacf_df: pd.DataFrame,
    max_lag_time: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Union[float, np.ndarray]]]:
    """
    未正規化 VACF データフレームから Green-Kubo 積分を行い、
    実験ごとの D_eff およびアンサンブル平均 D_eff を計算する。

    Parameters
    ----------
    evacf_df : pd.DataFrame
        'exp', 'lag time', 'VACF' カラムを持つデータフレーム（normalize=False で計算されたもの）
    max_lag_time : float, optional
        積分に用いる最大ラグ時間 [s]

    Returns
    -------
    df_exp_results : pd.DataFrame
        実験ごとの積分結果
    ensemble_summary : dict
        アンサンブル平均結果
    """
    if evacf_df.empty:
        return pd.DataFrame(), {}

    exp_records = []
    
    for exp_id, exp_group in evacf_df.groupby('exp'):
        exp_group = exp_group.sort_values('lag time')
        if max_lag_time is not None:
            exp_group = exp_group[exp_group['lag time'] <= max_lag_time]
            
        t = exp_group['lag time'].to_numpy(dtype=float)
        c_v = exp_group['VACF'].to_numpy(dtype=float)
        
        if len(t) < 2:
            continue
            
        # 1. 全体台形積分 D_eff = 0.5 * \int_0^T C_v(t) dt
        trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        d_eff_trapz_full = 0.5 * trapz_func(c_v, t)
        
        # 2. 最初のゼロクロス点までの台形積分
        zero_cross_idx = np.where(c_v < 0)[0]
        if len(zero_cross_idx) > 0 and zero_cross_idx[0] > 0:
            fc = zero_cross_idx[0]
            t_sub = t[:fc]
            c_sub = c_v[:fc]
            # ゼロクロス点での線形補間
            denom = c_v[fc] - c_v[fc - 1]
            if abs(denom) > 1e-12:
                t_cross = t[fc - 1] - c_v[fc - 1] * (t[fc] - t[fc - 1]) / denom
                t_sub = np.append(t_sub, t_cross)
                c_sub = np.append(c_sub, 0.0)
            d_eff_trapz_pos = 0.5 * trapz_func(c_sub, t_sub)
        else:
            d_eff_trapz_pos = d_eff_trapz_full
            
        # 3. 累積積分 D_eff(t)
        cum_d_eff = 0.5 * np.array([trapz_func(c_v[:j + 1], t[:j + 1]) for j in range(len(t))])
        
        # 4. 累積積分が最大になる点 (Peak)
        peak_idx = int(np.argmax(cum_d_eff))
        d_eff_max = float(cum_d_eff[peak_idx])
        t_peak = float(t[peak_idx])
        
        # 5. 累積積分系列の中央値 (Median)
        d_eff_median = float(np.median(cum_d_eff))
        
        exp_records.append({
            'exp': exp_id,
            'd_eff_full': d_eff_trapz_full,
            'd_eff_pos': d_eff_trapz_pos,
            'd_eff_max': d_eff_max,
            'd_eff_median': d_eff_median,
            't_peak': t_peak,
            'c_v_0': c_v[0] if len(c_v) > 0 else np.nan,
            't': t,
            'c_v': c_v,
            'cum_d_eff': cum_d_eff
        })

    if not exp_records:
        return pd.DataFrame(), {}

    df_exp_results = pd.DataFrame(exp_records)
    
    # アンサンブル集約
    ens_mean = evacf_df.groupby('lag time')['VACF'].mean()
    ens_std = evacf_df.groupby('lag time')['VACF'].std().fillna(0.0)
    
    if max_lag_time is not None:
        ens_mean = ens_mean[ens_mean.index <= max_lag_time]
        ens_std = ens_std[ens_std.index <= max_lag_time]
        
    t_ens = ens_mean.index.to_numpy(dtype=float)
    c_ens = ens_mean.to_numpy(dtype=float)
    
    trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    d_ens_full = 0.5 * trapz_func(c_ens, t_ens)
    
    z_idx = np.where(c_ens < 0)[0]
    if len(z_idx) > 0 and z_idx[0] > 0:
        fc = z_idx[0]
        t_sub = t_ens[:fc]
        c_sub = c_ens[:fc]
        denom = c_ens[fc] - c_ens[fc - 1]
        if abs(denom) > 1e-12:
            t_cross = t_ens[fc - 1] - c_ens[fc - 1] * (t_ens[fc] - t_ens[fc - 1]) / denom
            t_sub = np.append(t_sub, t_cross)
            c_sub = np.append(c_sub, 0.0)
        d_ens_pos = 0.5 * trapz_func(c_sub, t_sub)
    else:
        d_ens_pos = d_ens_full
        
    cum_d_ens = 0.5 * np.array([trapz_func(c_ens[:j + 1], t_ens[:j + 1]) for j in range(len(t_ens))])
    
    ens_peak_idx = int(np.argmax(cum_d_ens))
    d_ens_max = float(cum_d_ens[ens_peak_idx])
    t_ens_peak = float(t_ens[ens_peak_idx])
    d_ens_median = float(np.median(cum_d_ens))
    
    n_exps = len(df_exp_results)
    mean_exp_d = df_exp_results['d_eff_full'].mean()
    std_exp_d = df_exp_results['d_eff_full'].std()
    sem_exp_d = std_exp_d / np.sqrt(n_exps) if n_exps > 1 else 0.0

    mean_pos_d = df_exp_results['d_eff_pos'].mean()
    std_pos_d = df_exp_results['d_eff_pos'].std()
    sem_pos_d = std_pos_d / np.sqrt(n_exps) if n_exps > 1 else 0.0

    mean_max_d = df_exp_results['d_eff_max'].mean()
    std_max_d = df_exp_results['d_eff_max'].std()
    sem_max_d = std_max_d / np.sqrt(n_exps) if n_exps > 1 else 0.0
    mean_t_peak = df_exp_results['t_peak'].mean()

    mean_med_d = df_exp_results['d_eff_median'].mean()
    std_med_d = df_exp_results['d_eff_median'].std()
    sem_med_d = std_med_d / np.sqrt(n_exps) if n_exps > 1 else 0.0
    med_med_d = df_exp_results['d_eff_median'].median()

    ensemble_summary = {
        'n_exps': n_exps,
        'd_eff_mean': float(mean_exp_d),
        'd_eff_std': float(std_exp_d),
        'd_eff_sem': float(sem_exp_d),
        'd_eff_pos_mean': float(mean_pos_d),
        'd_eff_pos_std': float(std_pos_d),
        'd_eff_pos_sem': float(sem_pos_d),
        'd_eff_max_mean': float(mean_max_d),
        'd_eff_max_std': float(std_max_d),
        'd_eff_max_sem': float(sem_max_d),
        'd_eff_median_mean': float(mean_med_d),
        'd_eff_median_std': float(std_med_d),
        'd_eff_median_sem': float(sem_med_d),
        'd_eff_median_median': float(med_med_d),
        'mean_t_peak': float(mean_t_peak),
        'd_eff_ens_full': float(d_ens_full),
        'd_eff_ens_pos': float(d_ens_pos),
        'd_eff_ens_max': float(d_ens_max),
        'd_eff_ens_median': float(d_ens_median),
        't_ens_peak': float(t_ens_peak),
        'c_v_0': float(c_ens[0]) if len(c_ens) > 0 else 0.0,
        't_lag': t_ens,
        'c_v_mean': c_ens,
        'c_v_std': ens_std.to_numpy(dtype=float),
        'cum_d_ens': cum_d_ens
    }
    
    return df_exp_results, ensemble_summary


def calc_stokes_einstein_diffusion(
    diameter_um: Union[float, np.ndarray],
    temperature_K: float = 298.15,
    viscosity_Pa_s: float = 1.0e-3
) -> Union[float, np.ndarray]:
    """
    ストークス・アインシュタインの式に基づき、微粒子の直径から理論的な熱拡散係数 D_SE [um^2/s] を計算する。
    D_SE = k_B * T / (3 * pi * eta * d)
    
    Parameters
    ----------
    diameter_um : float or np.ndarray
        粒子径 [um]
    temperature_K : float, default 298.15 (25 degC)
        絶対温度 [K]
    viscosity_Pa_s : float, default 1.0e-3 (1 mPa*s, water at 20 degC)
        溶媒の粘性率 [Pa*s]

    Returns
    -------
    float or np.ndarray
        理論熱拡散係数 D_0 [um^2/s]
    """
    k_B = 1.380649e-23  # J/K (Boltzmann constant)
    d_m = np.asarray(diameter_um, dtype=float) * 1e-6
    # D [m^2/s] = k_B * T / (3 * pi * eta * d)
    d_m2_s = (k_B * temperature_K) / (3.0 * np.pi * viscosity_Pa_s * d_m)
    d_um2_s = d_m2_s * 1e12  # m^2/s -> um^2/s
    if np.ndim(diameter_um) == 0:
        return float(d_um2_s)
    return d_um2_s


def compute_rtp_theoretical_diffusion(
    hmm_summary_path: Path,
    angle_summary_path: Optional[Path] = None,
    abp_summary_path: Optional[Path] = None,
    state_msd_fits_path: Optional[Path] = None,
    state_msd_curves_path: Optional[Path] = None,
    autocorr_summary_path: Optional[Path] = None,
    temperature_K: float = 298.15,
    viscosity_Pa_s: float = 1.0e-3
) -> pd.DataFrame:
    """
    HMM パラメータ CSV、配向変化角 CSV、Run MSD フィッティング (ABP) CSV、
    Tumble 状態 MSD フィッティング CSV、および 状態別自己相関 (OACF) CSV から
    Run-and-Tumble Particle (RTP) モデルの理論的有効拡散係数を算出する。
    
    D_eff = D_0 + (1/2) * f_run * v_R^2 * tau_eff
    
    ここで、
    - D_0: Tumble 状態の粒子 MSD から見積もった受動拡散係数 D_0,tumble
           （または補助として Stokes-Einstein 理論熱拡散係数 D_SE(d) = k_B * T / (3 * pi * eta * d)）
    - tau_eff: 配向相関時間 tau_OACF と Run dwell time tau_dwell の調和結合
               1 / tau_eff = 1 / tau_OACF + 1 / tau_dwell
               tau_eff = 1 / (1 / tau_OACF + 1 / tau_dwell)

    Parameters
    ----------
    hmm_summary_path : Path
        `hmm_state_parameters_summary_k2.csv` のパス
    angle_summary_path : Path, optional
        `hmm_turning_angle_summary_k2.csv` のパス
    abp_summary_path : Path, optional
        `hmm_run_abp_fits_summary_k2.csv` のパス
    state_msd_fits_path : Path, optional
        `hmm_state_msd_fits_k2.csv` のパス
    state_msd_curves_path : Path, optional
        `hmm_state_msd_curves_k2.csv` のパス
    autocorr_summary_path : Path, optional
        `hmm_autocorrelation_summary_k2.csv` のパス
    temperature_K : float, default 298.15
        絶対温度 [K]
    viscosity_Pa_s : float, default 1.0e-3
        溶媒の粘度 [Pa*s]

    Returns
    -------
    pd.DataFrame
        粒子径ごとの理論有効拡散係数および微視的パラメータ
    """
    df_hmm = pd.read_csv(hmm_summary_path)
    df_angle = pd.read_csv(angle_summary_path) if (angle_summary_path and Path(angle_summary_path).exists()) else pd.DataFrame()
    df_abp = pd.read_csv(abp_summary_path) if (abp_summary_path and Path(abp_summary_path).exists()) else pd.DataFrame()
    df_msd_fits = pd.read_csv(state_msd_fits_path) if (state_msd_fits_path and Path(state_msd_fits_path).exists()) else pd.DataFrame()
    df_msd_curves = pd.read_csv(state_msd_curves_path) if (state_msd_curves_path and Path(state_msd_curves_path).exists()) else pd.DataFrame()
    df_autocorr = pd.read_csv(autocorr_summary_path) if (autocorr_summary_path and Path(autocorr_summary_path).exists()) else pd.DataFrame()
    
    records = []
    
    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        
        sub_hmm = df_hmm[df_hmm['diameter_um'] == d_um]
        if sub_hmm.empty:
            continue
            
        row_tumble = sub_hmm[sub_hmm['state'] == 0].iloc[0]
        row_run = sub_hmm[sub_hmm['state'] == 1].iloc[0]
        
        v_run = float(row_run['mean_speed_model_um_s'])
        v_run_geom = float(row_run['mean_speed_geom_um_s'])
        v_tumble = float(row_tumble['mean_speed_model_um_s'])
        
        tau_run_theo = float(row_run['theoretical_dwell_time_s'])
        tau_tumble_theo = float(row_tumble['theoretical_dwell_time_s'])
        
        tau_run_ccdf = float(row_run['tau_ccdf_s']) if 'tau_ccdf_s' in row_run else float(row_run['tau_fit_pdf_s'])
        tau_tumble_ccdf = float(row_tumble['tau_ccdf_s']) if 'tau_ccdf_s' in row_tumble else float(row_tumble['tau_fit_pdf_s'])
        
        tau_run_fit = float(row_run['tau_fit_pdf_s'])
        tau_tumble_fit = float(row_tumble['tau_fit_pdf_s'])
        tau_run_emp = float(row_run['mean_dwell_emp_s'])
        tau_tumble_emp = float(row_tumble['mean_dwell_emp_s'])
        
        pi_run = float(row_run['stationary_prob'])
        pi_tumble = float(row_tumble['stationary_prob'])
        f_run_emp = float(row_run['empirical_prob'])
        
        # 1. 粒子径から理論的に算出される Stokes-Einstein 熱拡散係数 D_0 [um^2/s]
        d_0_se = calc_stokes_einstein_diffusion(d_um, temperature_K=temperature_K, viscosity_Pa_s=viscosity_Pa_s)
        
        # 2. Tumble (state 0) 状態の粒子 MSD から見積もった受動拡散係数 D_0,tumble
        d_0_tumble_app = np.nan
        d_0_tumble_lin = np.nan
        alpha_tumble = np.nan
        
        if not df_msd_fits.empty:
            sub_mf = df_msd_fits[(df_msd_fits['diameter_um'] == d_um) & (df_msd_fits['state'] == 0)]
            if not sub_mf.empty:
                d_0_tumble_app = float(sub_mf.iloc[0]['D_apparent_um2_s'])
                alpha_tumble = float(sub_mf.iloc[0]['alpha'])
                
        if not df_msd_curves.empty:
            sub_mc = df_msd_curves[(df_msd_curves['diameter_um'] == d_um) & (df_msd_curves['state'] == 0) & (df_msd_curves['tau_step'] <= 10)]
            if not sub_mc.empty:
                t_arr = sub_mc['lag_time_s'].to_numpy(dtype=float)
                m_arr = sub_mc['msd_um2'].to_numpy(dtype=float)
                if len(t_arr) > 0 and np.sum(t_arr**2) > 0:
                    d_0_tumble_lin = float(np.sum(t_arr * m_arr) / (4.0 * np.sum(t_arr**2)))

        if np.isfinite(d_0_tumble_lin) and d_0_tumble_lin > 0:
            d_0_tumble = d_0_tumble_lin
        elif np.isfinite(d_0_tumble_app) and d_0_tumble_app > 0:
            d_0_tumble = d_0_tumble_app
        else:
            d_0_tumble = d_0_se

        # 3. Orientation 自己相関の積分時間 tau_OACF,int の取得
        tau_oacf_fit = np.nan
        tau_oacf_int = np.nan
        if not df_autocorr.empty:
            sub_ac = df_autocorr[(df_autocorr['diameter_um'] == d_um) & (df_autocorr['state'] == 1) & (df_autocorr['mode'] == 'oacf')]
            if not sub_ac.empty:
                val_corr = float(sub_ac.iloc[0]['tau_corr_s'])
                val_int = float(sub_ac.iloc[0]['tau_int_zero_s'])
                if np.isfinite(val_corr) and val_corr > 0:
                    tau_oacf_fit = val_corr
                if np.isfinite(val_int) and val_int > 0:
                    tau_oacf_int = val_int

        # 方向相関 <cos delta theta>
        mean_cos_run = 0.0
        if not df_angle.empty:
            sub_angle = df_angle[(df_angle['diameter_um'] == d_um) & (df_angle['state'] == 1)]
            if not sub_angle.empty:
                mean_cos_run = float(sub_angle.iloc[0]['mean_cos'])

        # Orientation 自己相関時間として「積分時間」tau_oacf_int を優先採用
        # ※ 0.63 um 粒子については tau_OACF = inf (1/tau_OACF = 0) とみなし tau_eff = tau_dwell とする
        if np.isclose(d_um, 0.63, atol=0.05):
            tau_oacf = np.inf
            tau_eff_theo = tau_run_theo
            tau_eff_ccdf = tau_run_ccdf
            tau_eff_fit = tau_run_fit
            tau_eff_emp = tau_run_emp
            tau_eff_exp_fit = tau_run_theo
        else:
            if np.isfinite(tau_oacf_int) and tau_oacf_int > 0:
                tau_oacf = tau_oacf_int
            elif np.isfinite(tau_oacf_fit) and tau_oacf_fit > 0:
                tau_oacf = tau_oacf_fit
            elif mean_cos_run > 0 and mean_cos_run < 1:
                tau_oacf = -4.0 / np.log(mean_cos_run)  # dt=4s
            else:
                tau_oacf = tau_run_theo

            # 4. 有効緩和時間 tau_eff = 1 / (1/tau_OACF,int + 1/tau_dwell) の算出
            # 1 / tau_eff = 1 / tau_OACF + 1 / tau_dwell
            tau_eff_theo = 1.0 / (1.0 / tau_oacf + 1.0 / tau_run_theo)
            tau_eff_ccdf = 1.0 / (1.0 / tau_oacf + 1.0 / tau_run_ccdf)
            tau_eff_fit = 1.0 / (1.0 / tau_oacf + 1.0 / tau_run_fit)
            tau_eff_emp = 1.0 / (1.0 / tau_oacf + 1.0 / tau_run_emp)

            # 指数フィット型 OACF を用いた場合の tau_eff_fit (参考用)
            if np.isfinite(tau_oacf_fit) and tau_oacf_fit > 0:
                tau_eff_exp_fit = 1.0 / (1.0 / tau_oacf_fit + 1.0 / tau_run_theo)
            else:
                tau_eff_exp_fit = tau_eff_theo

        # Run MSD フィッティングから得られた持続時間 tau_Run,MSD (参考比較用)
        tau_run_msd = np.nan
        Dt_msd = np.nan
        if not df_abp.empty:
            sub_abp = df_abp[df_abp['diameter_um'] == d_um]
            if not sub_abp.empty:
                tau_val = float(sub_abp.iloc[0]['tau_r_s'])
                if np.isfinite(tau_val) and tau_val > 0:
                    tau_run_msd = tau_val
                if 'Dt_um2_s' in sub_abp.columns:
                    Dt_msd = float(sub_abp.iloc[0]['Dt_um2_s'])

        # 5. 能動輸送項 D_active = (1/2) * f_run * v_R^2 * tau_eff
        d_active_theo = 0.5 * pi_run * (v_run ** 2) * tau_eff_theo
        d_active_ccdf = 0.5 * (tau_run_ccdf / (tau_run_ccdf + tau_tumble_ccdf)) * (v_run ** 2) * tau_eff_ccdf
        d_active_fit = 0.5 * (tau_run_fit / (tau_run_fit + tau_tumble_fit)) * (v_run ** 2) * tau_eff_fit
        d_active_emp = 0.5 * f_run_emp * (v_run ** 2) * tau_eff_emp
        d_active_exp_fit = 0.5 * pi_run * (v_run ** 2) * tau_eff_exp_fit

        # 提案モデル式: D_eff = D_SE + (1/2) * f_run * v_R^2 * tau_eff (Stokes-Einstein D_0 を採用)
        d_eff_se_theo = d_0_se + d_active_theo
        d_eff_se_ccdf = d_0_se + d_active_ccdf
        d_eff_se_fit = d_0_se + d_active_fit
        d_eff_se_emp = d_0_se + d_active_emp
        d_eff_se_exp_fit = d_0_se + d_active_exp_fit

        # Tumble 状態の粒子 MSD から見積もった D_0,tumble を用いたモデル値 (参考比較用)
        d_eff_tumble_theo = d_0_tumble + d_active_theo
        d_eff_tumble_ccdf = d_0_tumble + d_active_ccdf

        # 実験推定 Dt を用いた場合の D_eff = D_t + D_active
        d_eff_dt_theo = (Dt_msd if np.isfinite(Dt_msd) else d_0_se) + d_active_theo

        # 従来の min(tau_MSD, tau_dwell) のみを用いた理論値 (参考比較用)
        tau_run_min_msd = min(tau_run_msd, tau_run_theo) if np.isfinite(tau_run_msd) and tau_run_msd > 0 else tau_run_theo
        d_active_min_msd = 0.5 * pi_run * (v_run ** 2) * tau_run_min_msd

        records.append({
            'bead_name': b_name,
            'diameter_um': d_um,
            'v_run_um_s': v_run,
            'v_run_geom_um_s': v_run_geom,
            'v_tumble_um_s': v_tumble,
            'tau_oacf_s': tau_oacf,
            'tau_oacf_int_s': tau_oacf_int,
            'tau_oacf_fit_s': tau_oacf_fit,
            'tau_run_theo_s': tau_run_theo,
            'tau_tumble_theo_s': tau_tumble_theo,
            'tau_run_ccdf_s': tau_run_ccdf,
            'tau_tumble_ccdf_s': tau_tumble_ccdf,
            'tau_run_fit_s': tau_run_fit,
            'tau_tumble_fit_s': tau_tumble_fit,
            'tau_run_emp_s': tau_run_emp,
            'tau_eff_theo_s': tau_eff_theo,
            'tau_eff_ccdf_s': tau_eff_ccdf,
            'tau_eff_exp_fit_s': tau_eff_exp_fit,
            'tau_run_msd_s': tau_run_msd,
            'pi_run': pi_run,
            'pi_tumble': pi_tumble,
            'mean_cos_run': mean_cos_run,
            'D_0_SE': d_0_se,
            'D_0_tumble': d_0_tumble,
            'D_0_tumble_lin': d_0_tumble_lin,
            'D_0_tumble_app': d_0_tumble_app,
            'alpha_tumble': alpha_tumble,
            'Dt_msd': Dt_msd,
            'D_active_theo': d_active_theo,
            'D_active_ccdf': d_active_ccdf,
            'D_active_exp_fit': d_active_exp_fit,
            'D_active_min_msd': d_active_min_msd,
            'D_eff_theo': d_eff_se_theo,  # D_SE + active(tau_eff)
            'D_eff_ccdf': d_eff_se_ccdf,
            'D_eff_fit': d_eff_se_fit,
            'D_eff_emp': d_eff_se_emp,
            'D_eff_exp_fit': d_eff_se_exp_fit,
            'D_eff_tumble_theo': d_eff_tumble_theo,
            'D_eff_se_theo': d_eff_se_theo,
            'D_eff_Dt_theo': d_eff_dt_theo,
            'D_RTP_theo': d_active_theo,
        })

    return pd.DataFrame(records)



def plot_effective_diffusion_comparison(
    df_combined: pd.DataFrame,
    out_dir: Path,
    use_log_scale: bool = True
):
    """
    横軸を粒子径、縦軸を有効拡散係数として
    1. Green-Kubo 実測値
    2. 理論モデル予測: D_eff = D_0 + (1/2) * f_run * v_R^2 * tau_Run
    3. 粒子径から理論的に算出される純粋熱拡散係数: D_SE(d) = k_B * T / (3 * pi * eta * d) (Stokes-Einstein)
    4. 能動輸送増大項のみ: D_active = (1/2) * f_run * v_R^2 * tau_Run
    を比較描画するメイングラフを作成・保存する。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    
    # 粒子径
    d = df_combined['diameter_um'].to_numpy()
    
    # 1. 粒子径から理論的に算出される Stokes-Einstein 熱拡散曲線 D_SE(d)
    d_dense = np.logspace(np.log10(0.4), np.log10(28.0), 200)
    d_0_dense = calc_stokes_einstein_diffusion(d_dense)
    
    ax.plot(
        d_dense, d_0_dense,
        color='#999999',
        linestyle=':',
        linewidth=1.6,
        label=r'Stokes-Einstein Theory ($D_{\mathrm{SE}} \propto d^{-1}$)',
        zorder=2
    )

    # 2. Green-Kubo 実測値 (エラーバー付き: 累積積分系列の中央値 Median)
    if 'D_eff_GK_median_mean' in df_combined.columns:
        d_gk = df_combined['D_eff_GK_median_mean'].to_numpy()
        sem_gk = df_combined['D_eff_GK_median_sem'].to_numpy()
    elif 'D_eff_GK_mean' in df_combined.columns:
        d_gk = df_combined['D_eff_GK_mean'].to_numpy()
        sem_gk = df_combined['D_eff_GK_sem'].to_numpy()
    else:
        d_gk = df_combined['D_eff_GK_max_mean'].to_numpy()
        sem_gk = df_combined['D_eff_GK_max_sem'].to_numpy()
    
    # 終端積分値 (T_max=400s) も比較用にあれば取得
    d_gk_full = df_combined['D_eff_GK_full_mean'].to_numpy() if 'D_eff_GK_full_mean' in df_combined.columns else None
    
    # 終端値の薄いプロット（参考）
    if d_gk_full is not None:
        ax.plot(
            d, d_gk_full,
            marker='x',
            color='#1b9e77',
            linestyle='none',
            markersize=7.0,
            markeredgewidth=1.2,
            alpha=0.45,
            label=r'Green-Kubo Full Limit ($t = T_{\max}$)',
            zorder=6
        )
    
    ax.errorbar(
        d, d_gk, yerr=sem_gk,
        fmt='o',
        color='#1b9e77',
        ecolor='#1b9e77',
        elinewidth=2.2,
        capsize=5.5,
        capthick=1.6,
        markersize=8.5,
        label=r'Green-Kubo Median Integral ($D_{\mathrm{eff}} = \mathrm{median}_t \frac{1}{2}\int_0^t C_v dt''$)',
        zorder=8
    )
    
    # 3. 総合理論モデル: D_eff = D_SE + (1/2) * f_run * v_R^2 * tau_eff
    d_eff_theo = df_combined['D_eff_theo'].to_numpy()
    ax.plot(
        d, d_eff_theo,
        marker='s',
        color='#d95f02',
        linewidth=2.2,
        linestyle='-',
        markersize=7.5,
        label=r'Model: $D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2}f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$',
        zorder=7
    )

    # 4. 能動輸送項のみ: D_active = (1/2) * f_run * v_R^2 * tau_eff
    d_active = df_combined['D_active_theo'].to_numpy()
    ax.plot(
        d, d_active,
        marker='^',
        color='#2b83ba',
        linewidth=1.6,
        linestyle='--',
        markersize=6.5,
        label=r'Active Term: $\frac{1}{2}f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$',
        zorder=5
    )

    ax.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Diffusion Coefficient $D$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=12, fontweight='bold')
    ax.set_title(r'Effective Diffusion: Green-Kubo Median vs $D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2} f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$', fontsize=11, fontweight='bold', pad=10)
    
    if use_log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
        ax.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
        ax.set_xlim(0.45, 28.0)
        ax.set_ylim(0.0008, 3.0)
    else:
        ax.set_xlim(0, 22)
        
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=8.0, loc='upper right', framealpha=0.92)
    
    plt.tight_layout()
    svg_path = out_dir / 'D_eff_vs_diameter.svg'
    png_path = out_dir / 'D_eff_vs_diameter.png'
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] {svg_path}")
    print(f"  [保存完了] {png_path}")

    # 線形スケール版も作成
    fig_lin, ax_lin = plt.subplots(figsize=(7.5, 5.6))
    ax_lin.plot(
        d_dense, d_0_dense, color='#999999', linestyle=':', linewidth=1.6,
        label=r'Stokes-Einstein ($D_{\mathrm{SE}} \propto d^{-1}$)', zorder=2
    )
    if d_gk_full is not None:
        ax_lin.plot(
            d, d_gk_full, marker='x', color='#1b9e77', linestyle='none',
            markersize=7.0, markeredgewidth=1.2, alpha=0.45,
            label=r'Green-Kubo Full Limit ($t = T_{\max}$)', zorder=6
        )
    ax_lin.errorbar(
        d, d_gk, yerr=sem_gk,
        fmt='o', color='#1b9e77', ecolor='#1b9e77', elinewidth=2.2,
        capsize=5.5, capthick=1.6, markersize=8.5,
        label=r'Green-Kubo Median Integral ($D_{\mathrm{eff}} = \mathrm{median}_t \frac{1}{2}\int_0^t C_v dt''$)', zorder=8
    )
    ax_lin.plot(
        d, d_eff_theo, marker='s', color='#d95f02', linewidth=2.2, linestyle='-',
        markersize=7.5, label=r'Model: $D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2}f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$', zorder=7
    )
    ax_lin.plot(
        d, d_active, marker='^', color='#2b83ba', linewidth=1.6, linestyle='--',
        markersize=6.5, label=r'Active Term: $\frac{1}{2}f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$', zorder=5
    )
    ax_lin.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_lin.set_ylabel(r'Diffusion Coefficient $D$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=12, fontweight='bold')
    ax_lin.set_title(r'Effective Diffusion vs Diameter (Linear Scale - Green-Kubo Median)', fontsize=11, fontweight='bold', pad=10)
    ax_lin.set_xlim(0, 22)
    ax_lin.set_ylim(-0.02, 1.6)
    ax_lin.grid(True, linestyle='--', alpha=0.4)
    ax_lin.legend(frameon=True, fontsize=8.0, loc='upper right', framealpha=0.92)
    
    plt.tight_layout()
    svg_lin = out_dir / 'D_eff_vs_diameter_linear.svg'
    png_lin = out_dir / 'D_eff_vs_diameter_linear.png'
    fig_lin.savefig(svg_lin, bbox_inches='tight')
    fig_lin.savefig(png_lin, dpi=300, bbox_inches='tight')
    plt.close(fig_lin)


def plot_detailed_4panel_analysis(
    vacf_ensemble_dict: Dict[str, dict],
    df_combined: pd.DataFrame,
    out_dir: Path
):
    """
    4パネル詳細解析プロットを作成・保存する。
    (a) 未正規化 VACF 減衰曲線
    (b) 累積積分 D_eff(t) の収束挙動
    (c) 有効拡散係数 D_eff vs 粒子径 (Green-Kubo vs RTP理論)
    (d) HMM 微視的パラメータ (v_R, tau_R_eff, pi_R) の粒子径依存性
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.8))
    
    # -------------------------------------------------------------
    # Panel (a): 未正規化 VACF C_v(t) [um^2/s^2]
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in vacf_ensemble_dict:
            continue
        res = vacf_ensemble_dict[b_name]
        t = res['t_lag']
        c_mean = res['c_v_mean']
        
        ax_a.plot(
            t, c_mean,
            marker=item['marker'],
            color=item['color'],
            label=f"{item['label']}",
            linewidth=1.6,
            markersize=4.5,
            alpha=0.9
        )
        
    ax_a.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.7)
    ax_a.set_xlim(0, 40)
    ax_a.set_xlabel(r'Lag time $\Delta t$ [s]', fontsize=11, fontweight='bold')
    ax_a.set_ylabel(r'VACF $C_v(t) = \langle \mathbf{v}(t)\cdot\mathbf{v}(0)\rangle$ [$\mu\mathrm{m}^2/\mathrm{s}^2$]', fontsize=11, fontweight='bold')
    ax_a.set_title(r'(a) Unnormalized Velocity Autocorrelation Function', fontsize=12, fontweight='bold')
    ax_a.grid(True, linestyle='--', alpha=0.4)
    ax_a.legend(frameon=True, fontsize=8.5, loc='upper right')

    # -------------------------------------------------------------
    # Panel (b): 累積積分 D_eff(t) = 0.5 * \int_0^t C_v(t') dt'
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in vacf_ensemble_dict:
            continue
        res = vacf_ensemble_dict[b_name]
        t = res['t_lag']
        cum_d = res['cum_d_ens']
        
        ax_b.plot(
            t, cum_d,
            marker=item['marker'],
            color=item['color'],
            label=f"{item['label']}",
            linewidth=1.6,
            markersize=4.5,
            alpha=0.9
        )
        
    ax_b.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.7)
    ax_b.set_xlim(0, 60)
    ax_b.set_xlabel(r'Upper Integration Limit $t$ [s]', fontsize=11, fontweight='bold')
    ax_b.set_ylabel(r'Cumulative $D_{\mathrm{eff}}(t) = \frac{1}{2}\int_0^t C_v(t'') dt''$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=11, fontweight='bold')
    ax_b.set_title(r'(b) Green-Kubo Integration Convergence', fontsize=12, fontweight='bold')
    ax_b.grid(True, linestyle='--', alpha=0.4)
    ax_b.legend(frameon=True, fontsize=8.5, loc='upper right')

    # -------------------------------------------------------------
    # Panel (c): D_eff vs 粒子径
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    d = df_combined['diameter_um'].to_numpy()
    if 'D_eff_GK_median_mean' in df_combined.columns:
        d_gk = df_combined['D_eff_GK_median_mean'].to_numpy()
        sem_gk = df_combined['D_eff_GK_median_sem'].to_numpy()
    elif 'D_eff_GK_mean' in df_combined.columns:
        d_gk = df_combined['D_eff_GK_mean'].to_numpy()
        sem_gk = df_combined['D_eff_GK_sem'].to_numpy()
    else:
        d_gk = df_combined['D_eff_GK_max_mean'].to_numpy()
        sem_gk = df_combined['D_eff_GK_max_sem'].to_numpy()
    d_eff_theo = df_combined['D_eff_theo'].to_numpy()
    d_active = df_combined['D_active_theo'].to_numpy()
    
    # Stokes-Einstein 理論曲線
    d_dense = np.logspace(np.log10(0.4), np.log10(28.0), 150)
    d_0_dense = calc_stokes_einstein_diffusion(d_dense)
    ax_c.plot(
        d_dense, d_0_dense,
        color='#999999', linestyle=':', linewidth=1.6,
        label=r'Stokes-Einstein ($D_{\mathrm{SE}} \propto d^{-1}$)', zorder=2
    )
    
    ax_c.errorbar(
        d, d_gk, yerr=sem_gk,
        fmt='o', color='#1b9e77', ecolor='#1b9e77', elinewidth=2.0,
        capsize=4.5, capthick=1.5, markersize=8,
        label=r'Green-Kubo Integral (Median: $\mathrm{median}_t \frac{1}{2}\int C_v dt$)', zorder=6
    )
    ax_c.plot(
        d, d_eff_theo, marker='s', color='#d95f02', linewidth=2.0, linestyle='-',
        markersize=7, label=r'Model: $D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2}f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$', zorder=5
    )
    ax_c.plot(
        d, d_active, marker='^', color='#2b83ba', linewidth=1.5, linestyle='--',
        markersize=6, label=r'Active Term: $\frac{1}{2}f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$', zorder=4
    )
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_c.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax_c.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
    ax_c.set_xlim(0.45, 28.0)
    ax_c.set_ylim(0.0008, 3.0)
    ax_c.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax_c.set_ylabel(r'Diffusion Coefficient $D$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=11, fontweight='bold')
    ax_c.set_title(r'(c) Diffusion Coefficients vs Diameter ($D_0 = D_{\mathrm{SE}}$)', fontsize=12, fontweight='bold')
    ax_c.grid(True, which='both', linestyle='--', alpha=0.4)
    ax_c.legend(frameon=True, fontsize=7.5, loc='upper right')

    # -------------------------------------------------------------
    # Panel (d): 微視的持続時間 tau_OACF, tau_dwell, tau_eff vs 粒子径
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d2 = ax_d.twinx()
    
    # 左軸: tau_OACF, tau_dwell, tau_eff
    if 'tau_oacf_s' in df_combined.columns:
        ax_d.plot(d, df_combined['tau_oacf_s'], marker='o', color='#2b83ba', lw=1.6, linestyle=':', label=r'$\tau_{\mathrm{OACF, int}}$ (Orientation Integral Time)')
    ax_d.plot(d, df_combined['tau_run_theo_s'], marker='^', color='#abdda4', lw=1.6, linestyle='--', label=r'$\tau_{\mathrm{dwell}}$ (Run Dwell Time)')
    ax_d.plot(d, df_combined['tau_eff_theo_s'], marker='s', color='#d7191c', lw=2.2, linestyle='-', label=r'$\tau_{\mathrm{eff}} = (\tau_{\mathrm{OACF,int}}^{-1} + \tau_{\mathrm{dwell}}^{-1})^{-1}$')

    # 右軸: Run 速度 v_R
    l_v = ax_d2.plot(d, df_combined['v_run_um_s'], marker='d', color='#fdae61', lw=1.8, linestyle='-.', label=r'Run Speed $v_R$ [$\mu\mathrm{m}/\mathrm{s}$]')
    
    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    ax_d.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_d.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax_d.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
    ax_d.set_xlim(0.45, 28.0)
    
    ax_d.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax_d.set_ylabel(r'Timescales $\tau$ [s]', fontsize=11, fontweight='bold')
    ax_d2.set_ylabel(r'Run Speed $v_R$ [$\mu\mathrm{m}/\mathrm{s}$]', fontsize=11, fontweight='bold', color='#e66101')
    ax_d2.tick_params(axis='y', labelcolor='#e66101')
    
    lines_1, labels_1 = ax_d.get_legend_handles_labels()
    lines_2, labels_2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=True, fontsize=7.5, loc='lower left')
    ax_d.set_title(r'(d) Microscopic Timescales: $\tau_{\mathrm{OACF,int}}, \tau_{\mathrm{dwell}}, \tau_{\mathrm{eff}}$ & $v_R$', fontsize=12, fontweight='bold')
    ax_d.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    svg_4p = out_dir / 'effective_diffusion_detailed_4panel.svg'
    png_4p = out_dir / 'effective_diffusion_detailed_4panel.png'
    fig.savefig(svg_4p, bbox_inches='tight')
    fig.savefig(png_4p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 4パネル詳細図: {svg_4p}")


def plot_extended_cumulative_diffusion_6panel(
    vacf_ensemble_dict: Dict[str, dict],
    df_combined: pd.DataFrame,
    out_dir: Path,
    max_display_time_s: float = 400.0
):
    """
    全6ビーズサイズについて、未正規化 VACF C_v(t) と累積積分 D_eff(t) = 0.5 * int_0^t C_v(t') dt' の
    長時間発展（0 ~ max_display_time_s）およびプラトー収束を示す 6パネル詳細図。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), sharex=True)
    axes = axes.flatten()

    for idx, item in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = item["name"]
        d_um = item["diameter_um"]

        if b_name not in vacf_ensemble_dict:
            ax.set_visible(False)
            continue

        res = vacf_ensemble_dict[b_name]
        t = res['t_lag']
        c_mean = res['c_v_mean']
        cum_d = res['cum_d_ens']
        
        mask = t <= max_display_time_s
        t_sub = t[mask]
        c_sub = c_mean[mask]
        cum_sub = cum_d[mask]

        ax2 = ax.twinx()

        # 個別実験の累積積分（存在する場合）
        if 'exp_cum_d_list' in res:
            for exp_t, exp_cum in res['exp_cum_d_list']:
                m_exp = exp_t <= max_display_time_s
                ax2.plot(exp_t[m_exp], exp_cum[m_exp], color='#fc9272', alpha=0.35, lw=0.9)

        # 1. 左軸: VACF C_v(t)
        l1 = ax.plot(
            t_sub, c_sub,
            color=item['color'],
            marker=item['marker'],
            markersize=3.5,
            linewidth=1.6,
            label=r'VACF $C_v(t)$'
        )
        ax.axhline(0, color='gray', linestyle=':', lw=1.0, alpha=0.7)

        # 2. 右軸: 累積積分 D_eff(t)
        l2 = ax2.plot(
            t_sub, cum_sub,
            color='#de2d26',
            linewidth=2.2,
            linestyle='-',
            label=r'Cumulative $D_{\mathrm{eff}}(t)$'
        )

        # 最大ピーク値の検出とマーキング
        peak_idx = np.argmax(cum_sub)
        t_peak_val = t_sub[peak_idx]
        d_peak_val = cum_sub[peak_idx]
        l3 = ax2.plot(
            [t_peak_val], [d_peak_val],
            marker='*', color='#b10026', markersize=10.0,
            linestyle='none', label=r'Peak Maximum $D_{\max}$'
        )

        # 漸近プラトー値および中央値の表示
        d_final = cum_sub[-1] if len(cum_sub) > 0 else 0.0
        d_exp_med_mean = res.get('d_eff_median_mean', np.median(cum_sub))
        d_exp_med_sem = res.get('d_eff_median_sem', 0.0)
        d_exp_peak_mean = res.get('d_eff_max_mean', d_peak_val)
        d_exp_peak_sem = res.get('d_eff_max_sem', 0.0)
        
        # タイトルと注釈
        ax.set_title(f"{item['label']} ($d = {d_um}\\,\\mu\\mathrm{{m}}$)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

        # 軸ラベル
        ax.set_ylabel(r'VACF $C_v(t)$ [$\mu\mathrm{m}^2/\mathrm{s}^2$]', fontsize=10, fontweight='bold', color=item['color'])
        ax.tick_params(axis='y', labelcolor=item['color'])
        ax2.set_ylabel(r'$D_{\mathrm{eff}}(t)$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=10, fontweight='bold', color='#de2d26')
        ax2.tick_params(axis='y', labelcolor='#de2d26')

        # 注釈テキスト: 中央値、ピーク値、終端値
        txt = (
            f"$\\mathbf{{Median\\;D}} = {d_exp_med_mean:.4f} \\pm {d_exp_med_sem:.4f}\\,\\mu\\mathrm{{m^2/s}}$\n"
            f"$\\mathrm{{Peak\\;D}} = {d_exp_peak_mean:.4f} \\pm {d_exp_peak_sem:.4f}\\,\\mu\\mathrm{{m^2/s}}$ (at ${t_peak_val:.0f}\\mathrm{{s}}$)\n"
            f"$D({int(max_display_time_s)}\\mathrm{{s}}) = {d_final:.4f}\\,\\mu\\mathrm{{m^2/s}}$"
        )
        ax.text(
            0.45, 0.85, txt,
            transform=ax.transAxes,
            fontsize=8.0,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88, edgecolor='#cccccc'),
            zorder=6
        )

        if idx >= 3:
            ax.set_xlabel(r'Lag time $t$ [s]', fontsize=11, fontweight='bold')

        # 凡例 (第1パネルのみ)
        if idx == 0:
            lines = l1 + l2 + l3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='lower right', fontsize=8.0, frameon=True, framealpha=0.9)

    fig.suptitle(
        r"Extended Green-Kubo Integration: VACF $C_v(t)$ and Cumulative $D_{\mathrm{eff}}(t) = \frac{1}{2}\int_0^t C_v(t') dt'$ up to $T_{\max} = " + f"{int(max_display_time_s)}" + r"\,\mathrm{s}$",
        fontsize=13.5,
        fontweight='bold',
    )
    plt.tight_layout()

    svg_path = out_dir / 'D_eff_cumulative_extended_6panel.svg'
    png_path = out_dir / 'D_eff_cumulative_extended_6panel.png'
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 6パネル拡張累積積分図: {svg_path}")


def plot_diffusion_vs_cutoff_comparison(
    vacf_ensemble_dict: Dict[str, dict],
    df_combined: pd.DataFrame,
    out_dir: Path,
    cutoffs: List[float] = [50.0, 100.0, 200.0, 300.0, 400.0]
):
    """
    異なる積分上限時間 T_cutoff における Green-Kubo D_eff(T_cutoff) の粒子径依存性比較プロット。
    上限時間を増やした際の実効拡散係数の安定性（ロバスト性）を検証する。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.6))

    d = df_combined['diameter_um'].to_numpy()
    
    # 粒子径から理論的に算出される Stokes-Einstein 熱拡散曲線
    d_dense = np.logspace(np.log10(0.4), np.log10(28.0), 150)
    d_0_dense = calc_stokes_einstein_diffusion(d_dense)
    ax.plot(
        d_dense, d_0_dense,
        color='#7f7f7f', linestyle=':', linewidth=1.8,
        label=r'Stokes-Einstein Theory ($D_0(d) = \frac{k_B T}{3\pi\eta d}$)',
        zorder=2
    )

    # 各カットオフ時間における D_eff(T_cutoff)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cutoffs)))
    markers = ['o', 's', '^', 'd', 'v']

    for c_idx, t_cut in enumerate(cutoffs):
        d_vals = []
        for item in BEADS_INFO:
            b_name = item["name"]
            if b_name not in vacf_ensemble_dict:
                d_vals.append(np.nan)
                continue
            res = vacf_ensemble_dict[b_name]
            t = res['t_lag']
            cum_d = res['cum_d_ens']
            
            # t <= t_cut の最後の値
            sub_cum = cum_d[t <= t_cut]
            if len(sub_cum) > 0:
                d_vals.append(sub_cum[-1])
            else:
                d_vals.append(np.nan)
                
        d_vals = np.array(d_vals)
        ax.plot(
            d, d_vals,
            marker=markers[c_idx % len(markers)],
            color=colors[c_idx],
            linewidth=1.8,
            linestyle='-',
            markersize=7.0,
            label=f"$T_{{\\max}} = {int(t_cut)}\\,$s",
            zorder=4 + c_idx
        )

    # 理論モデル D_eff = D_SE + active
    d_eff_theo = df_combined['D_eff_theo'].to_numpy()
    ax.plot(
        d, d_eff_theo,
        marker='*',
        color='#d95f02',
        linewidth=2.0,
        linestyle='--',
        markersize=9.0,
        label=r'Theory: $D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2}f_{\mathrm{run}}v_R^2\tau_{\mathrm{eff}}$',
        zorder=10
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
    ax.set_xlim(0.45, 28.0)
    ax.set_ylim(0.005, 3.0)

    ax.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Effective Diffusion Coefficient $D_{\mathrm{eff}}$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=12, fontweight='bold')
    ax.set_title(r'Green-Kubo $D_{\mathrm{eff}}$ vs Upper Integration Limit $T_{\max}$', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=8.5, loc='upper right', framealpha=0.92)

    plt.tight_layout()
    svg_cut = out_dir / 'D_eff_vs_integration_cutoff.svg'
    png_cut = out_dir / 'D_eff_vs_integration_cutoff.png'
    fig.savefig(svg_cut, bbox_inches='tight')
    fig.savefig(png_cut, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 積分上限感度プロット: {svg_cut}")


def plot_tau_eff_vs_diameter(
    df_combined: pd.DataFrame,
    out_dir: Path
):
    """
    粒子径ごとの有効持続時間 tau_eff および構成要素 (tau_OACF,int, tau_dwell) の
    粒子径依存性グラフ (tau_eff_vs_diameter.png / .svg) を作成・保存する。
    1 / tau_eff = 1 / tau_OACF,int + 1 / tau_dwell
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = df_combined['diameter_um'].to_numpy()
    tau_eff = df_combined['tau_eff_theo_s'].to_numpy()
    tau_dwell = df_combined['tau_run_theo_s'].to_numpy()
    tau_oacf = df_combined['tau_oacf_int_s'].to_numpy() if 'tau_oacf_int_s' in df_combined.columns else df_combined['tau_oacf_s'].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5.6))

    # 1. tau_OACF,int (Orientation 自己相関積分時間)
    finite_oacf = np.isfinite(tau_oacf) & (tau_oacf > 0)
    ax.plot(
        d[finite_oacf], tau_oacf[finite_oacf],
        marker='o',
        color='#2b83ba',
        linewidth=1.8,
        linestyle=':',
        markersize=7.5,
        label=r'$\tau_{\mathrm{OACF, int}}$ (Orientation Integral Time)',
        zorder=3
    )

    # 2. tau_dwell (Run 状態滞在時間)
    ax.plot(
        d, tau_dwell,
        marker='^',
        color='#4dac26',
        linewidth=1.8,
        linestyle='--',
        markersize=7.5,
        label=r'$\tau_{\mathrm{dwell}}$ (Run Dwell Time: $\tau_{\mathrm{run}}$)',
        zorder=4
    )

    # 3. tau_eff (調和結合による有効持続時間) - 主系列
    ax.plot(
        d, tau_eff,
        marker='s',
        color='#d7191c',
        linewidth=2.6,
        linestyle='-',
        markersize=9.0,
        label=r'$\tau_{\mathrm{eff}} = \left(\tau_{\mathrm{OACF, int}}^{-1} + \tau_{\mathrm{dwell}}^{-1}\right)^{-1}$',
        zorder=5
    )

    # 各 tau_eff 点に数値アノテーション & 0.63 um 注釈
    for i_d, (x_val, y_val, t_o) in enumerate(zip(d, tau_eff, tau_oacf)):
        if not np.isfinite(t_o):
            ax.annotate(
                r'$\tau_{\mathrm{OACF}} = \infty$' + '\n' + r'($\tau_{\mathrm{eff}} = \tau_{\mathrm{dwell}}$)',
                xy=(x_val, y_val),
                xytext=(x_val * 1.15, y_val * 1.35),
                arrowprops=dict(arrowstyle="->", color='#2b83ba', lw=1.2),
                fontsize=8.5,
                fontweight='bold',
                color='#2b83ba',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2b83ba', alpha=0.9)
            )
        ax.annotate(
            f"{y_val:.2f}s",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, -16 if i_d == 0 else 10),
            ha='center',
            fontsize=8.5,
            fontweight='bold',
            color='#d7191c',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#d7191c', alpha=0.85)
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
    ax.set_xlim(0.45, 28.0)
    ax.set_ylim(0.5, 450.0)

    ax.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Effective Timescale $\tau$ [s]', fontsize=12, fontweight='bold')
    ax.set_title(r'Effective Persistence Time $\tau_{\mathrm{eff}}$ vs Particle Diameter', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=8.5, loc='upper right', framealpha=0.92)

    plt.tight_layout()
    svg_tau = out_dir / 'tau_eff_vs_diameter.svg'
    png_tau = out_dir / 'tau_eff_vs_diameter.png'
    fig.savefig(svg_tau, bbox_inches='tight')
    fig.savefig(png_tau, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] tau_eff 単独グラフ: {svg_tau}")
    print(f"  [保存完了] tau_eff 単独グラフ: {png_tau}")


def run_effective_diffusion_analysis(
    root_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    hmm_summary_path: Optional[Union[str, Path]] = None,
    angle_summary_path: Optional[Union[str, Path]] = None,
    abp_summary_path: Optional[Union[str, Path]] = None,
    state_msd_fits_path: Optional[Union[str, Path]] = None,
    state_msd_curves_path: Optional[Union[str, Path]] = None,
    autocorr_summary_path: Optional[Union[str, Path]] = None,
    max_timeshift_frames: int = 100,
    frame_interval: float = 4.0,
    scale: float = 0.11,
    max_lag_time: float = 400.0,
    temperature_K: float = 298.15,
    viscosity_Pa_s: float = 1.0e-3
) -> pd.DataFrame:
    """
    全ビーズサイズの Green-Kubo 積分および RTP 理論有効拡散係数を一括解析・可視化する。
    """
    if root_dir is None:
        root_dir = va.find_default_root()
    root_dir = Path(root_dir)
    
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / 'figure' / 'effective_diffusion'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    workspace_root = Path(__file__).parent.parent
    if hmm_summary_path is None:
        cand = [
            workspace_root / 'figure' / 'hmm_1d' / 'hmm_state_parameters_summary_k2.csv',
            root_dir / 'figure' / 'hmm_1d' / 'hmm_state_parameters_summary_k2.csv',
        ]
        hmm_summary_path = next((p for p in cand if p.exists()), cand[0])
    hmm_summary_path = Path(hmm_summary_path)

    if angle_summary_path is None:
        cand = [
            workspace_root / 'figure' / 'hmm_1d' / 'hmm_turning_angle_summary_k2.csv',
            root_dir / 'figure' / 'hmm_1d' / 'hmm_turning_angle_summary_k2.csv',
        ]
        angle_summary_path = next((p for p in cand if p.exists()), cand[0])
    angle_summary_path = Path(angle_summary_path)

    if abp_summary_path is None:
        cand = [
            workspace_root / 'figure' / 'hmm_1d' / 'hmm_run_abp_fits_summary_k2.csv',
            root_dir / 'figure' / 'hmm_1d' / 'hmm_run_abp_fits_summary_k2.csv',
        ]
        abp_summary_path = next((p for p in cand if p.exists()), cand[0])
    abp_summary_path = Path(abp_summary_path)

    if state_msd_fits_path is None:
        cand = [
            workspace_root / 'figure' / 'hmm_1d' / 'hmm_state_msd_fits_k2.csv',
            root_dir / 'figure' / 'hmm_1d' / 'hmm_state_msd_fits_k2.csv',
        ]
        state_msd_fits_path = next((p for p in cand if p.exists()), cand[0])
    state_msd_fits_path = Path(state_msd_fits_path)

    if state_msd_curves_path is None:
        cand = [
            workspace_root / 'figure' / 'hmm_1d' / 'hmm_state_msd_curves_k2.csv',
            root_dir / 'figure' / 'hmm_1d' / 'hmm_state_msd_curves_k2.csv',
        ]
        state_msd_curves_path = next((p for p in cand if p.exists()), cand[0])
    state_msd_curves_path = Path(state_msd_curves_path)

    if autocorr_summary_path is None:
        cand = [
            workspace_root / 'figure' / 'hmm_1d' / 'hmm_autocorrelation_summary_k2.csv',
            root_dir / 'figure' / 'hmm_1d' / 'hmm_autocorrelation_summary_k2.csv',
        ]
        autocorr_summary_path = next((p for p in cand if p.exists()), cand[0])
    autocorr_summary_path = Path(autocorr_summary_path)

    print(f"\n{'='*75}")
    print("有効拡散係数（Green-Kubo 積分 vs HMM RTP 理論予測 [D_0 = D_SE + active(tau_eff)]）一括解析")
    print(f"ルートディレクトリ: {root_dir}")
    print(f"出力先ディレクトリ: {out_dir}")
    print(f"HMM サマリーパス : {hmm_summary_path}")
    print(f"Tumble MSD パス  : {state_msd_fits_path}")
    print(f"OACF 自己相関パス: {autocorr_summary_path}")
    print(f"最大ラグ時間     : {max_timeshift_frames} frames ({max_timeshift_frames * frame_interval:.0f}s)")
    print(f"{'='*75}\n")

    # 1. VACF の計算 & Green-Kubo 積分
    gk_summary_list = []
    vacf_ensemble_dict = {}
    
    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        target_dir = root_dir / b_name
        
        if not target_dir.exists():
            print(f"  ディレクトリが見つかりません: {target_dir}")
            continue

        print(f"  [VACF Green-Kubo] 計算中: {item['label']} ({b_name}) ...", flush=True)
        evacf_df = va.vacf_dir(
            target_dir,
            max_timeshift_frames=max_timeshift_frames,
            frame_interval=frame_interval,
            scale=scale,
            mode='velocity',
            normalize=False
        )
        
        if evacf_df.empty:
            print(f"    -> データなし: {b_name}")
            continue

        df_exp_res, ens_summary = integrate_vacf_green_kubo(evacf_df, max_lag_time=max_lag_time)
        
        # 実験ごとの累積積分リストも保持
        exp_cum_list = []
        for _, r_exp in df_exp_res.iterrows():
            exp_cum_list.append((r_exp['t'], r_exp['cum_d_eff']))
        ens_summary['exp_cum_d_list'] = exp_cum_list

        vacf_ensemble_dict[b_name] = ens_summary

        gk_summary_list.append({
            'bead_name': b_name,
            'diameter_um': d_um,
            'n_experiments': ens_summary['n_exps'],
            'D_eff_GK_median_mean': ens_summary['d_eff_median_mean'],
            'D_eff_GK_median_std': ens_summary['d_eff_median_std'],
            'D_eff_GK_median_sem': ens_summary['d_eff_median_sem'],
            'D_eff_GK_median_median': ens_summary['d_eff_median_median'],
            'D_eff_GK_ens_median': ens_summary['d_eff_ens_median'],
            'D_eff_GK_max_mean': ens_summary['d_eff_max_mean'],
            'D_eff_GK_max_std': ens_summary['d_eff_max_std'],
            'D_eff_GK_max_sem': ens_summary['d_eff_max_sem'],
            't_peak_mean_s': ens_summary['mean_t_peak'],
            'D_eff_GK_ens_max': ens_summary['d_eff_ens_max'],
            't_ens_peak_s': ens_summary['t_ens_peak'],
            'D_eff_GK_full_mean': ens_summary['d_eff_mean'],
            'D_eff_GK_full_std': ens_summary['d_eff_std'],
            'D_eff_GK_full_sem': ens_summary['d_eff_sem'],
            'D_eff_GK_mean': ens_summary['d_eff_median_mean'],  # デフォルトを中央値（Median）に設定
            'D_eff_GK_sem': ens_summary['d_eff_median_sem'],
            'D_eff_GK_pos_mean': ens_summary['d_eff_pos_mean'],
            'D_eff_GK_pos_sem': ens_summary['d_eff_pos_sem'],
            'D_eff_GK_ens_trapz': ens_summary['d_eff_ens_full'],
            'D_eff_GK_ens_pos': ens_summary['d_eff_ens_pos'],
            'C_v_0': ens_summary['c_v_0']
        })
        print(f"    -> Green-Kubo Median D_eff = {ens_summary['d_eff_median_mean']:.4f} ± {ens_summary['d_eff_median_sem']:.4f} um^2/s (Peak: {ens_summary['d_eff_max_mean']:.4f} ± {ens_summary['d_eff_max_sem']:.4f}, N={ens_summary['n_exps']})")
        print(f"       (Full limit at {max_lag_time:.0f}s: {ens_summary['d_eff_mean']:.4f} ± {ens_summary['d_eff_sem']:.4f} um^2/s)")

    df_gk = pd.DataFrame(gk_summary_list)

    # 2. HMM パラメータからの RTP 理論計算 (D_0 = D_0,tumble & 1/tau_eff = 1/tau_OACF + 1/tau_dwell)
    if not hmm_summary_path.exists():
        raise FileNotFoundError(f"HMM state parameters file not found: {hmm_summary_path}")
        
    print(f"\n  [HMM RTP 理論モデル] パラメータ読み込み & 理論計算中 (D_0 = D_SE, 1/tau_eff = 1/tau_OACF + 1/tau_dwell) ...")
    df_rtp = compute_rtp_theoretical_diffusion(
        hmm_summary_path,
        angle_summary_path,
        abp_summary_path,
        state_msd_fits_path=state_msd_fits_path,
        state_msd_curves_path=state_msd_curves_path,
        autocorr_summary_path=autocorr_summary_path,
        temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s
    )

    # 3. データの統合
    df_combined = pd.merge(df_gk, df_rtp, on=['bead_name', 'diameter_um'])

    # 4. CSV サマリーの保存
    csv_path = out_dir / 'effective_diffusion_summary.csv'
    df_combined.to_csv(csv_path, index=False)
    print(f"\n  [保存完了] サマリー CSV: {csv_path}")

    # 5. グラフの作成
    print(f"\n  [プロット生成中] メイン比較図、4パネル詳細図、tau_eff 独立グラフ、6パネル長時間累積積分図、上限感度図 ...")
    plot_effective_diffusion_comparison(df_combined, out_dir, use_log_scale=True)
    plot_detailed_4panel_analysis(vacf_ensemble_dict, df_combined, out_dir)
    plot_tau_eff_vs_diameter(df_combined, out_dir)
    plot_extended_cumulative_diffusion_6panel(vacf_ensemble_dict, df_combined, out_dir, max_display_time_s=max_lag_time)
    plot_diffusion_vs_cutoff_comparison(
        vacf_ensemble_dict,
        df_combined,
        out_dir,
        cutoffs=[50.0, 100.0, 200.0, 300.0, min(max_lag_time, 400.0)]
    )

    print(f"\n{'='*75}")
    print("全解析・可視化が完了しました。")
    print(f"{'='*75}\n")

    return df_combined


