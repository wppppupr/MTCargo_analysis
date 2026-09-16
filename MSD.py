import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
from scipy.optimize import curve_fit
import os
import sys
import glob
from pathlib import Path

# libsモジュールの読み込み
from libs import fit_model as fm
from libs import displacement as dpm
from libs import cal_vel as cv

# スタイルの適用
style_path = Path(__file__).parent / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# データルートディレクトリの候補
POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s"},
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


mypass = find_default_root()


def calc_particle_alpha(sub_df: pd.DataFrame, min_t: float = 4.0, max_t: float = 300.0) -> float:
    """
    個別粒子MSD (iMSD) に対して対数空間でべき乗則 MSD ~ t^alpha をフィッティングし、
    異常拡散指数 alpha を算出する。
    """
    mask = (sub_df['lag time'] >= min_t) & (sub_df['lag time'] <= max_t) & (sub_df['MSD'] > 0)
    if np.sum(mask) < 3:
        mask = (sub_df['MSD'] > 0)
    if np.sum(mask) < 3:
        return np.nan
    dt = sub_df['lag time'][mask].to_numpy(dtype=float)
    msd = sub_df['MSD'][mask].to_numpy(dtype=float)
    slope, _ = np.polyfit(np.log10(dt), np.log10(msd), 1)
    return float(slope)


def concat_pooled_particles_MSD(folder, interval_list, scale = 0.11, alpha_threshold = 0.5):
    """
    全動画から追跡された全粒子の個別MSD (iMSD) を1つのプールに統合する。
    各粒子について異常拡散指数 alpha を算出し、alpha > alpha_threshold (デフォルト: 0.5) の粒子のみを抽出する。
    （異常値・スタック粒子 alpha <= 0.5 をフィルタリング除外）
    """
    all_imsds = []
    file_list = sorted(glob.glob(str(Path(folder) / "*" / "*" / "beads_tracks.csv")))
    for vid_idx, file_path in enumerate(file_list):
        interval = interval_list[vid_idx] if vid_idx < len(interval_list) else interval_list[-1]
        track = cv.cal(pd.read_csv(file_path), scale=scale, frame_interval=interval)
        imsd_df = dpm.imsd(track, scale=scale, time_scale=interval, display=False)
        imsd_df['video_idx'] = vid_idx
        imsd_df['unique_particle_id'] = f"vid{vid_idx}_" + imsd_df['particle'].astype(str)
        all_imsds.append(imsd_df)
        
    if len(all_imsds) == 0:
        empty_df = pd.DataFrame(columns=['unique_particle_id', 'video_idx', 'particle', 'lag time', 'MSD', 'alpha'])
        return empty_df, empty_df, 0, 0
        
    df_pool = pd.concat(all_imsds, ignore_index=True)
    df_pool['MSD'] = pd.to_numeric(df_pool['MSD'], errors='coerce')
    df_pool['lag time'] = pd.to_numeric(df_pool['lag time'], errors='coerce')
    df_pool = df_pool.dropna(subset=['MSD', 'lag time'])
    
    # 全個別粒子ごとの alpha を算出
    p_alphas = {}
    for pid, grp in df_pool.groupby('unique_particle_id'):
        p_alphas[pid] = calc_particle_alpha(grp, 4.0, 300.0)
        
    df_pool['alpha'] = df_pool['unique_particle_id'].map(p_alphas)
    
    n_total = df_pool['unique_particle_id'].nunique()
    df_filtered = df_pool[df_pool['alpha'] > alpha_threshold].copy()
    n_filtered = df_filtered['unique_particle_id'].nunique()
    
    return df_filtered, df_pool, n_total, n_filtered


def calc_pooled_MSD_stats(df_pool: pd.DataFrame, min_particles: int = 1) -> pd.DataFrame:
    """
    プールされた全粒子MSDから、lag timeごとのアンサンブル平均値 (mean)、
    四分位範囲 IQR (25%, 75%タイル)、10%, 90%タイル、標準偏差、粒子数を算出する。
    """
    if df_pool.empty:
        return pd.DataFrame(columns=['mean', 'q25', 'q75', 'q10', 'q90', 'median', 'std', 'count'])
    grouped = df_pool.groupby('lag time')['MSD']
    stats_df = pd.DataFrame({
        'mean': grouped.mean(),
        'q25': grouped.quantile(0.25),
        'q75': grouped.quantile(0.75),
        'q10': grouped.quantile(0.10),
        'q90': grouped.quantile(0.90),
        'median': grouped.median(),
        'std': grouped.std(),
        'count': grouped.count()
    })
    if min_particles > 1:
        stats_df = stats_df[stats_df['count'] >= min_particles]
    return stats_df


def concat_dimensionless_MSD(folder, interval_list, Rc, scale = 0.11, alpha_threshold = 0.5):
    all_imsds = []
    file_list = sorted(glob.glob(str(Path(folder) / "*" / "*" / "beads_tracks.csv")))
    for vid_idx, file_path in enumerate(file_list):
        interval = interval_list[vid_idx] if vid_idx < len(interval_list) else interval_list[-1]
        track = cv.cal(pd.read_csv(file_path), scale=scale, frame_interval=interval)
        imsd_df = dpm.imsd(track, scale=scale, time_scale=interval, display=False)
        imsd_df['video_idx'] = vid_idx
        imsd_df['unique_particle_id'] = f"vid{vid_idx}_" + imsd_df['particle'].astype(str)
        
        vel_path = Path(file_path).parent / "velocities_mean.csv"
        if vel_path.exists():
            v0 = pd.read_csv(vel_path)['mean_velocity'].mean()
        else:
            v0 = 1.0
            
        d_MT = 0.025
        Tc = d_MT / v0
        imsd_df['dim_lag_time'] = imsd_df['lag time'] / Tc
        imsd_df['dim_MSD'] = imsd_df['MSD'] / (d_MT**2)
        imsd_df['exp'] = vid_idx
        all_imsds.append(imsd_df)
        
    if len(all_imsds) == 0:
        return pd.DataFrame(columns=['unique_particle_id', 'exp', 'dim_lag_time', 'dim_MSD', 'alpha'])
        
    df_dim = pd.concat(all_imsds, ignore_index=True)
    df_dim['MSD'] = pd.to_numeric(df_dim['MSD'], errors='coerce')
    df_dim['lag time'] = pd.to_numeric(df_dim['lag time'], errors='coerce')
    df_dim = df_dim.dropna(subset=['MSD', 'lag time'])
    
    p_alphas = {}
    for pid, grp in df_dim.groupby('unique_particle_id'):
        p_alphas[pid] = calc_particle_alpha(grp, 4.0, 300.0)
    df_dim['alpha'] = df_dim['unique_particle_id'].map(p_alphas)
    
    df_dim_filtered = df_dim[df_dim['alpha'] > alpha_threshold].copy()
    return df_dim_filtered


def calc_dimensionless_MSD(dim_msd_df):
    grouped = dim_msd_df.groupby('lag time')
    emsd = grouped['dim_MSD'].mean().astype(float)
    elag = grouped['dim_lag_time'].mean().astype(float)
    N = 1 + int(dim_msd_df["exp"].max()) if not dim_msd_df.empty else 1
    emsd_err = grouped['dim_MSD'].std().astype(float)

    return elag, emsd, emsd_err, N


def fit_power_law(stats_df, min_t = 60, max_t = 300):
    mask = (stats_df.index >= min_t) & (stats_df.index <= max_t) & (stats_df['count'] >= 2)
    dt_vals = stats_df.index[mask].to_numpy(dtype=float)
    msd_vals = stats_df['mean'][mask].to_numpy(dtype=float)
    if len(dt_vals) > 2:
        popt, pcov = curve_fit(fm.ln_pl, np.log10(dt_vals), np.log10(msd_vals))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])


def fit_dimensionless(dim_msd_df, min_t=20, max_t=144):
    popt_list = []
    pcov_list = []
    for i in dim_msd_df["exp"].unique():
        sub_df = dim_msd_df[dim_msd_df["exp"] == i]
        mask = (sub_df['lag time'] >= min_t) & (sub_df['lag time'] <= max_t)
        if np.sum(mask) > 2:
            popt, pcov = curve_fit(fm.ln_pl, np.log10(np.float64(sub_df["dim_lag_time"][mask])), np.log10(np.float64(sub_df["dim_MSD"].to_numpy()[mask])))
            popt_list.append(popt)
            pcov_list.append(pcov)
    if len(popt_list) > 0:
        return np.array(popt_list), np.array(pcov_list)
    else:
        return np.array([[np.nan, np.nan]]), np.array([[[np.nan, np.nan], [np.nan, np.nan]]])


def calc_local_alpha(lag, msd, max_physical_lag=400):
    if hasattr(lag, 'index') and not isinstance(lag, pd.Index):
        phys_lag = lag.index.values
    else:
        phys_lag = np.array(lag)
        
    mask = phys_lag <= max_physical_lag
    
    lag_f = np.array(lag)[mask]
    msd_f = np.array(msd)[mask]
    
    log_lag = np.log10(np.float64(lag_f))
    log_msd = np.log10(np.float64(msd_f))
    alpha_local = np.gradient(log_msd, log_lag)
    return lag_f, alpha_local


def func(popt_list):
    mean = np.mean(popt_list, axis=0)
    err = np.std(popt_list, axis=0)
    return mean, err


def load_rtp_parameters(
    root_dir: Path = None,
    hmm_summary_path: Path = None,
    autocorr_summary_path: Path = None
) -> dict:
    """
    HMM 2状態モデルの解析結果CSV（hmm_state_parameters_summary_k2.csv）および
    状態別自己相関CSV（hmm_autocorrelation_summary_k2.csv）から最新の
    f_run, v_R, tau_dwell, tau_OACF_int を直接読み込み、
    1 / tau_eff = 1 / tau_OACF_int + 1 / tau_dwell
    tau_eff = 1 / (1 / tau_OACF_int + 1 / tau_dwell) を算出して辞書として返す。
    """
    workspace_dir = Path(__file__).parent.resolve()
    if root_dir is None:
        root_dir = find_default_root()
        
    if hmm_summary_path is None:
        cand1 = Path(root_dir) / "figure" / "hmm_1d" / "hmm_state_parameters_summary_k2.csv"
        cand2 = workspace_dir / "figure" / "hmm_1d" / "hmm_state_parameters_summary_k2.csv"
        hmm_summary_path = cand1 if cand1.exists() else cand2
        
    if autocorr_summary_path is None:
        cand1 = Path(root_dir) / "figure" / "hmm_1d" / "hmm_autocorrelation_summary_k2.csv"
        cand2 = workspace_dir / "figure" / "hmm_1d" / "hmm_autocorrelation_summary_k2.csv"
        autocorr_summary_path = cand1 if cand1.exists() else cand2
        
    print(f"Loading HMM state parameters from: {hmm_summary_path}")
    print(f"Loading Autocorrelation summary from: {autocorr_summary_path}")

    df_hmm = pd.read_csv(hmm_summary_path) if (hmm_summary_path and Path(hmm_summary_path).exists()) else pd.DataFrame()
    df_autocorr = pd.read_csv(autocorr_summary_path) if (autocorr_summary_path and Path(autocorr_summary_path).exists()) else pd.DataFrame()
    
    params_dict = {}
    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        
        f_run = np.nan
        v_R = np.nan
        tau_dwell = np.nan
        tau_oacf_int = np.nan
        
        if not df_hmm.empty:
            sub_hmm = df_hmm[(np.isclose(df_hmm['diameter_um'], d_um, atol=0.05)) & (df_hmm['state'] == 1)]
            if not sub_hmm.empty:
                f_run = float(sub_hmm.iloc[0]['stationary_prob'])
                v_R = float(sub_hmm.iloc[0]['mean_speed_model_um_s'])
                if 'tau_ccdf_s' in sub_hmm.columns and np.isfinite(sub_hmm.iloc[0]['tau_ccdf_s']):
                    tau_dwell = float(sub_hmm.iloc[0]['tau_ccdf_s'])
                elif 'theoretical_dwell_time_s' in sub_hmm.columns and np.isfinite(sub_hmm.iloc[0]['theoretical_dwell_time_s']):
                    tau_dwell = float(sub_hmm.iloc[0]['theoretical_dwell_time_s'])
                elif 'tau_fit_pdf_s' in sub_hmm.columns and np.isfinite(sub_hmm.iloc[0]['tau_fit_pdf_s']):
                    tau_dwell = float(sub_hmm.iloc[0]['tau_fit_pdf_s'])
                
        if not df_autocorr.empty:
            sub_ac = df_autocorr[(np.isclose(df_autocorr['diameter_um'], d_um, atol=0.05)) & (df_autocorr['state'] == 1) & (df_autocorr['mode'] == 'oacf')]
            if not sub_ac.empty:
                val_int = float(sub_ac.iloc[0]['tau_int_zero_s'])
                if np.isfinite(val_int) and val_int > 0:
                    tau_oacf_int = val_int
                
        # tau_eff の決定: 1 / tau_eff = 1 / tau_OACF,int + 1 / tau_dwell
        if np.isfinite(tau_oacf_int) and tau_oacf_int > 0 and np.isfinite(tau_dwell) and tau_dwell > 0:
            tau_eff = 1.0 / (1.0 / tau_oacf_int + 1.0 / tau_dwell)
            D_active = 0.5 * f_run * (v_R ** 2) * tau_eff
            model_type = "2-State RTP"
        elif np.isfinite(tau_dwell) and tau_dwell > 0:
            tau_eff = tau_dwell
            D_active = 0.5 * f_run * (v_R ** 2) * tau_eff
            model_type = "2-State RTP"
        else:
            tau_eff = np.nan
            D_active = 0.0
            model_type = "Pure Diffusion (4D0t)"
            
        params_dict[d_um] = {
            "bead_name": b_name,
            "diameter_um": d_um,
            "f_run": f_run,
            "v_R": v_R,
            "tau_dwell": tau_dwell,
            "tau_oacf_int": tau_oacf_int,
            "tau_eff": tau_eff,
            "D_active": D_active,
            "model_type": model_type
        }
    return params_dict


def fit_rtp_2state_msd(
    pool_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    f_run: float,
    v_R: float,
    tau_eff: float,
    min_t: float = 4.0,
    max_t: float = 300.0,
    min_particles: int = 2
) -> dict:
    """
    2状態RTP理論モデルに基づいてプール粒子のアンサンブル平均MSD（対数Log-Log空間）にフィッティングを実施し、
    D_0, sigma_noise_sq, D_eff を推定・算出する。
    個別粒子ごとのフィッティングも行い、粒子間のばらつき（標準偏差）を算出する。
    """
    # 1. 個別粒子ごとのフィッティング
    part_results = []
    for p_id in pool_df['unique_particle_id'].unique():
        sub_df = pool_df[pool_df['unique_particle_id'] == p_id]
        mask_p = (sub_df['lag time'] >= min_t) & (sub_df['lag time'] <= max_t)
        if np.sum(mask_p) < 3:
            continue
        dt_vals = sub_df['lag time'][mask_p].to_numpy(dtype=float)
        msd_vals = sub_df['MSD'][mask_p].to_numpy(dtype=float)
        
        def fit_model_p(dt, D0, sigma_sq):
            return fm.log_rtp_2state_msd(dt, D0, sigma_sq, f_run, v_R, tau_eff)
        target_p = np.log10(np.maximum(msd_vals, 1e-12))
        try:
            popt, _ = curve_fit(
                fit_model_p, dt_vals, target_p,
                p0=[1e-3, 1e-4],
                bounds=([0.0, 0.0], [100.0, 100.0]),
                maxfev=5000
            )
            D0_val, sigma_sq_val = popt[0], popt[1]
            D_active = 0.5 * f_run * (v_R ** 2) * tau_eff if (np.isfinite(tau_eff) and tau_eff > 0) else 0.0
            D_eff_val = D0_val + D_active
            part_results.append({
                "unique_particle_id": p_id,
                "D0": D0_val,
                "sigma_noise_sq": sigma_sq_val,
                "D_active": D_active,
                "D_eff": D_eff_val
            })
        except Exception:
            pass
            
    df_part_res = pd.DataFrame(part_results)
    
    # 2. プール粒子のアンサンブル平均MSDに対するフィッティング
    mask_ens = (stats_df.index >= min_t) & (stats_df.index <= max_t) & (stats_df['count'] >= min_particles)
    dt_ens = stats_df.index[mask_ens].to_numpy(dtype=float)
    msd_ens = stats_df['mean'][mask_ens].to_numpy(dtype=float)
    
    def fit_model_ens(dt, D0, sigma_sq):
        return fm.log_rtp_2state_msd(dt, D0, sigma_sq, f_run, v_R, tau_eff)
    target_ens = np.log10(np.maximum(msd_ens, 1e-12))
    
    try:
        popt_ens, pcov_ens = curve_fit(
            fit_model_ens, dt_ens, target_ens,
            p0=[1e-3, 1e-4],
            bounds=([0.0, 0.0], [100.0, 100.0]),
            maxfev=5000
        )
        D0_ens = float(popt_ens[0])
        sigma_sq_ens = float(popt_ens[1])
        perr_ens = np.sqrt(np.diag(pcov_ens)) if pcov_ens is not None else [np.nan, np.nan]
        D0_err = float(perr_ens[0]) if np.isfinite(perr_ens[0]) else np.nan
        sigma_sq_err = float(perr_ens[1]) if np.isfinite(perr_ens[1]) else np.nan
        
        pred_target = fit_model_ens(dt_ens, D0_ens, sigma_sq_ens)
        ss_res = np.sum((target_ens - pred_target)**2)
        ss_tot = np.sum((target_ens - np.mean(target_ens))**2)
        r2_ens = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else np.nan
    except Exception:
        D0_ens, D0_err = np.nan, np.nan
        sigma_sq_ens, sigma_sq_err = np.nan, np.nan
        r2_ens = np.nan
        
    D_active = 0.5 * f_run * (v_R ** 2) * tau_eff if (np.isfinite(tau_eff) and tau_eff > 0) else 0.0
    D_eff_ens = D0_ens + D_active if np.isfinite(D0_ens) else D_active
    
    fit_t_dense = np.logspace(np.log10(4.0), np.log10(1000.0), 300)
    fit_msd_dense = fm.rtp_2state_msd(fit_t_dense, D0_ens, sigma_sq_ens, f_run, v_R, tau_eff) if np.isfinite(D0_ens) else np.array([])
    
    summary = {
        "D0_ens": D0_ens,
        "D0_err": D0_err,
        "sigma_noise_sq_ens": sigma_sq_ens,
        "sigma_noise_sq_err": sigma_sq_err,
        "sigma_noise_ens": np.sqrt(max(0.0, sigma_sq_ens)) if np.isfinite(sigma_sq_ens) else np.nan,
        "D_active": D_active,
        "D_eff_ens": D_eff_ens,
        "r2_ens": r2_ens,
        "D0_mean_part": df_part_res["D0"].mean() if not df_part_res.empty else np.nan,
        "D0_std_part": df_part_res["D0"].std() if not df_part_res.empty else np.nan,
        "D_eff_mean_part": df_part_res["D_eff"].mean() if not df_part_res.empty else np.nan,
        "D_eff_std_part": df_part_res["D_eff"].std() if not df_part_res.empty else np.nan,
        "fit_t": fit_t_dense,
        "fit_msd": fit_msd_dense,
        "df_part": df_part_res
    }
    return summary


def calc_stokes_einstein_diffusion(diameter_um, temperature_K=298.15, viscosity_Pa_s=1.0e-3):
    """
    ストークス・アインシュタイン理論熱拡散係数 D_SE [um^2/s]
    """
    k_B = 1.380649e-23
    d_m = np.asarray(diameter_um, dtype=float) * 1e-6
    d_m2_s = (k_B * temperature_K) / (3.0 * np.pi * viscosity_Pa_s * d_m)
    return d_m2_s * 1e12


def save_figure_to_all(fig, basename: str, out_dirs: list[Path]):
    """Save matplotlib Figure as both .svg and .png to all valid output directories."""
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
                fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
                fig.savefig(d / f"{basename}.png", bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: list[Path]):
    """Save DataFrame as CSV to all valid output directories."""
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
                df.to_csv(d / filename, index=False)
        except Exception as e:
            print(f"Warning: Failed to save {filename} to {d}: {e}")


def clean_redundant_files(out_dirs: list[Path]):
    """Clean up old deprecated files so that only Ensemble Mean + IQR files remain."""
    redundant_patterns = [
        "MSD_median_IQR*",
        "MSD_mean_IQR*",
        "MSD_comparison_mean_vs_median*",
        "rtp_2state_msd_fit_summary_median.csv",
        "msd_pooled_median_iqr_summary.csv"
    ]
    for d in out_dirs:
        if d.exists():
            for pat in redundant_patterns:
                for f in d.glob(pat):
                    try:
                        f.unlink()
                    except Exception:
                        pass


def main():
    root_dir = find_default_root()
    workspace_dir = Path(__file__).parent.resolve()
    
    # 出力先ディレクトリ群
    out_dirs = [
        root_dir / "figure" / "msd",
        root_dir / "msd",
        root_dir / "figure",
        workspace_dir / "figure" / "msd",
        workspace_dir / "figure",
    ]
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    print(f"Data Root Directory: {root_dir}")
    print("Saving outputs to:")
    for d in out_dirs:
        if d.parent.exists():
            print(f"  {d}")

    clean_redundant_files(out_dirs)

    print("\n--- 1. Loading & Filtering MSD datasets (alpha > 0.5 only) ---")
    alpha_thresh = 0.5
    pool_06um, raw_06um, n_tot_06um, n_filt_06um = concat_pooled_particles_MSD(root_dir / "beads06um", [4, 4, 4, 4], alpha_threshold=alpha_thresh)
    pool_1um,  raw_1um,  n_tot_1um,  n_filt_1um  = concat_pooled_particles_MSD(root_dir / "beads1um",  [4, 4, 4, 4], alpha_threshold=alpha_thresh)
    pool_3um,  raw_3um,  n_tot_3um,  n_filt_3um  = concat_pooled_particles_MSD(root_dir / "beads3um",  [4, 4, 4, 4], alpha_threshold=alpha_thresh)
    pool_5um,  raw_5um,  n_tot_5um,  n_filt_5um  = concat_pooled_particles_MSD(root_dir / "beads5um",  [4, 4, 4, 4], alpha_threshold=alpha_thresh)
    pool_7um,  raw_7um,  n_tot_7um,  n_filt_7um  = concat_pooled_particles_MSD(root_dir / "beads7um",  [4, 4, 4],    alpha_threshold=alpha_thresh)
    pool_20um, raw_20um, n_tot_20um, n_filt_20um = concat_pooled_particles_MSD(root_dir / "beads20um", [4, 4, 4],    alpha_threshold=alpha_thresh)

    stats_06um = calc_pooled_MSD_stats(pool_06um)
    stats_1um = calc_pooled_MSD_stats(pool_1um)
    stats_3um = calc_pooled_MSD_stats(pool_3um)
    stats_5um = calc_pooled_MSD_stats(pool_5um)
    stats_7um = calc_pooled_MSD_stats(pool_7um)
    stats_20um = calc_pooled_MSD_stats(pool_20um)
    
    print(f"  beads06um: {n_tot_06um} -> {n_filt_06um} particles (removed {n_tot_06um - n_filt_06um})")
    print(f"  beads1um:  {n_tot_1um} -> {n_filt_1um} particles (removed {n_tot_1um - n_filt_1um})")
    print(f"  beads3um:  {n_tot_3um} -> {n_filt_3um} particles (removed {n_tot_3um - n_filt_3um})")
    print(f"  beads5um:  {n_tot_5um} -> {n_filt_5um} particles (removed {n_tot_5um - n_filt_5um})")
    print(f"  beads7um:  {n_tot_7um} -> {n_filt_7um} particles (removed {n_tot_7um - n_filt_7um})")
    print(f"  beads20um: {n_tot_20um} -> {n_filt_20um} particles (removed {n_tot_20um - n_filt_20um})")
    
    # 無次元化 MSD (alpha > 0.5 フィルタ済み)
    dim_msd06um = concat_dimensionless_MSD(root_dir / "beads06um", [4, 4, 4, 4], Rc=0.315, alpha_threshold=alpha_thresh)
    dim_msd1um  = concat_dimensionless_MSD(root_dir / "beads1um",  [4, 4, 4, 4], Rc=0.59,  alpha_threshold=alpha_thresh)
    dim_msd3um  = concat_dimensionless_MSD(root_dir / "beads3um",  [4, 4, 4, 4], Rc=1.685, alpha_threshold=alpha_thresh)
    dim_msd5um  = concat_dimensionless_MSD(root_dir / "beads5um",  [4, 4, 4, 4], Rc=2.5,   alpha_threshold=alpha_thresh)
    dim_msd7um  = concat_dimensionless_MSD(root_dir / "beads7um",  [4, 4, 4],    Rc=3.62,  alpha_threshold=alpha_thresh)
    dim_msd20um = concat_dimensionless_MSD(root_dir / "beads20um", [4, 4, 4],    Rc=10.0,  alpha_threshold=alpha_thresh)

    dim_elag_06um, dim_emsd_06um, dim_err_06um, _ = calc_dimensionless_MSD(dim_msd06um)
    dim_elag_1um,  dim_emsd_1um,  dim_err_1um, _  = calc_dimensionless_MSD(dim_msd1um)
    dim_elag_3um,  dim_emsd_3um,  dim_err_3um, _  = calc_dimensionless_MSD(dim_msd3um)
    dim_elag_5um,  dim_emsd_5um,  dim_err_5um, _  = calc_dimensionless_MSD(dim_msd5um)
    dim_elag_7um,  dim_emsd_7um,  dim_err_7um, _  = calc_dimensionless_MSD(dim_msd7um)
    dim_elag_20um, dim_emsd_20um, dim_err_20um, _ = calc_dimensionless_MSD(dim_msd20um)

    # ---------------------------------------------------------
    # 2. 2状態RTP大域フィッティングの実行 (Model A: 1/tau_eff = 1/tau_oacf + 1/tau_dwell)
    # ---------------------------------------------------------
    print("\n--- 2. Performing 2-state RTP Global Fitting (Filtered Ensemble Mean MSD) ---")
    rtp_params = load_rtp_parameters(root_dir=root_dir)
    
    beads_data = [
        {"name": "beads06um", "d_um": 0.63, "pool_df": pool_06um, "raw_df": raw_06um, "stats": stats_06um, "n_tot": n_tot_06um, "n_filt": n_filt_06um, "marker": "^", "color": style_colors[0]},
        {"name": "beads1um",  "d_um": 1.18, "pool_df": pool_1um,  "raw_df": raw_1um,  "stats": stats_1um,  "n_tot": n_tot_1um,  "n_filt": n_filt_1um,  "marker": "o", "color": style_colors[1]},
        {"name": "beads3um",  "d_um": 3.37, "pool_df": pool_3um,  "raw_df": raw_3um,  "stats": stats_3um,  "n_tot": n_tot_3um,  "n_filt": n_filt_3um,  "marker": "d", "color": style_colors[2]},
        {"name": "beads5um",  "d_um": 5.00, "pool_df": pool_5um,  "raw_df": raw_5um,  "stats": stats_5um,  "n_tot": n_tot_5um,  "n_filt": n_filt_5um,  "marker": "p", "color": style_colors[3]},
        {"name": "beads7um",  "d_um": 7.24, "pool_df": pool_7um,  "raw_df": raw_7um,  "stats": stats_7um,  "n_tot": n_tot_7um,  "n_filt": n_filt_7um,  "marker": "h", "color": style_colors[4]},
        {"name": "beads20um", "d_um": 20.0, "pool_df": pool_20um, "raw_df": raw_20um, "stats": stats_20um, "n_tot": n_tot_20um, "n_filt": n_filt_20um, "marker": "s", "color": style_colors[5]},
    ]
    
    rtp_fit_results = []
    print(f"\n[Filtered Ensemble Mean MSD RTP Fits (\u03b1 > 0.5)]")
    print(f"{'Name':<10} {'D_C [um]':<10} {'N_parts':<10} {'Model':<18} {'f_run':<8} {'v_R [um/s]':<12} {'tau_eff [s]':<14} {'D0 [um^2/s]':<14} {'D_active':<12} {'D_eff [um^2/s]':<14} {'R^2':<8}")
    print("-" * 140)
    
    for item in beads_data:
        d_um = item["d_um"]
        p = rtp_params[d_um]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]
        
        fit_res = fit_rtp_2state_msd(
            item["pool_df"],
            item["stats"],
            f_run=p["f_run"],
            v_R=p["v_R"],
            tau_eff=p["tau_eff"],
            min_t=4.0,
            max_t=300.0,
            min_particles=2
        )
        item["rtp_fit"] = fit_res
        
        rtp_fit_results.append({
            "bead_name": item["name"],
            "diameter_um": d_um,
            "n_filtered_particles": n_filt,
            "n_total_particles": n_tot,
            "alpha_threshold": alpha_thresh,
            "model_type": p["model_type"],
            "f_run": p["f_run"],
            "v_R_um_s": p["v_R"],
            "tau_Run_dwell_s": p["tau_dwell"],
            "tau_OACF_int_s": p["tau_oacf_int"],
            "tau_eff_s": p["tau_eff"],
            "D0_ens_um2_s": fit_res["D0_ens"],
            "D0_ens_err_um2_s": fit_res["D0_err"],
            "sigma_noise_sq_um2": fit_res["sigma_noise_sq_ens"],
            "sigma_noise_um": fit_res["sigma_noise_ens"],
            "D_active_um2_s": fit_res["D_active"],
            "D_eff_ens_um2_s": fit_res["D_eff_ens"],
            "r_squared_ens": fit_res["r2_ens"],
            "D0_mean_part_um2_s": fit_res["D0_mean_part"],
            "D0_std_part_um2_s": fit_res["D0_std_part"],
            "D_eff_mean_part_um2_s": fit_res["D_eff_mean_part"],
            "D_eff_std_part_um2_s": fit_res["D_eff_std_part"],
            "D_SE_um2_s": calc_stokes_einstein_diffusion(d_um)
        })
        
        tau_str = f"{p['tau_eff']:.2f}" if np.isfinite(p['tau_eff']) else "NaN (4D0t)"
        parts_str = f"{n_filt}/{n_tot}"
        print(f"{item['name']:<10} {d_um:<10.2f} {parts_str:<10} {p['model_type']:<18} {p['f_run']:<8.3f} {p['v_R']:<12.3f} {tau_str:<14} {fit_res['D0_ens']:<14.4e} {fit_res['D_active']:<12.4e} {fit_res['D_eff_ens']:<14.4e} {fit_res['r2_ens']:<8.4f}")
        
    df_rtp_summary = pd.DataFrame(rtp_fit_results)
    save_csv_to_all(df_rtp_summary, "rtp_2state_msd_fit_summary.csv", out_dirs)

    # 統計CSVの出力
    pooled_stats_rows = []
    for item in beads_data:
        st = item["stats"]
        b_name = item["name"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]
        for lag_val, row in st.iterrows():
            pooled_stats_rows.append({
                "bead_name": b_name,
                "diameter_um": d_um,
                "lag_time_s": lag_val,
                "mean_msd_um2": row['mean'],
                "q25_msd_um2": row['q25'],
                "q75_msd_um2": row['q75'],
                "q10_msd_um2": row['q10'],
                "q90_msd_um2": row['q90'],
                "median_msd_um2": row['median'],
                "std_msd_um2": row['std'],
                "n_active_particles": int(row['count']),
                "n_filtered_particles": n_filt,
                "n_total_particles": n_tot
            })
    df_pooled_summary = pd.DataFrame(pooled_stats_rows)
    save_csv_to_all(df_pooled_summary, "msd_pooled_mean_iqr_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 3. 従来のべき乗フィッティング
    # ---------------------------------------------------------
    min_t = 60
    max_t = 300

    popt_06um, err_popt_06um = fit_power_law(stats_06um, min_t, max_t)
    popt_1um,  err_popt_1um  = fit_power_law(stats_1um,  min_t, max_t)
    popt_3um,  err_popt_3um  = fit_power_law(stats_3um,  min_t, max_t)
    popt_5um,  err_popt_5um  = fit_power_law(stats_5um,  min_t, max_t)
    popt_7um,  err_popt_7um  = fit_power_law(stats_7um,  min_t, max_t)
    popt_20um, err_popt_20um = fit_power_law(stats_20um, min_t, max_t)

    dim_popt_06um, _ = fit_dimensionless(dim_msd06um, min_t, max_t)
    dim_popt_1um,  _ = fit_dimensionless(dim_msd1um,  min_t, max_t)
    dim_popt_3um,  _ = fit_dimensionless(dim_msd3um,  min_t, max_t)
    dim_popt_5um,  _ = fit_dimensionless(dim_msd5um,  min_t, max_t)
    dim_popt_7um,  _ = fit_dimensionless(dim_msd7um,  min_t, max_t)
    dim_popt_20um, _ = fit_dimensionless(dim_msd20um, min_t, max_t)

    dim_mean_popt_06um, dim_err_popt_06um = func(dim_popt_06um)
    dim_mean_popt_1um,  dim_err_popt_1um  = func(dim_popt_1um)
    dim_mean_popt_3um,  dim_err_popt_3um  = func(dim_popt_3um)
    dim_mean_popt_5um,  dim_err_popt_5um  = func(dim_popt_5um)
    dim_mean_popt_7um,  dim_err_popt_7um  = func(dim_popt_7um)
    dim_mean_popt_20um, dim_err_popt_20um = func(dim_popt_20um)
    
    alpha = 1
    marker_size = 8

    # =========================================================
    # 4. グラフ作成・出力 (代表値: Ensemble Mean, エラー帯: IQR, \alpha > 0.5 フィルタ適用)
    # =========================================================

    # ---------------------------------------------------------
    # 図1: MSD プロット (代表値: Ensemble Mean, エラー帯: 25-75% IQR & 10-90%)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]
        
        valid = st['count'] >= 2
        lags = st.index[valid]
        mean_v = st['mean'][valid]
        q25 = st['q25'][valid]
        q75 = st['q75'][valid]
        q10 = st['q10'][valid]
        q90 = st['q90'][valid]
        
        # 10% - 90% タイル帯 (淡色)
        ax.fill_between(lags, q10, q90, color=color, alpha=0.08, edgecolor='none')
        # 25% - 75% 四分位範囲 (IQR) 帯 (中間色)
        ax.fill_between(lags, q25, q75, color=color, alpha=0.22, edgecolor='none')
        # 主線: アンサンブル平均 Mean (実線 + マーカー)
        ax.plot(lags, mean_v, marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
                label=f'{d_um:.2f} \u03bcm ($N={n_filt}$)', color=color)

    ax.plot([80, 200], fm.power_law([80, 200], 1, popt_1um[1] * 1e-1) * 0.8, color='#333333', linestyle='--')
    ax.text(130, 3, f'$\\propto \\Delta t^{{1.0}}$', fontsize=12)
    ax.plot([80, 200], fm.power_law([80, 200], 2, popt_1um[1] * 1e-1) * 8, color='#333333', linestyle='--')
    ax.text(50, 400, f'$\\propto \\Delta t^{{2.0}}$', fontsize=12)

    ax.legend(fontsize=10.0, loc='upper left', framealpha=0.88, markerscale=0.8, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    ax.set(
        xlim=(4e-0, 1000),
        ylim=(1e-2, 1e4),
        xscale='log',
        yscale='log',
        xlabel='Lag time $\\Delta t$ [s]',
        ylabel='MSD $\\langle \\Delta \\boldsymbol{r}^2 \\rangle$ [$\\mu\\mathrm{m}^2$]',
        title='Ensemble Mean MSD \u00b1 IQR (Filtered: $\\alpha > 0.5$)'
    )
    save_figure_to_all(fig, "MSD", out_dirs)

    # ---------------------------------------------------------
    # 図1-B: 2状態RTPフィッティング重畳 MSD プロット (D_eff 注釈付き)
    # ---------------------------------------------------------
    fig_rtp, ax_rtp = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]
        fit_res = item["rtp_fit"]
        deff_val = fit_res["D_eff_ens"]
        
        valid = st['count'] >= 2
        lags = st.index[valid]
        mean_v = st['mean'][valid]
        q25 = st['q25'][valid]
        q75 = st['q75'][valid]
        
        ax_rtp.fill_between(lags, q25, q75, color=color, alpha=0.18, edgecolor='none')
        ax_rtp.plot(lags, mean_v, marker=marker, linestyle='none', markersize=7.5,
                    label=f'{d_um:.2f} \u03bcm ($N={n_filt}$, $D_{{\\mathrm{{eff}}}}={deff_val:.2f}$ \u03bcm$^2$/s)', color=color)
        if len(fit_res["fit_t"]) > 0:
            ax_rtp.plot(fit_res["fit_t"], fit_res["fit_msd"], color=color, linestyle='-', linewidth=2.0)

    ax_rtp.legend(fontsize=9.5, loc='upper left', framealpha=0.88, markerscale=0.8, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    ax_rtp.set(
        xlim=(4e-0, 1000),
        ylim=(1e-2, 1e4),
        xscale='log',
        yscale='log',
        xlabel='Lag time $\\Delta t$ [s]',
        ylabel='MSD $\\langle \\Delta \\boldsymbol{r}^2 \\rangle$ [$\\mu\\mathrm{m}^2$]',
        title='2-State RTP Fit on Filtered Ensemble Mean MSD ($\u03b1 > 0.5$)'
    )
    save_figure_to_all(fig_rtp, "MSD_RTP_fit", out_dirs)

    # ---------------------------------------------------------
    # 図1-C: 粒子種別ごとの 2状態RTPフィッティング個別パネル (2x3)
    # ---------------------------------------------------------
    fig_panels, axes_panels = plt.subplots(2, 3, figsize=(15.0, 9.5), sharex=True, sharey=True)
    axes_flat = axes_panels.flatten()
    
    for idx, item in enumerate(beads_data):
        ax_p = axes_flat[idx]
        pool_df = item["pool_df"]
        raw_df = item["raw_df"]
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]
        fit_res = item["rtp_fit"]
        p = rtp_params[d_um]
        
        # 1. フィルタ除外された粒子 (alpha <= 0.5) を薄いグレー点線で表示
        removed_ids = set(raw_df['unique_particle_id']) - set(pool_df['unique_particle_id'])
        for p_id in removed_ids:
            sub_rem = raw_df[raw_df['unique_particle_id'] == p_id]
            a_rem = sub_rem['alpha'].iloc[0] if 'alpha' in sub_rem.columns else np.nan
            ax_p.plot(sub_rem['lag time'], sub_rem['MSD'], color='#999999', alpha=0.35, linewidth=0.8, linestyle=':')
            
        # 2. 採用された全個別軌跡 (alpha > 0.5)
        for p_id in pool_df['unique_particle_id'].unique():
            sub_p = pool_df[pool_df['unique_particle_id'] == p_id]
            ax_p.plot(sub_p['lag time'], sub_p['MSD'], color=color, alpha=0.25, linewidth=0.9, linestyle='-')
            
        # 3. 10%-90%タイル & 25%-75% IQR 帯
        valid = st['count'] >= 2
        lags = st.index[valid]
        mean_v = st['mean'][valid]
        q25 = st['q25'][valid]
        q75 = st['q75'][valid]
        q10 = st['q10'][valid]
        q90 = st['q90'][valid]
        
        ax_p.fill_between(lags, q10, q90, color=color, alpha=0.10, edgecolor='none', label='10-90% Range')
        ax_p.fill_between(lags, q25, q75, color=color, alpha=0.25, edgecolor='none', label='IQR (25-75%)')
        
        # 4. 主線: アンサンブル平均 Mean（実線＋マーカー）
        ax_p.plot(lags, mean_v, marker=marker, linestyle='-', color=color, markersize=6.5, linewidth=2.0, label=f'Mean ($N={n_filt}/{n_tot}$)', zorder=5)
        
        # 5. 2状態RTP理論線
        if len(fit_res["fit_t"]) > 0:
            ax_p.plot(fit_res["fit_t"], fit_res["fit_msd"], color='#111111', linestyle='-', linewidth=2.2, label='2-State RTP Fit', zorder=6)
            
        # 6. 長時間漸近拡散線
        if np.isfinite(fit_res["D_eff_ens"]) and fit_res["D_eff_ens"] > 0:
            t_asymp = np.logspace(1.0, 3.0, 50)
            msd_asymp = 4.0 * fit_res["D_eff_ens"] * t_asymp
            ax_p.plot(t_asymp, msd_asymp, color='#666666', linestyle='--', linewidth=1.5, label=r'$4 D_{\mathrm{eff}} \Delta t$', zorder=5)
            
        deff_val = fit_res["D_eff_ens"]
        d0_val = fit_res["D0_ens"]
        dact_val = fit_res["D_active"]
        teff_val = p["tau_eff"]
        r2_val = fit_res["r2_ens"]
        r2_str = f"{r2_val:.3f}" if (np.isfinite(r2_val) and r2_val >= -10) else "N/A"
        
        param_text = (
            f"$D_C = {d_um:.2f}\\,\\mu\\mathrm{{m}}$ ($N={n_filt}/{n_tot}$)\n"
            f"$D_{{\\mathrm{{eff}}}} = {deff_val:.3e}\\,\\mu\\mathrm{{m}}^2/\\mathrm{{s}}$\n"
            f"$D_0 = {d0_val:.3e}\\,\\mu\\mathrm{{m}}^2/\\mathrm{{s}}$\n"
            f"$D_{{\\mathrm{{active}}}} = {dact_val:.3e}\\,\\mu\\mathrm{{m}}^2/\\mathrm{{s}}$\n"
            f"$\\tau_{{\\mathrm{{eff}}}} = {teff_val:.2f}\\,\\mathrm{{s}}$\n"
            f"$R^2 = {r2_str}$"
        )
        ax_p.text(0.04, 0.96, param_text, transform=ax_p.transAxes, verticalalignment='top', fontsize=9.0,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
        
        ax_p.set(
            xlim=(4e-0, 1000),
            ylim=(1e-2, 1e4),
            xscale='log',
            yscale='log',
            title=f'{item["name"]} ($D_C = {d_um:.2f}\\,\\mu\\mathrm{{m}}$, $N={n_filt}/{n_tot}$ parts)'
        )
        ax_p.legend(fontsize=8.0, loc='lower right', framealpha=0.85)
        
        if idx >= 3:
            ax_p.set_xlabel('Lag time $\\Delta t$ [s]', fontsize=14)
        if idx % 3 == 0:
            ax_p.set_ylabel('MSD $\\langle \\Delta \\boldsymbol{r}^2 \\rangle$ [$\\mu\\mathrm{m}^2$]', fontsize=14)
            
    fig_panels.tight_layout()
    save_figure_to_all(fig_panels, "MSD_RTP_fit_panels", out_dirs)

    # ---------------------------------------------------------
    # 図1-D: 粒子ごとの単独個別プロット保存 (代表値: Mean + IQR, フィルタ済み)
    # ---------------------------------------------------------
    for item in beads_data:
        d_um = item["d_um"]
        b_name = item["name"]
        color = item["color"]
        marker = item["marker"]
        fit_res = item["rtp_fit"]
        p = rtp_params[d_um]
        pool_df = item["pool_df"]
        raw_df = item["raw_df"]
        st = item["stats"]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]
        
        fig_single, ax_s = plt.subplots(figsize=(7.5, 5.8))
        
        # フィルタ除外粒子（薄いグレー点線）
        removed_ids = set(raw_df['unique_particle_id']) - set(pool_df['unique_particle_id'])
        for p_id in removed_ids:
            sub_rem = raw_df[raw_df['unique_particle_id'] == p_id]
            ax_s.plot(sub_rem['lag time'], sub_rem['MSD'], color='#999999', alpha=0.35, linewidth=0.9, linestyle=':')
            
        for p_id in pool_df['unique_particle_id'].unique():
            sub_p = pool_df[pool_df['unique_particle_id'] == p_id]
            ax_s.plot(sub_p['lag time'], sub_p['MSD'], color=color, alpha=0.25, linewidth=1.0, linestyle='-')
            
        valid = st['count'] >= 2
        lags = st.index[valid]
        mean_v = st['mean'][valid]
        q25 = st['q25'][valid]
        q75 = st['q75'][valid]
        q10 = st['q10'][valid]
        q90 = st['q90'][valid]
        
        ax_s.fill_between(lags, q10, q90, color=color, alpha=0.10, edgecolor='none', label='10-90% Range')
        ax_s.fill_between(lags, q25, q75, color=color, alpha=0.25, edgecolor='none', label='IQR (25-75%)')
        ax_s.plot(lags, mean_v, marker=marker, linestyle='-', color=color, markersize=8, linewidth=2.0, label=f'Mean ($N={n_filt}/{n_tot}$)', zorder=5)
        
        if len(fit_res["fit_t"]) > 0:
            ax_s.plot(fit_res["fit_t"], fit_res["fit_msd"], color='#111111', linestyle='-', linewidth=2.4, label='2-State RTP Fit', zorder=6)
            
        if np.isfinite(fit_res["D_eff_ens"]) and fit_res["D_eff_ens"] > 0:
            t_asymp = np.logspace(1.0, 3.0, 50)
            msd_asymp = 4.0 * fit_res["D_eff_ens"] * t_asymp
            ax_s.plot(t_asymp, msd_asymp, color='#666666', linestyle='--', linewidth=1.6, label=r'Asymptotic $4 D_{\mathrm{eff}} \Delta t$', zorder=5)

        deff_val = fit_res["D_eff_ens"]
        d0_val = fit_res["D0_ens"]
        dact_val = fit_res["D_active"]
        teff_val = p["tau_eff"]
        r2_val = fit_res["r2_ens"]
        r2_str = f"{r2_val:.3f}" if (np.isfinite(r2_val) and r2_val >= -10) else "N/A"
        
        param_text = (
            f"$D_C = {d_um:.2f}\\,\\mu\\mathrm{{m}}$ ($N={n_filt}/{n_tot}$ particles)\n"
            f"$D_{{\\mathrm{{eff}}}} = {deff_val:.3e}\\,\\mu\\mathrm{{m}}^2/\\mathrm{{s}}$\n"
            f"$D_0 = {d0_val:.3e}\\,\\mu\\mathrm{{m}}^2/\\mathrm{{s}}$\n"
            f"$D_{{\\mathrm{{active}}}} = {dact_val:.3e}\\,\\mu\\mathrm{{m}}^2/\\mathrm{{s}}$\n"
            f"$\\tau_{{\\mathrm{{eff}}}} = {teff_val:.2f}\\,\\mathrm{{s}}$\n"
            f"$R^2 = {r2_str}$"
        )
        ax_s.text(0.04, 0.96, param_text, transform=ax_s.transAxes, verticalalignment='top', fontsize=10.0,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
        
        ax_s.set(
            xlim=(4e-0, 1000),
            ylim=(1e-2, 1e4),
            xscale='log',
            yscale='log',
            xlabel='Lag time $\\Delta t$ [s]',
            ylabel='MSD $\\langle \\Delta \\boldsymbol{r}^2 \\rangle$ [$\\mu\\mathrm{m}^2$]',
            title=f'{b_name} ($D_C = {d_um:.2f}\\,\\mu\\mathrm{{m}}$) Filtered Mean \u00b1 IQR'
        )
        ax_s.legend(fontsize=8.5, loc='lower right', framealpha=0.88)
        
        save_figure_to_all(fig_single, f"MSD_RTP_fit_{b_name}", out_dirs)
        plt.close(fig_single)

    # ---------------------------------------------------------
    # 図: 有効拡散係数 D_eff vs 粒子径 D_C
    # ---------------------------------------------------------
    fig_deff, ax_deff = plt.subplots(figsize=(8.0, 5.8))
    d_vals = df_rtp_summary["diameter_um"].to_numpy()
    deff_ens = df_rtp_summary["D_eff_ens_um2_s"].to_numpy()
    deff_std_part = df_rtp_summary["D_eff_std_part_um2_s"].to_numpy()
    d_active = df_rtp_summary["D_active_um2_s"].to_numpy()
    d0_ens = df_rtp_summary["D0_ens_um2_s"].to_numpy()
    
    # Stokes-Einstein 理論線
    d_dense = np.logspace(np.log10(0.4), np.log10(28.0), 200)
    d_se_dense = calc_stokes_einstein_diffusion(d_dense)
    ax_deff.plot(d_dense, d_se_dense, color='#7f7f7f', linestyle=':', linewidth=1.8, label=r'Stokes-Einstein $D_0(D_C) = \frac{k_B T}{3\pi\eta D_C}$')
    
    # 能動輸送項 D_active (2-State RTP が適用された粒子のみ)
    valid_active = (d_active > 0) & np.isfinite(d_active)
    if np.any(valid_active):
        ax_deff.plot(d_vals[valid_active], d_active[valid_active], marker='^', linestyle='--', color='#2ca02c', markersize=7, label=r'Active Contribution $D_{\mathrm{active}} = \frac{1}{2} f_{\mathrm{run}} v_R^2 \tau_{\mathrm{eff}}$')
    
    # フィッティング熱拡散項 D_0
    ax_deff.plot(d_vals, d0_ens, marker='v', linestyle=':', color='#9467bd', markersize=7, label=r'Fitted Thermal $D_0$')
    
    # 2状態RTP 有効拡散係数 D_eff (アンサンブル大域フィッティング + 粒子間エラーバー)
    yerr_vals = np.where(np.isfinite(deff_std_part), deff_std_part, 0.0)
    ax_deff.errorbar(d_vals, deff_ens, yerr=yerr_vals, marker='o', linestyle='-', color='#d62728', linewidth=2.0, markersize=8, capsize=4, label=r'2-State RTP Fit $D_{\mathrm{eff}} = D_0 + D_{\mathrm{active}}$')

    ax_deff.set(
        xscale='log',
        yscale='log',
        xlim=(0.4, 28.0),
        ylim=(1e-3, 1e1),
        xlabel='Cargo Diameter $D_C$ [\u03bcm]',
        ylabel='Diffusion Coefficient [\u03bcm$^2$/s]',
        title='Effective Diffusion Coefficient $D_{\mathrm{eff}}$ vs Cargo Diameter'
    )
    ax_deff.legend(fontsize=10.0, loc='lower left', framealpha=0.85, markerscale=0.8, borderpad=0.3, labelspacing=0.25)
    save_figure_to_all(fig_deff, "rtp_effective_diffusion", out_dirs)

    # ---------------------------------------------------------
    # 図: 緩和時間 tau_OACF, tau_dwell, tau_eff vs 粒子径 D_C
    # (5, 7, 20 um の点は除外し、0.63, 1.18, 3.37 um のみプロット)
    # ---------------------------------------------------------
    fig_tau, ax_tau = plt.subplots(figsize=(8.0, 5.8))
    tau_dwell_vals = df_rtp_summary["tau_Run_dwell_s"].to_numpy()
    tau_oacf_vals = df_rtp_summary["tau_OACF_int_s"].to_numpy()
    tau_eff_vals = df_rtp_summary["tau_eff_s"].to_numpy()

    # 5, 7, 20 um を除外するマスク (Dc <= 3.5 um)
    mask_3beads = d_vals <= 3.5
    d_sub = d_vals[mask_3beads]
    tau_dwell_sub = tau_dwell_vals[mask_3beads]
    tau_oacf_sub = tau_oacf_vals[mask_3beads]
    tau_eff_sub = tau_eff_vals[mask_3beads]

    # 理論曲線: tau_OACF(Dc) = tau_0 * exp(-2 * Dc / (3 * R_0))
    # R0 = 2.7774 um (3R0 = 8.3321 um, plot_run_velocity.py フィッティングより固定)
    R0_vel = 2.7774
    d_dense_tau = np.linspace(0.0, 25.0, 300)
    
    # 3点の実測値から ln(y) 空間で tau_0 をフィッティング
    valid_oacf_sub = np.isfinite(tau_oacf_sub) & (tau_oacf_sub > 0)
    if np.any(valid_oacf_sub):
        ln_tau_0_vals = np.log(tau_oacf_sub[valid_oacf_sub]) + (2.0 / (3.0 * R0_vel)) * d_sub[valid_oacf_sub]
        ln_t0_fit = float(np.mean(ln_tau_0_vals))
        t0_fit = float(np.exp(ln_t0_fit))
    else:
        t0_fit = 14.00

    tau_theo_curve = t0_fit * np.exp(-2.0 * d_dense_tau / (3.0 * R0_vel))
    ax_tau.plot(
        d_dense_tau, tau_theo_curve,
        color='#1f78b4', linestyle='-', linewidth=2.2,
        label=rf'Theory: $\tau_{{\mathrm{{p}}}}(R_c) = \tau_0 \exp\left(-\frac{{4 R_c}}{{3 \xi}}\right)$' + '\n' + rf'  ($\tau_0 = {t0_fit:.2f}\,\mathrm{{s}},\ \xi = {R0_vel:.2f}\,\mu\mathrm{{m}}$)',
        zorder=3
    )

    # 1. tau_p (Orientation 持続時間 / 積分相関時間)
    if np.any(valid_oacf_sub):
        ax_tau.plot(
            d_sub[valid_oacf_sub], tau_oacf_sub[valid_oacf_sub],
            marker='o', color='#2b83ba', linewidth=1.8, linestyle=':', markersize=8.5,
            label=r'$\tau_{\mathrm{p}}$ (Measured Orientation persistence time)', zorder=4
        )

    # 2. tau_bound (Run 状態滞在時間 / 結合持続時間)
    ax_tau.plot(
        d_sub, tau_dwell_sub,
        marker='^', color='#4dac26', linewidth=1.8, linestyle='--', markersize=8.5,
        label=r'$\tau_{\mathrm{bound}}$ (Bound/Run dwell time)', zorder=5
    )

    # 3. tau_eff (有効緩和時間 1/tau_eff = 1/tau_p + 1/tau_bound)
    ax_tau.plot(
        d_sub, tau_eff_sub,
        marker='s', color='#d7191c', linewidth=2.5, linestyle='-', markersize=9.0,
        label=r'$\tau_{\mathrm{eff}} = \left(\tau_{\mathrm{p}}^{-1} + \tau_{\mathrm{bound}}^{-1}\right)^{-1}$', zorder=6
    )

    # tau_eff の数値注釈
    for i_d, (d_val, t_eff) in enumerate(zip(d_sub, tau_eff_sub)):
        ax_tau.annotate(
            f"{t_eff:.2f}s",
            (d_val, t_eff),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9.0,
            fontweight='bold',
            color='#d7191c',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#d7191c', alpha=0.9)
        )

    ax_tau.set_yscale('log')
    ax_tau.set_xlim(0, 25.0)
    ax_tau.set_ylim(0.02, 250.0)
    ax_tau.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax_tau.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax_tau.yaxis.set_major_locator(ticker.FixedLocator([0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]))
    ax_tau.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    ax_tau.set_xlabel(r'Particle Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_tau.set_ylabel(r'Timescale $\tau$ [s]', fontsize=12, fontweight='bold')
    ax_tau.set_title(r'Relaxation Times $\tau_{\mathrm{p}}, \tau_{\mathrm{bound}}, \tau_{\mathrm{eff}}$ vs Particle Diameter' + '\n' + rf'($\tau_0 = {t0_fit:.2f}\,\mathrm{{s}},\ \xi = {R0_vel:.2f}\,\mu\mathrm{{m}},\ x \in [0, 25]\,\mu\mathrm{{m}}$)', fontsize=12, fontweight='bold', pad=10)
    ax_tau.grid(True, which='both', linestyle='--', alpha=0.4)
    ax_tau.legend(frameon=True, fontsize=9.0, loc='upper right', framealpha=0.92)

    save_figure_to_all(fig_tau, "relaxation_times_vs_diameter", out_dirs)
    save_figure_to_all(fig_tau, "tau_eff_vs_diameter", out_dirs)

    # ---------------------------------------------------------
    # 図2: alpha vs Cargo Diameter
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.errorbar([0.63, 1.18, 3.37, 5.00, 7.24, 20.0], 
                 [popt_06um[0], popt_1um[0], popt_3um[0], popt_5um[0], popt_7um[0], popt_20um[0]], 
                 yerr=[err_popt_06um[0], err_popt_1um[0], err_popt_3um[0], err_popt_5um[0], err_popt_7um[0], err_popt_20um[0]], 
                 marker='o')
    ax2.set(xlabel=r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', ylabel=r'$\alpha$')
    save_figure_to_all(fig2, "alpha", out_dirs)
    
    # ---------------------------------------------------------
    # 図3: 無次元化 MSD
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots()
    ax3.plot(dim_elag_06um, dim_emsd_06um, marker='^', label=f'0.63 \u03bcm', alpha=alpha, markersize=marker_size, color=style_colors[0])
    ax3.fill_between(dim_elag_06um, dim_emsd_06um - dim_err_06um, dim_emsd_06um + dim_err_06um, edgecolor=style_colors[0], facecolor=mcolors.to_rgba(style_colors[0], alpha=0.2))
    ax3.plot(dim_elag_1um, dim_emsd_1um, marker='o', label=f'1.18 \u03bcm', alpha=alpha, markersize=marker_size, color=style_colors[1])
    ax3.fill_between(dim_elag_1um, dim_emsd_1um - dim_err_1um, dim_emsd_1um + dim_err_1um, edgecolor=style_colors[1], facecolor=mcolors.to_rgba(style_colors[1], alpha=0.2))
    ax3.plot(dim_elag_3um, dim_emsd_3um, marker='d', label=f'3.37 \u03bcm', alpha=alpha, markersize=marker_size, color=style_colors[2])
    ax3.fill_between(dim_elag_3um, dim_emsd_3um - dim_err_3um, dim_emsd_3um + dim_err_3um, edgecolor=style_colors[2], facecolor=mcolors.to_rgba(style_colors[2], alpha=0.2))
    ax3.plot(dim_elag_5um, dim_emsd_5um, marker=10, label=f'5.00 \u03bcm', alpha=alpha, markersize=marker_size, color=style_colors[3])
    ax3.fill_between(dim_elag_5um, dim_emsd_5um - dim_err_5um, dim_emsd_5um + dim_err_5um, edgecolor=style_colors[3], facecolor=mcolors.to_rgba(style_colors[3], alpha=0.2))
    ax3.plot(dim_elag_7um, dim_emsd_7um, marker=11, label=f'7.24 \u03bcm', alpha=alpha, markersize=marker_size, color=style_colors[4])  
    ax3.fill_between(dim_elag_7um, dim_emsd_7um - dim_err_7um, dim_emsd_7um + dim_err_7um, edgecolor=style_colors[4], facecolor=mcolors.to_rgba(style_colors[4], alpha=0.2))
    ax3.plot(dim_elag_20um, dim_emsd_20um, marker='s', label=f'20.0 \u03bcm', alpha=alpha, markersize=marker_size, color=style_colors[5])
    ax3.fill_between(dim_elag_20um, dim_emsd_20um - dim_err_20um, dim_emsd_20um + dim_err_20um, edgecolor=style_colors[5], facecolor=mcolors.to_rgba(style_colors[5], alpha=0.2))
    
    t_start, t_end = 100, 400
    max_A2 = 0
    min_A1 = np.inf
    
    for dim_elag, dim_emsd in [(dim_elag_06um, dim_emsd_06um), (dim_elag_1um, dim_emsd_1um), 
                               (dim_elag_3um, dim_emsd_3um), (dim_elag_5um, dim_emsd_5um), 
                               (dim_elag_7um, dim_emsd_7um), (dim_elag_20um, dim_emsd_20um)]:
        valid_idx = (dim_elag > t_start) & (dim_elag < t_end)
        if valid_idx.any():
            t_ref = dim_elag[valid_idx].values
            msd_ref = dim_emsd[valid_idx].values
            min_A1 = min(min_A1, np.min(msd_ref / t_ref))
            max_A2 = max(max_A2, np.max(msd_ref / (t_ref**2)))
            
    if max_A2 > 0 and min_A1 < np.inf:
        A1 = min_A1 * 0.3
        ax3.plot([t_start, t_end], [A1 * t_start, A1 * t_end], color='#333333')
        ax3.text(t_start * 1.5, A1 * (t_start * 1.5) * 0.5, r'$\propto \Delta\tilde{t}^{1.0}$')
        
        A2 = max_A2 * 3.0
        ax3.plot([t_start, t_end], [A2 * (t_start**2), A2 * (t_end**2)], color='#333333')
        ax3.text(t_start * 1.2, A2 * ((t_start * 1.2)**2) * 1.5, r'$\propto \Delta\tilde{t}^{2.0}$')
        
    ax3.legend()
    ax3.set(
        xscale='log',
        yscale='log',
        xlabel='Dimensionless lag time $\\Delta\\tilde{t}$',
        ylabel='Dimensionless MSD $\\langle\\Delta\\tilde{\\boldsymbol{r}}^2\\rangle$'
    )
    save_figure_to_all(fig3, "dimensionless_MSD", out_dirs)
    
    # ---------------------------------------------------------
    # 図4: 無次元化 alpha
    # ---------------------------------------------------------
    fig4, ax4 = plt.subplots()
    ax4.errorbar([0.63, 1.18, 3.37, 5.00, 7.24, 20.0], 
                 [dim_mean_popt_06um[0], dim_mean_popt_1um[0], dim_mean_popt_3um[0], dim_mean_popt_5um[0], dim_mean_popt_7um[0], dim_mean_popt_20um[0]], 
                 yerr=[dim_err_popt_06um[0], dim_err_popt_1um[0], dim_err_popt_3um[0], dim_err_popt_5um[0], dim_err_popt_7um[0], dim_err_popt_20um[0]], 
                 marker='o')
    ax4.set(xlabel='Cargo Diameter $D_C$ [\u03bcm]', ylabel='Dimensionless $\\alpha$')
    save_figure_to_all(fig4, "dimensionless_alpha", out_dirs)
    
    # ---------------------------------------------------------
    # 図5: 無次元局所 alpha
    # ---------------------------------------------------------
    fig5, ax5 = plt.subplots()
    ax5.plot(*calc_local_alpha(dim_elag_06um, dim_emsd_06um), label=f'0.63 \u03bcm', color=style_colors[0], alpha=0.8, marker='^', markersize=marker_size)
    ax5.plot(*calc_local_alpha(dim_elag_1um, dim_emsd_1um), label=f'1.18 \u03bcm', color=style_colors[1], alpha=0.8, marker='o', markersize=marker_size)
    ax5.plot(*calc_local_alpha(dim_elag_3um, dim_emsd_3um), label=f'3.37 \u03bcm', color=style_colors[2], alpha=0.8, marker='d', markersize=marker_size)
    ax5.plot(*calc_local_alpha(dim_elag_5um, dim_emsd_5um), label=f'5.00 \u03bcm', color=style_colors[3], alpha=0.8, marker=10, markersize=marker_size)
    ax5.plot(*calc_local_alpha(dim_elag_7um, dim_emsd_7um), label=f'7.24 \u03bcm', color=style_colors[4], alpha=0.8, marker=11, markersize=marker_size)
    ax5.plot(*calc_local_alpha(dim_elag_20um, dim_emsd_20um), label=f'20.0 \u03bcm', color=style_colors[5], alpha=0.8, marker='s', markersize=marker_size)
    
    ax5.legend()
    ax5.set(
        ylim=(0,2),
        xscale='log',
        xlabel='Dimensionless lag time $\\Delta\\tilde{t}$',
        ylabel='Local exponent $\\alpha(\\tilde{t}) = d\\log(\\widetilde{MSD}) / d\\log(\\Delta\\tilde{t})$'
    )
    save_figure_to_all(fig5, "dimensionless_local_alpha", out_dirs)

    # ---------------------------------------------------------
    # 図6: 局所 alpha (Ensemble Mean ベース)
    # ---------------------------------------------------------
    fig6, ax6 = plt.subplots()
    ax6.plot(*calc_local_alpha(stats_06um.index, stats_06um['mean']), label=f'0.63 \u03bcm', color=style_colors[0], alpha=0.8, marker='^', markersize=marker_size)
    ax6.plot(*calc_local_alpha(stats_1um.index, stats_1um['mean']), label=f'1.18 \u03bcm', color=style_colors[1], alpha=0.8, marker='o', markersize=marker_size)
    ax6.plot(*calc_local_alpha(stats_3um.index, stats_3um['mean']), label=f'3.37 \u03bcm', color=style_colors[2], alpha=0.8, marker='d', markersize=marker_size)
    ax6.plot(*calc_local_alpha(stats_5um.index, stats_5um['mean']), label=f'5.00 \u03bcm', color=style_colors[3], alpha=0.8, marker=10, markersize=marker_size)
    ax6.plot(*calc_local_alpha(stats_7um.index, stats_7um['mean']), label=f'7.24 \u03bcm', color=style_colors[4], alpha=0.8, marker=11, markersize=marker_size)
    ax6.plot(*calc_local_alpha(stats_20um.index, stats_20um['mean']), label=f'20.0 \u03bcm', color=style_colors[5], alpha=0.8, marker='s', markersize=marker_size)
    
    ax6.legend()
    ax6.set(
        xlim=(1, 400),
        ylim=(0, 2),
        xscale='log',
        xlabel='Lag time $\\Delta t$ [s]',
        ylabel='Local exponent $\\alpha(t) = d\\log(MSD) / d\\log(\\Delta t)$'
    )
    save_figure_to_all(fig6, "local_alpha", out_dirs)

    print("\n--- Completed all fits and figure outputs successfully! ---")


if __name__ == "__main__":
    main()