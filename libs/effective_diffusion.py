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
        
        exp_records.append({
            'exp': exp_id,
            'd_eff_full': d_eff_trapz_full,
            'd_eff_pos': d_eff_trapz_pos,
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
    
    n_exps = len(df_exp_results)
    mean_exp_d = df_exp_results['d_eff_full'].mean()
    std_exp_d = df_exp_results['d_eff_full'].std()
    sem_exp_d = std_exp_d / np.sqrt(n_exps) if n_exps > 1 else 0.0

    mean_pos_d = df_exp_results['d_eff_pos'].mean()
    std_pos_d = df_exp_results['d_eff_pos'].std()
    sem_pos_d = std_pos_d / np.sqrt(n_exps) if n_exps > 1 else 0.0

    ensemble_summary = {
        'n_exps': n_exps,
        'd_eff_mean': float(mean_exp_d),
        'd_eff_std': float(std_exp_d),
        'd_eff_sem': float(sem_exp_d),
        'd_eff_pos_mean': float(mean_pos_d),
        'd_eff_pos_std': float(std_pos_d),
        'd_eff_pos_sem': float(sem_pos_d),
        'd_eff_ens_full': float(d_ens_full),
        'd_eff_ens_pos': float(d_ens_pos),
        'c_v_0': float(c_ens[0]) if len(c_ens) > 0 else 0.0,
        't_lag': t_ens,
        'c_v_mean': c_ens,
        'c_v_std': ens_std.to_numpy(dtype=float),
        'cum_d_ens': cum_d_ens
    }
    
    return df_exp_results, ensemble_summary


def compute_rtp_theoretical_diffusion(
    hmm_summary_path: Path,
    angle_summary_path: Optional[Path] = None,
    abp_summary_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    HMM パラメータ CSV、配向変化角 CSV、および Run MSD フィッティング (ABP) CSV から
    Run-and-Tumble Particle (RTP) モデルの理論的有効拡散係数を算出する。
    
    tau_Run の値として、Run MSD フィッティングより求めた tau_Run,MSD と
    Run Dwell time より求めた tau_Run,dwell の小さい方 min(tau_Run,MSD, tau_Run,dwell) を使用します。

    Parameters
    ----------
    hmm_summary_path : Path
        `hmm_state_parameters_summary_k2.csv` のパス
    angle_summary_path : Path, optional
        `hmm_turning_angle_summary_k2.csv` のパス
    abp_summary_path : Path, optional
        `hmm_run_abp_fits_summary_k2.csv` のパス

    Returns
    -------
    pd.DataFrame
        粒子径ごとの理論有効拡散係数および微視的パラメータ
    """
    df_hmm = pd.read_csv(hmm_summary_path)
    df_angle = pd.read_csv(angle_summary_path) if (angle_summary_path and Path(angle_summary_path).exists()) else pd.DataFrame()
    df_abp = pd.read_csv(abp_summary_path) if (abp_summary_path and Path(abp_summary_path).exists()) else pd.DataFrame()
    
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
        
        # Run MSD フィッティングから得られた持続時間 tau_Run,MSD
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

        # min(tau_Run,MSD, tau_Run,dwell) の計算
        # 1) 理論マルコフ持続時間との最小値
        if np.isfinite(tau_run_msd) and tau_run_msd > 0:
            tau_run_eff_theo = min(tau_run_msd, tau_run_theo)
            tau_run_eff_ccdf = min(tau_run_msd, tau_run_ccdf)
            tau_run_eff_fit = min(tau_run_msd, tau_run_fit)
            tau_run_eff_emp = min(tau_run_msd, tau_run_emp)
        else:
            tau_run_eff_theo = tau_run_theo
            tau_run_eff_ccdf = tau_run_ccdf
            tau_run_eff_fit = tau_run_fit
            tau_run_eff_emp = tau_run_emp

        # 方向相関 <cos delta theta>
        mean_cos_run = 0.0
        if not df_angle.empty:
            sub_angle = df_angle[(df_angle['diameter_um'] == d_um) & (df_angle['state'] == 1)]
            if not sub_angle.empty:
                mean_cos_run = float(sub_angle.iloc[0]['mean_cos'])

        # 1. min(tau_MSD, tau_dwell) を用いた標準 2状態 RTP モデル
        # D_eff = 0.5 * pi_run * v_run^2 * min(tau_Run,MSD, tau_Run,dwell)
        d_rtp_min_theo = 0.5 * pi_run * (v_run ** 2) * tau_run_eff_theo
        d_rtp_min_ccdf = 0.5 * (tau_run_eff_ccdf / (tau_run_eff_ccdf + tau_tumble_ccdf)) * (v_run ** 2) * tau_run_eff_ccdf
        d_rtp_min_fit = 0.5 * (tau_run_eff_fit / (tau_run_eff_fit + tau_tumble_fit)) * (v_run ** 2) * tau_run_eff_fit
        d_rtp_min_emp = 0.5 * f_run_emp * (v_run ** 2) * tau_run_eff_emp

        # 2. 持続性考慮 (Persistence) RTP モデル (min(tau_MSD, tau_dwell) 使用)
        cos_clamped = np.clip(mean_cos_run, -0.85, 0.85)
        d_rtp_pers_min_theo = 0.5 * pi_run * (v_run ** 2) * (tau_run_eff_theo / (1.0 - cos_clamped))
        d_rtp_pers_min_ccdf = 0.5 * (tau_run_eff_ccdf / (tau_run_eff_ccdf + tau_tumble_ccdf)) * (v_run ** 2) * (tau_run_eff_ccdf / (1.0 - cos_clamped))

        # 3. 従来の純粋 Dwell time のみの RTP 理論値 (比較用)
        d_rtp_dwell_theo = 0.5 * pi_run * (v_run ** 2) * tau_run_theo
        d_rtp_dwell_ccdf = 0.5 * (tau_run_ccdf / (tau_run_ccdf + tau_tumble_ccdf)) * (v_run ** 2) * tau_run_ccdf

        # 4. 2状態一般化（Tumble時の速度成分も考慮）
        d_rtp_2state_min = 0.5 * (pi_run * (v_run ** 2) * tau_run_eff_theo + pi_tumble * (v_tumble ** 2) * tau_tumble_theo)

        records.append({
            'bead_name': b_name,
            'diameter_um': d_um,
            'v_run_um_s': v_run,
            'v_run_geom_um_s': v_run_geom,
            'v_tumble_um_s': v_tumble,
            'tau_run_msd_s': tau_run_msd,
            'tau_run_theo_s': tau_run_theo,
            'tau_tumble_theo_s': tau_tumble_theo,
            'tau_run_ccdf_s': tau_run_ccdf,
            'tau_tumble_ccdf_s': tau_tumble_ccdf,
            'tau_run_fit_s': tau_run_fit,
            'tau_tumble_fit_s': tau_tumble_fit,
            'tau_run_emp_s': tau_run_emp,
            'tau_run_eff_theo_s': tau_run_eff_theo,
            'tau_run_eff_ccdf_s': tau_run_eff_ccdf,
            'pi_run': pi_run,
            'pi_tumble': pi_tumble,
            'mean_cos_run': mean_cos_run,
            'D_RTP_theo': d_rtp_min_theo,
            'D_RTP_ccdf': d_rtp_min_ccdf,
            'D_RTP_fit': d_rtp_min_fit,
            'D_RTP_emp': d_rtp_min_emp,
            'D_RTP_pers_theo': d_rtp_pers_min_theo,
            'D_RTP_pers_ccdf': d_rtp_pers_min_ccdf,
            'D_RTP_dwell_theo': d_rtp_dwell_theo,
            'D_RTP_dwell_ccdf': d_rtp_dwell_ccdf,
            'D_RTP_2state_theo': d_rtp_2state_min
        })

    return pd.DataFrame(records)


def plot_effective_diffusion_comparison(
    df_combined: pd.DataFrame,
    out_dir: Path,
    use_log_scale: bool = True
):
    """
    横軸を粒子径、縦軸を有効拡散係数として
    Green-Kubo 実測値と RTP 理論予測 (tau_R = min(tau_MSD, tau_dwell)) を比較するメイングラフを作成・保存する。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    
    # 粒子径
    d = df_combined['diameter_um'].to_numpy()
    
    # 1. Green-Kubo 実測値
    d_gk = df_combined['D_eff_GK_mean'].to_numpy()
    sem_gk = df_combined['D_eff_GK_sem'].to_numpy()
    
    ax.errorbar(
        d, d_gk, yerr=sem_gk,
        fmt='o',
        color='#1b9e77',
        ecolor='#1b9e77',
        elinewidth=2.0,
        capsize=5.0,
        capthick=1.5,
        markersize=8.5,
        label=r'Green-Kubo Integral ($D_{\mathrm{eff}} = \frac{1}{2}\int_0^\infty C_v(t) dt$)',
        zorder=6
    )
    
    # 2. HMM RTP 理論予測 (tau_R = min(tau_MSD, tau_dwell))
    d_rtp_theo = df_combined['D_RTP_theo'].to_numpy()
    ax.plot(
        d, d_rtp_theo,
        marker='s',
        color='#d95f02',
        linewidth=2.2,
        linestyle='-',
        markersize=7.5,
        label=r'RTP Model: $\tau_R = \min(\tau_{\mathrm{MSD}}, \tau_{\mathrm{dwell}})$',
        zorder=5
    )

    # 3. HMM RTP 持続性考慮理論予測
    if 'D_RTP_pers_theo' in df_combined.columns:
        d_rtp_pers = df_combined['D_RTP_pers_theo'].to_numpy()
        ax.plot(
            d, d_rtp_pers,
            marker='^',
            color='#7570b3',
            linewidth=1.8,
            linestyle='--',
            markersize=7.0,
            label=r'RTP with Persistence ($\frac{\tau_R}{1 - \langle\cos\Delta\theta\rangle}$)',
            zorder=4
        )
        
    # 4. 従来の純粋 Dwell time のみの RTP 理論予測（参考比較）
    if 'D_RTP_dwell_theo' in df_combined.columns:
        d_rtp_dwell = df_combined['D_RTP_dwell_theo'].to_numpy()
        ax.plot(
            d, d_rtp_dwell,
            marker='x',
            color='#999999',
            linewidth=1.2,
            linestyle=':',
            markersize=6.0,
            alpha=0.75,
            label=r'RTP (Dwell-only $\tau_R = \tau_{\mathrm{dwell}}$)',
            zorder=2
        )

    ax.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Effective Diffusion Coefficient $D_{\mathrm{eff}}$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=12, fontweight='bold')
    ax.set_title('Effective Diffusion: Green-Kubo vs Microscopic RTP Model', fontsize=12, fontweight='bold', pad=10)
    
    if use_log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
        ax.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
        ax.set_xlim(0.45, 28.0)
    else:
        ax.set_xlim(0, 22)
        
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=8.8, loc='upper right', framealpha=0.92)
    
    plt.tight_layout()
    svg_path = out_dir / 'D_eff_vs_diameter.svg'
    png_path = out_dir / 'D_eff_vs_diameter.png'
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] {svg_path}")
    print(f"  [保存完了] {png_path}")

    # 線形スケール版も作成
    fig_lin, ax_lin = plt.subplots(figsize=(7.2, 5.4))
    ax_lin.errorbar(
        d, d_gk, yerr=sem_gk,
        fmt='o', color='#1b9e77', ecolor='#1b9e77', elinewidth=2.0,
        capsize=5.0, capthick=1.5, markersize=8.5,
        label=r'Green-Kubo Integral ($D_{\mathrm{eff}} = \frac{1}{2}\int_0^\infty C_v(t) dt$)', zorder=6
    )
    ax_lin.plot(
        d, d_rtp_theo, marker='s', color='#d95f02', linewidth=2.2, linestyle='-',
        markersize=7.5, label=r'RTP Model: $\tau_R = \min(\tau_{\mathrm{MSD}}, \tau_{\mathrm{dwell}})$', zorder=5
    )
    if 'D_RTP_pers_theo' in df_combined.columns:
        ax_lin.plot(
            d, df_combined['D_RTP_pers_theo'], marker='^', color='#7570b3', linewidth=1.8,
            linestyle='--', markersize=7.0, label=r'RTP with Persistence', zorder=4
        )
    if 'D_RTP_dwell_theo' in df_combined.columns:
        ax_lin.plot(
            d, df_combined['D_RTP_dwell_theo'], marker='x', color='#999999', linewidth=1.2,
            linestyle=':', markersize=6.0, alpha=0.75, label=r'RTP (Dwell-only)', zorder=2
        )
    ax_lin.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_lin.set_ylabel(r'Effective Diffusion Coefficient $D_{\mathrm{eff}}$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=12, fontweight='bold')
    ax_lin.set_title('Effective Diffusion vs Diameter (Linear Scale)', fontsize=12, fontweight='bold', pad=10)
    ax_lin.set_xlim(0, 22)
    ax_lin.grid(True, linestyle='--', alpha=0.4)
    ax_lin.legend(frameon=True, fontsize=8.8, loc='upper right', framealpha=0.92)
    
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
    d_gk = df_combined['D_eff_GK_mean'].to_numpy()
    sem_gk = df_combined['D_eff_GK_sem'].to_numpy()
    d_rtp_theo = df_combined['D_RTP_theo'].to_numpy()
    
    ax_c.errorbar(
        d, d_gk, yerr=sem_gk,
        fmt='o', color='#1b9e77', ecolor='#1b9e77', elinewidth=2.0,
        capsize=4.5, capthick=1.5, markersize=8,
        label=r'Green-Kubo Integral', zorder=5
    )
    ax_c.plot(
        d, d_rtp_theo, marker='s', color='#d95f02', linewidth=2.0, linestyle='-',
        markersize=7, label=r'RTP: $\tau_R = \min(\tau_{\mathrm{MSD}}, \tau_{\mathrm{dwell}})$', zorder=4
    )
    if 'D_RTP_pers_theo' in df_combined.columns:
        ax_c.plot(
            d, df_combined['D_RTP_pers_theo'], marker='^', color='#7570b3', linewidth=1.8,
            linestyle='--', markersize=7, label=r'RTP with Persistence', zorder=3
        )
    if 'D_RTP_dwell_theo' in df_combined.columns:
        ax_c.plot(
            d, df_combined['D_RTP_dwell_theo'], marker='x', color='#999999', linewidth=1.2,
            linestyle=':', markersize=5.5, alpha=0.75, label=r'RTP (Dwell-only)', zorder=2
        )
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_c.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax_c.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
    ax_c.set_xlim(0.45, 28.0)
    ax_c.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax_c.set_ylabel(r'Effective Diffusion $D_{\mathrm{eff}}$ [$\mu\mathrm{m}^2/\mathrm{s}$]', fontsize=11, fontweight='bold')
    ax_c.set_title(r'(c) Effective Diffusion: Green-Kubo vs RTP Model', fontsize=12, fontweight='bold')
    ax_c.grid(True, which='both', linestyle='--', alpha=0.4)
    ax_c.legend(frameon=True, fontsize=8.5, loc='upper right')

    # -------------------------------------------------------------
    # Panel (d): 微視的持続時間 tau_MSD, tau_dwell, tau_eff vs 粒子径
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d2 = ax_d.twinx()
    
    # 左軸: tau_MSD, tau_dwell, tau_eff
    if 'tau_run_msd_s' in df_combined.columns:
        ax_d.plot(d, df_combined['tau_run_msd_s'], marker='o', color='#2b83ba', lw=1.6, linestyle=':', label=r'$\tau_{\mathrm{Run, MSD}}$ (Active Brownian)')
    ax_d.plot(d, df_combined['tau_run_theo_s'], marker='^', color='#abdda4', lw=1.6, linestyle='--', label=r'$\tau_{\mathrm{Run, dwell}}$ (Markov)')
    ax_d.plot(d, df_combined['tau_run_eff_theo_s'], marker='s', color='#d7191c', lw=2.2, linestyle='-', label=r'$\tau_{\mathrm{Run}}^{\mathrm{eff}} = \min(\tau_{\mathrm{MSD}}, \tau_{\mathrm{dwell}})$')
    
    # 右軸: Run 速度 v_R
    l_v = ax_d2.plot(d, df_combined['v_run_um_s'], marker='d', color='#fdae61', lw=1.8, linestyle='-.', label=r'Run Speed $v_R$ [$\mu\mathrm{m}/\mathrm{s}$]')
    
    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    ax_d.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_d.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax_d.set_xticklabels(['0.63', '1.18', '3.37', '5.0', '7.24', '20'])
    ax_d.set_xlim(0.45, 28.0)
    
    ax_d.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=11, fontweight='bold')
    ax_d.set_ylabel(r'Run Timescales $\tau_R$ [s]', fontsize=11, fontweight='bold')
    ax_d2.set_ylabel(r'Run Speed $v_R$ [$\mu\mathrm{m}/\mathrm{s}$]', fontsize=11, fontweight='bold', color='#e66101')
    ax_d2.tick_params(axis='y', labelcolor='#e66101')
    
    lines_1, labels_1 = ax_d.get_legend_handles_labels()
    lines_2, labels_2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=True, fontsize=8.0, loc='lower left')
    ax_d.set_title(r'(d) Microscopic Run Parameters: $\tau_R$ & $v_R$ vs Diameter', fontsize=12, fontweight='bold')
    ax_d.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    svg_4p = out_dir / 'effective_diffusion_detailed_4panel.svg'
    png_4p = out_dir / 'effective_diffusion_detailed_4panel.png'
    fig.savefig(svg_4p, bbox_inches='tight')
    fig.savefig(png_4p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 4パネル詳細図: {svg_4p}")


def run_effective_diffusion_analysis(
    root_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    hmm_summary_path: Optional[Union[str, Path]] = None,
    angle_summary_path: Optional[Union[str, Path]] = None,
    abp_summary_path: Optional[Union[str, Path]] = None,
    max_timeshift_frames: int = 50,
    frame_interval: float = 4.0,
    scale: float = 0.11,
    max_lag_time: float = 200.0
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
        # ローカルまたは root_dir から探索
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

    print(f"\n{'='*75}")
    print("有効拡散係数（Green-Kubo 積分 vs HMM RTP 理論予測 [min(tau_MSD, tau_dwell)]）一括解析")
    print(f"ルートディレクトリ: {root_dir}")
    print(f"出力先ディレクトリ: {out_dir}")
    print(f"HMM サマリーパス : {hmm_summary_path}")
    print(f"ABP MSD パス    : {abp_summary_path}")
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
        vacf_ensemble_dict[b_name] = ens_summary

        gk_summary_list.append({
            'bead_name': b_name,
            'diameter_um': d_um,
            'n_experiments': ens_summary['n_exps'],
            'D_eff_GK_mean': ens_summary['d_eff_mean'],
            'D_eff_GK_std': ens_summary['d_eff_std'],
            'D_eff_GK_sem': ens_summary['d_eff_sem'],
            'D_eff_GK_pos_mean': ens_summary['d_eff_pos_mean'],
            'D_eff_GK_pos_sem': ens_summary['d_eff_pos_sem'],
            'D_eff_GK_ens_trapz': ens_summary['d_eff_ens_full'],
            'D_eff_GK_ens_pos': ens_summary['d_eff_ens_pos'],
            'C_v_0': ens_summary['c_v_0']
        })
        print(f"    -> Green-Kubo D_eff = {ens_summary['d_eff_mean']:.4f} ± {ens_summary['d_eff_sem']:.4f} um^2/s (N={ens_summary['n_exps']})")

    df_gk = pd.DataFrame(gk_summary_list)

    # 2. HMM パラメータからの RTP 理論計算 (min(tau_MSD, tau_dwell) 使用)
    if not hmm_summary_path.exists():
        raise FileNotFoundError(f"HMM state parameters file not found: {hmm_summary_path}")
        
    print(f"\n  [HMM RTP 理論モデル] パラメータ読み込み & 理論計算中 (tau_R = min(tau_MSD, tau_dwell)) ...")
    df_rtp = compute_rtp_theoretical_diffusion(hmm_summary_path, angle_summary_path, abp_summary_path)

    # 3. データの統合
    df_combined = pd.merge(df_gk, df_rtp, on=['bead_name', 'diameter_um'])

    # 4. CSV サマリーの保存
    csv_path = out_dir / 'effective_diffusion_summary.csv'
    df_combined.to_csv(csv_path, index=False)
    print(f"\n  [保存完了] サマリー CSV: {csv_path}")

    # 5. グラフの作成
    print(f"\n  [プロット生成中] メイン比較図 & 4パネル詳細図 ...")
    plot_effective_diffusion_comparison(df_combined, out_dir, use_log_scale=True)
    plot_detailed_4panel_analysis(vacf_ensemble_dict, df_combined, out_dir)

    print(f"\n{'='*75}")
    print("全解析・可視化が完了しました。")
    print(f"{'='*75}\n")

    return df_combined

