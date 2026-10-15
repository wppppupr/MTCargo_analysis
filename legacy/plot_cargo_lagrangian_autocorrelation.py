#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cargo_lagrangian_autocorrelation.py
========================================

貨物粒子（ビーズ）の軌跡に沿ってサンプリングされたイジングスピン磁化 M(t) の
ラグランジュ的自己相関関数（Lagrangian Autocorrelation Function）:

    C_M(tau) = < M(t) M(t + tau) >

および正規化自己相関関数:

    \\tilde{C}_M(tau) = < M(t) M(t + tau) > / < M(0)^2 >

および揺らぎの正規化自己相関関数（Connected Autocorrelation Function）:

    g_M(tau) = \\frac{< M(t) M(t + tau) > - <M>^2}{<M^2> - <M>^2}
             = \\frac{< \\delta M(t) \\delta M(t + tau) >}{< \\delta M^2 >}
             (\\text{ただし } \\delta M(t) = M(t) - <M>)

を各貨物粒子径（beads06um, beads1um, beads3um, beads5um, beads7um, beads20um）ごとに
広いラグ時間（既定 1200 s）にわたって計算・可視化し、線形スケールおよび
**縦軸対数スケール（Semilog-y / 片対数プロット）** で描画します。
指数減衰フィッティングにより磁場相関時間 tau_M および tau_{gM} を算出します。

【領域サイズ（Region size under cargo）】
評価領域の半径は、小粒子（0.63 um, 1.18 um）の微小管サンプリング数を担保したスケール:

    R_region = max( region_factor * R_c, min_region_um )

（既定: region_factor = 2.0, min_region_um = 3.37 um）に準拠します。
すでに計算済みの cargo_spin_velocity_points.csv が存在する場合はそこから高速に集計し、
--recompute を指定した場合は生フローデータ（GFP_flows.h5 / flow cache）から直接再抽出します。

【出力ファイル（既定: figure/cargo_lagrangian_autocorrelation/）】
[線形スケール]
1. cargo_lagrangian_autocorrelation_overlay.png / .svg
   - 全ビーズサイズの C_M(tau) = <M(t)M(t+tau)> 重ね描き比較（SEM エラーバンド付き）
2. cargo_lagrangian_autocorrelation_normalized_overlay.png / .svg
   - 正規化自己相関 \\tilde{C}_M(tau) の重ね描き比較
3. cargo_lagrangian_connected_autocorrelation_overlay.png / .svg
   - 揺らぎの正規化自己相関 g_M(tau) の重ね描き比較（SEM エラーバンド付き）
4. cargo_lagrangian_autocorrelation_panels.png / .svg
   - 粒子径ごとの C_M(tau) 個別パネル（個別トラック + アンサンブル平均 +- SEM + 指数減衰フィット）
5. cargo_lagrangian_connected_autocorrelation_panels.png / .svg
   - 粒子径ごとの g_M(tau) 個別パネル（個別トラック + アンサンブル平均 +- SEM + 指数減衰フィット）

[縦軸対数スケール（Semilog-y）]
6. cargo_lagrangian_autocorrelation_overlay_semilog.png / .svg
7. cargo_lagrangian_autocorrelation_normalized_overlay_semilog.png / .svg
8. cargo_lagrangian_connected_autocorrelation_overlay_semilog.png / .svg
9. cargo_lagrangian_autocorrelation_panels_semilog.png / .svg
10. cargo_lagrangian_connected_autocorrelation_panels_semilog.png / .svg

[スケーリング & データ]
11. cargo_lagrangian_correlation_time_vs_diameter.png / .svg
12. cargo_lagrangian_connected_correlation_time_vs_diameter.png / .svg
13. cargo_lagrangian_autocorrelation_curves.csv
14. cargo_lagrangian_autocorrelation_summary.csv
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# NAS / 共有ボリュームでの HDF5 ファイルロックエラー防止
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

# 親ディレクトリのパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import plot_mt_orientation_distribution as mt_ori
import plot_cargo_spin_velocity as cargo_spin

# スタイル適用
_style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if _style_path.exists():
    try:
        plt.style.use(str(_style_path))
    except Exception:
        pass

BEADS_INFO = mt_ori.BEADS_INFO
BEAD_LOOKUP = mt_ori.BEAD_LOOKUP


# =============================================================================
# 指数減衰モデル関数
# =============================================================================

def exp_decay_with_offset(t: np.ndarray, A: float, tau: float, C_inf: float) -> np.ndarray:
    """C(tau) = A * exp(-tau / tau_M) + C_inf"""
    return A * np.exp(-t / np.maximum(tau, 1e-6)) + C_inf


def fit_autocorrelation_decay(
    lags: np.ndarray,
    corr: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_lag_fit: Optional[float] = None,
    p0_custom: Optional[List[float]] = None,
) -> dict:
    """
    自己相関関数 C(tau) に指数減衰モデル C(tau) = A * exp(-tau / tau_M) + C_inf をフィッティングする。
    """
    valid = np.isfinite(lags) & np.isfinite(corr) & (lags >= 0)
    if max_lag_fit is not None:
        valid = valid & (lags <= max_lag_fit)

    t_fit = lags[valid]
    y_fit = corr[valid]

    if len(t_fit) < 4:
        return {
            'success': False,
            'tau_M': np.nan,
            'tau_M_err': np.nan,
            'A': np.nan,
            'C_inf': np.nan,
            'r2': np.nan,
            'fit_fn': None,
        }

    c0 = y_fit[0]
    cinf_guess = float(np.mean(y_fit[-max(1, len(y_fit) // 4):]))
    a_guess = float(c0 - cinf_guess)
    tau_guess = float(np.median(t_fit)) if len(t_fit) > 0 else 50.0

    sigma = None
    if weights is not None:
        w_fit = weights[valid]
        w_valid = np.isfinite(w_fit) & (w_fit > 0)
        if np.all(w_valid):
            sigma = 1.0 / np.maximum(w_fit, 1e-6)

    try:
        p0 = p0_custom if p0_custom is not None else [a_guess, max(1.0, tau_guess), cinf_guess]
        bounds = ([-2.0, 0.1, -1.5], [2.0, 5000.0, 1.5])
        popt, pcov = curve_fit(
            exp_decay_with_offset, t_fit, y_fit,
            p0=p0, bounds=bounds, sigma=sigma, maxfev=5000
        )
        a_fit, tau_fit, cinf_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan, np.nan]
        tau_err = perr[1]

        y_pred = exp_decay_with_offset(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            'success': True,
            'tau_M': float(tau_fit),
            'tau_M_err': float(tau_err),
            'A': float(a_fit),
            'C_inf': float(cinf_fit),
            'r2': float(r2),
            'fit_fn': lambda t: exp_decay_with_offset(t, a_fit, tau_fit, cinf_fit),
        }
    except Exception as e:
        return {
            'success': False,
            'tau_M': np.nan,
            'tau_M_err': np.nan,
            'A': np.nan,
            'C_inf': np.nan,
            'r2': np.nan,
            'fit_fn': None,
            'error': str(e),
        }


# =============================================================================
# ラグランジュ自己相関の計算
# =============================================================================

def compute_particle_lagrangian_autocorr(
    df_p: pd.DataFrame,
    max_lag_s: float = 1200.0,
    time_col: str = 'time_s',
    val_col: str = 'm_ising',
) -> Tuple[Dict[float, List[float]], Dict[float, List[float]]]:
    """
    1 粒子の軌跡 (t, M(t)) から、各ラグ時間 tau における
    1. 積ペア M(t) * M(t + tau)
    2. 偏差積ペア delta M(t) * delta M(t + tau) （粒子内平均基準）
    を計算する。
    """
    df_sorted = df_p.sort_values(time_col).dropna(subset=[time_col, val_col]).copy()
    times = df_sorted[time_col].to_numpy(dtype=float)
    vals = df_sorted[val_col].to_numpy(dtype=float)
    n = len(times)
    if n < 2:
        return {}, {}

    mean_p = float(np.mean(vals))
    pairs_by_lag: Dict[float, List[float]] = {}
    fluc_pairs_by_lag: Dict[float, List[float]] = {}

    for i in range(n):
        t0 = times[i]
        v0 = vals[i]
        dv0 = v0 - mean_p
        for j in range(i, n):
            dt = round(float(times[j] - t0), 3)
            if dt > max_lag_s:
                break
            prod = float(v0 * vals[j])
            fluc_prod = float(dv0 * (vals[j] - mean_p))

            if dt not in pairs_by_lag:
                pairs_by_lag[dt] = []
                fluc_pairs_by_lag[dt] = []
            pairs_by_lag[dt].append(prod)
            fluc_pairs_by_lag[dt].append(fluc_prod)

    return pairs_by_lag, fluc_pairs_by_lag


def compute_bead_autocorrelation(
    df_points: pd.DataFrame,
    bead_name: str,
    max_lag_s: float = 1200.0,
    min_pairs: int = 5,
    time_col: str = 'time_s',
    val_col: str = 'm_ising',
) -> dict:
    """
    あるビーズサイズ条件における全粒子のラグランジュ自己相関 C_M(tau) および
    揺らぎの正規化自己相関 g_M(tau) を計算・集計する。
    """
    df_b = df_points[df_points['bead_name'] == bead_name].copy()
    if df_b.empty:
        return {}

    bead = BEAD_LOOKUP.get(bead_name, {})
    diameter_um = float(bead.get('diameter_um', np.nan))
    radius_um = float(bead.get('radius_um', np.nan))

    # 全体平均 <M> と分散 <M^2> - <M>^2
    all_m = df_b[val_col].dropna().to_numpy(dtype=float)
    mean_m = float(np.mean(all_m)) if len(all_m) > 0 else 0.0
    mean_m_sq = float(np.mean(all_m ** 2)) if len(all_m) > 0 else 1.0
    var_m = float(mean_m_sq - mean_m ** 2)
    if var_m <= 0:
        var_m = float(np.var(all_m)) if len(all_m) > 0 else 1.0

    # 全粒子ごとのペアを収集
    grouped = df_b.groupby(['exp_dir', 'particle'])
    n_particles = grouped.ngroups

    all_pairs_by_lag: Dict[float, List[float]] = {}
    particle_curves: List[dict] = []

    for (exp_dir, pid), grp in grouped:
        p_pairs, p_fluc = compute_particle_lagrangian_autocorr(
            grp, max_lag_s=max_lag_s, time_col=time_col, val_col=val_col
        )
        if not p_pairs:
            continue

        p_lags = sorted(p_pairs.keys())
        p_means = [float(np.mean(p_pairs[l])) for l in p_lags]
        # 粒子ごとの g_M 曲線
        p_fluc_means = [float(np.mean(p_fluc[l])) for l in p_lags]
        p_var = float(np.mean(p_fluc.get(0.0, [1.0])))
        p_gm = [val / p_var if p_var > 0 else np.nan for val in p_fluc_means]

        particle_curves.append({
            'exp_dir': exp_dir,
            'particle': pid,
            'lags': np.array(p_lags, dtype=float),
            'corr': np.array(p_means, dtype=float),
            'g_m': np.array(p_gm, dtype=float),
            'n_pairs': np.array([len(p_pairs[l]) for l in p_lags], dtype=int),
        })

        for lag, prods in p_pairs.items():
            if lag not in all_pairs_by_lag:
                all_pairs_by_lag[lag] = []
            all_pairs_by_lag[lag].extend(prods)

    if not all_pairs_by_lag:
        return {}

    sorted_lags = sorted(all_pairs_by_lag.keys())
    lag_list = []
    mean_list = []
    sem_list = []
    std_list = []
    count_list = []
    norm_list = []
    gm_list = []
    gm_sem_list = []

    c0 = float(np.mean(all_pairs_by_lag.get(0.0, [1.0])))

    for lag in sorted_lags:
        prods = np.array(all_pairs_by_lag[lag], dtype=float)
        cnt = len(prods)
        if cnt < min_pairs:
            continue
        m_val = float(np.mean(prods))
        s_val = float(np.std(prods, ddof=1)) if cnt > 1 else 0.0
        sem_val = s_val / np.sqrt(cnt) if cnt > 1 else 0.0

        # g_M(tau) = (<M(t)M(t+tau)> - <M>^2) / (<M^2> - <M>^2)
        gm_val = (m_val - mean_m ** 2) / var_m if var_m != 0 else np.nan
        gm_sem = sem_val / var_m if var_m != 0 else np.nan

        lag_list.append(lag)
        mean_list.append(m_val)
        std_list.append(s_val)
        sem_list.append(sem_val)
        count_list.append(cnt)
        norm_list.append(m_val / c0 if c0 != 0 else np.nan)
        gm_list.append(gm_val)
        gm_sem_list.append(gm_sem)

    lags_arr = np.array(lag_list, dtype=float)
    mean_arr = np.array(mean_list, dtype=float)
    sem_arr = np.array(sem_list, dtype=float)
    count_arr = np.array(count_list, dtype=int)
    norm_arr = np.array(norm_list, dtype=float)
    gm_arr = np.array(gm_list, dtype=float)
    gm_sem_arr = np.array(gm_sem_list, dtype=float)

    # 積分相関時間 tau_int = integral g_M(tau) d tau
    if len(lags_arr) > 1 and not np.isnan(gm_arr).all():
        valid_int = (gm_arr > 0)
        if np.any(valid_int):
            first_neg = np.where(~valid_int)[0]
            cutoff_idx = first_neg[0] if len(first_neg) > 0 else len(lags_arr)
            x_int = lags_arr[:cutoff_idx]
            y_int = gm_arr[:cutoff_idx]
            if hasattr(np, 'trapezoid'):
                tau_int = float(np.trapezoid(y_int, x_int))
            else:
                tau_int = float(np.sum(0.5 * (y_int[:-1] + y_int[1:]) * np.diff(x_int)))
        else:
            tau_int = np.nan
    else:
        tau_int = np.nan

    # 1. C_M(tau) 指数減衰フィット
    fit_cm = fit_autocorrelation_decay(lags_arr, mean_arr, weights=(1.0 / np.maximum(sem_arr, 1e-4)))

    # 2. g_M(tau) 指数減衰フィット: g_M(tau) = A * exp(-tau / tau_{gM}) + C_inf (p0=[1.0, tau_guess, 0.0])
    fit_gm = fit_autocorrelation_decay(lags_arr, gm_arr, weights=(1.0 / np.maximum(gm_sem_arr, 1e-4)),
                                      p0_custom=[1.0, 30.0, 0.0])

    return {
        'bead_name': bead_name,
        'diameter_um': diameter_um,
        'radius_um': radius_um,
        'region_radius_um': float(df_b['region_radius_um'].iloc[0]) if 'region_radius_um' in df_b.columns else np.nan,
        'n_particles': n_particles,
        'n_points': len(df_b),
        'mean_m': mean_m,
        'mean_m_sq': mean_m_sq,
        'var_m': var_m,
        'c0': c0,
        'lags': lags_arr,
        'mean': mean_arr,
        'sem': sem_arr,
        'std': np.array(std_list, dtype=float),
        'count': count_arr,
        'normalized': norm_arr,
        'g_m': gm_arr,
        'g_m_sem': gm_sem_arr,
        'tau_int': tau_int,
        'fit': fit_cm,
        'fit_gm': fit_gm,
        'particle_curves': particle_curves,
    }


# =============================================================================
# 作図ルーチン (1: C_M(tau) 未正規化・正規化)
# =============================================================================

def plot_lagrangian_autocorr_overlay(
    results: Dict[str, dict],
    output_path: Path,
    max_lag_s: float = 1200.0,
    normalized: bool = False,
    yscale: str = 'linear',
    title_suffix: str = "",
) -> None:
    """
    全ビーズサイズのラグランジュ自己相関関数 C_M(tau) を 1 つの図に重ね描きする。
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)
    is_log = (yscale == 'log')

    for binfo in BEADS_INFO:
        bname = binfo['name']
        if bname not in results or not results[bname]:
            continue
        res = results[bname]
        lags = res['lags']
        if normalized:
            y = res['normalized']
            yerr = res['sem'] / res['c0'] if res['c0'] != 0 else res['sem']
        else:
            y = res['mean']
            yerr = res['sem']

        mask = (lags <= max_lag_s)
        if not np.any(mask):
            continue

        l_sub = lags[mask]
        y_sub = y[mask]
        err_sub = yerr[mask]

        if is_log:
            pos_mask = (y_sub > 1e-4)
            if not np.any(pos_mask):
                continue
            l_sub = l_sub[pos_mask]
            y_sub = y_sub[pos_mask]
            err_sub = err_sub[pos_mask]

        color = binfo.get('color', '#333333')
        marker = binfo.get('marker', 'o')
        label = f"{binfo['diameter_um']:.2f} $\\mu$m (N={res['n_particles']})"

        markevery = max(1, len(l_sub) // 20)
        ax.plot(l_sub, y_sub, marker=marker, color=color, markersize=4.2,
                linewidth=1.7, label=label, markevery=markevery)
        
        if is_log:
            y_low = np.maximum(y_sub - err_sub, 1e-4)
            y_high = y_sub + err_sub
            ax.fill_between(l_sub, y_low, y_high, color=color, alpha=0.15)
        else:
            ax.fill_between(l_sub, y_sub - err_sub, y_sub + err_sub, color=color, alpha=0.18)

    ax.set_xlabel(r'Lag time $\tau$ [s]', fontsize=11)
    if is_log:
        ax.set_yscale('log')
        if normalized:
            ax.set_ylim(1e-3, 1.6)
            ax.set_ylabel(r'Normalized Autocorrelation $\tilde{C}_M(\tau)$ (log scale)', fontsize=10.5)
            ax.set_title(r'Lagrangian Autocorrelation $\tilde{C}_M(\tau)$ [Semilog-y]' + title_suffix, fontsize=11.5, fontweight='bold')
        else:
            ax.set_ylim(1e-3, 1.2)
            ax.set_ylabel(r'Lagrangian Autocorrelation $C_M(\tau) = \langle M(t) M(t+\tau) \rangle$ (log scale)', fontsize=10.5)
            ax.set_title(r'Lagrangian Autocorrelation $C_M(\tau)$ [Semilog-y]' + title_suffix, fontsize=11.5, fontweight='bold')
    else:
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
        if normalized:
            ax.set_ylabel(r'Normalized Autocorrelation $\tilde{C}_M(\tau) = \frac{\langle M(t) M(t+\tau) \rangle}{\langle M(0)^2 \rangle}$', fontsize=10.5)
            ax.set_title(r'Lagrangian Magnetization Autocorrelation $\tilde{C}_M(\tau)$' + title_suffix, fontsize=11.5, fontweight='bold')
        else:
            ax.set_ylabel(r'Lagrangian Autocorrelation $C_M(\tau) = \langle M(t) M(t+\tau) \rangle$', fontsize=10.5)
            ax.set_title(r'Lagrangian Magnetization Autocorrelation $C_M(\tau) = \langle M(t) M(t+\tau) \rangle$' + title_suffix, fontsize=11.5, fontweight='bold')

    ax.set_xlim(0, max_lag_s)
    ax.grid(True, which='both' if is_log else 'major', linestyle=':', alpha=0.6)
    ax.legend(title='Cargo Diameter $2R_c$', frameon=True, fontsize=8.8, title_fontsize=9.2, loc='best')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path} and {svg_path}")


def plot_lagrangian_autocorr_panels(
    results: Dict[str, dict],
    output_path: Path,
    max_lag_s: float = 1200.0,
    ncols: int = 3,
    yscale: str = 'linear',
    title_suffix: str = "",
) -> None:
    """
    ビーズサイズごとに個別パネルを作成し、個別粒子トラック・アンサンブル平均・フィット曲線を表示する。
    """
    valid_beads = [b for b in BEADS_INFO if b['name'] in results and results[b['name']]]
    n_beads = len(valid_beads)
    if n_beads == 0:
        return

    is_log = (yscale == 'log')
    nrows = (n_beads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), dpi=300, squeeze=False)

    for idx, binfo in enumerate(valid_beads):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        bname = binfo['name']
        res = results[bname]
        color = binfo.get('color', '#333333')
        marker = binfo.get('marker', 'o')

        if is_log:
            ax.set_yscale('log')

        # 1. 個別粒子の曲線（薄い線）
        p_curves = res.get('particle_curves', [])
        for pc in p_curves:
            p_lags = pc['lags']
            p_corr = pc['corr']
            mask = (p_lags <= max_lag_s)
            if is_log:
                mask = mask & (p_corr > 1e-4)
            if np.any(mask):
                ax.plot(p_lags[mask], p_corr[mask], color=color, alpha=0.15, linewidth=0.8)

        # 2. アンサンブル平均 +- SEM
        lags = res['lags']
        mean = res['mean']
        sem = res['sem']
        mask = (lags <= max_lag_s)
        if is_log:
            mask = mask & (mean > 1e-4)
        l_sub = lags[mask]
        m_sub = mean[mask]
        s_sub = sem[mask]

        if len(l_sub) > 0:
            markevery = max(1, len(l_sub) // 15)
            ax.plot(l_sub, m_sub, marker=marker, color=color, markersize=3.8, linewidth=1.8,
                    label=r'Ensemble $\langle C_M(\tau) \rangle$', markevery=markevery, zorder=4)
            if is_log:
                y_low = np.maximum(m_sub - s_sub, 1e-4)
                y_high = m_sub + s_sub
                ax.fill_between(l_sub, y_low, y_high, color=color, alpha=0.22, zorder=3)
            else:
                ax.fill_between(l_sub, m_sub - s_sub, m_sub + s_sub, color=color, alpha=0.25, zorder=3)

        # 3. 指数減衰フィット
        fit = res.get('fit', {})
        if fit.get('success', False) and fit.get('fit_fn') is not None:
            t_fit_line = np.linspace(0, max_lag_s, 300)
            y_fit_line = fit['fit_fn'](t_fit_line)
            tau_m = fit['tau_M']
            tau_err = fit['tau_M_err']
            r2 = fit['r2']
            fit_lbl = f"Fit: $\\tau_M = {tau_m:.1f} \\pm {tau_err:.1f}$ s\n($R^2 = {r2:.2f}$)"
            if is_log:
                pos_fit = (y_fit_line > 1e-4)
                ax.plot(t_fit_line[pos_fit], y_fit_line[pos_fit], 'k--', linewidth=1.4, label=fit_lbl, zorder=5)
            else:
                ax.plot(t_fit_line, y_fit_line, 'k--', linewidth=1.4, label=fit_lbl, zorder=5)

        if is_log:
            ax.set_ylim(1e-3, 1.2)
        else:
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

        ax.set_xlim(0, max_lag_s)
        ax.set_xlabel(r'Lag time $\tau$ [s]', fontsize=9.5)
        ax.set_ylabel(r'$C_M(\tau)$' + (' (log)' if is_log else ''), fontsize=9.5)
        
        region_str = f", $R_{{\\mathrm{{reg}}}}={res['region_radius_um']:.2f}\\,\\mu$m" if np.isfinite(res.get('region_radius_um', np.nan)) else ""
        ax.set_title(f"{binfo['diameter_um']:.2f} $\\mu$m (N={res['n_particles']}{region_str})",
                     fontsize=10.5, fontweight='bold', color=color)
        ax.grid(True, which='both' if is_log else 'major', linestyle=':', alpha=0.5)
        ax.legend(frameon=True, fontsize=7.8, loc='best')

    # 未使用サブプロットの非表示
    for idx in range(n_beads, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')

    scale_tag = ' [Semilog-y]' if is_log else ''
    fig.suptitle(r'Cargo-Tracking Lagrangian Magnetization Autocorrelation $C_M(\tau)$ by Bead Size' + scale_tag + title_suffix,
                 fontsize=12.5, fontweight='bold', y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path} and {svg_path}")


# =============================================================================
# 作図ルーチン (2: g_M(tau) 揺らぎの正規化自己相関)
# =============================================================================

def plot_lagrangian_connected_autocorr_overlay(
    results: Dict[str, dict],
    output_path: Path,
    max_lag_s: float = 1200.0,
    yscale: str = 'linear',
    title_suffix: str = "",
) -> None:
    """
    全ビーズサイズの揺らぎ正規化自己相関関数
        g_M(tau) = (<M(t)M(t+tau)> - <M>^2) / (<M^2> - <M>^2)
    を 1 つの図に重ね描きする。
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)
    is_log = (yscale == 'log')

    for binfo in BEADS_INFO:
        bname = binfo['name']
        if bname not in results or not results[bname]:
            continue
        res = results[bname]
        lags = res['lags']
        y = res['g_m']
        yerr = res['g_m_sem']

        mask = (lags <= max_lag_s)
        if not np.any(mask):
            continue

        l_sub = lags[mask]
        y_sub = y[mask]
        err_sub = yerr[mask]

        if is_log:
            pos_mask = (y_sub > 1e-4)
            if not np.any(pos_mask):
                continue
            l_sub = l_sub[pos_mask]
            y_sub = y_sub[pos_mask]
            err_sub = err_sub[pos_mask]

        color = binfo.get('color', '#333333')
        marker = binfo.get('marker', 'o')
        label = f"{binfo['diameter_um']:.2f} $\\mu$m (N={res['n_particles']})"

        markevery = max(1, len(l_sub) // 20)
        ax.plot(l_sub, y_sub, marker=marker, color=color, markersize=4.2,
                linewidth=1.7, label=label, markevery=markevery)
        
        if is_log:
            y_low = np.maximum(y_sub - err_sub, 1e-4)
            y_high = y_sub + err_sub
            ax.fill_between(l_sub, y_low, y_high, color=color, alpha=0.15)
        else:
            ax.fill_between(l_sub, y_sub - err_sub, y_sub + err_sub, color=color, alpha=0.18)

    ax.set_xlabel(r'Lag time $\tau$ [s]', fontsize=11)
    if is_log:
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1.6)
        ax.set_ylabel(r'Connected Autocorrelation $g_M(\tau)$ (log scale)', fontsize=10.5)
        ax.set_title(r'Lagrangian Fluctuation Autocorrelation $g_M(\tau)$ [Semilog-y]' + title_suffix, fontsize=11.5, fontweight='bold')
    else:
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(r'Connected Autocorrelation $g_M(\tau) = \frac{\langle M(t) M(t+\tau) \rangle - \langle M \rangle^2}{\langle M^2 \rangle - \langle M \rangle^2}$', fontsize=10.5)
        ax.set_title(r'Lagrangian Magnetization Fluctuation Autocorrelation $g_M(\tau)$' + title_suffix, fontsize=11.5, fontweight='bold')
        ax.set_ylim(-0.35, 1.15)

    ax.set_xlim(0, max_lag_s)
    ax.grid(True, which='both' if is_log else 'major', linestyle=':', alpha=0.6)
    ax.legend(title='Cargo Diameter $2R_c$', frameon=True, fontsize=8.8, title_fontsize=9.2, loc='best')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path} and {svg_path}")


def plot_lagrangian_connected_autocorr_panels(
    results: Dict[str, dict],
    output_path: Path,
    max_lag_s: float = 1200.0,
    ncols: int = 3,
    yscale: str = 'linear',
    title_suffix: str = "",
) -> None:
    """
    ビーズサイズごとに個別パネルを作成し、揺らぎ自己相関 g_M(tau) の個別トラック・アンサンブル平均・フィット曲線を表示する。
    """
    valid_beads = [b for b in BEADS_INFO if b['name'] in results and results[b['name']]]
    n_beads = len(valid_beads)
    if n_beads == 0:
        return

    is_log = (yscale == 'log')
    nrows = (n_beads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), dpi=300, squeeze=False)

    for idx, binfo in enumerate(valid_beads):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        bname = binfo['name']
        res = results[bname]
        color = binfo.get('color', '#333333')
        marker = binfo.get('marker', 'o')

        if is_log:
            ax.set_yscale('log')

        # 1. 個別粒子の曲線（薄い線）
        p_curves = res.get('particle_curves', [])
        for pc in p_curves:
            p_lags = pc['lags']
            p_gm = pc['g_m']
            mask = (p_lags <= max_lag_s) & np.isfinite(p_gm)
            if is_log:
                mask = mask & (p_gm > 1e-4)
            if np.any(mask):
                ax.plot(p_lags[mask], p_gm[mask], color=color, alpha=0.15, linewidth=0.8)

        # 2. アンサンブル平均 +- SEM
        lags = res['lags']
        gm = res['g_m']
        gm_sem = res['g_m_sem']
        mask = (lags <= max_lag_s)
        if is_log:
            mask = mask & (gm > 1e-4)
        l_sub = lags[mask]
        g_sub = gm[mask]
        s_sub = gm_sem[mask]

        if len(l_sub) > 0:
            markevery = max(1, len(l_sub) // 15)
            ax.plot(l_sub, g_sub, marker=marker, color=color, markersize=3.8, linewidth=1.8,
                    label=r'Ensemble $g_M(\tau)$', markevery=markevery, zorder=4)
            if is_log:
                y_low = np.maximum(g_sub - s_sub, 1e-4)
                y_high = g_sub + s_sub
                ax.fill_between(l_sub, y_low, y_high, color=color, alpha=0.22, zorder=3)
            else:
                ax.fill_between(l_sub, g_sub - s_sub, g_sub + s_sub, color=color, alpha=0.25, zorder=3)

        # 3. 指数減衰フィット
        fit = res.get('fit_gm', {})
        if fit.get('success', False) and fit.get('fit_fn') is not None:
            t_fit_line = np.linspace(0, max_lag_s, 300)
            y_fit_line = fit['fit_fn'](t_fit_line)
            tau_gm = fit['tau_M']
            tau_err = fit['tau_M_err']
            r2 = fit['r2']
            fit_lbl = f"Fit: $\\tau_{{gM}} = {tau_gm:.1f} \\pm {tau_err:.1f}$ s\n($R^2 = {r2:.2f}$)"
            if is_log:
                pos_fit = (y_fit_line > 1e-4)
                ax.plot(t_fit_line[pos_fit], y_fit_line[pos_fit], 'k--', linewidth=1.4, label=fit_lbl, zorder=5)
            else:
                ax.plot(t_fit_line, y_fit_line, 'k--', linewidth=1.4, label=fit_lbl, zorder=5)

        if is_log:
            ax.set_ylim(1e-3, 1.6)
        else:
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.set_ylim(-0.4, 1.2)

        ax.set_xlim(0, max_lag_s)
        ax.set_xlabel(r'Lag time $\tau$ [s]', fontsize=9.5)
        ax.set_ylabel(r'$g_M(\tau)$' + (' (log)' if is_log else ''), fontsize=9.5)
        
        region_str = f", $R_{{\\mathrm{{reg}}}}={res['region_radius_um']:.2f}\\,\\mu$m" if np.isfinite(res.get('region_radius_um', np.nan)) else ""
        ax.set_title(f"{binfo['diameter_um']:.2f} $\\mu$m (N={res['n_particles']}{region_str})",
                     fontsize=10.5, fontweight='bold', color=color)
        ax.grid(True, which='both' if is_log else 'major', linestyle=':', alpha=0.5)
        ax.legend(frameon=True, fontsize=7.8, loc='best')

    # 未使用サブプロットの非表示
    for idx in range(n_beads, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')

    scale_tag = ' [Semilog-y]' if is_log else ''
    fig.suptitle(r'Cargo-Tracking Fluctuation Autocorrelation $g_M(\tau)$ by Bead Size' + scale_tag + title_suffix,
                 fontsize=12.5, fontweight='bold', y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path} and {svg_path}")


# =============================================================================
# 相関時間 vs 粒子径プロット
# =============================================================================

def plot_correlation_time_vs_diameter(
    results: Dict[str, dict],
    output_path: Path,
    use_gm: bool = False,
    title_suffix: str = "",
) -> None:
    """
    フィッティングから得られた磁場相関時間 tau_M (または tau_{gM}) および積分相関時間 tau_int vs 貨物粒子径をプロットする。
    """
    diameters = []
    radii = []
    tau_m_list = []
    tau_err_list = []
    tau_int_list = []
    colors = []
    markers = []
    labels = []

    for binfo in BEADS_INFO:
        bname = binfo['name']
        if bname not in results or not results[bname]:
            continue
        res = results[bname]
        d = float(binfo['diameter_um'])
        r = float(binfo['radius_um'])
        fit = res.get('fit_gm', {}) if use_gm else res.get('fit', {})
        tau_m = fit.get('tau_M', np.nan)
        tau_err = fit.get('tau_M_err', np.nan)
        tau_int = res.get('tau_int', np.nan)

        diameters.append(d)
        radii.append(r)
        tau_m_list.append(tau_m)
        tau_err_list.append(tau_err if np.isfinite(tau_err) else 0.0)
        tau_int_list.append(tau_int)
        colors.append(binfo.get('color', '#333333'))
        markers.append(binfo.get('marker', 'o'))
        labels.append(f"{d:.2f} $\\mu$m")

    if not diameters:
        return

    diameters = np.array(diameters)
    tau_m_arr = np.array(tau_m_list)
    tau_err_arr = np.array(tau_err_list)
    tau_int_arr = np.array(tau_int_list)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=300)

    # tau_fit (exponential fit)
    valid_fit = np.isfinite(tau_m_arr)
    fit_label = r'Fitted fluctuation timescale $\tau_{gM}$' if use_gm else r'Fitted correlation timescale $\tau_M$'
    if np.any(valid_fit):
        for d, t_m, t_err, c, m in zip(diameters[valid_fit], tau_m_arr[valid_fit], tau_err_arr[valid_fit],
                                      [colors[i] for i in range(len(colors)) if valid_fit[i]],
                                      [markers[i] for i in range(len(markers)) if valid_fit[i]]):
            ax.errorbar(d, t_m, yerr=t_err, fmt=m, color=c, ecolor=c, ms=7.0, capsize=3.5,
                        elinewidth=1.5, zorder=5)
        ax.plot(diameters[valid_fit], tau_m_arr[valid_fit], 'k-', alpha=0.6, linewidth=1.5,
                label=fit_label)

    # tau_int (Green-Kubo integral)
    valid_int = np.isfinite(tau_int_arr)
    if np.any(valid_int):
        ax.plot(diameters[valid_int], tau_int_arr[valid_int], 's--', color='darkorange', ms=6.0,
                linewidth=1.4, alpha=0.8, label=r'Integral timescale $\tau_{\mathrm{int}}$')

    ax.set_xlabel(r'Cargo Diameter $2 R_c$ [$\mu$m]', fontsize=11)
    ax.set_ylabel(r'Characteristic Correlation Time $\tau$ [s]', fontsize=11)
    title_str = r'Fluctuation Timescale $\tau_{gM}$ vs Cargo Diameter' if use_gm else r'Magnetization Correlation Timescale $\tau_M$ vs Cargo Diameter'
    ax.set_title(title_str + title_suffix, fontsize=11.5, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=9.5, loc='best')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path} and {svg_path}")


# =============================================================================
# CSV テーブル出力
# =============================================================================

def save_autocorr_tables(
    results: Dict[str, dict],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    1. 詳細な曲線データ (cargo_lagrangian_autocorrelation_curves.csv)
    2. サマリーテーブル (cargo_lagrangian_autocorrelation_summary.csv)
    を出力する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 曲線データ
    curve_rows = []
    for binfo in BEADS_INFO:
        bname = binfo['name']
        if bname not in results or not results[bname]:
            continue
        res = results[bname]
        lags = res['lags']
        for i, lag in enumerate(lags):
            curve_rows.append({
                'bead_name': bname,
                'diameter_um': res['diameter_um'],
                'radius_um': res['radius_um'],
                'region_radius_um': res.get('region_radius_um', np.nan),
                'lag_time_s': float(lag),
                'C_M_mean': float(res['mean'][i]),
                'C_M_sem': float(res['sem'][i]),
                'C_M_std': float(res['std'][i]),
                'n_pairs': int(res['count'][i]),
                'n_particles': int(res['n_particles']),
                'C_M_normalized': float(res['normalized'][i]),
                'g_M_connected': float(res['g_m'][i]),
                'g_M_sem': float(res['g_m_sem'][i]),
            })

    df_curves = pd.DataFrame(curve_rows)
    curves_csv = output_dir / 'cargo_lagrangian_autocorrelation_curves.csv'
    df_curves.to_csv(curves_csv, index=False)

    # 2. サマリーテーブル
    summary_rows = []
    for binfo in BEADS_INFO:
        bname = binfo['name']
        if bname not in results or not results[bname]:
            continue
        res = results[bname]
        fit = res.get('fit', {})
        fit_gm = res.get('fit_gm', {})
        summary_rows.append({
            'bead_name': bname,
            'diameter_um': res['diameter_um'],
            'radius_um': res['radius_um'],
            'region_radius_um': res.get('region_radius_um', np.nan),
            'n_particles': int(res['n_particles']),
            'n_points': int(res['n_points']),
            'mean_magnetization': float(res['mean_m']),
            'mean_square_magnetization': float(res['mean_m_sq']),
            'variance_magnetization': float(res['var_m']),
            'C_M_zero': float(res['c0']),
            # C_M fit
            'fit_CM_success': bool(fit.get('success', False)),
            'tau_M_s': float(fit.get('tau_M', np.nan)),
            'tau_M_err_s': float(fit.get('tau_M_err', np.nan)),
            'fit_CM_amp_A': float(fit.get('A', np.nan)),
            'fit_CM_C_inf': float(fit.get('C_inf', np.nan)),
            'fit_CM_r2': float(fit.get('r2', np.nan)),
            # g_M fit
            'fit_gM_success': bool(fit_gm.get('success', False)),
            'tau_gM_s': float(fit_gm.get('tau_M', np.nan)),
            'tau_gM_err_s': float(fit_gm.get('tau_M_err', np.nan)),
            'fit_gM_amp_A': float(fit_gm.get('A', np.nan)),
            'fit_gM_C_inf': float(fit_gm.get('C_inf', np.nan)),
            'fit_gM_r2': float(fit_gm.get('r2', np.nan)),
            'tau_int_s': float(res.get('tau_int', np.nan)),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / 'cargo_lagrangian_autocorrelation_summary.csv'
    df_summary.to_csv(summary_csv, index=False)

    print(f"Saved: {curves_csv}")
    print(f"Saved: {summary_csv}")
    return curves_csv, summary_csv


# =============================================================================
# CLI パーサー & メイン処理
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cargo-tracking (Lagrangian) Ising magnetization autocorrelation C_M(tau) and connected fluctuation autocorrelation g_M(tau) for all bead sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_csv', type=str,
                        default='figure/cargo_spin_velocity/cargo_spin_velocity_points.csv',
                        help="すでに抽出されたポイントデータ CSV のパス（存在すれば高速読み込み）")
    parser.add_argument('--root_dir', type=str, default=None,
                        help="生データルート（--recompute 時に使用）")
    parser.add_argument('--recompute', action='store_true',
                        help="CSV を使わず生フローデータから再抽出して計算する")
    parser.add_argument('--beads', type=str, nargs='+', default='all',
                        help="対象の粒子径（beads06um beads1um ... または all）")
    parser.add_argument('--output_dir', type=str, default='figure/cargo_lagrangian_autocorrelation',
                        help="出力ディレクトリ")
    parser.add_argument('--no_save_root', action='store_true',
                        help="データルート側への保存を行わない")
    parser.add_argument('--region_factor', type=float, default=2.0,
                        help="領域半径係数 R_region = max(region_factor * R_c, min_region_um)")
    parser.add_argument('--min_region_um', type=float, default=3.37,
                        help="領域半径の下限 [um]（小粒子のサンプリング担保用既定 3.37 um）")
    parser.add_argument('--pixel_stride', type=int, default=4,
                        help="フロー画素間引き幅（--recompute 時）")
    parser.add_argument('--frame_stride', type=int, default=5,
                        help="フレーム間引き幅（--recompute 時）")
    parser.add_argument('--max_lag_s', type=float, default=1200.0,
                        help="計算・プロットする最大ラグ時間 [s]（既定 1200 s = 20 分）")
    parser.add_argument('--min_pairs', type=int, default=5,
                        help="ラグビンに含める最小ペア数")
    parser.add_argument('--director', type=str, default='global', choices=['global', 'local'],
                        help="ディレクター軸の定義（--recompute 時）")
    parser.add_argument('--smooth_method', type=str, default='moving_average',
                        choices=['moving_average', 'savgol', 'none'],
                        help="軌跡平滑化方式（--recompute 時）")
    parser.add_argument('--smooth_window', type=int, default=3,
                        help="平滑化窓幅 [frames]（--recompute 時）")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = CURRENT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv_path = Path(args.input_csv)
    if not input_csv_path.is_absolute():
        input_csv_path = CURRENT_DIR / input_csv_path

    df_points: Optional[pd.DataFrame] = None

    if not args.recompute and input_csv_path.exists():
        print(f"Loading existing points from: {input_csv_path}")
        try:
            df_points = pd.read_csv(input_csv_path)
            print(f"Loaded {len(df_points)} records for beads: {df_points['bead_name'].unique().tolist()}")
        except Exception as e:
            print(f"Failed to read {input_csv_path}: {e}; will attempt recompute.", flush=True)

    if df_points is None:
        print("Extracting magnetization data from raw flow / tracks...")
        root_dir = Path(args.root_dir) if args.root_dir else mt_ori.find_default_root()
        if root_dir is None or not root_dir.exists():
            print(f"Error: Data root directory not found. Please specify with --root_dir.", file=sys.stderr)
            sys.exit(1)

        # 対象ビーズの選定
        if isinstance(args.beads, str) and args.beads == 'all':
            target_beads = BEADS_INFO
        elif isinstance(args.beads, list) and ('all' in args.beads or args.beads == ['all']):
            target_beads = BEADS_INFO
        else:
            raw_list = args.beads if isinstance(args.beads, list) else [args.beads]
            norm_names = set(mt_ori.normalize_bead_name(b) for b in raw_list)
            target_beads = [b for b in BEADS_INFO if b['name'] in norm_names]

        all_results = []
        for bead in target_beads:
            bname = bead['name']
            bead_dir = root_dir / bname
            if not bead_dir.exists():
                continue
            exp_dirs = sorted([d for d in bead_dir.glob('*/*') if (d / mt_ori.FLOW_NAME).exists()])
            for ed in exp_dirs:
                res = cargo_spin.process_experiment_cargo(
                    ed, bead,
                    pixel_stride=args.pixel_stride,
                    frame_stride=args.frame_stride,
                    region_factor=args.region_factor,
                    min_region_um=args.min_region_um,
                    director=args.director,
                    smooth_method=args.smooth_method,
                    smooth_window=args.smooth_window,
                )
                if res is not None:
                    all_results.append(res)

        if not all_results:
            print("Error: No data successfully extracted.", file=sys.stderr)
            sys.exit(1)

        df_points = cargo_spin.points_table(all_results, sign=1)
        print(f"Extracted {len(df_points)} points.")

    # 対象ビーズの決定
    if isinstance(args.beads, str) and args.beads == 'all':
        target_bead_names = [b['name'] for b in BEADS_INFO]
    elif isinstance(args.beads, list) and ('all' in args.beads or args.beads == ['all']):
        target_bead_names = [b['name'] for b in BEADS_INFO]
    else:
        raw_list = args.beads if isinstance(args.beads, list) else [args.beads]
        target_bead_names = [mt_ori.normalize_bead_name(b) for b in raw_list if mt_ori.normalize_bead_name(b)]

    # 各ビーズサイズの自己相関を計算
    results: Dict[str, dict] = {}
    print(f"\nComputing Lagrangian autocorrelation C_M(tau) & g_M(tau) (max_lag={args.max_lag_s:.0f} s) for beads: {target_bead_names}")
    for bname in target_bead_names:
        res = compute_bead_autocorrelation(
            df_points, bname, max_lag_s=args.max_lag_s, min_pairs=args.min_pairs
        )
        if res:
            results[bname] = res
            tau_m = res['fit'].get('tau_M', np.nan)
            r2_cm = res['fit'].get('r2', np.nan)
            tau_gm = res['fit_gm'].get('tau_M', np.nan)
            r2_gm = res['fit_gm'].get('r2', np.nan)
            print(f"  [{bname}] N_particles={res['n_particles']}, C_M(0)={res['c0']:.4f}, "
                  f"tau_M={tau_m:.2f} s (R^2={r2_cm:.2f}), tau_gM={tau_gm:.2f} s (R^2={r2_gm:.2f}), "
                  f"tau_int={res['tau_int']:.2f} s")

    if not results:
        print("No valid autocorrelation curves computed.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Generating Linear Scale Plots ---")
    # 1. C_M(tau) 線形重ね描きプロット
    overlay_path = output_dir / 'cargo_lagrangian_autocorrelation_overlay.png'
    plot_lagrangian_autocorr_overlay(results, overlay_path, max_lag_s=args.max_lag_s, normalized=False, yscale='linear')

    # 2. C_M(tau) / C_M(0) 線形正規化重ね描きプロット
    norm_overlay_path = output_dir / 'cargo_lagrangian_autocorrelation_normalized_overlay.png'
    plot_lagrangian_autocorr_overlay(results, norm_overlay_path, max_lag_s=args.max_lag_s, normalized=True, yscale='linear')

    # 3. g_M(tau) 線形揺らぎ正規化自己相関 重ね描きプロット
    conn_overlay_path = output_dir / 'cargo_lagrangian_connected_autocorrelation_overlay.png'
    plot_lagrangian_connected_autocorr_overlay(results, conn_overlay_path, max_lag_s=args.max_lag_s, yscale='linear')

    # 4. C_M(tau) 線形パネルプロット
    panels_path = output_dir / 'cargo_lagrangian_autocorrelation_panels.png'
    plot_lagrangian_autocorr_panels(results, panels_path, max_lag_s=args.max_lag_s, ncols=3, yscale='linear')

    # 5. g_M(tau) 線形パネルプロット
    conn_panels_path = output_dir / 'cargo_lagrangian_connected_autocorrelation_panels.png'
    plot_lagrangian_connected_autocorr_panels(results, conn_panels_path, max_lag_s=args.max_lag_s, ncols=3, yscale='linear')

    print("\n--- Generating Semilog-y (Log Scale) Plots ---")
    # 6. C_M(tau) 対数重ね描きプロット
    overlay_log_path = output_dir / 'cargo_lagrangian_autocorrelation_overlay_semilog.png'
    plot_lagrangian_autocorr_overlay(results, overlay_log_path, max_lag_s=args.max_lag_s, normalized=False, yscale='log')

    # 7. C_M(tau) / C_M(0) 対数正規化重ね描きプロット
    norm_overlay_log_path = output_dir / 'cargo_lagrangian_autocorrelation_normalized_overlay_semilog.png'
    plot_lagrangian_autocorr_overlay(results, norm_overlay_log_path, max_lag_s=args.max_lag_s, normalized=True, yscale='log')

    # 8. g_M(tau) 対数揺らぎ正規化自己相関 重ね描きプロット
    conn_overlay_log_path = output_dir / 'cargo_lagrangian_connected_autocorrelation_overlay_semilog.png'
    plot_lagrangian_connected_autocorr_overlay(results, conn_overlay_log_path, max_lag_s=args.max_lag_s, yscale='log')

    # 9. C_M(tau) 対数パネルプロット
    panels_log_path = output_dir / 'cargo_lagrangian_autocorrelation_panels_semilog.png'
    plot_lagrangian_autocorr_panels(results, panels_log_path, max_lag_s=args.max_lag_s, ncols=3, yscale='log')

    # 10. g_M(tau) 対数パネルプロット
    conn_panels_log_path = output_dir / 'cargo_lagrangian_connected_autocorrelation_panels_semilog.png'
    plot_lagrangian_connected_autocorr_panels(results, conn_panels_log_path, max_lag_s=args.max_lag_s, ncols=3, yscale='log')

    print("\n--- Generating Scaling Plots & CSV Tables ---")
    # 11. 相関時間 vs 粒子径
    scaling_path = output_dir / 'cargo_lagrangian_correlation_time_vs_diameter.png'
    plot_correlation_time_vs_diameter(results, scaling_path, use_gm=False)

    # 12. 揺らぎ相関時間 vs 粒子径
    conn_scaling_path = output_dir / 'cargo_lagrangian_connected_correlation_time_vs_diameter.png'
    plot_correlation_time_vs_diameter(results, conn_scaling_path, use_gm=True)

    # 13. CSV 保存
    save_autocorr_tables(results, output_dir)

    print("\n=== Lagrangian Autocorrelation Analysis Complete ===")


if __name__ == '__main__':
    main()
