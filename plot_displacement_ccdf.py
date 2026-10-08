#!/usr/bin/env python3
"""
plot_displacement_ccdf.py

貨物微粒子の変位についての累積分布（相補累積分布関数: CCDF P(R >= r)）を両対数（log-log）でプロットし、
CCDF 90% 到達距離 Delta r_90 (tau = 300s 等) の算出・CSV 出力、
および（オプション指定時）外側のテール（大変位域）のべき乗則減衰 P(R >= r) ~ r^(-mu) における傾き -mu の
高精度フィッティング（KS統計量最小化法 & N>=5残存上限）を行うスクリプト。

マーカーの色・形状・プロットスタイルは MSD.py を厳密に踏襲しています。
- beads06um (0.63 um): 三角 '^' (青系 / style_colors[0])
- beads1um  (1.18 um): 丸 'o'   (橙系 / style_colors[1])
- beads3um  (3.37 um): 菱形 'd' (緑系 / style_colors[2])
- beads5um  (5.00 um): 五角形 'p' (赤系 / style_colors[3])
- beads7um  (7.24 um): 六角形 'h' (紫系 / style_colors[4])
- beads20um (20.0 um): 正方形 's' (茶系 / style_colors[5])
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import linregress

# libsディレクトリのインポート
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import cal_vel as cv
from libs import displacement as dpm

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ビーズ条件設定 (MSD.py に準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]


def find_default_root() -> Path:
    """有効なデータルートディレクトリを自動検出する。"""
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path]):
    """Figure を SVG および PNG 形式で指定全ディレクトリへ保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            fig.savefig(d / f"{basename}.png", bbox_inches='tight')
        except Exception as e:
            print(f"[WARNING] Failed to save {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    """DataFrame を CSV 形式で指定全ディレクトリへ保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / filename, index=False)
        except Exception as e:
            print(f"[WARNING] Failed to save {filename} to {d}: {e}")


def calc_empirical_ccdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    データ配列から相補累積分布関数 (CCDF: P(X >= x)) を算出する。
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return np.array([]), np.array([])
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    # P(X >= x_k) = (N - k + 1) / N
    ccdf = (n - np.arange(n)) / float(n)
    return sorted_arr, ccdf


def sample_log_spaced_points(x_arr: np.ndarray, y_arr: np.ndarray, num_points: int = 65) -> Tuple[np.ndarray, np.ndarray]:
    """
    両対数プロットで見やすくするため、対数空間で均等に間引いた代表点列を抽出する。
    """
    if len(x_arr) <= num_points:
        return x_arr, y_arr
    log_min = np.log10(x_arr[0])
    log_max = np.log10(x_arr[-1])
    target_log_x = np.linspace(log_min, log_max, num_points)
    
    indices = np.searchsorted(x_arr, 10**target_log_x)
    indices = np.clip(indices, 0, len(x_arr) - 1)
    unique_indices = np.unique(indices)
    return x_arr[unique_indices], y_arr[unique_indices]


def fit_powerlaw_tail(
    x_vals: np.ndarray,
    min_points: int = 15,
    min_tail_count: int = 5
) -> Optional[Dict]:
    """
    CCDF P(X >= x) の外側テール部に対して、
    - 最大値 x_max: 累積カウント N(X >= x) >= min_tail_count (既定: 5件) が残っている最大値
    - 最小値 x_min: Kolmogorov-Smirnov (KS) 統計量最小化法により自動決定
    を行い、最適区間 [x_min, x_max] において対数空間 log10(P) = -mu * log10(x) + C の線形回帰により
    傾き -mu を算出する。
    """
    arr = np.asarray(x_vals, dtype=float)
    sorted_x = np.sort(arr[np.isfinite(arr) & (arr > 0)])
    N = len(sorted_x)
    
    if N < min_points + min_tail_count:
        return None

    # 1. x_max: 累積カウント N(X >= x) >= min_tail_count (5件) 残っている最大値
    idx_max = N - min_tail_count
    x_max = float(sorted_x[idx_max])

    # 2. x_min: KS 統計量最小化による最適下限の探索
    idx_start = int(N * 0.15)  # 下位15%以降から探索
    idx_end = idx_max - min_points
    if idx_start >= idx_end:
        idx_start = max(0, idx_end - 30)

    # 候補点列 (最大250点)
    cand_indices = np.linspace(idx_start, idx_end, min(250, idx_end - idx_start + 1), dtype=int)
    cand_indices = np.unique(cand_indices)

    best_D = np.inf
    best_xmin = None
    best_alpha = None
    best_idx = None

    for idx in cand_indices:
        x_min_cand = float(sorted_x[idx])
        sub = sorted_x[(sorted_x >= x_min_cand) & (sorted_x <= x_max)]
        n = len(sub)
        if n < min_points:
            continue

        # 連続べき乗分布の MLE: alpha = 1 + n / sum(ln(x_i / x_min))
        log_ratios = np.log(sub / x_min_cand)
        sum_log = np.sum(log_ratios)
        if sum_log <= 0:
            continue
        alpha = 1.0 + n / sum_log

        # 経験的 CDF (区間 [x_min, x_max] 上)
        emp_cdf = np.arange(1, n + 1) / float(n)
        # 理論モデル CDF: 1 - (x / x_min)^(-(alpha - 1))
        theo_cdf = 1.0 - (sub / x_min_cand) ** (-(alpha - 1.0))
        theo_cdf = np.clip(theo_cdf, 0.0, 1.0)

        # Kolmogorov-Smirnov 統計量 D
        D = float(np.max(np.abs(emp_cdf - theo_cdf)))
        if D < best_D:
            best_D = D
            best_xmin = x_min_cand
            best_alpha = alpha
            best_idx = idx

    if best_xmin is None:
        return None

    # 3. 最適区間 [best_xmin, x_max] における CCDF の線形回帰フィッティング
    sub_mask = (sorted_x >= best_xmin) & (sorted_x <= x_max)
    x_fit_pts = sorted_x[sub_mask]
    ranks = np.where(sub_mask)[0]
    y_fit_pts = (N - ranks) / float(N)

    log_x = np.log10(x_fit_pts)
    log_y = np.log10(y_fit_pts)

    res = linregress(log_x, log_y)
    slope = float(res.slope)
    mu = float(-res.slope)
    intercept = float(res.intercept)
    r2 = float(res.rvalue**2)
    stderr = float(res.stderr) if res.stderr is not None else 0.0

    # プロット用の外挿・内挿フィッティング直線 (テールの描画用)
    x_min_eval = max(best_xmin * 0.96, 1e-3)
    x_max_eval = x_max * 1.08
    fit_x = np.logspace(np.log10(x_min_eval), np.log10(x_max_eval), 120)
    fit_y = 10.0 ** (slope * np.log10(fit_x) + intercept)

    return {
        'slope': slope,
        'mu': mu,
        'intercept': intercept,
        'r2': r2,
        'stderr': stderr,
        'p_value': float(res.pvalue),
        'ks_D': best_D,
        'mle_alpha': best_alpha,
        'x_min': best_xmin,
        'x_max': x_max,
        'y_min': float(y_fit_pts[-1]),
        'y_max': float(y_fit_pts[0]),
        'fit_x': fit_x,
        'fit_y': fit_y,
        'n_tail_points': len(x_fit_pts),
        'total_points': N
    }


def calc_ccdf_reach_stats(
    beads_data: Dict[str, Dict],
    tau: int,
    frame_interval: float
) -> pd.DataFrame:
    """
    各ビーズ条件における CCDF 90% 到達距離 Delta r_90 (P(R >= r) = 0.90) および
    中央値 (P(R >= r) = 0.50)、上位10% 到達距離 (P(R >= r) = 0.10, 90th percentile) 等の統計量を算出する。
    """
    rows = []
    tau_sec = tau * frame_interval

    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        data_dict = beads_data.get(b_name)
        if data_dict is None or len(data_dict["displacements"]) == 0:
            continue

        arr = np.asarray(data_dict["displacements"], dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            continue

        # CCDF P(R >= r) = 0.90 -> 下位 10% タイル (全粒子の90%がこの距離以上移動)
        r90_ccdf = float(np.percentile(arr, 10))
        # CCDF P(R >= r) = 0.50 -> 中央値 50% タイル
        r50_median = float(np.percentile(arr, 50))
        # CCDF P(R >= r) = 0.10 -> 上位 10% 到達距離 (90% percentile)
        r10_top90 = float(np.percentile(arr, 90))
        
        r95_ccdf = float(np.percentile(arr, 5))
        r99_ccdf = float(np.percentile(arr, 1))

        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_v = float(std_v / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

        rows.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "tau_frames": tau,
            "lag_time_s": tau_sec,
            "delta_r_ccdf_90pct_um": r90_ccdf,
            "delta_r_ccdf_50pct_median_um": r50_median,
            "delta_r_ccdf_10pct_top90_um": r10_top90,
            "delta_r_ccdf_95pct_um": r95_ccdf,
            "delta_r_ccdf_99pct_um": r99_ccdf,
            "mean_displacement_um": mean_v,
            "std_displacement_um": std_v,
            "sem_displacement_um": sem_v,
            "n_displacements": len(arr),
            "n_experiments": data_dict["n_files"]
        })

    return pd.DataFrame(rows)


def load_all_displacements(
    root_dir: Path,
    tau: int = 1,
    scale: float = 0.11,
    component: str = 'norm',
    signed: bool = False
) -> Dict[str, Dict]:
    """
    全ビーズ条件の変位データを一括読み込みし、辞書形式で返す。
    """
    beads_data = {}
    for item in BEADS_INFO:
        b_name = item["name"]
        files = sorted(glob.glob(str(root_dir / b_name / "*" / "*" / "beads_tracks.csv")))
        all_disps = []
        per_exp_disps = []
        
        for f in files:
            try:
                df = pd.read_csv(f)
                disp = dpm.calc_displacement_magnitudes(df, tau=tau, scale=scale, component=component, signed=signed)
                disp = disp[np.isfinite(disp) & (disp > 0)]
                if len(disp) > 0:
                    per_exp_disps.append(disp)
                    all_disps.extend(disp)
            except Exception as e:
                print(f"[WARNING] Could not process {f}: {e}")
                
        all_disps = np.array(all_disps, dtype=float)
        beads_data[b_name] = {
            "info": item,
            "n_files": len(files),
            "displacements": all_disps,
            "per_exp": per_exp_disps,
            "n_total": len(all_disps)
        }
    return beads_data


def plot_ccdf_comparison(
    beads_data: Dict[str, Dict],
    tau: int,
    frame_interval: float,
    component: str,
    out_dirs: List[Path],
    fit: bool = False,
    min_points: int = 15,
    min_tail_count: int = 5
) -> List[Dict]:
    """
    全ビーズサイズの変位累積分布 (CCDF) を1枚の両対数プロットに重ね合わせる。
    fit=True の場合のみ外側テールの傾き -mu のフィッティング直線および数値を描画する。
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    tau_sec = tau * frame_interval
    fit_summary = []

    # 軸ラベルの設定
    if component in ['norm', '2d', 'magnitude', 'r']:
        comp_symbol = r'|\Delta \boldsymbol{r}|'
    elif component == 'x':
        comp_symbol = r'|\Delta x|'
    elif component == 'y':
        comp_symbol = r'|\Delta y|'
    elif component in ['parallel', 'par']:
        comp_symbol = r'|\Delta r_\parallel|'
    else:
        comp_symbol = r'\Delta r'
        
    xlabel = rf'Displacement ${comp_symbol}$ [$\mu\mathrm{{m}}$]'
    ylabel = rf'Complementary Cumulative Distribution $P(R \geq {comp_symbol})$'

    for item in BEADS_INFO:
        b_name = item["name"]
        data_dict = beads_data.get(b_name)
        if data_dict is None or len(data_dict["displacements"]) == 0:
            continue
            
        disps = data_dict["displacements"]
        x_full, y_full = calc_empirical_ccdf(disps)
        if len(x_full) == 0:
            continue
            
        d_um = item["diameter_um"]
        color = item["color"]
        marker = item["marker"]
        
        fit_res = None
        if fit:
            # テールフィッティング (x_min: KS 統計量最小化, x_max: 累積カウント >= 5 件)
            fit_res = fit_powerlaw_tail(disps, min_points=min_points, min_tail_count=min_tail_count)
            
        if fit and fit_res is not None:
            slope_val = fit_res['slope']
            mu_val = fit_res['mu']
            r2_val = fit_res['r2']
            label_text = (
                f'{d_um:.2f} $\\mu\\mathrm{{m}}$: '
                f'slope $-\\mu = {slope_val:.2f}$ '
                f'($\\mu = {mu_val:.2f}\\pm{fit_res["stderr"]:.2f}$, $R^2={r2_val:.2f}$)'
            )
            fit_summary.append({
                "bead_name": b_name,
                "diameter_um": d_um,
                "tau_frames": tau,
                "lag_time_s": tau_sec,
                "component": component,
                "total_displacements": len(disps),
                "tail_slope_minus_mu": slope_val,
                "tail_exponent_mu": mu_val,
                "tail_exponent_stderr": fit_res["stderr"],
                "r_squared": r2_val,
                "ks_distance_D": fit_res["ks_D"],
                "tail_fit_r_min_um": fit_res["x_min"],
                "tail_fit_r_max_um": fit_res["x_max"],
                "tail_fit_n_points": fit_res["n_tail_points"]
            })
        else:
            label_text = f'{d_um:.2f} $\\mu\\mathrm{{m}}$ ($N={len(disps):,}$)'

        # 1. 連続線の薄いプロット（背景）
        ax.plot(x_full, y_full, color=color, alpha=0.35, linewidth=1.0, zorder=2)

        # 2. 代表点マーカー（対数等間隔サンプリング、MSD.py スタイル）
        x_sample, y_sample = sample_log_spaced_points(x_full, y_full, num_points=60)
        ax.plot(
            x_sample, y_sample,
            marker=marker,
            color=color,
            linestyle='none',
            markersize=6.8,
            alpha=0.88,
            label=label_text,
            zorder=3
        )

        # 3. テールフィッティング直線（太い破線, fit=True 時のみ）
        if fit and fit_res is not None:
            ax.plot(
                fit_res['fit_x'],
                fit_res['fit_y'],
                linestyle='--',
                color=color,
                linewidth=2.0,
                alpha=0.95,
                zorder=4
            )

    # 4. 傾き比較用ガイド線 (fit=True 時のみ描画)
    if fit:
        g_x0, g_x1 = 2.0, 6.5
        g_y0 = 6e-3
        g_y1_m2 = g_y0 * (g_x1 / g_x0) ** (-2.0)
        g_y1_m3 = (g_y0 * 0.45) * (g_x1 / g_x0) ** (-3.0)
        
        ax.plot([g_x0, g_x1], [g_y0, g_y1_m2], color='#333333', linestyle=':', linewidth=1.5, zorder=1)
        ax.text(g_x1 * 1.05, g_y1_m2, r'$\propto r^{-2.0}$', fontsize=11.0, color='#333333', fontweight='bold', va='center')

        ax.plot([g_x0, g_x1], [g_y0 * 0.45, g_y1_m3], color='#333333', linestyle=':', linewidth=1.5, zorder=1)
        ax.text(g_x1 * 1.05, g_y1_m3, r'$\propto r^{-3.0}$', fontsize=11.0, color='#333333', fontweight='bold', va='center')

    # 軸・スケール・レイアウト設定
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(3e-2, 60.0)
    ax.set_ylim(6e-5, 1.4)
    ax.set_xlabel(xlabel, fontsize=12.5, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12.5, fontweight='bold')
    
    title_comp = r"2D Displacement Norm $|\Delta \boldsymbol{r}|$" if component in ['norm', '2d', 'r'] else f"Component {component}"
    if fit:
        title_str = (
            rf'Cargo Displacement CCDF & Power-Law Tail Fits ($-\mu$)' '\n'
            rf'({title_comp}, $\Delta t = {tau_sec:.1f}\,\mathrm{{s}}$, $\tau = {tau}$ frames)'
        )
    else:
        title_str = (
            rf'Cargo Displacement Complementary Cumulative Distribution (CCDF)' '\n'
            rf'({title_comp}, $\Delta t = {tau_sec:.1f}\,\mathrm{{s}}$, $\tau = {tau}$ frames)'
        )
    ax.set_title(title_str, fontsize=13.0, fontweight='bold', pad=12)
    
    ax.legend(
        loc='lower left',
        fontsize=9.2,
        framealpha=0.94,
        edgecolor='#cccccc',
        handlelength=1.5,
        borderpad=0.45,
        labelspacing=0.32
    )
    ax.grid(True, which="both", ls="--", alpha=0.35)

    plt.tight_layout()
    save_figure_to_all(fig, f"displacement_ccdf_comparison_tau{tau}", out_dirs)
    plt.close(fig)
    print(f"[SAVED] CCDF Comparison Plot for tau={tau} (dt={tau_sec}s, fit={fit})")
    return fit_summary


def plot_multitau_grid(
    root_dir: Path,
    taus: List[int],
    frame_interval: float,
    component: str,
    out_dirs: List[Path],
    fit: bool = False,
    min_points: int = 15,
    min_tail_count: int = 5
) -> List[Dict]:
    """
    複数のラグタイム (例: 4s, 20s, 60s, 120s) における変位 CCDF を
    2x2 のグリッド状に並べて比較するプロット。
    """
    n_taus = len(taus)
    if n_taus == 0:
        return []
    
    nrows = 2 if n_taus > 1 else 1
    ncols = 2 if n_taus > 2 else n_taus
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows), squeeze=False)
    all_summary = []

    comp_symbol = r'|\Delta \boldsymbol{r}|' if component in ['norm', '2d', 'r'] else r'\Delta r'
    xlabel = rf'Displacement ${comp_symbol}$ [$\mu\mathrm{{m}}$]'
    ylabel = rf'CCDF $P(R \geq {comp_symbol})$'

    for idx, tau in enumerate(taus):
        r_idx = idx // ncols
        c_idx = idx % ncols
        ax = axes[r_idx, c_idx]
        tau_sec = tau * frame_interval

        b_data = load_all_displacements(root_dir, tau=tau, component=component)

        for item in BEADS_INFO:
            b_name = item["name"]
            data_dict = b_data.get(b_name)
            if data_dict is None or len(data_dict["displacements"]) == 0:
                continue

            disps = data_dict["displacements"]
            x_full, y_full = calc_empirical_ccdf(disps)
            if len(x_full) == 0:
                continue

            d_um = item["diameter_um"]
            color = item["color"]
            marker = item["marker"]

            fit_res = None
            if fit:
                fit_res = fit_powerlaw_tail(disps, min_points=min_points, min_tail_count=min_tail_count)

            if fit and fit_res is not None:
                lbl = f'{d_um:.2f} $\\mu\\mathrm{{m}}$ ($-\\mu = {fit_res["slope"]:.2f}$)'
                all_summary.append({
                    "bead_name": b_name,
                    "diameter_um": d_um,
                    "tau_frames": tau,
                    "lag_time_s": tau_sec,
                    "component": component,
                    "total_displacements": len(disps),
                    "tail_slope_minus_mu": fit_res["slope"],
                    "tail_exponent_mu": fit_res["mu"],
                    "tail_exponent_stderr": fit_res["stderr"],
                    "r_squared": fit_res["r2"],
                    "ks_distance_D": fit_res["ks_D"],
                    "tail_fit_r_min_um": fit_res["x_min"],
                    "tail_fit_r_max_um": fit_res["x_max"],
                    "tail_fit_n_points": fit_res["n_tail_points"]
                })
            else:
                lbl = f'{d_um:.2f} $\\mu\\mathrm{{m}}$'

            # 連続線 & サンプリングマーカー
            ax.plot(x_full, y_full, color=color, alpha=0.3, linewidth=0.8)
            x_sample, y_sample = sample_log_spaced_points(x_full, y_full, num_points=50)
            ax.plot(x_sample, y_sample, marker=marker, color=color, linestyle='none',
                    markersize=5.5, alpha=0.85, label=lbl)

            if fit and fit_res is not None:
                ax.plot(fit_res['fit_x'], fit_res['fit_y'], linestyle='--', color=color, linewidth=1.6, alpha=0.95)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3e-2, 60.0)
        ax.set_ylim(8e-5, 1.3)
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(rf'$\Delta t = {tau_sec:.1f}\,\mathrm{{s}}$ ($\tau = {tau}$ frames)', fontsize=11.5, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8.0, framealpha=0.9)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.suptitle(
        rf'Cargo Displacement CCDF Multi-Lag Time Grid' + (r' & Tail Slope $-\mu$' if fit else ''),
        fontsize=13.5, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    save_figure_to_all(fig, "displacement_ccdf_multitau_grid", out_dirs)
    plt.close(fig)
    print(f"[SAVED] Multi-Lag Time Grid Plot ({taus}, fit={fit})")
    return all_summary


def plot_per_bead_6panel(
    beads_data: Dict[str, Dict],
    tau: int,
    frame_interval: float,
    component: str,
    out_dirs: List[Path],
    fit: bool = False,
    min_points: int = 15,
    min_tail_count: int = 5
):
    """
    各ビーズサイズごとに個別のサブプロットを割り当てた 2x3 の 6 パネル図を作成・保存する。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    tau_sec = tau * frame_interval
    comp_symbol = r'|\Delta \boldsymbol{r}|' if component in ['norm', '2d', 'r'] else r'\Delta r'

    for idx, item in enumerate(BEADS_INFO):
        r_idx = idx // 3
        c_idx = idx % 3
        ax = axes[r_idx, c_idx]
        
        b_name = item["name"]
        d_um = item["diameter_um"]
        color = item["color"]
        marker = item["marker"]
        
        data_dict = beads_data.get(b_name)
        if data_dict is None or len(data_dict["displacements"]) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        disps = data_dict["displacements"]
        x_full, y_full = calc_empirical_ccdf(disps)
        
        fit_res = None
        if fit:
            fit_res = fit_powerlaw_tail(disps, min_points=min_points, min_tail_count=min_tail_count)

        # 全データ（薄いステップ線）
        ax.plot(x_full, y_full, color=color, alpha=0.35, linewidth=1.2)

        # サンプル点
        x_sample, y_sample = sample_log_spaced_points(x_full, y_full, num_points=70)
        ax.plot(x_sample, y_sample, marker=marker, color=color, linestyle='none',
                markersize=6.5, alpha=0.85, label=f'Data ($N={len(disps):,}$)')

        if fit and fit_res is not None:
            # テールフィッティング直線
            ax.plot(
                fit_res['fit_x'],
                fit_res['fit_y'],
                linestyle='--',
                color='#111111',
                linewidth=2.2,
                alpha=0.9,
                label=(
                    f'Tail Fit: $P \\propto r^{{-\\mu}}$\n'
                    f'  $-\\mu = {fit_res["slope"]:.3f}$\n'
                    f'  $\\mu = {fit_res["mu"]:.3f} \\pm {fit_res["stderr"]:.3f}$\n'
                    f'  $R^2 = {fit_res["r2"]:.3f}$, $D_{{\\mathrm{{KS}}}} = {fit_res["ks_D"]:.3f}$\n'
                    f'  Range: $[{fit_res["x_min"]:.2f}, {fit_res["x_max"]:.2f}]\\,\\mu\\mathrm{{m}}$\n'
                    f'  ($N_{{\\mathrm{{tail}}}} = {fit_res["n_tail_points"]}$)'
                )
            )
            # フィット領域のハイライト（薄い背景色）
            ax.axvspan(fit_res['x_min'], fit_res['x_max'], color=color, alpha=0.12, label='Fit Tail Region (KS min)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3e-2, 60.0)
        ax.set_ylim(8e-5, 1.3)
        ax.set_xlabel(rf'Displacement ${comp_symbol}$ [$\mu\mathrm{{m}}$]', fontsize=11, fontweight='bold')
        ax.set_ylabel(rf'CCDF $P(R \geq {comp_symbol})$', fontsize=11, fontweight='bold')
        ax.set_title(rf'$2R_c = {d_um:.2f}\,\mu\mathrm{{m}}$ ({b_name})', fontsize=12, fontweight='bold', color=color)
        ax.legend(loc='lower left', fontsize=8.0, framealpha=0.92)
        ax.grid(True, which="both", ls="--", alpha=0.35)

    title_main = (
        rf'Cargo Displacement CCDF & Power-Law Tail Fits per Bead Size ($\Delta t = {tau_sec:.1f}\,\mathrm{{s}}$)'
        if fit else
        rf'Cargo Displacement CCDF per Bead Size ($\Delta t = {tau_sec:.1f}\,\mathrm{{s}}$)'
    )
    plt.suptitle(title_main, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_figure_to_all(fig, f"displacement_ccdf_per_bead_6panel_tau{tau}", out_dirs)
    plt.close(fig)
    print(f"[SAVED] Per-Bead 6-Panel Detailed Plot for tau={tau} (fit={fit})")


def plot_tail_slope_vs_diameter(
    df_summary: pd.DataFrame,
    out_dirs: List[Path]
):
    """
    テールの傾き -mu（およびベキ指数 mu）の粒子径 2R_c 依存性をプロットする。
    """
    if df_summary.empty:
        return

    fig, ax1 = plt.subplots(figsize=(7.8, 5.5))
    ax2 = ax1.twinx()

    d_vals = []
    slopes = []
    mus = []
    stderrs = []
    colors = []
    markers = []

    for item in BEADS_INFO:
        b_name = item["name"]
        row = df_summary[df_summary["bead_name"] == b_name]
        if not row.empty:
            d_vals.append(item["diameter_um"])
            slopes.append(row["tail_slope_minus_mu"].values[0])
            mus.append(row["tail_exponent_mu"].values[0])
            stderrs.append(row["tail_exponent_stderr"].values[0])
            colors.append(item["color"])
            markers.append(item["marker"])

    d_arr = np.array(d_vals)
    slope_arr = np.array(slopes)
    mu_arr = np.array(mus)
    err_arr = np.array(stderrs)

    # 左軸: テールの傾き -mu (負の値)
    line1 = ax1.errorbar(
        d_arr, slope_arr, yerr=err_arr,
        fmt='o-', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.6,
        capsize=4.5, capthick=1.2, markersize=8.5, linewidth=2.0,
        label=r'Tail Slope $-\mu$ ($P(R \geq r) \propto r^{-\mu}$)'
    )
    for x_d, y_s, c, m in zip(d_arr, slope_arr, colors, markers):
        ax1.plot(x_d, y_s, marker=m, color=c, markersize=9.0, zorder=5)

    # 右軸: ベキ指数 mu (正の値)
    line2 = ax2.errorbar(
        d_arr, mu_arr, yerr=err_arr,
        fmt='s--', color='#d62728', ecolor='#d62728', elinewidth=1.6,
        capsize=4.5, capthick=1.2, markersize=8.0, linewidth=2.0,
        label=r'Tail Exponent $\mu$'
    )

    # 軸設定 (横軸)
    ax1.set_xscale('log')
    ax1.set_xlim(0.4, 28.0)
    ax1.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax1.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')

    # 左軸設定
    ax1.set_ylim(-22.0, -1.0)
    ax1.set_ylabel(r'Tail Slope $-\mu$', fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # 右軸設定
    ax2.set_ylim(1.0, 22.0)
    ax2.set_ylabel(r'Tail Exponent $\mu$', fontsize=12, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    ax1.grid(True, which='both', linestyle='--', alpha=0.35)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.92, fontsize=10.0)

    ax1.set_title(
        r'Cargo Displacement Power-Law Tail Slope $-\mu$ vs Diameter $2R_c$',
        fontsize=12.5,
        fontweight='bold',
        pad=10
    )

    save_figure_to_all(fig, "displacement_tail_slope_vs_diameter", out_dirs)
    plt.close(fig)
    print(f"[SAVED] Tail Slope vs Diameter Plot")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Displacement CCDF in log-log scale with optional power-law tail slope -mu fitting and delta r_90 reach calculation."
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing beads folders.")
    parser.add_argument("--tau", type=int, default=1, help="Lag time in frames (default: 1, corresponding to 4s).")
    parser.add_argument("--component", type=str, default="norm", help="Displacement component ('norm', 'x', 'parallel').")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0s).")
    parser.add_argument("--fit", action="store_true", default=False, help="Enable power-law tail fitting (default: False).")
    parser.add_argument("--min_tail_count", type=int, default=5, help="Minimum remaining count threshold for x_max (default: 5).")
    parser.add_argument("--min_points", type=int, default=15, help="Minimum number of data points in tail range for fit (default: 15).")
    parser.add_argument("--signed", action="store_true", help="Use signed displacement (default: absolute magnitude).")
    args = parser.parse_args()

    # データルート設定
    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    workspace_dir = Path(__file__).resolve().parent
    
    out_dirs = [
        workspace_dir / "figure" / "displacement_ccdf",
        workspace_dir / "figure" / "displacement",
        root_dir / "figure" / "displacement_ccdf",
    ]
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    print(f"Data Root: {root_dir}")
    print(f"Fitting Mode: {'ENABLED (--fit)' if args.fit else 'DISABLED (default, raw CCDF)'}")
    print(f"Output Directories:")
    for d in out_dirs:
        print(f"  - {d}")

    # 1. 指定ラグタイムの変位データ読み込み
    print(f"\n--- 1. Loading Displacements (tau={args.tau} frames, dt={args.tau * args.frame_interval}s, comp='{args.component}') ---")
    beads_data = load_all_displacements(
        root_dir=root_dir,
        tau=args.tau,
        component=args.component,
        signed=args.signed
    )
    for b_name, d_dict in beads_data.items():
        print(f"  {b_name}: {d_dict['n_total']} displacements across {d_dict['n_files']} files")

    # 2. CCDF 90% 到達距離 (Delta r_90) および統計サマリーの算出と CSV 保存
    print(f"\n--- 2. Calculating CCDF 90% Reach (Delta r_90) at tau = {args.tau} frames ({args.tau * args.frame_interval}s) ---")
    df_reach_curr = calc_ccdf_reach_stats(beads_data, tau=args.tau, frame_interval=args.frame_interval)
    save_csv_to_all(df_reach_curr, f"displacement_ccdf_reach_summary_tau{args.tau}.csv", out_dirs)
    print(df_reach_curr[["bead_name", "diameter_um", "delta_r_ccdf_90pct_um", "delta_r_ccdf_50pct_median_um", "delta_r_ccdf_10pct_top90_um", "mean_displacement_um", "n_displacements"]].to_string(index=False))

    # tau = 300s (tau=75 frames) の CCDF 90% 到達距離も常に計算・保存
    tau_300s_frames = int(round(300.0 / args.frame_interval))
    if tau_300s_frames != args.tau:
        print(f"\n--- 2b. Calculating CCDF 90% Reach (Delta r_90) at tau = 300s (tau = {tau_300s_frames} frames) ---")
        beads_data_300s = load_all_displacements(root_dir=root_dir, tau=tau_300s_frames, component=args.component, signed=args.signed)
        df_reach_300s = calc_ccdf_reach_stats(beads_data_300s, tau=tau_300s_frames, frame_interval=args.frame_interval)
        save_csv_to_all(df_reach_300s, "displacement_ccdf_reach_summary_tau300s.csv", out_dirs)
        save_csv_to_all(df_reach_300s, "displacement_ccdf_reach_summary.csv", out_dirs)
        print(df_reach_300s[["bead_name", "diameter_um", "delta_r_ccdf_90pct_um", "delta_r_ccdf_50pct_median_um", "delta_r_ccdf_10pct_top90_um", "mean_displacement_um", "n_displacements"]].to_string(index=False))
    else:
        save_csv_to_all(df_reach_curr, "displacement_ccdf_reach_summary_tau300s.csv", out_dirs)
        save_csv_to_all(df_reach_curr, "displacement_ccdf_reach_summary.csv", out_dirs)

    # 3. CCDF 比較プロットの生成
    print(f"\n--- 3. Plotting CCDF Comparison (fit={args.fit}) ---")
    fit_summary = plot_ccdf_comparison(
        beads_data=beads_data,
        tau=args.tau,
        frame_interval=args.frame_interval,
        component=args.component,
        out_dirs=out_dirs,
        fit=args.fit,
        min_points=args.min_points,
        min_tail_count=args.min_tail_count
    )

    # 4. ビーズサイズ別 6 パネル詳細図
    print(f"\n--- 4. Plotting Detailed 6-Panel Breakdown per Bead Size (fit={args.fit}) ---")
    plot_per_bead_6panel(
        beads_data=beads_data,
        tau=args.tau,
        frame_interval=args.frame_interval,
        component=args.component,
        out_dirs=out_dirs,
        fit=args.fit,
        min_points=args.min_points,
        min_tail_count=args.min_tail_count
    )

    # 5. マルチラグタイム グリッド図
    print(f"\n--- 5. Plotting Multi-Lag Time Grid (tau=1, 5, 15, 30 frames, fit={args.fit}) ---")
    grid_summary = plot_multitau_grid(
        root_dir=root_dir,
        taus=[1, 5, 15, 30],
        frame_interval=args.frame_interval,
        component=args.component,
        out_dirs=out_dirs,
        fit=args.fit,
        min_points=args.min_points,
        min_tail_count=args.min_tail_count
    )

    # 6. フィッティングが有効な場合はサマリー CSV とテール傾き vs 粒子径プロットを出力
    if args.fit:
        print(f"\n--- 6. Generating Tail Slope Fits Summary and Tail Slope vs Diameter Plot ---")
        df_summary = pd.DataFrame(fit_summary)
        if not df_summary.empty:
            save_csv_to_all(df_summary, "displacement_ccdf_tail_slopes_summary.csv", out_dirs)
            plot_tail_slope_vs_diameter(df_summary, out_dirs)
            
            print("\n=== Power-Law Tail Slope (-mu) Summary (KS Minimization x_min & N>=5 x_max) ===")
            print(df_summary[["bead_name", "diameter_um", "tail_slope_minus_mu", "tail_exponent_mu", "tail_exponent_stderr", "r_squared", "ks_distance_D", "tail_fit_r_min_um", "tail_fit_r_max_um", "tail_fit_n_points"]].to_string(index=False))

    print("\nAll CCDF analyses and figure generations completed successfully!")


if __name__ == "__main__":
    main()
