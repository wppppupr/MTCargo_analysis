"""
displacement_analysis.py

貨物微粒子（蛍光ビーズ）の変位絶対値（2次元ノルム |Δr|、1次元 |Δx|, |Δy|、大域ネマチック主軸射影 |Δr_parallel|, |Δr_perp|）
を算出し、全ビーズ条件（0.6μm 〜 20μm）を対象に変位確率密度関数（PDF / ヒストグラム）
および統計量を一括解析・可視化するスクリプトです。
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

# libsディレクトリのインポートパス解決
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import displacement as dpm

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ビーズ条件設定
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": 10,  "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": 11,  "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTSingleBeads'),
]


def find_default_root():
    for r in POSSIBLE_ROOTS:
        if r.exists():
            # ビーズフォルダが実際に存在するか確認
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir, bead_condition):
    """
    指定ビーズ条件下の有効な実験ディレクトリ（beads_tracks.csvを含む）を探索する。
    """
    base = Path(root_dir) / bead_condition
    if not base.exists():
        return []

    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)

    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)

    return exp_dirs


def safe_save_csv(df, target_path, max_retries=5):
    """
    ネットワークドライブ等でのファイルロック(BlockingIOError)を防ぐため、
    リトライおよびフォールバック書き込みを行う。
    """
    target_path = Path(target_path)
    for attempt in range(max_retries):
        try:
            df.to_csv(target_path, index=False)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    csv_text = df.to_csv(index=False)
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(csv_text)
                    return
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    return
            import time
            time.sleep(0.5)


_THETA_CACHE = {}

def load_theta_array(exp_dir):
    """
    実験ディレクトリ内の MTs_im_theta.zarr から大域ネマチック平均配向角 theta(t) を取得する。
    メモリキャッシュを用いて高速化。
    """
    exp_dir_str = str(exp_dir)
    if exp_dir_str in _THETA_CACHE:
        return _THETA_CACHE[exp_dir_str]

    theta_path = Path(exp_dir) / "MTs_im_theta.zarr"
    if not theta_path.exists():
        _THETA_CACHE[exp_dir_str] = None
        return None
    try:
        MTs = zarr.open_array(str(theta_path), mode='r')
        theta = np.nanmean(MTs, axis=(1, 2))
        _THETA_CACHE[exp_dir_str] = theta
        return theta
    except Exception as e:
        print(f"[WARNING] Could not load theta from {theta_path}: {e}", flush=True)
        _THETA_CACHE[exp_dir_str] = None
        return None


_TRACKS_CACHE = {}

def collect_displacements(exp_dirs, tau, scale=0.11, component='norm', signed=False):
    """
    複数の実験ディレクトリから指定 tau の変位データを集約する。
    プールされた全変位配列と、実験ごとの変位配列リストを返す。
    トラックデータをメモリキャッシュして高速化。
    """
    all_disps = []
    exp_disps = []

    for d in exp_dirs:
        d_str = str(d)
        if d_str in _TRACKS_CACHE:
            df_tracks = _TRACKS_CACHE[d_str]
        else:
            tracks_path = d / "beads_tracks.csv"
            try:
                df_tracks = pd.read_csv(tracks_path)
                _TRACKS_CACHE[d_str] = df_tracks
            except Exception as e:
                print(f"[WARNING] Failed to read {tracks_path}: {e}", flush=True)
                _TRACKS_CACHE[d_str] = None
                continue

        if df_tracks is None:
            continue

        theta_array = None
        if component.lower() in ['parallel', 'par', 'perpendicular', 'perp']:
            theta_array = load_theta_array(d)
            if theta_array is None:
                continue

        try:
            disp = dpm.calc_displacement_magnitudes(
                df_tracks,
                tau=tau,
                scale=scale,
                component=component,
                theta_array=theta_array,
                signed=signed
            )
            if len(disp) > 0:
                all_disps.extend(disp)
                exp_disps.append(np.asarray(disp))
        except Exception as e:
            print(f"[ERROR] Error calculating displacement for {d}: {e}", flush=True)

    return {
        "pooled": np.array(all_disps),
        "per_exp": exp_disps
    }


def calc_ensemble_pdf(exp_disp_list, bins=50, bin_range=None, density=True):
    """
    実験ごとの変位データリストから、実験間平均PDFと標準偏差 (std) を算出する。
    """
    if not exp_disp_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_data = np.concatenate(exp_disp_list)
    if len(all_data) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if bin_range is None:
        bin_range = (float(np.nanmin(all_data)), float(np.nanmax(all_data)))

    _, bin_edges = np.histogram(all_data, bins=bins, range=bin_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pdf_matrix = []
    for exp_disp in exp_disp_list:
        counts, _ = np.histogram(exp_disp, bins=bin_edges, density=density)
        pdf_matrix.append(counts)

    pdf_matrix = np.array(pdf_matrix)  # shape: (n_exp, n_bins)
    mean_pdf = np.nanmean(pdf_matrix, axis=0)
    std_pdf = np.nanstd(pdf_matrix, axis=0, ddof=1) if len(exp_disp_list) > 1 else np.zeros_like(mean_pdf)

    return bin_centers, mean_pdf, std_pdf, bin_edges


def get_component_label(component, signed=False):
    """
    プロット軸ラベル用の文字列表現を取得する。
    """
    comp = component.lower()
    if comp in ['norm', '2d', 'magnitude', 'r']:
        return r'Displacement magnitude $|\Delta \mathbf{r}|$ [$\mu\mathrm{m}$]', r'$P(|\Delta \mathbf{r}|)$'
    elif comp == 'x':
        if signed:
            return r'Displacement $\Delta x$ [$\mu\mathrm{m}$]', r'$P(\Delta x)$'
        return r'Absolute displacement $|\Delta x|$ [$\mu\mathrm{m}$]', r'$P(|\Delta x|)$'
    elif comp == 'y':
        if signed:
            return r'Displacement $\Delta y$ [$\mu\mathrm{m}$]', r'$P(\Delta y)$'
        return r'Absolute displacement $|\Delta y|$ [$\mu\mathrm{m}$]', r'$P(|\Delta y|)$'
    elif comp == 'both_xy':
        if signed:
            return r'Displacement $\Delta x, \Delta y$ [$\mu\mathrm{m}$]', r'$P(\Delta x, \Delta y)$'
        return r'Absolute displacement $|\Delta x|, |\Delta y|$ [$\mu\mathrm{m}$]', r'$P(|\Delta x|, |\Delta y|)$'
    elif comp in ['parallel', 'par']:
        if signed:
            return r'Parallel displacement $\Delta r_\parallel$ [$\mu\mathrm{m}$]', r'$P(\Delta r_\parallel)$'
        return r'Parallel abs. displacement $|\Delta r_\parallel|$ [$\mu\mathrm{m}$]', r'$P(|\Delta r_\parallel|)$'
    elif comp in ['perpendicular', 'perp']:
        if signed:
            return r'Perpendicular displacement $\Delta r_\perp$ [$\mu\mathrm{m}$]', r'$P(\Delta r_\perp)$'
        return r'Perpendicular abs. displacement $|\Delta r_\perp|$ [$\mu\mathrm{m}$]', r'$P(|\Delta r_\perp|)$'
    return f'Displacement {component} [$\\mu\\mathrm{{m}}$]', f'PDF $P({component})$'


def plot_pdf_across_beads(beads_data, tau, frame_interval, component, signed, out_path, 
                          xscale='linear', yscale='log', bins=50, error_style='band', xlim=(0, 50),
                          fit_exp=True, fit_mode='log', fit_rmin=None, fit_rmax=None):
    """
    全ビーズサイズを1つの図で比較する変位PDFプロットを作成・保存する。
    実験ごとの標準偏差エラーバー／エラーバンドおよび指数関数フィッティングを描画。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tau_sec = tau * frame_interval
    xlabel, ylabel = get_component_label(component, signed)
    fit_results = {}

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_data or len(beads_data[b_name]["per_exp"]) == 0:
            continue

        exp_list = beads_data[b_name]["per_exp"]
        bin_range = xlim if xlim is not None else None
        centers, mean_pdf, std_pdf, edges = calc_ensemble_pdf(exp_list, bins=bins, bin_range=bin_range, density=True)
        
        valid = (mean_pdf > 0) & (centers > 0) if (xscale == 'log' or yscale == 'log') else (mean_pdf >= 0)
        
        n_exps = len(exp_list)
        label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($N={n_exps}$)'

        # 指数関数フィッティング P(r) = A * exp(-r / lambda) （デフォルトで対数空間 ln P(r) でフィット）
        fit_res = None
        if fit_exp and np.sum(valid) >= 3:
            fit_res = dpm.fit_exponential_pdf(
                centers[valid],
                mean_pdf[valid],
                pdf_std=std_pdf[valid] if np.all(std_pdf[valid] > 0) else None,
                r_min=fit_rmin,
                r_max=fit_rmax,
                fit_mode=fit_mode
            )
            if fit_res is not None:
                fit_results[b_name] = fit_res
                label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($\lambda={fit_res["lambda"]:.2f}\\,\\mu\\mathrm{{m}}$, $R^2={fit_res["r_squared"]:.2f}$)'

        # 1. 平均曲線のプロット
        ax.plot(
            centers[valid],
            mean_pdf[valid],
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            markersize=6,
            alpha=0.9,
            linestyle='none' if fit_exp else '-'
        )

        # 2. 指数フィッティング曲線の描画（破線）
        if fit_res is not None:
            ax.plot(
                fit_res['fit_x'],
                fit_res['fit_y'],
                linestyle='--',
                color=item["color"],
                alpha=0.85,
                linewidth=1.5
            )

        # 3. 実験間標準偏差エラーバー / エラーバンド（MSDと同様）
        if error_style in ['band', 'both']:
            ax.fill_between(
                centers[valid],
                np.clip(mean_pdf[valid] - std_pdf[valid], 1e-6 if yscale == 'log' else 0, None),
                mean_pdf[valid] + std_pdf[valid],
                edgecolor=item["color"],
                facecolor=mcolors.to_rgba(item["color"], alpha=0.2),
                linewidth=0.5
            )
        if error_style in ['bar', 'both']:
            ax.errorbar(
                centers[valid],
                mean_pdf[valid],
                yerr=std_pdf[valid],
                fmt='none',
                ecolor=item["color"],
                elinewidth=1.0,
                capsize=2,
                alpha=0.6
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(f'Displacement PDF & Exponential Fits ($\Delta t = {tau_sec:.1f}\\mathrm{{s}}$, $\\tau = {tau}$ frames)')
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)
    return fit_results


def plot_multitau_grid(all_tau_data, frame_interval, component, signed, out_path, 
                       xscale='linear', yscale='log', bins=40, error_style='band', xlim=(0, 50),
                       fit_exp=True, fit_mode='log', fit_rmin=None, fit_rmax=None):
    """
    複数のラグタイム tau をグリッド状に並べたサマリープロットを作成・保存する。
    """
    tau_list = list(all_tau_data.keys())
    n_tau = len(tau_list)
    if n_tau == 0:
        return

    n_cols = min(3, n_tau)
    n_rows = (n_tau + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.8 * n_rows), squeeze=False)
    xlabel, ylabel = get_component_label(component, signed)

    for idx, tau in enumerate(tau_list):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        tau_sec = tau * frame_interval
        beads_data = all_tau_data[tau]

        for item in BEADS_INFO:
            b_name = item["name"]
            if b_name not in beads_data or len(beads_data[b_name]["per_exp"]) == 0:
                continue

            exp_list = beads_data[b_name]["per_exp"]
            bin_range = xlim if xlim is not None else None
            centers, mean_pdf, std_pdf, _ = calc_ensemble_pdf(exp_list, bins=bins, bin_range=bin_range, density=True)
            valid = (mean_pdf > 0) & (centers > 0) if (xscale == 'log' or yscale == 'log') else (mean_pdf >= 0)

            label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$'
            fit_res = None
            if fit_exp and np.sum(valid) >= 3:
                fit_res = dpm.fit_exponential_pdf(
                    centers[valid],
                    mean_pdf[valid],
                    pdf_std=std_pdf[valid] if np.all(std_pdf[valid] > 0) else None,
                    r_min=fit_rmin,
                    r_max=fit_rmax,
                    fit_mode=fit_mode
                )
                if fit_res is not None:
                    label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($\lambda={fit_res["lambda"]:.1f}$)'

            ax.plot(
                centers[valid],
                mean_pdf[valid],
                marker=item["marker"],
                color=item["color"],
                label=label_text,
                markersize=5,
                alpha=0.9,
                linestyle='none' if fit_exp else '-'
            )

            if fit_res is not None:
                ax.plot(
                    fit_res['fit_x'],
                    fit_res['fit_y'],
                    linestyle='--',
                    color=item["color"],
                    alpha=0.85,
                    linewidth=1.2
                )

            # 実験間標準偏差エラーバンド (fill_between)
            if error_style in ['band', 'both']:
                ax.fill_between(
                    centers[valid],
                    np.clip(mean_pdf[valid] - std_pdf[valid], 1e-6 if yscale == 'log' else 0, None),
                    mean_pdf[valid] + std_pdf[valid],
                    edgecolor=item["color"],
                    facecolor=mcolors.to_rgba(item["color"], alpha=0.2),
                    linewidth=0.5
                )
            if error_style in ['bar', 'both']:
                ax.errorbar(
                    centers[valid],
                    mean_pdf[valid],
                    yerr=std_pdf[valid],
                    fmt='none',
                    ecolor=item["color"],
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.6
                )

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(f'$\Delta t = {tau_sec:.1f}\\mathrm{{s}}$ ($\Delta t = {tau}\\mathrm{{ frames}}$)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, frameon=True)

    # 余分なサブプロットを非表示
    for idx in range(n_tau, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

def plot_lambda_evolution(df_fits, component, signed, out_path, fit_powerlaw=True):
    """
    特性減衰長 lambda(Delta t) の時間発展プロットを作成・保存する。
    べき乗則 lambda(Delta t) = C * (Delta t)^alpha でフィッティングを行い、
    スケーリング指数 alpha を算出・描画する。
    """
    df_comp = df_fits[(df_fits['component'] == component) & (df_fits['signed'] == signed)]
    if df_comp.empty or 'fit_lambda_um' not in df_comp.columns:
        return []

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    powerlaw_records = []
    
    for item in BEADS_INFO:
        b_name = item["name"]
        df_b = df_comp[df_comp['bead_name'] == b_name].sort_values('lag_time_s')
        if df_b.empty:
            continue
            
        tau_s = df_b['lag_time_s'].to_numpy()
        lambda_val = df_b['fit_lambda_um'].to_numpy()
        lambda_err = df_b['fit_lambda_err'].to_numpy() if 'fit_lambda_err' in df_b.columns else np.zeros_like(lambda_val)
        
        valid = (tau_s > 0) & (lambda_val > 0) & np.isfinite(tau_s) & np.isfinite(lambda_val)
        x_val = tau_s[valid]
        y_val = lambda_val[valid]
        y_err = lambda_err[valid]
        
        label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$'
        
        # べき乗則フィッティング: ln lambda = ln C + alpha * ln(Delta t)
        if fit_powerlaw and len(x_val) >= 2:
            try:
                log_x = np.log(x_val)
                log_y = np.log(y_val)
                
                # 線形回帰 (対数空間)
                slope, intercept = np.polyfit(log_x, log_y, 1)
                alpha_fit = slope
                C_fit = np.exp(intercept)
                
                # 決定係数 R^2
                log_y_pred = intercept + slope * log_x
                ss_res = np.sum((log_y - log_y_pred) ** 2)
                ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # 誤差推定
                if len(x_val) > 2:
                    s_err = np.sqrt(ss_res / (len(x_val) - 2))
                    s_xx = np.sum((log_x - np.mean(log_x)) ** 2)
                    alpha_err = s_err / np.sqrt(s_xx) if s_xx > 0 else 0.0
                else:
                    alpha_err = 0.0

                label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($\\alpha={alpha_fit:.2f}$, $R^2={r2:.2f}$)'
                
                powerlaw_records.append({
                    'component': component,
                    'signed': signed,
                    'bead_name': b_name,
                    'diameter_um': item["diameter_um"],
                    'alpha': float(alpha_fit),
                    'alpha_err': float(alpha_err),
                    'C': float(C_fit),
                    'r_squared': float(r2),
                    'n_points': len(x_val)
                })
                
                # フィッティング曲線の描画（破線）
                x_fit_line = np.geomspace(np.min(x_val), np.max(x_val), 100)
                y_fit_line = C_fit * (x_fit_line ** alpha_fit)
                ax.plot(
                    x_fit_line,
                    y_fit_line,
                    linestyle='--',
                    color=item["color"],
                    alpha=0.85,
                    linewidth=1.5
                )
            except Exception as e:
                print(f"[WARNING] Powerlaw fit failed for {b_name}: {e}", flush=True)

        # エラーバー付きデータ点プロット
        ax.errorbar(
            x_val,
            y_val,
            yerr=y_err,
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            capsize=3,
            elinewidth=1.0,
            markersize=6,
            alpha=0.9,
            linestyle='none' if fit_powerlaw else '-'
        )

    # ガイドライン（拡散 ~ Delta t^0.5, 弾道 ~ Delta t^1.0）
    all_tau = df_comp['lag_time_s'].dropna().unique()
    if len(all_tau) >= 2:
        t_min, t_max = np.min(all_tau), np.max(all_tau)
        t_ref = np.logspace(np.log10(t_min), np.log10(t_max), 50)
        ref_val = float(np.nanmedian(df_comp['fit_lambda_um']))
        med_t = float(np.median(all_tau))
        ax.plot(t_ref, ref_val * (t_ref / med_t) ** 0.5, ':', color='gray', alpha=0.5, label=r'$\sim \Delta t^{0.5}$ (diffusive)')
        ax.plot(t_ref, ref_val * (t_ref / med_t) ** 1.0, '-.', color='gray', alpha=0.5, label=r'$\sim \Delta t^{1.0}$ (ballistic)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Lag time $\Delta t$ [$\mathrm{s}$]')
    ax.set_ylabel(r'$\lambda$ [$\mu\mathrm{m}$]')
    ax.legend(frameon=True, fontsize=8, loc='upper left')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)
    return powerlaw_records


def plot_scaled_pdf_master(all_tau_data, df_fits, frame_interval, component, signed, out_path,
                           plot_tau_list=None, xscale='linear', yscale='log', bins=50, xlim=(0, 10),
                           error_style='band'):
    """
    スケーリング変数 xi = Delta r / lambda(Delta t), y = lambda(Delta t) * P(Delta r) による
    全ビーズ・代表ラグタイムのデータコラップス（Master Scaling）プロットを作成・保存する。
    理論マスター曲線 f(xi) = exp(-xi) を破線でオーバーレイ。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    df_comp = df_fits[(df_fits['component'] == component) & (df_fits['signed'] == signed)]
    
    if plot_tau_list is None:
        plot_tau_list = sorted(list(all_tau_data.keys()))

    # 理論マスター曲線 f(xi) = exp(-xi)
    xi_max = xlim[1] if xlim is not None else 10.0
    xi_theory = np.linspace(0, xi_max, 200)
    ax.plot(xi_theory, np.exp(-xi_theory), 'k--', linewidth=2.0, alpha=0.8, label=r'Master curve $e^{-\xi}$', zorder=10)

    # 各ビーズ・ラグタイムのプロット
    for item in BEADS_INFO:
        b_name = item["name"]
        color = item["color"]

        for tau in plot_tau_list:
            if tau not in all_tau_data or b_name not in all_tau_data[tau]:
                continue
            exp_list = all_tau_data[tau][b_name]["per_exp"]
            if len(exp_list) == 0:
                continue
                
            # lambda の取得
            row = df_comp[(df_comp['bead_name'] == b_name) & (df_comp['tau_frame'] == tau)]
            if row.empty or 'fit_lambda_um' not in row.columns:
                continue
            lambda_val = float(row['fit_lambda_um'].iloc[0])
            if np.isnan(lambda_val) or lambda_val <= 0:
                continue

            centers, mean_pdf, std_pdf, _ = calc_ensemble_pdf(exp_list, bins=bins, bin_range=None, density=True)
            valid = (mean_pdf > 0) & (centers > 0)
            
            x_scaled = centers[valid] / lambda_val
            y_scaled = mean_pdf[valid] * lambda_val
            
            marker = item["marker"]
            alpha_val = 0.75 if tau == plot_tau_list[0] else 0.45
            label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$' if tau == plot_tau_list[0] else None

            ax.plot(
                x_scaled,
                y_scaled,
                marker=marker,
                color=color,
                label=label_text,
                markersize=4.5,
                alpha=alpha_val,
                linestyle='none'
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(r'Scaled displacement $\xi = \Delta r / \lambda(\Delta t)$')
    ax.set_ylabel(r'Scaled PDF $\lambda(\Delta t) P(\Delta r)$')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(bottom=1e-4, top=3.0)
    ax.legend(frameon=True, fontsize=8, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_scaled_pdf_per_bead(all_tau_data, df_fits, frame_interval, component, signed, out_path,
                             plot_tau_list=None, xscale='linear', yscale='log', bins=50, xlim=(0, 10),
                             error_style='band'):
    """
    ビーズ条件ごとにサブプロットを作成し、各ビーズにおける lag time Delta t による
    スケーリング（データコラップス）を可視化する（6パネルグリッド）。
    """
    n_beads = len(BEADS_INFO)
    n_cols = 3
    n_rows = (n_beads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False)
    df_comp = df_fits[(df_fits['component'] == component) & (df_fits['signed'] == signed)]
    
    if plot_tau_list is None:
        plot_tau_list = sorted(list(all_tau_data.keys()))

    tau_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_tau_list)))

    for idx, item in enumerate(BEADS_INFO):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        b_name = item["name"]

        # 理論マスター曲線 f(xi) = exp(-xi)
        xi_max = xlim[1] if xlim is not None else 10.0
        xi_theory = np.linspace(0, xi_max, 200)
        ax.plot(xi_theory, np.exp(-xi_theory), 'k--', linewidth=1.8, alpha=0.7, label=r'$e^{-\xi}$', zorder=10)

        for t_idx, tau in enumerate(plot_tau_list):
            if tau not in all_tau_data or b_name not in all_tau_data[tau]:
                continue
            exp_list = all_tau_data[tau][b_name]["per_exp"]
            if len(exp_list) == 0:
                continue

            row = df_comp[(df_comp['bead_name'] == b_name) & (df_comp['tau_frame'] == tau)]
            if row.empty or 'fit_lambda_um' not in row.columns:
                continue
            lambda_val = float(row['fit_lambda_um'].iloc[0])
            if np.isnan(lambda_val) or lambda_val <= 0:
                continue

            centers, mean_pdf, std_pdf, _ = calc_ensemble_pdf(exp_list, bins=bins, bin_range=None, density=True)
            valid = (mean_pdf > 0) & (centers > 0)

            x_scaled = centers[valid] / lambda_val
            y_scaled = mean_pdf[valid] * lambda_val
            std_scaled = std_pdf[valid] * lambda_val
            tau_sec = tau * frame_interval

            t_color = tau_colors[t_idx]
            label_text = f'$\Delta t = {tau_sec:.0f}\\mathrm{{s}}$'

            ax.plot(
                x_scaled,
                y_scaled,
                marker='o',
                color=t_color,
                label=label_text,
                markersize=4,
                alpha=0.85,
                linestyle='none'
            )
            if error_style in ['band', 'both']:
                ax.fill_between(
                    x_scaled,
                    np.clip(y_scaled - std_scaled, 1e-6, None),
                    y_scaled + std_scaled,
                    facecolor=mcolors.to_rgba(t_color, alpha=0.15),
                    edgecolor=t_color,
                    linewidth=0.5
                )

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(r'$\xi = \Delta r / \lambda(\Delta t)$')
        ax.set_ylabel(r'$\lambda(\Delta t) P(\Delta r)$')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_ylim(bottom=1e-4, top=3.0)
        ax.set_title(f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ({item["name"]})')
        ax.legend(fontsize=7, frameon=True, loc='upper right')
        ax.grid(True, which="both", ls="--", alpha=0.3)

    for idx in range(n_beads, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Cargo particle displacement absolute value and PDF analysis.")
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions (e.g. /Volumes/data/Sasaki/MTsingleBeads).")
    parser.add_argument('--beads', type=str, nargs='+', default=['all'],
                        help="Beads conditions to analyze (e.g. beads06um beads1um ... or 'all').")
    parser.add_argument('--component', type=str, default='norm',
                        choices=['norm', 'x', 'y', 'both_xy', 'parallel', 'perpendicular', 'all'],
                        help="Displacement component: 'norm' (2D magnitude |Δr|), 'x', 'y', 'both_xy', 'parallel' (|Δr_par|), 'perpendicular' (|Δr_perp|), or 'all'.")
    DEFAULT_FINE_TAU = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50]
    DEFAULT_PLOT_TAU = [1, 2, 5, 10, 25]

    parser.add_argument('--tau', type=int, nargs='+', default=DEFAULT_FINE_TAU,
                        help="Lag time list in frames for lambda(Δt) time evolution (default: 1..50 fine list).")
    parser.add_argument('--tau_seconds', type=float, nargs='+', default=None,
                        help="Lag time in seconds (overrides --tau if provided).")
    parser.add_argument('--plot_tau', type=int, nargs='+', default=DEFAULT_PLOT_TAU,
                        help="Representative lag times in frames to plot PDF histograms for (default: 1 2 5 10 25, 5 series).")
    parser.add_argument('--plot_tau_seconds', type=float, nargs='+', default=None,
                        help="Representative lag times in seconds to plot PDF histograms for (overrides --plot_tau).")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale (um/pixel, default: 0.11).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames in seconds (default: 4.0).")
    parser.add_argument('--signed', action='store_true',
                        help="Calculate signed displacement instead of absolute magnitude (for x, y, par, perp).")
    parser.add_argument('--bins', type=int, default=50,
                        help="Number of bins for PDF histogram (default: 50).")
    parser.add_argument('--xscale', type=str, default='linear', choices=['log', 'linear'],
                        help="X-axis scale for PDF plot (default: linear).")
    parser.add_argument('--yscale', type=str, default='log', choices=['log', 'linear'],
                        help="Y-axis scale for PDF plot (default: log).")
    parser.add_argument('--fit_exp', action='store_true', default=True,
                        help="Fit displacement PDF with exponential distribution P(r) = A * exp(-r/lambda) (default: True).")
    parser.add_argument('--no_fit_exp', dest='fit_exp', action='store_false',
                        help="Disable exponential fitting.")
    parser.add_argument('--fit_mode', type=str, default='log', choices=['log', 'linear'],
                        help="Fitting space: 'log' (fits ln P(r) = ln A - r/lambda, default), or 'linear'.")
    parser.add_argument('--fit_rmin', type=float, default=None,
                        help="Minimum r value for fitting exponential tail (default: None, all r >= 0).")
    parser.add_argument('--fit_rmax', type=float, default=None,
                        help="Maximum r value for fitting exponential tail (default: None).")
    parser.add_argument('--error_style', type=str, default='band', choices=['band', 'bar', 'both', 'none'],
                        help="Error representation across experiments: 'band' (shaded fill_between like MSD.py, default), 'bar' (error bars), 'both', or 'none'.")
    parser.add_argument('--xlim', type=float, nargs=2, default=[0.0, 50.0],
                        help="X-axis limits [min, max] (default: 0 50).")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Output directory to save plots and CSVs. Defaults to root_dir/figure.")
    args = parser.parse_args()

    # ルートディレクトリの確定
    if args.root_dir is not None:
        root_dir = Path(args.root_dir)
    else:
        root_dir = find_default_root()

    print(f"=== Cargo Particle Displacement Analysis ===", flush=True)
    print(f"Root Directory: {root_dir}", flush=True)

    # 出力先ディレクトリ
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = root_dir / 'figure'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 対象ビーズ条件の抽出
    if 'all' in args.beads:
        selected_beads = [b['name'] for b in BEADS_INFO]
    else:
        selected_beads = []
        for item in args.beads:
            for b in item.split(','):
                b = b.strip()
                if b == 'all':
                    selected_beads = [info['name'] for info in BEADS_INFO]
                    break
                if any(info['name'] == b for info in BEADS_INFO):
                    selected_beads.append(b)
                else:
                    print(f"[WARNING] Unknown bead condition: {b}", flush=True)

    # ラグタイムの決定 (評価用 & プロット用)
    if args.tau_seconds is not None:
        tau_list = sorted(list(set([max(1, int(round(ts / args.frame_interval))) for ts in args.tau_seconds])))
    else:
        tau_list = sorted(list(set(args.tau)))

    if args.plot_tau_seconds is not None:
        plot_tau_list = sorted(list(set([max(1, int(round(ts / args.frame_interval))) for ts in args.plot_tau_seconds])))
    else:
        plot_tau_list = sorted(list(set(args.plot_tau)))

    # plot_tau_list が tau_list に含まれるように統合
    for pt in plot_tau_list:
        if pt not in tau_list:
            tau_list.append(pt)
    tau_list = sorted(tau_list)

    print(f"Evaluation lag times (frames): {tau_list}", flush=True)
    print(f"Histogram plotting lag times (frames): {plot_tau_list}", flush=True)

    components_to_run = ['norm', 'parallel', 'perpendicular'] if args.component == 'all' else [args.component]

    stats_records = []
    all_comp_tau_data = {}
    xlim_tuple = tuple(args.xlim) if args.xlim is not None else None

    for comp in components_to_run:
        print(f"\n---> Processing Component: '{comp}' (signed={args.signed})", flush=True)
        plot_tau_data = {}

        for tau in tau_list:
            tau_sec = tau * args.frame_interval
            beads_data = {}

            for bead_name in selected_beads:
                exp_dirs = find_experiment_dirs(root_dir, bead_name)
                if not exp_dirs:
                    continue

                disp_dict = collect_displacements(exp_dirs, tau=tau, scale=args.scale, component=comp, signed=args.signed)
                beads_data[bead_name] = disp_dict

                pooled = disp_dict["pooled"]
                exp_list = disp_dict["per_exp"]
                if len(pooled) > 0:
                    exp_means = [float(np.mean(e)) for e in exp_list if len(e) > 0]
                    stats_records.append({
                        'component': comp,
                        'signed': args.signed,
                        'bead_name': bead_name,
                        'tau_frame': tau,
                        'lag_time_s': tau_sec,
                        'n_experiments': len(exp_list),
                        'count': len(pooled),
                        'mean': float(np.mean(pooled)),
                        'std': float(np.std(pooled)),
                        'median': float(np.median(pooled)),
                        'exp_mean_avg': float(np.mean(exp_means)) if exp_means else np.nan,
                        'exp_mean_std': float(np.std(exp_means, ddof=1)) if len(exp_means) > 1 else 0.0,
                        'p25': float(np.percentile(pooled, 25)),
                        'p75': float(np.percentile(pooled, 75)),
                        'msd': float(np.mean(pooled**2)),
                    })

            # 指数関数フィッティング計算
            fit_results = {}
            if args.fit_exp:
                for bead_name in selected_beads:
                    if bead_name in beads_data and len(beads_data[bead_name]["per_exp"]) > 0:
                        exp_list = beads_data[bead_name]["per_exp"]
                        bin_range = xlim_tuple if xlim_tuple is not None else None
                        centers, mean_pdf, std_pdf, _ = calc_ensemble_pdf(exp_list, bins=args.bins, bin_range=bin_range, density=True)
                        valid = (mean_pdf > 0) & (centers > 0) if (args.xscale == 'log' or args.yscale == 'log') else (mean_pdf >= 0)
                        if np.sum(valid) >= 3:
                            fit_res = dpm.fit_exponential_pdf(
                                centers[valid],
                                mean_pdf[valid],
                                pdf_std=std_pdf[valid] if np.all(std_pdf[valid] > 0) else None,
                                r_min=args.fit_rmin,
                                r_max=args.fit_rmax,
                                fit_mode=args.fit_mode
                            )
                            if fit_res is not None:
                                fit_results[bead_name] = fit_res

            # フィッティング結果を stats_records に紐付け
            if fit_results:
                for rec in stats_records:
                    if rec['component'] == comp and rec['tau_frame'] == tau and rec['bead_name'] in fit_results:
                        f_info = fit_results[rec['bead_name']]
                        rec['fit_lambda_um'] = f_info['lambda']
                        rec['fit_lambda_err'] = f_info['lambda_err']
                        rec['fit_A'] = f_info['A']
                        rec['fit_r2'] = f_info['r_squared']

            # 指定された代表ラグタイムのみヒストグラムPDFプロットを保存
            if tau in plot_tau_list:
                print(f"  Plotting PDF for tau={tau} frames ({tau_sec:.1f} s)...", flush=True)
                plot_tau_data[tau] = beads_data
                pdf_save_path = out_dir / f"displacement_PDF_{comp}_tau{tau_sec:.0f}s.svg"
                plot_pdf_across_beads(
                    beads_data,
                    tau=tau,
                    frame_interval=args.frame_interval,
                    component=comp,
                    signed=args.signed,
                    out_path=pdf_save_path,
                    xscale=args.xscale,
                    yscale=args.yscale,
                    bins=args.bins,
                    error_style=args.error_style,
                    xlim=xlim_tuple,
                    fit_exp=args.fit_exp,
                    fit_mode=args.fit_mode,
                    fit_rmin=args.fit_rmin,
                    fit_rmax=args.fit_rmax
                )
                plot_pdf_across_beads(
                    beads_data,
                    tau=tau,
                    frame_interval=args.frame_interval,
                    component=comp,
                    signed=args.signed,
                    out_path=pdf_save_path.with_suffix('.png'),
                    xscale=args.xscale,
                    yscale=args.yscale,
                    bins=args.bins,
                    error_style=args.error_style,
                    xlim=xlim_tuple,
                    fit_exp=args.fit_exp,
                    fit_mode=args.fit_mode,
                    fit_rmin=args.fit_rmin,
                    fit_rmax=args.fit_rmax
                )

        all_comp_tau_data[comp] = plot_tau_data

        # 代表ラグタイムのグリッドプロット保存
        if plot_tau_data:
            grid_save_path = out_dir / f"displacement_PDF_grid_{comp}.svg"
            plot_multitau_grid(
                plot_tau_data,
                frame_interval=args.frame_interval,
                component=comp,
                signed=args.signed,
                out_path=grid_save_path,
                xscale=args.xscale,
                yscale=args.yscale,
                bins=args.bins,
                error_style=args.error_style,
                xlim=xlim_tuple,
                fit_exp=args.fit_exp,
                fit_mode=args.fit_mode,
                fit_rmin=args.fit_rmin,
                fit_rmax=args.fit_rmax
            )
            plot_multitau_grid(
                plot_tau_data,
                frame_interval=args.frame_interval,
                component=comp,
                signed=args.signed,
                out_path=grid_save_path.with_suffix('.png'),
                xscale=args.xscale,
                yscale=args.yscale,
                bins=args.bins,
                error_style=args.error_style,
                xlim=xlim_tuple,
                fit_exp=args.fit_exp,
                fit_mode=args.fit_mode,
                fit_rmin=args.fit_rmin,
                fit_rmax=args.fit_rmax
            )

    # 統計サマリーの CSV 保存
    if stats_records:
        df_stats = pd.DataFrame(stats_records)
        csv_save_path = out_dir / "displacement_statistics_summary.csv"
        safe_save_csv(df_stats, csv_save_path)
        print(f"\n[SAVED] Statistics summary saved to {csv_save_path}", flush=True)

        if args.fit_exp and 'fit_lambda_um' in df_stats.columns:
            fit_cols = ['component', 'signed', 'bead_name', 'tau_frame', 'lag_time_s', 
                        'fit_lambda_um', 'fit_lambda_err', 'fit_A', 'fit_r2', 'count', 'mean', 'std', 'msd']
            present_cols = [c for c in fit_cols if c in df_stats.columns]
            df_fits = df_stats[present_cols].dropna(subset=['fit_lambda_um'])
            fits_csv_path = out_dir / "displacement_exponential_fits.csv"
            safe_save_csv(df_fits, fits_csv_path)
            print(f"[SAVED] Exponential fits summary saved to {fits_csv_path}", flush=True)

            # lambda(Delta t) の時間発展プロット保存 & べき乗フィッティング
            all_powerlaw_records = []
            for comp in components_to_run:
                lambda_plot_path = out_dir / f"displacement_lambda_evolution_{comp}.svg"
                pw_recs = plot_lambda_evolution(df_fits, component=comp, signed=args.signed, out_path=lambda_plot_path, fit_powerlaw=True)
                plot_lambda_evolution(df_fits, component=comp, signed=args.signed, out_path=lambda_plot_path.with_suffix('.png'), fit_powerlaw=True)
                if pw_recs:
                    all_powerlaw_records.extend(pw_recs)

            if all_powerlaw_records:
                df_pw = pd.DataFrame(all_powerlaw_records)
                pw_csv_path = out_dir / "displacement_lambda_powerlaw_fits.csv"
                safe_save_csv(df_pw, pw_csv_path)
                print(f"[SAVED] Lambda power-law fits summary saved to {pw_csv_path}", flush=True)

            # スケーリングプロット (Data collapse: xi = Delta r / lambda vs lambda * P(Delta r))
            for comp in components_to_run:
                if comp in all_comp_tau_data:
                    tau_data = all_comp_tau_data[comp]
                    scaled_master_path = out_dir / f"displacement_PDF_scaled_master_{comp}.svg"
                    plot_scaled_pdf_master(
                        tau_data, df_fits, frame_interval=args.frame_interval,
                        component=comp, signed=args.signed, out_path=scaled_master_path,
                        plot_tau_list=plot_tau_list, error_style=args.error_style
                    )
                    plot_scaled_pdf_master(
                        tau_data, df_fits, frame_interval=args.frame_interval,
                        component=comp, signed=args.signed, out_path=scaled_master_path.with_suffix('.png'),
                        plot_tau_list=plot_tau_list, error_style=args.error_style
                    )

                    scaled_per_bead_path = out_dir / f"displacement_PDF_scaled_per_bead_{comp}.svg"
                    plot_scaled_pdf_per_bead(
                        tau_data, df_fits, frame_interval=args.frame_interval,
                        component=comp, signed=args.signed, out_path=scaled_per_bead_path,
                        plot_tau_list=plot_tau_list, error_style=args.error_style
                    )
                    plot_scaled_pdf_per_bead(
                        tau_data, df_fits, frame_interval=args.frame_interval,
                        component=comp, signed=args.signed, out_path=scaled_per_bead_path.with_suffix('.png'),
                        plot_tau_list=plot_tau_list, error_style=args.error_style
                    )

    print(f"\n[DONE] Displacement analysis finished successfully!", flush=True)


if __name__ == '__main__':
    main()
