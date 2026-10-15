#!/usr/bin/env python3
"""
plot_cargo_dimensionless_velocity_distribution.py

微小管平均速度 v_MT で無次元化した貨物スカラー速度:
    \\tilde{v}_{\\mathrm{mag}} = |v| / v_{\\mathrm{MT}}
およびその対数:
    \\log_{10}(\\tilde{v}) = \\log_{10}(|v| / v_{\\mathrm{MT}})
    \\ln(\\tilde{v}) = \\ln(|v| / v_{\\mathrm{MT}})
の速度分布 P(\\tilde{v}), P(\\log_{10}\\tilde{v}), P(\\ln\\tilde{v})、
および対数正規 Q-Q プロット (Lognormal Q-Q Plot) を貨物サイズ（ビーズ径）ごとに算出し、
グリッドプロット（各サイズ別）、重ね合わせプロット（全サイズ比較）、
および統計サマリーCSVを出力するスクリプトです。

出力ファイル（既定: figure/dimensionless_velocity_distribution / <root>/figure/...）:
1. cargo_dimensionless_velocity_grid_linear.png / .svg
2. cargo_dimensionless_velocity_grid_semilog.png / .svg
3. cargo_dimensionless_velocity_overlay_linear.png / .svg
4. cargo_dimensionless_velocity_overlay_semilog.png / .svg
5. cargo_dimensionless_velocity_fit_params_vs_diameter.png / .svg
6. cargo_dimensionless_velocity_summary.csv
7. cargo_dimensionless_log10_velocity_grid.png / .svg (対数速度 log10(v_tilde) Grid)
8. cargo_dimensionless_log10_velocity_overlay.png / .svg (対数速度 log10(v_tilde) Overlay)
9. cargo_dimensionless_ln_velocity_grid.png / .svg (自然対数 ln(v_tilde) Grid)
10. cargo_dimensionless_ln_velocity_overlay.png / .svg (自然対数 ln(v_tilde) Overlay)
11. cargo_dimensionless_log_fit_params_vs_diameter.png / .svg (対数統計量 vs 粒子径)
12. cargo_dimensionless_log_velocity_summary.csv (対数速度の統計サマリーCSV)
13. cargo_dimensionless_lognormal_qq_grid.png / .svg (対数正規 Q-Q プロット Grid)
14. cargo_dimensionless_lognormal_qq_overlay.png / .svg (標準化 対数正規 Q-Q プロット Overlay)
15. cargo_dimensionless_qq_summary.csv (Q-Q プロット統計サマリーCSV)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, probplot, skew, kurtosis, kstest

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        style_colors = [f"#{c}" if not c.startswith('#') else c for c in style_colors]
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']

# ビーズ基本情報 (MSD.py / displacement_analysis.py と統一)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "label": "0.63 μm", "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "label": "1.18 μm", "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "label": "3.37 μm", "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "label": "5.00 μm", "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "label": "7.24 μm", "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "label": "20.0 μm", "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists():
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def apply_custom_style():
    style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
    if style_path.exists():
        try:
            plt.style.use(str(style_path))
        except Exception:
            pass
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'cm'


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300):
    """指定されたすべての出力ディレクトリに png と svg を保存する"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / f"{basename}.png"
        svg_path = d / f"{basename}.svg"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {basename}.png / .svg -> {len(out_dirs)} dir(s)")


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]):
    """指定されたすべての出力ディレクトリに CSV を保存する"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / f"{basename}.csv"
        df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {basename}.csv -> {len(out_dirs)} dir(s)")


def double_exponential_pdf(v: np.ndarray, alpha: float, vs: float, vf: float) -> np.ndarray:
    """
    正規化された2成分指数確率密度関数 (PDF):
    P(\\tilde{v}) = \\frac{1-\\alpha}{\\tilde{v}_s} e^{-\\tilde{v}/\\tilde{v}_s} + \\frac{\\alpha}{\\tilde{v}_f} e^{-\\tilde{v}/\\tilde{v}_f}
    """
    return (1.0 - alpha) / vs * np.exp(-v / vs) + alpha / vf * np.exp(-v / vf)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """正規分布 (ガウシアン) PDF"""
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_mean_mt_velocity(exp_dir: Path) -> float:
    """実験ディレクトリから velocities_mean.csv を読み込んで平均微小管速度を取得"""
    v_mt_path = exp_dir / "velocities_mean.csv"
    if not v_mt_path.exists():
        return float('nan')
    try:
        df_v_mt = pd.read_csv(v_mt_path)
        if 'mean_velocity' in df_v_mt.columns:
            s = pd.to_numeric(df_v_mt['mean_velocity'], errors='coerce').dropna()
            if len(s) > 0:
                return float(s.iloc[0])
        num_cols = df_v_mt.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            s = df_v_mt[num_cols[0]].dropna()
            if len(s) > 0:
                return float(s.iloc[0])
    except Exception:
        pass
    return float('nan')


def extract_from_points_csv(points_csv: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """既存の cargo_spin_velocity_points.csv から無次元速度 \\tilde{v}_{\\mathrm{mag}} を抽出"""
    df = pd.read_csv(points_csv)
    bead_velocities = {}
    bead_dfs = {}
    all_pooled = []

    for b in BEADS_INFO:
        b_name = b["name"]
        df_b = df[df["bead_name"] == b_name].copy()
        if len(df_b) == 0:
            continue

        if "v_mag_tilde" in df_b.columns and df_b["v_mag_tilde"].notna().any():
            vals = df_b["v_mag_tilde"].dropna().values
        elif "v_mag_um_s" in df_b.columns and "v_mt_um_s" in df_b.columns:
            vals = (df_b["v_mag_um_s"] / df_b["v_mt_um_s"]).dropna().values
        elif "v_track_tilde" in df_b.columns:
            vals = np.abs(df_b["v_track_tilde"].dropna().values)
        elif "v_tilde" in df_b.columns:
            vals = np.abs(df_b["v_tilde"].dropna().values)
        else:
            continue

        vals = vals[np.isfinite(vals) & (vals > 0)]
        bead_velocities[b_name] = vals
        bead_dfs[b_name] = df_b
        all_pooled.extend(vals)
        print(f"Loaded from points CSV [{b_name}]: N = {len(vals):,}, mean = {np.mean(vals):.4f}, median = {np.median(vals):.4f}")

    bead_velocities["overall_pooled"] = np.array(all_pooled)
    print(f"Total pooled dataset: N = {len(all_pooled):,}, mean = {np.mean(all_pooled):.4f}, median = {np.median(all_pooled):.4f}")
    return bead_velocities, bead_dfs


def extract_from_raw_tracks(
    root_dir: Path,
    scale: float = 0.11,
    frame_interval: float = 4.0
) -> Dict[str, np.ndarray]:
    """生軌跡データと各実験の velocities_mean.csv から無次元速度を計算して抽出"""
    bead_velocities = {}
    all_pooled = []

    for b in BEADS_INFO:
        b_name = b["name"]
        b_dir = root_dir / b_name
        track_files = sorted(b_dir.glob("*/*/beads_tracks.csv"))
        if not track_files:
            track_files = sorted(b_dir.glob("*/*/*beads_tracks.csv")) + sorted(b_dir.glob("*beads_tracks.csv"))
        track_files = sorted(list(set(track_files)))

        speeds_list = []
        for f in track_files:
            exp_dir = f.parent
            v_mt = load_mean_mt_velocity(exp_dir)
            if not np.isfinite(v_mt) or v_mt <= 0:
                continue

            try:
                df = pd.read_csv(f)
            except Exception:
                continue

            if not {'particle', 'x', 'y', 'frame'}.issubset(df.columns):
                continue

            for _, g in df.groupby('particle'):
                if len(g) < 2:
                    continue
                g = g.sort_values('frame')
                dx = np.diff(g['x'].values) * scale
                dy = np.diff(g['y'].values) * scale
                dt = np.diff(g['frame'].values) * frame_interval
                valid = dt > 0
                if np.sum(valid) == 0:
                    continue
                v_raw = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
                v_tilde = v_raw / v_mt
                v_tilde = v_tilde[np.isfinite(v_tilde) & (v_tilde > 0)]
                speeds_list.extend(v_tilde)

        arr = np.array(speeds_list)
        bead_velocities[b_name] = arr
        all_pooled.extend(speeds_list)
        print(f"Extracted from raw tracks [{b_name}]: N = {len(speeds_list):,}, mean = {np.mean(speeds_list):.4f}" if len(speeds_list) > 0 else f"No data for {b_name}")

    bead_velocities["overall_pooled"] = np.array(all_pooled)
    print(f"Total pooled dataset: N = {len(all_pooled):,}, mean = {np.mean(all_pooled):.4f}")
    return bead_velocities


def fit_dimensionless_velocity_distribution(
    v_tilde: np.ndarray,
    n_bins: int = 40,
    max_v: Optional[float] = None
) -> dict:
    """無次元速度分布のヒストグラム計算および2成分指数分布のフィッティング"""
    if len(v_tilde) < 10:
        return {"success": False}

    if max_v is None:
        max_v = float(np.percentile(v_tilde, 99.5) * 1.1)

    valid_v = v_tilde[(v_tilde >= 0) & (v_tilde <= max_v)]
    if len(valid_v) < 10:
        valid_v = v_tilde[v_tilde >= 0]

    counts, bin_edges = np.histogram(valid_v, bins=n_bins, range=(0, max_v), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mask = counts > 0
    x_fit = bin_centers[mask]
    y_fit = counts[mask]

    mean_v = float(np.mean(v_tilde))
    median_v = float(np.median(v_tilde))
    std_v = float(np.std(v_tilde))
    q25_v = float(np.percentile(v_tilde, 25))
    q75_v = float(np.percentile(v_tilde, 75))
    q90_v = float(np.percentile(v_tilde, 90))
    q95_v = float(np.percentile(v_tilde, 95))
    q99_v = float(np.percentile(v_tilde, 99))

    best_popt = None
    best_perr = None
    best_r2 = -float("inf")

    if len(x_fit) >= 4:
        init_guesses = [
            [0.3, mean_v * 0.4, mean_v * 1.8],
            [0.5, mean_v * 0.3, mean_v * 2.0],
            [0.2, mean_v * 0.5, mean_v * 1.5],
            [0.7, mean_v * 0.2, mean_v * 3.0],
        ]
        bounds = ([0.0, 1e-4, 1e-4], [1.0, max_v * 2.0, max_v * 5.0])

        for p0 in init_guesses:
            try:
                popt, pcov = curve_fit(
                    double_exponential_pdf, x_fit, y_fit,
                    p0=p0, bounds=bounds, maxfev=20000
                )
                if popt[1] > popt[2]:
                    alpha_new = 1.0 - popt[0]
                    vs_new = popt[2]
                    vf_new = popt[1]
                    popt = np.array([alpha_new, vs_new, vf_new])

                y_pred = double_exponential_pdf(x_fit, *popt)
                ss_res = np.sum((y_fit - y_pred) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                if r2 > best_r2:
                    best_r2 = r2
                    best_popt = popt
                    perr = np.sqrt(np.diag(pcov))
                    best_perr = perr
            except Exception:
                continue

    return {
        "success": True,
        "n_points": len(v_tilde),
        "mean_v": mean_v,
        "median_v": median_v,
        "std_v": std_v,
        "q25_v": q25_v,
        "q75_v": q75_v,
        "q90_v": q90_v,
        "q95_v": q95_v,
        "q99_v": q99_v,
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "counts": counts,
        "max_v": max_v,
        "popt_pdf": best_popt,
        "perr_pdf": best_perr,
        "r2_pdf": best_r2 if best_popt is not None else np.nan,
        "raw_velocities": v_tilde,
    }


def fit_log_velocity_distribution(
    v_tilde: np.ndarray,
    log_base: str = "log10",
    n_bins: int = 35,
    range_lim: Optional[Tuple[float, float]] = None
) -> dict:
    """対数無次元速度 log(v_tilde) のヒストグラム計算およびガウスフィッティング"""
    v_pos = v_tilde[np.isfinite(v_tilde) & (v_tilde > 0)]
    if len(v_pos) < 10:
        return {"success": False}

    if log_base == "log10":
        log_v = np.log10(v_pos)
        default_range = (-3.0, 0.8)
    else:  # ln
        log_v = np.log(v_pos)
        default_range = (-6.9, 1.8)

    hist_range = range_lim if range_lim is not None else default_range
    log_v_valid = log_v[(log_v >= hist_range[0]) & (log_v <= hist_range[1])]
    if len(log_v_valid) < 10:
        log_v_valid = log_v

    counts, bin_edges = np.histogram(log_v_valid, bins=n_bins, range=hist_range, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_log_v = float(np.mean(log_v))
    median_log_v = float(np.median(log_v))
    std_log_v = float(np.std(log_v))
    skew_log_v = float(skew(log_v))
    kurt_log_v = float(kurtosis(log_v))
    q25_log_v = float(np.percentile(log_v, 25))
    q75_log_v = float(np.percentile(log_v, 75))

    # ガウスフィッティング
    mask = counts > 0
    x_fit = bin_centers[mask]
    y_fit = counts[mask]
    gauss_mu = mean_log_v
    gauss_sigma = std_log_v
    gauss_r2 = np.nan

    if len(x_fit) >= 3:
        try:
            popt, pcov = curve_fit(
                gaussian_pdf, x_fit, y_fit,
                p0=[mean_log_v, std_log_v],
                bounds=([hist_range[0], 0.01], [hist_range[1], (hist_range[1] - hist_range[0])])
            )
            gauss_mu, gauss_sigma = popt[0], popt[1]
            y_pred = gaussian_pdf(x_fit, gauss_mu, gauss_sigma)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            gauss_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            pass

    return {
        "success": True,
        "log_base": log_base,
        "n_points": len(log_v),
        "mean_log_v": mean_log_v,
        "median_log_v": median_log_v,
        "std_log_v": std_log_v,
        "skew_log_v": skew_log_v,
        "kurt_log_v": kurt_log_v,
        "q25_log_v": q25_log_v,
        "q75_log_v": q75_log_v,
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "counts": counts,
        "hist_range": hist_range,
        "gauss_mu": gauss_mu,
        "gauss_sigma": gauss_sigma,
        "gauss_r2": gauss_r2,
        "raw_log_v": log_v,
    }


def compute_lognormal_qq(v_tilde: np.ndarray, log_base: str = "log10") -> dict:
    """対数正規 Q-Q プロット用データの算出（probplot による理論分位点とサンプル分位点）"""
    v_pos = v_tilde[np.isfinite(v_tilde) & (v_tilde > 0)]
    if len(v_pos) < 10:
        return {"success": False}

    log_v = np.log10(v_pos) if log_base == "log10" else np.log(v_pos)
    mean_val = float(np.mean(log_v))
    std_val = float(np.std(log_v))

    # probplot で理論分位点 (osm) と サンプル分位点 (osr) を算出
    (osm, osr), (slope, intercept, r) = probplot(log_v, dist="norm", fit=True)
    r2 = r ** 2

    # 標準化されたサンプル値 (Z-score)
    z_osr = (osr - mean_val) / std_val if std_val > 0 else osr

    # KS 検定 (正規分布との適合度)
    ks_stat, ks_p = kstest(log_v, 'norm', args=(mean_val, std_val))

    return {
        "success": True,
        "log_base": log_base,
        "n_points": len(log_v),
        "theoretical_quantiles": osm,
        "sample_quantiles": osr,
        "standardized_quantiles": z_osr,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mean": mean_val,
        "std": std_val,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
    }


def plot_dimensionless_velocity_grid(
    fit_results: Dict[str, dict],
    out_dirs: List[Path],
    semilog: bool = False
):
    """各粒子径および全体プールの無次元速度分布グリッドプロット"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    scale_str = "semilog" if semilog else "linear"

    for idx, b in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        b_name = b["name"]
        label = b["label"]
        color = b["color"]
        res = fit_results.get(b_name, {})

        if not res.get("success", False):
            ax.set_title(f"$d = {label}$ (No data)", fontsize=11)
            continue

        bin_centers = res["bin_centers"]
        counts = res["counts"]
        bin_edges = res["bin_edges"]
        max_v = res["max_v"]
        popt = res["popt_pdf"]
        v_dense = np.linspace(0, max_v, 500)

        ax.bar(
            bin_centers, counts, width=np.diff(bin_edges),
            align='center', alpha=0.45, color=color, edgecolor=color,
            label=f"$d = {label}$\n($N = {res['n_points']:,}$)"
        )

        if popt is not None and res.get("r2_pdf", -1) > 0.5:
            alpha, vs, vf = popt
            p_total = double_exponential_pdf(v_dense, alpha, vs, vf)
            p_slow = (1.0 - alpha) / vs * np.exp(-v_dense / vs)
            p_fast = alpha / vf * np.exp(-v_dense / vf)

            ax.plot(v_dense, p_total, color='black', lw=2.0, label=r'Fit $P(\tilde{v})$')
            ax.plot(v_dense, p_slow, color='#d62728', ls='--', lw=1.5, label=r'Slow: $\tilde{v}_s$')
            ax.plot(v_dense, p_fast, color='#1f77b4', ls=':', lw=1.7, label=r'Fast: $\tilde{v}_f$')

            text_str = (
                f"$\\tilde{{v}}_s = {vs:.3f}$\n"
                f"$\\tilde{{v}}_f = {vf:.3f}$\n"
                f"$\\alpha = {alpha:.3f}$\n"
                f"$R^2 = {res['r2_pdf']:.3f}$"
            )
            ax.text(
                0.95, 0.95, text_str, transform=ax.transAxes,
                va='top', ha='right', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85)
            )

        ax.set_title(f"$d = {label}$ (Diameter)", fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Dimensionless Velocity $\tilde{v} = |v| / v_{\mathrm{MT}}$", fontsize=10.5)
        ax.set_ylabel(r"Probability Density $P(\tilde{v})$", fontsize=10.5)
        ax.set_xlim(0, max_v)

        if semilog:
            ax.set_yscale('log')
            y_min = max(1e-3, np.min(counts[counts > 0]) * 0.5) if np.any(counts > 0) else 1e-3
            y_max = np.max(counts) * 3.0 if np.any(counts > 0) else 10.0
            ax.set_ylim(y_min, y_max)
            ax.legend(loc='lower left', fontsize=8.5, framealpha=0.8)
        else:
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=8.5, framealpha=0.8)

        ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.5)

    plt.suptitle(
        f"Dimensionless Cargo Velocity Distributions by Bead Size ({scale_str.capitalize()})\n"
        r"$\tilde{v} = |v| / v_{\mathrm{MT}}, \quad P(\tilde{v}) = (1-\alpha)\frac{1}{\tilde{v}_s}e^{-\tilde{v}/\tilde{v}_s} + \alpha\frac{1}{\tilde{v}_f}e^{-\tilde{v}/\tilde{v}_f}$",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, f"cargo_dimensionless_velocity_grid_{scale_str}", out_dirs)


def plot_dimensionless_velocity_overlay(
    fit_results: Dict[str, dict],
    out_dirs: List[Path],
    semilog: bool = False
):
    """全粒子径の無次元速度分布の重ね合わせプロット"""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scale_str = "semilog" if semilog else "linear"

    v_dense = np.linspace(0, 3.0, 500)

    for b in BEADS_INFO:
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        marker = b["marker"]
        res = fit_results.get(b_name, {})

        if not res.get("success", False):
            continue

        bin_centers = res["bin_centers"]
        counts = res["counts"]
        popt = res["popt_pdf"]

        ax.plot(
            bin_centers, counts, marker=marker, ms=5.5, ls='none',
            color=color, alpha=0.8, label=f"$d = {label}$ ($N={res['n_points']:,}$)"
        )

        if popt is not None and res.get("r2_pdf", -1) > 0.5:
            alpha, vs, vf = popt
            p_total = double_exponential_pdf(v_dense, alpha, vs, vf)
            ax.plot(v_dense, p_total, color=color, lw=1.8, alpha=0.85)

    ax.set_xlabel(r"Dimensionless Velocity $\tilde{v} = |v| / v_{\mathrm{MT}}$", fontsize=12.5)
    ax.set_ylabel(r"Probability Density $P(\tilde{v})$", fontsize=12.5)
    ax.set_xlim(0, 2.5)

    if semilog:
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 30.0)
        title_str = "Dimensionless Cargo Velocity Distributions (Semilog-y) & Fits"
    else:
        ax.set_ylim(0, 10.0)
        title_str = "Dimensionless Cargo Velocity Distributions (Linear) & Fits"

    ax.set_title(title_str, fontsize=13.5, fontweight='bold', pad=12)
    ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, f"cargo_dimensionless_velocity_overlay_{scale_str}", out_dirs)


def plot_log_velocity_grid(
    log_fit_results: Dict[str, dict],
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """対数無次元速度 log(v_tilde) のグリッドプロット (各粒子径 + ガウスフィット)"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    xlabel_str = r"$\log_{10}(\tilde{v}) = \log_{10}(|v| / v_{\mathrm{MT}})$" if log_base == "log10" else r"$\ln(\tilde{v}) = \ln(|v| / v_{\mathrm{MT}})$"
    ylabel_str = r"Probability Density $P(\log_{10}\tilde{v})$" if log_base == "log10" else r"Probability Density $P(\ln\tilde{v})$"
    hist_range = (-3.0, 0.8) if log_base == "log10" else (-6.9, 1.8)
    x_dense = np.linspace(hist_range[0], hist_range[1], 500)

    for idx, b in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        b_name = b["name"]
        label = b["label"]
        color = b["color"]
        res = log_fit_results.get(b_name, {})

        if not res.get("success", False):
            ax.set_title(f"$d = {label}$ (No data)", fontsize=11)
            continue

        bin_centers = res["bin_centers"]
        counts = res["counts"]
        bin_edges = res["bin_edges"]
        mu = res["gauss_mu"]
        sigma = res["gauss_sigma"]
        r2 = res.get("gauss_r2", np.nan)

        ax.bar(
            bin_centers, counts, width=np.diff(bin_edges),
            align='center', alpha=0.5, color=color, edgecolor=color,
            label=f"$d = {label}$\n($N = {res['n_points']:,}$)"
        )

        p_gauss = gaussian_pdf(x_dense, mu, sigma)
        ax.plot(x_dense, p_gauss, color='black', lw=2.0, label='Gaussian Fit')

        ax.axvline(res["mean_log_v"], color='#d62728', ls='--', lw=1.5, label=f"Mean: {res['mean_log_v']:.2f}")
        ax.axvline(res["median_log_v"], color='#2ca02c', ls=':', lw=1.5, label=f"Median: {res['median_log_v']:.2f}")

        text_str = (
            f"$\\mu = {mu:.3f}$\n"
            f"$\\sigma = {sigma:.3f}$\n"
            f"Skewness = ${res['skew_log_v']:.2f}$\n"
            f"$R^2 = {r2:.3f}$" if np.isfinite(r2) else f"$\\mu = {mu:.3f}$\n$\\sigma = {sigma:.3f}$"
        )
        ax.text(
            0.05, 0.95, text_str, transform=ax.transAxes,
            va='top', ha='left', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85)
        )

        ax.set_title(f"$d = {label}$ (Diameter)", fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel_str, fontsize=11)
        ax.set_ylabel(ylabel_str, fontsize=11)
        ax.set_xlim(hist_range[0], hist_range[1])
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.8)
        ax.grid(True, which='major', ls=':', alpha=0.6)

    title_main = (
        f"Logarithmic Dimensionless Cargo Velocity Distributions by Bead Size ({'Log10' if log_base == 'log10' else 'Natural Log'})\n"
        f"{xlabel_str}"
    )
    plt.suptitle(title_main, fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, f"cargo_dimensionless_{log_base}_velocity_grid", out_dirs)


def plot_log_velocity_overlay(
    log_fit_results: Dict[str, dict],
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """対数無次元速度 log(v_tilde) の全粒子径重ね合わせプロット"""
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    xlabel_str = r"$\log_{10}(\tilde{v}) = \log_{10}(|v| / v_{\mathrm{MT}})$" if log_base == "log10" else r"$\ln(\tilde{v}) = \ln(|v| / v_{\mathrm{MT}})$"
    ylabel_str = r"Probability Density $P(\log_{10}\tilde{v})$" if log_base == "log10" else r"Probability Density $P(\ln\tilde{v})$"
    hist_range = (-3.0, 0.8) if log_base == "log10" else (-6.9, 1.8)
    x_dense = np.linspace(hist_range[0], hist_range[1], 500)

    for b in BEADS_INFO:
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        marker = b["marker"]
        res = log_fit_results.get(b_name, {})

        if not res.get("success", False):
            continue

        bin_centers = res["bin_centers"]
        counts = res["counts"]
        mu = res["gauss_mu"]
        sigma = res["gauss_sigma"]

        ax.plot(
            bin_centers, counts, marker=marker, ms=5.5, ls='none',
            color=color, alpha=0.8, label=f"$d = {label}$ ($N={res['n_points']:,}$)"
        )
        p_gauss = gaussian_pdf(x_dense, mu, sigma)
        ax.plot(x_dense, p_gauss, color=color, lw=1.8, alpha=0.85)

    ax.set_xlabel(xlabel_str, fontsize=12.5)
    ax.set_ylabel(ylabel_str, fontsize=12.5)
    ax.set_xlim(hist_range[0], hist_range[1])
    ax.set_ylim(bottom=0)

    title_str = (
        f"Logarithmic Dimensionless Cargo Velocity Distributions ({'Log10' if log_base == 'log10' else 'Natural Log'}) & Gaussian Fits"
    )
    ax.set_title(title_str, fontsize=13.5, fontweight='bold', pad=12)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, f"cargo_dimensionless_{log_base}_velocity_overlay", out_dirs)


def plot_lognormal_qq_grid(
    qq_results: Dict[str, dict],
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """各粒子径および全体プールの対数正規 Q-Q プロット グリッド (2行4列: 6サイズ + 全体プール)"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]
    ylabel_str = r"Sample Quantiles: $\log_{10}(\tilde{v})$" if log_base == "log10" else r"Sample Quantiles: $\ln(\tilde{v})$"

    for idx, key in enumerate(target_keys):
        ax = axes_flat[idx]
        res = qq_results.get(key, {})

        if key == "overall_pooled":
            label = "All Pooled"
            color = "#333333"
            marker = "x"
        else:
            b_info = next(b for b in BEADS_INFO if b["name"] == key)
            label = b_info["label"]
            color = b_info["color"]
            marker = b_info["marker"]

        if not res.get("success", False):
            ax.set_title(f"{label} (No data)", fontsize=11)
            continue

        osm = res["theoretical_quantiles"]
        osr = res["sample_quantiles"]
        slope = res["slope"]
        intercept = res["intercept"]
        r2 = res["r2"]
        n_pts = res["n_points"]

        # データ点 (Q-Q 点)
        ax.plot(osm, osr, marker=marker, color=color, ms=4.5, alpha=0.65, ls='none', label=f"Data ($N={n_pts:,}$)")

        # 理論フィット直線
        x_line = np.array([np.min(osm), np.max(osm)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#d62728', lw=2.0, ls='--', label=f'Fit: $y = {slope:.2f}x {intercept:+.2f}$')

        # 統計テキスト
        ks_p = res.get("ks_p", np.nan)
        ks_text = f"KS $p = {ks_p:.2e}$" if ks_p < 0.001 else f"KS $p = {ks_p:.3f}$"
        text_str = (
            f"$R^2 = {r2:.4f}$\n"
            f"Slope $\\approx {slope:.3f}$\n"
            f"Intercept $\\approx {intercept:.3f}$\n"
            f"{ks_text}"
        )
        ax.text(
            0.05, 0.95, text_str, transform=ax.transAxes,
            va='top', ha='left', fontsize=9.0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85)
        )

        title_suffix = "Overall Pooled" if key == "overall_pooled" else f"$d = {label}$"
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Theoretical Quantiles (Standard Normal $z$)", fontsize=10.0)
        ax.set_ylabel(ylabel_str, fontsize=10.0)
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.8)

    # 8番目のパネルが余る場合は非表示
    if len(target_keys) < len(axes_flat):
        for rem_idx in range(len(target_keys), len(axes_flat)):
            axes_flat[rem_idx].axis('off')

    plt.suptitle(
        r"Lognormal Q-Q Plots for Dimensionless Cargo Velocity ($\log_{10}\tilde{v}$)" + "\n" +
        r"Comparison of $\log_{10}(\tilde{v})$ against Standard Normal Theoretical Quantiles",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, f"cargo_dimensionless_lognormal_qq_grid", out_dirs)


def plot_lognormal_qq_overlay(
    qq_results: Dict[str, dict],
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """全粒子径の標準化対数正規 Q-Q プロット (Standardized Q-Q Plot) 重ね合わせ"""
    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    for b in BEADS_INFO:
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        marker = b["marker"]
        res = qq_results.get(b_name, {})

        if not res.get("success", False):
            continue

        osm = res["theoretical_quantiles"]
        z_osr = res["standardized_quantiles"]
        r2 = res["r2"]

        ax.plot(
            osm, z_osr, marker=marker, ms=6.5, ls='none',
            color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=0.8,
            label=f"$d = {label}$ ($R^2={r2:.3f}$)"
        )

    # 理想的な対角線 y = x
    q_lim = 3.5
    ax.plot([-q_lim, q_lim], [-q_lim, q_lim], color='black', ls='--', lw=2.0, label='Standard Normal ($y = x$)')

    ax.set_xlabel(r"Theoretical Quantiles $z$ (Standard Normal)", fontsize=12.5)
    ax.set_ylabel(r"Standardized Sample Quantiles $Z = \frac{\log_{10}\tilde{v} - \mu}{\sigma}$", fontsize=12.5)
    ax.set_xlim(-q_lim, q_lim)
    ax.set_ylim(-q_lim, q_lim)

    ax.set_title(
        r"Standardized Lognormal Q-Q Plot Overlay Across All Particle Sizes" + "\n" +
        r"Deviation from $y=x$ indicates Non-Lognormal Behavior (Heavy Tails / Skewness)",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, "cargo_dimensionless_lognormal_qq_overlay", out_dirs)


def plot_fit_parameters_vs_diameter(
    fit_results: Dict[str, dict],
    out_dirs: List[Path]
):
    """粒子径に対する無次元フィッティングパラメータ (\\tilde{v}_s, \\tilde{v}_f, \\alpha) の依存性プロット (2パネル + 単体パネル)"""
    diameters = []
    vs_list, vs_err_list = [], []
    vf_list, vf_err_list = [], []
    alpha_list, alpha_err_list = [], []
    colors = []
    markers = []

    for b in BEADS_INFO:
        b_name = b["name"]
        res = fit_results.get(b_name, {})
        if not res.get("success", False) or res.get("popt_pdf") is None:
            continue

        d = b["diameter_um"]
        alpha, vs, vf = res["popt_pdf"]
        perr = res["perr_pdf"] if res["perr_pdf"] is not None else [0, 0, 0]

        diameters.append(d)
        alpha_list.append(alpha)
        alpha_err_list.append(perr[0] if np.isfinite(perr[0]) and perr[0] < 10 else 0.0)
        vs_list.append(vs)
        vs_err_list.append(perr[1] if np.isfinite(perr[1]) and perr[1] < 10 else 0.0)
        vf_list.append(vf)
        vf_err_list.append(perr[2] if np.isfinite(perr[2]) and perr[2] < 10 else 0.0)
        colors.append(b["color"])
        markers.append(b["marker"])

    if len(diameters) == 0:
        return

    diameters = np.array(diameters)
    vs_arr = np.array(vs_list)
    vf_arr = np.array(vf_list)
    alpha_arr = np.array(alpha_list)

    # 1. 2パネル統合プロット (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    for i in range(len(diameters)):
        ax1.errorbar(
            diameters[i], vs_arr[i], yerr=vs_err_list[i],
            fmt=markers[i], color='#882255', ecolor='black', elinewidth=1.6,
            capsize=4.5, capthick=1.2, markersize=9.5, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=r'Slow Scale $\tilde{v}_s$' if i == 0 else ""
        )
        ax1.errorbar(
            diameters[i], vf_arr[i], yerr=vf_err_list[i],
            fmt=markers[i], color='#44AA99', ecolor='black', elinewidth=1.6,
            capsize=4.5, capthick=1.2, markersize=9.5, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=r'Fast Scale $\tilde{v}_f$' if i == 0 else ""
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12.5)
    ax1.set_ylabel(r"Dimensionless Velocity Scale $\tilde{v}$", fontsize=12.5)
    ax1.set_title(r"Dimensionless Velocity Scales ($\tilde{v}_s < \tilde{v}_f$) vs Diameter", fontsize=13.0, fontweight='bold')
    ax1.grid(True, which='both', ls=':', alpha=0.6)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)

    for i in range(len(diameters)):
        ax2.errorbar(
            diameters[i], alpha_arr[i], yerr=alpha_err_list[i],
            fmt=markers[i], color=colors[i], ecolor='black', elinewidth=1.6,
            capsize=4.5, capthick=1.2, markersize=9.5, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {diameters[i]:.2f}\\,\\mu\\mathrm{{m}}$"
        )

    ax2.set_xscale('log')
    ax2.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12.5)
    ax2.set_ylabel(r"Fast Fraction $\alpha$", fontsize=12.5)
    ax2.set_title(r"Fast Component Fraction $\alpha$ vs Diameter", fontsize=13.0, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, which='both', ls=':', alpha=0.6)
    ax2.legend(loc='best', fontsize=9.5, framealpha=0.9)

    plt.suptitle(
        r"Double Exponential Fit Parameters vs Diameter: $P(\tilde{v}) = \frac{1-\alpha}{\tilde{v}_s}e^{-\tilde{v}/\tilde{v}_s} + \frac{\alpha}{\tilde{v}_f}e^{-\tilde{v}/\tilde{v}_f}$",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "cargo_dimensionless_velocity_fit_params_vs_diameter", out_dirs)

    # 2. 単体パネル 1: Velocity scales vs Diameter
    fig_vs, ax_vs = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        ax_vs.errorbar(
            diameters[i], vs_arr[i], yerr=vs_err_list[i],
            fmt=markers[i], color='#882255', ecolor='black', elinewidth=1.6,
            capsize=4.5, capthick=1.2, markersize=10.0, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=r'Slow Scale $\tilde{v}_s$' if i == 0 else ""
        )
        ax_vs.errorbar(
            diameters[i], vf_arr[i], yerr=vf_err_list[i],
            fmt=markers[i], color='#44AA99', ecolor='black', elinewidth=1.6,
            capsize=4.5, capthick=1.2, markersize=10.0, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=r'Fast Scale $\tilde{v}_f$' if i == 0 else ""
        )
    ax_vs.set_xscale('log')
    ax_vs.set_yscale('log')
    ax_vs.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_vs.set_ylabel(r"Dimensionless Velocity Scale $\tilde{v}$", fontsize=13.0)
    ax_vs.set_title(r"Velocity Scales $\tilde{v}_s, \tilde{v}_f$ vs Diameter", fontsize=13.5, fontweight='bold', pad=10)
    ax_vs.grid(True, which='both', ls=':', alpha=0.6)
    ax_vs.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_vs, "cargo_dimensionless_velocity_scales_vs_diameter", out_dirs)

    # 3. 単体パネル 2: Fast fraction alpha vs Diameter
    fig_alpha, ax_a = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        ax_a.errorbar(
            diameters[i], alpha_arr[i], yerr=alpha_err_list[i],
            fmt=markers[i], color=colors[i], ecolor='black', elinewidth=1.6,
            capsize=4.5, capthick=1.2, markersize=10.0, markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {diameters[i]:.2f}\\,\\mu\\mathrm{{m}}$"
        )
    ax_a.set_xscale('log')
    ax_a.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_a.set_ylabel(r"Fast Fraction $\alpha$", fontsize=13.0)
    ax_a.set_title(r"Fast Component Fraction $\alpha$ vs Diameter", fontsize=13.5, fontweight='bold', pad=10)
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.grid(True, which='both', ls=':', alpha=0.6)
    ax_a.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_alpha, "cargo_dimensionless_velocity_fraction_vs_diameter", out_dirs)


def plot_log_parameters_vs_diameter(
    log_fit_results: Dict[str, dict],
    out_dirs: List[Path]
):
    """粒子径に対する対数無次元速度統計量 (\\mu, \\sigma, median) の依存性プロット (2パネル + 単体パネル)"""
    diameters = []
    mu_list, sigma_list, med_list = [], [], []
    colors, markers = [], []

    for b in BEADS_INFO:
        b_name = b["name"]
        res = log_fit_results.get(b_name, {})
        if not res.get("success", False):
            continue

        diameters.append(b["diameter_um"])
        mu_list.append(res["gauss_mu"])
        sigma_list.append(res["gauss_sigma"])
        med_list.append(res["median_log_v"])
        colors.append(b["color"])
        markers.append(b["marker"])

    if len(diameters) == 0:
        return

    diameters = np.array(diameters)
    mu_arr = np.array(mu_list)
    sigma_arr = np.array(sigma_list)
    med_arr = np.array(med_list)

    # 1. 2パネル統合プロット (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    for i in range(len(diameters)):
        ax1.plot(
            diameters[i], mu_arr[i], marker=markers[i], color=colors[i], ms=9.5, ls='none',
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=r'Gaussian Center $\mu$' if i == 0 else ""
        )
        ax1.plot(
            diameters[i], med_arr[i], marker=markers[i], color=colors[i], ms=9.5, ls='none', fillstyle='none',
            markeredgecolor=colors[i], markeredgewidth=1.8, zorder=4,
            label=r'Median $\log_{10}(\tilde{v})$' if i == 0 else ""
        )

    ax1.set_xscale('log')
    ax1.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12.5)
    ax1.set_ylabel(r"$\log_{10}(\tilde{v})$ Center", fontsize=12.5)
    ax1.set_title(r"Center of $\log_{10}(\tilde{v})$ Distribution vs Diameter", fontsize=13.0, fontweight='bold')
    ax1.grid(True, which='both', ls=':', alpha=0.6)
    ax1.legend(loc='best', fontsize=10.5, framealpha=0.9)

    for i in range(len(diameters)):
        ax2.plot(
            diameters[i], sigma_arr[i], marker=markers[i], color=colors[i], ms=9.5, ls='none',
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {diameters[i]:.2f}\\,\\mu\\mathrm{{m}}$"
        )

    ax2.set_xscale('log')
    ax2.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12.5)
    ax2.set_ylabel(r"Standard Deviation $\sigma_{\log_{10}\tilde{v}}$", fontsize=12.5)
    ax2.set_title(r"Width of $\log_{10}(\tilde{v})$ Distribution vs Diameter", fontsize=13.0, fontweight='bold')
    ax2.set_ylim(0, max(sigma_arr) * 1.3)
    ax2.grid(True, which='both', ls=':', alpha=0.6)
    ax2.legend(loc='best', fontsize=9.5, framealpha=0.9)

    plt.suptitle(
        r"Logarithmic Dimensionless Velocity Parameters vs Particle Diameter ($\log_{10}\tilde{v}$)",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "cargo_dimensionless_log_fit_params_vs_diameter", out_dirs)

    # 2. 単体パネル 1: Center vs Diameter
    fig_c, ax_c = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        ax_c.plot(
            diameters[i], mu_arr[i], marker=markers[i], color=colors[i], ms=10.0, ls='none',
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=r'Gaussian Center $\mu$' if i == 0 else ""
        )
        ax_c.plot(
            diameters[i], med_arr[i], marker=markers[i], color=colors[i], ms=10.0, ls='none', fillstyle='none',
            markeredgecolor=colors[i], markeredgewidth=1.8, zorder=4,
            label=r'Median $\log_{10}(\tilde{v})$' if i == 0 else ""
        )
    ax_c.set_xscale('log')
    ax_c.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_c.set_ylabel(r"$\log_{10}(\tilde{v})$ Center", fontsize=13.0)
    ax_c.set_title(r"Distribution Center $\mu$, Median vs Diameter" + "\n($\\log_{10}\\tilde{v}$)", fontsize=13.5, fontweight='bold', pad=10)
    ax_c.grid(True, which='both', ls=':', alpha=0.6)
    ax_c.legend(loc='best', fontsize=10.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_c, "cargo_dimensionless_log_center_vs_diameter", out_dirs)

    # 3. 単体パネル 2: Width vs Diameter
    fig_w, ax_w = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        ax_w.plot(
            diameters[i], sigma_arr[i], marker=markers[i], color=colors[i], ms=10.0, ls='none',
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {diameters[i]:.2f}\\,\\mu\\mathrm{{m}}$"
        )
    ax_w.set_xscale('log')
    ax_w.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_w.set_ylabel(r"Standard Deviation $\sigma_{\log_{10}\tilde{v}}$", fontsize=13.0)
    ax_w.set_title(r"Distribution Width $\sigma$ vs Diameter" + "\n($\\log_{10}\\tilde{v}$)", fontsize=13.5, fontweight='bold', pad=10)
    ax_w.set_ylim(0, max(sigma_arr) * 1.3)
    ax_w.grid(True, which='both', ls=':', alpha=0.6)
    ax_w.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_w, "cargo_dimensionless_log_width_vs_diameter", out_dirs)


def save_summary_csv(fit_results: Dict[str, dict], out_dirs: List[Path]):
    """フィッティング結果および統計サマリーをCSVに保存"""
    records = []
    all_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for b_key in all_keys:
        res = fit_results.get(b_key, {})
        if not res.get("success", False):
            continue

        b_info = next((b for b in BEADS_INFO if b["name"] == b_key), None)
        d_um = b_info["diameter_um"] if b_info else np.nan

        popt_pdf = res.get("popt_pdf")
        perr_pdf = res.get("perr_pdf")

        rec = {
            "bead_name": b_key,
            "diameter_um": d_um,
            "n_points": res["n_points"],
            "mean_v_tilde": res["mean_v"],
            "median_v_tilde": res["median_v"],
            "std_v_tilde": res["std_v"],
            "q25_v_tilde": res["q25_v"],
            "q75_v_tilde": res["q75_v"],
            "q90_v_tilde": res["q90_v"],
            "q95_v_tilde": res["q95_v"],
            "q99_v_tilde": res["q99_v"],
            # PDF Model
            "pdf_alpha": popt_pdf[0] if popt_pdf is not None else np.nan,
            "pdf_alpha_err": perr_pdf[0] if perr_pdf is not None else np.nan,
            "pdf_vs_tilde": popt_pdf[1] if popt_pdf is not None else np.nan,
            "pdf_vs_err": perr_pdf[1] if perr_pdf is not None else np.nan,
            "pdf_vf_tilde": popt_pdf[2] if popt_pdf is not None else np.nan,
            "pdf_vf_err": perr_pdf[2] if perr_pdf is not None else np.nan,
            "pdf_r2": res.get("r2_pdf", np.nan),
        }
        records.append(rec)

    df_summary = pd.DataFrame(records)
    save_csv_to_all(df_summary, "cargo_dimensionless_velocity_summary", out_dirs)
    print("\n--- Dimensionless Velocity Summary Table ---")
    print(df_summary.to_string(index=False))


def save_log_summary_csv(
    log10_results: Dict[str, dict],
    ln_results: Dict[str, dict],
    out_dirs: List[Path]
):
    """対数無次元速度の統計サマリーCSVを保存"""
    records = []
    all_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for b_key in all_keys:
        res10 = log10_results.get(b_key, {})
        res_ln = ln_results.get(b_key, {})
        if not res10.get("success", False):
            continue

        b_info = next((b for b in BEADS_INFO if b["name"] == b_key), None)
        d_um = b_info["diameter_um"] if b_info else np.nan

        rec = {
            "bead_name": b_key,
            "diameter_um": d_um,
            "n_points": res10["n_points"],
            # log10 stats
            "mean_log10_v_tilde": res10["mean_log_v"],
            "median_log10_v_tilde": res10["median_log_v"],
            "std_log10_v_tilde": res10["std_log_v"],
            "skew_log10_v_tilde": res10["skew_log_v"],
            "kurt_log10_v_tilde": res10["kurt_log_v"],
            "gauss_mu_log10": res10["gauss_mu"],
            "gauss_sigma_log10": res10["gauss_sigma"],
            "gauss_r2_log10": res10.get("gauss_r2", np.nan),
            # ln stats
            "mean_ln_v_tilde": res_ln.get("mean_log_v", np.nan),
            "median_ln_v_tilde": res_ln.get("median_log_v", np.nan),
            "std_ln_v_tilde": res_ln.get("std_log_v", np.nan),
            "gauss_mu_ln": res_ln.get("gauss_mu", np.nan),
            "gauss_sigma_ln": res_ln.get("gauss_sigma", np.nan),
            "gauss_r2_ln": res_ln.get("gauss_r2", np.nan),
        }
        records.append(rec)

    df_log_summary = pd.DataFrame(records)
    save_csv_to_all(df_log_summary, "cargo_dimensionless_log_velocity_summary", out_dirs)
    print("\n--- Logarithmic Dimensionless Velocity Summary Table ---")
    print(df_log_summary.to_string(index=False))


def save_qq_summary_csv(qq_results: Dict[str, dict], out_dirs: List[Path]):
    """対数正規 Q-Q プロットの統計サマリーCSVを保存"""
    records = []
    all_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for b_key in all_keys:
        res = qq_results.get(b_key, {})
        if not res.get("success", False):
            continue

        b_info = next((b for b in BEADS_INFO if b["name"] == b_key), None)
        d_um = b_info["diameter_um"] if b_info else np.nan

        rec = {
            "bead_name": b_key,
            "diameter_um": d_um,
            "n_points": res["n_points"],
            "qq_linearity_r2": res["r2"],
            "qq_slope_sigma": res["slope"],
            "qq_intercept_mu": res["intercept"],
            "sample_mean_log10": res["mean"],
            "sample_std_log10": res["std"],
            "ks_statistic": res["ks_stat"],
            "ks_p_value": res["ks_p"],
        }
        records.append(rec)

    df_qq_summary = pd.DataFrame(records)
    save_csv_to_all(df_qq_summary, "cargo_dimensionless_qq_summary", out_dirs)
    print("\n--- Lognormal Q-Q Plot Summary Table ---")
    print(df_qq_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Plot dimensionless cargo velocity distribution P(\\tilde{v}), log(\\tilde{v}), and Q-Q plots by particle size."
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity" / "cargo_spin_velocity_points.csv",
        help="Path to cargo_spin_velocity_points.csv"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Root directory of raw experiment dataset"
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Extract directly from raw tracks instead of points CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "figure" / "dimensionless_velocity_distribution",
        help="Local output directory"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for velocity histogram"
    )

    args = parser.parse_args()
    apply_custom_style()

    root_dir = args.root_dir if args.root_dir else find_default_root()
    out_dirs = [args.output_dir]
    if root_dir.exists():
        nas_out_dir = root_dir / "figure" / "dimensionless_velocity_distribution"
        out_dirs.append(nas_out_dir)

    print(f"Output directories: {[str(d) for d in out_dirs]}")

    # データ読み込み
    if not args.from_raw and args.points_csv.exists():
        print(f"Loading from existing points CSV: {args.points_csv}")
        bead_velocities, _ = extract_from_points_csv(args.points_csv)
    else:
        print(f"Extracting from raw tracks under: {root_dir}")
        bead_velocities = extract_from_raw_tracks(root_dir)

    # 各ビーズ径および全体の分布計算・フィッティング・Q-Q 算出
    fit_results = {}
    log10_results = {}
    ln_results = {}
    qq_results = {}
    all_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for key in all_keys:
        if key not in bead_velocities or len(bead_velocities[key]) == 0:
            continue
        v_arr = bead_velocities[key]
        fit_results[key] = fit_dimensionless_velocity_distribution(v_arr, n_bins=args.bins)
        log10_results[key] = fit_log_velocity_distribution(v_arr, log_base="log10", n_bins=args.bins)
        ln_results[key] = fit_log_velocity_distribution(v_arr, log_base="ln", n_bins=args.bins)
        qq_results[key] = compute_lognormal_qq(v_arr, log_base="log10")

    # 1. 通常の無次元速度プロット (Linear & Semilog)
    plot_dimensionless_velocity_grid(fit_results, out_dirs, semilog=False)
    plot_dimensionless_velocity_grid(fit_results, out_dirs, semilog=True)
    plot_dimensionless_velocity_overlay(fit_results, out_dirs, semilog=False)
    plot_dimensionless_velocity_overlay(fit_results, out_dirs, semilog=True)
    plot_fit_parameters_vs_diameter(fit_results, out_dirs)
    save_summary_csv(fit_results, out_dirs)

    # 2. 対数無次元速度プロット (log10 & ln)
    plot_log_velocity_grid(log10_results, out_dirs, log_base="log10")
    plot_log_velocity_overlay(log10_results, out_dirs, log_base="log10")
    plot_log_velocity_grid(ln_results, out_dirs, log_base="ln")
    plot_log_velocity_overlay(ln_results, out_dirs, log_base="ln")
    plot_log_parameters_vs_diameter(log10_results, out_dirs)
    save_log_summary_csv(log10_results, ln_results, out_dirs)

    # 3. 対数正規 Q-Q プロット (Grid & Overlay)
    plot_lognormal_qq_grid(qq_results, out_dirs, log_base="log10")
    plot_lognormal_qq_overlay(qq_results, out_dirs, log_base="log10")
    save_qq_summary_csv(qq_results, out_dirs)

    print("\nAll dimensionless velocity (Linear, Logarithmic, & Lognormal Q-Q) plots and CSVs generated successfully!")


if __name__ == "__main__":
    main()
