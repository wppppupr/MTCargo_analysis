#!/usr/bin/env python3
"""
plot_cargo_velocity_distribution.py

貨物微粒子（ビーズ）の速度分布 P(v) を算出し、
2成分指数分布（Double Exponential Distribution）:
    P(v) = (1 - \alpha) e^{-v/v_s} + \alpha e^{-v/v_f}  (v_s < v_f)
または確率密度関数（正規化形式）:
    P(v) = \frac{1 - \alpha}{v_s} e^{-v/v_s} + \frac{\alpha}{v_f} e^{-v/v_f}  (v_s < v_f)
でフィッティングして可視化・パラメータ算出を行うスクリプトです。

出力ファイル:
1. figure/velocity_distribution/cargo_velocity_distribution_grid_linear.png / .svg
   (各粒子径および全体プール 速度分布 & 2成分指数フィット: Linear スケール)
2. figure/velocity_distribution/cargo_velocity_distribution_grid_semilog.png / .svg
   (各粒子径および全体プール 速度分布 & 2成分指数フィット: Semilog-y スケール)
3. figure/velocity_distribution/cargo_velocity_distribution_overlay_linear.png / .svg
   (全粒子径 重ね合わせプロット: Linear スケール)
4. figure/velocity_distribution/cargo_velocity_distribution_overlay_semilog.png / .svg
   (全粒子径 重ね合わせプロット: Semilog-y スケール)
5. figure/velocity_distribution/cargo_velocity_fit_params_vs_diameter.png / .svg
   (粒子径依存性: vs, vf, \alpha のプロット)
6. figure/velocity_distribution/cargo_velocity_distribution_fitting_summary.csv
   (フィッティングパラメータおよび統計値のサマリーCSV)
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

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ビーズ基本情報
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "label": "0.63 μm", "marker": "^", "color": "#1f77b4"},
    {"name": "beads1um",  "diameter_um": 1.18, "label": "1.18 μm", "marker": "o", "color": "#ff7f0e"},
    {"name": "beads3um",  "diameter_um": 3.37, "label": "3.37 μm", "marker": "d", "color": "#2ca02c"},
    {"name": "beads5um",  "diameter_um": 5.00, "label": "5.00 μm", "marker": "p", "color": "#d62728"},
    {"name": "beads7um",  "diameter_um": 7.24, "label": "7.24 μm", "marker": "h", "color": "#9467bd"},
    {"name": "beads20um", "diameter_um": 20.0, "label": "20.0 μm", "marker": "s", "color": "#8c564b"},
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
    確率密度関数 (Normalized PDF):
    P(v) = (1 - alpha)/vs * exp(-v/vs) + alpha/vf * exp(-v/vf)
    \int_0^\infty P(v) dv = 1  (vs < vf)
    """
    return (1.0 - alpha) / vs * np.exp(-v / vs) + alpha / vf * np.exp(-v / vf)


def double_exponential_amp(v: np.ndarray, A: float, alpha: float, vs: float, vf: float) -> np.ndarray:
    """
    振幅パラメータ A 付き 2成分指数関数:
    P(v) = A * [(1 - alpha) * exp(-v/vs) + alpha * exp(-v/vf)]
    """
    return A * ((1.0 - alpha) * np.exp(-v / vs) + alpha * np.exp(-v / vf))


def single_exponential_pdf(v: np.ndarray, v_mean: float) -> np.ndarray:
    """1成分指数関数: P(v) = (1/v_mean) * exp(-v/v_mean)"""
    return (1.0 / v_mean) * np.exp(-v / v_mean)


def extract_bead_velocities(root_dir: Path, scale: float = 0.11, frame_interval: float = 4.0) -> Dict[str, np.ndarray]:
    """各ビーズ径の全軌跡データから瞬時速度 [μm/s] を抽出"""
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
            try:
                df = pd.read_csv(f)
            except Exception:
                continue

            if not {'particle', 'x', 'y', 'frame'}.issubset(df.columns):
                continue

            for tid, g in df.groupby('particle'):
                if len(g) < 2:
                    continue
                g = g.sort_values('frame')
                dx = np.diff(g['x'].values) * scale
                dy = np.diff(g['y'].values) * scale
                dt = np.diff(g['frame'].values) * frame_interval
                valid = dt > 0
                if np.sum(valid) == 0:
                    continue
                sp = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
                sp = sp[np.isfinite(sp)]
                speeds_list.extend(sp)

        arr = np.array(speeds_list)
        bead_velocities[b_name] = arr
        all_pooled.extend(speeds_list)
        print(f"Loaded {b_name}: {len(speeds_list)} points, mean velocity = {np.mean(speeds_list):.4f} μm/s" if len(speeds_list) > 0 else f"No data for {b_name}")

    bead_velocities["overall_pooled"] = np.array(all_pooled)
    print(f"Total pooled dataset: {len(all_pooled)} points, mean = {np.mean(all_pooled):.4f} μm/s")
    return bead_velocities


def fit_velocity_distribution(
    speeds: np.ndarray,
    n_bins: int = 50,
    max_v: Optional[float] = None
) -> dict:
    """
    速度分布のヒストグラム計算および2成分指数分布のマルチスタート・フィッティングを行う
    """
    if len(speeds) < 10:
        return {"success": False}

    if max_v is None:
        max_v = float(np.percentile(speeds, 99.5) * 1.1)

    # 正の速度値のみを使用
    valid_speeds = speeds[(speeds >= 0) & (speeds <= max_v)]
    if len(valid_speeds) < 10:
        valid_speeds = speeds[speeds >= 0]

    counts, bin_edges = np.histogram(valid_speeds, bins=n_bins, range=(0, max_v), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    mask = counts > 0
    x_fit = bin_centers[mask]
    y_fit = counts[mask]

    mean_v = float(np.mean(valid_speeds))

    # 1. 確率密度関数 (PDF形式) フィッティング（マルチスタート）
    best_popt_pdf = None
    best_perr_pdf = None
    best_r2_pdf = -np.inf

    grid_p0 = [
        [0.3, mean_v * 0.3, mean_v * 1.5],
        [0.5, mean_v * 0.2, mean_v * 2.0],
        [0.1, mean_v * 0.5, mean_v * 3.0],
        [0.7, mean_v * 0.1, mean_v * 1.2],
        [0.4, 0.03, 0.18],
    ]
    bounds_pdf = ([0.0, 1e-4, 1e-4], [1.0, 5.0, 5.0])

    for p0 in grid_p0:
        try:
            popt_raw, pcov_raw = curve_fit(
                double_exponential_pdf, x_fit, y_fit,
                p0=p0, bounds=bounds_pdf, maxfev=15000
            )
            # vs < vf の順序を保証
            if popt_raw[1] > popt_raw[2]:
                alpha_c = 1.0 - popt_raw[0]
                vs_c = popt_raw[2]
                vf_c = popt_raw[1]
                perr = np.sqrt(np.diag(pcov_raw)) if pcov_raw is not None else np.zeros(3)
                perr_c = [perr[0], perr[2], perr[1]]
            else:
                alpha_c = popt_raw[0]
                vs_c = popt_raw[1]
                vf_c = popt_raw[2]
                perr_c = np.sqrt(np.diag(pcov_raw)).tolist() if pcov_raw is not None else [0, 0, 0]

            popt_c = [alpha_c, vs_c, vf_c]
            y_pred = double_exponential_pdf(x_fit, *popt_c)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2_c = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            if r2_c > best_r2_pdf:
                best_r2_pdf = r2_c
                best_popt_pdf = popt_c
                best_perr_pdf = perr_c
        except Exception:
            continue

    # 2. 振幅付きモデル フィッティング: P(v) = A * [(1-alpha) e^{-v/vs} + alpha e^{-v/vf}]
    best_popt_amp = None
    best_perr_amp = None
    best_r2_amp = -np.inf

    grid_p0_amp = [
        [y_fit[0], 0.3, mean_v * 0.3, mean_v * 1.5],
        [y_fit[0] * 1.2, 0.5, mean_v * 0.2, mean_v * 2.0],
        [y_fit[0] * 0.8, 0.2, mean_v * 0.5, mean_v * 3.0],
    ]
    bounds_amp = ([0.0, 0.0, 1e-4, 1e-4], [200.0, 1.0, 5.0, 5.0])

    for p0_a in grid_p0_amp:
        try:
            popt_raw_a, pcov_raw_a = curve_fit(
                double_exponential_amp, x_fit, y_fit,
                p0=p0_a, bounds=bounds_amp, maxfev=15000
            )
            if popt_raw_a[2] > popt_raw_a[3]:
                A_c = popt_raw_a[0]
                alpha_c = 1.0 - popt_raw_a[1]
                vs_c = popt_raw_a[3]
                vf_c = popt_raw_a[2]
                perr_a = np.sqrt(np.diag(pcov_raw_a)) if pcov_raw_a is not None else np.zeros(4)
                perr_c = [perr_a[0], perr_a[1], perr_a[3], perr_a[2]]
            else:
                A_c = popt_raw_a[0]
                alpha_c = popt_raw_a[1]
                vs_c = popt_raw_a[2]
                vf_c = popt_raw_a[3]
                perr_c = np.sqrt(np.diag(pcov_raw_a)).tolist() if pcov_raw_a is not None else [0, 0, 0, 0]

            popt_c = [A_c, alpha_c, vs_c, vf_c]
            y_pred_a = double_exponential_amp(x_fit, *popt_c)
            ss_res_a = np.sum((y_fit - y_pred_a) ** 2)
            ss_tot_a = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2_ca = 1.0 - (ss_res_a / ss_tot_a) if ss_tot_a > 0 else np.nan

            if r2_ca > best_r2_amp:
                best_r2_amp = r2_ca
                best_popt_amp = popt_c
                best_perr_amp = perr_c
        except Exception:
            continue

    return {
        "success": True,
        "valid_speeds": valid_speeds,
        "counts": counts,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_widths": bin_widths,
        "max_v": max_v,
        "mean_v": float(np.mean(valid_speeds)),
        "median_v": float(np.median(valid_speeds)),
        "std_v": float(np.std(valid_speeds, ddof=1)),
        "n_points": len(valid_speeds),
        "popt_pdf": best_popt_pdf,
        "perr_pdf": best_perr_pdf,
        "r2_pdf": best_r2_pdf if best_r2_pdf > -np.inf else np.nan,
        "popt_amp": best_popt_amp,
        "perr_amp": best_perr_amp,
        "r2_amp": best_r2_amp if best_r2_amp > -np.inf else np.nan,
    }


def plot_velocity_distributions_grid(
    fit_results: Dict[str, dict],
    output_dir: Path,
    semilog: bool = False
):
    """粒子径ごとおよび全体の速度分布とフィッティング曲線のグリッドプロット (2x3 + Overall)"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=False, sharey=False)
    axes = axes.flatten()
    scale_str = "semilog" if semilog else "linear"

    for idx, b in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        res = fit_results.get(b_name, {})

        if not res.get("success", False):
            ax.text(0.5, 0.5, f"No Data: {label}", ha='center', va='center', transform=ax.transAxes)
            continue

        bin_centers = res["bin_centers"]
        counts = res["counts"]
        bin_edges = res["bin_edges"]
        max_v = res["max_v"]
        popt = res["popt_pdf"]

        # ヒストグラムの描画
        ax.bar(
            bin_centers, counts, width=np.diff(bin_edges),
            align='center', alpha=0.45, color=color, edgecolor=color,
            label=f"Data ($N={res['n_points']:,}$)"
        )

        v_dense = np.linspace(0, max_v, 400)

        if popt is not None:
            alpha, vs, vf = popt
            p_total = double_exponential_pdf(v_dense, alpha, vs, vf)
            p_slow = (1.0 - alpha) / vs * np.exp(-v_dense / vs)
            p_fast = alpha / vf * np.exp(-v_dense / vf)

            ax.plot(v_dense, p_total, color='black', lw=2.2, label=r'Fit $P(v)$')
            ax.plot(v_dense, p_slow, color='#d62728', ls='--', lw=1.6, label=r'Slow: $(1-\alpha)e^{-v/v_s}/v_s$')
            ax.plot(v_dense, p_fast, color='#1f77b4', ls=':', lw=1.8, label=r'Fast: $\alpha e^{-v/v_f}/v_f$')

            param_text = (
                f"$v_s = {vs:.3f}\\ \\mu\\mathrm{{m/s}}$\n"
                f"$v_f = {vf:.3f}\\ \\mu\\mathrm{{m/s}}$\n"
                f"$\\alpha = {alpha:.3f}$\n"
                f"$R^2 = {res['r2_pdf']:.3f}$"
            )
            ax.text(
                0.95, 0.95, param_text, transform=ax.transAxes,
                va='top', ha='right',
                fontsize=10.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9)
            )

        ax.set_title(f"Diameter $d = {label}$", fontsize=13, fontweight='bold')
        ax.set_xlabel(r"Velocity $v$ [$\mu\mathrm{m/s}$]", fontsize=11.5)
        ax.set_ylabel(r"Probability Density $P(v)$ [$(\mu\mathrm{m/s})^{-1}$]", fontsize=11.5)
        ax.set_xlim(0, max_v)

        if semilog:
            ax.set_yscale('log')
            y_min = max(1e-3, np.min(counts[counts > 0]) * 0.5) if np.any(counts > 0) else 1e-3
            y_max = np.max(counts) * 2.5 if np.any(counts > 0) else 10.0
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(bottom=0)

        ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.6)
        ax.legend(loc='lower left' if semilog else 'upper right', fontsize=8.5, framealpha=0.85)

    plt.suptitle(
        f"Cargo Particle Velocity Distribution & Double Exponential Fit ({'Semilog-y' if semilog else 'Linear'})\n"
        r"$P(v) = (1-\alpha)\frac{1}{v_s}e^{-v/v_s} + \alpha\frac{1}{v_f}e^{-v/v_f} \quad (v_s < v_f)$",
        fontsize=14.5, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / f"cargo_velocity_distribution_grid_{scale_str}.png"
    svg_path = output_dir / f"cargo_velocity_distribution_grid_{scale_str}.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")


def plot_velocity_distributions_grid(
    fit_results: Dict[str, dict],
    out_dirs: List[Path],
    semilog: bool = False
):
    """
    全6粒子径 + 全体プール (7パネル) のグリッドプロット
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5), sharex=False, sharey=False)
    axes = axes.flatten()

    scale_str = "semilog" if semilog else "linear"
    v_dense = np.linspace(0, 2.0, 500)

    plot_order = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for i, b_key in enumerate(plot_order):
        ax = axes[i]
        res = fit_results.get(b_key, {})

        if not res.get("success", False):
            ax.text(0.5, 0.5, "Fit Failed or No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        bin_centers = res["bin_centers"]
        counts = res["counts"]
        bin_edges = res["bin_edges"]
        max_v = res["max_v"]
        popt_pdf = res["popt_pdf"]

        if b_key == "overall_pooled":
            color = "#333333"
            title = r"$\mathbf{All\ Beads\ Pooled}$"
        else:
            b_info = next(b for b in BEADS_INFO if b["name"] == b_key)
            color = b_info["color"]
            title = f"{b_info['label']} Beads"

        # ヒストグラム
        ax.bar(
            bin_centers, counts, width=np.diff(bin_edges),
            align='center', alpha=0.45, color=color, edgecolor=color,
            label=f'Data ($N={res["n_points"]:,}$)'
        )

        # 2成分指数フィット曲線
        if popt_pdf is not None:
            alpha, vs, vf = popt_pdf
            p_fit = double_exponential_pdf(v_dense, alpha, vs, vf)
            p_slow = (1.0 - alpha) / vs * np.exp(-v_dense / vs)
            p_fast = alpha / vf * np.exp(-v_dense / vf)

            ax.plot(v_dense, p_fit, color='black', lw=2.0, label='Fit (Double Exp)')
            ax.plot(v_dense, p_slow, color='#d62728', ls='--', lw=1.5, label=f'Slow ($v_s={vs:.2f}$)')
            ax.plot(v_dense, p_fast, color='#1f77b4', ls=':', lw=1.8, label=f'Fast ($v_f={vf:.2f}$)')

            param_box = (
                f"$v_s = {vs:.3f}\\ \\mu\\mathrm{{m/s}}$\n"
                f"$v_f = {vf:.3f}\\ \\mu\\mathrm{{m/s}}$\n"
                f"$\\alpha = {alpha:.3f}$\n"
                f"$R^2 = {res['r2_pdf']:.3f}$"
            )
            ax.text(
                0.95, 0.95, param_box, transform=ax.transAxes,
                va='top', ha='right', fontsize=9.0,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
            )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Velocity $v$ [$\mu\mathrm{m/s}$]", fontsize=10.5)
        ax.set_ylabel(r"Probability Density $P(v)$ [$(\mu\mathrm{m/s})^{-1}$]", fontsize=10.5)
        ax.set_xlim(0, max_v)

        if semilog:
            ax.set_yscale('log')
            y_min = max(1e-3, np.min(counts[counts > 0]) * 0.5) if np.any(counts > 0) else 1e-3
            y_max = np.max(counts) * 2.5 if np.any(counts > 0) else 10.0
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(bottom=0)

        ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.6)
        ax.legend(loc='lower left' if semilog else 'upper right', fontsize=8.5, framealpha=0.85)

    plt.suptitle(
        f"Cargo Particle Velocity Distribution & Double Exponential Fit ({'Semilog-y' if semilog else 'Linear'})\n"
        r"$P(v) = (1-\alpha)\frac{1}{v_s}e^{-v/v_s} + \alpha\frac{1}{v_f}e^{-v/v_f} \quad (v_s < v_f)$",
        fontsize=14.5, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure_to_all(fig, f"cargo_velocity_distribution_grid_{scale_str}", out_dirs)


def plot_overall_pooled_distribution(
    fit_results: Dict[str, dict],
    out_dirs: List[Path]
):
    """全粒子プール全体の速度分布とフィッティング (Linear & Semilog 2パネル)"""
    res = fit_results.get("overall_pooled", {})
    if not res.get("success", False):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    bin_centers = res["bin_centers"]
    counts = res["counts"]
    bin_edges = res["bin_edges"]
    max_v = res["max_v"]
    popt = res["popt_pdf"]
    v_dense = np.linspace(0, max_v, 500)

    for ax, semilog, scale_title in [(ax1, False, "Linear"), (ax2, True, "Semilog-y")]:
        ax.bar(
            bin_centers, counts, width=np.diff(bin_edges),
            align='center', alpha=0.45, color='#4c72b0', edgecolor='#4c72b0',
            label=f"All Beads Pooled ($N={res['n_points']:,}$)"
        )

        if popt is not None:
            alpha, vs, vf = popt
            p_total = double_exponential_pdf(v_dense, alpha, vs, vf)
            p_slow = (1.0 - alpha) / vs * np.exp(-v_dense / vs)
            p_fast = alpha / vf * np.exp(-v_dense / vf)

            ax.plot(v_dense, p_total, color='black', lw=2.4, label=r'Fit $P(v)$')
            ax.plot(v_dense, p_slow, color='#d62728', ls='--', lw=1.8, label=r'Slow: $(1-\alpha)e^{-v/v_s}/v_s$')
            ax.plot(v_dense, p_fast, color='#1f77b4', ls=':', lw=2.0, label=r'Fast: $\alpha e^{-v/v_f}/v_f$')

            param_text = (
                f"$v_s = {vs:.3f}\\ \\mu\\mathrm{{m/s}}$\n"
                f"$v_f = {vf:.3f}\\ \\mu\\mathrm{{m/s}}$\n"
                f"$\\alpha = {alpha:.3f}$\n"
                f"$R^2 = {res['r2_pdf']:.3f}$"
            )
            ax.text(
                0.95, 0.95, param_text, transform=ax.transAxes,
                va='top', ha='right',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9)
            )

        ax.set_title(f"Overall Pooled Velocity Distribution ({scale_title})", fontsize=12.5, fontweight='bold')
        ax.set_xlabel(r"Velocity $v$ [$\mu\mathrm{m/s}$]", fontsize=12)
        ax.set_ylabel(r"Probability Density $P(v)$ [$(\mu\mathrm{m/s})^{-1}$]", fontsize=12)
        ax.set_xlim(0, max_v)

        if semilog:
            ax.set_yscale('log')
            y_min = max(1e-3, np.min(counts[counts > 0]) * 0.5) if np.any(counts > 0) else 1e-3
            y_max = np.max(counts) * 2.5 if np.any(counts > 0) else 10.0
            ax.set_ylim(y_min, y_max)
            ax.legend(loc='lower left', fontsize=9.5, framealpha=0.85)
        else:
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=9.5, framealpha=0.85)

        ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.6)

    plt.suptitle(
        r"Overall Cargo Velocity Distribution (All Beads Pooled, $N=35,310$)" + "\n" +
        r"$P(v) = (1-\alpha)\frac{1}{v_s}e^{-v/v_s} + \alpha\frac{1}{v_f}e^{-v/v_f} \quad (v_s < v_f)$",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure_to_all(fig, "cargo_velocity_distribution_overall_pooled_2panel", out_dirs)


def plot_velocity_distributions_overlay(
    fit_results: Dict[str, dict],
    out_dirs: List[Path],
    semilog: bool = False
):
    """全粒子径の速度分布とフィッティング曲線の重ね合わせプロット"""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scale_str = "semilog" if semilog else "linear"

    v_dense = np.linspace(0, 1.5, 500)

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

        # データ点プロット
        ax.plot(
            bin_centers, counts, marker=marker, ms=5.5, ls='none',
            color=color, alpha=0.75, label=f"$d = {label}$"
        )

        # フィット曲線
        if popt is not None:
            alpha, vs, vf = popt
            p_total = double_exponential_pdf(v_dense, alpha, vs, vf)
            ax.plot(v_dense, p_total, color=color, lw=1.8, alpha=0.9)

    ax.set_xlabel(r"Velocity $v$ [$\mu\mathrm{m/s}$]", fontsize=12.5)
    ax.set_ylabel(r"Probability Density $P(v)$ [$(\mu\mathrm{m/s})^{-1}$]", fontsize=12.5)
    ax.set_xlim(0, 1.2)

    if semilog:
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 60.0)
        title_str = "Cargo Velocity Distributions (Semilog-y) & Double Exponential Fits"
    else:
        ax.set_ylim(0, 35.0)
        title_str = "Cargo Velocity Distributions (Linear) & Double Exponential Fits"

    ax.set_title(title_str, fontsize=13.5, fontweight='bold', pad=12)
    ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, f"cargo_velocity_distribution_overlay_{scale_str}", out_dirs)


def plot_fit_parameters_vs_diameter(
    fit_results: Dict[str, dict],
    out_dirs: List[Path]
):
    """粒子径に対するフィッティングパラメータ (vs, vf, \alpha) の依存性プロット"""
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

    diameters = np.array(diameters)
    vs_arr = np.array(vs_list)
    vf_arr = np.array(vf_list)
    alpha_arr = np.array(alpha_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # パネル 1: 特性速度スケール vs, vf vs 粒子径 d
    for i in range(len(diameters)):
        ax1.errorbar(
            diameters[i], vs_arr[i], yerr=vs_err_list[i],
            fmt=markers[i], color='#d62728', ms=8, capsize=4, elinewidth=1.5,
            label='Slow Scale $v_s$' if i == 0 else ""
        )
        ax1.errorbar(
            diameters[i], vf_arr[i], yerr=vf_err_list[i],
            fmt=markers[i], color='#1f77b4', ms=8, capsize=4, elinewidth=1.5,
            label='Fast Scale $v_f$' if i == 0 else ""
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12)
    ax1.set_ylabel(r"Characteristic Velocity Scale [$\mu\mathrm{m/s}$]", fontsize=12)
    ax1.set_title(r"Velocity Scales ($v_s < v_f$) vs Diameter", fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', ls=':', alpha=0.6)
    ax1.legend(loc='best', fontsize=11)

    # パネル 2: 高速成分比率 \alpha vs 粒子径 d
    for i in range(len(diameters)):
        ax2.errorbar(
            diameters[i], alpha_arr[i], yerr=alpha_err_list[i],
            fmt=markers[i], color=colors[i], ms=8, capsize=4, elinewidth=1.5
        )

    ax2.set_xscale('log')
    ax2.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12)
    ax2.set_ylabel(r"Fast Fraction $\alpha$", fontsize=12)
    ax2.set_title(r"Fast Component Fraction $\alpha$ vs Diameter", fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, which='both', ls=':', alpha=0.6)

    plt.suptitle(
        r"Double Exponential Fitting Parameters vs Particle Diameter: $P(v) = \frac{1-\alpha}{v_s}e^{-v/v_s} + \frac{\alpha}{v_f}e^{-v/v_f}$",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure_to_all(fig, "cargo_velocity_fit_params_vs_diameter", out_dirs)


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
        popt_amp = res.get("popt_amp")
        perr_amp = res.get("perr_amp")

        rec = {
            "bead_name": b_key,
            "diameter_um": d_um,
            "n_points": res["n_points"],
            "mean_velocity_um_s": res["mean_v"],
            "median_velocity_um_s": res["median_v"],
            "std_velocity_um_s": res["std_v"],
            # PDF Model
            "pdf_alpha": popt_pdf[0] if popt_pdf else np.nan,
            "pdf_alpha_err": perr_pdf[0] if perr_pdf else np.nan,
            "pdf_vs_um_s": popt_pdf[1] if popt_pdf else np.nan,
            "pdf_vs_err": perr_pdf[1] if perr_pdf else np.nan,
            "pdf_vf_um_s": popt_pdf[2] if popt_pdf else np.nan,
            "pdf_vf_err": perr_pdf[2] if perr_pdf else np.nan,
            "pdf_r2": res.get("r2_pdf", np.nan),
            # Amp Model
            "amp_A": popt_amp[0] if popt_amp else np.nan,
            "amp_A_err": perr_amp[0] if perr_amp else np.nan,
            "amp_alpha": popt_amp[1] if popt_amp else np.nan,
            "amp_alpha_err": perr_amp[1] if perr_amp else np.nan,
            "amp_vs_um_s": popt_amp[2] if popt_amp else np.nan,
            "amp_vs_err": perr_amp[2] if perr_amp else np.nan,
            "amp_vf_um_s": popt_amp[3] if popt_amp else np.nan,
            "amp_vf_err": perr_amp[3] if perr_amp else np.nan,
            "amp_r2": res.get("r2_amp", np.nan),
        }
        records.append(rec)

    df_summary = pd.DataFrame(records)
    save_csv_to_all(df_summary, "cargo_velocity_distribution_fitting_summary", out_dirs)
    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Cargo particle velocity distribution and double exponential fitting")
    parser.add_argument("--root-dir", type=Path, default=None, help="Root directory containing beads data")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "figure" / "velocity_distribution", help="Local output directory")
    parser.add_argument("--scale", type=float, default=0.11, help="Pixel scale (μm/px)")
    parser.add_argument("--frame-interval", type=float, default=4.0, help="Frame interval (s)")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    args = parser.parse_args()

    root_dir = args.root_dir if args.root_dir else find_default_root()
    out_dirs = [args.output_dir]
    if root_dir and root_dir.exists():
        nas_out_dir = root_dir / "figure" / "velocity_distribution"
        if nas_out_dir not in out_dirs:
            out_dirs.append(nas_out_dir)

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Using root directory: {root_dir}")
    print("Output directories:")
    for d in out_dirs:
        print(f"  - {d}")

    apply_custom_style()

    # 1. 速度データの抽出
    bead_velocities = extract_bead_velocities(
        root_dir=root_dir,
        scale=args.scale,
        frame_interval=args.frame_interval
    )

    # 2. 速度分布の計算 & フィッティング
    fit_results = {}
    for key, speeds in bead_velocities.items():
        fit_results[key] = fit_velocity_distribution(speeds, n_bins=args.bins)

    # 3. プロット生成
    # (a) グリッドプロット (Linear & Semilog)
    plot_velocity_distributions_grid(fit_results, out_dirs, semilog=False)
    plot_velocity_distributions_grid(fit_results, out_dirs, semilog=True)

    # (b) 全体プール2パネルプロット
    plot_overall_pooled_distribution(fit_results, out_dirs)

    # (c) 重ね合わせプロット (Linear & Semilog)
    plot_velocity_distributions_overlay(fit_results, out_dirs, semilog=False)
    plot_velocity_distributions_overlay(fit_results, out_dirs, semilog=True)

    # (d) パラメータ依存性プロット
    plot_fit_parameters_vs_diameter(fit_results, out_dirs)

    # 4. CSV サマリーの保存
    df_summary = save_summary_csv(fit_results, out_dirs)
    print("\n--- Fitting Summary ---")
    print(df_summary[["bead_name", "diameter_um", "n_points", "mean_velocity_um_s", "pdf_alpha", "pdf_vs_um_s", "pdf_vf_um_s", "pdf_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
