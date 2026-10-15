#!/usr/bin/env python3
"""
plot_xi_vs_velocity.py

貨物粒子（ビーズ）の各粒子 i・各時間フレーム t における
    ・瞬時局所微小管配向相関長 xi_{i,t} [um]（縦軸）
    ・瞬時速度 v_{i,t} [um/s]（横軸）
の散布図を、貨物粒子の直径（0.63, 1.18, 3.37, 5.0, 7.24, 20 um）ごとに作成するスクリプトです。

【データソース (各実験ディレクトリ内)】
- 軌跡: beads_tracks.csv  (frame, particle, x, y)
- 微小管アクティブフローの空間配向相関: angular_correlation_w.zarr  (distance, frame, particle)

【xi_{i,t} の算出】
粒子 i・フレーム t の近傍における空間配向相関 C(r) の縦軸の対数をとり、
    ln C(r) = ln a - r / xi
を r ∈ [min_r, max_r] の範囲で線形回帰し、xi = -1 / slope を得る。
- min_r = R_c * --min_r_factor（ビーズ自身が占めるマスク領域を除外）
- 対数をとれない C(r) < --min_corr_threshold の点、および範囲外の点は除外
- 有効点が --min_points 未満、または xi が [--xi_min, --xi_max] を外れる場合は棄却（NaN）

【v_{i,t} の算出】
    v_{i,t} = | r_i(t + tau) - r_i(t) | / (tau * --frame_interval)   [um/s]
（--scale = 0.11 um/pixel, --frame_interval = 4 s, --tau = 1）
連続フレーム対 (t, t+tau) が存在する場合のみ有効。

【出力ファイル】
1. figure/xi_vs_velocity/xi_vs_velocity_<bead>.png / .svg    : 直径ごとの散布図（全 xi_{i,t} 点）
2. figure/xi_vs_velocity/xi_vs_velocity_all_beads.png / .svg : 全直径一覧（2x3 パネル）
3. figure/xi_vs_velocity/xi_vs_velocity_<bead>.csv           : 各 (i, t) の生データ
4. figure/xi_vs_velocity/xi_vs_velocity_statistics.csv       : 直径ごとの統計量
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# プロジェクト設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import hmm_cargo as hc

# スタイル適用
_style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if _style_path.exists():
    plt.style.use(str(_style_path))

# データルート候補（存在するものを自動選択）
POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root() -> Optional[Path]:
    """存在するデータルートを返す（見つからない場合は None）。"""
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um']:
                if (r / b).exists():
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return None


def _style_colors(n: int) -> List[str]:
    """プロジェクトスタイルの色サイクルから n 色を取得する。"""
    try:
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        cols = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
    return [cols[i % len(cols)] for i in range(n)]


# 貨物粒子（ビーズ）の直径と半径
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s"},
]
for _b, _c in zip(BEADS_INFO, _style_colors(len(BEADS_INFO))):
    _b["color"] = _c
    _b["label"] = rf"${_b['diameter_um']:.2f}\,\mu\mathrm{{m}}$"

BEAD_LOOKUP = {b['name']: b for b in BEADS_INFO}


def master_function(x: np.ndarray) -> np.ndarray:
    """理論マスターカーブ g(x) = (2 / x^2) [1 - (1 + x) e^{-x}]"""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    small_mask = (x < 1e-4)
    if np.any(small_mask):
        xs = x[small_mask]
        g[small_mask] = 1.0 - (2.0 / 3.0) * xs + (1.0 / 4.0) * (xs ** 2) - (1.0 / 15.0) * (xs ** 3)
    reg_mask = ~small_mask
    if np.any(reg_mask):
        xr_ = x[reg_mask]
        g[reg_mask] = (2.0 / (xr_ ** 2)) * (1.0 - (1.0 + xr_) * np.exp(-xr_))
    return g


def fit_instantaneous_xi_vectorized(
    r_um: np.ndarray,
    curves: np.ndarray,
    min_r: float,
    max_r: float = 25.0,
    min_corr_threshold: float = 0.05,
    min_points: int = 4,
    xi_min: float = 0.1,
    xi_max: float = 50.0,
) -> np.ndarray:
    """
    複数の空間配向相関曲線 C(r) に対して、縦軸の対数をとった線形回帰を一括で行い、
    各曲線の瞬時相関長 xi を返す。

    Parameters
    ----------
    r_um : np.ndarray, shape (n_r,)
        距離座標 [um]
    curves : np.ndarray, shape (n_r, n_curves)
        各 (粒子, フレーム) の空間配向相関 C(r)
    min_r, max_r : float
        フィッティングに使用する距離範囲 [um]
    min_corr_threshold : float
        対数をとるために必要な相関の最小閾値
    min_points : int
        フィッティングに必要な最小有効点数
    xi_min, xi_max : float
        有効とみなす相関長の範囲 [um]

    Returns
    -------
    xi : np.ndarray, shape (n_curves,)
        相関長 xi [um]（フィット不可の場合は NaN）
    """
    r_col = np.asarray(r_um, dtype=float)[:, None]
    curves = np.asarray(curves, dtype=float)

    # 縦軸の対数をとれる点のみを使用
    mask = (r_col >= min_r) & (r_col <= max_r) & np.isfinite(curves) & (curves >= min_corr_threshold)

    n_points = mask.sum(axis=0)
    log_c = np.log(np.where(mask, curves, 1.0))

    sw = mask.sum(axis=0).astype(float)
    sx = (r_col * mask).sum(axis=0)
    sxx = (r_col * r_col * mask).sum(axis=0)
    sy = (log_c * mask).sum(axis=0)
    sxy = (r_col * log_c * mask).sum(axis=0)

    delta = sw * sxx - sx * sx
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = (sw * sxy - sx * sy) / delta
        xi = -1.0 / slope

    valid = (
        (n_points >= min_points)
        & np.isfinite(slope)
        & (slope < -1e-6)
        & np.isfinite(xi)
        & (xi >= xi_min)
        & (xi <= xi_max)
    )
    return np.where(valid, xi, np.nan)


def extract_xi_velocity_pairs(
    root_dir: Path,
    beads_info: List[dict],
    scale: float = 0.11,
    tau: int = 1,
    frame_interval: float = 4.0,
    min_r_factor: float = 1.1,
    max_r: float = 25.0,
    min_corr_threshold: float = 0.05,
    min_points: int = 4,
    xi_min: float = 0.1,
    xi_max: float = 50.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    貨物粒子の直径ごとに、全実験・全粒子・全フレームの (xi_{i,t}, v_{i,t}) ペアを抽出する。

    Parameters
    ----------
    root_dir : Path
        データルートディレクトリ（beads06um 等のサブディレクトリを含む）
    beads_info : List[dict]
        各粒子径の情報（'name', 'diameter_um', 'radius_um'）

    Returns
    -------
    df : pd.DataFrame
        'bead_name', 'diameter_um', 'radius_um', 'exp_dir', 'particle', 'frame',
        'v_um_s', 'xi_um' を含む全 (粒子, フレーム) レコード
    df_progress : pd.DataFrame
        直径ごとの有効点数・棄却率の集計
    """
    records: List[dict] = []
    progress: List[dict] = []

    for binfo in beads_info:
        bname = binfo['name']
        dia = binfo['diameter_um']
        rc = binfo['radius_um']
        base = root_dir / bname

        if not base.exists():
            if verbose:
                print(f"[SKIP] {bname}: directory not found ({base})")
            continue

        edirs = [
            p for p in sorted(base.glob('*/*'))
            if p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / 'angular_correlation_w.zarr').exists()
        ]
        if not edirs:
            edirs = [
                p for p in sorted(base.glob('*'))
                if p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / 'angular_correlation_w.zarr').exists()
            ]
        if not edirs:
            if verbose:
                print(f"[SKIP] {bname}: no experiment dir with beads_tracks.csv + angular_correlation_w.zarr")
            continue

        min_r = rc * min_r_factor
        n_total = 0
        n_valid = 0

        for edir in edirs:
            try:
                df_tracks = pd.read_csv(edir / 'beads_tracks.csv')
            except Exception as e:
                print(f"[WARNING] Failed to read {edir / 'beads_tracks.csv'}: {e}")
                continue

            try:
                ds_w = xr.open_zarr(str(edir / 'angular_correlation_w.zarr'), consolidated=False)
            except Exception as e:
                print(f"[WARNING] Failed to open {edir / 'angular_correlation_w.zarr'}: {e}")
                continue

            if 'angular_correlation' not in ds_w:
                ds_w.close()
                continue

            # 次元順序はファイルごとに異なるため (distance, frame, particle) に正規化する
            da = ds_w['angular_correlation'].transpose('distance', 'frame', 'particle')
            arr = da.values
            r_um = ds_w.coords['distance'].values.astype(float) * scale
            frame_index = {int(f): i for i, f in enumerate(ds_w.coords['frame'].values)}
            particle_index = {int(p): i for i, p in enumerate(ds_w.coords['particle'].values)}
            ds_w.close()

            # 各粒子・各フレームの速度 v_{i,t} [um/s] を HMM 特徴量抽出と同一の定義で計算
            _, _, df_obs = hc.extract_hmm_features(
                df_tracks, tau=tau, scale=scale, frame_interval=frame_interval, epsilon=1e-3
            )
            if df_obs.empty:
                continue

            f_idx = np.array([frame_index.get(int(f), -1) for f in df_obs['frame'].values])
            p_idx = np.array([particle_index.get(int(p), -1) for p in df_obs['particle'].values])
            in_scope = (f_idx >= 0) & (p_idx >= 0)

            if not np.any(in_scope):
                continue

            n_total += int(np.count_nonzero(in_scope))

            curves = arr[:, f_idx[in_scope], p_idx[in_scope]]
            xi_vals = fit_instantaneous_xi_vectorized(
                r_um, curves,
                min_r=min_r, max_r=max_r,
                min_corr_threshold=min_corr_threshold,
                min_points=min_points, xi_min=xi_min, xi_max=xi_max,
            )

            v_vals = df_obs['v'].values[in_scope]
            frames = df_obs['frame'].values[in_scope]
            particles = df_obs['particle'].values[in_scope]

            finite = np.isfinite(xi_vals)
            n_valid += int(np.count_nonzero(finite))
            if not np.any(finite):
                continue

            for v_v, xi_v, fr_v, pt_v in zip(v_vals[finite], xi_vals[finite], frames[finite], particles[finite]):
                records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'radius_um': rc,
                    'exp_dir': edir.name,
                    'particle': int(pt_v),
                    'frame': int(fr_v),
                    'v_um_s': float(v_v),
                    'xi_um': float(xi_v),
                })

        progress.append({
            'bead_name': bname,
            'diameter_um': dia,
            'n_experiments': len(edirs),
            'n_frame_pairs': n_total,
            'n_valid_xi': n_valid,
            'valid_ratio': (n_valid / n_total) if n_total > 0 else np.nan,
        })
        if verbose:
            print(f"  {bname:10s} (d = {dia:5.2f} um): {len(edirs)} exp dirs, "
                  f"{n_valid:,} / {n_total:,} valid (xi_ij) points")

    return pd.DataFrame(records), pd.DataFrame(progress)


def compute_binned_profile(v: np.ndarray, xi: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    速度 v の分位点で等点数ビンに分割し、各ビンにおける xi の中央値・四分位範囲・平均を算出する。
    （速度分布が強く裾を引くため、等幅ビンではなく等点数ビンを用いる）
    """
    v = np.asarray(v, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if len(v) == 0 or n_bins < 1:
        return pd.DataFrame()

    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame()

    records = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            sel = (v >= lo) & (v <= hi)
        else:
            sel = (v >= lo) & (v < hi)
        n_pts = int(np.count_nonzero(sel))
        if n_pts < 5:
            continue
        xi_sub = xi[sel]
        records.append({
            'v_bin_low': lo,
            'v_bin_high': hi,
            'v_center_um_s': float(np.median(v[sel])),
            'v_mean_um_s': float(np.mean(v[sel])),
            'xi_median_um': float(np.median(xi_sub)),
            'xi_mean_um': float(np.mean(xi_sub)),
            'xi_q25_um': float(np.quantile(xi_sub, 0.25)),
            'xi_q75_um': float(np.quantile(xi_sub, 0.75)),
            'xi_sem_um': float(np.std(xi_sub, ddof=1) / np.sqrt(n_pts)) if n_pts > 1 else 0.0,
            'n_points': n_pts,
        })
    return pd.DataFrame(records)


def save_figure_to_all(fig, basename: str, out_dirs: List[Path]):
    """全出力ディレクトリに PNG と SVG を保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.png", dpi=300, bbox_inches='tight')
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            print(f"Saved: {d / basename}.png")
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def _compute_axis_limits(values: np.ndarray, scale: str, upper_percentile: float):
    """散布図の軸範囲（下限, 上限）をデータから決める。"""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0

    upper = float(np.quantile(vals, upper_percentile / 100.0)) * 1.05

    if scale == 'log':
        pos = vals[vals > 0]
        if len(pos) > 0:
            # 下限は低分位点（1 - upper_percentile）を基準にし、極端な最小値には引きずられないようにする
            low_q = max(0.0, (100.0 - upper_percentile) / 100.0)
            lower = max(float(np.quantile(pos, low_q)) * 0.5, float(np.min(pos)) * 0.8)
        else:
            lower = 1e-2
        if not np.isfinite(lower) or lower <= 0:
            lower = 1e-2
        if upper <= lower:
            upper = lower * 10.0
    else:
        lower = 0.0
        if upper <= 0:
            upper = float(np.max(vals)) * 1.05

    return lower, upper


def plot_single_bead(
    df_bead: pd.DataFrame,
    binfo: dict,
    out_dirs: List[Path],
    n_bins: int = 10,
    xscale: str = 'linear',
    yscale: str = 'linear',
    xmax_percentile: float = 99.5,
    ymax_percentile: float = 99.5,
    show_theory: bool = False,
    figsize: Tuple[float, float] = (6.8, 5.6),
) -> dict:
    """
    1つの粒子径について、全 (粒子 i, フレーム t) の (v_{i,t}, xi_{i,t}) を散布図に描画し、
    相関係数・線形回帰などの統計量を返す。
    """
    bname = binfo['name']
    dia = binfo['diameter_um']
    rc = binfo['radius_um']
    color = binfo.get('color', '#333333')
    marker = binfo.get('marker', 'o')

    v = df_bead['v_um_s'].to_numpy(dtype=float)
    xi = df_bead['xi_um'].to_numpy(dtype=float)
    finite = np.isfinite(v) & np.isfinite(xi)
    v, xi = v[finite], xi[finite]
    n_pts = int(len(v))

    fig, ax = plt.subplots(figsize=figsize)

    # 全 (i, t) の xi_{i,t} を散布（点数を最大化）
    ax.scatter(
        v, xi, s=7.0, alpha=0.18, marker=marker, color=color,
        edgecolors='none', rasterized=True, zorder=2,
        label=rf"All $\xi_{{i,t}}$ ($N = {n_pts:,}$)",
    )

    # 速度ビンごとの中央値 ± 四分位範囲（等点数ビン）
    df_bin = compute_binned_profile(v, xi, n_bins=n_bins)
    if not df_bin.empty:
        ax.fill_between(
            df_bin['v_center_um_s'], df_bin['xi_q25_um'], df_bin['xi_q75_um'],
            color='#111111', alpha=0.15, zorder=3,
        )
        ax.plot(
            df_bin['v_center_um_s'], df_bin['xi_median_um'], '-',
            color='#111111', linewidth=2.0, marker='o', markersize=5.0, zorder=4,
            label=r"Binned median ($\pm$IQR)",
        )

    # OLS 線形回帰と相関係数
    lr = stats.linregress(v, xi)
    pr = stats.pearsonr(v, xi)
    sr = stats.spearmanr(v, xi)
    v_line = np.linspace(float(np.min(v)), float(np.max(v)), 100)
    ax.plot(
        v_line, lr.intercept + lr.slope * v_line, '--', color=color, linewidth=1.8, alpha=0.95, zorder=5,
        label=rf"OLS: $\xi = ({lr.slope:+.3f} \pm {lr.stderr:.3f})\,v {lr.intercept:+.2f}$",
    )

    # 理論マスターカーブ（任意）: v_{i,t} = v_0 * g(R_c / xi_{i,t})
    v0_theory = np.nan
    if show_theory:
        g = master_function(rc / xi)
        denom = float(np.sum(g ** 2))
        if denom > 0:
            v0_theory = float(np.sum(v * g) / denom)
            order = np.argsort(xi)
            ax.plot(
                v0_theory * g[order], xi[order], '-',
                color='#117733', linewidth=2.2, zorder=6,
                label=rf"Theory $v = v_0\,g(R_c/\xi)$, $v_0 = {v0_theory:.3f}\,\mu\mathrm{{m/s}}$",
            )

    # 軸範囲・スケール
    x_lo, x_hi = _compute_axis_limits(v, xscale, xmax_percentile)
    y_lo, y_hi = _compute_axis_limits(xi, yscale, ymax_percentile)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(r"Cargo Velocity $v_{i,t}$ [$\mu\mathrm{m/s}$]", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"MT Correlation Length $\xi_{i,t}$ [$\mu\mathrm{m}$]", fontsize=12, fontweight='bold')
    ax.set_title(rf"$\xi_{{i,t}}$ vs $v_{{i,t}}$ ($d = {dia:.2f}\,\mu\mathrm{{m}}$)", fontsize=13, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

    stat_text = (
        f"$N = {n_pts:,}$\n"
        f"Pearson $r = {pr.statistic:+.3f}$ ($p = {pr.pvalue:.1e}$)\n"
        f"Spearman $\\rho = {sr.statistic:+.3f}$ ($p = {sr.pvalue:.1e}$)\n"
        f"median: $\\xi = {np.median(xi):.2f}\\,\\mu\\mathrm{{m}}$, $v = {np.median(v):.3f}\\,\\mu\\mathrm{{m/s}}$"
    )
    ax.text(
        0.03, 0.97, stat_text, transform=ax.transAxes, fontsize=8.8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.85, edgecolor='lightgray'),
    )

    plt.tight_layout()
    save_figure_to_all(fig, f"xi_vs_velocity_{bname}", out_dirs)
    plt.close(fig)

    return {
        'bead_name': bname,
        'diameter_um': dia,
        'radius_um': rc,
        'n_points': n_pts,
        'xi_median_um': float(np.median(xi)),
        'xi_mean_um': float(np.mean(xi)),
        'xi_std_um': float(np.std(xi, ddof=1)) if n_pts > 1 else np.nan,
        'v_median_um_s': float(np.median(v)),
        'v_mean_um_s': float(np.mean(v)),
        'pearson_r': float(pr.statistic),
        'pearson_p': float(pr.pvalue),
        'spearman_rho': float(sr.statistic),
        'spearman_p': float(sr.pvalue),
        'ols_slope_um_per_um_s': float(lr.slope),
        'ols_slope_err': float(lr.stderr),
        'ols_intercept_um': float(lr.intercept),
        'v0_theory_um_s': v0_theory,
    }


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]) -> None:
    """CSV を全出力ディレクトリに保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / f"{basename}.csv", index=False)
            print(f"Saved: {d / basename}.csv")
        except Exception as e:
            print(f"Warning: Failed to save {basename}.csv to {d}: {e}")


def plot_all_beads(
    df_all: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
    n_bins: int = 10,
    xscale: str = 'linear',
    yscale: str = 'linear',
    xmax_percentile: float = 99.5,
    ymax_percentile: float = 99.5,
    ncols: int = 2,
) -> None:
    """全粒子径の xi_{i,t} vs v_{i,t} 散布図を1枚のパネル図にまとめて描画する。"""
    n_plots = len(beads_info)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']
        color = binfo.get('color', '#333333')
        marker = binfo.get('marker', 'o')

        sub = df_all[df_all['bead_name'] == bname]
        if sub.empty:
            ax.set_title(rf"$d = {dia:.2f}\,\mu\mathrm{{m}}$ (No data)", fontsize=11.5)
            ax.axis('off')
            continue

        v = sub['v_um_s'].to_numpy(dtype=float)
        xi = sub['xi_um'].to_numpy(dtype=float)

        ax.scatter(v, xi, s=5.0, alpha=0.15, marker=marker, color=color,
                   edgecolors='none', rasterized=True, zorder=2, label=r"All $\xi_{i,t}$")

        df_bin = compute_binned_profile(v, xi, n_bins=n_bins)
        if not df_bin.empty:
            ax.plot(df_bin['v_center_um_s'], df_bin['xi_median_um'], '-', color='#111111',
                    linewidth=1.8, marker='o', markersize=4.0, zorder=4, label='Binned median')

        sr = stats.spearmanr(v, xi)
        ax.text(
            0.03, 0.97,
            f"$N = {len(v):,}$\nSpearman $\\rho = {sr.statistic:+.3f}$ ($p = {sr.pvalue:.1e}$)",
            transform=ax.transAxes, fontsize=8.0, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor='lightgray'),
        )

        x_lo, x_hi = _compute_axis_limits(v, xscale, xmax_percentile)
        y_lo, y_hi = _compute_axis_limits(xi, yscale, ymax_percentile)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"Cargo Velocity $v_{i,t}$ [$\mu\mathrm{m/s}$]", fontsize=10)
        ax.set_ylabel(r"$\xi_{i,t}$ [$\mu\mathrm{m}$]", fontsize=10)
        ax.set_title(rf"$d = {dia:.2f}\,\mu\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=8.0, loc='upper right', framealpha=0.9)

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        r"Instantaneous MT Correlation Length $\xi_{i,t}$ vs Cargo Velocity $v_{i,t}$",
        fontsize=14, fontweight='bold', y=1.0,
    )
    plt.tight_layout()
    save_figure_to_all(fig, 'xi_vs_velocity_all_beads', out_dirs)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot instantaneous local MT correlation length xi_{i,t} versus cargo velocity v_{i,t} for each cargo bead diameter."
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Data root directory containing beads06um, beads1um, ... (auto-detected if omitted)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory for figures/CSV (default: <workspace>/figure/xi_vs_velocity and <root_dir>/figure/xi_vs_velocity)")
    parser.add_argument('--beads', type=str, nargs='+', default=None, choices=list(BEAD_LOOKUP.keys()),
                        help="Bead directory names to process (default: all 6 diameters)")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale [um/pixel] (default: 0.11)")
    parser.add_argument('--tau', type=int, default=1, help="Lag time for the velocity estimation [frames] (default: 1)")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="Frame interval [s] (default: 4.0)")
    parser.add_argument('--min_r_factor', type=float, default=1.1,
                        help="min_r = R_c * factor used for the xi fit (default: 1.1)")
    parser.add_argument('--max_r', type=float, default=25.0,
                        help="Maximum distance used for the xi fit [um] (default: 25.0)")
    parser.add_argument('--min_corr_threshold', type=float, default=0.05,
                        help="Minimum C(r) used for the log-space fit (default: 0.05)")
    parser.add_argument('--min_points', type=int, default=4,
                        help="Minimum number of valid points required for the xi fit (default: 4)")
    parser.add_argument('--xi_min', type=float, default=0.1, help="Minimum accepted xi [um] (default: 0.1)")
    parser.add_argument('--xi_max', type=float, default=50.0, help="Maximum accepted xi [um] (default: 50.0)")
    parser.add_argument('--n_bins', type=int, default=10,
                        help="Number of equal-count velocity bins for the median trend (default: 10)")
    parser.add_argument('--xscale', type=str, default='linear', choices=['linear', 'log'],
                        help="X-axis (velocity) scale (default: linear)")
    parser.add_argument('--yscale', type=str, default='linear', choices=['linear', 'log'],
                        help="Y-axis (xi) scale (default: linear)")
    parser.add_argument('--xmax_percentile', type=float, default=99.5,
                        help="Upper percentile used for the x-axis range (default: 99.5)")
    parser.add_argument('--ymax_percentile', type=float, default=99.5,
                        help="Upper percentile used for the y-axis range (default: 99.5)")
    parser.add_argument('--show_theory', action='store_true',
                        help="Overlay the theoretical master curve v = v_0 * g(R_c / xi)")
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else find_default_root()
    if root_dir is None or not root_dir.exists():
        raise FileNotFoundError("Data root directory not found. Please specify it with --root_dir.")

    beads_info = [BEAD_LOOKUP[b] for b in args.beads] if args.beads else list(BEADS_INFO)
    if not beads_info:
        raise ValueError("No valid bead names were given (--beads).")

    if args.output_dir:
        out_dirs = [Path(args.output_dir).expanduser()]
    else:
        out_dirs = [
            CURRENT_DIR / 'figure' / 'xi_vs_velocity',
            root_dir / 'figure' / 'xi_vs_velocity',
        ]
    out_dirs = list(dict.fromkeys(out_dirs))

    print("=" * 72)
    print(" Instantaneous MT Correlation Length xi_{i,t} vs Cargo Velocity v_{i,t}")
    print("=" * 72)
    print(f"Data Root Directory : {root_dir}")
    print("Output Directories  : " + ", ".join(str(d) for d in out_dirs))
    print(f"xi fit   : r in [{args.min_r_factor} * R_c, {args.max_r}] um, "
          f"C(r) >= {args.min_corr_threshold}, min_points = {args.min_points}, "
          f"xi in [{args.xi_min}, {args.xi_max}] um")
    print(f"velocity : v = |dr| / ({args.tau} * {args.frame_interval}) um/s, scale = {args.scale} um/pixel")

    print()
    print("Extracting (xi_{i,t}, v_{i,t}) pairs ...")
    df_all, df_progress = extract_xi_velocity_pairs(
        root_dir=root_dir,
        beads_info=beads_info,
        scale=args.scale,
        tau=args.tau,
        frame_interval=args.frame_interval,
        min_r_factor=args.min_r_factor,
        max_r=args.max_r,
        min_corr_threshold=args.min_corr_threshold,
        min_points=args.min_points,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
    )
    if df_all.empty:
        raise RuntimeError("No valid (xi_{i,t}, v_{i,t}) points were extracted. Check --root_dir and the fit parameters.")

    print()
    print(f"Total extracted (i, t) points: {len(df_all):,}")

    # 直径ごとに CSV 保存 + 散布図作成
    stats_records = []
    for binfo in beads_info:
        sub = df_all[df_all['bead_name'] == binfo['name']].copy()
        if sub.empty:
            print(f"[SKIP] {binfo['name']}: no valid points")
            continue

        sub = sub.sort_values(['exp_dir', 'particle', 'frame']).reset_index(drop=True)
        save_csv_to_all(sub, f"xi_vs_velocity_{binfo['name']}", out_dirs)

        rec = plot_single_bead(
            sub, binfo, out_dirs,
            n_bins=args.n_bins,
            xscale=args.xscale,
            yscale=args.yscale,
            xmax_percentile=args.xmax_percentile,
            ymax_percentile=args.ymax_percentile,
            show_theory=args.show_theory,
        )
        stats_records.append(rec)
        print(f"  d = {rec['diameter_um']:5.2f} um : N = {rec['n_points']:6,}, "
              f"xi_med = {rec['xi_median_um']:6.2f} um, v_med = {rec['v_median_um_s']:.3f} um/s, "
              f"Pearson r = {rec['pearson_r']:+.3f} (p = {rec['pearson_p']:.1e}), "
              f"Spearman rho = {rec['spearman_rho']:+.3f} (p = {rec['spearman_p']:.1e})")

    if not stats_records:
        raise RuntimeError("No figures were generated.")

    # 統計量・抽出状況 CSV
    df_stats = pd.DataFrame(stats_records)
    save_csv_to_all(df_stats, 'xi_vs_velocity_statistics', out_dirs)
    if not df_progress.empty:
        save_csv_to_all(df_progress, 'xi_vs_velocity_extraction_summary', out_dirs)

    # 全直径一覧図
    plot_all_beads(
        df_all, beads_info, out_dirs,
        n_bins=args.n_bins,
        xscale=args.xscale,
        yscale=args.yscale,
        xmax_percentile=args.xmax_percentile,
        ymax_percentile=args.ymax_percentile,
    )

    print()
    print("=" * 72)
    print(" Summary statistics")
    print("=" * 72)
    cols = ['bead_name', 'diameter_um', 'n_points', 'xi_median_um', 'v_median_um_s',
            'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', 'ols_slope_um_per_um_s']
    print(df_stats[[c for c in cols if c in df_stats.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()




