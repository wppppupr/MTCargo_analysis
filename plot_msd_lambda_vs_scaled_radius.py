#!/usr/bin/env python3
"""
plot_msd_lambda_vs_scaled_radius.py

スケール半径 x = R_c / xi に対する、貨物粒子（ビーズ）の長時間輸送特性のスケーリング図を作成する。

   左縦軸 (第1軸) : MSD(300 s) = < dr^2(Delta t = 300 s) >   [um^2]
   右縦軸 (第2軸) : lambda(100 s)                            [um]
   横軸           : x = R_c / xi

【xi の算出（各実験ディレクトリ）】
- 微小管アクティブフローの空間配向相関: angular_correlation_w.zarr (distance, frame, particle)
- 各 (粒子 i, フレーム t) について縦軸の対数をとった線形回帰
      ln C(r) = ln a - r / xi
  を行い xi_{i,t} を得る（plot_xi_vs_velocity.py と同一条件）:
      min_r = R_c * --min_r_factor, max_r = --max_r, C(r) >= --min_corr_threshold,
      min_points = --min_points, xi in [--xi_min, --xi_max] um
- 実験代表値 xi_exp は全 (i, t) の中央値（--xi_source global の場合は --xi_value を使用）

【MSD(300 s) の算出（各実験ディレクトリ）】
- 軌跡: beads_tracks.csv (frame, particle, x, y)
- libs.displacement.imsd による個別粒子 MSD（時間平均）を求め、MSD.py と同一条件で
  alpha (> --alpha_threshold, フィット範囲 --alpha_min_t .. --alpha_max_t s) の粒子のみを採用
  （全粒子が棄却された場合は未フィルタにフォールバックし、CSV に alpha_filtered=False を記録）
- lag time = --msd_lag_s の値について粒子間アンサンブル平均をとる（誤差は粒子間 SEM）

【lambda(100 s) の算出（各実験ディレクトリ）】
- 変位 |dr(Delta t = 100 s)| を libs.displacement.calc_displacement_magnitudes で計算
- 変位 PDF を --bins ビン（0 .. --bin_max um）でヒストグラム化し、対数空間で
      P(r) = A exp(-r / lambda)
  をフィッティング（libs.displacement.fit_exponential_pdf）
- 指数 PDF では lambda ~ <|dr|> であるため、フィット値が
  [--lambda_min_ratio, --lambda_max_ratio] x <|dr|> を外れる場合は統計不足による発散とみなし
  棄却する（NaN, lambda_rejected = True。統計不足の実験で単一ビンが効いて発散するのを防ぐ）
- 参考として実験間平均 PDF から求めた pooled lambda を CSV に記録する（lambda_pooled_um 列）が、
  図には描画しない（情報過多を避けるため）。

【出力ファイル】
デフォルトの出力先は (1) figure/scaling, (2) figure, (3) <root_dir>/figure/scaling の3箇所
（データルート側へ保存しない場合は --no_save_root を指定）。
1. msd300_lambda100_vs_scaled_radius.png / .svg       : 2軸図（横軸 x = R_c/xi、左: MSD(300 s), 右: lambda(100 s)）
2. msd300_lambda100_vs_scaled_radius_2panel.png / .svg: 2パネル版（参考）
3. msd300_lambda100_vs_radius_rc.png / .svg           : 横軸 = R_c（linear）, 左軸 = MSD（黒, log）,
                                                        右軸 = lambda（赤, linear）の 2軸図
4. msd300_lambda100_vs_scaled_radius_points.csv       : 実験ごとの生データ
5. msd300_lambda100_vs_scaled_radius_summary.csv      : 粒子径ごとの代表値（mean +/- SEM）
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import displacement as dpm
from libs import cal_vel as cv
# xi_{i,t} のフィッティングは plot_xi_vs_velocity.py の実装を再利用する（記法の単一情報源）
from plot_xi_vs_velocity import fit_instantaneous_xi_vectorized

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


# 貨物粒子（ビーズ）の直径・半径・マーカー
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


def find_experiment_dirs(root_dir: Path, bead_name: str, zarr_name: str) -> List[Path]:
    """beads_tracks.csv と angular_correlation zarr を両方含む実験ディレクトリを探索する。"""
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []

    def _ok(p: Path) -> bool:
        return p.is_dir() and (p / 'beads_tracks.csv').exists() and (p / zarr_name).exists()

    edirs = [p for p in sorted(base.glob('*/*')) if _ok(p)]
    if not edirs:
        edirs = [p for p in sorted(base.glob('*')) if _ok(p)]
    return edirs


def fit_experiment_xi(
    exp_dir: Path,
    radius_um: float,
    zarr_name: str = 'angular_correlation_w.zarr',
    scale: float = 0.11,
    min_r_factor: float = 1.1,
    max_r: float = 25.0,
    min_corr_threshold: float = 0.05,
    min_points: int = 4,
    xi_min: float = 0.1,
    xi_max: float = 50.0,
) -> Dict[str, float]:
    """
    1つの実験ディレクトリについて、全 (粒子 i, フレーム t) の空間配向相関長 xi_{i,t} を求め、
    代表値（中央値・平均・SEM）を返す。

    Returns
    -------
    dict
        'xi_median_um', 'xi_mean_um', 'xi_std_um', 'xi_sem_um', 'n_xi_valid', 'n_xi_total'
    """
    out = {
        'xi_median_um': np.nan, 'xi_mean_um': np.nan, 'xi_std_um': np.nan,
        'xi_sem_um': np.nan, 'n_xi_valid': 0, 'n_xi_total': 0,
    }
    tracks_path = exp_dir / 'beads_tracks.csv'
    zarr_path = exp_dir / zarr_name
    try:
        df_tracks = pd.read_csv(tracks_path, usecols=['particle', 'frame'])
    except Exception as e:
        print(f"[WARNING] Failed to read {tracks_path}: {e}")
        return out

    try:
        ds = xr.open_zarr(str(zarr_path), consolidated=False)
    except Exception as e:
        print(f"[WARNING] Failed to open {zarr_path}: {e}")
        return out

    if 'angular_correlation' not in ds:
        ds.close()
        return out

    # 次元順序はファイルごとに異なるため (distance, frame, particle) に正規化する
    da = ds['angular_correlation'].transpose('distance', 'frame', 'particle')
    arr = da.values
    r_um = ds.coords['distance'].values.astype(float) * scale
    frame_index = {int(f): i for i, f in enumerate(ds.coords['frame'].values)}
    particle_index = {int(p): i for i, p in enumerate(ds.coords['particle'].values)}
    ds.close()

    # 軌跡データに存在し、かつ zarr にも存在する (粒子, フレーム) のみを使用
    f_idx = np.array([frame_index.get(int(f), -1) for f in df_tracks['frame'].values])
    p_idx = np.array([particle_index.get(int(p), -1) for p in df_tracks['particle'].values])
    in_scope = (f_idx >= 0) & (p_idx >= 0)
    n_total = int(np.count_nonzero(in_scope))
    if n_total == 0:
        return out

    curves = arr[:, f_idx[in_scope], p_idx[in_scope]]
    xi_vals = fit_instantaneous_xi_vectorized(
        r_um, curves,
        min_r=radius_um * min_r_factor, max_r=max_r,
        min_corr_threshold=min_corr_threshold,
        min_points=min_points, xi_min=xi_min, xi_max=xi_max,
    )
    xi_valid = xi_vals[np.isfinite(xi_vals)]
    n_valid = int(xi_valid.size)

    out['n_xi_total'] = n_total
    out['n_xi_valid'] = n_valid
    if n_valid > 0:
        out['xi_median_um'] = float(np.median(xi_valid))
        out['xi_mean_um'] = float(np.mean(xi_valid))
        out['xi_std_um'] = float(np.std(xi_valid, ddof=1)) if n_valid > 1 else 0.0
        out['xi_sem_um'] = float(out['xi_std_um'] / np.sqrt(n_valid)) if n_valid > 1 else 0.0
    return out


def compute_msd_at_lag(
    df_tracks: pd.DataFrame,
    lag_s: float = 300.0,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    alpha_threshold: float = 0.5,
    alpha_min_t: float = 4.0,
    alpha_max_t: float = 300.0,
) -> Dict[str, float]:
    """
    1つの実験ディレクトリの MSD(Delta t = lag_s) [um^2] を、MSD.py と同一の手順で算出する。

    - libs.displacement.imsd による個別粒子 MSD（時間平均）
    - MSD.calc_particle_alpha により alpha を算出し、alpha > alpha_threshold の粒子のみ採用
    - lag time = lag_s における粒子間アンサンブル平均（誤差は粒子間 SEM）

    Returns
    -------
    dict
        'msd_um2', 'msd_median_um2', 'msd_std_um2', 'msd_sem_um2',
        'n_particles_msd', 'n_particles_all', 'msd_lag_s_actual', 'alpha_filtered'
    """
    out = {
        'msd_um2': np.nan, 'msd_median_um2': np.nan, 'msd_std_um2': np.nan,
        'msd_sem_um2': np.nan, 'n_particles_msd': 0, 'n_particles_all': 0,
        'msd_lag_s_actual': np.nan, 'alpha_filtered': False,
    }
    if df_tracks is None or df_tracks.empty:
        return out

    # MSD.py の alpha 定義を再利用（循環 import 回避のため遅延 import）
    from MSD import calc_particle_alpha

    try:
        track = cv.cal(df_tracks.copy(), scale=scale, frame_interval=frame_interval)
        imsd_df = dpm.imsd(track, scale=scale, time_scale=frame_interval, display=False)
    except Exception as e:
        print(f"[WARNING] MSD calculation failed: {e}")
        return out

    if imsd_df is None or imsd_df.empty:
        return out

    imsd_df = imsd_df.dropna(subset=['MSD', 'lag time'])
    out['n_particles_all'] = int(imsd_df['particle'].nunique())
    if imsd_df.empty:
        return out

    # --- alpha フィルタ（MSD.py と同じ: 4 .. 300 s の iMSD べき乗則指数） ---
    alphas = {
        pid: calc_particle_alpha(grp, alpha_min_t, alpha_max_t)
        for pid, grp in imsd_df.groupby('particle')
    }
    imsd_df = imsd_df.assign(alpha=imsd_df['particle'].map(alphas))
    if alpha_threshold is not None and np.isfinite(alpha_threshold):
        filt = imsd_df[imsd_df['alpha'] > alpha_threshold]
        if not filt.empty:
            imsd_df = filt
            out['alpha_filtered'] = True

    # --- 指定ラグタイムに最も近い（<= lag_s 側を優先した）実ラグを選択 ---
    lag_vals = np.unique(imsd_df['lag time'].to_numpy(dtype=float))
    lag_vals = lag_vals[np.isfinite(lag_vals)]
    if lag_vals.size == 0:
        return out

    exact = lag_vals[np.isclose(lag_vals, lag_s, atol=1e-6)]
    if exact.size > 0:
        lag_use = float(np.min(exact))
    else:
        below = lag_vals[lag_vals <= lag_s]
        lag_use = float(np.max(below)) if below.size > 0 else float(np.min(lag_vals))

    sub = imsd_df[np.isclose(imsd_df['lag time'].to_numpy(dtype=float), lag_use, atol=1e-6)]
    per_particle = sub.groupby('particle')['MSD'].mean().to_numpy(dtype=float)
    per_particle = per_particle[np.isfinite(per_particle) & (per_particle > 0)]
    n_pts = int(per_particle.size)
    if n_pts == 0:
        return out

    out['msd_um2'] = float(np.mean(per_particle))
    out['msd_median_um2'] = float(np.median(per_particle))
    out['msd_std_um2'] = float(np.std(per_particle, ddof=1)) if n_pts > 1 else 0.0
    out['msd_sem_um2'] = float(out['msd_std_um2'] / np.sqrt(n_pts)) if n_pts > 1 else 0.0
    out['n_particles_msd'] = n_pts
    out['msd_lag_s_actual'] = lag_use
    return out


def compute_lambda_at_lag(
    df_tracks: pd.DataFrame,
    lag_s: float = 100.0,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    component: str = 'norm',
    bins: int = 25,
    bin_max: float = 50.0,
    fit_mode: str = 'log',
    fit_rmin: Optional[float] = None,
    fit_rmax: Optional[float] = None,
    lambda_min_ratio: float = 0.1,
    lambda_max_ratio: float = 10.0,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    1つの実験ディレクトリの変位 PDF の指数減衰長 lambda(Delta t = lag_s) [um] を算出する。

    P(|dr|) = A exp(-|dr| / lambda) を対数空間でフィッティング
    （libs.displacement.fit_exponential_pdf）。

    指数 PDF では lambda は平均変位 <|dr|> と一致するため、フィット結果が
    [lambda_min_ratio, lambda_max_ratio] x <|dr|> の範囲外となる場合は
    ビン数不足・統計不足による発散とみなして棄却（NaN）する。

    Returns
    -------
    dict
        'lambda_um', 'lambda_err_um', 'lambda_r2_log', 'lambda_A',
        'lambda_fit_ratio', 'lambda_rejected', 'n_displacements', 'mean_disp_um',
        'tau_frames'
    np.ndarray or None
        条件代表値（実験間平均 PDF のフィット）用に返す変位配列
    """
    out = {
        'lambda_um': np.nan, 'lambda_err_um': np.nan, 'lambda_r2_log': np.nan,
        'lambda_A': np.nan, 'lambda_fit_ratio': np.nan, 'lambda_rejected': False,
        'n_displacements': 0, 'mean_disp_um': np.nan, 'tau_frames': 0,
    }
    if df_tracks is None or df_tracks.empty:
        return out, None

    tau = int(round(lag_s / frame_interval))
    out['tau_frames'] = tau
    if tau < 1:
        return out, None

    try:
        disp = dpm.calc_displacement_magnitudes(
            df_tracks, tau=tau, scale=scale, component=component
        )
    except Exception as e:
        print(f"[WARNING] Displacement calculation failed: {e}")
        return out, None

    disp = np.asarray(disp, dtype=float)
    disp = disp[np.isfinite(disp)]
    n_disp = int(disp.size)
    out['n_displacements'] = n_disp
    if n_disp < 10:
        return out, None
    mean_disp = float(np.mean(disp))
    out['mean_disp_um'] = mean_disp

    centers, pdf, _ = dpm.calc_displacement_pdf(
        disp, bins=bins, density=True, bin_range=(0.0, bin_max)
    )
    valid = np.isfinite(pdf) & (pdf > 0) & (centers > 0)
    if np.count_nonzero(valid) >= 3:
        fit = dpm.fit_exponential_pdf(
            centers[valid], pdf[valid],
            r_min=fit_rmin, r_max=fit_rmax, fit_mode=fit_mode,
        )
        if fit is not None:
            lam = float(fit['lambda'])
            ratio = lam / mean_disp if mean_disp > 0 else np.inf
            out['lambda_fit_ratio'] = float(ratio) if np.isfinite(ratio) else np.nan
            if (np.isfinite(lam) and lam > 0 and mean_disp > 0
                    and (lambda_min_ratio * mean_disp) <= lam <= (lambda_max_ratio * mean_disp)):
                out['lambda_um'] = lam
                out['lambda_err_um'] = float(fit['lambda_err'])
                out['lambda_r2_log'] = float(fit['r_squared'])
                out['lambda_A'] = float(fit['A'])
            else:
                out['lambda_rejected'] = True
                out['lambda_r2_log'] = float(fit['r_squared'])
                print(f"[WARNING] lambda fit rejected (lambda = {lam:.3g} um, "
                      f"mean |dr| = {mean_disp:.3g} um, N_disp = {n_disp}): outside "
                      f"[{lambda_min_ratio}, {lambda_max_ratio}] x mean displacement.")

    return out, disp


def fit_pooled_lambda(
    exp_disp_list: Sequence[np.ndarray],
    bins: int = 25,
    bin_max: float = 50.0,
    fit_mode: str = 'log',
    fit_rmin: Optional[float] = None,
    fit_rmax: Optional[float] = None,
) -> Dict[str, float]:
    """
    粒子径（条件）ごとに、実験間で平均した変位 PDF から lambda を求める。

    displacement_analysis.py の calc_ensemble_pdf + 指数フィットと同じ手順
    （実験ごとの PDF を等重みで平均してからフィット）であり、条件代表値として CSV に記録する。
    """
    out = {
        'lambda_pooled_um': np.nan, 'lambda_pooled_err_um': np.nan,
        'lambda_pooled_r2_log': np.nan, 'n_experiments_lambda': 0,
    }
    arrays = [a for a in (exp_disp_list or []) if a is not None]
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size > 0]
    if not arrays:
        return out

    all_data = np.concatenate(arrays)
    edges = np.histogram_bin_edges(all_data, bins=bins, range=(0.0, bin_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdfs = [np.histogram(a, bins=edges, density=True)[0] for a in arrays]
    mean_pdf = np.nanmean(np.array(pdfs), axis=0)

    valid = np.isfinite(mean_pdf) & (mean_pdf > 0) & (centers > 0)
    if np.count_nonzero(valid) < 3:
        return out

    fit = dpm.fit_exponential_pdf(
        centers[valid], mean_pdf[valid],
        r_min=fit_rmin, r_max=fit_rmax, fit_mode=fit_mode,
    )
    if fit is None:
        return out

    out['lambda_pooled_um'] = float(fit['lambda'])
    out['lambda_pooled_err_um'] = float(fit['lambda_err'])
    out['lambda_pooled_r2_log'] = float(fit['r_squared'])
    out['n_experiments_lambda'] = int(len(arrays))
    return out


def summarize_by_condition(
    df_points: pd.DataFrame,
    pooled_lambda_map: Dict[str, dict],
    beads_info: List[dict],
) -> pd.DataFrame:
    """
    粒子径ごとに、横軸 x = R_c / xi・MSD(300 s)・lambda(100 s) の代表値を
    実験間の mean +/- SEM として集計する（横軸 x が有限な実験のみを対象）。
    """
    rows: List[dict] = []
    if df_points is None or df_points.empty:
        return pd.DataFrame(rows)

    def _ms(values: np.ndarray) -> Tuple[float, float, int]:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            return np.nan, np.nan, 0
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return mean, sem, n

    for binfo in beads_info:
        sub = df_points[df_points['bead_name'] == binfo['name']]
        if sub.empty:
            continue

        x = sub['rc_over_xi'].to_numpy(dtype=float)
        finite_x = np.isfinite(x)

        msd = np.where(finite_x, sub['msd_um2'].to_numpy(dtype=float), np.nan)
        lam = np.where(finite_x, sub['lambda_um'].to_numpy(dtype=float), np.nan)

        x_mean, x_sem, n_x = _ms(x)
        msd_mean, msd_sem, n_msd = _ms(msd)
        lam_mean, lam_sem, n_lam = _ms(lam)

        rec = {
            'bead_name': binfo['name'],
            'diameter_um': binfo['diameter_um'],
            'radius_um': binfo['radius_um'],
            'n_experiments': int(len(sub)),
            'n_experiments_with_x': n_x,
            'rc_over_xi_mean': x_mean,
            'rc_over_xi_sem': x_sem,
            'xi_um_mean': _ms(sub['xi_um'].to_numpy(dtype=float))[0],
            'msd_um2_mean': msd_mean,
            'msd_um2_sem': msd_sem,
            'n_experiments_msd': n_msd,
            'lambda_um_mean': lam_mean,
            'lambda_um_sem': lam_sem,
            'n_experiments_lambda': n_lam,
        }
        rec.update(pooled_lambda_map.get(binfo['name'], {}))
        rows.append(rec)

    return pd.DataFrame(rows)


def _axis_limits(
    values: np.ndarray,
    scale: str = 'log',
    upper_percentile: float = 100.0,
    log_pad: float = 1.6,
    linear_pad: float = 1.10,
) -> Tuple[float, float]:
    """プロット軸の範囲 (下限, 上限) を有限値から決める。"""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    if scale == 'log':
        pos = vals[vals > 0]
        if pos.size == 0:
            return 1e-3, 1.0
        lower = float(np.min(pos)) / log_pad
        upper = float(np.quantile(pos, upper_percentile / 100.0)) * log_pad
        if not np.isfinite(lower) or lower <= 0:
            lower = float(np.min(pos)) * 0.5
        if upper <= lower:
            upper = lower * 10.0
    else:
        lower = min(0.0, float(np.min(vals)))
        upper = float(np.quantile(vals, upper_percentile / 100.0)) * linear_pad
        if upper <= lower:
            upper = lower + 1.0
    return lower, upper


def save_figure_to_all(fig, basename: str, out_dirs: List[Path]) -> None:
    """全出力ディレクトリに PNG と SVG を保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.png", dpi=300, bbox_inches='tight')
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            print(f"Saved: {d / basename}.png")
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]) -> None:
    """CSV を全出力ディレクトリに保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / f"{basename}.csv", index=False)
            print(f"Saved: {d / basename}.csv")
        except Exception as e:
            print(f"Warning: Failed to save {basename}.csv to {d}: {e}")


def _corr_stats(x, y) -> Tuple[float, float, int]:
    """log10 空間での Spearman rho / Pearson r と有効点数を返す。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n = int(np.count_nonzero(mask))
    if n < 3:
        return np.nan, np.nan, n
    sp = stats.spearmanr(np.log10(x[mask]), np.log10(y[mask]))
    pr = stats.pearsonr(np.log10(x[mask]), np.log10(y[mask]))
    return float(sp.statistic), float(pr.statistic), n


def _safe_sem(value) -> float:
    """NaN を 0 に置き換えた誤差（matplotlib 用）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def plot_scaling_twin_axis(
    df_pts: pd.DataFrame,
    df_cond: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
    xscale: str = 'log',
    yscale: str = 'log',
    msd_lag_s: float = 300.0,
    lambda_lag_s: float = 100.0,
    show_raw: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 6.4),
    x_col: str = 'rc_over_xi',
    x_col_cond: Optional[str] = 'rc_over_xi_mean',
    x_err_col: Optional[str] = 'rc_over_xi_sem',
    xlabel: Optional[str] = None,
    x_label_symbol: str = 'x',
    msd_color: str = 'bead',
    lambda_color: str = 'bead',
    msd_yscale: Optional[str] = None,
    lambda_yscale: Optional[str] = None,
    x_jitter: float = 0.0,
    basename: str = 'msd300_lambda100_vs_scaled_radius',
    legend_title_left: Optional[str] = None,
    legend_title_right: Optional[str] = None,
) -> Dict[str, float]:
    """
    第1縦軸 (左) = MSD(msd_lag_s) [um^2]、第2縦軸 (右) = lambda(lambda_lag_s) [um] の
    2軸スケーリング図を作成する（横軸は既定で x = R_c / xi、x_col で変更可能）。

    Layer 1: 実験ごとの生データ（左軸は塗りつぶしマーカー、右軸は白抜きマーカー）
    Layer 2: 粒子径ごとの代表値（Mean +/- SEM、黒縁マーカー）

    Parameters
    ----------
    x_col : str
        横軸に用いる生データ列（例: 'rc_over_xi', 'radius_um'）
    x_col_cond : str or None
        横軸の条件代表値列（'radius_um' のように実験間で共通の場合は生データ列と同じでよい）
    x_err_col : str or None
        横軸の誤差列（None の場合は横方向のエラーバーを描画しない）
    msd_color, lambda_color : str
        系列の色。'bead' で粒子径ごとの色分け、色文字列（例: 'black', '#d62728'）で単色。
        単色指定時は対応する縦軸（目盛・ラベル・スパイン）も同じ色に揃える。
    msd_yscale, lambda_yscale : str or None
        左右の縦軸スケール（None の場合は yscale を使用）
    x_jitter : float
        生データ点の横方向ジッタ（相対値, 0 で無効。同一 R_c に重なる実験点の分離用）
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    msd_color_mode = (msd_color == 'bead')
    lam_color_mode = (lambda_color == 'bead')
    msd_axis_color = '#000000' if msd_color_mode else msd_color
    lam_axis_color = '#000000' if lam_color_mode else lambda_color
    yscale_msd = msd_yscale if msd_yscale is not None else yscale
    yscale_lam = lambda_yscale if lambda_yscale is not None else yscale

    rng = np.random.default_rng(42)
    msd_handles: List = []
    lam_handles: List = []
    x_all, msd_all, lam_all = [], [], []
    n_exp_raw = 0

    for binfo in beads_info:
        sub = df_pts[df_pts['bead_name'] == binfo['name']]
        if sub.empty:
            continue
        color = binfo.get('color', '#333333')
        marker = binfo.get('marker', 'o')
        dia = binfo['diameter_um']
        msd_c = color if msd_color_mode else msd_color
        lam_c = color if lam_color_mode else lambda_color

        # 生データ点用の横方向ジッタ（同一 R_c に重なる実験点を分離）
        x_raw_all = sub[x_col].to_numpy(dtype=float)
        if x_jitter > 0:
            x_raw_all = x_raw_all * (1.0 + x_jitter * rng.normal(0.0, 1.0, size=x_raw_all.size))
        sub = sub.assign(_x_jit=x_raw_all)

        # --- Layer 1: 実験ごとの生データ ---
        if show_raw:
            s_msd = sub[np.isfinite(sub[x_col]) & np.isfinite(sub['msd_um2'])]
            if not s_msd.empty:
                ax.scatter(
                    s_msd['_x_jit'], s_msd['msd_um2'], s=34, marker=marker,
                    color=msd_c, alpha=0.45, edgecolors='none', zorder=2,
                )
                n_exp_raw += int(len(s_msd))
            s_lam = sub[np.isfinite(sub[x_col]) & np.isfinite(sub['lambda_um'])]
            if not s_lam.empty:
                if lam_color_mode:
                    # 色分けモードでは右軸側は白抜きマーカーで区別する
                    ax2.scatter(
                        s_lam['_x_jit'], s_lam['lambda_um'], s=40, marker=marker,
                        facecolors='none', edgecolors=lam_c, alpha=0.75,
                        linewidths=1.2, zorder=2,
                    )
                else:
                    ax2.scatter(
                        s_lam['_x_jit'], s_lam['lambda_um'], s=34, marker=marker,
                        color=lam_c, alpha=0.45, edgecolors='none', zorder=2,
                    )

        x_all.extend(sub[x_col].to_numpy(dtype=float))
        msd_all.extend(sub['msd_um2'].to_numpy(dtype=float))
        lam_all.extend(sub['lambda_um'].to_numpy(dtype=float))

        # --- Layer 2: 粒子径ごとの代表値 (Mean +/- SEM) ---
        if x_col_cond is None or x_col_cond not in df_cond.columns:
            continue
        row = df_cond[df_cond['bead_name'] == binfo['name']]
        if row.empty:
            continue
        row = row.iloc[0]
        x_val = float(row[x_col_cond])
        if not np.isfinite(x_val):
            continue
        x_err = _safe_sem(row[x_err_col]) if (x_err_col and x_err_col in row.index) else 0.0
        if x_label_symbol == 'x':
            label = rf"$2R_c = {dia:.2f}\,\mu\mathrm{{m}}$ ($x = {x_val:.2f}$)"
        else:
            label = (rf"$2R_c = {dia:.2f}\,\mu\mathrm{{m}}$ "
                     rf"(${x_label_symbol} = {x_val:.2f}\,\mu\mathrm{{m}}$)")

        msd_val = float(row['msd_um2_mean'])
        if np.isfinite(msd_val):
            msd_err = _safe_sem(row['msd_um2_sem'])
            ax.errorbar(
                x_val, msd_val,
                xerr=x_err if x_err > 0 else None,
                yerr=msd_err if msd_err > 0 else None,
                fmt=marker, color=msd_c, markerfacecolor=msd_c,
                markeredgecolor='black', markeredgewidth=1.1, markersize=10.5,
                ecolor='#333333', elinewidth=1.4, capsize=4.0, capthick=1.2, zorder=5,
            )
        msd_handles.append(
            Line2D([], [], marker=marker, linestyle='none', color=msd_c,
                   markerfacecolor=msd_c, markeredgecolor='black',
                   markeredgewidth=1.1, markersize=9.5, label=label)
        )

        lam_val = float(row['lambda_um_mean'])
        if lam_color_mode:
            lam_face, lam_edge = 'none', lam_c
        else:
            lam_face, lam_edge = lam_c, 'black'
        if np.isfinite(lam_val):
            lam_err = _safe_sem(row['lambda_um_sem'])
            ax2.errorbar(
                x_val, lam_val,
                xerr=x_err if x_err > 0 else None,
                yerr=lam_err if lam_err > 0 else None,
                fmt=marker, color=lam_c, markerfacecolor=lam_face,
                markeredgecolor=lam_edge, markeredgewidth=1.6, markersize=10.5,
                ecolor=lam_c, elinewidth=1.2, capsize=4.0, capthick=1.2, zorder=5,
            )
        lam_handles.append(
            Line2D([], [], marker=marker, linestyle='none', color=lam_c,
                   markerfacecolor=lam_face, markeredgecolor=lam_edge,
                   markeredgewidth=1.6, markersize=9.5, label=label)
        )

    # --- 軸設定 ---
    ax.set_xscale(xscale)
    ax.set_yscale(yscale_msd)
    ax2.set_yscale(yscale_lam)

    x_lo, x_hi = _axis_limits(np.asarray(x_all, dtype=float), xscale, 100.0, 1.6)
    msd_lo, msd_hi = _axis_limits(np.asarray(msd_all, dtype=float), yscale_msd, 100.0, 1.9)
    lam_lo, lam_hi = _axis_limits(np.asarray(lam_all, dtype=float), yscale_lam, 100.0, 1.9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(msd_lo, msd_hi)
    ax2.set_ylim(lam_lo, lam_hi)

    if xlabel is None:
        xlabel = r"Scaled Cargo Radius $x = R_c / \xi$"
    ax.set_xlabel(xlabel, fontsize=12.5, fontweight='bold')
    ax.set_ylabel(
        rf"MSD $\langle \Delta r^2(\Delta t = {msd_lag_s:.0f}\,\mathrm{{s}}) \rangle$"
        rf" [$\mu\mathrm{{m}}^2$]",
        fontsize=12.5, fontweight='bold',
    )
    ax2.set_ylabel(
        rf"Displacement Decay Length $\lambda(\Delta t = {lambda_lag_s:.0f}\,\mathrm{{s}})$"
        rf" [$\mu\mathrm{{m}}$]",
        fontsize=12.5, fontweight='bold',
    )

    # 縦軸（目盛・ラベル・スパイン）の色を系列色に合わせる
    ax.tick_params(axis='y', colors=msd_axis_color)
    ax.yaxis.label.set_color(msd_axis_color)
    ax.spines['left'].set_color(msd_axis_color)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_visible(False)
    ax2.tick_params(axis='y', colors=lam_axis_color)
    ax2.yaxis.label.set_color(lam_axis_color)
    ax2.spines['right'].set_color(lam_axis_color)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    if title is None:
        title = (rf"MSD$(\Delta t = {msd_lag_s:.0f}\,\mathrm{{s}})$ and "
                 rf"$\lambda(\Delta t = {lambda_lag_s:.0f}\,\mathrm{{s}})$ vs $x = R_c/\xi$")
    ax.set_title(title, fontsize=13.5, fontweight='bold', pad=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.35)

    # --- 統計量（横軸・縦軸とも対数変換可能な正値のみを使用） ---
    sp_msd, pr_msd, n_msd_pts = _corr_stats(x_all, msd_all)
    sp_lam, pr_lam, n_lam_pts = _corr_stats(x_all, lam_all)
    sp_msd_c, _, n_cond = _corr_stats(df_cond[x_col_cond], df_cond['msd_um2_mean'])
    sp_lam_c, _, _ = _corr_stats(df_cond[x_col_cond], df_cond['lambda_um_mean'])

    stats_text = (
        rf"$N_{{\mathrm{{exp}}}} = {n_msd_pts}$ (MSD), {n_lam_pts} ($\lambda$)"
        + (rf"; $N_{{\mathrm{{cond}}}} = {n_cond}$" if n_cond >= 3 else "") + "\n"
    )
    rho_msd_e = f"{sp_msd:+.2f}" if np.isfinite(sp_msd) else "n/a"
    rho_lam_e = f"{sp_lam:+.2f}" if np.isfinite(sp_lam) else "n/a"
    if n_cond >= 3:
        stats_text += (
            rf"MSD: Spearman $\rho = {rho_msd_e}$ (exp), ${sp_msd_c:+.2f}$ (cond)" + "\n"
            rf"$\lambda$: Spearman $\rho = {rho_lam_e}$ (exp), ${sp_lam_c:+.2f}$ (cond)"
        )
    else:
        stats_text += (
            rf"MSD: Spearman $\rho = {rho_msd_e}$ (exp)" + "\n"
            rf"$\lambda$: Spearman $\rho = {rho_lam_e}$ (exp)"
        )
    ax.text(
        0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=8.8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.88, edgecolor='lightgray'),
        zorder=8,
    )

    # --- 凡例（左軸: MSD、右軸: lambda） ---
    if show_raw:
        msd_handles.insert(0, Line2D(
            [], [], marker='o', linestyle='none',
            color='#666666' if msd_color_mode else msd_axis_color, alpha=0.6,
            markersize=6.0, label=rf"Individual experiments ($N = {n_exp_raw}$)",
        ))
    if msd_handles:
        leg1 = ax.legend(
            handles=msd_handles, loc='upper right', fontsize=8.2, frameon=True,
            framealpha=0.92, labelspacing=0.30,
            title=(legend_title_left if legend_title_left is not None
                   else rf"MSD$(\Delta t = {msd_lag_s:.0f}\,\mathrm{{s}})$"),
            title_fontsize=9.0,
        )
        if not msd_color_mode:
            for _t in leg1.get_texts():
                _t.set_color(msd_axis_color)
            if leg1.get_title() is not None:
                leg1.get_title().set_color(msd_axis_color)
        leg1.set_zorder(9)
        ax.add_artist(leg1)
    if lam_handles:
        leg2 = ax2.legend(
            handles=lam_handles, loc='lower left', fontsize=8.2, frameon=True,
            framealpha=0.92, labelspacing=0.30,
            title=(legend_title_right if legend_title_right is not None
                   else rf"$\lambda(\Delta t = {lambda_lag_s:.0f}\,\mathrm{{s}})$"),
            title_fontsize=9.0,
        )
        if not lam_color_mode:
            for _t in leg2.get_texts():
                _t.set_color(lam_axis_color)
            if leg2.get_title() is not None:
                leg2.get_title().set_color(lam_axis_color)
        leg2.set_zorder(9)

    plt.tight_layout()
    save_figure_to_all(fig, basename, out_dirs)
    plt.close(fig)

    return {
        'n_points_msd': n_msd_pts,
        'n_points_lambda': n_lam_pts,
        'n_conditions': n_cond,
        'spearman_msd_exp': sp_msd,
        'pearson_msd_exp': pr_msd,
        'spearman_lambda_exp': sp_lam,
        'pearson_lambda_exp': pr_lam,
        'spearman_msd_cond': sp_msd_c,
        'spearman_lambda_cond': sp_lam_c,
    }


def plot_scaling_two_panel(
    df_pts: pd.DataFrame,
    df_cond: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
    xscale: str = 'log',
    yscale: str = 'log',
    msd_lag_s: float = 300.0,
    lambda_lag_s: float = 100.0,
    show_raw: bool = True,
    figsize: Tuple[float, float] = (13.0, 5.6),
) -> None:
    """参考用: (a) MSD(msd_lag_s) vs x、(b) lambda(lambda_lag_s) vs x の2パネル版を作成する。"""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    panels = [
        {
            'ax': axes[0], 'raw': 'msd_um2', 'mean': 'msd_um2_mean', 'sem': 'msd_um2_sem',
            'ylabel': (rf"MSD $\langle \Delta r^2(\Delta t = {msd_lag_s:.0f}\,\mathrm{{s}}) \rangle$"
                       rf" [$\mu\mathrm{{m}}^2$]"),
            'title': rf"$\mathbf{{(a)}}$ MSD$(\Delta t = {msd_lag_s:.0f}\,\mathrm{{s}})$ vs $x = R_c/\xi$",
        },
        {
            'ax': axes[1], 'raw': 'lambda_um', 'mean': 'lambda_um_mean', 'sem': 'lambda_um_sem',
            'ylabel': (rf"Displacement Decay Length $\lambda(\Delta t = {lambda_lag_s:.0f}\,\mathrm{{s}})$"
                       rf" [$\mu\mathrm{{m}}$]"),
            'title': rf"$\mathbf{{(b)}}$ $\lambda(\Delta t = {lambda_lag_s:.0f}\,\mathrm{{s}})$ vs $x = R_c/\xi$",
        },
    ]

    for panel in panels:
        ax = panel['ax']
        handles: List = []
        x_rep, y_rep = [], []
        x_all, y_all = [], []

        for binfo in beads_info:
            sub = df_pts[df_pts['bead_name'] == binfo['name']]
            if sub.empty:
                continue
            color = binfo.get('color', '#333333')
            marker = binfo.get('marker', 'o')
            dia = binfo['diameter_um']

            if show_raw:
                s_raw = sub[np.isfinite(sub['rc_over_xi']) & np.isfinite(sub[panel['raw']])]
                if not s_raw.empty:
                    ax.scatter(
                        s_raw['rc_over_xi'], s_raw[panel['raw']], s=38, marker=marker,
                        color=color, alpha=0.40, edgecolors='none', zorder=2,
                    )
            x_all.extend(sub['rc_over_xi'].to_numpy(dtype=float))
            y_all.extend(sub[panel['raw']].to_numpy(dtype=float))

            row = df_cond[df_cond['bead_name'] == binfo['name']]
            if row.empty:
                continue
            row = row.iloc[0]
            x_val = float(row['rc_over_xi_mean'])
            y_val = float(row[panel['mean']])
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            x_err = _safe_sem(row['rc_over_xi_sem'])
            y_err = _safe_sem(row[panel['sem']])
            label = rf"$2R_c = {dia:.2f}\,\mu\mathrm{{m}}$ ($x = {x_val:.2f}$)"

            ax.errorbar(
                x_val, y_val,
                xerr=x_err if x_err > 0 else None,
                yerr=y_err if y_err > 0 else None,
                fmt=marker, color=color, markerfacecolor=color,
                markeredgecolor='black', markeredgewidth=1.1, markersize=10.5,
                ecolor='#333333', elinewidth=1.4, capsize=4.0, capthick=1.2, zorder=5,
            )
            handles.append(
                Line2D([], [], marker=marker, linestyle='none', color=color,
                       markerfacecolor=color, markeredgecolor='black',
                       markeredgewidth=1.1, markersize=9.5, label=label)
            )
            x_rep.append(x_val)
            y_rep.append(y_val)
        # 代表値の目安ライン（x 昇順）
        if len(x_rep) > 1:
            order = np.argsort(np.asarray(x_rep, dtype=float))
            ax.plot(
                np.asarray(x_rep, dtype=float)[order], np.asarray(y_rep, dtype=float)[order],
                linestyle='--', color='#777777', linewidth=1.4, alpha=0.8, zorder=4,
            )

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        x_lo, x_hi = _axis_limits(np.asarray(x_all, dtype=float), xscale, 100.0, 1.6)
        y_lo, y_hi = _axis_limits(np.asarray(y_all, dtype=float), yscale, 100.0, 1.9)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel(r"Scaled Cargo Radius $x = R_c / \xi$", fontsize=12.5, fontweight='bold')
        ax.set_ylabel(panel['ylabel'], fontsize=12.5, fontweight='bold')
        ax.set_title(panel['title'], fontsize=13.0, fontweight='bold', pad=10)
        ax.grid(True, which='both', linestyle='--', alpha=0.35)

        if show_raw:
            n_pts = int(np.count_nonzero(
                np.isfinite(np.asarray(x_all, dtype=float)) & np.isfinite(np.asarray(y_all, dtype=float))
            ))
            handles.insert(0, Line2D(
                [], [], marker='o', linestyle='none', color='#666666', alpha=0.6,
                markersize=6.0, label=rf"Individual experiments ($N = {n_pts}$)",
            ))
        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=8.4, frameon=True,
                      framealpha=0.92, labelspacing=0.30)

    fig.suptitle(
        rf"Transport Scaling vs Scaled Cargo Radius $x = R_c/\xi$ "
        rf"(MSD at $\Delta t = {msd_lag_s:.0f}\,\mathrm{{s}}$, "
        rf"$\lambda$ at $\Delta t = {lambda_lag_s:.0f}\,\mathrm{{s}}$)",
        fontsize=13.5, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    save_figure_to_all(fig, "msd300_lambda100_vs_scaled_radius_2panel", out_dirs)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Scaling plot of MSD(Delta t = 300 s) [left axis] and displacement "
                     "lambda(Delta t = 100 s) [right axis] versus x = R_c / xi.")
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions (e.g. /mnt/NAS-Ebanaru/Sasaki/MTsingleBeads).")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory for plots and CSVs (default: figure/scaling).")
    parser.add_argument('--beads', type=str, nargs='+', default=None,
                        help="Bead conditions to analyze (e.g. beads3um beads7um, or 'all'). Default: all.")
    parser.add_argument('--zarr_name', type=str, default='angular_correlation_w.zarr',
                        help="Angular correlation zarr file name inside each experiment directory.")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale [um/pixel] (default: 0.11).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames [s] (default: 4.0).")
    parser.add_argument('--msd_lag_s', type=float, default=300.0,
                        help="Lag time for the MSD (left axis) [s] (default: 300).")
    parser.add_argument('--lambda_lag_s', type=float, default=100.0,
                        help="Lag time for the displacement decay length lambda (right axis) [s] (default: 100).")
    parser.add_argument('--alpha_threshold', type=float, default=0.5,
                        help="MSD.py と同じ alpha フィルタ閾値（<= 0 で無効化, default: 0.5）.")
    parser.add_argument('--alpha_min_t', type=float, default=4.0,
                        help="alpha フィット範囲の下限 [s] (default: 4).")
    parser.add_argument('--alpha_max_t', type=float, default=300.0,
                        help="alpha フィット範囲の上限 [s] (default: 300).")
    parser.add_argument('--min_r_factor', type=float, default=1.1,
                        help="xi フィットの下限 min_r = min_r_factor * R_c (default: 1.1).")
    parser.add_argument('--max_r', type=float, default=25.0,
                        help="xi フィットの上限 [um] (default: 25).")
    parser.add_argument('--min_corr_threshold', type=float, default=0.05,
                        help="対数をとるための相関最小閾値 C(r) >= (default: 0.05).")
    parser.add_argument('--min_points', type=int, default=4,
                        help="xi フィットに必要な最小有効点数 (default: 4).")
    parser.add_argument('--xi_min', type=float, default=0.1,
                        help="有効な xi の下限 [um] (default: 0.1).")
    parser.add_argument('--xi_max', type=float, default=50.0,
                        help="有効な xi の上限 [um] (default: 50).")
    parser.add_argument('--xi_source', type=str, default='experiment', choices=['experiment', 'global'],
                        help="横軸に用いる xi: 'experiment'（実験ごとの測定値の中央値）または 'global'（--xi_value 固定値）.")
    parser.add_argument('--xi_value', type=float, default=2.7774,
                        help="--xi_source global のときの xi [um] (default: 2.7774).")
    parser.add_argument('--component', type=str, default='norm',
                        choices=['norm', 'x', 'y', 'parallel', 'perpendicular'],
                        help="lambda を求める変位成分 (default: norm = |dr|).")
    parser.add_argument('--bins', type=int, default=25,
                        help="変位 PDF のビン数 (default: 25).")
    parser.add_argument('--bin_max', type=float, default=50.0,
                        help="変位 PDF のビン範囲の上限 [um] (default: 50).")
    parser.add_argument('--fit_mode', type=str, default='log', choices=['log', 'linear'],
                        help="指数フィットの空間 (default: log).")
    parser.add_argument('--fit_rmin', type=float, default=None,
                        help="指数フィット範囲の下限 [um] (default: None = 全範囲).")
    parser.add_argument('--fit_rmax', type=float, default=None,
                        help="指数フィット範囲の上限 [um] (default: None = 全範囲).")
    parser.add_argument('--lambda_min_ratio', type=float, default=0.1,
                        help="lambda の妥当性判定: lambda >= ratio * <|dr|> (default: 0.1).")
    parser.add_argument('--lambda_max_ratio', type=float, default=10.0,
                        help="lambda の妥当性判定: lambda <= ratio * <|dr|> (default: 10.0).")
    parser.add_argument('--xscale', type=str, default='log', choices=['linear', 'log'],
                        help="X 軸スケール (default: log).")
    parser.add_argument('--yscale', type=str, default='log', choices=['linear', 'log'],
                        help="Y 軸スケール (default: log).")
    parser.add_argument('--no_raw', action='store_true',
                        help="実験ごとの生データ点（Layer 1）を描画しない.")
    parser.add_argument('--no_save_root', action='store_true',
                        help="データルート側 (root_dir/figure/scaling) への保存を行わない（既定では保存する）.")
    parser.add_argument('--title', type=str, default=None,
                        help="図のタイトル（未指定の場合は自動生成）.")
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else find_default_root()
    if root_dir is None or not root_dir.exists():
        raise FileNotFoundError("Data root directory not found. Please specify it with --root_dir.")

    # 対象ビーズ条件の決定
    if args.beads and 'all' not in args.beads:
        beads_info: List[dict] = []
        for item in args.beads:
            for b in str(item).split(','):
                b = b.strip()
                if not b:
                    continue
                if b in BEAD_LOOKUP:
                    beads_info.append(BEAD_LOOKUP[b])
                else:
                    print(f"[WARNING] Unknown bead condition: {b}")
    else:
        beads_info = list(BEADS_INFO)
    if not beads_info:
        raise ValueError("No valid bead names were given (--beads).")

    # 出力先ディレクトリ（既定でデータルート側 figure/scaling にも保存する）
    if args.output_dir:
        out_dirs = [Path(args.output_dir).expanduser()]
    else:
        out_dirs = [CURRENT_DIR / 'figure' / 'scaling', CURRENT_DIR / 'figure']
    if not args.no_save_root:
        out_dirs.append(root_dir / 'figure' / 'scaling')
    out_dirs = list(dict.fromkeys(out_dirs))

    print("=" * 78)
    print(" MSD(300 s) [left axis] and displacement lambda(100 s) [right axis] vs x = R_c / xi")
    print("=" * 78)
    print(f"Data Root Directory : {root_dir}")
    print("Output Directories  : " + ", ".join(str(d) for d in out_dirs))
    print(f"xi     : source = {args.xi_source}"
          + (f" (xi = {args.xi_value} um)" if args.xi_source == 'global' else "")
          + f", fit r in [{args.min_r_factor} * R_c, {args.max_r}] um,"
            f" C(r) >= {args.min_corr_threshold}, min_points = {args.min_points},"
            f" xi in [{args.xi_min}, {args.xi_max}] um")
    print(f"MSD    : Delta t = {args.msd_lag_s} s, alpha > {args.alpha_threshold} "
          f"({args.alpha_min_t}-{args.alpha_max_t} s), scale = {args.scale} um/px, "
          f"frame interval = {args.frame_interval} s")
    print(f"lambda : Delta t = {args.lambda_lag_s} s (component = {args.component}), "
          f"{args.bins} bins over [0, {args.bin_max}] um, fit_mode = {args.fit_mode}")
    print()
    print("Extracting per-experiment xi, MSD and lambda ...")

    df_points, df_cond = collect_experiment_records(
        root_dir=root_dir,
        beads_info=beads_info,
        zarr_name=args.zarr_name,
        scale=args.scale,
        frame_interval=args.frame_interval,
        msd_lag_s=args.msd_lag_s,
        lambda_lag_s=args.lambda_lag_s,
        alpha_threshold=args.alpha_threshold,
        alpha_min_t=args.alpha_min_t,
        alpha_max_t=args.alpha_max_t,
        min_r_factor=args.min_r_factor,
        max_r=args.max_r,
        min_corr_threshold=args.min_corr_threshold,
        min_points=args.min_points,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        xi_source=args.xi_source,
        xi_value=args.xi_value,
        component=args.component,
        bins=args.bins,
        bin_max=args.bin_max,
        fit_mode=args.fit_mode,
        fit_rmin=args.fit_rmin,
        fit_rmax=args.fit_rmax,
        lambda_min_ratio=args.lambda_min_ratio,
        lambda_max_ratio=args.lambda_max_ratio,
    )
    if df_points.empty:
        raise RuntimeError("No valid experiment records were extracted. "
                           "Check --root_dir and the fit parameters.")

    # --- CSV 出力（列順を整形） ---
    point_cols = [
        'bead_name', 'diameter_um', 'radius_um', 'exp_dir', 'xi_source', 'xi_um',
        'xi_mean_um', 'xi_std_um', 'xi_sem_um', 'n_xi_valid', 'n_xi_total', 'rc_over_xi',
        'msd_um2', 'msd_sem_um2', 'msd_median_um2', 'msd_std_um2', 'n_particles_msd',
        'n_particles_all', 'msd_lag_s_actual', 'alpha_filtered',
        'lambda_um', 'lambda_err_um', 'lambda_r2_log', 'lambda_fit_ratio', 'lambda_rejected',
        'mean_disp_um', 'n_displacements', 'tau_frames',
    ]
    ordered = [c for c in point_cols if c in df_points.columns]
    ordered += [c for c in df_points.columns if c not in ordered]
    df_points = df_points[ordered]

    save_csv_to_all(df_points, 'msd300_lambda100_vs_scaled_radius_points', out_dirs)
    save_csv_to_all(df_cond, 'msd300_lambda100_vs_scaled_radius_summary', out_dirs)

    # --- 作図 ---
    stats_dict = plot_scaling_twin_axis(
        df_points, df_cond, beads_info, out_dirs,
        xscale=args.xscale, yscale=args.yscale,
        msd_lag_s=args.msd_lag_s, lambda_lag_s=args.lambda_lag_s,
        show_raw=not args.no_raw,
        title=args.title,
    )
    plot_scaling_two_panel(
        df_points, df_cond, beads_info, out_dirs,
        xscale=args.xscale, yscale=args.yscale,
        msd_lag_s=args.msd_lag_s, lambda_lag_s=args.lambda_lag_s,
        show_raw=not args.no_raw,
    )

    # --- 参考図: 横軸 = R_c（linear）, 左軸 = MSD（黒, log）, 右軸 = lambda（赤, linear） ---
    plot_scaling_twin_axis(
        df_points, df_cond, beads_info, out_dirs,
        xscale='linear', msd_yscale='log', lambda_yscale='linear',
        msd_lag_s=args.msd_lag_s, lambda_lag_s=args.lambda_lag_s,
        show_raw=not args.no_raw,
        x_col='radius_um', x_col_cond='radius_um', x_err_col=None,
        xlabel=r"Cargo Radius $R_c$ [$\mu\mathrm{m}$]",
        x_label_symbol='R_c',
        msd_color='black', lambda_color='#d62728',
        x_jitter=0.06,
        basename='msd300_lambda100_vs_radius_rc',
        title=(rf"MSD$(\Delta t = {args.msd_lag_s:.0f}\,\mathrm{{s}})$ and "
               rf"$\lambda(\Delta t = {args.lambda_lag_s:.0f}\,\mathrm{{s}})$ "
               rf"vs Cargo Radius $R_c$"),
        legend_title_left=rf"MSD$(\Delta t = {args.msd_lag_s:.0f}\,\mathrm{{s}})$ [black axis]",
        legend_title_right=rf"$\lambda(\Delta t = {args.lambda_lag_s:.0f}\,\mathrm{{s}})$ [red axis]",
    )

    # --- ログ（代表値一覧・相関） ---
    print()
    print("=" * 78)
    print(" Per-condition representative values (mean +/- SEM)")
    print("=" * 78)
    show_cols = [
        'bead_name', 'diameter_um', 'n_experiments', 'rc_over_xi_mean', 'rc_over_xi_sem',
        'msd_um2_mean', 'msd_um2_sem', 'lambda_um_mean', 'lambda_um_sem', 'lambda_pooled_um',
    ]
    print(df_cond[[c for c in show_cols if c in df_cond.columns]].to_string(index=False))
    print()
    print(f"Spearman rho (log-log, experiments) : MSD {stats_dict['spearman_msd_exp']:+.3f}, "
          f"lambda {stats_dict['spearman_lambda_exp']:+.3f} "
          f"(N = {stats_dict['n_points_msd']})")
    print(f"Spearman rho (log-log, conditions)  : MSD {stats_dict['spearman_msd_cond']:+.3f}, "
          f"lambda {stats_dict['spearman_lambda_cond']:+.3f} "
          f"(N = {stats_dict['n_conditions']})")

    dev = df_points[
        np.isfinite(df_points['msd_lag_s_actual'])
        & ~np.isclose(df_points['msd_lag_s_actual'], args.msd_lag_s, atol=1e-6)
    ]
    if not dev.empty:
        print(f"[NOTE] {len(dev)} experiments had no exact lag of {args.msd_lag_s:.0f} s; "
              f"the nearest available lag was used (see msd_lag_s_actual).")
    n_nan_msd = int((~np.isfinite(df_points['msd_um2'])).sum())
    if n_nan_msd:
        print(f"[NOTE] MSD was NaN for {n_nan_msd} experiments (kept as NaN in the points CSV).")
    n_nan_lam = int((~np.isfinite(df_points['lambda_um'])).sum())
    if n_nan_lam:
        print(f"[NOTE] lambda was NaN for {n_nan_lam} experiments (kept as NaN in the points CSV).")
    print()
    print("Done.")


def collect_experiment_records(
    root_dir: Path,
    beads_info: List[dict],
    zarr_name: str = 'angular_correlation_w.zarr',
    scale: float = 0.11,
    frame_interval: float = 4.0,
    msd_lag_s: float = 300.0,
    lambda_lag_s: float = 100.0,
    alpha_threshold: float = 0.5,
    alpha_min_t: float = 4.0,
    alpha_max_t: float = 300.0,
    min_r_factor: float = 1.1,
    max_r: float = 25.0,
    min_corr_threshold: float = 0.05,
    min_points: int = 4,
    xi_min: float = 0.1,
    xi_max: float = 50.0,
    xi_source: str = 'experiment',
    xi_value: float = 2.7774,
    component: str = 'norm',
    bins: int = 25,
    bin_max: float = 50.0,
    fit_mode: str = 'log',
    fit_rmin: Optional[float] = None,
    fit_rmax: Optional[float] = None,
    lambda_min_ratio: float = 0.1,
    lambda_max_ratio: float = 10.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    全実験ディレクトリについて (xi_exp, MSD(Delta t = msd_lag_s), lambda(Delta t = lambda_lag_s)) を抽出する。

    Returns
    -------
    df_points : pd.DataFrame
        実験ごとの生データ（横軸 x = R_c / xi を含む）
    df_condition : pd.DataFrame
        粒子径ごとの代表値（mean +/- SEM）と、実験間平均 PDF から求めた lambda
    """
    records: List[dict] = []
    pooled_disp: Dict[str, List[np.ndarray]] = {}

    for binfo in beads_info:
        bname = binfo['name']
        dia = binfo['diameter_um']
        rc = binfo['radius_um']
        edirs = find_experiment_dirs(root_dir, bname, zarr_name)
        if not edirs:
            print(f"[SKIP] {bname}: no experiment dir with beads_tracks.csv + {zarr_name}")
            continue

        for edir in edirs:
            tracks_path = edir / 'beads_tracks.csv'
            try:
                df_tracks = pd.read_csv(tracks_path)
            except Exception as e:
                print(f"[WARNING] Failed to read {tracks_path}: {e}")
                continue

            xi_res = fit_experiment_xi(
                edir, rc,
                zarr_name=zarr_name, scale=scale,
                min_r_factor=min_r_factor, max_r=max_r,
                min_corr_threshold=min_corr_threshold,
                min_points=min_points, xi_min=xi_min, xi_max=xi_max,
            )
            if xi_source == 'global':
                xi_use = float(xi_value)
            else:
                xi_use = xi_res['xi_median_um']

            msd_res = compute_msd_at_lag(
                df_tracks, lag_s=msd_lag_s, scale=scale, frame_interval=frame_interval,
                alpha_threshold=alpha_threshold, alpha_min_t=alpha_min_t, alpha_max_t=alpha_max_t,
            )
            lam_res, disp_arr = compute_lambda_at_lag(
                df_tracks, lag_s=lambda_lag_s, scale=scale, frame_interval=frame_interval,
                component=component, bins=bins, bin_max=bin_max,
                fit_mode=fit_mode, fit_rmin=fit_rmin, fit_rmax=fit_rmax,
                lambda_min_ratio=lambda_min_ratio, lambda_max_ratio=lambda_max_ratio,
            )
            if disp_arr is not None:
                pooled_disp.setdefault(bname, []).append(disp_arr)

            rc_over_xi = rc / xi_use if (np.isfinite(xi_use) and xi_use > 0) else np.nan

            rec = {
                'bead_name': bname,
                'diameter_um': dia,
                'radius_um': rc,
                'exp_dir': edir.name,
                'xi_source': xi_source,
                'xi_um': xi_use,
            }
            rec.update({k: v for k, v in xi_res.items() if k not in ('xi_median_um',)})
            rec['rc_over_xi'] = rc_over_xi
            rec.update(msd_res)
            rec.update(lam_res)
            records.append(rec)

            if verbose:
                print(f"  {bname:10s} / {edir.name:18s} : "
                      f"x = {rc_over_xi:6.3f} (xi = {xi_use:5.2f} um, "
                      f"{xi_res['n_xi_valid']:5d}/{xi_res['n_xi_total']:5d} valid), "
                      f"MSD({msd_lag_s:.0f} s) = {msd_res['msd_um2']:8.3f} um^2 "
                      f"(N_p = {msd_res['n_particles_msd']}), "
                      f"lambda({lambda_lag_s:.0f} s) = {lam_res['lambda_um']:6.2f} um "
                      f"(N_disp = {lam_res['n_displacements']})")

    df_points = pd.DataFrame(records)

    pooled_lambda_map = {
        bname: fit_pooled_lambda(
            disps, bins=bins, bin_max=bin_max,
            fit_mode=fit_mode, fit_rmin=fit_rmin, fit_rmax=fit_rmax,
        )
        for bname, disps in pooled_disp.items()
    }
    df_condition = summarize_by_condition(df_points, pooled_lambda_map, beads_info)
    return df_points, df_condition


if __name__ == '__main__':
    main()
