"""
libs/spatial_heterogeneity.py

空間配向不均一性（Spatial Orientational Heterogeneity）を定量化するための解析モジュールです。

提供する3つの主要解析手法:
1. 4点配向相関関数 G_4(r) / chi_orient(r) (局所ペア相関の空間分散)
   chi_orient(r) = < [c(x, r) - C(r)]^2 >_x = < (u(x) · u(x+r))^2 >_x - C(r)^2
2. 内積確率密度関数 P(c; r) の非ガウス性パラメータ (NGP)
   alpha_{2, C}(r) = < c^4 >_r / (3 * < c^2 >_r^2) - 1   (c = u(x) · u(x+r) in [-1, 1])
3. 局所相関長 xi(x) の空間分布の直接評価 & 空間NGP
   alpha_{2, xi} = < xi^4 > / (3 * < xi^2 >^2) - 1
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# プロジェクト内の FFT 畳み込みモジュールをインポート
try:
    from libs.fft_convolution import FFTConvolver, create_spatial_kernel
except ImportError:
    from fft_convolution import FFTConvolver, create_spatial_kernel


def exp_decay_model(r: np.ndarray, xi: float, a: float = 1.0) -> np.ndarray:
    """指数減衰モデル: C(r) = a * exp(-r / xi)"""
    return a * np.exp(-r / np.maximum(xi, 1e-6))


def exp_decay_offset_model(r: np.ndarray, xi: float, a: float = 1.0, c0: float = 0.0) -> np.ndarray:
    """オフセット付き指数減衰モデル: C(r) = a * exp(-r / xi) + c0"""
    return a * np.exp(-r / np.maximum(xi, 1e-6)) + c0


# ==============================================================================
# 1. 4点配向相関関数 G_4(r) / chi_orient(r)
# ==============================================================================

def calc_4point_correlation_fft(
    ux: np.ndarray,
    uy: np.ndarray,
    distances_px: List[int],
    kernel_type: str = 'ring',
    shell_width: float = 2.0,
    valid_mask: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    2次元単位ベクトル場 (ux, uy) に対して、テンソル展開と2D FFT畳み込みを用いて
    2点配向相関 C(r) および4点配向相関 chi_orient(r) = <(u(x)·u(x+r))^2> - C(r)^2 を
    全空間で厳密かつ高速に計算する。

    展開理論:
      (u(x) · u(x+r))^2 = (ux(x)ux(x+r) + uy(x)uy(x+r))^2
                       = ux^2(x)ux^2(x+r) + uy^2(x)uy^2(x+r) + 2(ux*uy)(x)(ux*uy)(x+r)
      これはスカラー場 f1=ux^2, f2=uy^2, f3=sqrt(2)*ux*uy の自己相関の和に一致。

    Parameters
    ----------
    ux, uy : np.ndarray
        2次元の単位方向ベクトル配列 (H, W)
    distances_px : List[int]
        計算する空間距離 r (ピクセル) のリスト
    kernel_type : str, default 'ring'
        'ring' (円環シェル), 'disk', または 'gaussian'
    shell_width : float, default 2.0
        シェル幅 (ピクセル)
    valid_mask : np.ndarray, optional
        有効領域マスク (H, W) (1: 有効, 0: 無効)
    device : str, optional
        'cuda', 'scipy', または None (auto)

    Returns
    -------
    result : dict
        - 'distances_px': 距離配列 (px)
        - 'C_r': 通常の2点配向相関 C(r)
        - 'C2_r': 内積2乗の空間平均 <(u(x)·u(x+r))^2>
        - 'chi_orient': 4点配向相関 (局所相関の空間分散) chi_orient(r) = C2_r - C_r^2
        - 'chi_orient_local_var': 局所シェル平均 c(x, r) の空間分散 Var_x[c(x, r)]
    """
    H, W = ux.shape
    convolver = FFTConvolver(
        shape=(H, W),
        sizes=distances_px,
        kernel_type=kernel_type,
        shell_width=shell_width,
        device=device,
    )

    # 有効マスク
    if valid_mask is None:
        mag = np.hypot(ux, uy)
        mask = (mag > 1e-4).astype(np.float32)
    else:
        mask = valid_mask.astype(np.float32)

    # 単位ベクトル化の保証
    v_mag = np.hypot(ux, uy)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_x = np.where(v_mag > 1e-6, ux / v_mag, 0.0).astype(np.float32) * mask
        u_y = np.where(v_mag > 1e-6, uy / v_mag, 0.0).astype(np.float32) * mask

    # テンソル成分 (f1 = ux^2, f2 = uy^2, f3 = sqrt(2)*ux*uy)
    f1 = (u_x ** 2) * mask
    f2 = (u_y ** 2) * mask
    f3 = (np.sqrt(2.0, dtype=np.float32) * u_x * u_y) * mask

    num_d = len(distances_px)
    c_r_list = np.zeros(num_d, dtype=np.float32)
    c2_r_list = np.zeros(num_d, dtype=np.float32)
    local_c_var_list = np.zeros(num_d, dtype=np.float32)

    import scipy.fft

    fft_mask = scipy.fft.rfft2(mask, workers=4)
    fft_ux = scipy.fft.rfft2(u_x, workers=4)
    fft_uy = scipy.fft.rfft2(u_y, workers=4)
    fft_f1 = scipy.fft.rfft2(f1, workers=4)
    fft_f2 = scipy.fft.rfft2(f2, workers=4)
    fft_f3 = scipy.fft.rfft2(f3, workers=4)

    total_valid = np.sum(mask)
    if total_valid < 10:
        return {
            'distances_px': np.array(distances_px),
            'C_r': np.full(num_d, np.nan),
            'C2_r': np.full(num_d, np.nan),
            'chi_orient': np.full(num_d, np.nan),
            'chi_orient_local_var': np.full(num_d, np.nan),
        }

    for idx, k_fft in enumerate(convolver.kernel_ffts_np):
        # 1. 重み正規化用マスクの畳み込み
        v_mask = scipy.fft.irfft2(fft_mask * k_fft, s=(H, W), workers=4)
        valid_pixels = (mask > 0.5) & (v_mask > 1e-4)

        if not np.any(valid_pixels):
            c_r_list[idx] = np.nan
            c2_r_list[idx] = np.nan
            local_c_var_list[idx] = np.nan
            continue

        denom = np.where(v_mask > 1e-4, v_mask, 1.0)

        # 2. 2点相関 C(r) = <ux * conv(ux) + uy * conv(uy)> / denom
        conv_ux = scipy.fft.irfft2(fft_ux * k_fft, s=(H, W), workers=4)
        conv_uy = scipy.fft.irfft2(fft_uy * k_fft, s=(H, W), workers=4)

        # 局所相関 c(x, r) = u(x) · (K_r * u)(x)
        local_c = (u_x * conv_ux + u_y * conv_uy) / denom
        local_c_valid = local_c[valid_pixels]

        c_mean = float(np.mean(local_c_valid))
        c_r_list[idx] = c_mean
        local_c_var_list[idx] = float(np.var(local_c_valid))

        # 3. 4点相関 厳密ペア平均 < (u(x)·u(x+r))^2 >
        conv_f1 = scipy.fft.irfft2(fft_f1 * k_fft, s=(H, W), workers=4)
        conv_f2 = scipy.fft.irfft2(fft_f2 * k_fft, s=(H, W), workers=4)
        conv_f3 = scipy.fft.irfft2(fft_f3 * k_fft, s=(H, W), workers=4)

        local_c2_pair = (f1 * conv_f1 + f2 * conv_f2 + f3 * conv_f3) / denom
        c2_mean = float(np.mean(local_c2_pair[valid_pixels]))
        c2_r_list[idx] = c2_mean

    chi_orient = c2_r_list - (c_r_list ** 2)

    return {
        'distances_px': np.array(distances_px, dtype=np.float32),
        'C_r': c_r_list,
        'C2_r': c2_r_list,
        'chi_orient': chi_orient,
        'chi_orient_local_var': local_c_var_list,
    }


# ==============================================================================
# 2. 内積確率密度関数 P(c; r) の非ガウス性パラメータ (NGP)
# ==============================================================================

def sample_field_dot_products(
    ux: np.ndarray,
    uy: np.ndarray,
    distances_px: List[int],
    n_samples_per_dist: int = 20000,
    shell_width: float = 2.0,
    valid_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    """
    2次元ベクトル場から各距離 r におけるペア内積 c = u(x) · u(x+r) を均等にサンプリングする。

    Parameters
    ----------
    ux, uy : np.ndarray
        2次元の単位方向ベクトル配列 (H, W)
    distances_px : List[int]
        サンプリングする距離 (ピクセル)
    n_samples_per_dist : int, default 20000
        各距離におけるサンプリングペア数
    shell_width : float, default 2.0
        距離許容幅
    valid_mask : np.ndarray, optional
        有効マスク (H, W)

    Returns
    -------
    samples_by_dist : Dict[int, np.ndarray]
        各距離 r に対する内積値 c in [-1, 1] の配列
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = ux.shape
    if valid_mask is None:
        mag = np.hypot(ux, uy)
        valid_mask = mag > 1e-4

    valid_y, valid_x = np.where(valid_mask)
    n_valid = len(valid_y)
    if n_valid < 10:
        return {r: np.empty(0, dtype=np.float32) for r in distances_px}

    samples_by_dist = {}
    half_w = shell_width / 2.0

    for r in distances_px:
        # ランダムな方位角 phi を生成
        sample_indices = rng.integers(0, n_valid, size=n_samples_per_dist)
        y0 = valid_y[sample_indices]
        x0 = valid_x[sample_indices]

        phi = rng.uniform(0, 2 * np.pi, size=n_samples_per_dist)
        r_actual = rng.uniform(max(0.5, r - half_w), r + half_w, size=n_samples_per_dist)

        y1 = np.round(y0 + r_actual * np.sin(phi)).astype(int)
        x1 = np.round(x0 + r_actual * np.cos(phi)).astype(int)

        # 境界内判定
        in_bounds = (y1 >= 0) & (y1 < H) & (x1 >= 0) & (x1 < W)
        if not np.any(in_bounds):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        y0_b = y0[in_bounds]
        x0_b = x0[in_bounds]
        y1_b = y1[in_bounds]
        x1_b = x1[in_bounds]

        # 両点が有効領域か判定
        valid_pair = valid_mask[y0_b, x0_b] & valid_mask[y1_b, x1_b]
        if not np.any(valid_pair):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        y0_v = y0_b[valid_pair]
        x0_v = x0_b[valid_pair]
        y1_v = y1_b[valid_pair]
        x1_v = x1_b[valid_pair]

        u0_x = ux[y0_v, x0_v]
        u0_y = uy[y0_v, x0_v]
        u1_x = ux[y1_v, x1_v]
        u1_y = uy[y1_v, x1_v]

        mag0 = np.hypot(u0_x, u0_y)
        mag1 = np.hypot(u1_x, u1_y)
        valid_norm = (mag0 > 1e-4) & (mag1 > 1e-4)

        if not np.any(valid_norm):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        c_vals = (u0_x[valid_norm] * u1_x[valid_norm] + u0_y[valid_norm] * u1_y[valid_norm]) / (mag0[valid_norm] * mag1[valid_norm])
        c_vals = np.clip(c_vals, -1.0, 1.0)
        samples_by_dist[r] = c_vals.astype(np.float32)

    return samples_by_dist


def calc_dot_product_distribution_and_ngp(
    samples_by_dist: Dict[int, np.ndarray],
    n_bins: int = 50,
) -> Dict[str, Union[np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
    """
    距離別の内積サンプル集合から、確率密度関数 P(c; r) および NGP alpha_{2, C}(r) を計算する。

    定義:
      alpha_{2, C}(r) = < c^4 >_r / (3 * < c^2 >_r^2) - 1

    Parameters
    ----------
    samples_by_dist : Dict[int, np.ndarray]
        各距離 r に対する内積値 c in [-1, 1] の配列
    n_bins : int, default 50
        [-1, 1] のヒストグラムビン数

    Returns
    -------
    result : dict
        - 'distances': 距離配列
        - 'ngp': 各距離の alpha_{2, C}(r) 配列
        - 'c2_mean': 各距離の < c^2 >_r
        - 'c4_mean': 各距離の < c^4 >_r
        - 'pdf_dict': 各距離 r -> (c_centers, pdf)
    """
    distances = sorted(list(samples_by_dist.keys()))
    ngp_list = []
    c2_list = []
    c4_list = []
    pdf_dict = {}

    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for r in distances:
        c = samples_by_dist[r]
        if len(c) < 30:
            ngp_list.append(np.nan)
            c2_list.append(np.nan)
            c4_list.append(np.nan)
            pdf_dict[r] = (bin_centers, np.zeros_like(bin_centers))
            continue

        c2 = float(np.mean(c ** 2))
        c4 = float(np.mean(c ** 4))

        if c2 > 1e-8:
            alpha = (c4 / (3.0 * (c2 ** 2))) - 1.0
        else:
            alpha = np.nan

        ngp_list.append(alpha)
        c2_list.append(c2)
        c4_list.append(c4)

        hist, _ = np.histogram(c, bins=bin_edges, density=True)
        pdf_dict[r] = (bin_centers, hist)

    return {
        'distances': np.array(distances, dtype=np.float32),
        'ngp': np.array(ngp_list, dtype=np.float32),
        'c2_mean': np.array(c2_list, dtype=np.float32),
        'c4_mean': np.array(c4_list, dtype=np.float32),
        'pdf_dict': pdf_dict,
    }


# ==============================================================================
# 3. 局所相関長 xi(x) の空間分布の直接評価 & 空間NGP
# ==============================================================================

def calc_local_correlation_length_map(
    ux: np.ndarray,
    uy: np.ndarray,
    distances_px: List[int],
    scale: float = 0.11,
    grid_step: int = 8,
    kernel_type: str = 'ring',
    shell_width: float = 2.0,
    valid_mask: Optional[np.ndarray] = None,
    max_fit_dist_um: Optional[float] = None,
    min_r2: float = 0.5,
    device: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    視野全体をグリッド分割し、各地点を中心とした局所配向相関プロファイル C_x(r) を
    指数減衰 C(r) = a * exp(-r / xi) にフィッティングして局所相関長 xi(x) の空間マップを構築。
    得られた xi(x) から空間非ガウス性パラメータ alpha_{2, xi} を直接計算する。

    定義:
      alpha_{2, xi} = < xi^4 > / (3 * < xi^2 >^2) - 1

    Parameters
    ----------
    ux, uy : np.ndarray
        2次元方向ベクトル配列 (H, W)
    distances_px : List[int]
        相関を計算する距離リスト (ピクセル)
    scale : float, default 0.11
        空間スケール (um/pixel)
    grid_step : int, default 8
        グリッドサンプリング間隔 (ピクセル)
    kernel_type : str, default 'ring'
        畳み込みカーネル種別
    shell_width : float, default 2.0
        シェル幅 (ピクセル)
    valid_mask : np.ndarray, optional
        有効マスク (H, W)
    max_fit_dist_um : float, optional
        フィッティングに使用する最大距離 (um)
    min_r2 : float, default 0.5
        フィッティング決定係数 R^2 の最小許容値（下回る場合は NaN 扱い）
    device : str, optional
        'cuda', 'scipy', または None

    Returns
    -------
    result : dict
        - 'grid_x_um', 'grid_y_um': グリッド座標 (um)
        - 'xi_map_um': 局所相関長マップ (グリッド点 x グリッド点, um)
        - 'amp_map': フィッティング振幅マップ a(x)
        - 'r2_map': フィッティング決定係数マップ R^2(x)
        - 'xi_valid_um': 有効な局所相関長の1次元配列
        - 'alpha_2_xi': 相関長の空間非ガウス性パラメータ
        - 'xi_mean_um': 相関長の空間平均
        - 'xi_median_um': 相関長の中央値
        - 'xi_std_um': 相関長の空間標準偏差
    """
    H, W = ux.shape
    convolver = FFTConvolver(
        shape=(H, W),
        sizes=distances_px,
        kernel_type=kernel_type,
        shell_width=shell_width,
        device=device,
    )

    if valid_mask is None:
        mag = np.hypot(ux, uy)
        mask = (mag > 1e-4).astype(np.float32)
    else:
        mask = valid_mask.astype(np.float32)

    v_mag = np.hypot(ux, uy)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_x = np.where(v_mag > 1e-6, ux / v_mag, 0.0).astype(np.float32) * mask
        u_y = np.where(v_mag > 1e-6, uy / v_mag, 0.0).astype(np.float32) * mask

    import scipy.fft

    fft_mask = scipy.fft.rfft2(mask, workers=4)
    fft_ux = scipy.fft.rfft2(u_x, workers=4)
    fft_uy = scipy.fft.rfft2(u_y, workers=4)

    distances_um = np.array(distances_px, dtype=np.float32) * scale
    num_d = len(distances_px)

    # グリッド座標の作成
    y_coords = np.arange(0, H, grid_step)
    x_coords = np.arange(0, W, grid_step)
    nY = len(y_coords)
    nX = len(x_coords)

    # 局所相関テンソル (num_d, nY, nX)
    local_corr_grid = np.full((num_d, nY, nX), np.nan, dtype=np.float32)

    for idx, k_fft in enumerate(convolver.kernel_ffts_np):
        v_mask = scipy.fft.irfft2(fft_mask * k_fft, s=(H, W), workers=4)
        conv_ux = scipy.fft.irfft2(fft_ux * k_fft, s=(H, W), workers=4)
        conv_uy = scipy.fft.irfft2(fft_uy * k_fft, s=(H, W), workers=4)

        denom = np.where(v_mask > 1e-4, v_mask, 1.0)
        c_field = (u_x * conv_ux + u_y * conv_uy) / denom
        c_field = np.where((mask > 0.5) & (v_mask > 1e-4), c_field, np.nan)

        # グリッドサンプリング
        local_corr_grid[idx, :, :] = c_field[np.ix_(y_coords, x_coords)]

    # フィッティングによる相関長抽出
    xi_map = np.full((nY, nX), np.nan, dtype=np.float32)
    amp_map = np.full((nY, nX), np.nan, dtype=np.float32)
    r2_map = np.full((nY, nX), np.nan, dtype=np.float32)

    fit_mask = np.ones(num_d, dtype=bool)
    if max_fit_dist_um is not None:
        fit_mask = distances_um <= max_fit_dist_um

    x_fit_all = distances_um[fit_mask]

    # グリッドごとのフィッティングループ
    for iy in range(nY):
        for ix in range(nX):
            y_curve = local_corr_grid[fit_mask, iy, ix]
            valid_pt = ~np.isnan(y_curve)
            if np.count_nonzero(valid_pt) < 3:
                continue

            x_v = x_fit_all[valid_pt]
            y_v = y_curve[valid_pt]

            # 最初の点が極端に小さい/負の場合はスキップ
            if y_v[0] <= 0.05:
                continue

            a_init = float(np.clip(y_v[0], 0.1, 1.0))
            p0 = [10.0 * scale, a_init]
            bounds = ([0.05 * scale, 0.0], [200.0 * scale, 1.5])

            try:
                popt, _ = curve_fit(exp_decay_model, x_v, y_v, p0=p0, bounds=bounds, maxfev=400)
                xi_est, a_est = popt

                # R^2 決定係数
                residuals = y_v - exp_decay_model(x_v, *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
                r2 = 1.0 - (ss_res / (ss_tot + 1e-9)) if ss_tot > 1e-9 else 0.0

                if r2 >= min_r2 or (r2 < min_r2 and xi_est < 2.0 * scale):
                    xi_map[iy, ix] = xi_est
                    amp_map[iy, ix] = a_est
                    r2_map[iy, ix] = r2
            except Exception:
                continue

    # 有効な相関長配列
    valid_xi = xi_map[~np.isnan(xi_map)]

    if len(valid_xi) >= 10:
        xi2_mean = float(np.mean(valid_xi ** 2))
        xi4_mean = float(np.mean(valid_xi ** 4))
        if xi2_mean > 1e-8:
            alpha_2_xi = (xi4_mean / (3.0 * (xi2_mean ** 2))) - 1.0
        else:
            alpha_2_xi = np.nan
        xi_mean = float(np.mean(valid_xi))
        xi_median = float(np.median(valid_xi))
        xi_std = float(np.std(valid_xi))
    else:
        alpha_2_xi = np.nan
        xi_mean = np.nan
        xi_median = np.nan
        xi_std = np.nan

    return {
        'grid_x_um': x_coords * scale,
        'grid_y_um': y_coords * scale,
        'xi_map_um': xi_map,
        'amp_map': amp_map,
        'r2_map': r2_map,
        'xi_valid_um': valid_xi,
        'alpha_2_xi': float(alpha_2_xi),
        'xi_mean_um': float(xi_mean),
        'xi_median_um': float(xi_median),
        'xi_std_um': float(xi_std),
    }


# ==============================================================================
# 4. 可視化・プロット用ユーティリティ
# ==============================================================================

def plot_spatial_heterogeneity_summary(
    distances_um: np.ndarray,
    four_point_result: dict,
    ngp_result: dict,
    local_xi_result: dict,
    condition_name: str = "",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    1つの実験または条件に対して、3つの不均一性解析結果をまとめた総合ダッシュボード図を作成する。

    構成 (2x2 パネル):
      [Panel A] 4点配向相関関数 chi_orient(r) vs r (相関長ピーク R0 の検出)
      [Panel B] 内積確率密度関数 P(c; r) の分布推移 & NGP alpha_{2, C}(r)
      [Panel C] 局所相関長 xi(x) の空間マップ (高速道路 vs ジャンクション)
      [Panel D] 相関長分布 P(xi) と空間 NGP alpha_{2, xi}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    # -------------------------------------------------------------
    # Panel A: 4点配向相関関数 chi_orient(r)
    # -------------------------------------------------------------
    ax_a = axes[0]
    r_um = distances_um
    chi = four_point_result['chi_orient']
    c_r = four_point_result['C_r']

    ax_a.plot(r_um, chi, color='#d95f02', lw=2.5, marker='o', ms=4, label=r'$\chi_{\mathrm{orient}}(r) = \langle c^2 \rangle - C(r)^2$')
    if 'chi_orient_local_var' in four_point_result:
        ax_a.plot(r_um, four_point_result['chi_orient_local_var'], color='#7570b3', lw=1.8, ls='--', label=r'$\mathrm{Var}_{\mathbf{x}}[c(\mathbf{x}, r)]$ (Local Shell)')
    ax_a.set_xlabel(r'Distance $r\ [\mu\mathrm{m}]$', fontsize=12)
    ax_a.set_ylabel(r'4-Point Susceptibility $\chi_{\mathrm{orient}}(r)$', fontsize=12)
    ax_a.set_title(r'(a) 4-Point Orientational Correlation $G_4(r)$', fontsize=13, fontweight='bold')
    ax_a.grid(True, linestyle=':', alpha=0.6)
    ax_a.legend(loc='best', frameon=True, fontsize=10)

    # ピーク位置の注釈
    valid_idx = ~np.isnan(chi)
    if np.any(valid_idx):
        max_idx = np.argmax(chi[valid_idx])
        peak_r = r_um[valid_idx][max_idx]
        peak_val = chi[valid_idx][max_idx]
        ax_a.annotate(
            f'Peak: $r \\approx {peak_r:.1f}\\,\\mu\\mathrm{{m}}$\n$\\chi_{{max}} = {peak_val:.3f}$',
            xy=(peak_r, peak_val),
            xytext=(peak_r + 5, peak_val * 0.9),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="orange", lw=1),
        )

    # -------------------------------------------------------------
    # Panel B: 内積確率密度関数 P(c; r) & NGP alpha_{2, C}(r)
    # -------------------------------------------------------------
    ax_b = axes[1]
    ngp = ngp_result['ngp']
    r_ngp = ngp_result['distances'] * (r_um[0] / four_point_result['distances_px'][0] if len(r_um) > 0 else 0.11)

    # 主軸: alpha_{2, C}(r)
    line_ngp = ax_b.plot(r_ngp, ngp, color='#1b9e77', lw=2.5, marker='s', ms=4, label=r'$\alpha_{2, C}(r)$ (NGP)')
    ax_b.set_xlabel(r'Distance $r\ [\mu\mathrm{m}]$', fontsize=12)
    ax_b.set_ylabel(r'Non-Gaussian Parameter $\alpha_{2, C}(r)$', color='#1b9e77', fontsize=12)
    ax_b.tick_params(axis='y', labelcolor='#1b9e77')
    ax_b.grid(True, linestyle=':', alpha=0.6)
    ax_b.set_title(r'(b) Dot Product NGP $\alpha_{2, C}(r)$', fontsize=13, fontweight='bold')

    # 代表的な距離での P(c; r) の挿入図 (Inset)
    pdf_dict = ngp_result.get('pdf_dict', {})
    if pdf_dict:
        # 近距離、中距離 (相関長付近)、長距離 の3つを選択
        avail_dists = sorted(list(pdf_dict.keys()))
        selected_dists = [avail_dists[0], avail_dists[len(avail_dists) // 3], avail_dists[-1]]
        colors = ['#2b83ba', '#fdae61', '#d7191c']

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax_b, width="42%", height="40%", loc='lower right', borderpad=1.2)
        for d, col in zip(selected_dists, colors):
            c_vals, p_vals = pdf_dict[d]
            d_um = d * (r_um[0] / four_point_result['distances_px'][0] if len(r_um) > 0 else 0.11)
            ax_inset.plot(c_vals, p_vals, color=col, lw=1.6, label=f'$r={d_um:.1f}\\mu\\mathrm{{m}}$')
        ax_inset.set_xlabel(r'$c = \mathbf{u}_1 \cdot \mathbf{u}_2$', fontsize=8)
        ax_inset.set_ylabel(r'$P(c; r)$', fontsize=8)
        ax_inset.tick_params(labelsize=8)
        ax_inset.legend(loc='upper center', fontsize=7, framealpha=0.8)
        ax_inset.grid(True, linestyle=':', alpha=0.4)

    # -------------------------------------------------------------
    # Panel C: 局所相関長 xi(x) 空間マップ
    # -------------------------------------------------------------
    ax_c = axes[2]
    xi_map = local_xi_result['xi_map_um']
    gx = local_xi_result['grid_x_um']
    gy = local_xi_result['grid_y_um']

    if len(gx) > 1 and len(gy) > 1:
        extent = [gx[0], gx[-1], gy[-1], gy[0]]
        vmax = float(np.nanpercentile(xi_map, 98)) if np.any(~np.isnan(xi_map)) else 20.0
        vmin = float(np.nanpercentile(xi_map, 2)) if np.any(~np.isnan(xi_map)) else 0.5

        im = ax_c.imshow(xi_map, extent=extent, cmap='viridis', vmin=max(0, vmin), vmax=max(vmin + 1, vmax), aspect='auto')
        cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label(r'Local Correlation Length $\xi(\mathbf{x})\ [\mu\mathrm{m}]$', fontsize=11)
        ax_c.set_xlabel(r'$x\ [\mu\mathrm{m}]$', fontsize=12)
        ax_c.set_ylabel(r'$y\ [\mu\mathrm{m}]$', fontsize=12)
        ax_c.set_title(r'(c) Local Correlation Length Map $\xi(\mathbf{x})$', fontsize=13, fontweight='bold')
    else:
        ax_c.text(0.5, 0.5, 'Insufficient grid points', ha='center', va='center')

    # -------------------------------------------------------------
    # Panel D: 局所相関長分布 P(xi) & 空間 NGP alpha_{2, xi}
    # -------------------------------------------------------------
    ax_d = axes[3]
    valid_xi = local_xi_result['xi_valid_um']
    alpha_xi = local_xi_result['alpha_2_xi']
    xi_mean = local_xi_result['xi_mean_um']
    xi_std = local_xi_result['xi_std_um']

    if len(valid_xi) > 0:
        counts, bins, _ = ax_d.hist(valid_xi, bins=25, density=True, color='#882255', alpha=0.65, edgecolor='black')
        ax_d.axvline(xi_mean, color='black', lw=2, ls='--', label=f'Mean $\\langle \\xi \\rangle = {xi_mean:.2f}\\,\\mu\\mathrm{{m}}$')
        ax_d.axvline(local_xi_result['xi_median_um'], color='blue', lw=1.5, ls=':', label=f'Median = {local_xi_result["xi_median_um"]:.2f}\\,\\mu\\mathrm{{m}}$')

        # KDE 曲線のプロット
        try:
            kde = stats.gaussian_kde(valid_xi)
            x_eval = np.linspace(np.min(valid_xi), np.max(valid_xi), 200)
            ax_d.plot(x_eval, kde(x_eval), color='#882255', lw=2.2)
        except Exception:
            pass

        # 空間 NGP の情報ボックス
        text_info = (
            f"$\\mathbf{{\\alpha_{{2, \\xi}} = {alpha_xi:.3f}}}$\n"
            f"$\\langle \\xi \\rangle = {xi_mean:.2f} \\pm {xi_std:.2f}\\,\\mu\\mathrm{{m}}$\n"
            f"$N_{{\\mathrm{{grid}}}} = {len(valid_xi)}$"
        )
        ax_d.text(
            0.95, 0.85, text_info,
            transform=ax_d.transAxes,
            ha='right', va='top',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="gray", lw=1.2)
        )

        ax_d.set_xlabel(r'Correlation Length $\xi\ [\mu\mathrm{m}]$', fontsize=12)
        ax_d.set_ylabel(r'Probability Density $P(\xi)$', fontsize=12)
        ax_d.set_title(r'(d) Distribution $P(\xi)$ & Spatial NGP $\alpha_{2, \xi}$', fontsize=13, fontweight='bold')
        ax_d.grid(True, linestyle=':', alpha=0.6)
        ax_d.legend(loc='upper right', bbox_to_anchor=(0.95, 0.60), fontsize=9.5)
    else:
        ax_d.text(0.5, 0.5, 'No valid correlation lengths', ha='center', va='center')

    title_str = f"Spatial Orientational Heterogeneity Analysis: {condition_name}" if condition_name else "Spatial Orientational Heterogeneity Analysis"
    fig.suptitle(title_str, fontsize=15, fontweight='bold', y=0.995)
    try:
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    except Exception:
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.3, wspace=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved heterogeneity dashboard to {save_path}")

    return fig
