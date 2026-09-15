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
import scipy.fft

# PyTorch / CUDA サポートの確認
HAS_TORCH = False
TORCH_CUDA = False
try:
    import torch
    HAS_TORCH = True
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    pass

# Numba サポートの確認
HAS_NUMBA = False
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:
    pass

# プロジェクト内の FFT 畳み込みモジュールをインポート
try:
    from libs.fft_convolution import FFTConvolver, create_spatial_kernel
except ImportError:
    from fft_convolution import FFTConvolver, create_spatial_kernel


def _get_device(device_str: Optional[str] = None) -> Tuple[str, Optional[object]]:
    """実行デバイス ('cuda' or 'cpu') と torch.device を判定して返す。"""
    if device_str is not None:
        if device_str == 'cuda' and HAS_TORCH and TORCH_CUDA:
            return 'cuda', torch.device('cuda')
        return 'cpu', torch.device('cpu') if HAS_TORCH else None

    if HAS_TORCH and TORCH_CUDA:
        try:
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            if free_mem > 150:
                return 'cuda', torch.device('cuda')
        except Exception:
            pass
    return 'cpu', torch.device('cpu') if HAS_TORCH else None


def exp_decay_model(r: np.ndarray, xi: float, a: float = 1.0) -> np.ndarray:
    """指数減衰モデル: C(r) = a * exp(-r / xi)"""
    return a * np.exp(-r / np.maximum(xi, 1e-6))


def exp_decay_offset_model(r: np.ndarray, xi: float, a: float = 1.0, c0: float = 0.0) -> np.ndarray:
    """オフセット付き指数減衰モデル: C(r) = a * exp(-r / xi) + c0"""
    return a * np.exp(-r / np.maximum(xi, 1e-6)) + c0


# ==============================================================================
# 1. 4点配向相関関数 G_4(r) / chi_orient(r) (GPU & バッチ最適化)
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
    全空間で厳密かつ超高速（GPU/バッチFFT対応）に計算する。

    展開理論:
      (u(x) · u(x+r))^2 = (ux(x)ux(x+r) + uy(x)uy(x+r))^2
                       = ux^2(x)ux^2(x+r) + uy^2(x)uy^2(x+r) + 2(ux*uy)(x)(ux*uy)(x+r)
      これはスカラー場 f1=ux^2, f2=uy^2, f3=sqrt(2)*ux*uy の自己相関の和に一致。
    """
    H, W = ux.shape
    num_d = len(distances_px)

    # 有効マスク
    if valid_mask is None:
        mag = np.hypot(ux, uy)
        mask = (mag > 1e-4).astype(np.float32)
    else:
        mask = valid_mask.astype(np.float32)

    total_valid = np.sum(mask)
    if total_valid < 10:
        return {
            'distances_px': np.array(distances_px, dtype=np.float32),
            'C_r': np.full(num_d, np.nan, dtype=np.float32),
            'C2_r': np.full(num_d, np.nan, dtype=np.float32),
            'chi_orient': np.full(num_d, np.nan, dtype=np.float32),
            'chi_orient_local_var': np.full(num_d, np.nan, dtype=np.float32),
        }

    # 単位ベクトル化
    v_mag = np.hypot(ux, uy)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_x = np.where(v_mag > 1e-6, ux / v_mag, 0.0).astype(np.float32) * mask
        u_y = np.where(v_mag > 1e-6, uy / v_mag, 0.0).astype(np.float32) * mask

    f1 = (u_x ** 2) * mask
    f2 = (u_y ** 2) * mask
    f3 = (np.sqrt(2.0, dtype=np.float32) * u_x * u_y) * mask

    dev_type, t_device = _get_device(device)

    # FFTConvolver で事前計算カーネルを取得
    convolver = FFTConvolver(
        shape=(H, W),
        sizes=distances_px,
        kernel_type=kernel_type,
        shell_width=shell_width,
        device=dev_type,
    )

    # -------------------------------------------------------------
    # GPU (PyTorch CUDA) による超高速バッチ実行
    # -------------------------------------------------------------
    if dev_type == 'cuda' and HAS_TORCH:
        try:
            with torch.no_grad():
                # 6つのスカラー場を 1つのテンソル (6, H, W) として GPU に転送
                # fields: [u_x, u_y, mask, f1, f2, f3]
                fields_np = np.stack([u_x, u_y, mask, f1, f2, f3], axis=0)
                t_fields = torch.from_numpy(fields_np).to(t_device, non_blocking=True)

                # 一括 2D 実フーリエ変換 (6, H, W_rfft)
                fft_fields = torch.fft.rfft2(t_fields)

                c_r_list = np.zeros(num_d, dtype=np.float32)
                c2_r_list = np.zeros(num_d, dtype=np.float32)
                local_c_var_list = np.zeros(num_d, dtype=np.float32)

                d_mask = t_fields[2]
                d_ux = t_fields[0]
                d_uy = t_fields[1]
                d_f1 = t_fields[3]
                d_f2 = t_fields[4]
                d_f3 = t_fields[5]

                for idx, k_fft_cpu in enumerate(convolver.kernel_ffts_torch):
                    k_fft = k_fft_cpu.to(t_device, non_blocking=True)
                    # バッチ乗算 & 逆フーリエ変換 (6, H, W)
                    conv_all = torch.fft.irfft2(fft_fields * k_fft, s=(H, W))

                    conv_ux = conv_all[0]
                    conv_uy = conv_all[1]
                    v_mask = conv_all[2]
                    conv_f1 = conv_all[3]
                    conv_f2 = conv_all[4]
                    conv_f3 = conv_all[5]

                    valid_pixels = (d_mask > 0.5) & (v_mask > 1e-4)
                    if not torch.any(valid_pixels):
                        c_r_list[idx] = np.nan
                        c2_r_list[idx] = np.nan
                        local_c_var_list[idx] = np.nan
                        continue

                    denom = torch.where(v_mask > 1e-4, v_mask, torch.tensor(1.0, device=t_device))

                    # 局所相関 c(x, r)
                    local_c = (d_ux * conv_ux + d_uy * conv_uy) / denom
                    local_c_valid = local_c[valid_pixels]

                    c_r_list[idx] = float(torch.mean(local_c_valid).cpu().numpy())
                    local_c_var_list[idx] = float(torch.var(local_c_valid).cpu().numpy())

                    # 4点ペア相関 <(u(x)·u(x+r))^2>
                    local_c2_pair = (d_f1 * conv_f1 + d_f2 * conv_f2 + d_f3 * conv_f3) / denom
                    c2_r_list[idx] = float(torch.mean(local_c2_pair[valid_pixels]).cpu().numpy())

                chi_orient = c2_r_list - (c_r_list ** 2)

                return {
                    'distances_px': np.array(distances_px, dtype=np.float32),
                    'C_r': c_r_list,
                    'C2_r': c2_r_list,
                    'chi_orient': chi_orient,
                    'chi_orient_local_var': local_c_var_list,
                }
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # CPU (マルチスレッド SciPy) によるフォールバック
    # -------------------------------------------------------------
    fields_np = np.stack([u_x, u_y, mask, f1, f2, f3], axis=0)
    fft_fields = scipy.fft.rfft2(fields_np, axes=(-2, -1), workers=4)

    c_r_list = np.zeros(num_d, dtype=np.float32)
    c2_r_list = np.zeros(num_d, dtype=np.float32)
    local_c_var_list = np.zeros(num_d, dtype=np.float32)

    for idx, k_fft in enumerate(convolver.kernel_ffts_np):
        conv_all = scipy.fft.irfft2(fft_fields * k_fft, s=(H, W), axes=(-2, -1), workers=4)
        conv_ux = conv_all[0]
        conv_uy = conv_all[1]
        v_mask = conv_all[2]
        conv_f1 = conv_all[3]
        conv_f2 = conv_all[4]
        conv_f3 = conv_all[5]

        valid_pixels = (mask > 0.5) & (v_mask > 1e-4)
        if not np.any(valid_pixels):
            c_r_list[idx] = np.nan
            c2_r_list[idx] = np.nan
            local_c_var_list[idx] = np.nan
            continue

        denom = np.where(v_mask > 1e-4, v_mask, 1.0)
        local_c = (u_x * conv_ux + u_y * conv_uy) / denom
        local_c_valid = local_c[valid_pixels]

        c_r_list[idx] = float(np.mean(local_c_valid))
        local_c_var_list[idx] = float(np.var(local_c_valid))

        local_c2_pair = (f1 * conv_f1 + f2 * conv_f2 + f3 * conv_f3) / denom
        c2_r_list[idx] = float(np.mean(local_c2_pair[valid_pixels]))

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

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _sample_dot_products_numba(
        ux: np.ndarray,
        uy: np.ndarray,
        valid_y: np.ndarray,
        valid_x: np.ndarray,
        valid_mask: np.ndarray,
        distances: np.ndarray,
        n_samples: int,
        half_w: float,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba 並列サンプリングカーネル"""
        np.random.seed(seed)
        num_d = len(distances)
        n_valid = len(valid_y)
        H, W = ux.shape

        out_c = np.empty((num_d, n_samples), dtype=np.float32)
        out_counts = np.zeros(num_d, dtype=np.int32)

        for d_idx in prange(num_d):
            r = distances[d_idx]
            r_min = max(0.5, r - half_w)
            r_max = r + half_w
            count = 0

            # 余裕を持ったループ回数
            max_trials = n_samples * 3
            for _ in range(max_trials):
                if count >= n_samples:
                    break
                idx = np.random.randint(0, n_valid)
                y0 = valid_y[idx]
                x0 = valid_x[idx]

                phi = np.random.uniform(0.0, 2.0 * np.pi)
                r_act = np.random.uniform(r_min, r_max)

                y1 = int(np.round(y0 + r_act * np.sin(phi)))
                x1 = int(np.round(x0 + r_act * np.cos(phi)))

                if y1 >= 0 and y1 < H and x1 >= 0 and x1 < W:
                    if valid_mask[y1, x1]:
                        u0_x = ux[y0, x0]
                        u0_y = uy[y0, x0]
                        u1_x = ux[y1, x1]
                        u1_y = uy[y1, x1]

                        mag0 = np.hypot(u0_x, u0_y)
                        mag1 = np.hypot(u1_x, u1_y)

                        if mag0 > 1e-4 and mag1 > 1e-4:
                            dot = (u0_x * u1_x + u0_y * u1_y) / (mag0 * mag1)
                            if dot > 1.0:
                                dot = 1.0
                            elif dot < -1.0:
                                dot = -1.0
                            out_c[d_idx, count] = np.float32(dot)
                            count += 1

            out_counts[d_idx] = count

        return out_c, out_counts


def _sample_dot_products_torch(
    ux: np.ndarray,
    uy: np.ndarray,
    valid_mask: np.ndarray,
    distances_px: List[int],
    n_samples_per_dist: int,
    half_w: float,
    t_device: "torch.device",
) -> Dict[int, np.ndarray]:
    """PyTorch GPU による超高速ベクトル化サンプリング"""
    H, W = ux.shape
    d_ux = torch.from_numpy(ux).to(t_device)
    d_uy = torch.from_numpy(uy).to(t_device)
    d_mask = torch.from_numpy(valid_mask).to(t_device)

    valid_y, valid_x = torch.where(d_mask)
    n_valid = len(valid_y)
    if n_valid < 10:
        return {r: np.empty(0, dtype=np.float32) for r in distances_px}

    samples_by_dist = {}
    n_fetch = int(n_samples_per_dist * 1.8)

    for r in distances_px:
        # ランダムインデックス生成
        sample_indices = torch.randint(0, n_valid, (n_fetch,), device=t_device)
        y0 = valid_y[sample_indices]
        x0 = valid_x[sample_indices]

        phi = torch.rand(n_fetch, device=t_device) * (2.0 * np.pi)
        r_min = max(0.5, r - half_w)
        r_max = r + half_w
        r_act = torch.rand(n_fetch, device=t_device) * (r_max - r_min) + r_min

        y1 = torch.round(y0.float() + r_act * torch.sin(phi)).long()
        x1 = torch.round(x0.float() + r_act * torch.cos(phi)).long()

        in_bounds = (y1 >= 0) & (y1 < H) & (x1 >= 0) & (x1 < W)
        if not torch.any(in_bounds):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        y0_b, x0_b = y0[in_bounds], x0[in_bounds]
        y1_b, x1_b = y1[in_bounds], x1[in_bounds]

        valid_pair = d_mask[y0_b, x0_b] & d_mask[y1_b, x1_b]
        if not torch.any(valid_pair):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        y0_v, x0_v = y0_b[valid_pair], x0_b[valid_pair]
        y1_v, x1_v = y1_b[valid_pair], x1_b[valid_pair]

        u0_x, u0_y = d_ux[y0_v, x0_v], d_uy[y0_v, x0_v]
        u1_x, u1_y = d_ux[y1_v, x1_v], d_uy[y1_v, x1_v]

        mag0 = torch.hypot(u0_x, u0_y)
        mag1 = torch.hypot(u1_x, u1_y)
        valid_norm = (mag0 > 1e-4) & (mag1 > 1e-4)

        if not torch.any(valid_norm):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        c_vals = (u0_x[valid_norm] * u1_x[valid_norm] + u0_y[valid_norm] * u1_y[valid_norm]) / (mag0[valid_norm] * mag1[valid_norm])
        c_vals = torch.clamp(c_vals, -1.0, 1.0)
        c_res = c_vals[:n_samples_per_dist].cpu().numpy().astype(np.float32)
        samples_by_dist[r] = c_res

    return samples_by_dist


def sample_field_dot_products(
    ux: np.ndarray,
    uy: np.ndarray,
    distances_px: List[int],
    n_samples_per_dist: int = 20000,
    shell_width: float = 2.0,
    valid_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    device: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    2次元ベクトル場から各距離 r におけるペア内積 c = u(x) · u(x+r) を均等かつ超高速（GPU/Numba対応）にサンプリングする。
    """
    H, W = ux.shape
    if valid_mask is None:
        mag = np.hypot(ux, uy)
        valid_mask = (mag > 1e-4)

    half_w = float(shell_width) / 2.0
    dev_type, t_device = _get_device(device)

    # 1. GPU (PyTorch CUDA) が使える場合
    if dev_type == 'cuda' and HAS_TORCH:
        try:
            with torch.no_grad():
                return _sample_dot_products_torch(
                    ux, uy, valid_mask.astype(bool), distances_px, n_samples_per_dist, half_w, t_device
                )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()

    # 2. CPU Numba 並列サンプリング
    if rng is None:
        rng = np.random.default_rng()

    valid_y, valid_x = np.where(valid_mask)
    n_valid = len(valid_y)
    if n_valid < 10:
        return {r: np.empty(0, dtype=np.float32) for r in distances_px}

    dist_arr = np.array(distances_px, dtype=np.float32)

    if HAS_NUMBA:
        seed = int(rng.integers(0, 1000000))
        out_c, out_counts = _sample_dot_products_numba(
            ux.astype(np.float32),
            uy.astype(np.float32),
            valid_y.astype(np.int32),
            valid_x.astype(np.int32),
            valid_mask.astype(bool),
            dist_arr,
            n_samples_per_dist,
            half_w,
            seed,
        )
        samples_by_dist = {}
        for idx, r in enumerate(distances_px):
            c_len = out_counts[idx]
            samples_by_dist[r] = out_c[idx, :c_len].copy()
        return samples_by_dist

    # 3. Vectorized NumPy fallback
    samples_by_dist = {}
    for r in distances_px:
        n_fetch = int(n_samples_per_dist * 1.5)
        sample_indices = rng.integers(0, n_valid, size=n_fetch)
        y0 = valid_y[sample_indices]
        x0 = valid_x[sample_indices]

        phi = rng.uniform(0, 2 * np.pi, size=n_fetch)
        r_actual = rng.uniform(max(0.5, r - half_w), r + half_w, size=n_fetch)

        y1 = np.round(y0 + r_actual * np.sin(phi)).astype(int)
        x1 = np.round(x0 + r_actual * np.cos(phi)).astype(int)

        in_bounds = (y1 >= 0) & (y1 < H) & (x1 >= 0) & (x1 < W)
        if not np.any(in_bounds):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        y0_b, x0_b = y0[in_bounds], x0[in_bounds]
        y1_b, x1_b = y1[in_bounds], x1[in_bounds]

        valid_pair = valid_mask[y0_b, x0_b] & valid_mask[y1_b, x1_b]
        if not np.any(valid_pair):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        y0_v, x0_v = y0_b[valid_pair], x0_b[valid_pair]
        y1_v, x1_v = y1_b[valid_pair], x1_b[valid_pair]

        u0_x, u0_y = ux[y0_v, x0_v], uy[y0_v, x0_v]
        u1_x, u1_y = ux[y1_v, x1_v], uy[y1_v, x1_v]

        mag0 = np.hypot(u0_x, u0_y)
        mag1 = np.hypot(u1_x, u1_y)
        valid_norm = (mag0 > 1e-4) & (mag1 > 1e-4)

        if not np.any(valid_norm):
            samples_by_dist[r] = np.empty(0, dtype=np.float32)
            continue

        c_vals = (u0_x[valid_norm] * u1_x[valid_norm] + u0_y[valid_norm] * u1_y[valid_norm]) / (mag0[valid_norm] * mag1[valid_norm])
        c_vals = np.clip(c_vals, -1.0, 1.0)
        samples_by_dist[r] = c_vals[:n_samples_per_dist].astype(np.float32)

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
# 3. 局所相関長 xi(x) の空間分布の直接評価 & 空間NGP (GPU / バッチ並列)
# ==============================================================================

if HAS_NUMBA:
    @njit(fastmath=True)
    def _fit_single_curve_numba(x_v: np.ndarray, y_v: np.ndarray, scale: float) -> Tuple[float, float, float]:
        """1本の相関減衰曲線に対する高速 Gauss-Newton 非線形フィッティング"""
        n = len(x_v)
        if n < 3 or y_v[0] <= 0.05:
            return np.nan, np.nan, np.nan

        # 初期値推定 (対数線形回帰)
        sum_w = 0.0
        sum_x = 0.0
        sum_xx = 0.0
        sum_y = 0.0
        sum_xy = 0.0

        for i in range(n):
            yi = y_v[i]
            if yi > 0.01:
                xi = x_v[i]
                ly = np.log(yi)
                sum_w += 1.0
                sum_x += xi
                sum_xx += xi * xi
                sum_y += ly
                sum_xy += xi * ly

        delta = sum_w * sum_xx - sum_x * sum_x
        if delta > 1e-7 and sum_w >= 2.0:
            b_init = -(sum_w * sum_xy - sum_x * sum_y) / delta
            log_a = (sum_xx * sum_y - sum_x * sum_xy) / delta
            a_est = np.exp(log_a)
            if b_init > 1e-4:
                xi_est = 1.0 / b_init
            else:
                xi_est = 10.0 * scale
        else:
            xi_est = 10.0 * scale
            a_est = y_v[0]

        a_est = min(max(a_est, 0.1), 1.5)
        xi_est = min(max(xi_est, 0.05 * scale), 200.0 * scale)

        # Gauss-Newton / LM 反復
        lam = 1e-3
        for _ in range(8):
            b = 1.0 / max(xi_est, 1e-6)
            h00, h11, h01, g0, g1 = 0.0, 0.0, 0.0, 0.0, 0.0

            for i in range(n):
                xi = x_v[i]
                yi = y_v[i]
                exp_term = np.exp(-b * xi)
                y_hat = a_est * exp_term
                res = y_hat - yi

                ja = exp_term
                jxi = a_est * xi * (1.0 / (xi_est * xi_est)) * exp_term

                h00 += ja * ja
                h11 += jxi * jxi
                h01 += ja * jxi
                g0 += ja * res
                g1 += jxi * res

            h00 += lam
            h11 += lam
            det = h00 * h11 - h01 * h01

            if det > 1e-9:
                da = -(h11 * g0 - h01 * g1) / det
                dxi = -(h00 * g1 - h01 * g0) / det

                a_est = min(max(a_est + da, 0.0), 1.5)
                xi_est = min(max(xi_est + dxi, 0.05 * scale), 200.0 * scale)
                if abs(da) < 1e-4 and abs(dxi) < 1e-4 * scale:
                    break

        # R^2 の算出
        ss_res = 0.0
        y_mean = 0.0
        for i in range(n):
            y_mean += y_v[i]
        y_mean /= n

        ss_tot = 0.0
        b_final = 1.0 / max(xi_est, 1e-6)
        for i in range(n):
            y_hat = a_est * np.exp(-b_final * x_v[i])
            ss_res += (y_v[i] - y_hat) ** 2
            ss_tot += (y_v[i] - y_mean) ** 2

        r2 = 1.0 - (ss_res / (ss_tot + 1e-9)) if ss_tot > 1e-9 else 0.0
        return xi_est, a_est, r2

    @njit(parallel=True, fastmath=True)
    def _fit_grid_numba(
        local_corr_grid: np.ndarray,
        distances_um: np.ndarray,
        fit_mask: np.ndarray,
        scale: float,
        min_r2: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba 並列グリッドフィッティング"""
        num_d, nY, nX = local_corr_grid.shape
        xi_map = np.full((nY, nX), np.nan, dtype=np.float32)
        amp_map = np.full((nY, nX), np.nan, dtype=np.float32)
        r2_map = np.full((nY, nX), np.nan, dtype=np.float32)

        x_fit_all = distances_um[fit_mask]
        num_fit = len(x_fit_all)

        for iy in prange(nY):
            for ix in range(nX):
                # 有効点の抽出
                n_valid = 0
                for k in range(num_fit):
                    val = local_corr_grid[fit_mask, iy, ix][k]
                    if not np.isnan(val):
                        n_valid += 1

                if n_valid < 3:
                    continue

                x_v = np.empty(n_valid, dtype=np.float32)
                y_v = np.empty(n_valid, dtype=np.float32)
                idx = 0
                for k in range(num_fit):
                    val = local_corr_grid[fit_mask, iy, ix][k]
                    if not np.isnan(val):
                        x_v[idx] = x_fit_all[k]
                        y_v[idx] = val
                        idx += 1

                xi_est, a_est, r2 = _fit_single_curve_numba(x_v, y_v, scale)
                if not np.isnan(xi_est):
                    if r2 >= min_r2 or (r2 < min_r2 and xi_est < 2.0 * scale):
                        xi_map[iy, ix] = np.float32(xi_est)
                        amp_map[iy, ix] = np.float32(a_est)
                        r2_map[iy, ix] = np.float32(r2)

        return xi_map, amp_map, r2_map


def _fit_grid_torch(
    corr_tensor: "torch.Tensor",
    x_fit: "torch.Tensor",
    scale: float,
    min_r2: float,
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch CUDA による全グリッド（数万点）一括バッチ非線形最小二乗フィッティング。
    """
    # corr_tensor shape: (num_fit, N_pts)
    num_fit, N_pts = corr_tensor.shape
    x = x_fit.view(num_fit, 1)  # (num_fit, 1)

    # 有効マスク
    valid_mask = ~torch.isnan(corr_tensor) & (corr_tensor > -0.5)
    valid_counts = valid_mask.sum(dim=0)  # (N_pts,)
    first_pt_valid = torch.where(valid_mask[0], corr_tensor[0], torch.tensor(0.0, device=device)) > 0.05
    fit_eligible = (valid_counts >= 3) & first_pt_valid

    # 初期化
    xi_out = torch.full((N_pts,), float('nan'), dtype=torch.float32, device=device)
    amp_out = torch.full((N_pts,), float('nan'), dtype=torch.float32, device=device)
    r2_out = torch.full((N_pts,), float('nan'), dtype=torch.float32, device=device)

    if not torch.any(fit_eligible):
        return xi_out.cpu().numpy(), amp_out.cpu().numpy(), r2_out.cpu().numpy()

    # 対象点のみ抽出
    sub_y = corr_tensor[:, fit_eligible]  # (num_fit, M)
    sub_mask = valid_mask[:, fit_eligible].float()  # (num_fit, M)
    M = sub_y.shape[1]

    # 対数線形回帰による初期値推定
    pos_mask = sub_mask * (sub_y > 0.01).float()
    safe_y = torch.where(pos_mask > 0, sub_y, torch.tensor(1.0, device=device))
    log_y = torch.log(safe_y) * pos_mask

    sw = pos_mask.sum(dim=0)
    sx = (pos_mask * x).sum(dim=0)
    sxx = (pos_mask * (x ** 2)).sum(dim=0)
    sy = log_y.sum(dim=0)
    sxy = (x * log_y).sum(dim=0)

    delta = sw * sxx - sx * sx
    valid_delta = (delta > 1e-7) & (sw >= 2)

    b_init = torch.where(valid_delta, -(sw * sxy - sx * sy) / torch.clamp(delta, min=1e-7), torch.tensor(1.0 / (10.0 * scale), device=device))
    log_a = torch.where(valid_delta, (sxx * sy - sx * sxy) / torch.clamp(delta, min=1e-7), torch.tensor(0.0, device=device))
    a_init = torch.where(valid_delta, torch.exp(log_a), sub_y[0])

    a = torch.clamp(a_init, 0.1, 1.5).view(1, M)
    xi = torch.clamp(1.0 / torch.clamp(b_init, min=1e-4), 0.05 * scale, 200.0 * scale).view(1, M)

    # Gauss-Newton バッチ反復 (6ステップ)
    lam = 1e-3
    for _ in range(6):
        b = 1.0 / torch.clamp(xi, min=1e-6)  # (1, M)
        exp_term = torch.exp(-b * x)  # (num_fit, M)
        y_hat = a * exp_term  # (num_fit, M)
        res = (y_hat - sub_y) * sub_mask  # (num_fit, M)

        ja = exp_term * sub_mask
        jxi = (a * x / torch.clamp(xi ** 2, min=1e-8)) * exp_term * sub_mask

        h00 = (ja * ja).sum(dim=0) + lam  # (M,)
        h11 = (jxi * jxi).sum(dim=0) + lam
        h01 = (ja * jxi).sum(dim=0)
        g0 = (ja * res).sum(dim=0)
        g1 = (jxi * res).sum(dim=0)

        det = h00 * h11 - h01 * h01
        det_safe = torch.clamp(det, min=1e-9)

        da = -(h11 * g0 - h01 * g1) / det_safe
        dxi = -(h00 * g1 - h01 * g0) / det_safe

        a = torch.clamp(a + da.view(1, M), 0.0, 1.5)
        xi = torch.clamp(xi + dxi.view(1, M), 0.05 * scale, 200.0 * scale)

    # R^2 算出
    b_fin = 1.0 / torch.clamp(xi, min=1e-6)
    y_hat = a * torch.exp(-b_fin * x)
    ss_res = (((sub_y - y_hat) * sub_mask) ** 2).sum(dim=0)
    y_mean = (sub_y * sub_mask).sum(dim=0) / torch.clamp(sub_mask.sum(dim=0), min=1.0)
    ss_tot = (((sub_y - y_mean.view(1, M)) * sub_mask) ** 2).sum(dim=0)
    r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-9))

    a_res = a.view(-1)
    xi_res = xi.view(-1)
    passed = (r2 >= min_r2) | ((r2 < min_r2) & (xi_res < 2.0 * scale))

    # 結果をテンソルに書き戻し
    xi_sub = torch.where(passed, xi_res, torch.tensor(float('nan'), device=device))
    a_sub = torch.where(passed, a_res, torch.tensor(float('nan'), device=device))
    r2_sub = torch.where(passed, r2, torch.tensor(float('nan'), device=device))

    xi_out[fit_eligible] = xi_sub
    amp_out[fit_eligible] = a_sub
    r2_out[fit_eligible] = r2_sub

    return xi_out.cpu().numpy(), amp_out.cpu().numpy(), r2_out.cpu().numpy()


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
    指数減衰 C(r) = a * exp(-r / xi) にフィッティングして局所相関長 xi(x) の空間マップを超高速に構築。
    """
    H, W = ux.shape
    num_d = len(distances_px)
    distances_um = np.array(distances_px, dtype=np.float32) * scale

    if valid_mask is None:
        mag = np.hypot(ux, uy)
        mask = (mag > 1e-4).astype(np.float32)
    else:
        mask = valid_mask.astype(np.float32)

    v_mag = np.hypot(ux, uy)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_x = np.where(v_mag > 1e-6, ux / v_mag, 0.0).astype(np.float32) * mask
        u_y = np.where(v_mag > 1e-6, uy / v_mag, 0.0).astype(np.float32) * mask

    dev_type, t_device = _get_device(device)

    # グリッド座標の作成
    y_coords = np.arange(0, H, grid_step)
    x_coords = np.arange(0, W, grid_step)
    nY = len(y_coords)
    nX = len(x_coords)

    # フィッティング対象距離マスク
    fit_mask = np.ones(num_d, dtype=bool)
    if max_fit_dist_um is not None:
        fit_mask = distances_um <= max_fit_dist_um

    convolver = FFTConvolver(
        shape=(H, W),
        sizes=distances_px,
        kernel_type=kernel_type,
        shell_width=shell_width,
        device=dev_type,
    )

    # -------------------------------------------------------------
    # GPU (PyTorch CUDA) による超高速バッチ実行
    # -------------------------------------------------------------
    if dev_type == 'cuda' and HAS_TORCH:
        try:
            with torch.no_grad():
                fields_np = np.stack([u_x, u_y, mask], axis=0)
                t_fields = torch.from_numpy(fields_np).to(t_device, non_blocking=True)
                fft_fields = torch.fft.rfft2(t_fields)

                d_ux = t_fields[0]
                d_uy = t_fields[1]
                d_mask = t_fields[2]

                # グリッド座標のテンソル
                t_y_coords = torch.from_numpy(y_coords).to(t_device)
                t_x_coords = torch.from_numpy(x_coords).to(t_device)

                local_corr_grid_t = torch.full((num_d, nY, nX), float('nan'), dtype=torch.float32, device=t_device)

                for idx, k_fft_cpu in enumerate(convolver.kernel_ffts_torch):
                    k_fft = k_fft_cpu.to(t_device, non_blocking=True)
                    conv_all = torch.fft.irfft2(fft_fields * k_fft, s=(H, W))

                    conv_ux = conv_all[0]
                    conv_uy = conv_all[1]
                    v_mask = conv_all[2]

                    denom = torch.where(v_mask > 1e-4, v_mask, torch.tensor(1.0, device=t_device))
                    c_field = (d_ux * conv_ux + d_uy * conv_uy) / denom
                    c_field = torch.where((d_mask > 0.5) & (v_mask > 1e-4), c_field, torch.tensor(float('nan'), device=t_device))

                    # グリッドサンプリング
                    local_corr_grid_t[idx] = c_field[t_y_coords[:, None], t_x_coords[None, :]]

                # GPU バッチ非線形フィッティング
                corr_sub = local_corr_grid_t[fit_mask].view(-1, nY * nX)  # (num_fit, N_pts)
                x_fit_t = torch.from_numpy(distances_um[fit_mask]).to(t_device)

                xi_arr, amp_arr, r2_arr = _fit_grid_torch(corr_sub, x_fit_t, scale, min_r2, t_device)

                xi_map = xi_arr.reshape(nY, nX)
                amp_map = amp_arr.reshape(nY, nX)
                r2_map = r2_arr.reshape(nY, nX)

                valid_xi = xi_map[~np.isnan(xi_map)]
                if len(valid_xi) >= 10:
                    xi2_mean = float(np.mean(valid_xi ** 2))
                    xi4_mean = float(np.mean(valid_xi ** 4))
                    alpha_2_xi = (xi4_mean / (3.0 * (xi2_mean ** 2))) - 1.0 if xi2_mean > 1e-8 else np.nan
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
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # CPU (マルチスレッド SciPy & Numba) によるフォールバック
    # -------------------------------------------------------------
    fields_np = np.stack([u_x, u_y, mask], axis=0)
    fft_fields = scipy.fft.rfft2(fields_np, axes=(-2, -1), workers=4)

    local_corr_grid = np.full((num_d, nY, nX), np.nan, dtype=np.float32)

    for idx, k_fft in enumerate(convolver.kernel_ffts_np):
        conv_all = scipy.fft.irfft2(fft_fields * k_fft, s=(H, W), axes=(-2, -1), workers=4)
        conv_ux = conv_all[0]
        conv_uy = conv_all[1]
        v_mask = conv_all[2]

        denom = np.where(v_mask > 1e-4, v_mask, 1.0)
        c_field = (u_x * conv_ux + u_y * conv_uy) / denom
        c_field = np.where((mask > 0.5) & (v_mask > 1e-4), c_field, np.nan)

        local_corr_grid[idx, :, :] = c_field[np.ix_(y_coords, x_coords)]

    if HAS_NUMBA:
        xi_map, amp_map, r2_map = _fit_grid_numba(local_corr_grid, distances_um, fit_mask, scale, min_r2)
    else:
        # SciPy curve_fit fallback
        xi_map = np.full((nY, nX), np.nan, dtype=np.float32)
        amp_map = np.full((nY, nX), np.nan, dtype=np.float32)
        r2_map = np.full((nY, nX), np.nan, dtype=np.float32)
        x_fit_all = distances_um[fit_mask]

        for iy in range(nY):
            for ix in range(nX):
                y_curve = local_corr_grid[fit_mask, iy, ix]
                valid_pt = ~np.isnan(y_curve)
                if np.count_nonzero(valid_pt) < 3 or y_curve[valid_pt][0] <= 0.05:
                    continue
                x_v = x_fit_all[valid_pt]
                y_v = y_curve[valid_pt]
                a_init = float(np.clip(y_v[0], 0.1, 1.0))
                p0 = [10.0 * scale, a_init]
                bounds = ([0.05 * scale, 0.0], [200.0 * scale, 1.5])
                try:
                    popt, _ = curve_fit(exp_decay_model, x_v, y_v, p0=p0, bounds=bounds, maxfev=400)
                    xi_est, a_est = popt
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

    valid_xi = xi_map[~np.isnan(xi_map)]
    if len(valid_xi) >= 10:
        xi2_mean = float(np.mean(valid_xi ** 2))
        xi4_mean = float(np.mean(valid_xi ** 4))
        alpha_2_xi = (xi4_mean / (3.0 * (xi2_mean ** 2))) - 1.0 if xi2_mean > 1e-8 else np.nan
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
    four_point_result: Optional[dict] = None,
    ngp_result: Optional[dict] = None,
    local_xi_result: Optional[dict] = None,
    tracks_data: Optional[Union[pd.DataFrame, str, Path]] = None,
    scale: float = 0.11,
    condition_name: str = "",
    save_path: Optional[Union[str, Path]] = None,
    track_color: str = '#FF007F',
    track_alpha: float = 0.75,
    track_lw: float = 0.9,
) -> plt.Figure:
    """
    1つの実験または条件に対して、局所相関長マップと相関長分布・空間NGPをまとめた
    2パネル (1x2) のサマリーダッシュボード図を作成する。

    構成 (1x2 パネル):
      [Panel A] 局所相関長 xi(x) の空間マップ + 貨物粒子の軌跡オーバーレイ
      [Panel B] 局所相関長分布 P(xi) と空間 NGP alpha_{2, xi}
    """
    from matplotlib.collections import LineCollection
    import matplotlib.lines as mlines

    if local_xi_result is None and four_point_result is not None and 'xi_map_um' in four_point_result:
        local_xi_result = four_point_result

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # -------------------------------------------------------------
    # Panel A: 局所相関長 xi(x) 空間マップ + 貨物粒子軌跡
    # -------------------------------------------------------------
    ax_a = axes[0]
    has_tracks = False
    if local_xi_result and 'xi_map_um' in local_xi_result:
        xi_map = local_xi_result['xi_map_um']
        gx = local_xi_result['grid_x_um']
        gy = local_xi_result['grid_y_um']

        if len(gx) > 1 and len(gy) > 1 and xi_map.size > 0:
            extent = [gx[0], gx[-1], gy[-1], gy[0]]
            vmax = float(np.nanpercentile(xi_map, 98)) if np.any(~np.isnan(xi_map)) else 20.0
            vmin = float(np.nanpercentile(xi_map, 2)) if np.any(~np.isnan(xi_map)) else 0.5

            im = ax_a.imshow(xi_map, extent=extent, cmap='viridis', vmin=max(0, vmin), vmax=max(vmin + 1, vmax), aspect='auto')
            cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
            cbar.set_label(r'Local Correlation Length $\xi(\mathbf{x})\ [\mu\mathrm{m}]$', fontsize=11)
            ax_a.set_xlabel(r'$x\ [\mu\mathrm{m}]$', fontsize=12)
            ax_a.set_ylabel(r'$y\ [\mu\mathrm{m}]$', fontsize=12)

            # 軌跡データのオーバーレイ描画 (Viridis と高コントラストなマゼンタ + 視認性を高める微細アウトライン)
            if tracks_data is not None:
                if isinstance(tracks_data, (str, Path)):
                    t_path = Path(tracks_data)
                    if t_path.exists():
                        try:
                            df_t = pd.read_csv(t_path)
                        except Exception:
                            df_t = None
                    else:
                        df_t = None
                elif isinstance(tracks_data, pd.DataFrame):
                    df_t = tracks_data
                else:
                    df_t = None

                if df_t is not None and not df_t.empty and 'x' in df_t.columns and 'y' in df_t.columns:
                    p_col = 'particle' if 'particle' in df_t.columns else ('track_id' if 'track_id' in df_t.columns else None)
                    if p_col:
                        lines = []
                        for _, group in df_t.groupby(p_col):
                            if len(group) > 1:
                                tx = group['x'].values * scale
                                ty = group['y'].values * scale
                                lines.append(np.column_stack([tx, ty]))
                        if lines:
                            # 視認性向上のための薄い黒縁取り (アウトライン)
                            lc_outline = LineCollection(lines, colors='black', alpha=0.35, linewidths=track_lw + 0.8, zorder=3)
                            ax_a.add_collection(lc_outline)
                            # 前面の鮮やかな軌跡線
                            lc_fg = LineCollection(lines, colors=track_color, alpha=track_alpha, linewidths=track_lw, zorder=4)
                            ax_a.add_collection(lc_fg)
                            has_tracks = True
                    else:
                        ax_a.plot(df_t['x'] * scale, df_t['y'] * scale, '.', color=track_color, alpha=track_alpha, ms=1.5, zorder=4)
                        has_tracks = True

            ax_a.set_xlim(extent[0], extent[1])
            ax_a.set_ylim(extent[2], extent[3])

            title_a = r'(a) Local Correlation Length Map $\xi(\mathbf{x})$'
            if has_tracks:
                title_a += ' & Cargo Trajectories'
                legend_line = mlines.Line2D([], [], color=track_color, alpha=0.9, lw=track_lw * 2.0, label='Cargo Trajectories')
                ax_a.legend(handles=[legend_line], loc='upper right', fontsize=9, framealpha=0.85, facecolor='white', edgecolor='#cccccc', labelcolor='#222222')

            ax_a.set_title(title_a, fontsize=13, fontweight='bold')
        else:
            ax_a.text(0.5, 0.5, 'Insufficient grid points', ha='center', va='center')
    else:
        ax_a.text(0.5, 0.5, 'No local correlation data', ha='center', va='center')

    # -------------------------------------------------------------
    # Panel B: 局所相関長分布 P(xi) & 空間 NGP alpha_{2, xi}
    # -------------------------------------------------------------
    ax_b = axes[1]
    if local_xi_result and 'xi_valid_um' in local_xi_result:
        valid_xi = local_xi_result['xi_valid_um']
        alpha_xi = local_xi_result.get('alpha_2_xi', np.nan)
        xi_mean = local_xi_result.get('xi_mean_um', np.nan)
        xi_std = local_xi_result.get('xi_std_um', np.nan)
        xi_median = local_xi_result.get('xi_median_um', np.nan)

        if len(valid_xi) > 0:
            counts, bins, _ = ax_b.hist(valid_xi, bins=25, density=True, color='#882255', alpha=0.65, edgecolor='black')
            ax_b.axvline(xi_mean, color='black', lw=2, ls='--', label=f'Mean $\\langle \\xi \\rangle = {xi_mean:.2f}\\,\\mu\\mathrm{{m}}$')
            ax_b.axvline(xi_median, color='blue', lw=1.5, ls=':', label=f'Median $= {xi_median:.2f}\\,\\mu\\mathrm{{m}}$')

            # KDE 曲線のプロット
            try:
                kde = stats.gaussian_kde(valid_xi)
                x_eval = np.linspace(np.min(valid_xi), np.max(valid_xi), 200)
                ax_b.plot(x_eval, kde(x_eval), color='#882255', lw=2.2)
            except Exception:
                pass

            # 空間 NGP の情報ボックス
            text_info = (
                f"$\\mathbf{{\\alpha_{{2, \\xi}} = {alpha_xi:.3f}}}$\n"
                f"$\\langle \\xi \\rangle = {xi_mean:.2f} \\pm {xi_std:.2f}\\,\\mu\\mathrm{{m}}$\n"
                f"$N_{{\\mathrm{{grid}}}} = {len(valid_xi)}$"
            )
            ax_b.text(
                0.95, 0.85, text_info,
                transform=ax_b.transAxes,
                ha='right', va='top',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="gray", lw=1.2)
            )

            ax_b.set_xlabel(r'Correlation Length $\xi\ [\mu\mathrm{m}]$', fontsize=12)
            ax_b.set_ylabel(r'Probability Density $P(\xi)$', fontsize=12)
            ax_b.set_title(r'(b) Distribution $P(\xi)$ & Spatial NGP $\alpha_{2, \xi}$', fontsize=13, fontweight='bold')
            ax_b.grid(True, linestyle=':', alpha=0.6)
            ax_b.legend(loc='upper right', bbox_to_anchor=(0.95, 0.60), fontsize=9.5)
        else:
            ax_b.text(0.5, 0.5, 'No valid correlation lengths', ha='center', va='center')
    else:
        ax_b.text(0.5, 0.5, 'No valid correlation lengths', ha='center', va='center')

    title_str = f"Spatial Heterogeneity Analysis: {condition_name}" if condition_name else "Spatial Heterogeneity Analysis"
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)
    try:
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    except Exception:
        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, wspace=0.25)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path = save_path.with_suffix('.svg')
        png_path = save_path.with_suffix('.png')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved heterogeneity dashboard to {svg_path} and {png_path}")

    return fig
