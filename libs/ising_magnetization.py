"""
libs/ising_magnetization.py
==========================

光学フロー（Optical Flow）から得た流速場 u(x, y, t) = (u_x, u_y) を用いて、
イジングスピン（Ising spin）

    sigma_i(t) = sign( u_i(t) . n )        n : 配向ベクトル（ディレクター）

を定義し、窓サイズ（window size）R のブロックごとの磁化

    M_Ising(R) = (1 / N_R) * sum_{i in block} sigma_i

の絶対値のアンサンブル平均 <|M_Ising(R)|> を計算するためのコアモジュールです。

期待されるスケーリング（2 次元）
--------------------------------
- 完全に配向がそろった場（sigma がすべて +1 または -1）: <|M(R)|> = 1（R に依存しない）
- 無秩序な場（sigma が空間的にランダム）              : <|M(R)|> ~ R^{-1}
- 臨界（2D Ising, beta/nu = 1/8 = 0.125）             : <|M(R)|> ~ R^{-1/8}

したがって <|M(R)|> vs R を両対数プロットし、その局所勾配

    p(R) = - d ln <|M(R)|> / d ln R

を見ることで、流速場が「どのスケールまで配向ドメイン（平行 / 反平行領域）を保つか」
（= 動的イジング的ドメイン構造）を定量化できます。

定義上の注意
------------
- sigma = sign(u . n) は n -> -n の置き換えで全画素の符号が反転する。したがって
  ブロック磁化の絶対値 |M| は n と -n で不変であり、軸の向き（head/tail）の不定性に
  依存しない。一方、符号付き平均 <sigma>（極性バイアス）は符号規約の影響を受ける。
- 窓内の平均は「有効画素（|u| が閾値以上かつマスク外）」のみで行い、有効画素の割合が
  min_valid_fraction 未満のブロックは無効（NaN）として除外する。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'integral_image',
    'default_window_sizes',
    'ising_spin_field',
    'block_magnetizations',
    'block_magnetization_stats',
    'magnetization_curve',
    'spin_fractions',
    'local_log_slope',
    'fit_power_law_exponent',
]


# =============================================================================
# 基本ユーティリティ
# =============================================================================

def integral_image(a: np.ndarray) -> np.ndarray:
    """
    2 次元配列の積分画像（累積和, summed-area table）を返す。

    Parameters
    ----------
    a : ndarray, shape (H, W)

    Returns
    -------
    ndarray, shape (H + 1, W + 1)
        C[i, j] = sum_{y < i, x < j} a[y, x] （左端 / 上端は 0 埋め）
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"integral_image expects a 2-D array, got shape {a.shape}")
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    c[1:, 1:] = a
    np.cumsum(c[1:, 1:], axis=0, out=c[1:, 1:])
    np.cumsum(c[1:, 1:], axis=1, out=c[1:, 1:])
    return c


def default_window_sizes(max_window: int, min_window: int = 1, n_steps: int = 24) -> np.ndarray:
    """
    既定の窓サイズ列（等比級数）を作る。

    Parameters
    ----------
    max_window : int
        最大窓サイズ（格子単位）。
    min_window : int
        最小窓サイズ（格子単位）。既定 1。
    n_steps : int
        分割数（重複を除いた整数列を返す）。

    Returns
    -------
    ndarray of int
        昇順・重複なしの窓サイズ。
    """
    max_window = int(max_window)
    min_window = max(1, int(min_window))
    if max_window < min_window:
        return np.empty(0, dtype=int)
    if max_window == min_window:
        return np.array([min_window], dtype=int)
    if n_steps <= 2:
        return np.unique(np.array([min_window, max_window], dtype=int))
    ws = np.geomspace(float(min_window), float(max_window), int(n_steps))
    ws = np.unique(np.round(ws).astype(int))
    return ws[(ws >= min_window) & (ws <= max_window)]


# =============================================================================
# イジングスピン場
# =============================================================================

def ising_spin_field(
    mx: np.ndarray,
    my: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    min_flow_mag: float = 0.0,
    exclude_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    流速場 u = (mx, my) とディレクター n = (cos theta, sin theta) からイジングスピン

        sigma = sign( u . n )

    を計算する。

    Parameters
    ----------
    mx, my : ndarray
        フロー流速場（x = 列方向, y = 行方向の成分）。
    cos_theta, sin_theta : ndarray or float
        ディレクターの成分。（全画素共通の大域軸ならスカラー、局所配向なら配列）
    min_flow_mag : float
        |u| がこの値以下の画素はスピン未定義として無効化する（既定 0 = |u| > 0 のみ有効）。
    exclude_mask : ndarray of bool, optional
        True の画素を無効化する（貨物粒子近傍マスク等）。

    Returns
    -------
    sigma : ndarray, shape (H, W)
        有効画素では +1 / -1、無効画素では 0。
    valid : ndarray of bool, shape (H, W)
        有効画素マスク（True = 有効）。
    """
    mx = np.asarray(mx, dtype=np.float64)
    my = np.asarray(my, dtype=np.float64)
    if mx.shape != my.shape:
        raise ValueError(f"mx and my must share the shape, got {mx.shape} and {my.shape}")

    ct = np.asarray(cos_theta, dtype=np.float64)
    st = np.asarray(sin_theta, dtype=np.float64)
    dot = mx * ct + my * st

    mag = np.hypot(mx, my)
    valid = np.isfinite(mx) & np.isfinite(my) & (mag > max(float(min_flow_mag), 0.0))
    if exclude_mask is not None:
        valid = valid & ~np.asarray(exclude_mask, dtype=bool)

    sigma = np.sign(dot)
    sigma[~valid] = 0.0
    return sigma, valid


# =============================================================================
# ブロック磁化 M_Ising(R)
# =============================================================================

def block_magnetizations(
    sigma: np.ndarray,
    valid: np.ndarray,
    window: int,
    step: Optional[int] = None,
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """
    スピン場を一辺 window 画素のブロックに区切り、ブロック磁化

        m_b = (1 / N_R^{(b)}) * sum_{i in block b} sigma_i

    を返す（和は有効画素のみ、N_R^{(b)} はブロック内の有効画素数）。

    Parameters
    ----------
    sigma : ndarray, shape (H, W)
        イジングスピン（無効画素は 0）。
    valid : ndarray of bool, shape (H, W)
        有効画素マスク。
    window : int
        ブロックの一辺（画素 = 間引き格子上の単位）。
    step : int, optional
        ブロック中心の間隔。None なら window（= 非重複タイル）。
    min_valid_fraction : float
        ブロックを有効とみなすのに必要な有効画素の割合（既定 0.5）。

    Returns
    -------
    ndarray, shape (n_y, n_x)
        各ブロックの磁化。窓が画像より大きい場合は shape (0, 0)。
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if sigma.shape != valid.shape:
        raise ValueError(f"sigma and valid must share the shape, got {sigma.shape} and {valid.shape}")

    rows, cols = sigma.shape
    w = int(window)
    if w < 1 or w > min(rows, cols):
        return np.empty((0, 0), dtype=np.float64)

    st = int(step) if step else w
    st = max(1, st)

    s = np.where(valid, sigma, 0.0)
    cs = integral_image(s)
    cv = integral_image(valid.astype(np.float64))

    ys = np.arange(0, rows - w + 1, st)
    xs = np.arange(0, cols - w + 1, st)
    cc = np.ix_(ys + w, xs + w)
    yy = np.ix_(ys, xs + w)
    xx = np.ix_(ys + w, xs)
    oo = np.ix_(ys, xs)

    num = cs[cc] - cs[yy] - cs[xx] + cs[oo]
    den = cv[cc] - cv[yy] - cv[xx] + cv[oo]

    ok = (den > 0.0) & (den >= float(min_valid_fraction) * float(w * w))
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(ok, num / np.where(den > 0.0, den, 1.0), np.nan)
    return m


def block_magnetization_stats(
    sigma: np.ndarray,
    valid: np.ndarray,
    window: int,
    step: Optional[int] = None,
    min_valid_fraction: float = 0.5,
) -> Dict[str, float]:
    """
    ブロック磁化の統計量（有効ブロック上の平均）を返す。

    Returns
    -------
    dict
        abs_mean      : <|m_b|>_b        （= M_Ising(R) の絶対値平均）
        signed_mean   : <m_b>_b
        squared_mean  : <m_b^2>_b
        n_blocks      : 有効ブロック数（int）
    """
    m = block_magnetizations(sigma, valid, window, step=step,
                             min_valid_fraction=min_valid_fraction)
    flat = m[np.isfinite(m)] if m.size else np.empty(0, dtype=np.float64)
    if flat.size == 0:
        return {'abs_mean': float('nan'), 'signed_mean': float('nan'),
                'squared_mean': float('nan'), 'n_blocks': 0}
    return {
        'abs_mean': float(np.mean(np.abs(flat))),
        'signed_mean': float(np.mean(flat)),
        'squared_mean': float(np.mean(flat ** 2)),
        'n_blocks': int(flat.size),
    }


def magnetization_curve(
    sigma: np.ndarray,
    valid: np.ndarray,
    windows: Sequence[int],
    overlap: float = 0.0,
    min_valid_fraction: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    複数の窓サイズに対する <|M_Ising(R)|> を 1 フレーム分まとめて計算する。

    Parameters
    ----------
    sigma, valid : ndarray
        ising_spin_field() の出力。
    windows : sequence of int
        窓サイズ列（間引き格子上の一辺）。
    overlap : float
        隣接ブロックの重なり率（0 = 非重複タイル, 0.5 = 50% 重複）。
    min_valid_fraction : float
        ブロックを有効とみなすのに必要な有効画素の割合。

    Returns
    -------
    dict of ndarray
        window / abs_mean / signed_mean / squared_mean / n_blocks
    """
    wins = np.atleast_1d(np.asarray(windows, dtype=int))
    n_w = int(wins.size)
    out = {
        'window': wins,
        'abs_mean': np.full(n_w, np.nan, dtype=np.float64),
        'signed_mean': np.full(n_w, np.nan, dtype=np.float64),
        'squared_mean': np.full(n_w, np.nan, dtype=np.float64),
        'n_blocks': np.zeros(n_w, dtype=np.int64),
    }
    if n_w == 0:
        return out

    ov = float(np.clip(overlap, 0.0, 0.95))
    for i, w in enumerate(wins):
        step = int(max(1, round(float(w) * (1.0 - ov))))
        stats = block_magnetization_stats(sigma, valid, int(w), step=step,
                                         min_valid_fraction=min_valid_fraction)
        out['abs_mean'][i] = stats['abs_mean']
        out['signed_mean'][i] = stats['signed_mean']
        out['squared_mean'][i] = stats['squared_mean']
        out['n_blocks'][i] = stats['n_blocks']
    return out


def spin_fractions(sigma: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """
    有効画素に対するスピンの占有比を返す。

    Returns
    -------
    dict
        frac_plus  : +1 スピンの割合
        frac_minus : -1 スピンの割合
        polar_bias : <sigma> = frac_plus - frac_minus（ディレクター軸方向の極性バイアス）
        n_valid    : 有効画素数（float）
    """
    sigma = np.asarray(sigma)
    valid = np.asarray(valid, dtype=bool)
    n_valid = float(np.count_nonzero(valid))
    if n_valid <= 0:
        return {'frac_plus': float('nan'), 'frac_minus': float('nan'),
                'polar_bias': float('nan'), 'n_valid': 0.0}
    s = sigma[valid]
    n_plus = float(np.count_nonzero(s > 0))
    n_minus = float(np.count_nonzero(s < 0))
    n_def = n_plus + n_minus
    if n_def <= 0:
        return {'frac_plus': float('nan'), 'frac_minus': float('nan'),
                'polar_bias': float('nan'), 'n_valid': n_valid}
    return {
        'frac_plus': n_plus / n_def,
        'frac_minus': n_minus / n_def,
        'polar_bias': (n_plus - n_minus) / n_def,
        'n_valid': n_valid,
    }


# =============================================================================
# スケーリング解析
# =============================================================================

def local_log_slope(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    両対数プロットの局所勾配 p = - d ln y / d ln x（隣接点の差分）を返す。

    Returns
    -------
    x_mid : ndarray
        隣接点の中点（対数軸上の中点 = 幾何平均）。
    p : ndarray
        局所指数。y <= 0 または非有限の点は除外する。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return np.empty(0), np.empty(0)
    lx, ly = np.log(x), np.log(y)
    dlx = np.diff(lx)
    dly = np.diff(ly)
    x_mid = np.sqrt(x[:-1] * x[1:])
    with np.errstate(invalid='ignore', divide='ignore'):
        p = -dly / dlx
    p = np.where(np.isfinite(p), p, np.nan)
    return x_mid, p


def fit_power_law_exponent(
    x: np.ndarray,
    y: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Tuple[float, float, float, int]:
    """
    y = A * x^{-p} を両対数最小二乗でフィッティングし、指数 p を返す。

    Returns
    -------
    p : float
        べき指数（y ~ x^{-p}）。フィット不能なら NaN。
    r2 : float
        決定係数（対数空間）。フィット不能なら NaN。
    amp : float
        振幅 A（y = A x^{-p} の係数）。
    n : int
        フィットに用いた点数。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if x_min is not None:
        keep &= x >= float(x_min)
    if x_max is not None:
        keep &= x <= float(x_max)
    xs, ys = x[keep], y[keep]
    n = int(xs.size)
    if n < 2:
        return float('nan'), float('nan'), float('nan'), n

    lx, ly = np.log(xs), np.log(ys)
    slope, intercept = np.polyfit(lx, ly, 1)
    p = -float(slope)
    amp = float(np.exp(intercept))
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return p, r2, amp, n
