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

局所ポーラーオーダーとの対応関係
--------------------------------
同じブロックに対して、向きだけを見た秩序変数（局所ポーラーオーダー）

    P(R) = | (1 / N_R) * sum_{i in block} u_hat_i | ,   u_hat_i = u_i / |u_i|

も同時に計算できる（libs/calc_local_polar.py の円形カーネル版と同じ「単位ベクトル平均の
ノルム」規約）。2 つの秩序変数は次の意味で相補的である。

- ±反平行な 2 状態場（u = ±(cos theta0, sin theta0) が混在）では
  P(R) = |1 - 2 f| = |M(R)| が厳密に成り立つ（f = 反平行ドメインの面積比）。
  このとき相対差 Delta(R) = |P - |M|| / P は 0 になり、「スピン秩序 = 向きの秩序」。
- 同符号側に角度広がり delta がある場合は P / |M| = 平均 cos delta < 1 となるため、
  Delta(R) は「符号はそろっているが向きは揃っていない度合い」を測る。

定義上の注意
------------
- sigma = sign(u . n) は n -> -n の置き換えで全画素の符号が反転する。したがって
  ブロック磁化の絶対値 |M| は n と -n で不変であり、軸の向き（head/tail）の不定性に
  依存しない。一方、符号付き平均 <sigma>（極性バイアス）は符号規約の影響を受ける。
  ポーラーオーダー P はディレクター n に一切依存しない。
- 窓内の平均は「有効画素（|u| が閾値以上かつマスク外）」のみで行い、有効画素の割合が
  min_valid_fraction 未満のブロックは無効（NaN）として除外する。磁化とポーラー
  オーダーは同一のブロック格子・同一の有効判定を共有するので、ブロック単位で対応する。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'integral_image',
    'default_window_sizes',
    'ising_spin_field',
    'unit_flow_components',
    'block_magnetizations',
    'block_magnetization_stats',
    'block_polar_orders',
    'block_polar_stats',
    'block_order_pairs',
    'magnetization_curve',
    'polar_order_curve',
    'spin_fractions',
    'local_log_slope',
    'fit_power_law_exponent',
    'relative_gap',
    'even_stride_indices',
    'pearson_correlation',
    'spearman_rank_correlation',
    'slope_through_origin',
    'paired_correlation_stats',
    'binned_median',
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
# 単位ベクトル場とブロックポーラーオーダー P(R)
# =============================================================================

def unit_flow_components(
    mx: np.ndarray,
    my: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    流速場を画素ごとの単位ベクトル u_hat = u / |u| に正規化する。

    ポーラーオーダー（向きの秩序）は流速の大きさに依存させないため、大きさを落として
    向きだけを使う。無効画素（|u| = 0 や NaN、valid=False）は 0 ベクトルになる。

    Parameters
    ----------
    mx, my : ndarray
        フロー流速場（x = 列方向, y = 行方向の成分）。
    valid : ndarray of bool, optional
        有効画素マスク。None なら |u| > 0 かつ有限な画素を有効とする。

    Returns
    -------
    ux, uy : ndarray
        単位ベクトル成分（無効画素は 0）。
    """
    mx = np.asarray(mx, dtype=np.float64)
    my = np.asarray(my, dtype=np.float64)
    if mx.shape != my.shape:
        raise ValueError(f"mx and my must share the shape, got {mx.shape} and {my.shape}")

    mag = np.hypot(mx, my)
    ok = np.isfinite(mag) & (mag > 0.0)
    if valid is not None:
        ok = ok & np.asarray(valid, dtype=bool)

    ux = np.zeros_like(mx, dtype=np.float64)
    uy = np.zeros_like(my, dtype=np.float64)
    np.divide(mx, mag, out=ux, where=ok)
    np.divide(my, mag, out=uy, where=ok)
    return ux, uy


def _block_offsets(n: int, window: int, step: Optional[int]) -> np.ndarray:
    """ブロック左上隅の座標列（間引き格子単位）。窓が大きすぎる場合は空配列。"""
    w = int(window)
    if w < 1 or w > int(n):
        return np.zeros(0, dtype=int)
    st = int(step) if step else w
    st = max(1, st)
    return np.arange(0, int(n) - w + 1, st)


def _block_reduce(cum: np.ndarray, ys: np.ndarray, xs: np.ndarray, window: int) -> np.ndarray:
    """積分画像 cum から、ブロック左上隅 (ys, xs) の総和を O(1) で取り出す。"""
    w = int(window)
    cc = np.ix_(ys + w, xs + w)
    yy = np.ix_(ys, xs + w)
    xx = np.ix_(ys + w, xs)
    oo = np.ix_(ys, xs)
    return cum[cc] - cum[yy] - cum[xx] + cum[oo]


def _valid_block_mask(den: np.ndarray, window: int, min_valid_fraction: float) -> np.ndarray:
    """ブロックを有効とみなすマスク（有効画素数 > 0 かつ割合が閾値以上）。"""
    return (den > 0.0) & (den >= float(min_valid_fraction) * float(int(window) ** 2))


def block_polar_orders(
    ux: np.ndarray,
    uy: np.ndarray,
    valid: np.ndarray,
    window: int,
    step: Optional[int] = None,
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """
    単位ベクトル場を一辺 window 画素のブロックに区切り、ブロックポーラーオーダー

        P_b = | (1 / N_R^{(b)}) * sum_{i in block b} u_hat_i |

    を返す（N_R^{(b)} はブロック内の有効画素数）。ブロック磁化 block_magnetizations と
    まったく同じブロック格子・同じ有効判定を使うので、要素ごとに対応する。

    Parameters
    ----------
    ux, uy : ndarray, shape (H, W)
        unit_flow_components() の出力（無効画素は 0）。
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
        各ブロックのポーラーオーダー（無効ブロックは NaN）。窓が画像より大きい場合は (0, 0)。
    """
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if ux.shape != uy.shape or ux.shape != valid.shape:
        raise ValueError("ux, uy, valid must share the shape, got "
                         f"{ux.shape}, {uy.shape}, {valid.shape}")

    rows, cols = ux.shape
    w = int(window)
    if w < 1 or w > min(rows, cols):
        return np.empty((0, 0), dtype=np.float64)

    ys = _block_offsets(rows, w, step)
    xs = _block_offsets(cols, w, step)

    cx = integral_image(np.where(valid, ux, 0.0))
    cy = integral_image(np.where(valid, uy, 0.0))
    cv = integral_image(valid.astype(np.float64))

    den = _block_reduce(cv, ys, xs, w)
    ok = _valid_block_mask(den, w, min_valid_fraction)
    sx = _block_reduce(cx, ys, xs, w)
    sy = _block_reduce(cy, ys, xs, w)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(ok, np.hypot(sx, sy) / np.where(den > 0.0, den, 1.0), np.nan)
    return p


def block_polar_stats(
    ux: np.ndarray,
    uy: np.ndarray,
    valid: np.ndarray,
    window: int,
    step: Optional[int] = None,
    min_valid_fraction: float = 0.5,
) -> Dict[str, float]:
    """
    ブロックポーラーオーダーの統計量（有効ブロック上の平均）を返す。

    Returns
    -------
    dict
        polar_mean : <P_b>_b（= <P(R)>）
        n_blocks   : 有効ブロック数（int）
    """
    p = block_polar_orders(ux, uy, valid, window, step=step,
                           min_valid_fraction=min_valid_fraction)
    flat = p[np.isfinite(p)] if p.size else np.empty(0, dtype=np.float64)
    if flat.size == 0:
        return {'polar_mean': float('nan'), 'n_blocks': 0}
    return {'polar_mean': float(np.mean(flat)), 'n_blocks': int(flat.size)}


def block_order_pairs(
    sigma: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    valid: np.ndarray,
    window: int,
    step: Optional[int] = None,
    min_valid_fraction: float = 0.5,
) -> Dict[str, object]:
    """
    同一のブロック格子で、イジング磁化とポーラーオーダーのペアを同時に計算する。

    ブロックごとに

        signed_m = (1 / N_R) * sum sigma_i      （符号付き磁化）
        abs_m    = |signed_m|
        polar    = | (1 / N_R) * sum u_hat_i |  （ポーラーオーダー）

    を返す（無効ブロックは NaN）。散布図・相関解析の 1 サンプル = 1 ブロックに対応する。

    Returns
    -------
    dict
        signed_m / abs_m / polar : ndarray, shape (n_y, n_x)
        n_blocks                 : 有効ブロック数（int）
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (sigma.shape == ux.shape == uy.shape == valid.shape):
        raise ValueError("sigma, ux, uy, valid must share the shape, got "
                         f"{sigma.shape}, {ux.shape}, {uy.shape}, {valid.shape}")

    rows, cols = sigma.shape
    w = int(window)
    empty = np.empty((0, 0), dtype=np.float64)
    if w < 1 or w > min(rows, cols):
        return {'signed_m': empty, 'abs_m': empty, 'polar': empty, 'n_blocks': 0}

    ys = _block_offsets(rows, w, step)
    xs = _block_offsets(cols, w, step)

    cs = integral_image(np.where(valid, sigma, 0.0))
    cx = integral_image(np.where(valid, ux, 0.0))
    cy = integral_image(np.where(valid, uy, 0.0))
    cv = integral_image(valid.astype(np.float64))

    den = _block_reduce(cv, ys, xs, w)
    ok = _valid_block_mask(den, w, min_valid_fraction)
    num = _block_reduce(cs, ys, xs, w)
    sx = _block_reduce(cx, ys, xs, w)
    sy = _block_reduce(cy, ys, xs, w)

    safe_den = np.where(den > 0.0, den, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        signed_m = np.where(ok, num / safe_den, np.nan)
        abs_m = np.where(ok, np.abs(num) / safe_den, np.nan)
        polar = np.where(ok, np.hypot(sx, sy) / safe_den, np.nan)
    return {'signed_m': signed_m, 'abs_m': abs_m, 'polar': polar,
            'n_blocks': int(np.count_nonzero(ok))}


def polar_order_curve(
    ux: np.ndarray,
    uy: np.ndarray,
    valid: np.ndarray,
    windows: Sequence[int],
    overlap: float = 0.0,
    min_valid_fraction: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    複数の窓サイズに対する <P(R)> を 1 フレーム分まとめて計算する。

    magnetization_curve() と完全に対応する（窓サイズ・間隔・有効判定が同一）。

    Returns
    -------
    dict of ndarray
        window / polar_mean / n_blocks
    """
    wins = np.atleast_1d(np.asarray(windows, dtype=int))
    n_w = int(wins.size)
    out = {
        'window': wins,
        'polar_mean': np.full(n_w, np.nan, dtype=np.float64),
        'n_blocks': np.zeros(n_w, dtype=np.int64),
    }
    if n_w == 0:
        return out

    ov = float(np.clip(overlap, 0.0, 0.95))
    for i, w in enumerate(wins):
        step = int(max(1, round(float(w) * (1.0 - ov))))
        stats = block_polar_stats(ux, uy, valid, int(w), step=step,
                                  min_valid_fraction=min_valid_fraction)
        out['polar_mean'][i] = stats['polar_mean']
        out['n_blocks'][i] = stats['n_blocks']
    return out


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

    ys = _block_offsets(rows, w, step)
    xs = _block_offsets(cols, w, step)

    s = np.where(valid, sigma, 0.0)
    cs = integral_image(s)
    cv = integral_image(valid.astype(np.float64))

    num = _block_reduce(cs, ys, xs, w)
    den = _block_reduce(cv, ys, xs, w)

    ok = _valid_block_mask(den, w, min_valid_fraction)
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


# =============================================================================
# ポーラーオーダーと磁化の比較（散布図・相関・相対差 Delta）
# =============================================================================

def relative_gap(polar: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """
    2 つの秩序変数の相対差

        Delta(R) = | <P(R)> - <|M_Ising(R)|> | / <P(R)>

    を返す。P = 0 または非有限の点は NaN。

    解釈
    ----
    - Delta = 0: スピンの符号秩序が「向きの秩序」と完全に一致（±反平行 2 状態場）。
    - Delta > 0: 符号はそろっているが向きが揃っていない（同符号側の角度広がり）。
    """
    p = np.asarray(polar, dtype=np.float64)
    m = np.asarray(mag, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(m) & (p > 0.0)
    safe = np.where(p > 0.0, p, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(ok, np.abs(safe - m) / safe, np.nan)
    return d


def even_stride_indices(n: int, max_count: int) -> np.ndarray:
    """
    0 .. n-1 から等間隔に最大 max_count 個の index を選ぶ（決定的・乱数不使用）。

    max_count <= 0 または max_count >= n なら全 index を返す。大量のブロックを
    空間的に均等に間引いて散布図・相関解析用のサンプルを作るために使う。
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=int)
    mc = int(max_count)
    if mc <= 0 or mc >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, mc)).astype(int))


def _finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """両方が有限な要素だけを残した (x, y) を返す。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must share the size, got {x.size} and {y.size}")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相関係数（非有限値は除外、分散 0 や n < 2 なら NaN）。"""
    xs, ys = _finite_pair(x, y)
    n = int(xs.size)
    if n < 2:
        return float('nan')
    xd = xs - np.mean(xs)
    yd = ys - np.mean(ys)
    denom = float(np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2)))
    if denom <= 0.0:
        return float('nan')
    return float(np.sum(xd * yd) / denom)


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """同順位を平均順位で処理した順位（1 始まり）を返す。"""
    a = np.asarray(a, dtype=np.float64)
    n = int(a.size)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = float(np.mean(ranks[order[i:j]]))
        i = j
    return ranks


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 順位相関係数（同順位は平均順位）。n < 2 なら NaN。"""
    xs, ys = _finite_pair(x, y)
    if xs.size < 2:
        return float('nan')
    return pearson_correlation(_average_ranks(xs), _average_ranks(ys))


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    原点を通る最小二乗直線 y = a * x の傾き a と、非中心 R^2 を返す。

    R^2（非中心）= 1 - sum((y - a x)^2) / sum(y^2)。原点を通すフィットの標準的な
    定義で、x -> y の比例性（P から |M| をどこまで予測できるか）を表す。
    """
    xs, ys = _finite_pair(x, y)
    if xs.size == 0:
        return float('nan'), float('nan')
    sxx = float(np.sum(xs ** 2))
    if sxx <= 0.0:
        return float('nan'), float('nan')
    a = float(np.sum(xs * ys) / sxx)
    ss_res = float(np.sum((ys - a * xs) ** 2))
    ss_tot = float(np.sum(ys ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return a, r2


def paired_correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    (x, y) のペア（1 サンプル = 1 ブロック）に対する相関統計をまとめて返す。

    Returns
    -------
    dict
        n             : 有効サンプル数
        pearson_r     : Pearson 相関係数
        spearman_rho  : Spearman 順位相関係数
        slope_origin  : 原点通過最小二乗の傾き（y = a x）
        r2_origin     : 原点通過フィットの非中心 R^2
        slope_ols     : 通常最小二乗の傾き
        intercept_ols : 通常最小二乗の切片
        r2_ols        : 通常最小二乗の決定係数（中心化）
        mean_x, mean_y: 平均値
    """
    xs, ys = _finite_pair(x, y)
    n = int(xs.size)
    out = {k: float('nan') for k in
           ('pearson_r', 'spearman_rho', 'slope_origin', 'r2_origin',
            'slope_ols', 'intercept_ols', 'r2_ols', 'mean_x', 'mean_y')}
    out['n'] = n
    if n == 0:
        return out
    out['mean_x'] = float(np.mean(xs))
    out['mean_y'] = float(np.mean(ys))
    out['pearson_r'] = pearson_correlation(xs, ys)
    out['spearman_rho'] = spearman_rank_correlation(xs, ys)
    a0, r20 = slope_through_origin(xs, ys)
    out['slope_origin'] = a0
    out['r2_origin'] = r20
    if n >= 2 and float(np.var(xs)) > 0.0:
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        out['slope_ols'] = float(slope)
        out['intercept_ols'] = float(intercept)
        out['r2_ols'] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return out


def binned_median(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    x_min: float = 0.0,
    x_max: float = 1.0,
    min_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x を等幅ビンに分け、ビンごとの y の中央値を返す（散布図のトレンド線用）。

    Returns
    -------
    centers : ndarray
        ビン中心。
    medians : ndarray
        y の中央値（サンプル数が min_count 未満のビンは NaN）。
    counts : ndarray of int
        ビンごとのサンプル数。
    """
    xs, ys = _finite_pair(x, y)
    nb = max(1, int(n_bins))
    lo, hi = float(x_min), float(x_max)
    if not (hi > lo):
        lo, hi = 0.0, 1.0
    edges = np.linspace(lo, hi, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(nb, np.nan, dtype=np.float64)
    counts = np.zeros(nb, dtype=np.int64)
    if xs.size == 0:
        return centers, medians, counts

    bin_idx = np.clip(np.digitize(xs, edges) - 1, 0, nb - 1)
    inside = (xs >= lo) & (xs <= hi)
    for b in range(nb):
        vals = ys[inside & (bin_idx == b)]
        counts[b] = int(vals.size)
        if vals.size >= max(1, int(min_count)):
            medians[b] = float(np.median(vals))
    return centers, medians, counts
