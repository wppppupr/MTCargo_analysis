#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ising_magnetization.py
===========================

光学フロー（Optical Flow, GFP_flows.h5）から得た微小管流速場 u(x, y, t) を用いて、
イジングスピン

    sigma(x, y, t) = sign( u(x, y, t) . n )

を定義し、窓サイズ（window size）R のブロック磁化

    M_Ising(R) = (1 / N_R) * sum_{i in block} sigma_i

の絶対値のアンサンブル平均 <|M_Ising(R)|> を「様々なレンジの窓サイズ」について計算し、
その窓サイズ依存性（両対数プロット + 局所スケーリング指数）を条件（貨物粒子径）ごとに
可視化するスクリプトです。

【ディレクター n の与え方（--director）】
- global（既定）: フレームごとの大域ネマチック主軸 n(t) = (cos theta_nem(t), sin theta_nem(t))。
  theta_nem(t) は MTs_im_theta.zarr（局所配向角マップ）の 2 テンソル平均
  theta_nem(t) = 0.5 * arctan2(<sin 2 theta>, <cos 2 theta>)
  で求め（libs.calc_bg_angular_correlation.load_nematic_thetas と同一定義）、
  zarr が無い場合はそのフレームのフロー配向から同じ定義で推定する。
- local: 各画素の局所配向 n(x, y, t)（MTs_im_theta.zarr を最近傍でフロー格子へリサンプル）。
  「微小管バンドルに沿った輸送の向き」をそのままスピン化する。

【角度規約（ミラー）の自動整合】
libs/AFT_tools.py は least-moment 角を -1 倍する実装のため、MTs_im_theta.zarr の角度規約は
光学フローの座標系（x = 列, y = 行, phi = arctan2(u_y, u_x)）とミラー関係にあるデータが
ある。本スクリプトは theta -> -theta の両方を同時に評価し、プール <cos 2 Delta theta>
（= フローがディレクター軸に沿っている度合い）が大きい側を --theta_sign auto（既定）で
自動採用する。採用符号は CSV・図タイトルに記録される。

  注意: |M_Ising(R)| は n -> -n の置き換えで不変（全画素のスピンが反転し |m_b| は同じ）
  であるため、global モードの <|M(R)|> はディレクターの符号規約に依存しない。符号規約が
  効くのは符号付き極性バイアス <sigma> と、ミラーが画素ごとの反射になる local モードである。

【窓サイズ（--window_sizes）】
元画像ピクセル単位で指定する（例: '8:512:8' = 8, 16, ..., 512 px、'64,128,256,512'）。
内部では間引き格子単位（window_px / pixel_stride）に換算し、pixel_stride の倍数へ丸める。
--window_sizes auto（既定）では 1 格子（= pixel_stride px）から min(H, W) / 2 までを
等比級数（--window_steps, 既定 24）で自動生成する。
ブロックは --window_overlap（既定 0 = 非重複タイル, 0.5 = 50% 重複）で走査する。

【出力ファイル】
1. ising_magnetization_vs_window.png/.svg        : <|M(R)|> vs R（両対数）+ 局所指数パネル
2. ising_magnetization_vs_window_linear.png/.svg : 同上（線形軸、狭いレンジの確認用）
3. ising_magnetization_per_experiment.png/.svg   : 条件別パネル（実験ごとの曲線 + 条件平均）
4. ising_polar_bias.png/.svg                     : 条件別の極性バイアス <sigma> と +1 スピン比
5. ising_spin_map_examples.png/.svg              : 各条件の代表スピン場 sigma(x, y)（検証用）
CSV:
  ising_magnetization_curve.csv          : 条件 x 窓サイズ（実験間平均 ± SEM + プール値）
  ising_magnetization_per_experiment.csv : 実験 x 窓サイズの生値
  ising_magnetization_summary.csv        : 条件ごとの代表値・べき指数フィット・極性バイアス

【データ源】
各実験ディレクトリの GFP_flows.h5（shape = (frame, 2, y, x) もしくは (frame, y, x, 2)）。
フレーム読み出しは plot_mt_orientation_distribution.FlowFrameReader を再利用する
（1 アクセスで 2ch 読み + 間引きキャッシュにより NAS の I/O を最小化）。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# NAS / 共有ボリュームでの HDF5 ファイルロックエラー防止（h5py import 前に設定が必要）
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# 親ディレクトリのパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import ising_magnetization as ising
import plot_mt_orientation_distribution as mt_ori

# スタイル適用
_style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if _style_path.exists():
    try:
        plt.style.use(str(_style_path))
    except Exception:
        pass

FLOW_NAME = mt_ori.FLOW_NAME
TRACKS_NAME = mt_ori.TRACKS_NAME
BEADS_INFO = mt_ori.BEADS_INFO
BEAD_LOOKUP = mt_ori.BEAD_LOOKUP

VARIANT_PLUS = 'plus'
VARIANT_MINUS = 'minus'
VARIANT_NAMES = (VARIANT_PLUS, VARIANT_MINUS)

ISING_CRITICAL_EXPONENT = 0.125   # 2D Ising: beta / nu = 1/8
UNCORRELATED_EXPONENT = 1.0       # 空間無相関なスピン: <|M(R)|> ~ R^{-1}


# =============================================================================
# 窓サイズ指定の解決
# =============================================================================

def parse_int_spec(spec: str) -> List[int]:
    """
    '8:512:8,1024' 形式（start:stop:step と単一値の混在）の整数指定をパースする。
    libs/spatial_heterogeneity.py の距離指定と同じ記法。
    """
    out = set()
    for token in str(spec).replace(',', ' ').split():
        if not token:
            continue
        if ':' in token:
            parts = token.split(':')
            start = int(round(float(parts[0])))
            stop = int(round(float(parts[1]))) if len(parts) > 1 else start
            step = int(round(float(parts[2]))) if len(parts) > 2 else 1
            step = max(1, step)
            out.update(range(start, stop + 1, step))
        else:
            out.add(int(round(float(token))))
    return sorted(v for v in out if v > 0)


def resolve_windows(
    spec: Optional[str],
    pixel_stride: int,
    max_window_grid: int,
    n_steps: int = 24,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, Optional[int]]]]:
    """
    窓サイズ指定（元画像 px）を間引き格子単位へ換算する。

    Returns
    -------
    windows_grid : ndarray of int
        間引き格子単位の窓サイズ（昇順・重複なし）。
    windows_px : ndarray of int
        実際に使用する窓サイズ（元画像 px = windows_grid * pixel_stride）。
    adjustments : list of (requested_px, actual_px or None)
        丸め / 除外された指定の記録（None は「大きすぎて除外」）。
    """
    st = max(1, int(pixel_stride))
    max_window_grid = max(1, int(max_window_grid))

    if spec is None or str(spec).strip().lower() in ('', 'auto', 'none'):
        grid = ising.default_window_sizes(max_window_grid, 1, max(2, int(n_steps)))
        return grid, grid * st, []

    adjustments: List[Tuple[int, Optional[int]]] = []
    grid_set = set()
    for w_px in parse_int_spec(spec):
        g = int(round(float(w_px) / float(st)))
        if g < 1:
            g = 1
            adjustments.append((int(w_px), int(g * st)))
        elif g > max_window_grid:
            adjustments.append((int(w_px), None))
            continue
        if int(w_px) != int(g * st):
            adjustments.append((int(w_px), int(g * st)))
        grid_set.add(g)

    grid = np.array(sorted(grid_set), dtype=int)
    return grid, grid * st, adjustments


def _nearest_indices(out_n: int, in_n: int, stride: int, full_n: int) -> np.ndarray:
    """
    間引きフロー格子（stride ごと）の各サンプル位置に対応する、リサンプル元配列の最近傍 index。
    """
    if out_n <= 0 or in_n <= 0:
        return np.zeros(0, dtype=int)
    pos = np.arange(int(out_n), dtype=np.float64) * float(max(1, int(stride)))
    idx = np.floor(pos * float(in_n) / float(max(int(full_n), 1))).astype(np.int64)
    return np.clip(idx, 0, int(in_n) - 1)


def load_local_director_maps(
    exp_dir: Path,
    frame_ids: Sequence[int],
    full_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    pixel_stride: int,
) -> Optional[np.ndarray]:
    """
    MTs_im_theta.zarr（局所配向角マップ）を、フローの間引き格子へ最近傍リサンプルして返す。

    Returns
    -------
    ndarray, shape (len(frame_ids), out_rows, out_cols) or None
        局所配向角 theta(x, y) [rad]。zarr が無い / 読めない場合は None。
    """
    path = Path(exp_dir) / "MTs_im_theta.zarr"
    if not path.exists():
        return None
    try:
        import zarr
        z = zarr.open_array(str(path), mode='r')
        n_t = int(z.shape[0])
        out_rows, out_cols = int(out_shape[0]), int(out_shape[1])
        full_rows, full_cols = int(full_shape[0]), int(full_shape[1])
        maps = np.full((len(frame_ids), out_rows, out_cols), np.nan, dtype=np.float32)
        idx_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for i, t in enumerate(frame_ids):
            tt = int(min(int(t), n_t - 1))
            arr = np.asarray(z[tt], dtype=np.float64)
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                continue
            in_rows, in_cols = int(arr.shape[0]), int(arr.shape[1])
            key = (in_rows, in_cols)
            if key not in idx_cache:
                idx_cache[key] = (
                    _nearest_indices(out_rows, in_rows, pixel_stride, full_rows),
                    _nearest_indices(out_cols, in_cols, pixel_stride, full_cols),
                )
            iy, ix = idx_cache[key]
            maps[i] = arr[np.ix_(iy, ix)]
        print(f"    Loaded local directors from {path.name} (shape {z.shape})", flush=True)
        return maps
    except Exception as e:
        print(f"    [WARNING] Failed to load local directors from {path.name}: {e}. "
              f"Falling back to the global nematic axis.", flush=True)
        return None


# =============================================================================
# 実験ディレクトリごとのイジング磁化集計
# =============================================================================

def _new_accumulator(n_w: int) -> dict:
    """1 つのディレクター符号バリアント用の累積バッファを作る。"""
    return {
        'sum_abs': np.zeros(n_w, dtype=np.float64),
        'sum_m': np.zeros(n_w, dtype=np.float64),
        'sum_m2': np.zeros(n_w, dtype=np.float64),
        'n_blocks': np.zeros(n_w, dtype=np.int64),
        'frame_abs': [[] for _ in range(n_w)],
        'frame_signed': [[] for _ in range(n_w)],
        'cos2_sum': 0.0,      # sum cos 2 Delta theta（符号判定用）
        'n_valid_sum': 0.0,   # 有効画素数（cos2 の分母）
        'n_plus': 0.0,
        'n_minus': 0.0,
    }


def _finalize_variant(
    acc: dict,
    windows_px: np.ndarray,
    windows_um: np.ndarray,
    n_frames_used: int,
) -> dict:
    """
    フレームごとの累積値から、実験レベルの <|M(R)|>（フレーム平均 ± SEM）と
    プール値（全ブロック・全フレーム平均）を確定する。
    """
    n_w = int(windows_px.size)
    abs_mean = np.full(n_w, np.nan)
    abs_sem = np.full(n_w, np.nan)
    signed_mean = np.full(n_w, np.nan)
    signed_sem = np.full(n_w, np.nan)
    pooled_abs = np.full(n_w, np.nan)
    pooled_signed = np.full(n_w, np.nan)
    squared_mean = np.full(n_w, np.nan)

    for wi in range(n_w):
        vals = np.asarray(acc['frame_abs'][wi], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            abs_mean[wi] = float(np.mean(vals))
            abs_sem[wi] = (float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                           if vals.size > 1 else 0.0)
        svals = np.asarray(acc['frame_signed'][wi], dtype=np.float64)
        svals = svals[np.isfinite(svals)]
        if svals.size:
            signed_mean[wi] = float(np.mean(svals))
            signed_sem[wi] = (float(np.std(svals, ddof=1) / np.sqrt(svals.size))
                              if svals.size > 1 else 0.0)
        nb = int(acc['n_blocks'][wi])
        if nb > 0:
            pooled_abs[wi] = float(acc['sum_abs'][wi] / nb)
            pooled_signed[wi] = float(acc['sum_m'][wi] / nb)
            squared_mean[wi] = float(acc['sum_m2'][wi] / nb)

    n_def = acc['n_plus'] + acc['n_minus']
    frac_plus = acc['n_plus'] / n_def if n_def > 0 else float('nan')
    frac_minus = acc['n_minus'] / n_def if n_def > 0 else float('nan')
    n_valid = float(acc['n_valid_sum'])
    cos2 = acc['cos2_sum'] / n_valid if n_valid > 0 else float('nan')

    return {
        'windows_px': windows_px,
        'windows_um': windows_um,
        'abs_mean': abs_mean,
        'abs_sem': abs_sem,
        'signed_mean': signed_mean,
        'signed_sem': signed_sem,
        'pooled_abs_mean': pooled_abs,
        'pooled_signed_mean': pooled_signed,
        'squared_mean': squared_mean,
        'n_blocks': acc['n_blocks'].astype(np.int64),
        'n_frames_used': int(n_frames_used),
        'frac_plus': float(frac_plus),
        'frac_minus': float(frac_minus),
        'polar_bias': float(frac_plus - frac_minus) if n_def > 0 else float('nan'),
        'nematic_order_cos2': float(cos2),
        'cos2_sum': float(acc['cos2_sum']),
        'n_valid_sum': float(n_valid),
    }


def process_experiment_ising(
    exp_dir: Path,
    bead: dict,
    windows_grid: np.ndarray,
    pixel_stride: int = 8,
    frame_stride: int = 5,
    max_frames_per_exp: Optional[int] = None,
    min_flow_mag: float = 1e-4,
    window_overlap: float = 0.0,
    min_valid_fraction: float = 0.5,
    mask_radius_factor: float = 0.0,
    min_mask_radius_px: float = 0.0,
    scale: float = 0.11,
    director: str = 'global',
    flow_cache: str = 'auto',
    flow_cache_name: Optional[str] = None,
    progress: bool = True,
) -> Optional[dict]:
    """
    1 つの実験ディレクトリの GFP_flows.h5 を読み、フレームごとにイジングスピン

        sigma = sign( u . n )

    を計算し、複数の窓サイズ R に対する <|M_Ising(R)|> を集計して返す。

    ディレクター符号（theta -> -theta のミラー）の両バリアントを同時に集計し、
    後段でプール <cos 2 Delta theta> が大きい側を採用できるようにする。

    Returns
    -------
    dict or None
        windows_px / windows_um、variants（{'plus': ..., 'minus': ...}）、
        スピン場の代表例（spin_example_plus / spin_example_minus）、
        theta_source, theta_series_rad, n_frames_used などを含む辞書。
        フローも有効なキャッシュも無い場合は None。
    """
    exp_dir = Path(exp_dir)
    st = max(1, int(pixel_stride))
    fs = max(1, int(frame_stride))
    windows_grid = np.atleast_1d(np.asarray(windows_grid, dtype=int))
    windows_px = (windows_grid * st).astype(np.int64)
    windows_um = windows_px.astype(np.float64) * float(scale)
    n_w = int(windows_grid.size)

    # 貨物粒子近傍マスク（半径 [px] = max(min_mask_radius_px, factor * R_c / scale)）
    mask_radius_px = float(max(float(min_mask_radius_px),
                               float(mask_radius_factor) * float(bead.get('radius_um', 0.0))
                               / float(scale)))
    cargo_pos: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if mask_radius_px > 0:
        tracks_path = exp_dir / TRACKS_NAME
        if tracks_path.exists():
            try:
                cargo_pos = mt_ori.cargo_positions_by_frame(pd.read_csv(tracks_path))
            except Exception as e:
                print(f"    [WARNING] could not read {tracks_path}: {e}", flush=True)
        else:
            print(f"    [WARNING] {TRACKS_NAME} not found; cargo masking disabled for "
                  f"{exp_dir.name}", flush=True)

    try:
        reader = mt_ori.FlowFrameReader(exp_dir, pixel_stride=st, frame_stride=fs,
                                        max_frames_per_exp=max_frames_per_exp,
                                        flow_cache=flow_cache, cache_name=flow_cache_name)
    except Exception as e:
        print(f"    [WARNING] cannot open flow for {exp_dir.name}: {e}", flush=True)
        return None

    with reader as src:
        if src.source is None:
            print(f"    [WARNING] cannot read flow for {exp_dir.name}: {src.error}", flush=True)
            return None

        n_frames_total = int(src.n_frames_total)
        shape = tuple(int(s) for s in (src.dataset_shape or ()))
        channel_first = bool(src.channel_first)
        frame_ids = [int(t) for t in src.frame_ids]
        full_rows, full_cols = int(src.rows), int(src.cols)
        out_shape = (int(src.out_rows), int(src.out_cols))

        # --- ディレクター（n）の決定 ---
        thetas: Optional[np.ndarray] = None
        theta_maps: Optional[np.ndarray] = None
        if str(director) == 'local':
            theta_maps = load_local_director_maps(
                exp_dir, frame_ids, (full_rows, full_cols), out_shape, st)
            if theta_maps is not None:
                theta_source = 'local:zarr'
            else:
                thetas = mt_ori.load_nematic_directors(exp_dir, frame_ids, n_frames_total)
                theta_source = 'local:zarr-fallback' if thetas is not None else 'local:flow-fallback'
        else:
            thetas = mt_ori.load_nematic_directors(exp_dir, frame_ids, n_frames_total)
            theta_source = 'global:zarr' if thetas is not None else 'global:flow'

        accs = {name: _new_accumulator(n_w) for name in VARIANT_NAMES}
        spin_example_plus: Optional[np.ndarray] = None
        spin_example_minus: Optional[np.ndarray] = None
        theta_series = np.full(len(frame_ids), np.nan, dtype=np.float64)
        n_frames_used = 0

        iterator = range(len(frame_ids))
        if progress:
            iterator = tqdm(iterator, desc=f"Ising {exp_dir.parent.name}/{exp_dir.name}",
                            leave=False)

        for i in iterator:
            t = frame_ids[i]
            mx, my = src.get(i)

            # --- 有効画素マスク（流速閾値 + 貨物粒子近傍） ---
            mag = np.hypot(mx, my)
            valid = np.isfinite(mx) & np.isfinite(my) & (mag > max(float(min_flow_mag), 0.0))
            if cargo_pos:
                pos = cargo_pos.get(t)
                if pos is not None:
                    exclude = mt_ori.make_cargo_mask(pos[0], pos[1], mx.shape, st, mask_radius_px)
                    valid = valid & ~exclude

            # --- ディレクター角 theta の決定 ---
            if theta_maps is not None:
                th = theta_maps[i]
                valid = valid & np.isfinite(th)
            elif thetas is not None:
                th = float(thetas[min(t, thetas.size - 1)])
                theta_series[i] = th
            else:
                phi_v = np.arctan2(my[valid], mx[valid])
                th = mt_ori.global_nematic_theta_from_angles(phi_v)
                theta_series[i] = th

            if not np.any(valid):
                continue
            n_frames_used += 1
            n_valid_frame = float(np.count_nonzero(valid))

            # --- スピン場 sigma = sign(u . n)（theta -> -theta の両バリアント） ---
            ct = np.cos(th)
            stt = np.sin(th)
            c2t = np.cos(2.0 * th)
            s2t = np.sin(2.0 * th)
            phi = np.arctan2(my, mx)
            c2f = np.cos(2.0 * phi)
            s2f = np.sin(2.0 * phi)

            dots = {
                VARIANT_PLUS: mx * ct + my * stt,      # n = (cos theta, sin theta)
                VARIANT_MINUS: mx * ct - my * stt,     # n = (cos theta, -sin theta)
            }
            cos2_terms = {
                VARIANT_PLUS: c2f * c2t + s2f * s2t,   # cos 2(phi - theta)
                VARIANT_MINUS: c2f * c2t - s2f * s2t,  # cos 2(phi + theta)
            }

            for name in VARIANT_NAMES:
                acc = accs[name]
                acc['n_valid_sum'] += n_valid_frame
                acc['cos2_sum'] += float(np.sum(cos2_terms[name][valid]))

                sigma = np.sign(dots[name])
                sigma[~valid] = 0.0
                acc['n_plus'] += float(np.count_nonzero(sigma > 0))
                acc['n_minus'] += float(np.count_nonzero(sigma < 0))

                if name == VARIANT_PLUS and spin_example_plus is None:
                    spin_example_plus = sigma
                elif name == VARIANT_MINUS and spin_example_minus is None:
                    spin_example_minus = sigma

                curve = ising.magnetization_curve(
                    sigma, valid, windows_grid,
                    overlap=window_overlap, min_valid_fraction=min_valid_fraction)
                for wi in range(n_w):
                    nb = int(curve['n_blocks'][wi])
                    if nb <= 0:
                        continue
                    a_val = float(curve['abs_mean'][wi])
                    s_val = float(curve['signed_mean'][wi])
                    acc['frame_abs'][wi].append(a_val)
                    acc['frame_signed'][wi].append(s_val)
                    acc['sum_abs'][wi] += a_val * nb
                    acc['sum_m'][wi] += s_val * nb
                    acc['sum_m2'][wi] += float(curve['squared_mean'][wi]) * nb
                    acc['n_blocks'][wi] += nb

    variants = {
        name: _finalize_variant(accs[name], windows_px, windows_um, n_frames_used)
        for name in VARIANT_NAMES
    }
    finite_theta = theta_series[np.isfinite(theta_series)]
    return {
        'exp_dir': str(exp_dir),
        'bead_name': bead['name'],
        'n_frames_used': int(n_frames_used),
        'n_frames_total': int(n_frames_total),
        'dataset_shape': shape,
        'channel_first': bool(channel_first),
        'grid_shape': out_shape,
        'pixel_stride': st,
        'director': str(director),
        'theta_source': theta_source,
        'theta_mean_rad': float(np.mean(finite_theta)) if finite_theta.size else float('nan'),
        'theta_series_rad': theta_series,
        'theta_frames': np.asarray(frame_ids, dtype=int),
        'flow_cache_source': src.source,
        'flow_cache_path': str(src.cache_path) if src.source == 'cache' else '',
        'mask_radius_px': float(mask_radius_px),
        'min_flow_mag': float(min_flow_mag),
        'windows_px': windows_px,
        'windows_um': windows_um,
        'variants': variants,
        'spin_example_plus': spin_example_plus,
        'spin_example_minus': spin_example_minus,
    }


# =============================================================================
# ディレクター符号（ミラー）の選択
# =============================================================================

def choose_director_sign(
    results: Sequence[dict],
    theta_sign: str = 'auto',
    margin: float = 0.05,
) -> Tuple[int, Dict[str, float]]:
    """
    プール <cos 2 Delta theta> を比較し、採用するディレクター符号を決める。

    theta -> -theta は座標系のミラーに対応し、片方だけが「フローがディレクター軸に
    沿う（Delta theta = 0 / pi）」を与える。両バリアントを同時に集計してあるので、
    ここで大きい側を選ぶ（--theta_sign +1 / -1 で強制指定も可能）。

    Returns
    -------
    sign : int
        +1 または -1。
    info : dict
        S2_plus / S2_minus / decision / n_experiments を含む判定情報。
    """
    num_plus = num_minus = den = 0.0
    n_exp = 0
    for r in results:
        vp = r['variants'][VARIANT_PLUS]
        vm = r['variants'][VARIANT_MINUS]
        nv = float(vp['n_valid_sum'])
        if nv <= 0:
            continue
        num_plus += float(vp['cos2_sum'])
        num_minus += float(vm['cos2_sum'])
        den += nv
        n_exp += 1

    s2_plus = num_plus / den if den > 0 else float('nan')
    s2_minus = num_minus / den if den > 0 else float('nan')
    info = {'S2_plus': s2_plus, 'S2_minus': s2_minus, 'n_experiments': n_exp}

    if str(theta_sign) in ('+1', '1', '+', 'plus'):
        info['decision'] = 'forced +1'
        return 1, info
    if str(theta_sign) in ('-1', '-', 'minus'):
        info['decision'] = 'forced -1'
        return -1, info

    if not np.isfinite(s2_plus) or not np.isfinite(s2_minus):
        info['decision'] = 'no data -> default +1'
        return 1, info
    if abs(s2_plus - s2_minus) < float(margin):
        info['decision'] = 'ambiguous (|S2+ - S2-| < margin) -> default +1'
        return 1, info
    if s2_minus > s2_plus:
        info['decision'] = 'mirrored convention detected -> -1'
        return -1, info
    info['decision'] = 'zarr convention matches the flow axes -> +1'
    return 1, info


def select_variant(result: dict, sign: int) -> dict:
    """採用符号に対応するバリアント（磁化カーブ）を返す。"""
    return result['variants'][VARIANT_PLUS if int(sign) >= 0 else VARIANT_MINUS]


def select_spin_example(result: dict, sign: int) -> Optional[np.ndarray]:
    """採用符号に対応する代表スピン場 sigma(x, y) を返す。"""
    return result['spin_example_plus' if int(sign) >= 0 else 'spin_example_minus']


def _sem(values: Sequence[float]) -> Tuple[float, float, int]:
    """平均 ± SEM（標本数 < 2 なら SEM = 0）を返す。"""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem, int(arr.size)


# =============================================================================
# 集計テーブル
# =============================================================================

def per_experiment_table(results: Sequence[dict], sign: int = 1) -> pd.DataFrame:
    """実験 x 窓サイズの生値テーブル（long 形式）。"""
    rows: List[dict] = []
    for r in results:
        v = select_variant(r, sign)
        windows_px = v['windows_px']
        windows_um = v['windows_um']
        for wi in range(len(windows_px)):
            if not np.isfinite(v['abs_mean'][wi]) and int(v['n_blocks'][wi]) == 0:
                continue
            rows.append({
                'bead_name': r['bead_name'],
                'exp_dir': r['exp_dir'],
                'director': r['director'],
                'theta_source': r['theta_source'],
                'theta_sign': int(1 if int(sign) >= 0 else -1),
                'n_frames_used': int(v['n_frames_used']),
                'n_frames_total': int(r['n_frames_total']),
                'pixel_stride': int(r['pixel_stride']),
                'grid_rows': int(r['grid_shape'][0]),
                'grid_cols': int(r['grid_shape'][1]),
                'window_px': int(windows_px[wi]),
                'window_um': float(windows_um[wi]),
                'window_grid': int(round(float(windows_px[wi]) / max(1, int(r['pixel_stride'])))),
                'abs_mean': float(v['abs_mean'][wi]),
                'abs_sem': float(v['abs_sem'][wi]),
                'signed_mean': float(v['signed_mean'][wi]),
                'signed_sem': float(v['signed_sem'][wi]),
                'pooled_abs_mean': float(v['pooled_abs_mean'][wi]),
                'pooled_signed_mean': float(v['pooled_signed_mean'][wi]),
                'squared_mean': float(v['squared_mean'][wi]),
                'n_blocks': int(v['n_blocks'][wi]),
                'frac_plus': float(v['frac_plus']),
                'frac_minus': float(v['frac_minus']),
                'polar_bias': float(v['polar_bias']),
                'nematic_order_cos2': float(v['nematic_order_cos2']),
            })
    return pd.DataFrame(rows)


def _condition_arrays(
    results: Sequence[dict],
    bead_name: str,
    sign: int,
) -> Optional[dict]:
    """1 条件分の窓サイズごとの代表値（実験間平均 ± SEM とプール値）をまとめる。"""
    rs = [r for r in results if r['bead_name'] == bead_name and r['n_frames_used'] > 0]
    if not rs:
        return None
    windows_px = np.asarray(rs[0]['windows_px'])
    windows_um = np.asarray(rs[0]['windows_um'])
    n_w = int(windows_px.size)

    out = {
        'windows_px': windows_px,
        'windows_um': windows_um,
        'abs_mean': np.full(n_w, np.nan),
        'abs_sem': np.full(n_w, np.nan),
        'signed_mean': np.full(n_w, np.nan),
        'signed_sem': np.full(n_w, np.nan),
        'pooled_abs_mean': np.full(n_w, np.nan),
        'pooled_signed_mean': np.full(n_w, np.nan),
        'squared_mean': np.full(n_w, np.nan),
        'n_blocks': np.zeros(n_w, dtype=np.int64),
        'n_experiments': np.zeros(n_w, dtype=int),
    }
    for wi in range(n_w):
        abs_vals: List[float] = []
        signed_vals: List[float] = []
        sum_abs = sum_m = sum_m2 = 0.0
        nb_tot = 0
        for r in rs:
            v = select_variant(r, sign)
            a = float(v['abs_mean'][wi])
            s = float(v['signed_mean'][wi])
            if np.isfinite(a):
                abs_vals.append(a)
            if np.isfinite(s):
                signed_vals.append(s)
            nb = int(v['n_blocks'][wi])
            if nb > 0:
                pa = float(v['pooled_abs_mean'][wi])
                ps = float(v['pooled_signed_mean'][wi])
                p2 = float(v['squared_mean'][wi])
                if np.isfinite(pa):
                    sum_abs += pa * nb
                if np.isfinite(ps):
                    sum_m += ps * nb
                if np.isfinite(p2):
                    sum_m2 += p2 * nb
                nb_tot += nb
        m, se, n = _sem(abs_vals)
        out['abs_mean'][wi], out['abs_sem'][wi], out['n_experiments'][wi] = m, se, n
        out['signed_mean'][wi], out['signed_sem'][wi], _ = _sem(signed_vals)
        if nb_tot > 0:
            out['pooled_abs_mean'][wi] = sum_abs / nb_tot
            out['pooled_signed_mean'][wi] = sum_m / nb_tot
            out['squared_mean'][wi] = sum_m2 / nb_tot
        out['n_blocks'][wi] = nb_tot
    return out


def condition_curve_table(
    results: Sequence[dict],
    beads: Sequence[dict],
    sign: int = 1,
) -> pd.DataFrame:
    """条件 x 窓サイズの <|M(R)|>（実験間平均 ± SEM とプール値）テーブル。"""
    rows: List[dict] = []
    for bead in beads:
        arr = _condition_arrays(results, bead['name'], sign)
        if arr is None:
            continue
        for wi in range(len(arr['windows_px'])):
            rows.append({
                'bead_name': bead['name'],
                'diameter_um': float(bead.get('diameter_um', np.nan)),
                'window_px': int(arr['windows_px'][wi]),
                'window_um': float(arr['windows_um'][wi]),
                'n_experiments': int(arr['n_experiments'][wi]),
                'abs_mean': float(arr['abs_mean'][wi]),
                'abs_sem': float(arr['abs_sem'][wi]),
                'signed_mean': float(arr['signed_mean'][wi]),
                'signed_sem': float(arr['signed_sem'][wi]),
                'pooled_abs_mean': float(arr['pooled_abs_mean'][wi]),
                'pooled_signed_mean': float(arr['pooled_signed_mean'][wi]),
                'squared_mean': float(arr['squared_mean'][wi]),
                'n_blocks_total': int(arr['n_blocks'][wi]),
            })
    return pd.DataFrame(rows)


def condition_summary_table(
    results: Sequence[dict],
    beads: Sequence[dict],
    sign: int = 1,
) -> pd.DataFrame:
    """
    条件ごとの代表値サマリー。

    べき指数は、条件平均カーブ <|M(R)|> vs R を全窓で両対数フィット（power_law_*）した値と、
    窓サイズの上位半分（大 R 側, ドメイン構造のスケーリング域）だけでフィットした値
    （power_law_upper_*）を併記する。
    """
    rows: List[dict] = []
    for bead in beads:
        rs = [r for r in results if r['bead_name'] == bead['name'] and r['n_frames_used'] > 0]
        arr = _condition_arrays(results, bead['name'], sign)
        if arr is None or not rs:
            continue

        p_all, r2_all, amp_all, n_all = ising.fit_power_law_exponent(
            arr['windows_um'], arr['abs_mean'])
        med = float(np.nanmedian(arr['windows_um'])) if arr['windows_um'].size else np.nan
        p_up, r2_up, amp_up, n_up = ising.fit_power_law_exponent(
            arr['windows_um'], arr['abs_mean'], x_min=med)

        pig = [float(select_variant(r, sign)['polar_bias']) for r in rs]
        fpig = [float(select_variant(r, sign)['frac_plus']) for r in rs]
        cos2 = [float(select_variant(r, sign)['nematic_order_cos2']) for r in rs]
        theta_src = sorted({str(r['theta_source']) for r in rs})
        pb_mean, pb_sem, _ = _sem(pig)
        fp_mean, fp_sem, _ = _sem(fpig)
        c2_mean, c2_sem, _ = _sem(cos2)

        rows.append({
            'bead_name': bead['name'],
            'diameter_um': float(bead.get('diameter_um', np.nan)),
            'marker': str(bead.get('marker', '')),
            'theta_sign': int(1 if int(sign) >= 0 else -1),
            'director': str(rs[0]['director']),
            'theta_source': '|'.join(theta_src),
            'n_experiments': len(rs),
            'n_frames_used': int(sum(int(select_variant(r, sign)['n_frames_used']) for r in rs)),
            'n_frames_total': int(sum(int(r['n_frames_total']) for r in rs)),
            'n_valid_pixels': float(sum(float(select_variant(r, sign)['n_valid_sum']) for r in rs)),
            'pixel_stride': int(rs[0]['pixel_stride']),
            'grid_rows': int(rs[0]['grid_shape'][0]),
            'grid_cols': int(rs[0]['grid_shape'][1]),
            'mask_radius_px': float(np.mean([float(r['mask_radius_px']) for r in rs])),
            'window_min_px': int(arr['windows_px'][0]) if arr['windows_px'].size else 0,
            'window_max_px': int(arr['windows_px'][-1]) if arr['windows_px'].size else 0,
            'window_min_um': float(arr['windows_um'][0]) if arr['windows_um'].size else np.nan,
            'window_max_um': float(arr['windows_um'][-1]) if arr['windows_um'].size else np.nan,
            'abs_mean_at_min_R': float(arr['abs_mean'][0]) if arr['abs_mean'].size else np.nan,
            'abs_mean_at_max_R': float(arr['abs_mean'][-1]) if arr['abs_mean'].size else np.nan,
            'pooled_abs_mean_at_min_R': (float(arr['pooled_abs_mean'][0])
                                         if arr['pooled_abs_mean'].size else np.nan),
            'pooled_abs_mean_at_max_R': (float(arr['pooled_abs_mean'][-1])
                                         if arr['pooled_abs_mean'].size else np.nan),
            'power_law_exponent': p_all,
            'power_law_r2': r2_all,
            'power_law_amp': amp_all,
            'power_law_n': int(n_all),
            'power_law_upper_exponent': p_up,
            'power_law_upper_r2': r2_up,
            'power_law_upper_n': int(n_up),
            'polar_bias_mean': pb_mean,
            'polar_bias_sem': pb_sem,
            'frac_plus_mean': fp_mean,
            'frac_plus_sem': fp_sem,
            'nematic_order_cos2_mean': c2_mean,
            'nematic_order_cos2_sem': c2_sem,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 作図
# =============================================================================

def _anchored_power_law(x: np.ndarray, x0: float, y0: float, exponent: float) -> np.ndarray:
    """基準点 (x0, y0) を通り傾き -p の両対数直線 y = y0 * (x / x0)^{-p} を返す。"""
    return float(y0) * (np.asarray(x, dtype=float) / float(x0)) ** (-float(exponent))


def _compact_log_ticks(ax, which: str = 'xy', numticks: int = 12) -> None:
    """
    対数軸の目盛りラベルを簡潔な数値表記（'2', '10', '50' など）に整える。

    既定スタイルは 20 pt フォントのため、10^k 表記のラベルが窓サイズ軸で重なる。
    主要目盛りを 1 / 2 / 5 × 10^k に置き、ScalarFormatter で素の数値を表示、
    副目盛りのラベルは消す（縦横とも 1 桁に収まり重ならない）。
    """
    import matplotlib.ticker as mticker
    for axis, key in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
        if key not in which:
            continue
        axis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                                  numticks=int(numticks)))
        axis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        axis.set_minor_formatter(mticker.NullFormatter())


def _bead_label(bead: dict) -> str:
    return f"{bead['name']} ({bead.get('diameter_um', float('nan')):.2f} μm)"


def _sorted_bead_df(df: pd.DataFrame, bead: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df['bead_name'] == bead['name']]
    return d.sort_values('window_um') if not d.empty else d


def plot_magnetization_vs_window(
    df_curve: pd.DataFrame,
    beads: Sequence[dict],
    out_dirs: Sequence[Path],
    sign_note: str = '',
    basename: str = 'ising_magnetization_vs_window',
) -> None:
    """
    <|M_Ising(R)|> vs 窓サイズ R（両対数）と、その局所スケーリング指数

        p(R) = - d ln <|M(R)|> / d ln R

    を条件（貨物粒子径）ごとに重ねて描画する。
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 7.8), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.0]})
    ax, ax2 = axes
    guide: Optional[Tuple[np.ndarray, np.ndarray]] = None

    for bead in beads:
        df = _sorted_bead_df(df_curve, bead)
        if df.empty:
            continue
        x = df['window_um'].to_numpy(dtype=float)
        y = df['abs_mean'].to_numpy(dtype=float)
        ye = df['abs_sem'].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if not np.any(ok):
            continue
        color = bead.get('color', None)
        marker = bead.get('marker', 'o')
        ax.errorbar(
            x[ok], y[ok], yerr=np.where(np.isfinite(ye[ok]), ye[ok], 0.0),
            marker=marker, ms=5.0, lw=1.5, color=color, elinewidth=1.0,
            capsize=2.0, label=_bead_label(bead), zorder=4)
        low = np.maximum(y[ok] - np.where(np.isfinite(ye[ok]), ye[ok], 0.0), y[ok] * 0.3)
        high = y[ok] + np.where(np.isfinite(ye[ok]), ye[ok], 0.0)
        ax.fill_between(x[ok], low, high, color=color, alpha=0.15, lw=0, zorder=3)
        if guide is None:
            guide = (x[ok], y[ok])

        xm, p = ising.local_log_slope(x, y)
        if xm.size:
            ax2.plot(xm, p, marker=marker, ms=4.0, lw=1.4, color=color, alpha=0.9)

    if guide is not None:
        x0 = np.geomspace(max(float(np.nanmin(guide[0])), 1e-6),
                          float(np.nanmax(guide[0])), 60)
        anchor_x = float(guide[0][0])
        anchor_y = float(guide[1][0])
        for expo, label, ls in (
            (UNCORRELATED_EXPONENT, r'$R^{-1}$ (space-uncorrelated)', ':'),
            (ISING_CRITICAL_EXPONENT, r'$R^{-1/8}$ (2D Ising critical)', '--'),
        ):
            ax.plot(x0, _anchored_power_law(x0, anchor_x, anchor_y, expo),
                    ls=ls, lw=1.3, color='0.35', zorder=1, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelbottom=False)   # sharex で下段のみに表示
    _compact_log_ticks(ax, 'xy')
    ax.set_ylabel(r'$\langle |M_{\mathrm{Ising}}(R)| \rangle$')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    title = ('Ising spin magnetization from the MT optical flow  '
             r'$\sigma_i = \mathrm{sign}(\mathbf{u}_i \cdot n)$')
    if sign_note:
        title += f"\n{sign_note}"
    ax.set_title(title, fontsize=10)

    ax2.axhline(UNCORRELATED_EXPONENT, ls=':', color='0.35', lw=1.3)
    ax2.axhline(ISING_CRITICAL_EXPONENT, ls='--', color='0.35', lw=1.3)
    ax2.text(0.99, UNCORRELATED_EXPONENT, r' $p=1$',
             transform=ax2.get_yaxis_transform(), va='bottom', ha='right',
             fontsize=8, color='0.35')
    ax2.text(0.99, ISING_CRITICAL_EXPONENT, r' $p=1/8$',
             transform=ax2.get_yaxis_transform(), va='bottom', ha='right',
             fontsize=8, color='0.35')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Window size $R$ [μm]')
    ax2.set_ylabel(r'local slope $p(R)$')
    ax2.grid(True, which='both', alpha=0.25)
    _compact_log_ticks(ax2, 'x')

    # スタイル (figure.autolayout: True) がレイアウトを自動調整する。GridSpec に
    # hspace / wspace などのサブプロットパラメータを直接渡すと autolayout が当該
    # サブプロットを非互換と判定して警告を出すため、間隔は autolayout に任せる。
    mt_ori.save_figure_to_all(fig, basename, list(out_dirs))
    plt.close(fig)


def plot_magnetization_vs_window_linear(
    df_curve: pd.DataFrame,
    beads: Sequence[dict],
    out_dirs: Sequence[Path],
    sign_note: str = '',
    basename: str = 'ising_magnetization_vs_window_linear',
) -> None:
    """<|M_Ising(R)|> vs R を線形軸で描画する（狭い窓サイズ域の確認用）。"""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for bead in beads:
        df = _sorted_bead_df(df_curve, bead)
        if df.empty:
            continue
        x = df['window_um'].to_numpy(dtype=float)
        y = df['abs_mean'].to_numpy(dtype=float)
        ye = df['abs_sem'].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not np.any(ok):
            continue
        ax.errorbar(x[ok], y[ok], yerr=np.where(np.isfinite(ye[ok]), ye[ok], 0.0),
                    marker=bead.get('marker', 'o'), ms=5.0, lw=1.5,
                    color=bead.get('color', None), elinewidth=1.0, capsize=2.0,
                    label=_bead_label(bead))
    ax.set_xlabel(r'Window size $R$ [μm]')
    ax.set_ylabel(r'$\langle |M_{\mathrm{Ising}}(R)| \rangle$')
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    if sign_note:
        ax.set_title(sign_note, fontsize=9)
    fig.tight_layout()
    mt_ori.save_figure_to_all(fig, basename, list(out_dirs))
    plt.close(fig)


def plot_per_experiment(
    df_exp: pd.DataFrame,
    beads: Sequence[dict],
    out_dirs: Sequence[Path],
    sign_note: str = '',
    ncols: int = 3,
    basename: str = 'ising_magnetization_per_experiment',
) -> None:
    """条件ごとのパネルに、実験ごとの <|M(R)|> 曲線と条件平均 ± SEM を描画する。"""
    selected = [b for b in beads if not _sorted_bead_df(df_exp, b).empty]
    if not selected:
        return
    ncols = max(1, min(int(ncols), len(selected)))
    nrows = int(np.ceil(len(selected) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.3 * nrows),
                             squeeze=False)
    for k, bead in enumerate(selected):
        ax = axes[k // ncols][k % ncols]
        d = _sorted_bead_df(df_exp, bead)
        color = bead.get('color', 'C0')
        for _, g in d.groupby('exp_dir'):
            g = g.sort_values('window_um')
            x = g['window_um'].to_numpy(dtype=float)
            y = g['abs_mean'].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if np.any(ok):
                ax.plot(x[ok], y[ok], '-', lw=1.0, color=color, alpha=0.35)
        grp = d.groupby('window_um')['abs_mean']
        xm = grp.mean().index.to_numpy(dtype=float)
        ym = grp.mean().to_numpy(dtype=float)
        cnt = grp.count().to_numpy(dtype=float)
        sd = grp.std(ddof=1).to_numpy(dtype=float) if len(d) > 1 else np.zeros_like(ym)
        yerr = np.where(cnt > 1, np.nan_to_num(sd) / np.sqrt(np.maximum(cnt, 1.0)), 0.0)
        okm = np.isfinite(xm) & np.isfinite(ym) & (ym > 0)
        if np.any(okm):
            ax.errorbar(xm[okm], ym[okm], yerr=yerr[okm], marker=bead.get('marker', 'o'),
                        ms=5.0, lw=2.0, color=color, elinewidth=1.0, capsize=2.0,
                        label='condition mean ± SEM', zorder=4)
            ax.fill_between(xm[okm], np.maximum(ym[okm] - yerr[okm], ym[okm] * 0.3),
                            ym[okm] + yerr[okm], color=color, alpha=0.15, lw=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        _compact_log_ticks(ax, 'xy')
        ax.set_title(_bead_label(bead), fontsize=9)
        ax.grid(True, which='both', alpha=0.22)
        ax.set_xlabel(r'$R$ [μm]')
        ax.set_ylabel(r'$\langle |M(R)| \rangle$')
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(len(selected), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    if sign_note:
        fig.suptitle(sign_note, fontsize=9)
    fig.tight_layout()
    mt_ori.save_figure_to_all(fig, basename, list(out_dirs))
    plt.close(fig)


def plot_polar_bias(
    df_summary: pd.DataFrame,
    out_dirs: Sequence[Path],
    sign_note: str = '',
    basename: str = 'ising_polar_bias',
) -> None:
    """条件別の極性バイアス <sigma> と +1 スピン比を棒グラフで描画する。"""
    if df_summary is None or df_summary.empty:
        return
    d = df_summary.sort_values('diameter_um') if 'diameter_um' in df_summary.columns else df_summary
    labels = [str(v) for v in d['bead_name']]
    xpos = np.arange(len(d), dtype=float)
    colors = [BEAD_LOOKUP.get(name, {}).get('color', 'C0') for name in labels]

    fig, axes = plt.subplots(1, 2, figsize=(4.6 * 2, 4.4))
    ax = axes[0]
    ax.bar(xpos, d['polar_bias_mean'].to_numpy(dtype=float), yerr=d['polar_bias_sem'].to_numpy(dtype=float),
           color=colors, edgecolor='black', linewidth=0.6, capsize=3.0, alpha=0.85)
    ax.axhline(0.0, color='black', lw=1.0)
    ax.axhline(1.0, color='0.5', ls='--', lw=1.0)
    ax.axhline(-1.0, color='0.5', ls='--', lw=1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r'$\langle \sigma \rangle$ (polar bias along $n$)')
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax2 = axes[1]
    ax2.bar(xpos, d['frac_plus_mean'].to_numpy(dtype=float), yerr=d['frac_plus_sem'].to_numpy(dtype=float),
            color=colors, edgecolor='black', linewidth=0.6, capsize=3.0, alpha=0.85)
    ax2.axhline(0.5, color='black', lw=1.0, ls='--')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel(r'fraction of $\sigma = +1$')
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')

    if sign_note:
        fig.suptitle(sign_note, fontsize=9)
    fig.tight_layout()
    mt_ori.save_figure_to_all(fig, basename, list(out_dirs))
    plt.close(fig)


def plot_spin_map_examples(
    examples: Dict[str, Tuple[np.ndarray, str]],
    out_dirs: Sequence[Path],
    sign_note: str = '',
    ncols: int = 3,
    basename: str = 'ising_spin_map_examples',
) -> None:
    """
    各条件の代表スピン場 sigma(x, y) ∈ {-1, 0, +1} を可視化する（定義の検証用）。

    examples は {bead_name: (sigma, ラベル)} の辞書。sigma = 0 の画素は無効画素。
    """
    items = [(k, v) for k, v in examples.items() if v is not None and v[0] is not None]
    if not items:
        return
    ncols = max(1, min(int(ncols), len(items)))
    nrows = int(np.ceil(len(items) / float(ncols)))
    # colorbar 用の列を GridSpec 側に確保する（専用 Axes も subplotspec を持つので
    # figure.autolayout と競合せず、パネルに重ならない）。wspace / hspace などの
    # サブプロットパラメータを GridSpec に直接渡すと autolayout が無効化されるため、
    # ここでは width_ratios のみを与えて間隔は autolayout に任せる。
    fig = plt.figure(figsize=(3.6 * ncols + 0.7, 3.2 * nrows))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1.0] * ncols + [0.04])
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cmap = plt.get_cmap('RdBu_r').with_extremes(bad='0.85')
    im = None
    for k, (bead_name, (sigma, label)) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        arr = np.asarray(sigma, dtype=float)
        arr = np.where(arr == 0.0, np.nan, arr)
        im = ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation='nearest')
        ax.set_title(f"{bead_name}\n{label}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(len(items), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    if im is not None:
        cbar = fig.colorbar(im, cax=fig.add_subplot(gs[:, ncols]))
        cbar.set_label(r'$\sigma = \mathrm{sign}(\mathbf{u} \cdot n)$')
    fig.suptitle((sign_note + '\n' if sign_note else '') +
                 'Representative Ising spin field (0 = invalid pixel)', fontsize=9)
    mt_ori.save_figure_to_all(fig, basename, list(out_dirs))
    plt.close(fig)


# =============================================================================
# 画像サイズのプローブ（窓サイズ auto の上限決定用）と main
# =============================================================================

def probe_grid_shape(
    exp_dir: Path,
    pixel_stride: int,
    frame_stride: int,
    max_frames_per_exp: Optional[int] = None,
    flow_cache_name: Optional[str] = None,
) -> Optional[Tuple[int, int, int, int, int]]:
    """
    フローのメタデータだけを読み、画像 / 間引き格子のサイズを返す。

    ※ ここでは flow_cache='off' を強制する。'auto' のままだと FlowFrameReader が
      この時点でキャッシュファイルを作成してしまい、本解析パスでその「中身が空の
      キャッシュ」を有効とみなして読み込んでしまう（全画素が 0 = 無効になる）ため。

    Returns
    -------
    tuple or None
        (rows, cols, out_rows, out_cols, n_frames_total)
    """
    try:
        reader = mt_ori.FlowFrameReader(
            Path(exp_dir), pixel_stride=max(1, int(pixel_stride)),
            frame_stride=max(1, int(frame_stride)),
            max_frames_per_exp=max_frames_per_exp,
            flow_cache='off', cache_name=flow_cache_name)
        with reader as src:
            if src.source is None:
                return None
            return (int(src.rows), int(src.cols), int(src.out_rows), int(src.out_cols),
                    int(src.n_frames_total))
    except Exception as e:
        print(f"[WARNING] could not probe flow shape in {exp_dir}: {e}", flush=True)
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Optical flow velocity field -> Ising spins sigma = sign(u . n) -> "
                     "block magnetization <|M_Ising(R)|> vs window size R."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="データルート（<root>/<beads>/<date>/<exp>/GFP_flows.h5）")
    parser.add_argument('--beads', type=str, nargs='+', default='all',
                        help="対象条件（beads06um など）。'all' で全条件。")
    parser.add_argument('--output_dir', type=str, default='figure/ising_magnetization',
                        help="出力ディレクトリ（相対パスはスクリプトディレクトリ基準）")
    parser.add_argument('--no_save_root', action='store_true',
                        help="<root_dir>/figure/ising_magnetization への保存を行わない")

    parser.add_argument('--pixel_stride', type=int, default=8,
                        help="フローの画素間引き幅（元画像 px）。窓サイズの最小単位になる。")
    parser.add_argument('--frame_stride', type=int, default=5,
                        help="フローのフレーム間引き幅")
    parser.add_argument('--max_frames_per_exp', type=int, default=None,
                        help="（デバッグ用）実験あたりの最大使用フレーム数")

    parser.add_argument('--window_sizes', type=str, default='auto',
                        help="窓サイズ [元画像 px]（例 '8:512:8' / '64,128,256'）。auto で自動生成。")
    parser.add_argument('--window_steps', type=int, default=24,
                        help="--window_sizes auto の等比分割数")
    parser.add_argument('--window_overlap', type=float, default=0.0,
                        help="ブロックの重なり率（0 = 非重複タイル, 0.5 = 50%% 重複）")

    parser.add_argument('--director', type=str, default='global', choices=['global', 'local'],
                        help="n の与え方（global = フレームごとの大域ネマチック主軸, "
                             "local = MTs_im_theta.zarr の局所配向場）")
    parser.add_argument('--theta_sign', type=str, default='auto', choices=['auto', '+1', '-1'],
                        help="MTs_im_theta.zarr のミラー規約補正（auto で自動判定）")
    parser.add_argument('--mirror_margin', type=float, default=0.05,
                        help="auto 判定で +1 / -1 を区別しない閾値（|S2+ - S2-|）")

    parser.add_argument('--min_flow_mag', type=float, default=1e-4,
                        help="この流速ノルム以下の画素はスピン未定義として無効化")
    parser.add_argument('--min_valid_fraction', type=float, default=0.5,
                        help="ブロックを有効とみなすのに必要な有効画素の割合")
    parser.add_argument('--mask_radius_factor', type=float, default=0.0,
                        help="貨物粒子近傍マスク半径 = factor x R_c（0 で無効）")
    parser.add_argument('--min_mask_radius_px', type=float, default=0.0,
                        help="貨物粒子近傍マスク半径の下限 [px]")

    parser.add_argument('--scale', type=float, default=0.11,
                        help="空間スケール [μm/px]（窓サイズの μm 換算に使用）")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="フレーム間隔 [s]（記録用）")

    parser.add_argument('--flow_cache', type=str, default='auto',
                        choices=['auto', 'off', 'refresh'],
                        help="間引きフローキャッシュの扱い")
    parser.add_argument('--flow_cache_name', type=str, default=None,
                        help="キャッシュファイル名（既定 mt_flow_cache_s{st}_f{fs}.h5）")
    parser.add_argument('--flow_cache_dir', type=str, default=None,
                        help="キャッシュを置くディレクトリ（NAS が遅いときにローカルへ）")

    parser.add_argument('--ncols', type=int, default=3, help="パネル図の列数")
    parser.add_argument('--no_progress', action='store_true', help="tqdm を無効化")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else mt_ori.find_default_root()
    if root_dir is None or not Path(root_dir).exists():
        raise FileNotFoundError("Data root directory not found. Please specify it with --root_dir.")

    out_arg = Path(args.output_dir).expanduser()
    out_dirs: List[Path] = [out_arg if out_arg.is_absolute() else (CURRENT_DIR / out_arg)]
    if not args.no_save_root:
        out_dirs.append(Path(root_dir) / 'figure' / 'ising_magnetization')
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    target_beads = mt_ori.parse_target_beads(args.beads, BEADS_INFO)
    if not target_beads:
        raise RuntimeError("No target bead conditions selected.")

    # --- 窓サイズ（auto の上限は最初の実験のフロー形状から決める） ---
    probe = None
    for bead in target_beads:
        exp_dirs = mt_ori.find_experiment_dirs(root_dir, bead['name'])
        for exp_dir in exp_dirs:
            cache_name = mt_ori.resolve_flow_cache_name(
                args.flow_cache_dir, root_dir, exp_dir,
                args.pixel_stride, args.frame_stride, args.flow_cache_name)
            probe = probe_grid_shape(exp_dir, args.pixel_stride, args.frame_stride,
                                     args.max_frames_per_exp, cache_name)
            if probe is not None:
                break
        if probe is not None:
            break
    if probe is None:
        raise RuntimeError("Could not probe any GFP_flows.h5; nothing to analyse.")

    rows, cols, out_rows, out_cols, n_frames_probe = probe
    max_window_grid = max(1, min(int(out_rows), int(out_cols)) // 2)
    windows_grid, windows_px, adjustments = resolve_windows(
        args.window_sizes, args.pixel_stride, max_window_grid, args.window_steps)
    if windows_grid.size == 0:
        raise RuntimeError("No usable window sizes resolved.")
    windows_um = windows_px.astype(float) * float(args.scale)

    print("=" * 78)
    print(" Ising Spin Magnetization <|M_Ising(R)|> vs Window Size from the MT Optical Flow")
    print("=" * 78)
    print(f"Data Root Directory : {root_dir}")
    print(f"Output Directories  : {', '.join(str(d) for d in out_dirs)}")
    print(f"Target Beads        : {[b['name'] for b in target_beads]}")
    print(f"Flow resolution     : {rows} x {cols} px (frames ~ {n_frames_probe})")
    print(f"Pixel / frame stride: {args.pixel_stride} px / {args.frame_stride} frames")
    print(f"Spin grid resolution: {out_rows} x {out_cols}")
    print(f"Director            : {args.director} (theta_sign = {args.theta_sign})")
    print(f"Window sizes [{len(windows_px)}]   : "
          f"{windows_px[0]} px ({windows_um[0]:.2f} um) ... "
          f"{windows_px[-1]} px ({windows_um[-1]:.2f} um), overlap = {args.window_overlap}")
    print(f"Min |u| / valid frac: {args.min_flow_mag} / {args.min_valid_fraction}")
    if adjustments:
        preview = ', '.join(f"{a}->{'dropped' if b is None else b}" for a, b in adjustments[:6])
        print(f"[NOTE] --window_sizes の丸め / 除外: {preview}"
              f"{' ...' if len(adjustments) > 6 else ''}")
    print("-" * 78)

    # --- 各実験の集計 ---
    results: List[dict] = []
    for bead in target_beads:
        exp_dirs = mt_ori.find_experiment_dirs(root_dir, bead['name'])
        print(f"[{bead['name']}] {len(exp_dirs)} experiment dir(s) with {FLOW_NAME}")
        for exp_dir in exp_dirs:
            cache_name = mt_ori.resolve_flow_cache_name(
                args.flow_cache_dir, root_dir, exp_dir,
                args.pixel_stride, args.frame_stride, args.flow_cache_name)
            res = process_experiment_ising(
                exp_dir, bead, windows_grid,
                pixel_stride=args.pixel_stride,
                frame_stride=args.frame_stride,
                max_frames_per_exp=args.max_frames_per_exp,
                min_flow_mag=args.min_flow_mag,
                window_overlap=args.window_overlap,
                min_valid_fraction=args.min_valid_fraction,
                mask_radius_factor=args.mask_radius_factor,
                min_mask_radius_px=args.min_mask_radius_px,
                scale=args.scale,
                director=args.director,
                flow_cache=args.flow_cache,
                flow_cache_name=cache_name,
                progress=not args.no_progress,
            )
            if res is None:
                continue
            results.append(res)
            vp = res['variants'][VARIANT_PLUS]
            print(f"      -> {exp_dir}: {res['n_frames_used']}/{res['n_frames_total']} frames, "
                  f"theta = {res['theta_source']} "
                  f"(mean {np.rad2deg(res['theta_mean_rad']):.1f} deg), "
                  f"|M(R_min)| = {vp['abs_mean'][0]:.4f}, "
                  f"|M(R_max)| = {vp['abs_mean'][-1]:.4f}, "
                  f"flow = {res['flow_cache_source']}", flush=True)

    if not results:
        print("[ERROR] No experiment could be processed.")
        return

    sign, sign_info = choose_director_sign(results, args.theta_sign, args.mirror_margin)
    print("-" * 78)
    print(f" Director sign       : {sign}  ({sign_info['decision']})")
    print(f"   pooled <cos 2 dtheta>: +theta = {sign_info['S2_plus']:.4f}, "
          f"-theta = {sign_info['S2_minus']:.4f}  ({sign_info['n_experiments']} experiments)")
    if sign < 0:
        print("   [NOTE] MTs_im_theta.zarr の角度規約が光学フローとミラー関係にあるため、"
              "theta -> -theta として採用しました。")
    print("-" * 78)

    # --- 集計テーブル ---
    df_exp = per_experiment_table(results, sign)
    df_curve = condition_curve_table(results, target_beads, sign)
    df_summary = condition_summary_table(results, target_beads, sign)

    sign_note = (f"director = {args.director}, theta_sign = {sign} "
                 f"({sign_info['decision']}), pixel_stride = {args.pixel_stride}, "
                 f"frame_stride = {args.frame_stride}, overlap = {args.window_overlap}")
    mt_ori.save_csv_to_all(df_exp, 'ising_magnetization_per_experiment', out_dirs)
    mt_ori.save_csv_to_all(df_curve, 'ising_magnetization_curve', out_dirs)
    mt_ori.save_csv_to_all(df_summary, 'ising_magnetization_summary', out_dirs)

    # --- 作図 ---
    plot_magnetization_vs_window(df_curve, target_beads, out_dirs, sign_note=sign_note)
    plot_magnetization_vs_window_linear(df_curve, target_beads, out_dirs, sign_note=sign_note)
    plot_per_experiment(df_exp, target_beads, out_dirs, sign_note=sign_note, ncols=args.ncols)
    plot_polar_bias(df_summary, out_dirs, sign_note=sign_note)

    examples: Dict[str, Tuple[Optional[np.ndarray], str]] = {}
    for bead in target_beads:
        rs = [r for r in results if r['bead_name'] == bead['name'] and r['n_frames_used'] > 0]
        if not rs:
            continue
        r0 = rs[0]
        examples[bead['name']] = (select_spin_example(r0, sign), Path(r0['exp_dir']).name)
    plot_spin_map_examples(examples, out_dirs, sign_note=sign_note, ncols=args.ncols)

    # --- ログ ---
    print("-" * 78)
    print(" Per-condition summary (experiment-level mean +/- SEM)")
    cols = ['bead_name', 'n_experiments', 'n_frames_used', 'window_min_px', 'window_max_px',
            'window_max_um', 'abs_mean_at_min_R', 'abs_mean_at_max_R',
            'pooled_abs_mean_at_max_R', 'power_law_exponent', 'power_law_r2',
            'power_law_upper_exponent', 'polar_bias_mean', 'frac_plus_mean',
            'nematic_order_cos2_mean', 'theta_source']
    print(df_summary[[c for c in cols if c in df_summary.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()

