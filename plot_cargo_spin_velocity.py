#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cargo_spin_velocity.py
===========================

光学フロー（GFP_flows.h5）から得た MT 流速場 u(x, y, t) にイジングスピン

    sigma(x, y, t) = sign( u(x, y, t) . n(x, y, t) )

を定義し、**貨物粒子（ビーズ）i の下の領域**（粒子中心を中心とする円板）における
スピン平均

    M_{i,t} = < sigma >_{region(i, t)}                ( -1 <= M <= +1 )

と、同じ円板領域における局所ポーラーオーダー

    P_{i,t} = | < u_hat >_{region(i, t)} |,   u_hat = u / |u|   ( 0 <= P <= 1 )

を計算し、貨物粒子 i のフレーム t における速度

    v_{i,t} = | r_i(t + tau) - r_i(t) | / ( tau * dt )   [um/s]
    （dt = --frame_interval [s], tau = --tau [frames]、beads_tracks.csv の位置から算出）

に対する散布図を作ります。出力は 2 系統です:

1. 横軸 = M_{i,t}          : magnetization_vs_velocity_*  （イジングスピン平均）
2. 横軸 = P_{i,t}          : polar_order_vs_velocity_*   （局所ポーラーオーダー）

【データ源（各実験ディレクトリ内）】
- 流速場     : GFP_flows.h5（shape = (frame, 2, y, x) もしくは (frame, y, x, 2)）。
               フレーム読み出しは plot_mt_orientation_distribution.FlowFrameReader を
               再利用する（1 アクセスで 2ch 読み + 間引きキャッシュで NAS I/O を最小化）。
- 軌跡・速度 : beads_tracks.csv（'particle', 'frame', 'x', 'y'）。
               速度は libs.hmm_cargo.extract_hmm_features と同一の定義
               （連続フレーム対 t -> t+tau のみを使用）で計算する。
- ディレクター: MTs_im_theta.zarr（局所配向角マップ、--director local）もしくは
               大域ネマチック主軸 theta_nem(t)（--director global, 既定）。
               zarr が読めない場合はフロー配向から同じ定義（2 テンソル平均）で推定する。

【0.11 um/px、4 s/frame、光学フローの単位】
GFP_flows.h5 の (u_x, u_y) は **px/frame の変位**（実測: 平均 |u| = 10.2 px/frame
= 0.28 um/s @ 0.11 um/px, 4 s）なので、フロー速度は u * scale / dt で um/s に換算できる。
M / P は単位ベクトルと符号だけを使うため、この換算の影響を受けない
（--velocity flow を指定したときだけ v がフロー速度になる）。

【「貨物粒子の下の領域」の定義】
ビーズ中心の円板（半径 R_region = max(--region_factor * R_c, --min_region_um)）で、
--region_inner_factor > 0 の場合は内側（ビーズ自身の占有領域）をくり抜いた円環になる
（ビーズ直下のフローは遮蔽・補間の影響を受けやすいため、感度チェック用）。
円板の内側で「流速が閾値以上（--min_flow_mag）」の有効画素だけを平均し、
有効画素数が --min_region_pixels 未満、または有効率が --min_valid_fraction 未満の
(i, t) は棄却する。既定は pixel_stride = 4（1 um の円板でも 10 点以上を確保するため）。
--pixel_stride を大きくすると円板内のサンプル数が減り M が離散化されて見かけの
相関が弱まる点に注意（実行ヘッダに円板内画素数を表示する）。

【M の符号規約（ディレクター n のミラー）】
n -> -n で sigma は反転するため、M の符号（したがって v との相関の符号）はディレクターの
向きに依存する。MTs_im_theta.zarr の角度規約は光学フローとミラー関係にあるデータが
あるため、plot_ising_magnetization.choose_director_sign と同一の判定（プール
< cos 2 Delta theta > が大きい側を採用、--theta_sign auto / --mirror_margin）を共有し、
採用符号を CSV（dir_sign）と図の注記に記録する。P は n に依存しないため影響を受けない。

【出力ファイル（既定: ./figure/cargo_spin_velocity と <root>/figure/cargo_spin_velocity）】
1. cargo_spin_velocity_points.csv       : すべての (実験, 粒子 i, フレーム t) の生データ
                                          （M, P, v, 有効画素数, 円板半径, theta ...）
2. cargo_spin_velocity_summary.csv      : 条件（粒子径）ごとの統計量。M / P と v の
                                          Pearson r, Spearman rho, OLS 傾き・切片・R^2,
                                          粒子内（時間変動のみ）の相関 within_r など
3. cargo_spin_velocity_binned.csv       : 条件 x 横軸変数ごとの等点数ビン統計（中央値・
                                          四分位・SEM）＝図のトレンド線の数値
4. cargo_spin_velocity_extraction.csv   : 実験ごとのフレーム数・採用点数・棄却数・
                                          theta_source・円板半径・フローキャッシュ元
5. magnetization_vs_velocity_<bead>.png/.svg    : 条件別 M_{i,t} vs v_{i,t}
6. polar_order_vs_velocity_<bead>.png/.svg       : 条件別 P_{i,t} vs v_{i,t}
7. magnetization_vs_velocity_all_beads.png/.svg  : 条件別パネル（M）
8. polar_order_vs_velocity_all_beads.png/.svg    : 条件別パネル（P）
9. magnetization_vs_velocity_overlay.png/.svg    : 全条件を 1 軸に重ね描き（M）
10. polar_order_vs_velocity_overlay.png/.svg     : 全条件を 1 軸に重ね描き（P）
11. magnetization_vs_velocity_heatmap.png/.svg  : 全条件プールの 2D ヒストグラム
                                                  （横軸 = M_{i,t}, 縦軸 = v_{i,t}, 色 = P(v, M)）
12. polar_order_vs_velocity_heatmap.png/.svg     : 同（横軸 = P_{i,t}）
    （--heatmap_per_condition で条件別の ..._heatmap_<bead>.png/.svg も出力）
13. *_heatmap_abs.png/.svg                        : --heatmap_abs を指定したときの絶対値版
                                                  （横軸 = |M_{i,t}| / P_{i,t}, 縦軸 = |v_{i,t}|）
14. cargo_spin_velocity_heatmap.csv              : 2D ヒストグラムの各ビン（境界・個数・
                                                  同時確率密度・abs_values フラグ）。
                                                  count = 0 のビンは省略

【2D ヒートマップ（色 = 同時確率密度 P(v, M)）】
散布図は 1 点 = 1 (i, t) をそのまま描くため、点数が多いと密度の偏りが見えにくい。
--no_heatmap を指定しない限り、全条件・全粒子をプールした 2D ヒストグラムも出力する。

    P(v, M) = count(v, M) / ( N_in * dM * dv )      [ (um/s)^-1 ]

N_in はビン範囲内の点数（横軸は M / P の物理範囲、縦軸は速度の分位点まで）なので、
図の範囲内で ∫ P dv dM = 1 になる。色は既定で線形（--heatmap_log_color で対数）、
上限は 0 でないビンの --heatmap_vmax_percentile 分位点（既定 99）に取って
1 ビンだけが突出して他が白飛びするのを防ぐ。縦軸は速度分布が裾を引くため、既定では
**対数等間隔ビン**（--heatmap_y_edges log）にして低速度側の分解能を確保する。
白線は散布図と同じ等点数ビンの中央値 ± IQR（色 = 個数の右肩上がりとは独立な、
v の条件付き分布の代表値）。周辺分布（上 = x の個数、右 = v の個数）も併置する。

【絶対値版（--heatmap_abs）】
M はディレクターの符号規約（dir_sign）に依存し、v も変位ベクトルの向きに依存するため、
符号を落とした「強さ」だけで見たい場合は --heatmap_abs を付ける。横軸 = |M|（P は元から
0 <= P <= 1 なのでそのまま）、縦軸 = |v| に折り畳んで集計し（v = 0 の点は対数ビンに
入らないため範囲外として図中に個数を表示）、統計量（r, rho, 傾き）も |v| vs |M| で
計算し直す。散布図は従来どおり符号付きのまま。

【実行例】
    pixi run python plot_cargo_spin_velocity.py --beads all \\
        --pixel_stride 4 --frame_stride 5 --flow_cache_dir /tmp/mtcache
    pixi run python plot_cargo_spin_velocity.py --beads 1um 3um \\
        --region_factor 3 --velocity flow --yscale log
    pixi run python plot_cargo_spin_velocity.py --beads 1um \\
        --heatmap_per_condition --heatmap_log_color            # ヒートマップを条件別 + 対数色
    pixi run python plot_cargo_spin_velocity.py --heatmap_abs   # |v| vs |M| のヒートマップも
    pixi run python plot_cargo_spin_velocity.py --no_heatmap   # 散布図だけを出力
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# NAS / 共有ボリュームでの HDF5 ファイルロックエラー防止（h5py import 前に設定が必要）
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats as scipy_stats
from tqdm import tqdm

# 親ディレクトリのパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import hmm_cargo as hc
from libs import ising_magnetization as ising
import plot_mt_orientation_distribution as mt_ori
import plot_ising_magnetization as ising_plot

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

VARIANT_PLUS = ising_plot.VARIANT_PLUS
VARIANT_MINUS = ising_plot.VARIANT_MINUS
VARIANT_NAMES = ising_plot.VARIANT_NAMES

# 横軸に取る秩序変数（図・CSV の系統）
X_VARIABLES = ('m_ising', 'polar')
X_LABELS = {
    'm_ising': r'Ising spin average $M_{i,t}$ (region under cargo)',
    'polar': r'local polar order $P_{i,t}$ (region under cargo)',
}
X_LIMITS = {
    'm_ising': (-1.05, 1.05),
    'polar': (-0.02, 1.05),
}
X_FILE_TAG = {
    'm_ising': 'magnetization_vs_velocity',
    'polar': 'polar_order_vs_velocity',
}

# 秩序変数の物理的な範囲（2D ヒートマップの横軸ビンに使う）
X_PHYS_LIMITS = {
    'm_ising': (-1.0, 1.0),
    'polar': (0.0, 1.0),
}

# --heatmap_abs: 符号をもつ量（v と M）を絶対値に折り畳んだヒートマップ用の定義。
# P は 0 <= P <= 1 で元から非負なので折り畳んでも変わらない。
X_PHYS_LIMITS_ABS = {
    'm_ising': (0.0, 1.0),
    'polar': (0.0, 1.0),
}
X_LABELS_ABS = {
    'm_ising': r'Ising spin average $|M_{i,t}|$ (region under cargo)',
    'polar': X_LABELS['polar'],
}

# 2D ヒートマップの色の量（同時確率密度）のラベルと単位
JOINT_LABELS = {
    'm_ising': r'$P(v_{i,t},\, M_{i,t})$',
    'polar': r'$P(v_{i,t},\, P_{i,t})$',
}
JOINT_LABELS_ABS = {
    'm_ising': r'$P(|v_{i,t}|,\, |M_{i,t}|)$',
    'polar': r'$P(|v_{i,t}|,\, P_{i,t})$',
}
JOINT_DENSITY_UNIT = r'[($\mu$m/s)$^{-1}$] '

VELOCITY_LABEL = r'cargo velocity $v_{i,t}$ [$\mu$m/s]'
VELOCITY_LABEL_ABS = r'cargo speed $|v_{i,t}|$ [$\mu$m/s]'



# =============================================================================
# 領域（貨物粒子の下の円板）と速度のユーティリティ
# =============================================================================

def region_radius_um(bead: dict, region_factor: float = 2.0,
                     min_region_um: float = 1.0) -> float:
    """
    「貨物粒子の下の領域」の半径 [um] を返す。

        R_region = max( region_factor * R_c , min_region_um )

    既定（factor = 2.0, min = 1.0 um）は「ビーズ直径と同程度の円板」で、かつ最小径の
    ビーズ (0.63 um) でも間引き格子上で十分な画素数（~16 点 @ pixel_stride 4）を確保する。
    """
    r_bead = float(bead.get('radius_um', 0.0))
    return float(max(float(region_factor) * r_bead, float(min_region_um)))


def cargo_positions_with_ids(
    df_tracks: Optional[pd.DataFrame],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """beads_tracks.csv から frame -> (particle, x, y) の辞書を作る。"""
    positions: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if df_tracks is None or df_tracks.empty:
        return positions
    if not {'frame', 'particle', 'x', 'y'}.issubset(set(df_tracks.columns)):
        return positions
    for frame, grp in df_tracks.groupby('frame'):
        positions[int(frame)] = (
            grp['particle'].to_numpy(dtype=np.int64),
            grp['x'].to_numpy(dtype=float),
            grp['y'].to_numpy(dtype=float),
        )
    return positions


def tracked_velocity_lookup(
    df_tracks: Optional[pd.DataFrame],
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
) -> Dict[Tuple[int, int], float]:
    """
    beads_tracks.csv から (particle, frame) -> v_{i,t} [um/s] の辞書を作る。

    定義は libs.hmm_cargo.extract_hmm_features と同一:

        v_{i,t} = | r_i(t + tau) - r_i(t) | / (tau * frame_interval)

    連続フレーム対 (t, t + tau) が存在する場合のみ値を持つ（欠損フレームは含まれない）。
    """
    if df_tracks is None or df_tracks.empty:
        return {}
    try:
        _, _, df_obs = hc.extract_hmm_features(
            df_tracks, tau=int(tau), scale=float(scale),
            frame_interval=float(frame_interval), epsilon=1e-3)
    except Exception as e:
        print(f"    [WARNING] could not compute velocities: {e}", flush=True)
        return {}
    if df_obs is None or df_obs.empty:
        return {}
    out: Dict[Tuple[int, int], float] = {}
    for p, f, v in zip(df_obs['particle'].to_numpy(dtype=np.int64),
                       df_obs['frame'].to_numpy(dtype=np.int64),
                       df_obs['v'].to_numpy(dtype=float)):
        out[(int(p), int(f))] = float(v)
    return out



def disk_region_sums(
    mx: np.ndarray,
    my: np.ndarray,
    sigma_plus: np.ndarray,
    sigma_minus: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    valid: np.ndarray,
    cx: float,
    cy: float,
    r_out: float,
    r_in: float = 0.0,
) -> Optional[dict]:
    """
    間引き格子（1 unit = pixel_stride px）上で、中心 (cx, cy)・外半径 r_out・内半径 r_in の
    円板領域（r_in = 0 なら円板、> 0 なら円環）について和をまとめて計算する。

    Parameters
    ----------
    mx, my : ndarray
        光学フロー（px/frame）。間引き格子。
    sigma_plus, sigma_minus : ndarray
        イジングスピン sign(u . n)（+theta / -theta の 2 バリアント）。無効画素は 0。
    ux, uy : ndarray
        単位ベクトル場 u / |u|（無効画素は 0）。
    valid : ndarray of bool
        有効画素マスク。
    cx, cy, r_out, r_in : float
        **間引き格子の単位**（フル解像度 px / pixel_stride）での中心と半径。

    領域が画像外 / 有効画素が 0 の場合は None を返す。
    """
    rows, cols = mx.shape
    y0 = max(int(np.floor(cy - r_out)), 0)
    y1 = min(int(np.ceil(cy + r_out)) + 1, int(rows))
    x0 = max(int(np.floor(cx - r_out)), 0)
    x1 = min(int(np.ceil(cx + r_out)) + 1, int(cols))
    if y1 <= y0 or x1 <= x0:
        return None

    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    disk = d2 <= float(r_out) ** 2
    if float(r_in) > 0.0:
        disk &= d2 > float(r_in) ** 2
    n_disk = int(np.count_nonzero(disk))
    if n_disk == 0:
        return None

    sl = (slice(y0, y1), slice(x0, x1))
    m = disk & valid[sl]
    n_valid = int(np.count_nonzero(m))
    if n_valid == 0:
        return None

    sub_mx = mx[sl][m]
    sub_my = my[sl][m]
    return {
        'n_disk': n_disk,
        'n_valid': n_valid,
        'sum_mx': float(np.sum(sub_mx)),
        'sum_my': float(np.sum(sub_my)),
        'sum_mag': float(np.sum(np.hypot(sub_mx, sub_my))),
        'sum_ux': float(np.sum(ux[sl][m])),
        'sum_uy': float(np.sum(uy[sl][m])),
        'sum_sigma_plus': float(np.sum(sigma_plus[sl][m])),
        'sum_sigma_minus': float(np.sum(sigma_minus[sl][m])),
    }


# =============================================================================
# 実験ディレクトリごとの処理
# =============================================================================

def process_experiment_cargo(
    exp_dir: Path,
    bead: dict,
    pixel_stride: int = 4,
    frame_stride: int = 5,
    max_frames_per_exp: Optional[int] = None,
    min_flow_mag: float = 1e-4,
    min_valid_fraction: float = 0.5,
    min_region_pixels: int = 6,
    region_factor: float = 2.0,
    min_region_um: float = 1.0,
    region_inner_factor: float = 0.0,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    tau: int = 1,
    director: str = 'global',
    velocity: str = 'tracked',
    flow_cache: str = 'auto',
    flow_cache_name: Optional[str] = None,
    progress: bool = True,
) -> Optional[dict]:
    """
    1 つの実験ディレクトリについて、各フレーム t・各貨物粒子 i の

        M_{i,t} = < sign(u . n) >_{region}   ( -1 <= M <= +1 )
        P_{i,t} = | < u / |u| >_{region} |   (  0 <= P <= 1 )

    を計算し、(粒子, フレーム) ごとのレコードと集計メタ情報を返す。

    M はディレクターの符号（n -> -n）で反転するため、+theta / -theta の両バリアントを
    同時に計算してレコードに保持し（m_ising_plus / m_ising_minus）、プールした
    < cos 2 Delta theta > で後段（choose_director_sign）が採用符号を決める。
    P はディレクター非依存なので 1 回だけ計算して共有する。

    引数 velocity は散布図の縦軸に使う速度の定義（'tracked' = 軌跡から
    v = |dr| / (tau * dt)、'flow' = 円板内の平均フロー |<u>| * scale / dt、
    すなわちビーズ直下の MT 流速）。どちらも CSV には常に記録する
    （v_track_um_s / v_flow_um_s / v_flow_absmean_um_s）。

    Returns
    -------
    dict or None
        records（list of dict）/ n_frames_used / n_frames_total / theta_source /
        variants（符号判定用の cos2_sum, n_valid_sum）/ 円板半径 / 棄却カウンタ など。
        フローも有効なキャッシュも無い場合は None。
    """
    exp_dir = Path(exp_dir)
    st = max(1, int(pixel_stride))
    fs = max(1, int(frame_stride))

    r_um = region_radius_um(bead, region_factor, min_region_um)
    r_px = r_um / float(scale)
    r_grid = r_px / float(st)
    r_in_um = float(max(0.0, float(region_inner_factor))) * float(bead.get('radius_um', 0.0))
    r_in_px = r_in_um / float(scale)
    r_in_grid = r_in_px / float(st)

    # --- 軌跡（位置と速度） ---
    tracks_path = exp_dir / TRACKS_NAME
    df_tracks: Optional[pd.DataFrame] = None
    if tracks_path.exists():
        try:
            df_tracks = pd.read_csv(tracks_path)
        except Exception as e:
            print(f"    [WARNING] could not read {tracks_path}: {e}", flush=True)
    else:
        print(f"    [WARNING] {TRACKS_NAME} not found in {exp_dir.name}; skipped", flush=True)
        return None

    positions = cargo_positions_with_ids(df_tracks)
    v_track = tracked_velocity_lookup(df_tracks, tau=tau, scale=scale,
                                      frame_interval=frame_interval)
    if not positions:
        print(f"    [WARNING] no usable cargo positions in {tracks_path}", flush=True)
        return None

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
        frame_ids = [int(t) for t in src.frame_ids]
        full_rows, full_cols = int(src.rows), int(src.cols)
        out_shape = (int(src.out_rows), int(src.out_cols))

        # --- ディレクター（n）の決定 ---
        thetas: Optional[np.ndarray] = None
        theta_maps: Optional[np.ndarray] = None
        if str(director) == 'local':
            theta_maps = ising_plot.load_local_director_maps(
                exp_dir, frame_ids, (full_rows, full_cols), out_shape, st)
            if theta_maps is not None:
                theta_source = 'local:zarr'
            else:
                thetas = mt_ori.load_nematic_directors(exp_dir, frame_ids, n_frames_total)
                theta_source = ('local:zarr-fallback' if thetas is not None
                                else 'local:flow-fallback')
        else:
            thetas = mt_ori.load_nematic_directors(exp_dir, frame_ids, n_frames_total)
            theta_source = 'global:zarr' if thetas is not None else 'global:flow'

        records: List[dict] = []
        cos2_sum = {name: 0.0 for name in VARIANT_NAMES}
        n_valid_sum = {name: 0.0 for name in VARIANT_NAMES}
        n_frames_used = 0
        n_rejected_region = 0
        n_rejected_no_velocity = 0
        n_no_positions = 0

        iterator = range(len(frame_ids))
        if progress:
            iterator = tqdm(iterator, desc=f"CargoIsing {exp_dir.parent.name}/{exp_dir.name}",
                            leave=False)

        for i in iterator:
            t = int(frame_ids[i])
            mx, my = src.get(i)
            mag = np.hypot(mx, my)
            valid = np.isfinite(mx) & np.isfinite(my) & (mag > max(float(min_flow_mag), 0.0))

            # --- ディレクター角 theta の決定 ---
            theta_frame = float('nan')
            if theta_maps is not None:
                th = theta_maps[i]
                valid = valid & np.isfinite(th)
            elif thetas is not None:
                th = float(thetas[min(t, thetas.size - 1)])
                theta_frame = th
            else:
                phi_v = np.arctan2(my[valid], mx[valid])
                th = mt_ori.global_nematic_theta_from_angles(phi_v)
                theta_frame = th

            if not np.any(valid):
                continue
            n_frames_used += 1
            n_valid_frame = float(np.count_nonzero(valid))

            ct = float(np.cos(th))
            stt = float(np.sin(th))
            c2t = float(np.cos(2.0 * th))
            s2t = float(np.sin(2.0 * th))

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

            # 単位ベクトル場 u_hat（ポーラーオーダー用・ディレクター非依存）
            ux = np.cos(phi)
            uy = np.sin(phi)
            ux[~valid] = 0.0
            uy[~valid] = 0.0

            sigmas: Dict[str, np.ndarray] = {}
            for name in VARIANT_NAMES:
                n_valid_sum[name] += n_valid_frame
                cos2_sum[name] += float(np.sum(cos2_terms[name][valid]))
                sigma = np.sign(dots[name])
                sigma[~valid] = 0.0
                sigmas[name] = sigma

            pos = positions.get(t)
            if pos is None:
                n_no_positions += 1
                continue
            parts, xs, ys = pos

            for pid, x, y in zip(parts, xs, ys):
                res = disk_region_sums(
                    mx, my, sigmas[VARIANT_PLUS], sigmas[VARIANT_MINUS], ux, uy, valid,
                    cx=float(x) / float(st), cy=float(y) / float(st),
                    r_out=r_grid, r_in=r_in_grid)
                if res is None:
                    n_rejected_region += 1
                    continue
                n_valid_region = int(res['n_valid'])
                valid_frac_region = n_valid_region / float(max(1, int(res['n_disk'])))
                if (n_valid_region < int(min_region_pixels)
                        or valid_frac_region < float(min_valid_fraction)):
                    n_rejected_region += 1
                    continue

                v_t = float(v_track.get((int(pid), t), float('nan')))
                v_flow = float(np.hypot(res['sum_mx'], res['sum_my']) / n_valid_region
                               * float(scale) / float(frame_interval))
                v_flow_absmean = float(res['sum_mag'] / n_valid_region
                                       * float(scale) / float(frame_interval))
                v_selected = v_t if str(velocity) == 'tracked' else v_flow
                if not np.isfinite(v_selected):
                    n_rejected_no_velocity += 1
                    continue

                records.append({
                    'bead_name': bead['name'],
                    'exp_dir': exp_dir.name,
                    'particle': int(pid),
                    'frame': int(t),
                    'time_s': float(t) * float(frame_interval),
                    'x_um': float(x) * float(scale),
                    'y_um': float(y) * float(scale),
                    'm_ising_plus': float(res['sum_sigma_plus'] / n_valid_region),
                    'm_ising_minus': float(res['sum_sigma_minus'] / n_valid_region),
                    'polar': float(np.hypot(res['sum_ux'], res['sum_uy']) / n_valid_region),
                    'v_track_um_s': v_t,
                    'v_flow_um_s': v_flow,
                    'v_flow_absmean_um_s': v_flow_absmean,
                    'region_n_valid': n_valid_region,
                    'region_n_disk': int(res['n_disk']),
                    'region_valid_fraction': float(valid_frac_region),
                    'theta_rad': float(theta_frame),
                })

        cache_source = src.source
        cache_path = str(src.cache_path) if src.source == 'cache' else ''

    return {
        'exp_dir': str(exp_dir),
        'bead_name': bead['name'],
        'records': records,
        'n_frames_used': int(n_frames_used),
        'n_frames_total': int(n_frames_total),
        'dataset_shape': shape,
        'grid_shape': out_shape,
        'pixel_stride': st,
        'frame_stride': fs,
        'director': str(director),
        'theta_source': theta_source,
        'variants': {name: {'cos2_sum': float(cos2_sum[name]),
                            'n_valid_sum': float(n_valid_sum[name])}
                     for name in VARIANT_NAMES},
        'region_radius_um': float(r_um),
        'region_radius_px': float(r_px),
        'region_inner_um': float(r_in_um),
        'n_rejected_region': int(n_rejected_region),
        'n_rejected_no_velocity': int(n_rejected_no_velocity),
        'n_frames_without_positions': int(n_no_positions),
        'flow_cache_source': cache_source,
        'flow_cache_path': cache_path,
        'tau': int(tau),
        'velocity': str(velocity),
    }


# =============================================================================
# テーブル化と統計量
# =============================================================================

def points_table(
    results: Sequence[dict],
    sign: int = 1,
    velocity: str = 'tracked',
) -> pd.DataFrame:
    """
    全実験のレコードを 1 つの DataFrame にまとめる。

    採用したディレクター符号 sign（+1 / -1）に対応する磁化を m_ising 列に入れ、
    縦軸に使う速度を v_um_s 列に入れる（v_track_um_s / v_flow_um_s は常に保持）。
    """
    sgn = 1 if int(sign) >= 0 else -1
    vkey = 'v_track_um_s' if str(velocity) == 'tracked' else 'v_flow_um_s'
    rows: List[dict] = []
    for res in results:
        bead = BEAD_LOOKUP.get(res['bead_name'], {})
        for rec in res['records']:
            m = rec['m_ising_plus'] if sgn > 0 else rec['m_ising_minus']
            rows.append({
                'bead_name': rec['bead_name'],
                'diameter_um': float(bead.get('diameter_um', np.nan)),
                'radius_um': float(bead.get('radius_um', np.nan)),
                'exp_dir': rec['exp_dir'],
                'particle': int(rec['particle']),
                'frame': int(rec['frame']),
                'time_s': rec['time_s'],
                'x_um': rec['x_um'],
                'y_um': rec['y_um'],
                'm_ising': m,
                'polar': rec['polar'],
                'v_um_s': rec[vkey],
                'v_track_um_s': rec['v_track_um_s'],
                'v_flow_um_s': rec['v_flow_um_s'],
                'v_flow_absmean_um_s': rec['v_flow_absmean_um_s'],
                'region_n_valid': int(rec['region_n_valid']),
                'region_n_disk': int(rec['region_n_disk']),
                'region_valid_fraction': rec['region_valid_fraction'],
                'region_radius_um': float(res['region_radius_um']),
                'region_inner_um': float(res['region_inner_um']),
                'theta_rad': rec['theta_rad'],
                'dir_sign': sgn,
                'velocity_source': str(velocity),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(['bead_name', 'exp_dir', 'particle', 'frame']).reset_index(drop=True)


def bin_profile_records(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 12,
    min_count: int = 10,
) -> List[dict]:
    """
    x を**等点数（分位点）ビン**に分割し、各ビンでの y の中央値・平均・四分位・SEM を返す。

    速度 v も M / P も裾を引く分布になるため、等幅ビンではなく等点数ビンを使う
    （libs/ising_magnetization.binned_median の等幅版とは用途が異なる）。
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0 or int(n_bins) < 1:
        return []

    edges = np.unique(np.quantile(xs, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 3:
        return []

    out: List[dict] = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (xs >= lo) & (xs <= hi) if i == edges.size - 2 else (xs >= lo) & (xs < hi)
        n_pts = int(np.count_nonzero(sel))
        if n_pts < max(1, int(min_count)):
            continue
        vals = ys[sel]
        out.append({
            'x_low': lo,
            'x_high': hi,
            'x_center': float(np.median(xs[sel])),
            'n_points': n_pts,
            'y_median': float(np.median(vals)),
            'y_mean': float(np.mean(vals)),
            'y_q25': float(np.quantile(vals, 0.25)),
            'y_q75': float(np.quantile(vals, 0.75)),
            'y_sem': (float(np.std(vals, ddof=1) / np.sqrt(n_pts)) if n_pts > 1 else 0.0),
        })
    return out


def binned_table(
    df_points: pd.DataFrame,
    n_bins: int = 12,
    min_count: int = 10,
) -> pd.DataFrame:
    """
    条件 x 横軸変数（M / P）ごとの等点数ビン統計（図のトレンド線の数値）。

    ビン数は指定値を上限として、点数が少ない場合に自動的に減らす
    （n_bins_eff = max(1, min(n_bins, N / min_count))）。実際に使ったビン数を
    n_bins_effective 列に記録する。
    """
    if df_points is None or df_points.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    for (bname, x_var), grp in df_points.groupby(['bead_name', 'x_var'], sort=True):
        x = grp['x_value'].to_numpy(dtype=float)
        v = grp['v_um_s'].to_numpy(dtype=float)
        n_pts = int(np.count_nonzero(np.isfinite(x) & np.isfinite(v)))
        n_eff = max(1, min(int(n_bins), n_pts // max(1, int(min_count))))
        recs = bin_profile_records(x, v, n_bins=n_eff, min_count=min_count)
        for rec in recs:
            rows.append({'bead_name': bname, 'x_variable': x_var,
                         'diameter_um': float(grp['diameter_um'].iloc[0]),
                         'n_bins_effective': int(n_eff), **rec})
    return pd.DataFrame(rows)


def long_points_table(df_points: pd.DataFrame) -> pd.DataFrame:
    """
    points テーブル（1 行 = 1 (i, t)）を「横軸変数」で縦持ちにしたものを作る。

    x_var  : 'm_ising' / 'polar'
    x_value: 対応する秩序変数の値
    """
    if df_points is None or df_points.empty:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for x_var in X_VARIABLES:
        if x_var not in df_points.columns:
            continue
        sub = df_points.copy()
        sub['x_var'] = x_var
        sub['x_value'] = sub[x_var].to_numpy(dtype=float)
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def within_group_correlation(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
) -> Tuple[float, int]:
    """
    群（実験 x 粒子）内で平均を引いた後の Pearson 相関（群内変動のみの相関）を返す。

    条件全体の相関が「粒子ごとの平均レベルの違い」で生じているのか、
    「同じ粒子の時間変動」で生じているのかを切り分けるために使う。
    """
    df = pd.DataFrame({
        'x': np.asarray(x, dtype=float),
        'y': np.asarray(y, dtype=float),
        'g': list(groups),
    })
    df = df[np.isfinite(df['x']) & np.isfinite(df['y'])]
    if df.empty:
        return float('nan'), 0
    gb = df.groupby('g')
    xr = df['x'] - gb['x'].transform('mean')
    yr = df['y'] - gb['y'].transform('mean')
    return float(ising.pearson_correlation(xr.to_numpy(), yr.to_numpy())), int(len(df))


def _corr_stats(x: np.ndarray, y: np.ndarray, groups: Sequence) -> dict:
    """
    (x, y) の相関統計（Pearson / Spearman と p 値、OLS、群内相関）をまとめて返す。

    分散 0 やサンプル不足で定義できない量は NaN（libs.ising_magnetization と同じ規約）。
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    out = {
        'pearson_r': float('nan'), 'pearson_p': float('nan'),
        'spearman_rho': float('nan'), 'spearman_p': float('nan'),
        'slope_ols': float('nan'), 'slope_stderr': float('nan'),
        'intercept_ols': float('nan'), 'r2_ols': float('nan'),
        'within_r': float('nan'), 'within_n': 0,
    }
    if xs.size >= 3 and float(np.std(xs)) > 0.0 and float(np.std(ys)) > 0.0:
        r, p = scipy_stats.pearsonr(xs, ys)
        rho, p_s = scipy_stats.spearmanr(xs, ys)
        lr = scipy_stats.linregress(xs, ys)
        out.update({
            'pearson_r': float(r), 'pearson_p': float(p),
            'spearman_rho': float(rho), 'spearman_p': float(p_s),
            'slope_ols': float(lr.slope), 'slope_stderr': float(lr.stderr),
            'intercept_ols': float(lr.intercept), 'r2_ols': float(lr.rvalue ** 2),
        })
    g_all = np.asarray(list(groups), dtype=object)
    wr, wn = within_group_correlation(xs, ys, g_all[keep])
    out['within_r'] = wr
    out['within_n'] = wn
    return out


def summary_table(
    df_long: pd.DataFrame,
    velocity: str = 'tracked',
    n_bins: int = 12,
    min_count: int = 10,
) -> pd.DataFrame:
    """
    条件（粒子径）x 横軸変数（M / P）ごとの統計量テーブル。

    相関係数は 1 サンプル = 1 (粒子 i, フレーム t)。within_r は実験 x 粒子ごとに
    平均を引いた後の相関（= 同一粒子の時間変動だけを見た相関）。
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    for (bname, x_var), grp in df_long.groupby(['bead_name', 'x_var'], sort=True):
        x = grp['x_value'].to_numpy(dtype=float)
        v = grp['v_um_s'].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(v)
        groups = np.asarray(
            [f"{e}|{int(pp)}" for e, pp in
             zip(grp['exp_dir'].to_numpy(), grp['particle'].to_numpy())], dtype=object)
        n_keep = int(np.count_nonzero(keep))
        rec = {
            'bead_name': bname,
            'x_variable': x_var,
            'diameter_um': float(grp['diameter_um'].iloc[0]),
            'radius_um': float(grp['radius_um'].iloc[0]),
            'region_radius_um': float(grp['region_radius_um'].iloc[0]),
            'n_points': n_keep,
            'n_experiments': int(grp['exp_dir'].nunique()),
            'n_particles': int(grp.groupby(['exp_dir', 'particle']).ngroups),
            'x_mean': (float(np.mean(x[keep])) if n_keep else float('nan')),
            'x_median': (float(np.median(x[keep])) if n_keep else float('nan')),
            'x_std': (float(np.std(x[keep], ddof=1)) if n_keep > 1 else float('nan')),
            'v_median_um_s': (float(np.median(v[keep])) if n_keep else float('nan')),
            'v_mean_um_s': (float(np.mean(v[keep])) if n_keep else float('nan')),
            'v_p90_um_s': (float(np.quantile(v[keep], 0.9)) if n_keep else float('nan')),
            'velocity_source': str(velocity),
            'n_bins': int(n_bins),
            'bin_min_count': int(min_count),
        }
        rec.update(_corr_stats(x, v, groups))
        rows.append(rec)
    return pd.DataFrame(rows)


def extraction_table(results: Sequence[dict]) -> pd.DataFrame:
    """実験ごとの処理状況（使用フレーム数・採用点数・棄却数・円板半径・キャッシュ元）。"""
    rows: List[dict] = []
    for res in results:
        recs = res['records']
        n_valid = [r['region_n_valid'] for r in recs]
        rows.append({
            'bead_name': res['bead_name'],
            'exp_dir': Path(res['exp_dir']).name,
            'n_frames_used': int(res['n_frames_used']),
            'n_frames_total': int(res['n_frames_total']),
            'n_points': int(len(recs)),
            'n_particles': int(len({(r['exp_dir'], r['particle']) for r in recs})),
            'region_radius_um': float(res['region_radius_um']),
            'region_radius_px': float(res['region_radius_px']),
            'region_inner_um': float(res['region_inner_um']),
            'region_n_valid_median': (float(np.median(n_valid)) if n_valid else float('nan')),
            'region_n_valid_min': (int(np.min(n_valid)) if n_valid else 0),
            'n_rejected_region': int(res['n_rejected_region']),
            'n_rejected_no_velocity': int(res['n_rejected_no_velocity']),
            'n_frames_without_positions': int(res['n_frames_without_positions']),
            'theta_source': res['theta_source'],
            'flow_cache_source': res['flow_cache_source'],
            'flow_cache_path': res['flow_cache_path'],
            'tau': int(res['tau']),
            'velocity_source': res['velocity'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 作図
# =============================================================================

def _bead_label(bead: dict) -> str:
    """図のタイトル用の条件ラベル（例: 'beads1um (1.18 um)'）。"""
    return f"{bead.get('name', '?')} ({bead.get('diameter_um', float('nan')):.2f} $\\mu$m)"


def _velocity_limits(v: np.ndarray, yscale: str = 'linear',
                     upper_percentile: float = 99.5) -> Tuple[float, float]:
    """速度軸の範囲をデータから決める（対数軸では正の分位点を基準にする）。"""
    vals = np.asarray(v, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    upper = float(np.quantile(vals, upper_percentile / 100.0)) * 1.05
    if str(yscale) == 'log':
        pos = vals[vals > 0.0]
        if pos.size:
            low_q = max(0.0, (100.0 - upper_percentile) / 100.0)
            lower = max(float(np.quantile(pos, low_q)) * 0.5, float(np.min(pos)) * 0.8)
        else:
            lower = 1e-2
        if not np.isfinite(lower) or lower <= 0.0:
            lower = 1e-2
        if upper <= lower:
            upper = lower * 10.0
        return lower, upper
    if upper <= 0.0:
        upper = float(np.max(vals)) * 1.05
    return 0.0, upper


def _condition_arrays(
    df_long: pd.DataFrame,
    bead_name: str,
    x_var: str,
    max_points: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1 条件・1 横軸変数の (x, v, particle, group) を返す。

    有限値のみを残し、max_points > 0 なら空間的な順序を保ったまま等間隔に間引く
    （乱数を使わないので再現性がある）。group は (実験ディレクトリ名, 粒子 id)。
    """
    d = df_long[(df_long['bead_name'] == bead_name) & (df_long['x_var'] == x_var)]
    if d.empty:
        empty = np.empty(0)
        return empty, empty, empty.astype(np.int64), np.empty(0, dtype=object)
    x = d['x_value'].to_numpy(dtype=float)
    v = d['v_um_s'].to_numpy(dtype=float)
    p = d['particle'].to_numpy(dtype=np.int64)
    g = np.asarray([f"{e}|{int(pp)}" for e, pp in zip(d['exp_dir'].to_numpy(), p)],
                   dtype=object)
    keep = np.isfinite(x) & np.isfinite(v)
    x, v, p, g = x[keep], v[keep], p[keep], g[keep]
    if int(max_points) > 0 and x.size > int(max_points):
        idx = ising.even_stride_indices(x.size, int(max_points))
        x, v, p, g = x[idx], v[idx], p[idx], g[idx]
    return x, v, p, g


def _binned_curve(
    df_binned: Optional[pd.DataFrame],
    bead_name: str,
    x_var: str,
) -> Optional[pd.DataFrame]:
    """binned テーブルから 1 条件・1 変数のビンを取り出す（x 昇順）。"""
    if df_binned is None or df_binned.empty:
        return None
    d = df_binned[(df_binned['bead_name'] == bead_name)
                  & (df_binned['x_variable'] == x_var)]
    if d.empty:
        return None
    return d.sort_values('x_center')


def _stats_text(
    x: np.ndarray,
    v: np.ndarray,
    groups: Sequence,
    x_var: str,
    abs_values: bool = False,
) -> str:
    """図中に載せる統計量のテキスト（N, r, rho, OLS 傾き, 粒子内相関）。"""
    if str(x_var) == 'm_ising':
        head = ('Ising spin $|M_{i,t}|$' if abs_values else 'Ising spin $M_{i,t}$')
    else:
        head = 'polar order $P_{i,t}$'
    st = _corr_stats(x, v, groups)
    tail = ("\ny = $|v_{i,t}|$ (speed), folded about 0" if abs_values else "")
    return (f"{head}\n"
            f"$N$ = {st['within_n']:,} (i, t) points\n"
            f"Pearson $r$ = {st['pearson_r']:+.3f} ($p$ = {st['pearson_p']:.1e})\n"
            f"Spearman $\\rho$ = {st['spearman_rho']:+.3f}\n"
            f"OLS slope = {st['slope_ols']:+.4f} $\\pm$ {st['slope_stderr']:.4f} $\\mu$m/s per unit\n"
            f"within-particle $r$ = {st['within_r']:+.3f}{tail}")


# =============================================================================
# 2D ヒストグラム（同時分布 P(v, M) / P(v, P)）
# =============================================================================

def pooled_arrays(
    df_long: pd.DataFrame,
    target_beads: Sequence[dict],
    x_var: str,
    max_points: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    複数条件（粒子径）の (x, v, particle, group) を 1 つにプールする。

    各条件は _condition_arrays（有限値のみ / max_points > 0 なら等間隔間引き）で
    取り出す。ヒートマップは個数を数えるため、既定（max_points = 0）では間引かずに
    全点を使う。
    """
    xs: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    gs: List[np.ndarray] = []
    for bead in (target_beads or []):
        x, v, p, g = _condition_arrays(df_long, bead['name'], x_var, max_points=max_points)
        if x.size == 0:
            continue
        xs.append(x)
        vs.append(v)
        ps.append(p)
        gs.append(g)
    if not xs:
        empty = np.empty(0)
        return empty, empty, empty.astype(np.int64), np.empty(0, dtype=object)
    return (np.concatenate(xs), np.concatenate(vs),
            np.concatenate(ps), np.concatenate(gs))


def _monotone_edges(edges: Sequence[float], n_bins: int,
                    fallback: Tuple[float, float]) -> np.ndarray:
    """
    ビン境界を狭義単調増加に整形する（非有限・重複を除去）。

    データが退化してビン境界が 2 本未満しか作れない場合は fallback の範囲で等幅に戻す。
    """
    e = np.asarray(list(edges), dtype=float)
    e = np.unique(e[np.isfinite(e)])
    if e.size >= 3:
        return e
    lo, hi = float(fallback[0]), float(fallback[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = 0.0, 1.0
    return np.linspace(lo, hi, max(2, int(n_bins)) + 1)


def heatmap_edges(
    x: np.ndarray,
    v: np.ndarray,
    x_var: str,
    x_bins: int = 25,
    y_bins: int = 30,
    x_edges_mode: str = 'uniform',
    y_edges_mode: str = 'log',
    upper_percentile: float = 99.5,
    x_limits: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D ヒストグラム（横軸 = 秩序変数 x、縦軸 = 速度 v）のビン境界を作る。

    - 横軸: 物理的な範囲（M は [-1, 1]、P は [0, 1]）を既定で等幅に切る
      （x_edges_mode = 'quantile' では等点数ビン）。x_limits を渡すとその範囲を
      使う（--heatmap_abs では |M| の [0, 1] を渡す）。
    - 縦軸: 速度は裾が重いので、既定（'log'）では**対数等間隔**ビンにして低速度側の
      分解能を確保し、上限を upper_percentile 分位点で打ち切る（それより速い点は
      ヒストグラムの範囲外になる）。'linear' は 0 から上限までの等幅、'quantile' は
      等点数ビン。

    Returns
    -------
    x_edges, y_edges : ndarray
        昇順のビン境界。
    """
    xs = np.asarray(x, dtype=float)
    vs = np.asarray(v, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(vs)
    xs, vs = xs[keep], vs[keep]

    default_lo = float(np.min(xs)) if xs.size else 0.0
    default_hi = float(np.max(xs)) if xs.size else 1.0
    limits = x_limits if x_limits is not None else X_PHYS_LIMITS.get(str(x_var))
    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    else:
        lo, hi = default_lo, default_hi
    n_x = max(2, int(x_bins))
    if str(x_edges_mode) == 'quantile' and xs.size >= n_x:
        x_edges = _monotone_edges(np.quantile(xs, np.linspace(0.0, 1.0, n_x + 1)),
                                  n_x, (float(lo), float(hi)))
    else:
        x_edges = np.linspace(float(lo), float(hi), n_x + 1)

    y_lo, y_hi = _velocity_limits(vs, 'log' if str(y_edges_mode) == 'log' else 'linear',
                                  upper_percentile=float(upper_percentile))
    n_y = max(2, int(y_bins))
    if str(y_edges_mode) == 'log':
        y_edges = _monotone_edges(
            np.geomspace(max(float(y_lo), np.finfo(float).tiny), float(y_hi), n_y + 1),
            n_y, (0.0, 1.0))
    elif str(y_edges_mode) == 'quantile' and vs.size >= n_y:
        y_edges = _monotone_edges(np.quantile(vs, np.linspace(0.0, 1.0, n_y + 1)),
                                  n_y, (float(y_lo), float(y_hi)))
    else:
        y_edges = np.linspace(float(y_lo), float(y_hi), n_y + 1)
    return x_edges, y_edges


def joint_density(
    x: np.ndarray,
    v: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    2D ヒストグラムの個数と同時確率密度（ヒートマップの色の量）を返す。

        P(v, x) = count / ( N_in * dx * dv )      [ (um/s)^-1 ]

    N_in はビン範囲内の点数なので、sum(count) = N_in かつ図の範囲内で
    ∫ P(v, x) dv dx = 1 になる。返り値の shape は (ny, nx)（行 = v、列 = x。
    np.histogram2d の (nx, ny) を転置した pcolormesh 規約）。
    """
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    counts, _, _ = np.histogram2d(np.asarray(x, dtype=float), np.asarray(v, dtype=float),
                                  bins=[xe, ye])
    counts = counts.T                      # (nx, ny) -> (ny, nx)
    n_in = int(counts.sum())
    area = np.diff(ye)[:, None] * np.diff(xe)[None, :]
    density = counts / float(max(1, n_in)) / area
    return counts, density, n_in


def heatmap_density_data(
    df_long: pd.DataFrame,
    target_beads: Sequence[dict],
    x_var: str,
    x_bins: int = 25,
    y_bins: int = 30,
    x_edges_mode: str = 'uniform',
    y_edges_mode: str = 'log',
    upper_percentile: float = 99.5,
    abs_values: bool = False,
) -> Optional[dict]:
    """
    条件（粒子径）をまたいでプールした (x, v) から 2D ヒストグラム（同時分布）を作る。

    abs_values = True では符号をもつ量（速度 v と、横軸が M のときは M）を絶対値に
    折り畳んでから集計する（|M| ∈ [0, 1]、|v| >= 0。P は元から非負なのでそのまま）。
    横軸の範囲も [0, 1] に切り替わる。

    Returns
    -------
    dict or None
        x / v / group（生データ: トレンド線と統計量の再計算に使う）、x_edges /
        y_edges / counts / density / n_points / n_points_in_range / ビンモード /
        abs_values。点が 1 つも無ければ None。
    """
    x, v, _, g = pooled_arrays(df_long, target_beads, x_var)
    if x.size == 0:
        return None
    if abs_values:                        # |v| と |M| に折り畳む（P は非負なので不変）
        x = np.abs(x)
        v = np.abs(v)
    x_limits = (X_PHYS_LIMITS_ABS if abs_values else X_PHYS_LIMITS).get(str(x_var))
    x_edges, y_edges = heatmap_edges(x, v, x_var, x_bins=x_bins, y_bins=y_bins,
                                     x_edges_mode=x_edges_mode,
                                     y_edges_mode=y_edges_mode,
                                     upper_percentile=upper_percentile,
                                     x_limits=x_limits)
    counts, density, n_in = joint_density(x, v, x_edges, y_edges)
    return {
        'x_var': str(x_var),
        'x': x, 'v': v, 'group': g,
        'x_edges': x_edges, 'y_edges': y_edges,
        'counts': counts, 'density': density,
        'n_points': int(x.size), 'n_points_in_range': int(n_in),
        'x_edges_mode': str(x_edges_mode), 'y_edges_mode': str(y_edges_mode),
        'abs_values': bool(abs_values),
        'bead_names': [b.get('name') for b in (target_beads or [])],
    }


def heatmap_density_table(data: Optional[dict], bead_label: str = 'all') -> pd.DataFrame:
    """
    2D ヒストグラムの中身を long 形式（1 行 = 1 ビン）のテーブルにする。

    count = 0 のビンは省く。y_center は対数等間隔ビンのときは幾何平均
    （= 対数軸上でのビンの中心）を入れる。abs_values は |v| / |M| に折り畳んだ
    ヒートマップ（--heatmap_abs）かどうかのフラグ。列: bead_name / x_variable /
    x_low / x_high / x_center / y_low / y_high / y_center / count / prob_density /
    bin_area / n_points / n_points_in_range / abs_values。
    """
    columns = ['bead_name', 'x_variable',
               'x_low', 'x_high', 'x_center', 'y_low', 'y_high', 'y_center',
               'count', 'prob_density', 'bin_area', 'n_points', 'n_points_in_range',
               'abs_values']
    if not data:
        return pd.DataFrame(columns=columns)
    xe = np.asarray(data['x_edges'], dtype=float)
    ye = np.asarray(data['y_edges'], dtype=float)
    counts = np.asarray(data['counts'], dtype=float)
    density = np.asarray(data['density'], dtype=float)
    iy, ix = np.nonzero(counts)          # counts は (ny, nx)
    if ix.size == 0:
        return pd.DataFrame(columns=columns)

    y_center = 0.5 * (ye[iy] + ye[iy + 1])
    if str(data.get('y_edges_mode', '')) == 'log' and bool(np.all(ye > 0.0)):
        y_center = np.sqrt(ye[iy] * ye[iy + 1])
    df = pd.DataFrame({
        'bead_name': str(bead_label),
        'x_variable': str(data['x_var']),
        'x_low': xe[ix], 'x_high': xe[ix + 1], 'x_center': 0.5 * (xe[ix] + xe[ix + 1]),
        'y_low': ye[iy], 'y_high': ye[iy + 1], 'y_center': y_center,
        'count': counts[iy, ix].astype(np.int64),
        'prob_density': density[iy, ix],
        'bin_area': (ye[iy + 1] - ye[iy]) * (xe[ix + 1] - xe[ix]),
        'n_points': int(data['n_points']),
        'n_points_in_range': int(data['n_points_in_range']),
        'abs_values': bool(data.get('abs_values', False)),
    })
    return df.sort_values(['y_low', 'x_low']).reset_index(drop=True)


def plot_heatmap(
    data: Optional[dict],
    x_var: str,
    out_dirs: Sequence[Path],
    basename: Optional[str] = None,
    title: str = '',
    sign_note: str = '',
    cmap: str = 'magma',
    log_color: bool = False,
    vmax_percentile: float = 99.0,
    marginals: bool = True,
    trend_bins: int = 12,
    trend_min_count: int = 10,
) -> None:
    """
    2D ヒートマップ（横軸 = M / P、縦軸 = v、色 = 同時確率密度 P(v, M)）を描く。

    色はビン内の個数を (N_in * dM * dv) で割った密度なので、図の範囲内で
    ∫ P dv dM = 1。白線は散布図と同一の等点数ビン中央値 ± IQR、上と右の周辺分布は
    個数ヒストグラム（ビン境界は 2D ヒストグラムと共通）で、y 軸は速度分布が
    大きく裾を引くことを踏まえて既定で対数等間隔ビン（= 対数軸表示）になる。

    data['abs_values'] = True（--heatmap_abs）のときは |v| と |M| に折り畳んだ図と
    して、軸・カラーバー・統計ボックスのラベルを絶対値用に切り替える。
    """
    if not data:
        print(f"  [SKIP] no heatmap data for {x_var}")
        return

    use_abs = bool(data.get('abs_values', False))
    x_label = X_LABELS_ABS[x_var] if use_abs else X_LABELS[x_var]
    y_label = VELOCITY_LABEL_ABS if use_abs else VELOCITY_LABEL
    joint_label = JOINT_LABELS_ABS[x_var] if use_abs else JOINT_LABELS[x_var]

    xe = np.asarray(data['x_edges'], dtype=float)
    ye = np.asarray(data['y_edges'], dtype=float)
    density = np.asarray(data['density'], dtype=float)
    x = np.asarray(data['x'], dtype=float)
    v = np.asarray(data['v'], dtype=float)
    g = np.asarray(data['group'], dtype=object)
    n_points = int(data['n_points'])
    n_in = int(data['n_points_in_range'])
    log_axis = (str(data.get('y_edges_mode', '')) == 'log' and float(ye[0]) > 0.0)

    # --- 色の正規化（1 ビンだけが突出して他が白飛びするのを防ぐ） ---
    pos = density[density > 0.0]
    norm: Normalize
    if pos.size == 0:
        norm = Normalize(vmin=0.0, vmax=1.0)
    elif log_color:
        vmin_c = float(np.min(pos))
        vmax_c = float(np.percentile(pos, float(vmax_percentile)))
        norm = LogNorm(vmin=vmin_c, vmax=max(vmax_c, vmin_c * 1.001))
    else:
        vmax_c = float(np.percentile(pos, float(vmax_percentile)))
        norm = Normalize(vmin=0.0, vmax=max(vmax_c, np.finfo(float).tiny))

    fig = plt.figure(figsize=(7.6, 6.4))
    try:  # 既定スタイルの autolayout を止めて、下の GridSpec の余白指定を守る
        fig.set_layout_engine('none')
    except Exception:
        pass

    ax_top = ax_right = cax = None
    if marginals:
        gs = fig.add_gridspec(2, 2, width_ratios=(4.6, 1.0), height_ratios=(1.0, 4.6),
                              left=0.135, right=0.925, bottom=0.20, top=0.85,
                              wspace=0.05, hspace=0.05)
        ax_top = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        cax = fig.add_subplot(gs[0, 1])
    else:
        gs = fig.add_gridspec(1, 1, left=0.135, right=0.90, bottom=0.20, top=0.90)
        ax = fig.add_subplot(gs[0, 0])

    mesh = ax.pcolormesh(xe, ye, density, cmap=cmap, norm=norm, shading='flat')
    ax.set_xscale('linear')
    ax.set_yscale('log' if log_axis else 'linear')
    ax.set_xlim(float(xe[0]), float(xe[-1]))
    ax.set_ylim(float(ye[0]), float(ye[-1]))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # 横軸はビン数が多いので目盛りを間引く（既定スタイルの大きいラベルの重なり防止）
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(False)

    # --- 周辺分布（個数。ビン境界は 2D ヒストグラムと共通） ---
    if marginals and ax_top is not None and ax_right is not None:
        ax_top.hist(x, bins=xe, color='0.5', lw=0)
        ax_top.set_yscale('log')
        ax_top.set_ylabel('counts', fontsize=10)
        ax_top.tick_params(labelbottom=False, labelsize=9, length=3)
        ax_top.grid(False)
        ax_top.set_title(title or '', fontsize=13, pad=8)
        ax_right.hist(v, bins=ye, orientation='horizontal', color='0.5', lw=0)
        ax_right.set_xscale('log')
        ax_right.set_xlabel('counts', fontsize=10)
        ax_right.tick_params(labelleft=False, labelsize=9, length=3)
        ax_right.grid(False)
    else:
        ax.set_title(title or '', fontsize=13)

    # --- カラーバー ---
    if cax is not None:
        fig.colorbar(mesh, cax=cax, orientation='horizontal')
        cax.set_title(f"{joint_label}\n{JOINT_DENSITY_UNIT.strip()}",
                      fontsize=10, pad=6)
        cax.tick_params(labelsize=8, length=3)
    else:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label(f"{joint_label} {JOINT_DENSITY_UNIT.strip()}", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    # --- トレンド線（等点数ビンの中央値 ± IQR。散布図と同じ定義） ---
    curve = bin_profile_records(x, v, n_bins=int(trend_bins), min_count=int(trend_min_count))
    if curve:
        cx = np.array([c['x_center'] for c in curve], dtype=float)
        cy = np.array([c['y_median'] for c in curve], dtype=float)
        cq25 = np.array([c['y_q25'] for c in curve], dtype=float)
        cq75 = np.array([c['y_q75'] for c in curve], dtype=float)
        if log_axis:  # 対数軸では下側の誤差が 0 以下にならないよう下限を切る
            keep = cy > 0.0
            cx, cy, cq25, cq75 = cx[keep], cy[keep], cq25[keep], cq75[keep]
            cq25 = np.maximum(cq25, cy * 1e-3)
        if cx.size:
            yerr = np.vstack([np.maximum(cy - cq25, 0.0), np.maximum(cq75 - cy, 0.0)])
            ax.errorbar(cx, cy, yerr=yerr, fmt='o-', color='white', ecolor='white',
                        ms=4.0, lw=1.8, capsize=2.0, markeredgecolor='0.25',
                        markeredgewidth=0.7, zorder=6)

    txt = _stats_text(x, v, g, x_var, abs_values=use_abs)
    txt += f"\n{n_in:,} / {n_points:,} points in range"
    if curve:
        txt += "\nwhite line: binned median $\\pm$ IQR"
    ax.text(0.025, 0.975, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color='0.15',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.88))
    if sign_note:
        # 注記は 1 行に収まらないことがあるため折り返して中央に置く
        wrapped = '\n'.join(textwrap.wrap(str(sign_note), width=115)) or str(sign_note)
        fig.text(0.5, 0.008, wrapped, ha='center', va='bottom', fontsize=8.0,
                 color='0.3', multialignment='center')

    mt_ori.save_figure_to_all(fig, basename or f"{X_FILE_TAG[x_var]}_heatmap", list(out_dirs))
    plt.close(fig)


def plot_per_condition(
    df_long: pd.DataFrame,
    df_binned: Optional[pd.DataFrame],
    bead: dict,
    x_var: str,
    out_dirs: Sequence[Path],
    yscale: str = 'linear',
    max_points: int = 4000,
    sign_note: str = '',
) -> None:
    """
    1 つの粒子径について、1 点 = 1 (貨物粒子 i, フレーム t) の散布図を描く。

    点の色は粒子 i（個体差が傾向を支配していないかを目視できる）、黒太線は等点数ビン
    ごとの v の中央値 ± 四分位（IQR）、右上に相関統計を表示する。
    """
    x, v, p, g = _condition_arrays(df_long, bead['name'], x_var, max_points=max_points)
    if x.size == 0:
        print(f"  [SKIP] no points for {bead['name']} / {x_var}")
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    parts = np.unique(p)
    cmap = plt.get_cmap('viridis')
    handles = []
    for k, pid in enumerate(parts):
        sel = p == pid
        col = cmap(0.05 + 0.85 * (k / max(1, len(parts) - 1))) if len(parts) > 1 else cmap(0.3)
        ax.scatter(x[sel], v[sel], s=16, alpha=0.45, lw=0, color=col,
                   rasterized=True, zorder=2)
        handles.append(Line2D([], [], ls='', marker='o', ms=7, color=col,
                              label=f'particle {int(pid)}'))

    curve = _binned_curve(df_binned, bead['name'], x_var)
    if curve is not None and len(curve) > 0:
        yerr = np.vstack([curve['y_median'] - curve['y_q25'],
                          curve['y_q75'] - curve['y_median']])
        ax.errorbar(curve['x_center'], curve['y_median'], yerr=yerr, fmt='o-',
                    color='0.1', ms=5, lw=2.0, capsize=2.5, zorder=5,
                    label='binned median $\\pm$ IQR')

    xlo, xhi = X_LIMITS[x_var]
    ax.set_xlim(xlo, xhi)
    ax.set_xscale('linear')
    ax.set_yscale(str(yscale))
    ylo, yhi = _velocity_limits(v, yscale)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(X_LABELS[x_var])
    ax.set_ylabel(VELOCITY_LABEL)
    ax.set_title(_bead_label(bead), pad=30)
    ax.grid(True, which='both', alpha=0.35)

    ax.text(0.02, 0.98, _stats_text(x, v, g, x_var), transform=ax.transAxes,
            ha='left', va='top', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.7', alpha=0.9))
    # 凡例はデータを隠さないよう軸の外側（上）に横 1 列で置く
    trend_handle = Line2D([], [], ls='-', marker='o', color='0.1',
                          label='binned median $\\pm$ IQR')
    if len(parts) <= 8:
        leg_handles = handles + [trend_handle]
        ncol = min(6, len(leg_handles))
    else:
        # 粒子数が多いときは凡例が大きくなりすぎるため、トレンド線のみ表示する
        leg_handles = [trend_handle]
        ncol = 1
    ax.legend(handles=leg_handles, fontsize=8, loc='lower left',
              bbox_to_anchor=(0.0, 1.005), ncol=ncol, framealpha=0.95,
              borderaxespad=0.0, handletextpad=0.35, columnspacing=0.8)
    if sign_note:
        fig.text(0.5, 0.005, sign_note, ha='center', va='bottom', fontsize=9, color='0.3')

    mt_ori.save_figure_to_all(fig, f"{X_FILE_TAG[x_var]}_{bead['name']}", list(out_dirs))
    plt.close(fig)


def plot_panels(
    df_long: pd.DataFrame,
    df_binned: Optional[pd.DataFrame],
    target_beads: Sequence[dict],
    x_var: str,
    out_dirs: Sequence[Path],
    ncols: int = 3,
    yscale: str = 'linear',
    max_points: int = 3000,
    sign_note: str = '',
) -> None:
    """
    条件（粒子径）ごとのパネル図（横軸 1 系統: M または P）。

    各パネルは 1 点 = 1 (粒子, フレーム) の散布図 + 等点数ビンの中央値 ± IQR。
    """
    selected = [b for b in target_beads
                if not df_long[(df_long['bead_name'] == b['name'])
                               & (df_long['x_var'] == x_var)].empty]
    if not selected:
        return

    n = len(selected)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows),
                             squeeze=False)

    y_all = df_long[df_long['bead_name'].isin([b['name'] for b in selected])]['v_um_s']
    if str(yscale) == 'log':
        y_lo, y_hi = _velocity_limits(y_all.to_numpy(dtype=float), yscale)
    else:
        y_lo, y_hi = 0.0, float(np.nanmax(y_all.to_numpy(dtype=float))) * 1.05

    xlo, xhi = X_LIMITS[x_var]
    for k, bead in enumerate(selected):
        ax = axes[k // ncols][k % ncols]
        x, v, p, g = _condition_arrays(df_long, bead['name'], x_var, max_points=max_points)
        col = bead.get('color', '#332288')
        ax.scatter(x, v, s=9, alpha=0.35, lw=0, color=col, rasterized=True, zorder=2)

        curve = _binned_curve(df_binned, bead['name'], x_var)
        if curve is not None and len(curve) > 0:
            yerr = np.vstack([curve['y_median'] - curve['y_q25'],
                              curve['y_q75'] - curve['y_median']])
            ax.errorbar(curve['x_center'], curve['y_median'], yerr=yerr, fmt='o-',
                        color='0.1', ms=3.5, lw=1.6, capsize=2.0, zorder=5)

        st = _corr_stats(x, v, g)
        ax.set_xlim(xlo, xhi)
        ax.set_yscale(str(yscale))
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis='both', which='both', labelsize=9)
        ax.set_xlabel(X_LABELS[x_var], fontsize=10)
        ax.set_ylabel(VELOCITY_LABEL, fontsize=10)
        ax.set_title(_bead_label(bead), fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.text(0.03, 0.97,
                f"$N$ = {st['within_n']:,}\n"
                f"$r$ = {st['pearson_r']:+.3f}\n"
                f"$\\rho$ = {st['spearman_rho']:+.3f}\n"
                f"within-$r$ = {st['within_r']:+.3f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.75', alpha=0.9))

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')

    fig.suptitle(f"{X_LABELS[x_var]} vs {VELOCITY_LABEL}", fontsize=13)
    if sign_note:
        fig.text(0.5, 0.005, sign_note, ha='center', va='bottom', fontsize=8.5, color='0.3')
    mt_ori.save_figure_to_all(fig, f"{X_FILE_TAG[x_var]}_all_beads", list(out_dirs))
    plt.close(fig)


def plot_overlay(
    df_long: pd.DataFrame,
    df_binned: Optional[pd.DataFrame],
    target_beads: Sequence[dict],
    x_var: str,
    out_dirs: Sequence[Path],
    yscale: str = 'linear',
    max_points: int = 3000,
    sign_note: str = '',
) -> None:
    """
    全条件を 1 つの軸に重ねた図（条件ごとに色・マーカーを変え、ビン中央値 ± IQR を重ねる）。
    """
    selected = [b for b in target_beads
                if not df_long[(df_long['bead_name'] == b['name'])
                               & (df_long['x_var'] == x_var)].empty]
    if not selected:
        return

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    xlo, xhi = X_LIMITS[x_var]
    lines: List[str] = []
    v_all: List[np.ndarray] = []
    for bead in selected:
        x, v, p, g = _condition_arrays(df_long, bead['name'], x_var, max_points=max_points)
        col = bead.get('color', '#332288')
        mk = bead.get('marker', 'o')
        ax.scatter(x, v, s=12, alpha=0.22, lw=0, color=col, rasterized=True, zorder=2)
        v_all.append(v)

        curve = _binned_curve(df_binned, bead['name'], x_var)
        st = _corr_stats(x, v, g)
        if curve is not None and len(curve) > 0:
            yerr = np.vstack([curve['y_median'] - curve['y_q25'],
                              curve['y_q75'] - curve['y_median']])
            ax.errorbar(curve['x_center'], curve['y_median'], yerr=yerr, fmt=mk + '-',
                        color=col, ms=5, lw=2.2, capsize=2.5, alpha=0.95, zorder=5,
                        label=f"{_bead_label(bead)}: $r$ = {st['pearson_r']:+.3f}")
        else:
            ax.plot([], [], mk, color=col,
                    label=f"{_bead_label(bead)}: $r$ = {st['pearson_r']:+.3f}")
        lines.append(f"{bead['name']:10s} N = {st['within_n']:6,}  "
                     f"$r$ = {st['pearson_r']:+.3f}  $\\rho$ = {st['spearman_rho']:+.3f}  "
                     f"within-$r$ = {st['within_r']:+.3f}")

    pooled = np.concatenate(v_all) if v_all else np.empty(0)
    y_lo, y_hi = _velocity_limits(pooled, yscale)
    ax.set_xlim(xlo, xhi)
    ax.set_yscale(str(yscale))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(X_LABELS[x_var])
    ax.set_ylabel(VELOCITY_LABEL)
    ax.set_title(f"{X_LABELS[x_var]} vs {VELOCITY_LABEL}")
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.92)
    ax.text(0.99, 0.02, '\n'.join(lines), transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.75', alpha=0.9))
    if sign_note:
        fig.text(0.5, 0.005, sign_note, ha='center', va='bottom', fontsize=9, color='0.3')

    mt_ori.save_figure_to_all(fig, f"{X_FILE_TAG[x_var]}_overlay", list(out_dirs))
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Cargo velocity v_{i,t} vs the mean Ising spin M_{i,t} and the local polar "
                     "order P_{i,t} in the region under each cargo particle."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="データルート（<root>/<beads>/<date>/<exp>/GFP_flows.h5）")
    parser.add_argument('--beads', type=str, nargs='+', default='all',
                        help="対象の粒子径（例: beads1um 1um 3um / all）")
    parser.add_argument('--output_dir', type=str, default='figure/cargo_spin_velocity',
                        help="出力ディレクトリ（相対パスはスクリプト基準）")
    parser.add_argument('--no_save_root', action='store_true',
                        help="データルート側の figure/ への保存を行わない")
    parser.add_argument('--pixel_stride', type=int, default=4,
                        help="フローの画素間引き幅（円板内のサンプル数を確保するため既定 4）")
    parser.add_argument('--frame_stride', type=int, default=5, help="フレーム間引き幅")
    parser.add_argument('--max_frames_per_exp', type=int, default=None,
                        help="（デバッグ用）1 実験あたりの使用フレーム数上限")
    parser.add_argument('--region_factor', type=float, default=2.0,
                        help="円板半径 = region_factor * R_c（ビーズ半径の何倍か）")
    parser.add_argument('--min_region_um', type=float, default=1.0,
                        help="円板半径の下限 [um]（小さいビーズでサンプル数を確保する）")
    parser.add_argument('--region_inner_factor', type=float, default=0.0,
                        help="内側くり抜き半径 = factor * R_c（0 = 円板、1 = ビーズ直下を除外した円環）")
    parser.add_argument('--min_region_pixels', type=int, default=6,
                        help="円板内の有効画素数の下限（未満は棄却）")
    parser.add_argument('--min_valid_fraction', type=float, default=0.5,
                        help="円板内の有効画素率の下限（未満は棄却）")
    parser.add_argument('--min_flow_mag', type=float, default=1e-4,
                        help="有効画素とみなす最小流速 |u|（px/frame）")
    parser.add_argument('--director', type=str, default='global', choices=['global', 'local'],
                        help="イジングスピンの基準軸（global = 大域ネマチック主軸 theta_nem(t)、"
                             "local = MTs_im_theta.zarr の局所配向場）")
    parser.add_argument('--theta_sign', type=str, default='auto', choices=['auto', '+1', '-1'],
                        help="MTs_im_theta.zarr のミラー規約補正（auto で自動判定）")
    parser.add_argument('--mirror_margin', type=float, default=0.05,
                        help="--theta_sign auto の判定マージン（プール <cos 2 Delta theta>）")
    parser.add_argument('--scale', type=float, default=0.11, help="空間スケール [um/px]")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="フレーム間隔 [s]")
    parser.add_argument('--tau', type=int, default=1,
                        help="速度計算のラグ [frames]（1 = 連続フレーム、光学フローと同時刻）")
    parser.add_argument('--velocity', type=str, default='tracked', choices=['tracked', 'flow'],
                        help="縦軸の速度: tracked = 軌跡から、flow = 円板内の平均フロー")
    parser.add_argument('--v_bins', type=int, default=12, help="トレンド線の等点数ビン数")
    parser.add_argument('--v_bin_min_count', type=int, default=10,
                        help="ビンに必要な最小点数（未満のビンは非表示）")
    parser.add_argument('--yscale', type=str, default='linear', choices=['linear', 'log'],
                        help="速度軸のスケール")
    parser.add_argument('--ncols', type=int, default=3, help="パネル図の列数")
    parser.add_argument('--max_points_per_panel', type=int, default=3000,
                        help="パネル図 1 枚あたりの最大描画点数（0 以下で全点）")
    parser.add_argument('--max_points_single', type=int, default=6000,
                        help="条件別図 1 枚あたりの最大描画点数（0 以下で全点）")
    parser.add_argument('--no_heatmap', action='store_true',
                        help="2D ヒートマップ（横軸 M / P, 縦軸 v, 色 = 同時確率密度）を出力しない")
    parser.add_argument('--heatmap_per_condition', action='store_true',
                        help="ヒートマップを条件（粒子径）ごとにも出力する（既定は全条件プールのみ）")
    parser.add_argument('--heatmap_abs', action='store_true',
                        help="符号をもつ量（速度 v とスピン平均 M）を絶対値に折り畳んだ"
                             "ヒートマップ（横軸 |M|, 縦軸 |v|）も追加出力する")
    parser.add_argument('--heatmap_bins_x', type=int, default=25,
                        help="ヒートマップ横軸（M / P）のビン数")
    parser.add_argument('--heatmap_bins_y', type=int, default=30,
                        help="ヒートマップ縦軸（v）のビン数")
    parser.add_argument('--heatmap_x_edges', type=str, default='uniform',
                        choices=['uniform', 'quantile'],
                        help="ヒートマップ横軸のビン境界（uniform = 物理範囲を等幅、quantile = 等点数）")
    parser.add_argument('--heatmap_y_edges', type=str, default='log',
                        choices=['log', 'linear', 'quantile'],
                        help="ヒートマップ縦軸（速度）のビン境界（log = 対数等間隔で低速度側を分解、既定）")
    parser.add_argument('--heatmap_upper_percentile', type=float, default=99.5,
                        help="縦軸の上限を決める速度の分位点 [%%]（これより速い点は範囲外）")
    parser.add_argument('--heatmap_vmax_percentile', type=float, default=99.0,
                        help="色の上限を決める密度の分位点 [%%]（1 ビンだけが突出するのを防ぐ）")
    parser.add_argument('--heatmap_log_color', action='store_true',
                        help="ヒートマップの色を対数スケールにする")
    parser.add_argument('--heatmap_cmap', type=str, default='magma',
                        help="ヒートマップのカラーマップ名")
    parser.add_argument('--no_heatmap_marginals', action='store_true',
                        help="ヒートマップに周辺分布（上 = x の個数、右 = v の個数）を付けない")
    parser.add_argument('--flow_cache', type=str, default='auto',
                        choices=['auto', 'off', 'refresh'],
                        help="光学フローの間引きキャッシュ（plot_mt_orientation_distribution と共通）")
    parser.add_argument('--flow_cache_name', type=str, default=None,
                        help="キャッシュファイル名（既定 mt_flow_cache_s{st}_f{fs}.h5）")
    parser.add_argument('--flow_cache_dir', type=str, default=None,
                        help="キャッシュを置くディレクトリ（NAS が遅いときはローカルを指定）")
    parser.add_argument('--no_progress', action='store_true', help="tqdm を無効化")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else mt_ori.find_default_root()
    if root_dir is None or not Path(root_dir).exists():
        raise FileNotFoundError("Data root directory not found. Please specify it with --root_dir.")

    out_arg = Path(args.output_dir).expanduser()
    out_dirs: List[Path] = [out_arg if out_arg.is_absolute() else (CURRENT_DIR / out_arg)]
    if not args.no_save_root:
        out_dirs.append(Path(root_dir) / 'figure' / 'cargo_spin_velocity')
    out_dirs = list(dict.fromkeys(out_dirs))
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    target_beads = mt_ori.parse_target_beads(args.beads, BEADS_INFO)
    if not target_beads:
        raise RuntimeError("No target bead conditions selected.")

    print("=" * 78)
    print(" Cargo velocity v_{i,t} vs Ising spin mean M_{i,t} and local polar order P_{i,t}")
    print("=" * 78)
    print(f"Data Root Directory : {root_dir}")
    print(f"Output Directories  : {', '.join(str(d) for d in out_dirs)}")
    print(f"Target Beads        : {[b['name'] for b in target_beads]}")
    print(f"Pixel / frame stride: {args.pixel_stride} px / {args.frame_stride} frames")
    print(f"Director            : {args.director} (theta_sign = {args.theta_sign})")
    print(f"Velocity            : {args.velocity} "
          f"(tracked: v = |dr| / ({args.tau} x {args.frame_interval} s), scale = {args.scale} um/px)")
    print(f"Region under cargo  : R = max({args.region_factor} R_c, {args.min_region_um} um), "
          f"inner = {args.region_inner_factor} R_c, min pixels = {args.min_region_pixels}, "
          f"min valid frac = {args.min_valid_fraction}")
    for bead in target_beads:
        r_um = region_radius_um(bead, args.region_factor, args.min_region_um)
        print(f"    {bead['name']:10s}: R_c = {bead['radius_um']:.3f} um -> "
              f"R_region = {r_um:.3f} um = {r_um / args.scale:.1f} px "
              f"({r_um / args.scale / max(1, args.pixel_stride):.1f} grid px)")
    print("-" * 78)

    results: List[dict] = []
    for bead in target_beads:
        exp_dirs = [p for p in mt_ori.find_experiment_dirs(root_dir, bead['name'])
                    if (p / TRACKS_NAME).exists()]
        if not exp_dirs:
            print(f"[SKIP] {bead['name']}: no experiment dir with {TRACKS_NAME}", flush=True)
            continue
        print(f"[{bead['name']}] {len(exp_dirs)} experiment dir(s)", flush=True)
        for exp_dir in exp_dirs:
            cache_name = mt_ori.resolve_flow_cache_name(
                args.flow_cache_dir, root_dir, exp_dir,
                args.pixel_stride, args.frame_stride, args.flow_cache_name)
            res = process_experiment_cargo(
                exp_dir, bead,
                pixel_stride=args.pixel_stride, frame_stride=args.frame_stride,
                max_frames_per_exp=args.max_frames_per_exp,
                min_flow_mag=args.min_flow_mag,
                min_valid_fraction=args.min_valid_fraction,
                min_region_pixels=args.min_region_pixels,
                region_factor=args.region_factor,
                min_region_um=args.min_region_um,
                region_inner_factor=args.region_inner_factor,
                scale=args.scale, frame_interval=args.frame_interval, tau=args.tau,
                director=args.director, velocity=args.velocity,
                flow_cache=args.flow_cache, flow_cache_name=cache_name,
                progress=not args.no_progress)
            if res is None:
                continue
            print(f"    {exp_dir.name}: {res['n_frames_used']}/{res['n_frames_total']} frames, "
                  f"{len(res['records'])} points, theta = {res['theta_source']}, "
                  f"flow = {res['flow_cache_source']}, "
                  f"rejected(region/no-v) = {res['n_rejected_region']}/"
                  f"{res['n_rejected_no_velocity']}", flush=True)
            results.append(res)

    if not results:
        raise RuntimeError("No experiment could be processed (check --root_dir / --beads).")

    # --- ディレクター符号（n -> -n のミラー）の選択 ---
    # plot_ising_magnetization.choose_director_sign と共通の判定を使う（プール
    # <cos 2 Delta theta> が大きい側を採用）。P はこの影響を受けない。
    sign, sign_info = ising_plot.choose_director_sign(
        results, args.theta_sign, args.mirror_margin)
    print("-" * 78)
    print(f"Director sign decision: {sign:+d} ({sign_info['decision']})")
    print(f"    pooled <cos 2 Delta theta>: +theta = {sign_info['S2_plus']:+.4f}, "
          f"-theta = {sign_info['S2_minus']:+.4f}")
    if int(sign) < 0 and str(args.theta_sign) == 'auto':
        print("   [NOTE] MTs_im_theta.zarr の角度規約が光学フローとミラー関係にあるため、"
              "theta -> -theta として採用しました。")
    print("-" * 78)

    # --- テーブル ---
    df_points = points_table(results, sign=sign, velocity=args.velocity)
    if df_points.empty:
        raise RuntimeError("No (M, v) pairs were extracted. Check --region_factor / "
                           "--min_region_pixels / --min_flow_mag.")
    df_long = long_points_table(df_points)
    df_summary = summary_table(df_long, velocity=args.velocity,
                               n_bins=args.v_bins, min_count=args.v_bin_min_count)
    df_binned = binned_table(df_long, n_bins=args.v_bins,
                             min_count=args.v_bin_min_count)
    df_extract = extraction_table(results)

    sign_note = (f"region = max({args.region_factor} R_c, {args.min_region_um} um) disk under "
                 f"the cargo, velocity = {args.velocity} (tau = {args.tau} frames), "
                 f"director = {args.director}, dir_sign = {sign} ({sign_info['decision']}), "
                 f"pixel/frame stride = {args.pixel_stride}/{args.frame_stride}")
    heat_note = (f"region = max({args.region_factor} R_c, {args.min_region_um} um) disk, "
                 f"velocity = {args.velocity} (tau = {args.tau} frames), "
                 f"director = {args.director}, dir_sign = {sign}, "
                 f"pixel/frame stride = {args.pixel_stride}/{args.frame_stride}")

    print(f"Total points (i, t): {len(df_points):,} "
          f"({df_long['bead_name'].nunique()} condition(s))")
    mt_ori.save_csv_to_all(df_points, 'cargo_spin_velocity_points', out_dirs)
    mt_ori.save_csv_to_all(df_summary, 'cargo_spin_velocity_summary', out_dirs)
    mt_ori.save_csv_to_all(df_binned, 'cargo_spin_velocity_binned', out_dirs)
    mt_ori.save_csv_to_all(df_extract, 'cargo_spin_velocity_extraction', out_dirs)

    # --- 作図（横軸 2 系統 x 3 種類 + 2D ヒートマップ） ---
    df_heat_frames: List[pd.DataFrame] = []
    for x_var in X_VARIABLES:
        for bead in target_beads:
            plot_per_condition(df_long, df_binned, bead, x_var, out_dirs,
                               yscale=args.yscale,
                               max_points=args.max_points_single,
                               sign_note=sign_note)
        plot_panels(df_long, df_binned, target_beads, x_var, out_dirs,
                    ncols=args.ncols, yscale=args.yscale,
                    max_points=args.max_points_per_panel, sign_note=sign_note)
        plot_overlay(df_long, df_binned, target_beads, x_var, out_dirs,
                     yscale=args.yscale, max_points=args.max_points_per_panel,
                     sign_note=sign_note)

        # --- 2D ヒートマップ（全条件プール、色 = 同時確率密度 P(v, M)） ---
        if args.no_heatmap:
            continue
        heat_kwargs = dict(
            x_bins=args.heatmap_bins_x, y_bins=args.heatmap_bins_y,
            x_edges_mode=args.heatmap_x_edges, y_edges_mode=args.heatmap_y_edges,
            upper_percentile=args.heatmap_upper_percentile)
        plot_kwargs = dict(
            sign_note=heat_note, cmap=args.heatmap_cmap,
            log_color=args.heatmap_log_color,
            vmax_percentile=args.heatmap_vmax_percentile,
            marginals=not args.no_heatmap_marginals,
            trend_bins=args.v_bins, trend_min_count=args.v_bin_min_count)

        # --heatmap_abs では符号をもつ量（v と M）を絶対値に折り畳んだ版も追加する
        abs_modes = (False, True) if args.heatmap_abs else (False,)
        for use_abs in abs_modes:
            suffix = '_heatmap_abs' if use_abs else '_heatmap'
            joint_label = JOINT_LABELS_ABS[x_var] if use_abs else JOINT_LABELS[x_var]
            data = heatmap_density_data(df_long, target_beads, x_var,
                                        abs_values=use_abs, **heat_kwargs)
            if data is not None:
                df_heat_frames.append(heatmap_density_table(data, 'all'))
                print(f"  Heatmap {X_FILE_TAG[x_var]}{suffix}: "
                      f"{data['n_points_in_range']:,} / {data['n_points']:,} points, "
                      f"{data['counts'].shape[1]} x {data['counts'].shape[0]} bins")
                plot_heatmap(data, x_var, out_dirs,
                             basename=f"{X_FILE_TAG[x_var]}{suffix}",
                             title=(f"{joint_label} : all conditions "
                                    f"($N$ = {data['n_points']:,})"),
                             **plot_kwargs)
            for bead in (target_beads if args.heatmap_per_condition else []):
                d_bead = heatmap_density_data(df_long, [bead], x_var,
                                              abs_values=use_abs, **heat_kwargs)
                if d_bead is None:
                    continue
                df_heat_frames.append(heatmap_density_table(d_bead, bead['name']))
                plot_heatmap(d_bead, x_var, out_dirs,
                             basename=f"{X_FILE_TAG[x_var]}{suffix}_{bead['name']}",
                             title=(f"{joint_label} : {_bead_label(bead)} "
                                    f"($N$ = {d_bead['n_points']:,})"),
                             **plot_kwargs)

    if df_heat_frames:
        mt_ori.save_csv_to_all(pd.concat(df_heat_frames, ignore_index=True),
                               'cargo_spin_velocity_heatmap', out_dirs)

    # --- コンソール要約 ---
    print("-" * 78)
    print(" Summary (per condition x x-variable)")
    cols = ['bead_name', 'x_variable', 'n_points', 'n_experiments', 'n_particles',
            'x_mean', 'x_std', 'v_median_um_s', 'pearson_r', 'pearson_p',
            'spearman_rho', 'slope_ols', 'r2_ols', 'within_r']
    print(df_summary[[c for c in cols if c in df_summary.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()


