#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_mt_orientation_distribution.py
===================================

光学フロー (Optical Flow) から求めた微小管（MT）フロー配向角 phi(x, y, t) の分布を、
各フレームの**大域ネマチック主軸角 theta_nem(t)** を基準にした角度差

    Delta theta(x, y, t) = wrap_pi( phi(x, y, t) - theta_nem(t) )

のヒストグラムおよびレーダーチャート（極座標ローズ図）として可視化するスクリプトです。

配向角は headless（向きを持たない軸）なので、フローがネマチック主軸に沿って流れる場合、
Delta theta の分布は 0 と +-pi に 2 つのピークを持つ対称な形になります
（ネマチック軸の表裏を区別しないため）。本スクリプトはこの「0 / pi ピーク構造」を
条件（貨物粒子径 0.63 - 20 um）ごとに定量化します。

【データ源】
- 各実験ディレクトリの GFP_flows.h5（shape = (frame, 2, y, x) もしくは (frame, y, x, 2)）
  からフローベクトル (u_x, u_y) をピクセル単位で読み、phi = arctan2(u_y, u_x) を計算する。
- 大域ネマチック主軸角 theta_nem(t) は libs.calc_bg_angular_correlation.load_nematic_thetas と
  同一の 2 テンソル平均（0.5 * arctan2(<sin 2 theta>, <cos 2 theta>)）で、MTs_im_theta.zarr の
  局所配向角マップから算出する（使用フレームのみを対象とする高速版 load_nematic_directors を使用）。
  MTs_im_theta.zarr が無い場合は、統計に用いる間引き格子のフロー配向 phi に同じ定義を適用して
  フレームごとに推定する。

【注意（0 と +-pi は同一方向）】
光学フローの向きは符号を持つベクトルなので Delta theta の分布は 0 と +-pi に 2 つのピークを持つ。
+pi と -pi は同一方向のため、展開分布ヒストグラムでは 1 つの「反平行ローブ」が左右両端のビンに
半分ずつ現れる（レーダーチャートでは pi の位置に連続した 1 つのローブとして描かれる）。
ネマチック折り返し分布（mod pi）では 0 に単一ピークとなる。

【角度規約の整合（--theta_sign auto）】
MTs_im_theta.zarr の角度規約（libs/AFT_tools.py は least-moment 角を -1 倍する実装）は、
光学フローの座標系（x = 列, y = 行, phi = arctan2(u_y, u_x)）とミラー関係にあるデータセットがある。
実際に GFP 生画像から (i) 構造テンソル配向、(ii) 輝度不変則 (Lucas-Kanade) で推定した MT バンドル方向と
比較すると、本データセットではフロー配向が -theta_zarr（ミラー側）と強く一致する
（プール <cos 2 Delta theta> = 0.72 (-theta) 対 -0.28 (+theta)）。
そこで本スクリプトは ±theta_nem の両方を評価し、Delta theta が 0 / pi に鋭くピークを持つ側を自動採用する
（--theta_sign auto, 既定）。採用した符号は CSV の theta_sign 列と図タイトルに記録される。
--theta_sign +1 / -1 で強制指定でき、判定閾値は --mirror_margin で変更できる。

【ネマチック主軸の時間変化（導入済み）】
theta_nem は実験ごとの**フレームごと（= 時間変化あり）**に算出し、Delta theta は各フレーム自身の
主軸 theta_nem(t) を基準に計算する（フレーム間で固定の軸ではない）。theta_nem(t) は
1. 時系列図 mt_orientation_theta_time_series.png/.svg（条件別に実験ごとの theta_nem(t) を描画）
2. long 形式 CSV mt_orientation_theta_nem_timeseries.csv（frame / time_s / theta_nem_deg ...）
3. 実験別 CSV の統計列 theta_nem_mean_deg, theta_nem_std_deg_continuous, theta_nem_range_deg,
   theta_nem_drift_deg, theta_nem_axis_order
として確認できる。theta_nem は headless な軸（pi 周期）なので、2 theta を unwrap して連続化した
代表値でレンジ / ドリフト / 標準偏差を評価する（時間軸は --frame_interval, 既定 4.0 s）。

【ヒストグラムの定義】
- 展開分布: Delta theta を [-pi, pi) にラップ（+pi は -pi と同一方向として左端ビンに代表される）。
  ピークは 0 と +-pi。等方分布（ランダム配向）の期待値は 1/(2 pi)。
- ネマチック折り返し分布: Delta theta を mod pi で (-pi/2, pi/2] に折り返す。
  ネマチック秩序を 1 つのピーク（0）で評価でき、秩序度 <cos 2 Delta theta> と直接対応する。
  等方分布の期待値は 1/pi。

【重み付け】
--weighting none      : 有効画素を等重みで計数（既定。角度分布そのもの）
--weighting magnitude : 流速ノルム |u| で重み付け（速い領域を重視）

【サンプリングと計算コスト（実測ベース）】
GFP_flows.h5 は 1 実験あたり 10 GB を超え、h5py のチャンクは 1 フレーム = (1, 2, H, W) = 22 MB
（2160x2560 float16 の場合）であるため、ストライド読み出しでも「使用フレーム数 x フレームサイズ」の
バイト数を NAS から読むことになる（pixel_stride を下げても I/O は増えない / 計算だけが増える）。
実測（NAS 上, 2160x2560, pixel_stride 8）:
  - チャンク読み出し: 292 ms/frame（cold, ~76 MB/s）〜 1771 ms/frame（NAS 混雑時）
    ※ ch0/ch1 を別アクセスで読むと同一チャンクを 2 回読むため 581 ms/frame（1 アクセス統合で 2 倍高速）
  - 角度計算 + ビン集計: 6.1 ms/frame（bincount 化前は 12.9 ms/frame）→ I/O が支配的（>95%）
  - 複数実験の並列読みは NAS では逆効果（実測: 3 並列で合計 16.6 MB/s < 単一 69 MB/s）
したがって本スクリプトは「1 アクセス 2ch 読み + 単一パス集計 + フローキャッシュ」で速度を確保する:
  - --flow_cache auto（既定）: 解析で使うフレームの間引き済みコピー (mx, my) を
    mt_flow_cache_s{pixel_stride}_f{frame_stride}.h5 として最初の解析パス中に書き出し
    （追加 I/O 無し）、次回以降はそれを読む。実測 106 フレームで 187.7 s -> 17.1 s（11 倍高速）。
  - --flow_cache_dir を指定するとキャッシュをローカルディスク等へ置ける（NAS が遅いときに有効）。
  - 統計量を最大限使いたい場合は --pixel_stride 1 --frame_stride 1 を指定する。
    pixel_stride を下げるコストは計算分のみ（I/O はほぼ不変）。

【出力ファイル】
既定の出力先は (1) <作業ディレクトリ>/figure/mt_orientation と
(2) <root_dir>/figure/mt_orientation の 2 箇所（--no_save_root で (2) を省略, --output_dir で (1) を変更）。
1. mt_orientation_histogram_panels.png / .svg      : 条件別 Delta theta ヒストグラム（実験間平均 +- SEM）
2. mt_orientation_histogram_overlay.png / .svg     : 全条件重ね書き Delta theta ヒストグラム
3. mt_orientation_radar_panels.png / .svg          : 条件別レーダーチャート（極座標ローズ図）
4. mt_orientation_radar_overlay.png / .svg         : 全条件重ね書きレーダーチャート
5. mt_orientation_nematic_folded.png / .svg        : ネマチック折り返し分布 (Delta theta mod pi) の全条件比較
6. mt_orientation_theta_time_series.png / .svg     : フレームごとの大域ネマチック主軸角 theta_nem(t) の時間発展
7. mt_orientation_per_experiment.csv               : 実験ごとの円統計（R, <cos>, <cos 2>, |Delta theta|, 平行/反平行比,
                                                     theta_nem(t) の変化統計）
8. mt_orientation_histogram.csv                    : 条件 x 角度ビンごとの平均密度 +- SEM（展開分布）
9. mt_orientation_nematic_folded_histogram.csv     : 同上（ネマチック折り返し分布）
10. mt_orientation_theta_nem_timeseries.csv        : 実験 x フレームごとの theta_nem(t)（long 形式）
11. mt_orientation_summary.csv                     : 条件ごとの代表値（mean +- SEM, プール値, サンプル数）
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# NAS / 共有ボリュームでの HDF5 ファイルロックエラー防止（h5py import 前に設定が必要）
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# 親ディレクトリのパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# 大域ネマチック主軸角 theta_nem(t) の定義は libs.calc_bg_angular_correlation.load_nematic_thetas と
# 同一（局所配向角マップの 2 テンソル平均）。本スクリプトでは使用フレームのみを対象とする高速版
# load_nematic_directors を用いる（同関数は毎回全フレームを走査するため大規模フローでは I/O が重い）。

# スタイル適用
_style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if _style_path.exists():
    try:
        plt.style.use(str(_style_path))
    except Exception:
        pass

FLOW_NAME = 'GFP_flows.h5'
TRACKS_NAME = 'beads_tracks.csv'

# データルート候補（存在するものを自動選択）
POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]

# 貨物粒子（ビーズ）の直径・半径・マーカー
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s"},
]


def _style_colors(n: int) -> List[str]:
    """プロジェクトスタイルの色サイクルから n 色を取得する（mplstyle は '#' 無し表記）。"""
    try:
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        cols = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
    out = []
    for i in range(n):
        c = cols[i % len(cols)]
        out.append('#' + c if (isinstance(c, str) and not c.startswith('#')) else c)
    return out


for _b, _c in zip(BEADS_INFO, _style_colors(len(BEADS_INFO))):
    _b["color"] = _c

BEAD_LOOKUP = {b['name']: b for b in BEADS_INFO}


def find_default_root() -> Optional[Path]:
    """存在するデータルートを返す（見つからない場合は None）。"""
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um']:
                if (r / b).exists() and len(list((r / b).glob(f'*/*/{FLOW_NAME}'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return None


def normalize_bead_name(raw_name: str) -> Optional[str]:
    """
    入力文字列（例: 'beads06um', 'bead06um', '06um', '0.6um', '0.6', '1um', '1', etc.）を
    BEADS_INFO の標準名 ('beads06um' 等) に正規化する。
    """
    s = raw_name.strip().lower()
    mapping = {
        'beads06um': 'beads06um', 'bead06um': 'beads06um', '06um': 'beads06um', '0.6um': 'beads06um', '0.6': 'beads06um', '06': 'beads06um',
        'beads1um': 'beads1um', 'bead1um': 'beads1um', '1um': 'beads1um', '1.0um': 'beads1um', '1.18um': 'beads1um', '1': 'beads1um',
        'beads3um': 'beads3um', 'bead3um': 'beads3um', '3um': 'beads3um', '3.0um': 'beads3um', '3.37um': 'beads3um', '3': 'beads3um',
        'beads5um': 'beads5um', 'bead5um': 'beads5um', '5um': 'beads5um', '5.0um': 'beads5um', '5': 'beads5um',
        'beads7um': 'beads7um', 'bead7um': 'beads7um', '7um': 'beads7um', '7.0um': 'beads7um', '7.24um': 'beads7um', '7': 'beads7um',
        'beads20um': 'beads20um', 'bead20um': 'beads20um', '20um': 'beads20um', '20.0um': 'beads20um', '20': 'beads20um',
    }
    return mapping.get(s, None)


def parse_target_beads(beads_args: Union[str, List[str]], beads_info: List[dict] = BEADS_INFO) -> List[dict]:
    """argparse の引数（リストまたはカンマ/スペース区切りの文字列）から対象ビーズ情報のリストを抽出する。"""
    if isinstance(beads_args, str):
        raw_list = [beads_args]
    else:
        raw_list = list(beads_args)

    tokens: List[str] = []
    for item in raw_list:
        tokens.extend([t for t in item.replace(',', ' ').split() if t])

    if not tokens or 'all' in [t.lower() for t in tokens]:
        return list(beads_info)

    selected_names = set()
    for token in tokens:
        norm = normalize_bead_name(token)
        if norm:
            selected_names.add(norm)
        else:
            found = False
            for b in beads_info:
                if b['name'].lower() == token.lower():
                    selected_names.add(b['name'])
                    found = True
                    break
            if not found:
                print(f"[WARNING] Unrecognized bead specification: '{token}'. "
                      f"Available: {[b['name'] for b in beads_info]}")

    return [b for b in beads_info if b['name'] in selected_names]


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    """<root_dir>/<bead_name>/{date/}exp 配下で GFP_flows.h5 を持つ実験ディレクトリを返す。"""
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []

    def _valid(p: Path) -> bool:
        return p.is_dir() and (p / FLOW_NAME).exists()

    exp_dirs = [p for p in sorted(base.glob('*/*')) if _valid(p)]
    if not exp_dirs:
        exp_dirs = [p for p in sorted(base.glob('*')) if _valid(p)]
    return exp_dirs



def safe_save_csv(df: pd.DataFrame, target_path: Path, max_retries: int = 5) -> None:
    """NAS 等の一時的な I/O 失敗に耐える CSV 保存（既存スクリプトと同じ方式）。"""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            df.to_csv(target_path, index=False)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(df.to_csv(index=False))
                    return
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    return
            time.sleep(0.5)


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]) -> List[Path]:
    """すべての出力ディレクトリへ CSV を保存する。"""
    saved = []
    for d in out_dirs:
        target = Path(d) / f"{basename}.csv"
        safe_save_csv(df, target)
        saved.append(target)
    print(f"  Saved CSV: {basename}.csv -> {len(saved)} dir(s)")
    return saved


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300) -> List[Path]:
    """すべての出力ディレクトリへ PNG / SVG を保存する。"""
    saved = []
    for d in out_dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'svg'):
            target = d / f"{basename}.{ext}"
            fig.savefig(target, dpi=dpi, bbox_inches='tight')
            saved.append(target)
    print(f"  Saved figure: {basename}.png/.svg -> {len(out_dirs)} dir(s)")
    return saved


# =========================================================================
# 角度演算ユーティリティ
# =========================================================================

def load_nematic_directors(
    exp_dir: Path,
    frame_ids: Sequence[int],
    num_frames: int,
    max_zarr_mb: float = 256.0,
) -> Optional[np.ndarray]:
    """
    MTs_im_theta.zarr（局所配向角マップ）から、指定フレームの大域ネマチック主軸角
    theta_nem(t) を算出する。ファイルが無い / 読めない場合は None を返す。

    定義は libs.calc_bg_angular_correlation.load_nematic_thetas と同一（2 テンソル平均）:
        theta_nem(t) = 0.5 * arctan2(<sin 2 theta>, <cos 2 theta>)

    ※ 使用フレームのみを対象とし、小さな配列は一括読み込み + ベクトル化で高速化する
      （同関数は毎回全フレームを走査するため大規模フローでは I/O が支配的になる）。
    """
    path = Path(exp_dir) / "MTs_im_theta.zarr"
    if not path.exists():
        return None

    try:
        import zarr
        z = zarr.open_array(str(path), mode='r')
        n = int(min(int(z.shape[0]), int(num_frames)))
        if n <= 0:
            return None
        thetas = np.zeros(int(num_frames), dtype=np.float64)
        reduce_axes = tuple(range(1, len(z.shape)))
        nbytes = float(np.prod(z.shape)) * float(np.dtype(z.dtype).itemsize)

        if nbytes <= float(max_zarr_mb) * 1e6:
            arr = np.asarray(z[:n], dtype=np.float64)
            s2 = np.nanmean(np.sin(2.0 * arr), axis=reduce_axes)
            c2 = np.nanmean(np.cos(2.0 * arr), axis=reduce_axes)
            with np.errstate(invalid='ignore'):
                th = 0.5 * np.arctan2(s2, c2)
            bad = ~np.isfinite(th)
            if np.any(bad):
                fallback = np.nanmean(arr[bad], axis=reduce_axes) if arr.ndim > 1 else np.nanmean(arr)
                th[bad] = np.asarray(fallback, dtype=np.float64).ravel()
            thetas[:n] = np.where(np.isfinite(th), th, 0.0)
        else:
            for t in frame_ids:
                if t >= n:
                    continue
                th_frame = np.asarray(z[int(t)], dtype=np.float64)
                s2 = np.nanmean(np.sin(2.0 * th_frame))
                c2 = np.nanmean(np.cos(2.0 * th_frame))
                if not np.isfinite(s2) or not np.isfinite(c2) or (s2 == 0 and c2 == 0):
                    v = float(np.nanmean(th_frame))
                    thetas[int(t)] = v if np.isfinite(v) else 0.0
                else:
                    thetas[int(t)] = 0.5 * np.arctan2(s2, c2)

        if n < int(num_frames):
            thetas[n:] = thetas[n - 1]
        print(f"    Loaded global nematic angles from {path.name} "
              f"(mean theta = {np.mean(thetas[:n]):.3f} rad)", flush=True)
        return thetas.astype(np.float32)
    except Exception as e:
        print(f"    [WARNING] Failed to load {path.name}: {e}. Falling back to flow nematic axis.",
              flush=True)
        return None


def global_nematic_theta_from_angles(phi: np.ndarray) -> float:
    """配向角 phi の集合から 2 テンソル平均による大域ネマチック主軸角を返す。"""
    s2 = float(np.mean(np.sin(2.0 * phi)))
    c2 = float(np.mean(np.cos(2.0 * phi)))
    if not np.isfinite(s2) or not np.isfinite(c2) or (s2 == 0.0 and c2 == 0.0):
        return 0.0
    return float(0.5 * np.arctan2(s2, c2))


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """角度を [-pi, pi) にラップする（+pi は同一方向の -pi として代表される）。"""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def fold_to_half_pi(angle: np.ndarray) -> np.ndarray:
    """ネマチック折り返し: 角度を mod pi で (-pi/2, pi/2] に折り返す。"""
    a = np.asarray(angle, dtype=np.float64)
    return (a + 0.5 * np.pi) % np.pi - 0.5 * np.pi


def make_angle_bins(bins: int, fold: bool = False) -> np.ndarray:
    """ヒストグラム用のビン境界を作る（fold=True で (-pi/2, pi/2]）。"""
    if fold:
        return np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(bins) + 1)
    return np.linspace(-np.pi, np.pi, int(bins) + 1)


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """ビン境界からビン中心を返す。"""
    edges = np.asarray(bin_edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])



def histogram_uniform(values: np.ndarray, bin_edges: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    一様ビンに対する高速ヒストグラム。

    np.histogram(values, bins=bin_edges, weights=weights) と等価な結果を
    searchsorted ベースの np.histogram より高速な np.bincount で計算する
    （ビン境界が等間隔であることを利用）。右端の境界値は np.histogram と同じく
    最後のビンに含める。
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = edges.size - 1
    lo = float(edges[0])
    hi = float(edges[-1])
    width = (hi - lo) / n_bins
    idx = np.floor((np.asarray(values, dtype=np.float64) - lo) / width).astype(np.int64)
    np.clip(idx, 0, n_bins - 1, out=idx)
    return np.bincount(idx, weights=weights, minlength=n_bins).astype(np.float64)


def delta_theta_histogram(
    mx: np.ndarray,
    my: np.ndarray,
    theta_nem: Optional[float],
    bin_edges: np.ndarray,
    bin_edges_fold: np.ndarray,
    min_flow_mag: float = 1e-4,
    valid_mask: Optional[np.ndarray] = None,
    weighting: str = 'none',
    align_tol_deg: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    1 フレーム（または任意の配列）のフローベクトル場から Delta theta = wrap_pi(phi - theta_nem)
    のヒストグラムと、円統計用の重み付き和を計算する。

    Parameters
    ----------
    mx, my : ndarray
        フローベクトルの x / y 成分（同一 shape）。
    theta_nem : float or None
        そのフレームの大域ネマチック主軸角 [rad]（向き無しの軸; 0 と pi は等価）。
        None を渡すと有効画素の配向角 phi から 2 テンソル平均で自動推定する
        （MTs_im_theta.zarr が無い場合のフローへのフォールバック）。
    bin_edges, bin_edges_fold : ndarray
        展開分布 [-pi, pi) とネマチック折り返し分布 (-pi/2, pi/2] のビン境界。
    min_flow_mag : float
        有効画素判定の閾値（|u| > min_flow_mag の画素のみ使用）。
    valid_mask : ndarray of bool, optional
        追加の有効画素マスク（貨物粒子近傍の除外など）。True = 使用する。
    weighting : {'none', 'magnitude'}
        'none' は等重み計数、'magnitude' は流速ノルムで重み付け。
    align_tol_deg : float
        平行 / 反平行とみなす許容角度（度）。

    Returns
    -------
    hist, hist_fold : ndarray
        展開分布 / ネマティック折り返し分布のヒストグラム（重み付き）。
    stats : dict
        wsum, n_pixels, c1, s1, c2, abs_sum, n_align, n_anti, theta_nem の重み付き和。
    """
    mx = np.asarray(mx, dtype=np.float32)
    my = np.asarray(my, dtype=np.float32)
    mag = np.hypot(mx, my)

    valid = mag > float(min_flow_mag)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    n_bins = len(bin_edges) - 1
    n_bins_fold = len(bin_edges_fold) - 1
    if not np.any(valid):
        return (np.zeros(n_bins, dtype=np.float64), np.zeros(n_bins_fold, dtype=np.float64),
                {'wsum': 0.0, 'n_pixels': 0.0, 'c1': 0.0, 's1': 0.0, 'c2': 0.0,
                 'abs_sum': 0.0, 'n_align': 0.0, 'n_anti': 0.0, 'theta_nem': np.nan})

    phi = np.arctan2(my[valid], mx[valid])
    theta_used = float(theta_nem) if theta_nem is not None else global_nematic_theta_from_angles(phi)
    dth = wrap_to_pi(phi - theta_used)

    w: Optional[np.ndarray] = None
    if str(weighting) == 'magnitude':
        w = mag[valid].astype(np.float64)

    hist = histogram_uniform(dth, bin_edges, weights=w)
    hist_fold = histogram_uniform(fold_to_half_pi(dth), bin_edges_fold, weights=w)

    ww = np.ones_like(dth, dtype=np.float64) if w is None else w
    cs = np.cos(dth)
    sn = np.sin(dth)
    wc = ww * cs
    ws = ww * sn
    abs_dth = np.abs(dth)
    tol = np.deg2rad(float(align_tol_deg))

    stats = {
        'wsum': float(ww.sum()),
        'n_pixels': float(dth.size),
        'c1': float(wc.sum()),
        's1': float(ws.sum()),
        # w * cos(2 dth) = w * (cos^2 - sin^2): 三角関数の再計算を避ける
        'c2': float((wc * cs - ws * sn).sum()),
        'abs_sum': float(np.dot(ww, abs_dth)),
        'n_align': float(np.dot(ww, abs_dth <= tol)),
        'n_anti': float(np.dot(ww, abs_dth >= np.pi - tol)),
        'theta_nem': float(theta_used),
    }
    return hist, hist_fold, stats


def delta_theta_variants(
    mx: np.ndarray,
    my: np.ndarray,
    theta_nem: Optional[float],
    bin_edges: np.ndarray,
    bin_edges_fold: np.ndarray,
    **kwargs,
) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]]:
    """
    theta_nem とそのミラー (-theta_nem) の両方を基準軸とした Delta theta 統計を計算する。

    MTs_im_theta.zarr の角度規約（libs/AFT_tools.py は least-moment 角を -1 倍する）は
    光学フローの座標系（x = 列, y = 行, phi = arctan2(u_y, u_x)）とミラー関係にある場合がある。
    そこで両方の基準軸を計算しておき、どちらが「Delta theta = 0 / pi にピークを持つか」
    （= フローとの整合性が高いか）を後段で判定できるようにする。

    Returns
    -------
    dict
        {'plus': (hist, hist_fold, stats), 'minus': (hist, hist_fold, stats) or None}
        theta_nem が None（フローから主軸を推定する場合）はミラー判定が無意味なため
        'minus' は None となる。
    """
    plus = delta_theta_histogram(mx, my, theta_nem, bin_edges, bin_edges_fold, **kwargs)
    if theta_nem is None:
        return {'plus': plus, 'minus': None}
    minus = delta_theta_histogram(mx, my, -float(theta_nem), bin_edges, bin_edges_fold, **kwargs)
    return {'plus': plus, 'minus': minus}


def resolve_flow_cache_name(
    flow_cache_dir: Optional[Union[str, Path]],
    root_dir: Path,
    exp_dir: Path,
    pixel_stride: int,
    frame_stride: int,
    user_name: Optional[str] = None,
) -> Optional[str]:
    """
    フローキャッシュのパスを決める。

    --flow_cache_dir 未指定なら None（= 実験ディレクトリ内の既定名を使う）。
    指定された場合は <flow_cache_dir>/<root_dir からの相対パス>/mt_flow_cache_s{st}_f{fs}.h5
    を返す（NAS が遅いときにローカルディスクへキャッシュするため）。
    """
    if flow_cache_dir is None or str(flow_cache_dir).strip() in ('', 'none', 'None'):
        return user_name
    base = Path(str(flow_cache_dir)).expanduser()
    try:
        rel = Path(exp_dir).resolve().relative_to(Path(root_dir).resolve())
    except Exception:
        rel = Path(Path(exp_dir).name)
    fname = user_name or f"mt_flow_cache_s{int(pixel_stride)}_f{int(frame_stride)}.h5"
    return str(base / rel / fname)


def theta_timeseries_stats(theta_series_rad: np.ndarray) -> Dict[str, float]:
    """
    フレームごとの大域ネマチック主軸角 theta_nem(t) の時間変化統計を返す。

    theta_nem は headless な軸（pi 周期）なので、2*theta を unwrap して連続化した
    代表値を用いて標準偏差・レンジ・ドリフトを評価する。

    Returns
    -------
    dict
        theta_nem_mean_deg            : 円周平均（2 テンソル平均）[deg]
        theta_nem_std_deg_continuous  : 連続化した軸角の標準偏差 [deg]（時間変動の大きさ）
        theta_nem_range_deg           : 連続化した軸角のレンジ（最大 - 最小）[deg]
        theta_nem_drift_deg           : 連続化した軸角の最初 - 最後の差 [deg]
        theta_nem_axis_order          : 軸としての時間的一貫性 |<e^{i 2 theta}>| (0-1)
        n_frames                      : 有効フレーム数
    """
    th = np.asarray(theta_series_rad, dtype=np.float64)
    th = th[np.isfinite(th)]
    if th.size == 0:
        return {
            'theta_nem_mean_deg': np.nan, 'theta_nem_std_deg_continuous': np.nan,
            'theta_nem_range_deg': np.nan, 'theta_nem_drift_deg': np.nan,
            'theta_nem_axis_order': np.nan, 'n_frames': 0,
        }

    cont = np.unwrap(2.0 * th) / 2.0          # pi 周期を考慮した連続化
    s2 = float(np.mean(np.sin(2.0 * th)))
    c2 = float(np.mean(np.cos(2.0 * th)))
    axis_order = float(np.hypot(s2, c2))
    return {
        'theta_nem_mean_deg': float(np.rad2deg(0.5 * np.arctan2(s2, c2))),
        'theta_nem_std_deg_continuous': float(np.rad2deg(np.std(cont))),
        'theta_nem_range_deg': float(np.rad2deg(np.ptp(cont))),
        'theta_nem_drift_deg': float(np.rad2deg(cont[-1] - cont[0])),
        'theta_nem_axis_order': axis_order,
        'n_frames': int(th.size),
    }


def theta_timeseries_table(
    results: Sequence[dict],
    frame_interval: float = 4.0,
) -> pd.DataFrame:
    """
    実験ごとの theta_nem(t) 時系列（連続化した軸角）を long 形式の表にする。

    columns: bead_name / exp_dir / sample_index / frame / time_s / theta_nem_rad /
             theta_nem_deg / theta_nem_deg_continuous / theta_nem_deg_relative
    （theta_nem_deg_continuous は pi 周期を考慮して連続化した値、
      theta_nem_deg_relative はその最初のフレームからの差）
    """
    rows = []
    for res in results:
        series = res.get('theta_series_rad')
        frames = res.get('theta_frames')
        if series is None or frames is None or len(series) == 0:
            continue
        series = np.asarray(series, dtype=np.float64)
        frames = np.asarray(frames, dtype=int)
        finite = np.isfinite(series)
        if not np.any(finite):
            continue
        th = series[finite]
        fr = frames[finite]
        cont = np.rad2deg(np.unwrap(2.0 * th) / 2.0)
        for i in range(th.size):
            rows.append({
                'bead_name': res['bead_name'],
                'exp_dir': res['exp_dir'],
                'sample_index': int(i),
                'frame': int(fr[i]),
                'time_s': float(fr[i]) * float(frame_interval),
                'theta_nem_rad': float(th[i]),
                'theta_nem_deg': float(np.rad2deg(th[i])),
                'theta_nem_deg_continuous': float(cont[i]),
                'theta_nem_deg_relative': float(cont[i] - cont[0]),
            })
    return pd.DataFrame(rows)


def summarized_circular_stats(stats: Dict[str, float]) -> Dict[str, float]:
    """重み付き和（delta_theta_histogram の出力）から円統計量を計算する。"""
    wsum = float(stats.get('wsum', 0.0))
    if wsum <= 0:
        return {
            'R': np.nan, 'mean_cos': np.nan, 'mean_sin': np.nan, 'S2': np.nan,
            'mean_abs_deg': np.nan, 'frac_aligned': np.nan, 'frac_anti': np.nan,
        }
    c1 = float(stats['c1']) / wsum
    s1 = float(stats['s1']) / wsum
    return {
        'R': float(np.hypot(c1, s1)),          # 平均合成ベクトル長（極性秩序）
        'mean_cos': c1,                        # <cos Delta theta>
        'mean_sin': s1,
        'S2': float(stats['c2']) / wsum,        # <cos 2 Delta theta>（ネマチック秩序）
        'mean_abs_deg': float(np.rad2deg(float(stats['abs_sum']) / wsum)),
        'frac_aligned': float(stats['n_align']) / wsum,   # |Delta theta| <= tol
        'frac_anti': float(stats['n_anti']) / wsum,       # |Delta theta| >= pi - tol
    }


def make_cargo_mask(
    xs: Optional[np.ndarray],
    ys: Optional[np.ndarray],
    shape: Tuple[int, int],
    stride: int,
    radius_px: float,
) -> np.ndarray:
    """
    フレーム内の貨物粒子位置（フル解像度 px 座標）から、間引き格子（stride）上での
    貨物近傍マスク（True = 除外）を作る。xs / ys が None の場合は全 False を返す。
    """
    mask = np.zeros(shape, dtype=bool)
    if xs is None or ys is None or len(xs) == 0 or radius_px <= 0:
        return mask

    rows, cols = int(shape[0]), int(shape[1])
    st = max(1, int(stride))
    r_str = float(radius_px) / st
    for x, y in zip(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)):
        cx = x / st
        cy = y / st
        x0 = max(int(np.floor(cx - r_str)), 0)
        x1 = min(int(np.ceil(cx + r_str)) + 1, cols)
        y0 = max(int(np.floor(cy - r_str)), 0)
        y1 = min(int(np.ceil(cy + r_str)) + 1, rows)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= ((yy - cy) ** 2 + (xx - cx) ** 2) <= r_str ** 2
    return mask


def cargo_positions_by_frame(df_tracks: Optional[pd.DataFrame]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """beads_tracks.csv から frame -> (xs, ys) の辞書を作る。"""
    positions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if df_tracks is None or df_tracks.empty:
        return positions
    if not {'frame', 'x', 'y'}.issubset(set(df_tracks.columns)):
        return positions
    for frame, grp in df_tracks.groupby('frame'):
        positions[int(frame)] = (grp['x'].to_numpy(dtype=float), grp['y'].to_numpy(dtype=float))
    return positions



# =========================================================================
# 実験ディレクトリごとの Delta theta 集計
# =========================================================================

class FlowFrameReader:
    """
    GFP_flows.h5（またはその間引きキャッシュ）からフレームを読み出す。

    実測されたコスト構造（NAS 上の実データ, 2160x2560 float16 2ch）:
      - h5py のチャンクは 1 フレーム = (1, 2, H, W) = 22 MB。ch0 と ch1 を別々に読むと同じ
        チャンクを 2 回読むことになり、コールド読み込みが 581 ms/frame -> 1 アクセスに
        統合すると 292 ms/frame（約 2 倍高速）。
      - コールド読み込みは NAS 帯域（実測 ~76 MB/s）律速で、pixel_stride を変えても
        読み込みバイト数は減らない（チャンク粒度のため）。ウォーム時は 35 ms/frame。
      - 複数実験の並列読みは NAS では逆効果（実測: 3 並列で合計 16.6 MB/s に低下）。
    そこで解析に使うフレームの間引き済みコピー (mx, my) を HDF5 にキャッシュし、
    同じ (pixel_stride, frame_stride) での再実行を ~60 倍高速化する
    （キャッシュは最初の解析パス中に書き込むため追加 I/O は無い）。

    Parameters
    ----------
    exp_dir : Path
        実験ディレクトリ。
    pixel_stride, frame_stride : int
        画素・フレームの間引き幅（キャッシュのキーにもなる）。
    max_frames_per_exp : int, optional
        （デバッグ用）先頭 N フレームのみ使用。
    flow_cache : {'auto', 'off', 'refresh'}
        'auto'    : 有効なキャッシュがあれば読み、無ければ解析パス中に作成する（既定）
        'off'     : キャッシュを使用しない
        'refresh' : 常にキャッシュを作り直す
    cache_name : str, optional
        キャッシュファイル名（既定 mt_flow_cache_s{st}_f{fs}.h5）。
    """

    VERSION = 1

    def __init__(self, exp_dir: Path, pixel_stride: int = 8, frame_stride: int = 5,
                 max_frames_per_exp: Optional[int] = None, flow_cache: str = 'auto',
                 cache_name: Optional[str] = None, flow_name: str = FLOW_NAME):
        self.exp_dir = Path(exp_dir)
        self.flow_path = self.exp_dir / flow_name
        self.pixel_stride = max(1, int(pixel_stride))
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames_per_exp = max_frames_per_exp
        self.flow_cache = str(flow_cache)
        cache_fname = cache_name or f"mt_flow_cache_s{self.pixel_stride}_f{self.frame_stride}.h5"
        self.cache_path = self.exp_dir / cache_fname
        try:
            if self.flow_cache != 'off':
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._flow_file = None
        self._cache_file = None
        self._write_cache = False
        self._dset_key = 'flows'
        self.source: Optional[str] = None
        self.channel_first = True
        self.dataset_shape: Optional[Tuple[int, ...]] = None
        self.n_frames_total = 0
        self.rows = 0
        self.cols = 0
        self.out_rows = 0
        self.out_cols = 0
        self.frame_ids: List[int] = []
        self.error: str = ''

    # --- キャッシュ整合性チェック -------------------------------------------------
    def _cache_attrs(self) -> Optional[dict]:
        try:
            with h5py.File(str(self.cache_path), 'r') as f:
                return dict(f.attrs)
        except Exception:
            return None

    def _cache_is_valid(self, attrs: Optional[dict], source_size: Optional[int]) -> bool:
        if not attrs:
            return False
        try:
            if int(attrs.get('version', -1)) != self.VERSION:
                return False
            if int(attrs.get('pixel_stride', -1)) != self.pixel_stride:
                return False
            if int(attrs.get('frame_stride', -1)) != self.frame_stride:
                return False
            if int(attrs.get('max_frames_per_exp', 0)) != int(self.max_frames_per_exp or 0):
                return False
            if source_size is not None and int(attrs.get('source_size', -1)) != int(source_size):
                return False
            return True
        except Exception:
            return False

    def _source_size(self) -> Optional[int]:
        try:
            return int(self.flow_path.stat().st_size)
        except Exception:
            return None

    # --- ライフサイクル ----------------------------------------------------------
    def __enter__(self) -> 'FlowFrameReader':
        try:
            self._enter()
        except Exception as e:                      # ファイル欠損 / 破損 / 形状不整合など
            self.error = str(e)
            self.source = None
            for f in (self._flow_file, self._cache_file):
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass
            self._flow_file = None
            self._cache_file = None
        return self

    def _enter(self) -> None:
        flow_exists = self.flow_path.exists()
        source_size = self._source_size() if flow_exists else None
        attrs = self._cache_attrs() if self.flow_cache != 'off' else None
        use_cache = self._cache_is_valid(attrs, source_size)

        if self.flow_cache == 'off' or (not use_cache and not flow_exists):
            if not flow_exists:
                raise FileNotFoundError(f"missing {self.flow_path} (and no valid flow cache)")
            self._open_flow()
            self.source = 'flow'
            return

        if use_cache and self.flow_cache != 'refresh':
            self._open_cache(attrs)
            self.source = 'cache'
            return

        # auto / refresh: フローを読みながらキャッシュも書き出す（追加 I/O なし）
        self._open_flow()
        self._create_cache()
        self.source = 'flow'

    def __exit__(self, exc_type, exc, tb) -> None:
        for f in (self._flow_file, self._cache_file):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._flow_file = None
        self._cache_file = None

    # --- オープン処理 ------------------------------------------------------------
    def _open_flow(self) -> None:
        self._flow_file = h5py.File(str(self.flow_path), 'r')
        key = 'flows' if 'flows' in self._flow_file else list(self._flow_file.keys())[0]
        self._dset_key = key
        shape = tuple(int(s) for s in self._flow_file[key].shape)
        if len(shape) != 4:
            raise ValueError(f"unsupported flow shape {shape} in {self.flow_path}")
        self.dataset_shape = shape
        self.channel_first = (shape[1] == 2)
        if not self.channel_first and shape[3] != 2:
            self.channel_first = True
        self.n_frames_total = shape[0]
        if self.channel_first:
            self.rows, self.cols = shape[2], shape[3]
        else:
            self.rows, self.cols = shape[1], shape[2]
        self._build_frame_ids()

    def _build_frame_ids(self) -> None:
        ids = list(range(0, self.n_frames_total, self.frame_stride))
        if self.max_frames_per_exp is not None and int(self.max_frames_per_exp) > 0:
            ids = ids[:int(self.max_frames_per_exp)]
        self.frame_ids = ids
        self.out_rows = len(range(0, self.rows, self.pixel_stride))
        self.out_cols = len(range(0, self.cols, self.pixel_stride))

    def _open_cache(self, attrs: dict) -> None:
        self._cache_file = h5py.File(str(self.cache_path), 'r')
        self.n_frames_total = int(attrs.get('n_frames_total', 0))
        self.channel_first = bool(int(attrs.get('channel_first', 1)))
        self.dataset_shape = tuple(int(s) for s in attrs.get('dataset_shape', ()))
        self.frame_ids = [int(v) for v in attrs.get('frame_ids', [])]
        self.rows = int(attrs.get('rows', 0))
        self.cols = int(attrs.get('cols', 0))
        self.out_rows = int(attrs.get('out_rows', 0))
        self.out_cols = int(attrs.get('out_cols', 0))

    def _create_cache(self) -> None:
        n = len(self.frame_ids)
        self._cache_file = h5py.File(str(self.cache_path), 'w')
        for name in ('mx', 'my'):
            self._cache_file.create_dataset(
                name, shape=(n, self.out_rows, self.out_cols), dtype=np.float16,
                chunks=(1, self.out_rows, self.out_cols), compression='lzf')
        a = self._cache_file.attrs
        a['version'] = self.VERSION
        a['pixel_stride'] = self.pixel_stride
        a['frame_stride'] = self.frame_stride
        a['max_frames_per_exp'] = int(self.max_frames_per_exp or 0)
        a['n_frames_total'] = int(self.n_frames_total)
        a['channel_first'] = int(self.channel_first)
        a['dataset_shape'] = [int(s) for s in (self.dataset_shape or ())]
        a['frame_ids'] = [int(v) for v in self.frame_ids]
        a['rows'] = int(self.rows)
        a['cols'] = int(self.cols)
        a['out_rows'] = int(self.out_rows)
        a['out_cols'] = int(self.out_cols)
        a['source_name'] = self.flow_path.name
        a['source_size'] = int(self._source_size() or -1)
        a['source_shape'] = [int(s) for s in (self.dataset_shape or ())]
        self._write_cache = True

    # --- フレーム取得 ------------------------------------------------------------
    def get(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """frame_ids[index] に対応する (mx, my) を返す（キャッシュ / フローどちらでも）。"""
        if self.source == 'cache':
            return (np.asarray(self._cache_file['mx'][index], dtype=np.float32),
                    np.asarray(self._cache_file['my'][index], dtype=np.float32))

        t = self.frame_ids[index]
        dset = self._flow_file[self._dset_key]
        st = self.pixel_stride
        # 1 アクセスで 2 チャンネルまとめて読む（チャンク = 1 フレームの二重読みを回避）
        if self.channel_first:
            arr = np.asarray(dset[t, :, ::st, ::st], dtype=np.float32)
            mx, my = arr[0], arr[1]
        else:
            arr = np.asarray(dset[t, ::st, ::st, :], dtype=np.float32)
            mx, my = arr[..., 0], arr[..., 1]

        if self._write_cache:
            self._cache_file['mx'][index] = mx.astype(np.float16)
            self._cache_file['my'][index] = my.astype(np.float16)
        return mx, my


def process_experiment(
    exp_dir: Path,
    bead: dict,
    bin_edges: np.ndarray,
    bin_edges_fold: np.ndarray,
    pixel_stride: int = 8,
    frame_stride: int = 5,
    max_frames_per_exp: Optional[int] = None,
    min_flow_mag: float = 1e-4,
    weighting: str = 'none',
    align_tol_deg: float = 30.0,
    mask_radius_factor: float = 0.0,
    min_mask_radius_px: float = 0.0,
    scale: float = 0.11,
    progress: bool = True,
    flow_cache: str = 'auto',
    flow_cache_name: Optional[str] = None,
) -> Optional[dict]:
    """
    1 つの実験ディレクトリの GFP_flows.h5 を読み、フレームごとに
    Delta theta = wrap_pi(phi - theta_nem) のヒストグラムと重み付き和を累積して返す。

    フレーム読み出しは FlowFrameReader（1 アクセス 2ch 読み + 間引きキャッシュ）に委譲し、
    読み込んだフレームについてのみ valid mask -> arctan2 -> 2 符号分のビン集計（bincount）
    を行う単一パス O(使用フレーム数 x 使用画素数) のアルゴリズムである。

    Returns
    -------
    dict or None
        hist / hist_fold（重み付き累積）、stats（重み付き和）、n_frames_used, n_frames_total,
        dataset_shape, channel_first, theta_source, theta_sign, flow_cache_source,
        mask_radius_px, weight_sum, n_pixels を含む辞書。加えて基準軸のミラー候補として
        hist_minus / hist_fold_minus / stats_minus と S2_plus / S2_minus を保持する
        （--theta_sign auto の判定に使用）。フローも有効なキャッシュも無い場合は None。
    """
    exp_dir = Path(exp_dir)
    st = max(1, int(pixel_stride))
    fs = max(1, int(frame_stride))
    n_bins = len(bin_edges) - 1
    n_bins_fold = len(bin_edges_fold) - 1

    # 貨物粒子近傍マスク（--mask_radius_factor / --min_mask_radius_px が 0 なら無効）
    mask_radius_px = float(max(float(min_mask_radius_px),
                               float(mask_radius_factor) * float(bead.get('radius_um', 0.0)) / float(scale)))
    cargo_pos: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if mask_radius_px > 0:
        tracks_path = exp_dir / TRACKS_NAME
        if tracks_path.exists():
            try:
                df_tracks = pd.read_csv(tracks_path)
                cargo_pos = cargo_positions_by_frame(df_tracks)
            except Exception as e:
                print(f"    [WARNING] could not read {tracks_path}: {e}", flush=True)
        else:
            print(f"    [WARNING] {TRACKS_NAME} not found; cargo masking disabled for {exp_dir.name}",
                  flush=True)

    def _new_acc() -> dict:
        return {
            'hist': np.zeros(n_bins, dtype=np.float64),
            'hist_fold': np.zeros(n_bins_fold, dtype=np.float64),
            'stats': {'wsum': 0.0, 'n_pixels': 0.0, 'c1': 0.0, 's1': 0.0, 'c2': 0.0,
                      'abs_sum': 0.0, 'n_align': 0.0, 'n_anti': 0.0},
        }

    hist = np.zeros(n_bins, dtype=np.float64)
    hist_fold = np.zeros(n_bins_fold, dtype=np.float64)
    stats = {'wsum': 0.0, 'n_pixels': 0.0, 'c1': 0.0, 's1': 0.0, 'c2': 0.0,
             'abs_sum': 0.0, 'n_align': 0.0, 'n_anti': 0.0}
    acc_minus: Optional[dict] = None

    try:
        reader = FlowFrameReader(exp_dir, pixel_stride=st, frame_stride=fs,
                                 max_frames_per_exp=max_frames_per_exp,
                                 flow_cache=flow_cache, cache_name=flow_cache_name)
    except Exception as e:
        print(f"    [WARNING] cannot open flow for {exp_dir.name}: {e}", flush=True)
        return None

    with reader as src:
        if src.source is None:
            print(f"    [WARNING] cannot read flow for {exp_dir.name}: {src.error}", flush=True)
            return None
        shape = src.dataset_shape
        if shape is None or len(shape) != 4:
            print(f"    [WARNING] unsupported flow shape {shape} in {exp_dir}", flush=True)
            return None
        channel_first = src.channel_first
        n_frames_total = src.n_frames_total
        frame_ids = list(src.frame_ids)

        # 大域ネマチック主軸角 theta_nem(t)（MTs_im_theta.zarr 優先）
        # None の場合はフレームごとにフロー配向から 2 テンソル平均で推定する
        thetas = load_nematic_directors(exp_dir, frame_ids, n_frames_total)
        theta_source = "MTs_im_theta.zarr" if thetas is not None else "flow"
        theta_values: List[float] = []
        theta_series = np.full(len(frame_ids), np.nan, dtype=np.float64)

        for i, t in enumerate(tqdm(frame_ids, desc=f"  {bead['name']}/{exp_dir.name}",
                                   leave=False, disable=not progress)):
            # 1 アクセスで 2 チャンネル読み出し（チャンク二重読み回避、キャッシュ時はその読み出し）
            mx, my = src.get(i)

            valid_mask = None
            if cargo_pos:
                pos = cargo_pos.get(t)
                if pos is not None:
                    valid_mask = ~make_cargo_mask(pos[0], pos[1], mx.shape, st, mask_radius_px)


            theta_t = float(thetas[t]) if thetas is not None else None
            variants = delta_theta_variants(
                mx, my, theta_t, bin_edges, bin_edges_fold,
                min_flow_mag=min_flow_mag, valid_mask=valid_mask,
                weighting=weighting, align_tol_deg=align_tol_deg,
            )
            h, hf, s = variants['plus']
            hist += h
            hist_fold += hf
            for k in stats:
                stats[k] += s[k]

            v_minus = variants['minus']
            if v_minus is not None:
                if acc_minus is None:
                    acc_minus = _new_acc()
                hm, hfm, sm = v_minus
                acc_minus['hist'] += hm
                acc_minus['hist_fold'] += hfm
                for k in acc_minus['stats']:
                    acc_minus['stats'][k] += sm[k]

            if np.isfinite(s.get('theta_nem', np.nan)):
                theta_values.append(float(s['theta_nem']))
                theta_series[i] = float(s['theta_nem'])

    theta_mean = float(np.mean(theta_values)) if theta_values else np.nan

    # 符号ごとのプール ネマチック秩序 <cos 2 Delta theta>（基準軸の妥当性判定に使用）
    s2_plus = (stats['c2'] / stats['wsum']) if stats['wsum'] > 0 else np.nan
    s2_minus = (acc_minus['stats']['c2'] / acc_minus['stats']['wsum']
                if (acc_minus is not None and acc_minus['stats']['wsum'] > 0) else np.nan)

    def _base_ret() -> dict:
        return {
            'exp_dir': str(exp_dir),
            'bead_name': bead['name'],
            'hist': hist,
            'hist_fold': hist_fold,
            'stats': stats,
            'n_frames_total': n_frames_total,
            'dataset_shape': shape,
            'channel_first': channel_first,
            'theta_source': theta_source,
            'flow_cache_source': src.source,
            'flow_cache_path': str(src.cache_path) if src.source == 'cache' else '',
            'mask_radius_px': mask_radius_px,
            'weight_sum': float(stats['wsum']),
            'n_pixels': int(stats['n_pixels']),
            'theta_mean_rad': theta_mean,
            'theta_series_rad': theta_series,
            'theta_frames': np.asarray(frame_ids, dtype=int),
            # ミラー符号候補（--theta_sign auto の判定材料）
            'hist_minus': acc_minus['hist'] if acc_minus is not None else None,
            'hist_fold_minus': acc_minus['hist_fold'] if acc_minus is not None else None,
            'stats_minus': acc_minus['stats'] if acc_minus is not None else None,
            'S2_plus': float(s2_plus) if np.isfinite(s2_plus) else np.nan,
            'S2_minus': float(s2_minus) if np.isfinite(s2_minus) else np.nan,
            'theta_sign': 1,
        }

    if stats['wsum'] <= 0:
        ret = _base_ret()
        ret['n_frames_used'] = 0
        ret['n_pixels'] = 0
        ret['weight_sum'] = 0.0
        return ret

    ret = _base_ret()
    ret['n_frames_used'] = len(frame_ids)
    return ret


def choose_theta_sign(
    results: Sequence[dict],
    theta_sign: str = 'auto',
    margin: float = 0.05,
) -> Tuple[int, Dict[str, float]]:
    """
    基準軸の符号（+1 / -1）を決める。

    '+1' / '-1' が指定された場合はその符号をそのまま採用する。'auto' の場合は
    全実験をプールしたネマチック秩序 <cos 2 Delta theta> が大きい方（= Delta theta 分布が
    0 / pi に鋭くピークを持つ方）を採用する。MTs_im_theta.zarr の角度規約が光学フローの
    座標系とミラー関係にあるデータセットでは -1 が選ばれる。
    差が margin 未満の場合は規約の曖昧さが無いとみなして +1 を維持する。

    Returns
    -------
    sign : int
        +1 または -1。
    info : dict
        s2_plus / s2_minus / n_experiments_with_mirror / decision。
    """
    s2_plus = float(np.nansum([r.get('stats', {}).get('c2', 0.0) for r in results]))
    w_plus = float(np.nansum([r.get('stats', {}).get('wsum', 0.0) for r in results]))
    s2_minus = float(np.nansum([r['stats_minus']['c2'] for r in results if r.get('stats_minus')]))
    w_minus = float(np.nansum([r['stats_minus']['wsum'] for r in results if r.get('stats_minus')]))
    val_plus = (s2_plus / w_plus) if w_plus > 0 else np.nan
    val_minus = (s2_minus / w_minus) if w_minus > 0 else np.nan
    n_mirror = int(sum(1 for r in results if r.get('stats_minus')))

    info = {
        'S2_pooled_plus': val_plus,
        'S2_pooled_minus': val_minus,
        'n_experiments_with_mirror': n_mirror,
    }

    forced = str(theta_sign).strip()
    if forced in ('+1', '1'):
        info['decision'] = 'forced +1'
        return 1, info
    if forced == '-1':
        info['decision'] = 'forced -1'
        return -1, info

    if n_mirror == 0 or not np.isfinite(val_minus):
        info['decision'] = 'auto -> +1 (no mirror candidate; theta from flow)'
        return 1, info
    if val_minus - val_plus > float(margin):
        info['decision'] = f'auto -> -1 (mirror corrected: S2 {val_minus:.3f} > {val_plus:.3f})'
        return -1, info
    info['decision'] = f'auto -> +1 (S2 +1: {val_plus:.3f} vs -1: {val_minus:.3f})'
    return 1, info


def select_sign_variant(result: dict, sign: int) -> dict:
    """指定した符号の基準軸に対応するヒストグラム / 統計を主要フィールドとして持つ辞書を返す。"""
    out = dict(result)
    if int(sign) < 0 and result.get('stats_minus') is not None:
        out['hist'] = result['hist_minus']
        out['hist_fold'] = result['hist_fold_minus']
        out['stats'] = result['stats_minus']
        out['weight_sum'] = float(result['stats_minus']['wsum'])
        out['n_pixels'] = int(result['stats_minus']['n_pixels'])
        out['theta_sign'] = -1
        # Delta theta の基準として実際に使った軸（-theta_nem）を時系列にも反映する
        series = result.get('theta_series_rad')
        if series is not None:
            out['theta_series_rad'] = -np.asarray(series, dtype=np.float64)
        mean_rad = result.get('theta_mean_rad', np.nan)
        if np.isfinite(mean_rad):
            out['theta_mean_rad'] = -float(mean_rad)
    else:
        # フローから主軸を推定した場合は符号の概念が無いため +1 として記録する
        out['theta_sign'] = 1
    return out



# =========================================================================
# 集計（実験 -> 条件）
# =========================================================================

def normalize_density(hist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """ヒストグラムを確率密度（積分 = 1, 単位 rad^-1）に規格化する。"""
    hist = np.asarray(hist, dtype=np.float64)
    total = float(np.sum(hist))
    if total <= 0:
        return np.full(hist.shape, np.nan, dtype=np.float64)
    widths = np.diff(np.asarray(bin_edges, dtype=np.float64))
    return hist / total / widths


def sem_or_nan(values: np.ndarray) -> float:
    """標準誤差（サンプル数 2 未満なら NaN）。"""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float('nan')
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def per_experiment_table(results: Sequence[dict], align_tol_deg: float = 30.0) -> pd.DataFrame:
    """実験ごとの円統計サマリー表を作る。"""
    rows = []
    for res in results:
        cs = summarized_circular_stats(res['stats'])
        ts = theta_timeseries_stats(res.get('theta_series_rad', np.array([])))
        rows.append({
            'bead_name': res['bead_name'],
            'exp_dir': res['exp_dir'],
            'theta_source': res['theta_source'],
            'flow_cache_source': res.get('flow_cache_source', ''),
            'theta_sign': res.get('theta_sign', 1),
            'mask_radius_px': res['mask_radius_px'],
            'n_frames_used': res['n_frames_used'],
            'n_frames_total': res['n_frames_total'],
            'n_pixels': res['n_pixels'],
            'weight_sum': res['weight_sum'],
            'theta_mean_rad': res.get('theta_mean_rad', np.nan),
            'theta_nem_mean_deg': ts['theta_nem_mean_deg'],
            'theta_nem_std_deg_continuous': ts['theta_nem_std_deg_continuous'],
            'theta_nem_range_deg': ts['theta_nem_range_deg'],
            'theta_nem_drift_deg': ts['theta_nem_drift_deg'],
            'theta_nem_axis_order': ts['theta_nem_axis_order'],
            'mean_resultant_length_R': cs['R'],
            'mean_cos_delta_theta': cs['mean_cos'],
            'nematic_order_cos2': cs['S2'],
            'mean_abs_delta_theta_deg': cs['mean_abs_deg'],
            'frac_parallel_within_tol': cs['frac_aligned'],
            'frac_antiparallel_within_tol': cs['frac_anti'],
            'align_tol_deg': align_tol_deg,
            'nematic_order_cos2_signplus': res.get('S2_plus', np.nan),
            'nematic_order_cos2_signminus': res.get('S2_minus', np.nan),
        })
    return pd.DataFrame(rows)



def condition_histogram_table(
    results: Sequence[dict],
    target_beads: Sequence[dict],
    bin_edges: np.ndarray,
    bin_edges_fold: np.ndarray,
    weighting: str = 'none',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    条件（粒子径）ごとに、実験ごとの規格化密度の平均 ± SEM とプール密度をまとめる。

    Returns
    -------
    df_curve, df_curve_fold : pd.DataFrame
        bead_name / diameter_um / angle_center_rad / angle_center_deg /
        mean_density / sem_density / pooled_density / n_experiments / n_samples
    """
    centers = bin_centers(bin_edges)
    centers_fold = bin_centers(bin_edges_fold)

    rows, rows_fold = [], []
    for bead in target_beads:
        sub = [r for r in results if r['bead_name'] == bead['name']]
        if not sub:
            continue

        dens = np.array([normalize_density(r['hist'], bin_edges) for r in sub], dtype=np.float64)
        dens_fold = np.array([normalize_density(r['hist_fold'], bin_edges_fold) for r in sub], dtype=np.float64)

        mean = np.nanmean(dens, axis=0)
        sem = np.array([sem_or_nan(dens[:, i]) for i in range(dens.shape[1])])
        pooled = normalize_density(np.sum([r['hist'] for r in sub], axis=0), bin_edges)
        mean_f = np.nanmean(dens_fold, axis=0)
        sem_f = np.array([sem_or_nan(dens_fold[:, i]) for i in range(dens_fold.shape[1])])
        pooled_fold = normalize_density(np.sum([r['hist_fold'] for r in sub], axis=0), bin_edges_fold)

        n_samples = int(np.sum([r['n_pixels'] for r in sub]))
        n_exp = len(sub)
        for i, c in enumerate(centers):
            rows.append({
                'bead_name': bead['name'],
                'diameter_um': bead['diameter_um'],
                'weighting': weighting,
                'angle_center_rad': float(c),
                'angle_center_deg': float(np.rad2deg(c)),
                'mean_density': float(mean[i]),
                'sem_density': float(sem[i]),
                'pooled_density': float(pooled[i]),
                'n_experiments': n_exp,
                'n_samples': n_samples,
            })
        for i, c in enumerate(centers_fold):
            rows_fold.append({
                'bead_name': bead['name'],
                'diameter_um': bead['diameter_um'],
                'weighting': weighting,
                'angle_center_rad': float(c),
                'angle_center_deg': float(np.rad2deg(c)),
                'mean_density': float(mean_f[i]),
                'sem_density': float(sem_f[i]),
                'pooled_density': float(pooled_fold[i]),
                'n_experiments': n_exp,
                'n_samples': n_samples,
            })

    return pd.DataFrame(rows), pd.DataFrame(rows_fold)



def condition_summary_table(
    results: Sequence[dict],
    target_beads: Sequence[dict],
    align_tol_deg: float = 30.0,
) -> pd.DataFrame:
    """条件ごとの代表値（円統計の実験間 mean ± SEM とプール値）をまとめる。

    プール値（*_pooled）はビン化誤差を避けるため、実験ごとの重み付き和を合算して
    直接計算した円統計量（全サンプルプール）である。
    """
    rows = []
    for bead in target_beads:
        sub = [r for r in results if r['bead_name'] == bead['name']]
        if not sub:
            continue
        cs = [summarized_circular_stats(r['stats']) for r in sub]
        ts_list = [theta_timeseries_stats(r.get('theta_series_rad', np.array([]))) for r in sub]

        # プール値はビン化誤差の無い「生の重み付き和」から直接計算する
        sum_wsum = float(np.sum([r['stats']['wsum'] for r in sub]))
        if sum_wsum > 0:
            s2_pooled = float(np.sum([r['stats']['c2'] for r in sub])) / sum_wsum
            abs_pooled = float(np.sum([r['stats']['abs_sum'] for r in sub])) / sum_wsum
            r1_pooled = float(np.hypot(np.sum([r['stats']['c1'] for r in sub]),
                                       np.sum([r['stats']['s1'] for r in sub]))) / sum_wsum
        else:
            s2_pooled, abs_pooled, r1_pooled = np.nan, np.nan, np.nan

        def _agg(key: str) -> Tuple[float, float]:
            vals = np.array([c[key] for c in cs], dtype=np.float64)
            return float(np.nanmean(vals)), sem_or_nan(vals)

        def _agg_ts(key: str) -> Tuple[float, float]:
            vals = np.array([t[key] for t in ts_list], dtype=np.float64)
            return float(np.nanmean(vals)), sem_or_nan(vals)

        rows.append({
            'bead_name': bead['name'],
            'diameter_um': bead['diameter_um'],
            'radius_um': bead.get('radius_um', np.nan),
            'n_experiments': len(sub),
            'n_frames_total': int(np.sum([r['n_frames_used'] for r in sub])),
            'n_samples': int(np.sum([r['n_pixels'] for r in sub])),
            'mask_radius_px_min': float(np.min([r['mask_radius_px'] for r in sub])),
            'mask_radius_px_max': float(np.max([r['mask_radius_px'] for r in sub])),
            'theta_sources': ','.join(sorted({r['theta_source'] for r in sub})),
            'theta_sign': int(sub[0].get('theta_sign', 1)),
            'R_mean': _agg('R')[0], 'R_sem': _agg('R')[1],
            'mean_cos_delta_theta_mean': _agg('mean_cos')[0],
            'mean_cos_delta_theta_sem': _agg('mean_cos')[1],
            'nematic_order_cos2_mean': _agg('S2')[0],
            'nematic_order_cos2_sem': _agg('S2')[1],
            'nematic_order_cos2_pooled': s2_pooled,
            'R_pooled': r1_pooled,
            'mean_abs_delta_theta_deg_mean': _agg('mean_abs_deg')[0],
            'mean_abs_delta_theta_deg_sem': _agg('mean_abs_deg')[1],
            'mean_abs_delta_theta_deg_pooled': float(np.rad2deg(abs_pooled)) if np.isfinite(abs_pooled) else np.nan,
            'frac_parallel_within_tol_mean': _agg('frac_aligned')[0],
            'frac_parallel_within_tol_sem': _agg('frac_aligned')[1],
            'frac_antiparallel_within_tol_mean': _agg('frac_anti')[0],
            'frac_antiparallel_within_tol_sem': _agg('frac_anti')[1],
            'align_tol_deg': align_tol_deg,
            'theta_nem_std_deg_continuous_mean': _agg_ts('theta_nem_std_deg_continuous')[0],
            'theta_nem_std_deg_continuous_sem': _agg_ts('theta_nem_std_deg_continuous')[1],
            'theta_nem_range_deg_mean': _agg_ts('theta_nem_range_deg')[0],
            'theta_nem_axis_order_mean': _agg_ts('theta_nem_axis_order')[0],
            'theta_nem_axis_order_sem': _agg_ts('theta_nem_axis_order')[1],
            'uniform_density_unfolded': 1.0 / (2.0 * np.pi),
            'uniform_density_folded': 1.0 / np.pi,
        })
    return pd.DataFrame(rows)



# =========================================================================
# 作図
# =========================================================================

def smooth_circular(y: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    """周期境界でガウシアン平滑化する（レーダーチャートの輪郭用）。"""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if sigma_bins is None or float(sigma_bins) <= 0 or n < 3:
        return y.copy()
    k = int(np.ceil(3.0 * float(sigma_bins)))
    k = min(k, max(1, (n - 1) // 2))
    x = np.arange(-k, k + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (x / float(sigma_bins)) ** 2)
    ker /= ker.sum()
    y_pad = np.concatenate([y[-k:], y, y[:k]])
    return np.convolve(y_pad, ker, mode='same')[k:k + n]


def _radial_order(df_curve: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    展開分布のビン中心を [0, 2 pi) に写像し、昇順に並べ替えた (theta, index) を返す。
    -pi と +pi のビンは円周上で pi の両隣に並ぶため、重複なく全周を覆う。
    """
    centers = np.asarray(df_curve['angle_center_rad'].values, dtype=float)
    theta = np.mod(centers, 2.0 * np.pi)
    order = np.argsort(theta)
    return theta[order], order


def _condition_title(bead: dict, row: Optional[pd.Series]) -> str:
    """パネルタイトル（直径 + ネマチック秩序 + 実験数）を作る。"""
    title = rf"$2R_c = {bead['diameter_um']:.2f}\,\mu\mathrm{{m}}$"
    if row is None:
        return title
    parts = []
    if np.isfinite(row.get('nematic_order_cos2_pooled', np.nan)):
        parts.append(rf"$\langle\cos 2\Delta\theta\rangle = {row['nematic_order_cos2_pooled']:.3f}$")
    if np.isfinite(row.get('n_experiments', np.nan)):
        parts.append(rf"$N_{{\mathrm{{exp}}}} = {int(row['n_experiments'])}$")
    if parts:
        title += "\n" + ", ".join(parts)
    return title


def plot_theta_timeseries(
    df_theta_series: pd.DataFrame,
    df_summary: pd.DataFrame,
    target_beads: Sequence[dict],
    out_dirs: List[Path],
    theta_note: str = '',
    ncols: int = 3,
) -> None:
    """
    各フレームで評価した大域ネマチック主軸角 theta_nem(t) の時間発展を条件別に描画する。

    縦軸は各実験の最初のフレームからの相対変化（degree, pi 周期を考慮して連続化）とし、
    実験間比較ができるようにする。theta_nem(t) は Delta theta の基準軸として
    フレームごとに用いられている（時間変化を導入済み）。
    """
    if df_theta_series.empty:
        print("[WARNING] theta_nem(t) 時系列データが空のためスキップします")
        return

    conds = [b for b in target_beads
             if not df_theta_series[df_theta_series['bead_name'] == b['name']].empty]
    if not conds:
        return

    ncols = int(min(ncols, len(conds)))
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows),
                             sharex=False, sharey=True, squeeze=False)
    axes = axes.ravel()

    for i, bead in enumerate(conds):
        ax = axes[i]
        sub = df_theta_series[df_theta_series['bead_name'] == bead['name']]
        color = bead.get('color', 'C0')
        for k, (exp_dir, grp) in enumerate(sub.groupby('exp_dir')):
            grp = grp.sort_values('sample_index')
            ax.plot(grp['time_s'] / 60.0, grp['theta_nem_deg_relative'],
                    color=color, lw=1.4, alpha=0.55,
                    label='experiments' if k == 0 else None)

        row = df_summary[df_summary['bead_name'] == bead['name']]
        title = rf"$2R_c = {bead['diameter_um']:.2f}\,\mu\mathrm{{m}}$"
        if not row.empty:
            title += (rf" ($\sigma_t = {row['theta_nem_std_deg_continuous_mean'].iloc[0]:.1f}"
                      rf" \pm {row['theta_nem_std_deg_continuous_sem'].iloc[0]:.1f}^\circ$, "
                      rf"$N_{{\mathrm{{exp}}}} = {int(row['n_experiments'].iloc[0])}$)")
        ax.set_title(title, fontsize=12.0)
        ax.axhline(0.0, color='#333333', ls='--', lw=1.0, alpha=0.6)
        ax.grid(True, ls='--', alpha=0.4)
        if i % ncols == 0:
            ax.set_ylabel(r"$\theta_{\mathrm{nem}}(t) - \theta_{\mathrm{nem}}(0)$ [deg]", fontsize=12.5)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"Time [min]", fontsize=12.5)
        if i == 0:
            ax.legend(fontsize=10.0, loc='upper left', framealpha=0.92)

    for j in range(len(conds), axes.size):
        axes[j].axis('off')

    fig.suptitle(r"Temporal Evolution of the Per-Frame Global Nematic Axis "
                 rf"$\theta_{{\mathrm{{nem}}}}(t)${theta_note}", fontsize=14.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_to_all(fig, 'mt_orientation_theta_time_series', out_dirs)
    plt.close(fig)


def theta_sign_note(theta_sign: int) -> str:
    """基準軸の符号補正の有無を示す注記文字列（図タイトル用）。"""
    return '' if int(theta_sign) > 0 else r" ($\theta_{\mathrm{nem}}$ mirror-corrected)"


def plot_histogram_panels(
    df_curve: pd.DataFrame,
    df_summary: pd.DataFrame,
    target_beads: Sequence[dict],
    out_dirs: List[Path],
    weighting: str = 'none',
    ncols: int = 3,
    theta_note: str = '',
) -> None:
    """条件別 Delta theta ヒストグラム（[-pi, pi)）の 2xN パネル図。"""
    conds = [b for b in target_beads if not df_curve[df_curve['bead_name'] == b['name']].empty]
    if not conds:
        print("[WARNING] ヒストグラム用データが空のためスキップします")
        return

    ncols = int(min(ncols, len(conds)))
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows),
                             sharex=True, sharey=False, squeeze=False)
    axes = axes.ravel()

    uniform = 1.0 / (2.0 * np.pi)
    for i, bead in enumerate(conds):
        ax = axes[i]
        sub = df_curve[df_curve['bead_name'] == bead['name']].sort_values('angle_center_rad')
        centers = sub['angle_center_rad'].to_numpy(dtype=float)
        mean = sub['mean_density'].to_numpy(dtype=float)
        sem = np.nan_to_num(sub['sem_density'].to_numpy(dtype=float), nan=0.0)
        color = bead.get('color', 'C0')

        ax.fill_between(centers, 0.0, mean, step='mid', color=color, alpha=0.25, lw=0)
        ax.fill_between(centers, np.clip(mean - sem, 0, None), mean + sem, step='mid',
                        color=color, alpha=0.45, lw=0)
        ax.plot(centers, mean, color=color, lw=2.2, zorder=3)
        ax.axhline(uniform, color='gray', ls=':', lw=1.4, zorder=2,
                   label=r"Uniform $1/2\pi$" if i == 0 else None)
        for xv in (-np.pi, 0.0, np.pi):
            ax.axvline(xv, color='#333333', ls='--', lw=1.0, alpha=0.6, zorder=1)

        row = df_summary[df_summary['bead_name'] == bead['name']]
        ax.set_title(_condition_title(bead, row.iloc[0] if not row.empty else None), fontsize=12.5)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"], fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)

        if i == 0:
            ax.annotate(r"aligned", xy=(0.0, mean.max() if mean.size else 1.0),
                        xytext=(0.28, mean.max() * 0.98 if mean.size else 0.9),
                        fontsize=10.5, color='#333333')
            ax.annotate(r"anti-aligned", xy=(np.pi, mean.max() * 0.98 if mean.size else 0.9),
                        xytext=(-np.pi * 0.55, mean.max() * 0.98 if mean.size else 0.9),
                        fontsize=10.5, color='#333333')
            ax.legend(fontsize=10.0, loc='upper center', framealpha=0.92)
        if i % ncols == 0:
            ax.set_ylabel(r"$P(\Delta\theta)$ [$\mathrm{rad}^{-1}$]", fontsize=13)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\Delta\theta = \phi - \theta_{\mathrm{nem}}$ [rad]", fontsize=13)

    for j in range(len(conds), axes.size):
        axes[j].axis('off')

    note = " (counts)" if str(weighting) == 'none' else " (weighted by $|u|$)"
    fig.suptitle(r"MT Flow Orientation Relative to the Nematic Axis: "
                 rf"$P(\Delta\theta)$ Histogram{note}{theta_note}", fontsize=14.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_to_all(fig, 'mt_orientation_histogram_panels', out_dirs)
    plt.close(fig)



def plot_radar_panels(
    df_curve: pd.DataFrame,
    df_summary: pd.DataFrame,
    target_beads: Sequence[dict],
    out_dirs: List[Path],
    weight_magnitude: bool = False,
    smooth_sigma_bins: float = 1.0,
    ncols: int = 3,
    theta_note: str = '',
) -> None:
    """条件別レーダーチャート（極座標ローズ図）。0 と pi に 2 つのピークが現れる。"""
    conds = [b for b in target_beads if not df_curve[df_curve['bead_name'] == b['name']].empty]
    if not conds:
        print("[WARNING] レーダーチャート用データが空のためスキップします")
        return

    ncols = int(min(ncols, len(conds)))
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 4.9 * nrows),
                             subplot_kw=dict(polar=True), squeeze=False)
    axes = axes.ravel()

    uniform = 1.0 / (2.0 * np.pi)
    th_circ = np.linspace(0.0, 2.0 * np.pi, 361)
    bandwidth = None

    for i, bead in enumerate(conds):
        ax = axes[i]
        sub = df_curve[df_curve['bead_name'] == bead['name']]
        theta_sorted, order = _radial_order(sub)
        centers_sorted = np.asarray(sub['angle_center_rad'].values, dtype=float)[order]
        mean_sorted = np.asarray(sub['mean_density'].values, dtype=float)[order]
        color = bead.get('color', 'C0')

        if bandwidth is None and centers_sorted.size > 1:
            bandwidth = float(np.diff(np.sort(centers_sorted)).max())

        ax.bar(theta_sorted, mean_sorted, width=bandwidth, color=color, alpha=0.35,
               edgecolor=color, linewidth=0.7, zorder=1)
        sm = smooth_circular(mean_sorted, smooth_sigma_bins)
        th_closed = np.concatenate([theta_sorted, [theta_sorted[0] + 2.0 * np.pi]])
        sm_closed = np.concatenate([sm, [sm[0]]])
        ax.plot(th_closed, sm_closed, color=color, lw=2.4, zorder=3,
                label=r"smoothed $P(\Delta\theta)$" if i == 0 else None)
        ax.plot(th_circ, np.full_like(th_circ, uniform), color='gray', ls=':', lw=1.3, zorder=2,
                label=r"Uniform $1/2\pi$" if i == 0 else None)

        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_xticks([0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2])
        ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"], fontsize=12)
        ax.set_title(_condition_title(bead, df_summary[df_summary['bead_name'] == bead['name']].iloc[0]
                                      if not df_summary[df_summary['bead_name'] == bead['name']].empty else None),
                     fontsize=12.0, pad=16)
        ax.grid(True, ls='--', alpha=0.45)
        if i == 0:
            ax.legend(loc='lower left', bbox_to_anchor=(-0.22, -0.18), fontsize=10.0,
                      frameon=True, framealpha=0.92)

    for j in range(len(conds), axes.size):
        axes[j].axis('off')

    wnote = r" (weighted by $|u|$)" if weight_magnitude else ""
    fig.suptitle(r"MT Flow Orientation Radar Chart Relative to the Nematic Axis"
                 rf"{wnote}{theta_note}: Two Lobes at $\Delta\theta = 0$ and $\pi$",
                 fontsize=14.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure_to_all(fig, 'mt_orientation_radar_panels', out_dirs)
    plt.close(fig)


def plot_radar_overlay(
    df_curve: pd.DataFrame,
    target_beads: Sequence[dict],
    out_dirs: List[Path],
    weight_magnitude: bool = False,
    smooth_sigma_bins: float = 1.0,
    theta_note: str = '',
) -> None:
    """全条件を重ね書きしたレーダーチャート（1 枚の極座標プロット）。"""
    conds = [b for b in target_beads if not df_curve[df_curve['bead_name'] == b['name']].empty]
    if not conds:
        return

    fig, ax = plt.subplots(figsize=(9.0, 8.2), subplot_kw=dict(polar=True))
    uniform = 1.0 / (2.0 * np.pi)
    th_circ = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(th_circ, np.full_like(th_circ, uniform), color='gray', ls=':', lw=1.5,
            zorder=2, label=r"Uniform $1/2\pi$")

    bandwidth = None
    curves = []
    for bead in conds:
        sub = df_curve[df_curve['bead_name'] == bead['name']]
        theta_sorted, order = _radial_order(sub)
        mean_sorted = np.asarray(sub['mean_density'].values, dtype=float)[order]
        if bandwidth is None and theta_sorted.size > 1:
            bandwidth = float(np.diff(np.sort(theta_sorted)).max())
        sm = smooth_circular(mean_sorted, smooth_sigma_bins)
        th_closed = np.concatenate([theta_sorted, [theta_sorted[0] + 2.0 * np.pi]])
        sm_closed = np.concatenate([sm, [sm[0]]])
        curves.append((bead, th_closed, sm_closed))

    for bead, th_closed, sm_closed in curves:
        color = bead.get('color', 'C0')
        ax.plot(th_closed, sm_closed, color=color, lw=2.0,
                label=rf"$2R_c = {bead['diameter_um']:.2f}\,\mu\mathrm{{m}}$")
        ax.fill(th_closed, np.clip(sm_closed, 0, None), color=color, alpha=0.12, lw=0)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks([0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"], fontsize=13)
    ax.grid(True, ls='--', alpha=0.45)
    ax.set_title(r"MT Flow Orientation Relative to the Nematic Axis: "
                 rf"Cargo-Size Comparison ($\Delta\theta = 0, \pi$ peaks){theta_note}",
                 fontsize=14.0, fontweight='bold', pad=22)
    ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fontsize=11.5,
              frameon=True, framealpha=0.93)
    fig.tight_layout()
    save_figure_to_all(fig, 'mt_orientation_radar_overlay', out_dirs)
    plt.close(fig)



def plot_nematic_folded(
    df_curve_fold: pd.DataFrame,
    df_summary: pd.DataFrame,
    target_beads: Sequence[dict],
    out_dirs: List[Path],
    weight_magnitude: bool = False,
    theta_note: str = '',
) -> None:
    """ネマチック折り返し分布 (Delta theta mod pi, -pi/2 から pi/2) の全条件比較図。"""
    conds = [b for b in target_beads if not df_curve_fold[df_curve_fold['bead_name'] == b['name']].empty]
    if not conds:
        return

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    uniform = 1.0 / np.pi
    ax.axhline(uniform, color='gray', ls=':', lw=1.5, label=r"Uniform $1/\pi$")

    for bead in conds:
        sub = df_curve_fold[df_curve_fold['bead_name'] == bead['name']].sort_values('angle_center_rad')
        centers = sub['angle_center_rad'].to_numpy(dtype=float)
        mean = sub['mean_density'].to_numpy(dtype=float)
        sem = np.nan_to_num(sub['sem_density'].to_numpy(dtype=float), nan=0.0)
        color = bead.get('color', 'C0')
        row = df_summary[df_summary['bead_name'] == bead['name']]
        s2 = row['nematic_order_cos2_pooled'].iloc[0] if not row.empty else np.nan
        lbl = rf"$2R_c = {bead['diameter_um']:.2f}\,\mu\mathrm{{m}}$"
        if np.isfinite(s2):
            lbl += rf" ($\langle\cos 2\Delta\theta\rangle = {s2:.3f}$)"
        ax.plot(centers, mean, color=color, lw=2.4, label=lbl)
        ax.fill_between(centers, np.clip(mean - sem, 0, None), mean + sem,
                        color=color, alpha=0.18, lw=0)

    ax.axvline(0.0, color='#333333', ls='--', lw=1.2)
    ax.set_xlim(-0.5 * np.pi, 0.5 * np.pi)
    ax.set_xticks([-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2])
    ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"], fontsize=13)
    ax.set_xlabel(r"$\Delta\theta\ \mathrm{mod}\ \pi$ relative to nematic axis [rad]", fontsize=14)
    ax.set_ylabel(r"$P(\Delta\theta)$ [$\mathrm{rad}^{-1}$]", fontsize=14)
    wnote = r" (weighted by $|u|$)" if weight_magnitude else ""
    ax.set_title(rf"Nematic-Folded MT Flow Orientation Distribution{wnote}{theta_note}",
                 fontsize=14.5, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=10.5, loc='upper right', framealpha=0.93)
    fig.tight_layout()
    save_figure_to_all(fig, 'mt_orientation_nematic_folded', out_dirs)
    plt.close(fig)


def plot_histogram_overlay(
    df_curve: pd.DataFrame,
    target_beads: Sequence[dict],
    out_dirs: List[Path],
    weight_magnitude: bool = False,
    theta_note: str = '',
) -> None:
    """全条件を重ね書きした Delta theta ヒストグラム（1 枚の線形プロット）。"""
    conds = [b for b in target_beads if not df_curve[df_curve['bead_name'] == b['name']].empty]
    if not conds:
        return

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.axhline(1.0 / (2.0 * np.pi), color='gray', ls=':', lw=1.5, label=r"Uniform $1/2\pi$")
    for bead in conds:
        sub = df_curve[df_curve['bead_name'] == bead['name']].sort_values('angle_center_rad')
        centers = sub['angle_center_rad'].to_numpy(dtype=float)
        mean = sub['mean_density'].to_numpy(dtype=float)
        ax.plot(centers, mean, color=bead.get('color', 'C0'), lw=2.2,
                label=rf"$2R_c = {bead['diameter_um']:.2f}\,\mu\mathrm{{m}}$")

    for xv in (-np.pi, 0.0, np.pi):
        ax.axvline(xv, color='#333333', ls='--', lw=1.0, alpha=0.6)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"], fontsize=13)
    ax.set_xlabel(r"$\Delta\theta = \phi - \theta_{\mathrm{nem}}$ [rad]", fontsize=14)
    ax.set_ylabel(r"$P(\Delta\theta)$ [$\mathrm{rad}^{-1}$]", fontsize=14)
    wnote = r" (weighted by $|u|$)" if weight_magnitude else ""
    ax.set_title(rf"MT Flow Orientation Histogram Relative to the Nematic Axis{wnote}{theta_note}",
                 fontsize=14.5, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=10.5, loc='upper center', framealpha=0.93)
    fig.tight_layout()
    save_figure_to_all(fig, 'mt_orientation_histogram_overlay', out_dirs)
    plt.close(fig)



# =========================================================================
# CLI
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optical flow (GFP_flows.h5) から求めた MT フロー配向角 phi の、"
            "大域ネマチック主軸角 theta_nem に対する角度差 Delta theta の "
            "ヒストグラム & レーダーチャートを作成する。"
        )
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="データルート（既定: /Volumes/... の既存候補を自動選択）.")
    parser.add_argument('--beads', type=str, nargs='+', default='all',
                        help="対象条件 (e.g. 'all', 'beads3um beads7um', 'beads06um,beads1um').")
    parser.add_argument('--output_dir', type=str, default='figure/mt_orientation',
                        help="出力ディレクトリ（相対パスはスクリプト基準. 既定 figure/mt_orientation）.")
    parser.add_argument('--no_save_root', action='store_true',
                        help="<root_dir>/figure/mt_orientation への保存を省略する.")

    # --- サンプリング ---
    parser.add_argument('--pixel_stride', type=int, default=8,
                        help="画素間引き（1 = 全画素. 既定 8）.")
    parser.add_argument('--frame_stride', type=int, default=5,
                        help="フレーム間引き（1 = 全フレーム. 既定 5）.")
    parser.add_argument('--max_frames_per_exp', type=int, default=None,
                        help="（デバッグ用）各実験で先頭 N フレームのみを使用.")
    parser.add_argument('--min_flow_mag', type=float, default=1e-4,
                        help="フロー有効判定の閾値（|u| > この値の画素のみ使用. 既定 1e-4）.")
    parser.add_argument('--weighting', type=str, default='none', choices=['none', 'magnitude'],
                        help="'none' = 等重み計数（既定）, 'magnitude' = 流速ノルムで重み付け.")
    parser.add_argument('--mask_radius_factor', type=float, default=0.0,
                        help="貨物粒子近傍除外半径 = factor x R_c [px]（0 で無効. 既定 0）.")
    parser.add_argument('--min_mask_radius_px', type=float, default=0.0,
                        help="貨物粒子近傍除外半径の下限 [px]（0 で無効. 既定 0）.")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="フレーム間隔 [s]（theta_nem(t) 時系列の時間軸に使用. 既定 4.0）.")
    parser.add_argument('--flow_cache', type=str, default='auto', choices=['auto', 'off', 'refresh'],
                        help="間引きフローのキャッシュ (mt_flow_cache_s{st}_f{fs}.h5)。"
                             "'auto'（既定）は有効なキャッシュがあれば再利用し、無ければ解析パス中に作成する"
                             "（追加 I/O 無しで、同じ pixel_stride/frame_stride での再実行が ~60 倍高速）。"
                             "'off' はキャッシュを使わない。'refresh' は常に作り直す。")
    parser.add_argument('--flow_cache_name', type=str, default=None,
                        help="キャッシュファイル名（既定 mt_flow_cache_s{pixel_stride}_f{frame_stride}.h5）.")
    parser.add_argument('--flow_cache_dir', type=str, default=None,
                        help="キャッシュの保存先ルート（未指定なら実験ディレクトリ内）。"
                             "NAS が遅い場合はローカルディスク（例 /tmp/mt_flow_cache）を指定すると再実行が高速になります。")
    parser.add_argument('--theta_sign', type=str, default='auto', choices=['auto', '+1', '-1'],
                        help="基準軸 theta_nem の符号。'auto'（既定）は +/- 両方を評価し、"
                             "Delta theta 分布が 0/pi に強くピークする方（= 光学フローと整合する方）を採用する。"
                             "MTs_im_theta.zarr の角度規約がフローの座標系とミラーしているデータセットでは -1 が選ばれる。")
    parser.add_argument('--mirror_margin', type=float, default=0.05,
                        help="--theta_sign auto で符号 -1 を採用するのに必要な <cos 2 Delta theta> の改善量（既定 0.05）.")

    # --- ヒストグラム / 作図 ---
    parser.add_argument('--bins', type=int, default=72,
                        help="展開分布 [-pi, pi) のビン数（既定 72 = 5 deg ビン）.")
    parser.add_argument('--bins_fold', type=int, default=36,
                        help="ネマチック折り返し分布 (-pi/2, pi/2] のビン数（既定 36 = 5 deg ビン）.")
    parser.add_argument('--align_tol_deg', type=float, default=30.0,
                        help="平行 / 反平行とみなす許容角度 [deg]（既定 30）.")
    parser.add_argument('--smooth_sigma_bins', type=float, default=1.0,
                        help="レーダーチャート輪郭の平滑化幅 [bins]（0 で無効. 既定 1）.")
    parser.add_argument('--ncols', type=int, default=3, help="パネル図の列数（既定 3）.")
    parser.add_argument('--no_progress', action='store_true', help="tqdm プログレスバーを無効化.")
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else find_default_root()
    if root_dir is None or not Path(root_dir).exists():
        raise FileNotFoundError("Data root directory not found. Please specify it with --root_dir.")

    out_arg = Path(args.output_dir).expanduser()
    out_dirs: List[Path] = [out_arg if out_arg.is_absolute() else (CURRENT_DIR / out_arg)]
    if not args.no_save_root:
        out_dirs.append(Path(root_dir) / 'figure' / 'mt_orientation')
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    bin_edges = make_angle_bins(args.bins, fold=False)
    bin_edges_fold = make_angle_bins(args.bins_fold, fold=True)
    target_beads = parse_target_beads(args.beads, BEADS_INFO)
    if not target_beads:
        raise RuntimeError("No target bead conditions selected.")

    weight_magnitude = (str(args.weighting) == 'magnitude')

    print("=" * 78)
    print(" MT Flow Orientation Distribution Relative to the Nematic Axis")
    print("=" * 78)
    print(f"Data Root Directory : {root_dir}")
    print(f"Output Directories  : {', '.join(str(d) for d in out_dirs)}")
    print(f"Target Beads        : {[b['name'] for b in target_beads]}")
    print(f"Pixel / frame stride: {args.pixel_stride} px / {args.frame_stride} frames")
    print(f"Bins                : {args.bins} (unfolded), {args.bins_fold} (nematic-folded)")
    print(f"Weighting           : {args.weighting}")
    print(f"Mask radius         : max({args.min_mask_radius_px:.0f} px, "
          f"{args.mask_radius_factor:.1f} x R_c) / {args.scale} um/px")
    print(f"Alignment tolerance : {args.align_tol_deg:.1f} deg")
    print(f"Theta sign          : {args.theta_sign} (mirror margin {args.mirror_margin:.2f})")
    print("-" * 78)

    # --- 各実験の角度ヒストグラムを集計 ---
    results: List[dict] = []
    for bead in target_beads:
        exp_dirs = find_experiment_dirs(root_dir, bead['name'])
        print(f"[{bead['name']}] {len(exp_dirs)} experiment dir(s) with {FLOW_NAME}")
        for exp_dir in exp_dirs:
            cache_name = resolve_flow_cache_name(
                args.flow_cache_dir, root_dir, exp_dir,
                args.pixel_stride, args.frame_stride, args.flow_cache_name)
            res = process_experiment(
                exp_dir, bead, bin_edges, bin_edges_fold,
                pixel_stride=args.pixel_stride,
                frame_stride=args.frame_stride,
                max_frames_per_exp=args.max_frames_per_exp,
                min_flow_mag=args.min_flow_mag,
                weighting=args.weighting,
                align_tol_deg=args.align_tol_deg,
                mask_radius_factor=args.mask_radius_factor,
                min_mask_radius_px=args.min_mask_radius_px,
                scale=args.scale,
                progress=not args.no_progress,
                flow_cache=args.flow_cache,
                flow_cache_name=cache_name,
            )
            if res is None:
                continue
            results.append(res)
            print(f"      -> {exp_dir}: {res['n_frames_used']}/{res['n_frames_total']} frames, "
                  f"{res['n_pixels']:,} px, theta = {res['theta_source']} "
                  f"(mean {np.rad2deg(res['theta_mean_rad']):.1f} deg), "
                  f"flow = {res['flow_cache_source']}, "
                  f"mask = {res['mask_radius_px']:.1f} px", flush=True)

    if not results:
        raise RuntimeError("No flow samples were extracted. Check --root_dir, --beads, and "
                           "the presence of GFP_flows.h5.")

    # --- 基準軸 theta_nem の符号（ミラー規約）判定 ---
    sign, sign_info = choose_theta_sign(results, theta_sign=args.theta_sign,
                                        margin=args.mirror_margin)
    results = [select_sign_variant(r, sign) for r in results]
    theta_note = theta_sign_note(sign)
    print("-" * 78)
    print(" Nematic axis sign check (theta_nem vs optical-flow frame)")
    print(f"   pooled <cos 2 dtheta> : +theta = {sign_info['S2_pooled_plus']:.4f}, "
          f"-theta = {sign_info['S2_pooled_minus']:.4f} "
          f"(mirror candidates: {sign_info['n_experiments_with_mirror']}/{len(results)} experiments)")
    print(f"   decision              : {sign_info['decision']}")
    if sign < 0:
        print("   [NOTE] MTs_im_theta.zarr の角度規約が光学フローの座標系とミラー関係にあるため、"
              "theta_nem -> -theta_nem として 0/pi ピークが現れる向きに補正しました。")
    print("-" * 78)

    # --- 集計 & CSV 出力 ---
    df_exp = per_experiment_table(results, align_tol_deg=args.align_tol_deg)
    df_curve, df_curve_fold = condition_histogram_table(
        results, target_beads, bin_edges, bin_edges_fold, weighting=args.weighting)
    df_summary = condition_summary_table(
        results, target_beads, align_tol_deg=args.align_tol_deg)
    df_theta_series = theta_timeseries_table(results, frame_interval=args.frame_interval)

    save_csv_to_all(df_exp, 'mt_orientation_per_experiment', out_dirs)
    save_csv_to_all(df_curve, 'mt_orientation_histogram', out_dirs)
    save_csv_to_all(df_curve_fold, 'mt_orientation_nematic_folded_histogram', out_dirs)
    save_csv_to_all(df_summary, 'mt_orientation_summary', out_dirs)
    if not df_theta_series.empty:
        save_csv_to_all(df_theta_series, 'mt_orientation_theta_nem_timeseries', out_dirs)

    # --- 作図 ---
    plot_histogram_panels(df_curve, df_summary, target_beads, out_dirs,
                          weighting=args.weighting, ncols=args.ncols, theta_note=theta_note)
    plot_histogram_overlay(df_curve, target_beads, out_dirs,
                           weight_magnitude=weight_magnitude, theta_note=theta_note)
    plot_radar_panels(df_curve, df_summary, target_beads, out_dirs,
                      weight_magnitude=weight_magnitude,
                      smooth_sigma_bins=args.smooth_sigma_bins, ncols=args.ncols,
                      theta_note=theta_note)
    plot_radar_overlay(df_curve, target_beads, out_dirs,
                       weight_magnitude=weight_magnitude,
                       smooth_sigma_bins=args.smooth_sigma_bins, theta_note=theta_note)
    plot_nematic_folded(df_curve_fold, df_summary, target_beads, out_dirs,
                        weight_magnitude=weight_magnitude, theta_note=theta_note)
    plot_theta_timeseries(df_theta_series, df_summary, target_beads, out_dirs,
                          theta_note=theta_note, ncols=args.ncols)

    # --- ログ出力 ---
    print("-" * 78)
    print(f" theta_nem sign = {sign}  ({sign_info['decision']})")
    print(" Per-experiment circular statistics")
    cols = ['bead_name', 'exp_dir', 'theta_source', 'n_frames_used', 'n_pixels',
            'mean_resultant_length_R', 'mean_cos_delta_theta', 'nematic_order_cos2',
            'nematic_order_cos2_signplus', 'nematic_order_cos2_signminus',
            'mean_abs_delta_theta_deg', 'frac_parallel_within_tol', 'frac_antiparallel_within_tol']
    print(df_exp[[c for c in cols if c in df_exp.columns]].to_string(index=False))
    print()
    print(" Per-condition summary (experiment-level mean +/- SEM)")
    cols2 = ['bead_name', 'diameter_um', 'n_experiments', 'n_samples', 'theta_sign',
             'nematic_order_cos2_mean', 'nematic_order_cos2_sem', 'nematic_order_cos2_pooled',
             'mean_abs_delta_theta_deg_mean', 'mean_abs_delta_theta_deg_pooled',
             'theta_nem_std_deg_continuous_mean', 'theta_nem_std_deg_continuous_sem',
             'theta_nem_axis_order_mean',
             'frac_parallel_within_tol_mean', 'frac_antiparallel_within_tol_mean']
    print(df_summary[[c for c in cols2 if c in df_summary.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()

