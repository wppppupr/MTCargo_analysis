#!/usr/bin/env python3
"""
plot_bg_angular_correlation.py

貨物粒子（ビーズ）近傍を除外した「バックグラウンド（バルク）微小管フロー」の空間配向相関

    C_bg(r) = < u(x) · u(x + r) >_{|r| ~= r},    u = 正規化オプティカルフロー

を、多数の仮想粒子（ランダムコントロール点）近傍で評価して統計量を稼ぎ、
粒子径（貨物サイズ）条件ごとに C_bg(r) 曲線と配向相関長 xi_bg を定量・可視化するスクリプトです。

【データ源と計算方法（hmm_flow_correlation_analysis.py と同じ FFTConvolver を使用）】
- 各実験ディレクトリの GFP_flows.h5（(frame, 2, y, x) のオプティカルフロー, channel-first）を読み、
  フレームごとに以下の領域を除外した上で、--n_virtual_points 個の仮想粒子位置をランダム抽出する
  （--seed 固定で再現可能）:
    1. 追跡貨物粒子（beads_tracks.csv）近傍: 半径 max(--min_mask_radius_px, --mask_radius_factor * R_c / scale) px
    2. フロー無効画素（|v| <= --min_flow_mag）
    3. 画像境界から --border_margin_px 以内（既定 = max(--distances) px。周期 FFT のラップアラウンド回避）
- libs.fft_convolution.FFTConvolver（ring カーネル, GPU 自動判定）で各仮想粒子まわりの
  C(r)（全成分）, C_parallel(r)（ネマチック主軸に平行）, C_perp(r)（垂直）を --distances（px グリッド）上で計算する。
  リング平均は有効マスク面積（貨物近傍と無効画素を除いた面積）で正規化する。
- 計算結果は実験ディレクトリ内に --cache_name（既定 angular_correlation_bg_vp.zarr,
  dims = distance x frame x virtual_point）として保存し、計算パラメータが一致すれば再利用する
  （--force_recompute で再計算）。
- --source existing を指定すると、既存の angular_correlation_bg.zarr（仮想粒子型 = dims に random_point を含むもの）を
  優先して読み込む（例: 0.63 / 1.18 / 3.37 um 条件の既存データ。迅速な確認用）。

【解析】
- 実験ごとに全 (frame x virtual point) サンプルの平均から C_bg(r) を求め、縦軸を ln C_bg にとった
  重み付き線形フィット ln C = ln a - r / xi（libs.hmm_flow_correlation.fit_flow_correlation_length、
  hmm_flow_correlation_analysis.py と同一実装）により実験ごとの相関長 xi_bg を算出する
  （--fit_range / --min_corr_threshold）。
- 条件（粒子径）ごとに実験間の mean ± SEM を集計し、サンプル数（フレーム x 仮想粒子数 x 実験数）も記録する。

【出力ファイル】
既定の出力先は (1) <作業ディレクトリ>/figure/bg_angular_correlation と
(2) <root_dir>/figure/bg_angular_correlation の 2 箇所（--no_save_root で (2) を省略, --output_dir で (1) を変更）。
1. bg_angular_correlation_Cr.png / .svg            : 条件別 C_bg(r)（実験別生カーブ + 実験間平均 ± SEM + 指数フィット）
2. bg_angular_correlation_xi_vs_diameter.png / .svg: xi_bg vs 貨物直径 2R_c（実験点 + 条件平均 ± SEM + 全体系平均）
3. bg_angular_correlation_par_perp.png / .svg      : 条件別のネマチック主軸分解（total / parallel / perpendicular）
4. bg_angular_correlation_points.csv               : 実験 x 距離ごとの C_bg(r)（平均・SEM・サンプル数, par/perp 含む）
5. bg_angular_correlation_curves.csv               : 条件 x 距離ごとの平均曲線（実験間平均 ± SEM, プール SEM, サンプル数）
6. bg_angular_correlation_length_per_experiment.csv: 実験ごとの xi_bg（フィット品質・サンプル数付き）
7. bg_angular_correlation_length_summary.csv       : 条件ごとの xi_bg 代表値（mean ± SEM, プールフィット値）
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# 親ディレクトリのパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs.calc_bg_angular_correlation import load_nematic_thetas, parse_distances
from libs.fft_convolution import FFTConvolver
from libs.hmm_flow_correlation import fit_flow_correlation_length

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

FLOW_NAME = 'GFP_flows.h5'
TRACKS_NAME = 'beads_tracks.csv'


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

BEAD_LOOKUP = {b['name']: b for b in BEADS_INFO}


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
        return beads_info

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

    try:
        return [b for b in beads_info if b['name'] in selected_names]
    except Exception:
        return list(beads_info)


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


def mask_radius_px_for_bead(radius_um: float, scale: float, factor: float, min_px: float) -> float:
    """
    貨物粒子近傍を除外するマスク半径（px）を返す。

    radius_px = max(min_px, factor * R_c / scale)
    = 粒子半径そのものではなく「粒子近傍（流れが乱されている領域）」を除外するための半径。
    """
    radius_px = float(radius_um) / float(scale)
    return float(max(float(min_px), float(factor) * radius_px))


def safe_save_csv(df: pd.DataFrame, target_path: Path, max_retries: int = 5) -> None:
    """NAS 等の一時的な I/O 失敗に耐える CSV 保存（hmm_flow_correlation_analysis.py と同じ方式）。"""
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
        target = d / f"{basename}.csv"
        safe_save_csv(df, target)
        saved.append(target)
    print(f"  Saved CSV: {basename}.csv -> {len(saved)} dir(s)")
    return saved


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300) -> List[Path]:
    """すべての出力ディレクトリへ PNG / SVG を保存する。"""
    saved = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'svg'):
            target = d / f"{basename}.{ext}"
            fig.savefig(target, dpi=dpi, bbox_inches='tight')
            saved.append(target)
    print(f"  Saved figure: {basename}.png/.svg -> {len(out_dirs)} dir(s)")
    return saved


# =========================================================================
# 仮想粒子（コントロール点）サンプリングによるバックグラウンド配向相関の計算
# =========================================================================

VIRTUAL_POINT_DIM = 'virtual_point'
_EXISTING_VP_DIM = 'random_point'


def open_virtual_point_dataset(path: Path) -> Optional[xr.Dataset]:
    """仮想粒子型（dims に virtual_point / random_point を含む）の zarr を開く。該当しなければ None。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        ds = xr.open_zarr(str(path), consolidated=False)
    except Exception as e:
        print(f"[WARNING] Failed to open {path}: {e}")
        return None
    if not ((VIRTUAL_POINT_DIM in ds.sizes) or (_EXISTING_VP_DIM in ds.sizes)) or ('angular_correlation' not in ds):
        ds.close()
        return None
    return ds


def virtual_point_samples(ds: xr.Dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int, str]:
    """(distance, frame, virtual_point) のサンプル配列・距離座標・仮想粒子数を取り出す。"""
    dim = VIRTUAL_POINT_DIM if VIRTUAL_POINT_DIM in ds.sizes else _EXISTING_VP_DIM
    dist_px = np.asarray(ds.coords['distance'].values, dtype=float)
    n_pts = int(ds.sizes[dim])

    def _get(name: str) -> Optional[np.ndarray]:
        if name not in ds:
            return None
        arr = ds[name]
        try:
            arr = arr.transpose('distance', 'frame', dim)
        except Exception:
            return None
        return np.asarray(arr.values, dtype=np.float32)

    return (_get('angular_correlation'), _get('angular_correlation_parallel'),
            _get('angular_correlation_perpendicular'), dist_px, n_pts, dim)


def cache_matches(
    ds: xr.Dataset,
    distances: Sequence[float],
    kernel_type: str,
    shell_width: float,
    n_virtual_points: int,
    mask_radius_px: float,
    border_margin_px: int,
    min_flow_mag: float,
) -> bool:
    """キャッシュ zarr の attrs / 距離座標が要求パラメータと一致するかを判定する。"""
    try:
        attrs = ds.attrs
        if int(attrs.get('n_virtual_points', -1)) != int(n_virtual_points):
            return False
        if abs(float(attrs.get('mask_radius_px', -1.0)) - float(mask_radius_px)) > 1e-6:
            return False
        if str(attrs.get('kernel_type', '')) != str(kernel_type):
            return False
        if abs(float(attrs.get('shell_width', -1.0)) - float(shell_width)) > 1e-6:
            return False
        if int(attrs.get('border_margin_px', -1)) != int(border_margin_px):
            return False
        if abs(float(attrs.get('min_flow_mag', -1.0)) - float(min_flow_mag)) > 1e-12:
            return False
        if not np.allclose(np.asarray(ds.coords['distance'].values, dtype=float),
                           np.asarray(distances, dtype=float)):
            return False
        return True
    except Exception:
        return False


def compute_virtual_point_correlations(
    exp_dir: Path,
    distances: Sequence[float],
    n_virtual_points: int,
    mask_radius_px: float,
    border_margin_px: int,
    kernel_type: str = 'ring',
    shell_width: float = 2.0,
    seed: int = 42,
    device: Optional[str] = None,
    min_flow_mag: float = 1e-4,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    verbose: bool = True,
) -> Optional[xr.Dataset]:
    """
    1 つの実験ディレクトリについて、貨物粒子近傍・無効画素・画像境界を除外した領域から
    仮想粒子（コントロール点）を n_virtual_points 個 / フレーム抽出し、
    各点まわりの角空間相関 C(r), C_parallel(r), C_perp(r) を計算して xarray.Dataset を返す。

    Returns
    -------
    ds : xr.Dataset or None
        dims = (distance, frame, virtual_point) の 3 変数（total / parallel / perpendicular）と
        ネマチック角 theta_nematic を保持。フローが読めない場合は None。
    """
    exp_dir = Path(exp_dir)
    flow_path = exp_dir / FLOW_NAME
    tracks_path = exp_dir / TRACKS_NAME
    if not flow_path.exists():
        return None

    df_tracks: Optional[pd.DataFrame] = None
    if tracks_path.exists():
        try:
            df_tracks = pd.read_csv(tracks_path)
        except Exception as e:
            print(f"[WARNING] Failed to read {tracks_path}: {e}")
    else:
        print(f"[WARNING] {TRACKS_NAME} not found in {exp_dir.name}: 貨物粒子近傍マスクなしで実行します")

    rng = np.random.default_rng(int(seed))
    distances = [float(d) for d in distances]
    n_pts = int(n_virtual_points)

    with h5py.File(str(flow_path), 'r') as f:
        key = list(f.keys())[0]
        flow = f[key]
        shape = flow.shape
        num_frames = int(shape[0])
        channel_first = not (shape[-1] == 2)
        rows, cols = (int(shape[2]), int(shape[3])) if channel_first else (int(shape[1]), int(shape[2]))

        # ネマチック主軸角 theta(t)（MTs_im_theta.zarr を優先、無ければフローから算出）
        thetas = load_nematic_thetas(exp_dir, num_frames, flow_data=flow, channel_first=channel_first)

        frames = list(range(0, num_frames, max(1, int(frame_stride))))
        if max_frames is not None:
            frames = frames[:int(max_frames)]
        if not frames:
            return None

        # 画像境界マージン（周期 FFT のラップアラウンド回避: 最大距離ぶん内側のみを候補にする）
        y0 = int(min(max(0, border_margin_px), rows // 2))
        x0 = int(min(max(0, border_margin_px), cols // 2))
        y1, x1 = rows - y0, cols - x0
        if y1 - y0 < 8 or x1 - x0 < 8:
            y0 = x0 = 0
            y1, x1 = rows, cols

        if verbose:
            print(f"    FFTConvolver init ({len(distances)} distances, kernel={kernel_type}, "
                  f"device={device or 'auto'})...", flush=True)
        convolver = FFTConvolver(shape=(rows, cols), sizes=distances, kernel_type=kernel_type,
                                 shell_width=shell_width, device=device)
        if verbose:
            print(f"    backend={convolver.device_type}; sampling {n_pts} virtual points/frame "
                  f"(mask radius = {mask_radius_px:.1f} px, border margin = {y0} px)", flush=True)

        n_d = len(distances)
        arr_total = np.full((n_d, len(frames), n_pts), np.nan, dtype=np.float32)
        arr_par = np.full((n_d, len(frames), n_pts), np.nan, dtype=np.float32)
        arr_perp = np.full((n_d, len(frames), n_pts), np.nan, dtype=np.float32)

        rows_axis = np.arange(rows, dtype=np.float32)[:, None]
        cols_axis = np.arange(cols, dtype=np.float32)[None, :]
        r_sq = float(mask_radius_px) ** 2
        grouped = df_tracks.groupby('frame') if df_tracks is not None else None

        iterator = tqdm(frames, desc=f"    {exp_dir.name}", leave=False) if verbose else frames
        for fi, t in enumerate(iterator):
            m_x = (flow[t, 0] if channel_first else flow[t, ..., 0]).astype(np.float32)
            m_y = (flow[t, 1] if channel_first else flow[t, ..., 1]).astype(np.float32)
            v_mag = np.hypot(m_x, m_y)
            valid = (v_mag > float(min_flow_mag))

            # 1. 貨物粒子近傍の除外（粒子中心から mask_radius_px 以内）
            if grouped is not None and t in grouped.groups:
                sub = df_tracks.loc[grouped.groups[t]]
                for py, px in zip(sub['y'].to_numpy(dtype=float), sub['x'].to_numpy(dtype=float)):
                    if not (np.isfinite(py) and np.isfinite(px)):
                        continue
                    ylo = int(max(0, np.floor(py - mask_radius_px)))
                    yhi = int(min(rows, np.ceil(py + mask_radius_px) + 1))
                    xlo = int(max(0, np.floor(px - mask_radius_px)))
                    xhi = int(min(cols, np.ceil(px + mask_radius_px) + 1))
                    if yhi <= ylo or xhi <= xlo:
                        continue
                    dist_sq = (rows_axis[ylo:yhi] - py) ** 2 + (cols_axis[:, xlo:xhi] - px) ** 2
                    valid[ylo:yhi, xlo:xhi] &= (dist_sq > r_sq)

            # 2. 境界マージンを除いた領域から仮想粒子位置をランダム抽出
            cand_y, cand_x = np.nonzero(valid[y0:y1, x0:x1])
            if cand_y.size == 0:
                continue
            k = int(min(n_pts, cand_y.size))
            sel = rng.choice(cand_y.size, size=k, replace=False)
            sel_y = (cand_y[sel] + y0).astype(np.int64)
            sel_x = (cand_x[sel] + x0).astype(np.int64)

            with np.errstate(divide='ignore', invalid='ignore'):
                m_ux = np.where(valid, m_x / np.maximum(v_mag, 1e-6), 0.0).astype(np.float32)
                m_uy = np.where(valid, m_y / np.maximum(v_mag, 1e-6), 0.0).astype(np.float32)

            c_ux = np.asarray(m_ux[sel_y, sel_x], dtype=np.float32)
            c_uy = np.asarray(m_uy[sel_y, sel_x], dtype=np.float32)
            th_t = float(thetas[t]) if t < len(thetas) else 0.0

            # 3. リング平均は有効マスク面積で正規化（valid_mask を渡す）
            res = convolver.convolve_and_sample_angular_correlation(
                m_ux=m_ux, m_uy=m_uy, valid_mask=valid.astype(np.float32),
                y_idx=sel_y, x_idx=sel_x,
                center_flow_ux=c_ux, center_flow_uy=c_uy,
                b_ux=c_ux, b_uy=c_uy,
                theta=th_t, roi_slice=None,
            )
            arr_total[:, fi, :k] = res['flow_total']
            arr_par[:, fi, :k] = res['flow_par']
            arr_perp[:, fi, :k] = res['flow_perp']

    dist_coord = np.asarray(distances, dtype=float)
    frame_coord = np.asarray(frames, dtype=int)
    ds = xr.Dataset(
        data_vars={
            'angular_correlation': xr.DataArray(
                arr_total, dims=['distance', 'frame', VIRTUAL_POINT_DIM],
                coords={'distance': dist_coord, 'frame': frame_coord,
                        VIRTUAL_POINT_DIM: np.arange(n_pts)}),
            'angular_correlation_parallel': xr.DataArray(
                arr_par, dims=['distance', 'frame', VIRTUAL_POINT_DIM],
                coords={'distance': dist_coord, 'frame': frame_coord,
                        VIRTUAL_POINT_DIM: np.arange(n_pts)}),
            'angular_correlation_perpendicular': xr.DataArray(
                arr_perp, dims=['distance', 'frame', VIRTUAL_POINT_DIM],
                coords={'distance': dist_coord, 'frame': frame_coord,
                        VIRTUAL_POINT_DIM: np.arange(n_pts)}),
            'theta_nematic': xr.DataArray(
                np.asarray(thetas, dtype=np.float32), dims=['frame'],
                coords={'frame': np.arange(len(thetas))}),
        },
        attrs={
            'description': ('Background (cargo-vicinity excluded) MT flow angular spatial correlation '
                            'from random virtual particle (control point) sampling'),
            'method': 'Random Virtual Points + FFTConvolver ring kernel (mask-area normalized)',
            'source_file': FLOW_NAME,
            'n_virtual_points': int(n_pts),
            'mask_radius_px': float(mask_radius_px),
            'border_margin_px': int(y0),
            'kernel_type': str(kernel_type),
            'shell_width': float(shell_width),
            'min_flow_mag': float(min_flow_mag),
            'seed': int(seed),
            'frame_stride': int(frame_stride),
            'n_frames': int(len(frames)),
            'image_shape': [int(rows), int(cols)],
        },
    )
    return ds


def save_dataset_cache(ds: xr.Dataset, cache_path: Path) -> None:
    """計算結果を zarr として保存する（既存があれば削除して上書き）。"""
    cache_path = Path(cache_path)
    try:
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
        ds.to_zarr(str(cache_path), mode='w', consolidated=False)
        print(f"    Saved cache: {cache_path.name}", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to save cache {cache_path}: {e}")


def load_experiment_correlation(
    exp_dir: Path,
    binfo: dict,
    args: argparse.Namespace,
    distances: Sequence[float],
    verbose: bool = True,
) -> Optional[dict]:
    """
    1 実験の仮想粒子バックグラウンド相関サンプルを取得する。

    取得順:
      1. --source existing: 既存の angular_correlation_bg.zarr（仮想粒子型 = random_point 次元を持つもの）
      2. --cache_name のキャッシュ zarr（計算パラメータが一致する場合。--force_recompute で無効化）
      3. GFP_flows.h5 から計算し、キャッシュ zarr として保存
    """
    exp_dir = Path(exp_dir)
    mask_px = mask_radius_px_for_bead(binfo['radius_um'], args.scale,
                                      args.mask_radius_factor, args.min_mask_radius_px)
    border_px = int(args.border_margin_px) if args.border_margin_px is not None else int(np.ceil(max(distances)))

    samples: Optional[np.ndarray] = None
    par: Optional[np.ndarray] = None
    perp: Optional[np.ndarray] = None
    dist_px = np.asarray(distances, dtype=float)
    n_pts = int(args.n_virtual_points)
    mask_out = float(mask_px)
    source = ''

    if args.source == 'existing':
        ds_ex = open_virtual_point_dataset(exp_dir / args.existing_zarr_name)
        if ds_ex is not None:
            out = virtual_point_samples(ds_ex)
            mask_attr = ds_ex.attrs.get('particle_mask_radius', ds_ex.attrs.get('mask_radius_px', None))
            ds_ex.close()
            if out is not None and out[0] is not None:
                samples, par, perp, dist_px, n_pts, _ = out
                mask_out = float(mask_attr) if mask_attr is not None else np.nan
                source = f"existing:{args.existing_zarr_name}:{n_pts}pts"
                print(f"    [existing] {exp_dir.name}: dims = ({len(dist_px)}, {samples.shape[1]}, {n_pts})")
            else:
                print(f"    [existing] {exp_dir.name}: {args.existing_zarr_name} にサンプル変数がありません")
        else:
            print(f"    [existing] {exp_dir.name}: 仮想粒子型 {args.existing_zarr_name} が無いため計算します")

    if samples is None:
        cache_path = exp_dir / args.cache_name
        if cache_path.exists() and not args.force_recompute:
            ds_c = open_virtual_point_dataset(cache_path)
            if ds_c is not None:
                is_ok = cache_matches(ds_c, distances, args.kernel_type, args.shell_width,
                                      int(args.n_virtual_points), mask_px, border_px, args.min_flow_mag)
                out = virtual_point_samples(ds_c) if is_ok else None
                ds_c.close()
                if is_ok and out is not None and out[0] is not None:
                    samples, par, perp, dist_px, n_pts, _ = out
                    source = f"cache:{args.cache_name}:{n_pts}pts"
                    print(f"    [cache] {exp_dir.name}: dims = ({len(dist_px)}, {samples.shape[1]}, {n_pts})")

    if samples is None:
        if verbose:
            print(f"    [compute] {exp_dir.name}: {FLOW_NAME} から仮想粒子 "
                  f"{int(args.n_virtual_points)} 点/フレームを計算します", flush=True)
        ds = compute_virtual_point_correlations(
            exp_dir, distances, int(args.n_virtual_points), mask_px, border_px,
            kernel_type=args.kernel_type, shell_width=args.shell_width,
            seed=args.seed, device=args.device, min_flow_mag=args.min_flow_mag,
            max_frames=args.max_frames, frame_stride=args.frame_stride, verbose=verbose,
        )
        if ds is None:
            print(f"[WARNING] {exp_dir.name}: {FLOW_NAME} を読み込めませんでした（スキップ）")
            return None
        if not args.no_cache:
            save_dataset_cache(ds, exp_dir / args.cache_name)
        out = virtual_point_samples(ds)
        ds.close()
        if out is None or out[0] is None:
            return None
        samples, par, perp, dist_px, n_pts, _ = out
        source = f"computed:{int(args.n_virtual_points)}pts"

    return {
        'bead_name': binfo['name'],
        'diameter_um': float(binfo['diameter_um']),
        'radius_um': float(binfo['radius_um']),
        'exp_dir': exp_dir.name,
        'exp_path': str(exp_dir),
        'distances_px': np.asarray(dist_px, dtype=float),
        'distances_um': np.asarray(dist_px, dtype=float) * float(args.scale),
        'samples': samples,
        'samples_par': par,
        'samples_perp': perp,
        'n_frames': int(samples.shape[1]),
        'n_virtual_points': int(n_pts),
        'mask_radius_px': float(mask_out),
        'source': source,
    }


def summarize_samples(samples: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance, frame, virtual_point) サンプル配列 -> 距離ごとの (mean, sem, n)（NaN は無視）。"""
    if samples is None:
        return np.array([]), np.array([]), np.array([])
    n_d = samples.shape[0]
    flat = samples.reshape(n_d, -1)
    n = np.sum(np.isfinite(flat), axis=1).astype(int)
    vals = np.where(np.isfinite(flat), flat, np.nan)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(vals, axis=1)
        std = np.nanstd(vals, axis=1, ddof=1)
    sem = np.where(n > 1, std / np.sqrt(np.maximum(n, 1)), 0.0)
    return np.asarray(mean, dtype=float), np.asarray(sem, dtype=float), n


def fit_xi_from_curve(
    distance_um: np.ndarray,
    mean_c: np.ndarray,
    sem_c: np.ndarray,
    min_fit_dist: float,
    max_fit_dist: float,
    min_corr_threshold: float,
) -> Dict[str, object]:
    """
    C_bg(r) 曲線から ln C = ln a - r / xi の重み付き線形フィットにより xi_bg を求める。

    フィット実装は hmm_flow_correlation_analysis.py と共通
    （libs.hmm_flow_correlation.fit_flow_correlation_length）で、これにより
    既存の xi_flow（粒子近傍・BG モード）と同一の定義・同一のフィット範囲条件で比較できる。
    """
    empty: Dict[str, object] = {
        'xi_um': np.nan, 'xi_err_um': np.nan, 'r2_log': np.nan, 'r2': np.nan,
        'r_peak_um': np.nan, 'r_fit_min_um': np.nan, 'r_fit_max_um': np.nan,
        'n_fit_points': 0, 'fit_r': np.array([]), 'fit_c': np.array([]),
    }
    distance_um = np.asarray(distance_um, dtype=float)
    mean_c = np.asarray(mean_c, dtype=float)
    sem_c = np.asarray(sem_c, dtype=float) if sem_c is not None else np.zeros_like(distance_um)
    if distance_um.size < 3:
        return empty

    df = pd.DataFrame({
        'distance_um': distance_um,
        'mean_correlation': mean_c,
        'sem_correlation': sem_c,
    })
    res = fit_flow_correlation_length(df, min_fit_dist=float(min_fit_dist),
                                      max_fit_dist=float(max_fit_dist),
                                      min_corr_threshold=float(min_corr_threshold))
    out = dict(empty)
    for key in ('xi_um', 'xi_err_um', 'r2', 'r2_log', 'r_peak_um', 'r_fit_min_um', 'r_fit_max_um',
                'fit_r', 'fit_c'):
        if key in res:
            out[key] = res[key]
    r_min = out.get('r_fit_min_um', np.nan)
    r_max = out.get('r_fit_max_um', np.nan)
    if np.isfinite(r_min) and np.isfinite(r_max):
        sel = np.isfinite(distance_um) & np.isfinite(mean_c)
        out['n_fit_points'] = int(np.count_nonzero(sel & (distance_um >= r_min) & (distance_um <= r_max)))
    return out


# =========================================================================
# 集計（実験 -> 条件）
# =========================================================================

def build_tables(
    exp_results: List[dict],
    fit_range: Tuple[float, float],
    min_corr_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, int], List[float]]]:
    """
    実験ごとの C_bg(r) 表（points）と xi_bg 表（xi）を作成し、
    条件ごとのプール統計（n, sum, sum of squares）を同時に蓄積して返す。
    """
    point_rows: List[dict] = []
    xi_rows: List[dict] = []
    pool: Dict[Tuple[str, str, int], List[float]] = {}

    def _accum(kind: str, bead: str, samples: Optional[np.ndarray]) -> None:
        if samples is None:
            return
        flat = samples.reshape(samples.shape[0], -1)
        for i in range(flat.shape[0]):
            v = flat[i][np.isfinite(flat[i])]
            if v.size == 0:
                continue
            acc = pool.setdefault((kind, bead, i), [0.0, 0.0, 0.0])
            v64 = v.astype(np.float64)
            acc[0] += float(v.size)
            acc[1] += float(np.sum(v64))
            acc[2] += float(np.sum(v64 ** 2))

    for res in exp_results:
        bead = res['bead_name']
        r_um = res['distances_um']
        mean_c, sem_c, n_c = summarize_samples(res['samples'])
        mean_par, sem_par, _ = summarize_samples(res['samples_par'])
        mean_perp, sem_perp, _ = summarize_samples(res['samples_perp'])
        _accum('total', bead, res['samples'])
        _accum('par', bead, res['samples_par'])
        _accum('perp', bead, res['samples_perp'])

        fit = fit_xi_from_curve(r_um, mean_c, sem_c, fit_range[0], fit_range[1], min_corr_threshold)

        for i, r in enumerate(r_um):
            point_rows.append({
                'bead_name': bead,
                'diameter_um': res['diameter_um'],
                'exp_dir': res['exp_dir'],
                'source': res['source'],
                'distance_px': float(res['distances_px'][i]),
                'distance_um': float(r),
                'mean_c': float(mean_c[i]) if i < mean_c.size else np.nan,
                'sem_c': float(sem_c[i]) if i < sem_c.size else np.nan,
                'n_samples': int(n_c[i]) if i < n_c.size else 0,
                'mean_c_par': float(mean_par[i]) if i < mean_par.size else np.nan,
                'sem_c_par': float(sem_par[i]) if i < sem_par.size else np.nan,
                'mean_c_perp': float(mean_perp[i]) if i < mean_perp.size else np.nan,
                'sem_c_perp': float(sem_perp[i]) if i < sem_perp.size else np.nan,
            })

        xi_rows.append({
            'bead_name': bead,
            'diameter_um': res['diameter_um'],
            'exp_dir': res['exp_dir'],
            'source': res['source'],
            'n_frames': res['n_frames'],
            'n_virtual_points': res['n_virtual_points'],
            'mask_radius_px': res['mask_radius_px'],
            'n_samples_total': int(np.sum(n_c)),
            'xi_um': float(fit['xi_um']),
            'xi_err_um': float(fit['xi_err_um']),
            'xi_r2_log': float(fit['r2_log']),
            'xi_r2': float(fit['r2']),
            'n_fit_points': int(fit['n_fit_points']),
            'r_peak_um': float(fit['r_peak_um']),
            'fit_min_um': float(fit['r_fit_min_um']),
            'fit_max_um': float(fit['r_fit_max_um']),
        })

    return pd.DataFrame(point_rows), pd.DataFrame(xi_rows), pool


def pool_stat(
    pool: Dict[Tuple[str, str, int], List[float]],
    kind: str,
    bead: str,
    index: int,
) -> Tuple[float, float, int]:
    """プール統計 (n, sum, sumsq) から (mean, sem, n) を計算する（全実験・全フレーム・全仮想粒子を合算）。"""
    acc = pool.get((kind, bead, index))
    if acc is None or acc[0] <= 1:
        return np.nan, np.nan, 0
    n, s, s2 = acc
    mean = s / n
    var = max(s2 / n - mean ** 2, 0.0) * (n / (n - 1.0))
    return float(mean), float(np.sqrt(var / n)), int(n)


def summarize_conditions(
    df_points: pd.DataFrame,
    df_xi: pd.DataFrame,
    pool: Dict[Tuple[str, str, int], List[float]],
    beads_info: List[dict],
    grid_distances: np.ndarray,
    fit_range: Tuple[float, float],
    min_corr_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    条件（粒子径）ごとに C_bg(r) の実験間平均 ± SEM・プール平均 ± SEM・サンプル数を集計し、
    条件平均曲線とプール曲線をそれぞれフィットして xi_bg を求める。
    """
    curve_rows: List[dict] = []
    summary_rows: List[dict] = []
    grid = np.asarray(grid_distances, dtype=float)

    for binfo in beads_info:
        bead = binfo['name']
        sub = df_points[df_points['bead_name'] == bead]
        sub_xi = df_xi[df_xi['bead_name'] == bead]
        if sub.empty or sub_xi.empty:
            continue
        n_exp = int(sub['exp_dir'].nunique())
        n_d = grid.size

        exp_mean = np.full(n_d, np.nan)
        exp_sem = np.full(n_d, np.nan)
        par_mean = np.full(n_d, np.nan)
        par_sem = np.full(n_d, np.nan)
        perp_mean = np.full(n_d, np.nan)
        perp_sem = np.full(n_d, np.nan)
        pooled_mean = np.full(n_d, np.nan)
        pooled_sem = np.full(n_d, np.nan)
        pooled_n = np.zeros(n_d, dtype=int)

        for i, r in enumerate(grid):
            g = sub[np.isclose(sub['distance_um'], r)]
            if g.empty:
                continue
            vals = g['mean_c'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                exp_mean[i] = float(np.mean(vals))
                exp_sem[i] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            for mean_arr, sem_arr, mcol in ((par_mean, par_sem, 'mean_c_par'),
                                            (perp_mean, perp_sem, 'mean_c_perp')):
                vm = g[mcol].to_numpy(dtype=float)
                vm = vm[np.isfinite(vm)]
                if vm.size:
                    mean_arr[i] = float(np.mean(vm))
                    sem_arr[i] = float(np.std(vm, ddof=1) / np.sqrt(vm.size)) if vm.size > 1 else 0.0
            pm, ps, pn = pool_stat(pool, 'total', bead, i)
            pooled_mean[i], pooled_sem[i], pooled_n[i] = pm, ps, pn

        fit_cond = fit_xi_from_curve(grid, exp_mean, exp_sem, fit_range[0], fit_range[1], min_corr_threshold)
        fit_pooled = fit_xi_from_curve(grid, pooled_mean, pooled_sem, fit_range[0], fit_range[1], min_corr_threshold)

        for i, r in enumerate(grid):
            curve_rows.append({
                'bead_name': bead,
                'diameter_um': binfo['diameter_um'],
                'distance_um': float(r),
                'mean_c': float(exp_mean[i]),
                'sem_c': float(exp_sem[i]),
                'n_experiments': n_exp,
                'mean_c_par': float(par_mean[i]),
                'sem_c_par': float(par_sem[i]),
                'mean_c_perp': float(perp_mean[i]),
                'sem_c_perp': float(perp_sem[i]),
                'pooled_mean_c': float(pooled_mean[i]),
                'pooled_sem_c': float(pooled_sem[i]),
                'pooled_n_samples': int(pooled_n[i]),
                'xi_bg_cond_fit_um': float(fit_cond['xi_um']),
                'xi_bg_cond_fit_err_um': float(fit_cond['xi_err_um']),
                'xi_bg_cond_fit_r2_log': float(fit_cond['r2_log']),
                'xi_bg_pooled_um': float(fit_pooled['xi_um']),
                'xi_bg_pooled_err_um': float(fit_pooled['xi_err_um']),
                'xi_bg_pooled_r2_log': float(fit_pooled['r2_log']),
            })

        xi_vals = sub_xi['xi_um'].to_numpy(dtype=float)
        xi_vals = xi_vals[np.isfinite(xi_vals)]
        summary_rows.append({
            'bead_name': bead,
            'diameter_um': binfo['diameter_um'],
            'radius_um': binfo['radius_um'],
            'n_experiments': n_exp,
            'n_frames_total': int(sub_xi['n_frames'].sum()),
            'n_samples_total': int(sub_xi['n_samples_total'].sum()),
            'n_virtual_points': int(sub_xi['n_virtual_points'].max()),
            'mask_radius_px': float(sub_xi['mask_radius_px'].mean()),
            'xi_bg_mean_um': float(np.mean(xi_vals)) if xi_vals.size else np.nan,
            'xi_bg_sem_um': float(np.std(xi_vals, ddof=1) / np.sqrt(xi_vals.size)) if xi_vals.size > 1 else 0.0,
            'xi_bg_std_um': float(np.std(xi_vals, ddof=1)) if xi_vals.size > 1 else 0.0,
            'xi_bg_median_um': float(np.median(xi_vals)) if xi_vals.size else np.nan,
            'xi_bg_n_experiments': int(xi_vals.size),
            'xi_bg_cond_fit_um': float(fit_cond['xi_um']),
            'xi_bg_cond_fit_err_um': float(fit_cond['xi_err_um']),
            'xi_bg_cond_fit_r2_log': float(fit_cond['r2_log']),
            'xi_bg_pooled_um': float(fit_pooled['xi_um']),
            'xi_bg_pooled_err_um': float(fit_pooled['xi_err_um']),
            'xi_bg_pooled_r2_log': float(fit_pooled['r2_log']),
        })

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def weighted_global_xi(df_xi: pd.DataFrame) -> Tuple[float, float, int]:
    """全実験の xi_bg を 1/err^2 重みで合算した代表値 (mean, sem, n) を返す。"""
    sub = df_xi[np.isfinite(df_xi['xi_um'])].copy()
    if sub.empty:
        return np.nan, np.nan, 0
    vals = sub['xi_um'].to_numpy(dtype=float)
    errs = sub['xi_err_um'].to_numpy(dtype=float)
    w = np.where(np.isfinite(errs) & (errs > 0), 1.0 / np.maximum(errs, 1e-9) ** 2, 1.0)
    w_sum = float(np.sum(w))
    mean = float(np.sum(w * vals) / w_sum)
    var = float(np.sum(w * (vals - mean) ** 2) / w_sum) * (w.size / max(w.size - 1, 1))
    sem = float(np.sqrt(max(var, 0.0) / w.size))
    return mean, sem, int(vals.size)


# =========================================================================
# 作図
# =========================================================================

def plot_condition_curves(
    df_curves: pd.DataFrame,
    df_points: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    meta: dict,
    fit_range: Tuple[float, float],
    min_corr_threshold: float,
    xscale: str = 'log',
    max_dist: float = 60.0,
    ylim: Tuple[float, float] = (-0.2, 1.05),
) -> None:
    """条件（粒子径）ごとの C_bg(r) を 1 軸に描く（実験別生カーブ + 実験間平均 ± SEM + 指数フィット）。"""
    if df_curves.empty:
        print("[WARNING] C_bg(r) 曲線データが空のため図 1 をスキップします")
        return

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    exp_dirs = sorted(df_points['exp_dir'].unique())
    n_exp = len(exp_dirs)

    for i, e in enumerate(exp_dirs):
        g = df_points[df_points['exp_dir'] == e].sort_values('distance_um')
        ax.plot(g['distance_um'], g['mean_c'], color='#b3b3b3', lw=0.9, alpha=0.75, zorder=1,
                label=(rf"Individual experiments ($N = {n_exp}$)" if i == 0 else None))

    for binfo in BEADS_INFO:
        sub = df_curves[df_curves['bead_name'] == binfo['name']].sort_values('distance_um')
        if sub.empty:
            continue
        row = df_summary[df_summary['bead_name'] == binfo['name']]
        xi_mean = float(row['xi_bg_mean_um'].iloc[0]) if not row.empty else np.nan
        xi_sem = float(row['xi_bg_sem_um'].iloc[0]) if not row.empty else np.nan
        if np.isfinite(xi_mean) and np.isfinite(xi_sem) and xi_sem > 0:
            label = (rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$ "
                     rf"($\xi_{{\mathrm{{bg}}}} = {xi_mean:.1f} \pm {xi_sem:.1f}\,\mu\mathrm{{m}}$)")
        elif np.isfinite(xi_mean):
            label = (rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$ "
                     rf"($\xi_{{\mathrm{{bg}}}} = {xi_mean:.1f}\,\mu\mathrm{{m}}$)")
        else:
            label = rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$"

        ax.errorbar(sub['distance_um'], sub['mean_c'], yerr=sub['sem_c'],
                    fmt=binfo['marker'], ms=7.0, color=binfo['color'],
                    mfc=binfo['color'], mec='black', mew=0.8,
                    elinewidth=1.1, capsize=2.5, lw=1.5, alpha=0.95, zorder=4, label=label)

        fit = fit_xi_from_curve(sub['distance_um'].to_numpy(dtype=float),
                                sub['mean_c'].to_numpy(dtype=float),
                                sub['sem_c'].to_numpy(dtype=float),
                                fit_range[0], fit_range[1], min_corr_threshold)
        fit_r = np.asarray(fit.get('fit_r', np.array([])), dtype=float)
        fit_c = np.asarray(fit.get('fit_c', np.array([])), dtype=float)
        if fit_r.size and np.isfinite(fit.get('xi_um', np.nan)):
            ax.plot(fit_r, fit_c, ls='--', color=binfo['color'], lw=1.5, alpha=0.9, zorder=3)

    ax.axhline(0.0, color='gray', ls='--', lw=0.9, alpha=0.6)
    ax.set_xscale(xscale)
    x_min = float(max(np.nanmin(df_curves['distance_um']) * 0.8, 1e-3)) if xscale == 'log' else 0.0
    ax.set_xlim(x_min, max_dist)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"Distance $r$ from Virtual Particle Center [$\mu\mathrm{m}$]",
                  fontsize=13, fontweight='bold')
    ax.set_ylabel(r"Background Flow Spatial Correlation $C_{\mathrm{bg}}(r)$",
                  fontsize=13, fontweight='bold')
    ax.set_title("Background MT Flow Spatial Correlation\n"
                 r"(Cargo Vicinity Excluded, Virtual-Particle Sampling)",
                 fontsize=13.5, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.35)

    info = (rf"$N_{{\mathrm{{exp}}}} = {meta.get('n_exp', n_exp)}$; "
            rf"$N_{{\mathrm{{virtual}}}} = {meta.get('n_virtual_points', 0)}$/frame" + "\n"
            rf"$N_{{\mathrm{{frames}}}} = {meta.get('n_frames', 0)}$; "
            rf"$N_{{\mathrm{{samples}}}} = {meta.get('n_samples', 0):,}$" + "\n"
            rf"mask $R = {meta.get('mask_min_px', np.nan):.0f}$-${meta.get('mask_max_px', np.nan):.0f}$ px; "
            rf"fit range ${fit_range[0]:.0f}$-${fit_range[1]:.0f}\,\mu\mathrm{{m}}$")
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va='top', ha='left', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.88))

    ax.legend(fontsize=9.0, loc='upper right', framealpha=0.93,
              title=r"Background Flow $C_{\mathrm{bg}}(r) = a\exp(-r/\xi_{\mathrm{bg}})$",
              title_fontsize=9.5)
    fig.tight_layout()
    save_figure_to_all(fig, 'bg_angular_correlation_Cr', out_dirs)
    plt.close(fig)


def plot_xi_vs_diameter(
    df_xi: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    global_xi: Tuple[float, float, int],
    xscale: str = 'log',
) -> None:
    """xi_bg vs 貨物直径 2R_c（実験点 + 条件平均 ± SEM + 全実験代表値 + プールフィット値）。"""
    if df_summary.empty:
        print("[WARNING] xi_bg 集計データが空のため図 2 をスキップします")
        return

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    rng = np.random.default_rng(7)
    plotted_any = False

    for binfo in BEADS_INFO:
        sub = df_xi[(df_xi['bead_name'] == binfo['name']) & np.isfinite(df_xi['xi_um'])]
        row = df_summary[df_summary['bead_name'] == binfo['name']]
        if sub.empty or row.empty:
            continue
        plotted_any = True
        dia = float(binfo['diameter_um'])
        if xscale == 'log':
            x_pts = dia * rng.uniform(0.90, 1.10, size=len(sub))
        else:
            x_pts = dia + rng.uniform(-0.08, 0.08, size=len(sub)) * dia
        ax.plot(x_pts, sub['xi_um'], marker=binfo['marker'], ls='none', ms=6.0,
                mfc='none', mec=binfo['color'], mew=1.0, alpha=0.8, zorder=3)

        xi_mean = float(row['xi_bg_mean_um'].iloc[0])
        xi_sem = float(row['xi_bg_sem_um'].iloc[0]) if np.isfinite(row['xi_bg_sem_um'].iloc[0]) else 0.0
        if np.isfinite(xi_mean):
            label = (rf"$2R_c = {dia:.2f}\,\mu\mathrm{{m}}$ "
                     rf"($N = {int(row['xi_bg_n_experiments'].iloc[0])}$)")
            ax.errorbar([dia], [xi_mean], yerr=[xi_sem], fmt=binfo['marker'], ms=11.0,
                        color=binfo['color'], mfc=binfo['color'], mec='black', mew=0.9,
                        elinewidth=1.4, capsize=3.5, zorder=5, label=label)
        xi_pooled = float(row['xi_bg_pooled_um'].iloc[0])
        if np.isfinite(xi_pooled):
            ax.plot([dia], [xi_pooled], marker='s', ls='none', ms=6.5, mfc='white',
                    mec=binfo['color'], mew=1.4, alpha=0.9, zorder=6)

    g_mean, g_sem, g_n = global_xi
    if np.isfinite(g_mean):
        g_err = g_sem if np.isfinite(g_sem) else 0.0
        ax.axhspan(g_mean - g_err, g_mean + g_err, color='#333333', alpha=0.10, zorder=1)
        ax.axhline(g_mean, color='#333333', ls='--', lw=1.3, zorder=2,
                   label=rf"All experiments: $\xi_{{\mathrm{{bg}}}} = {g_mean:.2f} \pm {g_err:.2f}\,"
                         rf"\mu\mathrm{{m}}$ ($N = {g_n}$)")

    if not plotted_any:
        print("[WARNING] 有効な xi_bg が無いため図 2 をスキップします")
        plt.close(fig)
        return

    ax.set_xscale(xscale)
    ax.set_xlabel(r"Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]", fontsize=13, fontweight='bold')
    ax.set_ylabel(r"Background Correlation Length $\xi_{\mathrm{bg}}$ [$\mu\mathrm{m}$]",
                  fontsize=13, fontweight='bold')
    ax.set_title(r"Background MT Flow Correlation Length vs Cargo Size",
                 fontsize=13.5, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.35)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9.5, loc='best', framealpha=0.93,
                  title=r"Per-experiment fit ($N$ experiments), pooled-sample fit (open square)",
                  title_fontsize=9.5)
    fig.tight_layout()
    save_figure_to_all(fig, 'bg_angular_correlation_xi_vs_diameter', out_dirs)
    plt.close(fig)


def plot_par_perp_panels(
    df_curves: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    xscale: str = 'log',
    max_dist: float = 60.0,
    ylim: Tuple[float, float] = (-0.35, 1.05),
) -> None:
    """条件ごとのネマチック主軸分解（total / parallel / perpendicular）をパネル表示する。"""
    conds = [b for b in BEADS_INFO if not df_curves[df_curves['bead_name'] == b['name']].empty]
    if not conds:
        print("[WARNING] par/perp データが空のため図 3 をスキップします")
        return

    ncols = 3
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, binfo in enumerate(conds):
        ax = axes[i]
        sub = df_curves[df_curves['bead_name'] == binfo['name']].sort_values('distance_um')
        ax.plot(sub['distance_um'], sub['mean_c'], color='#333333', ls=':', lw=1.3, label=r"Total $C_{\mathrm{bg}}(r)$")
        ax.errorbar(sub['distance_um'], sub['mean_c_par'], yerr=sub['sem_c_par'],
                    color='#1b9e77', fmt='o', ms=3.0, lw=1.2, capsize=2.0, alpha=0.9,
                    label=r"$\parallel$ nematic axis")
        ax.errorbar(sub['distance_um'], sub['mean_c_perp'], yerr=sub['sem_c_perp'],
                    color='#e7298a', fmt='^', ms=3.0, lw=1.2, capsize=2.0, alpha=0.9,
                    label=r"$\perp$ nematic axis")

        row = df_summary[df_summary['bead_name'] == binfo['name']]
        if not row.empty and np.isfinite(row['xi_bg_mean_um'].iloc[0]):
            title = (rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$ "
                     rf"($\xi_{{\mathrm{{bg}}}} = {row['xi_bg_mean_um'].iloc[0]:.1f}\,"
                     rf"\pm {row['xi_bg_sem_um'].iloc[0]:.1f}\,\mu\mathrm{{m}}$, "
                     rf"$N_{{\mathrm{{exp}}}} = {int(row['n_experiments'].iloc[0])}$)")
        else:
            title = rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$"
        ax.set_title(title, fontsize=10.5, fontweight='bold')
        ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xscale(xscale)
        ax.set_xlim(float(max(np.nanmin(df_curves['distance_um']) * 0.8, 1e-3)) if xscale == 'log' else 0.0,
                    max_dist)
        ax.set_ylim(*ylim)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel(r"$C_{\mathrm{bg}}(r)$ (nematic decomposition)", fontsize=11)
        if i == 0:
            ax.legend(fontsize=8.0, loc='upper right', framealpha=0.9)

    for j in range(len(conds), axes.size):
        axes[j].axis('off')

    fig.suptitle(r"Background MT Flow Spatial Correlation: Nematic Axis Decomposition "
                 r"(Cargo Vicinity Excluded)", fontsize=13.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_to_all(fig, 'bg_angular_correlation_par_perp', out_dirs)
    plt.close(fig)


# =========================================================================
# メイン処理
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Background (cargo-vicinity excluded) MT flow spatial orientational correlation analysis "
                     "using random virtual particle (control point) sampling.")
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figure/bg_angular_correlation',
                        help="図・CSV の出力先（相対パスはカレントディレクトリ基準）.")
    parser.add_argument('--no_save_root', action='store_true',
                        help="データルート側 (<root_dir>/figure/bg_angular_correlation) への保存を行わない.")
    parser.add_argument('--beads', type=str, nargs='+', default=['all'],
                        help="対象ビーズ条件 (e.g. 'all', 'beads3um beads7um', 'beads06um,beads1um').")

    # --- データ源 ---
    parser.add_argument('--source', type=str, default='compute', choices=['compute', 'existing'],
                        help="'compute': GFP_flows.h5 から仮想粒子サンプリングを計算（既定, キャッシュ利用）; "
                             "'existing': 既存の angular_correlation_bg.zarr（仮想粒子型）を優先して使う.")
    parser.add_argument('--cache_name', type=str, default='angular_correlation_bg_vp.zarr',
                        help="計算結果のキャッシュ zarr 名（実験ディレクトリ内に保存）.")
    parser.add_argument('--existing_zarr_name', type=str, default='angular_correlation_bg.zarr',
                        help="--source existing で読み込む既存 zarr 名.")
    parser.add_argument('--force_recompute', action='store_true',
                        help="キャッシュ zarr を無視して必ず再計算する.")
    parser.add_argument('--no_cache', action='store_true', help="計算結果のキャッシュ保存を行わない.")

    # --- 仮想粒子サンプリング / FFT 計算パラメータ ---
    parser.add_argument('--distances', '--windows', dest='distances', type=str, nargs='+',
                        default=['2:100:2', '120:500:20'],
                        help="距離 r（px）。'start:stop:step' 形式または数値の羅列.")
    parser.add_argument('--kernel_type', type=str, default='ring', choices=['ring', 'disk', 'gaussian'],
                        help="リング平均カーネル（既定 ring）.")
    parser.add_argument('--shell_width', type=float, default=2.0, help="ring カーネルのシェル幅 (px).")
    parser.add_argument('--n_virtual_points', type=int, default=50,
                        help="フレームあたりにサンプリングする仮想粒子（コントロール点）数（既定 50）.")
    parser.add_argument('--mask_radius_factor', type=float, default=2.0,
                        help="貨物粒子近傍除外半径 = factor x R_c [px]. 既定 2.0.")
    parser.add_argument('--min_mask_radius_px', type=float, default=15.0,
                        help="貨物粒子近傍除外半径の下限 [px]（小さい粒子での下限。既定 15）.")
    parser.add_argument('--border_margin_px', type=float, default=None,
                        help="仮想粒子サンプリングを禁止する画像境界マージン [px]（既定 = max(--distances)）.")
    parser.add_argument('--min_flow_mag', type=float, default=1e-4,
                        help="フロー有効判定の閾値（|v| > この値の画素のみ有効. 既定 1e-4）.")
    parser.add_argument('--seed', type=int, default=42, help="仮想粒子サンプリングの乱数シード（再現性）.")
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', 'torch_cpu', 'scipy'],
                        help="FFT 計算バックエンド（既定: GPU 自動判定）.")
    parser.add_argument('--frame_stride', type=int, default=1, help="使用フレームの間引き（既定 1 = 全フレーム）.")
    parser.add_argument('--max_frames', type=int, default=None,
                        help="（デバッグ用）各実験で先頭 N フレームのみを使用.")

    # --- 解析・作図 ---
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--fit_range', type=float, nargs=2, default=[0.0, 20.0], metavar=('MIN', 'MAX'),
                        help="xi_bg フィットに使う距離範囲 [um]（既定 0 20）.")
    parser.add_argument('--min_corr_threshold', type=float, default=0.01,
                        help="対数をとる C_bg の下限閾値（既定 0.01）.")
    parser.add_argument('--max_dist', type=float, default=60.0, help="C_bg(r) 図の横軸上限 [um].")
    parser.add_argument('--xscale', type=str, default='log', choices=['log', 'linear'],
                        help="C_bg(r) 図の横軸スケール（既定 log）.")
    parser.add_argument('--xi_xscale', type=str, default='log', choices=['log', 'linear'],
                        help="xi_bg vs 2R_c 図の横軸スケール（既定 log）.")
    parser.add_argument('--ylim', type=float, nargs=2, default=[-0.2, 1.05], metavar=('MIN', 'MAX'),
                        help="C_bg(r) 図の縦軸範囲（既定 -0.2 1.05）.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    fit_range = (float(args.fit_range[0]), float(args.fit_range[1]))
    ylim = (float(args.ylim[0]), float(args.ylim[1]))

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else find_default_root()
    if root_dir is None or not Path(root_dir).exists():
        raise FileNotFoundError("Data root directory not found. Please specify it with --root_dir.")

    out_arg = Path(args.output_dir).expanduser()
    out_dirs: List[Path] = [out_arg if out_arg.is_absolute() else (CURRENT_DIR / out_arg)]
    if not args.no_save_root:
        out_dirs.append(Path(root_dir) / 'figure' / 'bg_angular_correlation')
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    distances = parse_distances(args.distances)
    target_beads = parse_target_beads(args.beads, BEADS_INFO)
    if not target_beads:
        raise RuntimeError("No target bead conditions selected.")

    print("=" * 78)
    print(" Background (Cargo-Vicinity Excluded) MT Flow Spatial Correlation Analysis")
    print("=" * 78)
    print(f"Data Root Directory : {root_dir}")
    print(f"Output Directories  : {', '.join(str(d) for d in out_dirs)}")
    print(f"Target Beads        : {[b['name'] for b in target_beads]}")
    print(f"Distances (px)      : {len(distances)} values ({min(distances)}-{max(distances)} px)")
    print(f"Virtual points      : {args.n_virtual_points} / frame (seed = {args.seed})")
    print(f"Mask radius         : max({args.min_mask_radius_px:.0f} px, "
          f"{args.mask_radius_factor:.1f} x R_c) / {args.scale} um/px")
    print(f"Source              : {args.source} (cache: {args.cache_name})")
    print(f"Fit range           : {fit_range[0]:.2f} - {fit_range[1]:.2f} um")
    print("-" * 78)

    # --- 各実験の仮想粒子バックグラウンド相関を取得（キャッシュ / 計算） ---
    exp_results: List[dict] = []
    for binfo in target_beads:
        exp_dirs = find_experiment_dirs(root_dir, binfo['name'])
        print(f"[{binfo['name']}] {len(exp_dirs)} experiment dir(s) with {FLOW_NAME}")
        for exp_dir in exp_dirs:
            res = load_experiment_correlation(exp_dir, binfo, args, distances)
            if res is None:
                continue
            exp_results.append(res)
            print(f"      -> {res['exp_dir']}: {res['n_frames']} frames x {res['n_virtual_points']} pts "
                  f"({res['source']}), mask = {res['mask_radius_px']:.1f} px", flush=True)

    if not exp_results:
        raise RuntimeError("No background correlation samples were extracted. "
                           "Check --root_dir, --beads, and the presence of GFP_flows.h5.")

    grids = [np.asarray(r['distances_px'], dtype=float) for r in exp_results]
    grid_px = grids[0]
    if not all(g.size == grid_px.size and np.allclose(g, grid_px) for g in grids):
        print("[WARNING] 距離グリッドが実験間で一致しません（先頭実験のグリッドで集計します）")

    # --- 集計（実験 -> 条件） ---
    df_points, df_xi, pool = build_tables(exp_results, fit_range, args.min_corr_threshold)
    meta = {
        'n_exp': len(exp_results),
        'n_virtual_points': int(df_xi['n_virtual_points'].max()),
        'n_frames': int(df_xi['n_frames'].sum()),
        'n_samples': int(df_points.groupby('exp_dir')['n_samples'].max().sum()),
        'mask_min_px': float(df_xi['mask_radius_px'].min()),
        'mask_max_px': float(df_xi['mask_radius_px'].max()),
    }
    df_curves, df_summary = summarize_conditions(df_points, df_xi, pool, target_beads,
                                                 grid_px, fit_range, args.min_corr_threshold)
    global_xi = weighted_global_xi(df_xi)

    # --- CSV 出力 ---
    print("-" * 78)
    save_csv_to_all(df_points, 'bg_angular_correlation_points', out_dirs)
    save_csv_to_all(df_curves, 'bg_angular_correlation_curves', out_dirs)
    save_csv_to_all(df_xi, 'bg_angular_correlation_length_per_experiment', out_dirs)
    save_csv_to_all(df_summary, 'bg_angular_correlation_length_summary', out_dirs)

    # --- 作図 ---
    plot_condition_curves(df_curves, df_points, df_summary, out_dirs, meta,
                          fit_range, args.min_corr_threshold,
                          xscale=args.xscale, max_dist=args.max_dist, ylim=ylim)
    plot_xi_vs_diameter(df_xi, df_summary, out_dirs, global_xi, xscale=args.xi_xscale)
    plot_par_perp_panels(df_curves, df_summary, out_dirs,
                         xscale=args.xscale, max_dist=args.max_dist)

    # --- ログ出力 ---
    print("-" * 78)
    print(" Global statistics")
    print(f"   experiments          : {meta['n_exp']}")
    print(f"   frames (total)       : {meta['n_frames']}")
    print(f"   virtual points       : {meta['n_virtual_points']} / frame")
    print(f"   mask radius          : {meta['mask_min_px']:.1f} - {meta['mask_max_px']:.1f} px")
    print(f"   samples (total)      : {meta['n_samples']:,} (= frames x virtual points x experiments)")
    print(f"   weighted mean xi_bg  : {global_xi[0]:.2f} +/- {global_xi[1]:.2f} um (N = {global_xi[2]} experiments)")
    print()
    print(" Per-experiment xi_bg")
    cols = ['bead_name', 'exp_dir', 'source', 'n_frames', 'n_virtual_points', 'mask_radius_px',
            'n_samples_total', 'xi_um', 'xi_err_um', 'xi_r2_log', 'n_fit_points']
    print(df_xi[[c for c in cols if c in df_xi.columns]].to_string(index=False))
    print()
    print(" Per-condition summary (xi_bg: experiment-level mean +/- SEM, pooled-sample fit)")
    cols2 = ['bead_name', 'diameter_um', 'n_experiments', 'n_samples_total',
             'xi_bg_mean_um', 'xi_bg_sem_um', 'xi_bg_pooled_um', 'xi_bg_pooled_r2_log']
    print(df_summary[[c for c in cols2 if c in df_summary.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()
