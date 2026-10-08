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
  dims = distance x frame x virtual_point）として保存する。キャッシュは「要求より密なサンプリング」
  （仮想粒子数が要求以上、フレーム間引きが要求以下で割り切れる）であれば、
  要求パラメータに合わせて点・フレームを部分抽出して再利用する（統計量を落とさず計算時間を節約。
  --force_recompute で再計算、--no_cache で保存しない）。
- --source existing を指定すると、既存の angular_correlation_bg.zarr（仮想粒子型 = dims に random_point を含むもの）を
  優先して読み込む（例: 0.63 / 1.18 / 3.37 um 条件の既存データ。迅速な確認用）。

【統計量を増やすための設定】
- --n_virtual_points（既定 100 点/フレーム）: フレームあたりの仮想粒子数。既往の 50 点/フレーム設定
  （angular_correlation_bg_vp_s4.zarr）は n が要求未満なので再計算の対象になる。
- --frame_stride（既定 1 = 全フレーム）: 全フレームを使うと NAS 読み出し量が最大になるため、
  --n_workers で実験ディレクトリ単位の並列計算（プロセス並列）を行い、全フレーム条件を実用的な時間で実行できる。
- --max_frames はデバッグ用（先頭 N フレームのみ）。

【解析・誤差評価】
- 実験ごとに全 (frame x virtual point) サンプルの平均から C_bg(r) を求め、縦軸を ln C_bg にとった
  重み付き線形フィット ln C = ln a - r / xi（libs.hmm_flow_correlation.fit_flow_correlation_length、
  hmm_flow_correlation_analysis.py と同一実装）により実験ごとの相関長 xi_bg を算出する
  （--fit_range / --min_corr_threshold）。
- 誤差は 2 通りを常に併記する（--error_mode でフィット重みと図のエラーバーを選択。既定 frame）:
    1. sample SEM: 距離ごとの全サンプル（フレーム x 仮想粒子）間の標準誤差。単純だが
       同一フレーム内の仮想粒子は空間相関を持つため、独立サンプル数を過大評価（誤差を過小評価）する。
    2. frame SEM（frame-block SEM）: まずフレームごとに仮想粒子平均を取り、そのフレーム平均間の標準誤差を
       sqrt(N_frames) で評価する。実効独立サンプル数 ≈ フレーム数とみなす保守的（正直な）誤差。
- 条件（粒子径）ごとに実験間の mean ± SEM を集計し、サンプル数（フレーム x 仮想粒子数 x 実験数）も記録する。
  CSV には sample/frame 両方の SEM と両方の誤差重みによる xi_bg を併記し、図では frame SEM の誤差帯を重ねる。

【出力ファイル】
既定の出力先は (1) <作業ディレクトリ>/figure/bg_angular_correlation と
(2) <root_dir>/figure/bg_angular_correlation の 2 箇所（--no_save_root で (2) を省略, --output_dir で (1) を変更）。
※ 図 1・図 3 の C_bg(r) は横軸 r を linear、縦軸を log（半対数表示）で描画する（--xscale / --yscale で変更可）。
   これにより指数減衰モデル ln C = ln a - r / xi（フィットが実際に線形化している式）が図上で直線として現れる。
1. bg_angular_correlation_Cr.png / .svg            : 条件別 C_bg(r)（実験別生カーブ + 実験間平均 ± SEM（エラーバー）
   + プールした frame-block SEM（帯）+ ln C で線形フィットした指数減衰曲線）
2. bg_angular_correlation_xi_vs_diameter.png / .svg: xi_bg vs 貨物直径 2R_c（実験点 ± フィット誤差 + 条件平均 ± SEM
   + 全体系平均 + プールフィット）
3. bg_angular_correlation_par_perp.png / .svg      : 条件別のネマチック主軸分解（total / parallel / perpendicular,
   各曲線にプールした frame-block SEM の帯を付加。縦軸 log, 横軸 linear）
4. bg_angular_correlation_points.csv               : 実験 x 距離ごとの C_bg(r)（平均・SEM・サンプル数, par/perp 含む）
5. bg_angular_correlation_curves.csv               : 条件 x 距離ごとの平均曲線（実験間平均 ± SEM, プール SEM, サンプル数）
6. bg_angular_correlation_length_per_experiment.csv: 実験ごとの xi_bg（フィット品質・サンプル数付き）
7. bg_angular_correlation_length_summary.csv       : 条件ごとの xi_bg 代表値（mean ± SEM, プールフィット値）
"""

import argparse
import multiprocessing as mp
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# 貨物粒子（ビーズ）およびコントロール条件の定義
BEADS_INFO = [
    {
        "name": "control",
        "dir_name": "control/MTs8uM",
        "display_name": "w/o Cargo",
        "diameter_um": np.nan,
        "radius_um": 0.0,
        "marker": None,
        "color": "black",
        "is_control": True,
    },
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s"},
]
cargo_beads = [b for b in BEADS_INFO if not b.get("is_control", False)]
for _b, _c in zip(cargo_beads, _style_colors(len(cargo_beads))):
    _b["color"] = _c

BEAD_LOOKUP = {b['name']: b for b in BEADS_INFO}


def normalize_bead_name(raw_name: str) -> Optional[str]:
    """
    入力文字列（例: 'control', 'w/o cargo', 'beads06um', '06um', etc.）を
    BEADS_INFO の標準名 ('control', 'beads06um' 等) に正規化する。
    """
    s = raw_name.strip().lower()
    mapping = {
        'control': 'control', 'nocargo': 'control', 'no_cargo': 'control',
        'w/o cargo': 'control', 'w/o_cargo': 'control', 'wo_cargo': 'control',
        'control/mts8um': 'control', 'control_mts8um': 'control',
        'mts8um': 'control', '8um': 'control', 'control_8um': 'control',
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


def find_experiment_dirs(root_dir: Path, bead_name: str, dir_name: Optional[str] = None) -> List[Path]:
    """<root_dir>/<target_rel>/{date/}exp 配下で GFP_flows.h5 を持つ実験ディレクトリを返す。"""
    target_rel = dir_name or bead_name
    base = Path(root_dir) / target_rel
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
    control 等で radius_um <= 0 または非有限値の場合は 0.0 px（マスクなし）を返す。
    """
    if not np.isfinite(radius_um) or radius_um <= 0:
        return 0.0
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


def virtual_point_samples(
    ds: xr.Dataset,
    n_points_limit: Optional[int] = None,
    frame_stride: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int, str]:
    """
    (distance, frame, virtual_point) のサンプル配列・距離座標・仮想粒子数を取り出す。

    要求より密なキャッシュ（仮想粒子数・フレーム数が多い）を再利用できるように、
    n_points_limit / frame_stride を与えると点およびフレームを部分抽出する
    （点は交換可能な i.i.d. サンプルなので先頭 k 点の部分抽出で偏りは生じない）。
    """
    dim = VIRTUAL_POINT_DIM if VIRTUAL_POINT_DIM in ds.sizes else _EXISTING_VP_DIM
    dist_px = np.asarray(ds.coords['distance'].values, dtype=float)
    n_pts = int(ds.sizes[dim])
    k = min(n_pts, int(n_points_limit)) if n_points_limit else n_pts

    frames_all = np.asarray(ds.coords['frame'].values, dtype=np.int64)
    if frame_stride is not None and int(frame_stride) > 1:
        sel_frames = np.nonzero(np.mod(frames_all, int(frame_stride)) == 0)[0]
        if sel_frames.size == 0:
            sel_frames = np.arange(frames_all.size)
    else:
        sel_frames = np.arange(frames_all.size)

    def _get(name: str) -> Optional[np.ndarray]:
        if name not in ds:
            return None
        arr = ds[name]
        try:
            arr = arr.transpose('distance', 'frame', dim)
        except Exception:
            return None
        vals = np.asarray(arr.values)
        if sel_frames.size != frames_all.size:
            vals = vals[:, sel_frames, :]
        if k < n_pts:
            vals = vals[..., :k]
        return np.asarray(vals, dtype=np.float32)

    return (_get('angular_correlation'), _get('angular_correlation_parallel'),
            _get('angular_correlation_perpendicular'), dist_px, k, dim)


def cache_reuse_plan(
    ds: xr.Dataset,
    distances: Sequence[float],
    kernel_type: str,
    shell_width: float,
    n_virtual_points: int,
    mask_radius_px: float,
    border_margin_px: int,
    min_flow_mag: float,
    frame_stride: int = 1,
) -> Optional[Dict[str, int]]:
    """
    キャッシュ zarr を要求パラメータでどのように再利用できるかを判定する。

    距離グリッド・カーネル種別・シェル幅・マスク半径・境界マージン・フロー閾値が一致し、かつ
    キャッシュのサンプリング密度が要求以上（仮想粒子数 >= 要求、フレーム間引き <= 要求 かつ 割り切れる）
    である場合に、使用する点・フレーム数を表す辞書を返す。条件を満たさない場合は None（= 再計算）。

    Returns
    -------
    plan : dict or None
        {'n_points', 'n_points_cache', 'frame_stride', 'frame_stride_cache'}
    """
    try:
        attrs = ds.attrs
        if str(attrs.get('kernel_type', '')) != str(kernel_type):
            return None
        if abs(float(attrs.get('shell_width', -1.0)) - float(shell_width)) > 1e-6:
            return None
        mask_attr = attrs.get('mask_radius_px', attrs.get('particle_mask_radius', None))
        if mask_attr is None or abs(float(mask_attr) - float(mask_radius_px)) > 1e-6:
            return None
        if int(attrs.get('border_margin_px', -1)) != int(border_margin_px):
            return None
        if abs(float(attrs.get('min_flow_mag', -1.0)) - float(min_flow_mag)) > 1e-12:
            return None
        if not np.allclose(np.asarray(ds.coords['distance'].values, dtype=float),
                           np.asarray(distances, dtype=float)):
            return None

        dim = VIRTUAL_POINT_DIM if VIRTUAL_POINT_DIM in ds.sizes else _EXISTING_VP_DIM
        n_pts_cache = int(attrs.get('n_virtual_points', attrs.get('n_random_points', ds.sizes[dim])))
        stride_cache = max(1, int(attrs.get('frame_stride', 1) or 1))
        req_pts = max(1, int(n_virtual_points))
        req_stride = max(1, int(frame_stride))
        if n_pts_cache < req_pts:
            return None
        if stride_cache > req_stride or (req_stride % stride_cache) != 0:
            return None
        return {
            'n_points': req_pts,
            'n_points_cache': n_pts_cache,
            'frame_stride': req_stride,
            'frame_stride_cache': stride_cache,
        }
    except Exception:
        return None


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
    """（後方互換）キャッシュが要求パラメータと一致するかを判定する。"""
    return cache_reuse_plan(ds, distances, kernel_type, shell_width, n_virtual_points,
                            mask_radius_px, border_margin_px, min_flow_mag, 1) is not None



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
        u_sq_list: List[float] = []

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

            val_cnt = np.sum(valid)
            if val_cnt > 50:
                mean_ux = float(np.sum(m_ux) / val_cnt)
                mean_uy = float(np.sum(m_uy) / val_cnt)
                u_sq_list.append(mean_ux ** 2 + mean_uy ** 2)

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
    mean_u_sq_attr = float(np.mean(u_sq_list)) if u_sq_list else 0.0
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
                np.asarray([float(thetas[t]) if t < len(thetas) else 0.0 for t in frames],
                           dtype=np.float32),
                dims=['frame'], coords={'frame': frame_coord}),
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
            'mean_u_bar_sq': mean_u_sq_attr,
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


def h5_num_frames(exp_dir: Path, flow_name: str = FLOW_NAME) -> int:
    """GFP_flows.h5 の総フレーム数を返す（読めない場合は -1）。"""
    path = Path(exp_dir) / flow_name
    if not path.exists():
        return -1
    try:
        with h5py.File(str(path), 'r') as f:
            key = list(f.keys())[0]
            return int(f[key].shape[0])
    except Exception:
        return -1


def compute_mean_u_bar_sq(
    exp_dir: Path,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    min_flow_mag: float = 1e-4,
    max_sample_frames: int = 50,
    cached_val: Optional[float] = None,
) -> float:
    """
    GFP_flows.h5 から各フレームの空間平均配向ベクトル u_bar(t) を求め、
    その自乗の時間平均 <|u_bar(t)|^2>_t を高速に算出する（HDF5一括読み込み & 最大 max_sample_frames で等間隔サンプリング）。
    """
    if cached_val is not None and np.isfinite(cached_val) and cached_val >= 0:
        return float(cached_val)

    exp_dir = Path(exp_dir)
    flow_path = exp_dir / FLOW_NAME
    if not flow_path.exists():
        return 0.0

    try:
        with h5py.File(str(flow_path), 'r') as f:
            key = list(f.keys())[0]
            flow = f[key]
            num_frames = flow.shape[0]
            channel_first = not (flow.shape[-1] == 2)

            stride = max(1, int(frame_stride))
            frames = list(range(0, num_frames, stride))
            if max_frames is not None:
                frames = frames[:int(max_frames)]
            if not frames:
                return 0.0

            # 高速化: 最大 max_sample_frames フレームで等間隔サンプリング
            if len(frames) > max_sample_frames:
                step = max(1, len(frames) // max_sample_frames)
                frames = frames[::step][:max_sample_frames]

            # HDF5から一括読み込み（NAS I/Oのレイテンシを最小化）
            data = flow[frames]
            if channel_first:
                vx = data[:, 0].astype(np.float32)
                vy = data[:, 1].astype(np.float32)
            else:
                vx = data[..., 0].astype(np.float32)
                vy = data[..., 1].astype(np.float32)

            vmag = np.hypot(vx, vy)
            val = vmag > float(min_flow_mag)

            u_sq_list = []
            for i in range(len(frames)):
                m = val[i]
                cnt = int(np.sum(m))
                if cnt > 50:
                    ux = float(np.sum(vx[i][m] / vmag[i][m])) / cnt
                    uy = float(np.sum(vy[i][m] / vmag[i][m])) / cnt
                    u_sq_list.append(ux**2 + uy**2)
            return float(np.mean(u_sq_list)) if u_sq_list else 0.0
    except Exception as e:
        print(f"[WARNING] Failed to compute <|u_bar(t)|^2> in {exp_dir.name}: {e}")
        return 0.0


def load_experiment_correlation(
    exp_dir: Path,
    binfo: dict,
    args: argparse.Namespace,
    distances: Sequence[float],
    verbose: bool = True,
) -> Optional[dict]:
    """
    1 実験の仮想粒子バックグラウンド相関サンプルを取得し、
    大域配向ゆらぎの二乗時間平均 <|u_bar(t)|^2>_t によるベースライン補正:
    C_corrected(r) = (C_raw(r) - <|u_bar|^2>) / (1 - <|u_bar|^2>)
    を適用して返す。
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
    cached_u_sq: Optional[float] = None

    if args.source == 'existing':
        ds_ex = open_virtual_point_dataset(exp_dir / args.existing_zarr_name)
        if ds_ex is not None:
            out = virtual_point_samples(ds_ex)
            mask_attr = ds_ex.attrs.get('particle_mask_radius', ds_ex.attrs.get('mask_radius_px', None))
            cached_u_sq = ds_ex.attrs.get('mean_u_bar_sq', None)
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
                plan = cache_reuse_plan(ds_c, distances, args.kernel_type, args.shell_width,
                                        int(args.n_virtual_points), mask_px, border_px,
                                        args.min_flow_mag, int(args.frame_stride))
                out = None
                if plan is not None:
                    out = virtual_point_samples(ds_c, n_points_limit=plan['n_points'],
                                                frame_stride=plan['frame_stride'])
                    cached_u_sq = ds_c.attrs.get('mean_u_bar_sq', None)
                ds_c.close()
                if plan is not None and out is not None and out[0] is not None:
                    samples, par, perp, dist_px, n_pts, _ = out
                    subset = ''
                    if (plan['n_points_cache'] > plan['n_points']
                            or plan['frame_stride_cache'] < plan['frame_stride']):
                        subset = (f" [subsampled from {plan['n_points_cache']}pts/"
                                  f"stride{plan['frame_stride_cache']}]")
                    source = f"cache:{args.cache_name}:{n_pts}pts{subset}"
                    print(f"    [cache] {exp_dir.name}: dims = ({len(dist_px)}, "
                          f"{samples.shape[1]}, {n_pts}){subset}")

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
        cached_u_sq = ds.attrs.get('mean_u_bar_sq', None)
        if not args.no_cache:
            save_dataset_cache(ds, exp_dir / args.cache_name)
        out = virtual_point_samples(ds)
        ds.close()
        if out is None or out[0] is None:
            return None
        samples, par, perp, dist_px, n_pts, _ = out
        source = f"computed:{int(args.n_virtual_points)}pts"

    if args.max_frames is None and args.source != 'existing':
        n_total = h5_num_frames(exp_dir)
        stride = max(1, int(args.frame_stride))
        expect = len(range(0, n_total, stride)) if n_total > 0 else -1
        if expect > 0 and samples.shape[1] < expect:
            print(f"    [WARNING] {exp_dir.name}: 利用可能なフレーム数 {samples.shape[1]} < 期待値 {expect} "
                  f"(--frame_stride {stride})。全フレームで再計算するには "
                  f"--force_recompute を指定してください")

    # 大域配向ゆらぎの二乗時間平均 <|u_bar(t)|^2>_t によるベースライン補正:
    # C_corrected(r) = (C_raw(r) - <|u_bar|^2>) / (1 - <|u_bar|^2>)
    u_bar_sq = compute_mean_u_bar_sq(exp_dir, frame_stride=args.frame_stride,
                                     max_frames=args.max_frames, min_flow_mag=args.min_flow_mag,
                                     cached_val=cached_u_sq)
    denom = max(1.0 - u_bar_sq, 1e-4)
    samples_raw = np.copy(samples) if samples is not None else None
    samples_corr = (samples - u_bar_sq) / denom if samples is not None else None
    par_corr = (par - u_bar_sq) / denom if par is not None else None
    perp_corr = (perp - u_bar_sq) / denom if perp is not None else None

    return {
        'bead_name': binfo['name'],
        'diameter_um': float(binfo['diameter_um']),
        'radius_um': float(binfo['radius_um']),
        'exp_dir': exp_dir.name,
        'exp_path': str(exp_dir),
        'distances_px': np.asarray(dist_px, dtype=float),
        'distances_um': np.asarray(dist_px, dtype=float) * float(args.scale),
        'samples': samples_corr,
        'samples_raw': samples_raw,
        'samples_par': par_corr,
        'samples_perp': perp_corr,
        'u_bar_sq_mean': float(u_bar_sq),
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


def frame_block_stats(samples: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (distance, frame, virtual_point) サンプル配列 -> 距離ごとのフレームブロック統計。

    まずフレームごとに仮想粒子平均を取り（frame mean）、そのフレーム平均間の標準誤差を
    sqrt(N_frames) で評価する。同一フレーム内の仮想粒子は空間相関をもつため、
    サンプル単位の SEM（summarize_samples）は誤差を過小評価する。
    実効独立サンプル数 ≈ フレーム数 とみなすこの評価が統計的に正直な誤差である。

    Returns
    -------
    mean_frame : np.ndarray  フレーム平均の平均（= 距離ごとの C_bg(r)）
    sem_frame  : np.ndarray  フレーム平均間の SEM
    n_frames   : np.ndarray  有効フレーム数
    """
    if samples is None:
        return np.array([]), np.array([]), np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        frame_mean = np.nanmean(samples, axis=2)               # (distance, frame)
        n_frames = np.sum(np.isfinite(frame_mean), axis=1).astype(int)
        mean_frame = np.nanmean(frame_mean, axis=1)
        std_frame = np.nanstd(frame_mean, axis=1, ddof=1)
    sem_frame = np.where(n_frames > 1, std_frame / np.sqrt(np.maximum(n_frames, 1)), 0.0)
    return (np.asarray(mean_frame, dtype=float), np.asarray(sem_frame, dtype=float),
            np.asarray(n_frames, dtype=int))


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
    error_mode: str = 'frame',
) -> Tuple[pd.DataFrame, pd.DataFrame,
           Dict[Tuple[str, str, int], List[float]],
           Dict[Tuple[str, str, int], List[float]]]:
    """
    実験ごとの C_bg(r) 表（points）と xi_bg 表（xi）を作成し、
    条件ごとのプール統計を同時に蓄積して返す。

    - pool       : 全サンプル（フレーム x 仮想粒子）の (n, sum, sumsq)
    - pool_frame : フレーム平均の (n, sum, sumsq)（実効独立サンプル数 ≈ フレーム数）

    error_mode ('frame' | 'sample') は xi_bg フィットの重み（sigma_ln C = SEM / C）に使う誤差を選ぶ。
    どちらのモードでも両方の誤差による xi_bg を算出し CSV に併記する。
    """
    point_rows: List[dict] = []
    xi_rows: List[dict] = []
    pool: Dict[Tuple[str, str, int], List[float]] = {}
    pool_frame: Dict[Tuple[str, str, int], List[float]] = {}

    def _accum(target: Dict[Tuple[str, str, int], List[float]], kind: str, bead: str,
               values_2d: Optional[np.ndarray]) -> None:
        """distance x N の 2 次元配列について (n, sum, sumsq) を蓄積する。"""
        if values_2d is None or values_2d.size == 0:
            return
        for i in range(values_2d.shape[0]):
            row = values_2d[i]
            v = row[np.isfinite(row)]
            if v.size == 0:
                continue
            acc = target.setdefault((kind, bead, i), [0.0, 0.0, 0.0])
            v64 = v.astype(np.float64)
            acc[0] += float(v.size)
            acc[1] += float(np.sum(v64))
            acc[2] += float(np.sum(v64 ** 2))

    def _accum_samples(kind: str, bead: str, samples: Optional[np.ndarray]) -> None:
        if samples is None:
            return
        _accum(pool, kind, bead, samples.reshape(samples.shape[0], -1))

    def _accum_frame(kind: str, bead: str, samples: Optional[np.ndarray]) -> None:
        if samples is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            frame_mean = np.nanmean(samples, axis=2)  # (distance, frame)
        _accum(pool_frame, kind, bead, frame_mean)

    for res in exp_results:
        bead = res['bead_name']
        r_um = res['distances_um']
        u_bar_sq = float(res.get('u_bar_sq_mean', 0.0))
        mean_c, sem_c, n_c = summarize_samples(res['samples'])
        mean_c_raw, sem_c_raw, _ = summarize_samples(res.get('samples_raw'))
        mean_cf, sem_cf, n_frames_c = frame_block_stats(res['samples'])
        mean_par, sem_par, _ = summarize_samples(res['samples_par'])
        mean_perp, sem_perp, _ = summarize_samples(res['samples_perp'])
        _, sem_parf, _ = frame_block_stats(res['samples_par'])
        _, sem_perpf, _ = frame_block_stats(res['samples_perp'])
        _accum_samples('total', bead, res['samples'])
        _accum_samples('par', bead, res['samples_par'])
        _accum_samples('perp', bead, res['samples_perp'])
        _accum_frame('total', bead, res['samples'])
        _accum_frame('par', bead, res['samples_par'])
        _accum_frame('perp', bead, res['samples_perp'])

        # 誤差モードに応じた主フィット ＋ 両モードの比較用フィット
        sem_primary = sem_cf if str(error_mode) == 'frame' else sem_c
        fit = fit_xi_from_curve(r_um, mean_c, sem_primary, fit_range[0], fit_range[1],
                                min_corr_threshold)
        fit_sample = (fit if str(error_mode) == 'sample'
                      else fit_xi_from_curve(r_um, mean_c, sem_c, fit_range[0], fit_range[1],
                                             min_corr_threshold))
        fit_frame = (fit if str(error_mode) == 'frame'
                     else fit_xi_from_curve(r_um, mean_c, sem_cf, fit_range[0], fit_range[1],
                                            min_corr_threshold))

        for i, r in enumerate(r_um):
            point_rows.append({
                'bead_name': bead,
                'diameter_um': res['diameter_um'],
                'exp_dir': res['exp_dir'],
                'source': res['source'],
                'u_bar_sq_mean': u_bar_sq,
                'distance_px': float(res['distances_px'][i]),
                'distance_um': float(r),
                'mean_c': float(mean_c[i]) if i < mean_c.size else np.nan,
                'sem_c': float(sem_c[i]) if i < sem_c.size else np.nan,
                'mean_c_raw': float(mean_c_raw[i]) if i < mean_c_raw.size else np.nan,
                'sem_c_raw': float(sem_c_raw[i]) if i < sem_c_raw.size else np.nan,
                'n_samples': int(n_c[i]) if i < n_c.size else 0,
                'mean_c_frame': float(mean_cf[i]) if i < mean_cf.size else np.nan,
                'sem_c_frame': float(sem_cf[i]) if i < sem_cf.size else np.nan,
                'n_frames_used': int(n_frames_c[i]) if i < n_frames_c.size else 0,
                'mean_c_par': float(mean_par[i]) if i < mean_par.size else np.nan,
                'sem_c_par': float(sem_par[i]) if i < sem_par.size else np.nan,
                'sem_c_par_frame': float(sem_parf[i]) if i < sem_parf.size else np.nan,
                'mean_c_perp': float(mean_perp[i]) if i < mean_perp.size else np.nan,
                'sem_c_perp': float(sem_perp[i]) if i < sem_perp.size else np.nan,
                'sem_c_perp_frame': float(sem_perpf[i]) if i < sem_perpf.size else np.nan,
            })

        xi_rows.append({
            'bead_name': bead,
            'diameter_um': res['diameter_um'],
            'exp_dir': res['exp_dir'],
            'source': res['source'],
            'u_bar_sq_mean': u_bar_sq,
            'n_frames': res['n_frames'],
            'n_virtual_points': res['n_virtual_points'],
            'mask_radius_px': res['mask_radius_px'],
            'n_samples_total': int(np.sum(np.any(np.isfinite(res['samples']), axis=0))),
            'xi_um': float(fit['xi_um']),
            'xi_err_um': float(fit['xi_err_um']),
            'xi_r2_log': float(fit['r2_log']),
            'xi_r2': float(fit['r2']),
            'n_fit_points': int(fit['n_fit_points']),
            'r_peak_um': float(fit['r_peak_um']),
            'fit_min_um': float(fit['r_fit_min_um']),
            'fit_max_um': float(fit['r_fit_max_um']),
            'error_mode': str(error_mode),
            'xi_um_sample_sem': float(fit_sample['xi_um']),
            'xi_err_um_sample_sem': float(fit_sample['xi_err_um']),
            'xi_r2_log_sample_sem': float(fit_sample['r2_log']),
            'xi_um_frame_sem': float(fit_frame['xi_um']),
            'xi_err_um_frame_sem': float(fit_frame['xi_err_um']),
            'xi_r2_log_frame_sem': float(fit_frame['r2_log']),
        })

    return pd.DataFrame(point_rows), pd.DataFrame(xi_rows), pool, pool_frame


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
    pool_frame: Optional[Dict[Tuple[str, str, int], List[float]]] = None,
    error_mode: str = 'frame',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    条件（粒子径）ごとに C_bg(r) の実験間平均 ± SEM・プール平均 ± SEM・サンプル数を集計し、
    条件平均曲線とプール曲線をそれぞれフィットして xi_bg を求める。

    プール曲線は 2 通りを併記する:
      - サンプル単位プール（pool）: 全サンプルの平均 ± サンプル SEM
      - フレームブロックプール（pool_frame）: フレーム平均の平均 ± フレーム平均間 SEM
        （実効独立サンプル数 ≈ フレーム数。誤差帯の描画に用いる）
    error_mode ('frame' | 'sample') で主フィット（xi_bg_pooled_um）の誤差重みを選ぶ。
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
        pooled_mean_f = np.full(n_d, np.nan)
        pooled_sem_f = np.full(n_d, np.nan)
        pooled_n_frames = np.zeros(n_d, dtype=int)
        pooled_par_f = np.full(n_d, np.nan)
        pooled_sem_par_f = np.full(n_d, np.nan)
        pooled_perp_f = np.full(n_d, np.nan)
        pooled_sem_perp_f = np.full(n_d, np.nan)

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
            pp_par = pool_stat(pool, 'par', bead, i)
            pp_perp = pool_stat(pool, 'perp', bead, i)
            if pool_frame is not None:
                fm, fs, fn = pool_stat(pool_frame, 'total', bead, i)
                pooled_mean_f[i], pooled_sem_f[i], pooled_n_frames[i] = fm, fs, fn
                pfp = pool_stat(pool_frame, 'par', bead, i)
                pooled_par_f[i], pooled_sem_par_f[i] = pfp[0], pfp[1]
                pfp = pool_stat(pool_frame, 'perp', bead, i)
                pooled_perp_f[i], pooled_sem_perp_f[i] = pfp[0], pfp[1]

        fit_cond = fit_xi_from_curve(grid, exp_mean, exp_sem, fit_range[0], fit_range[1], min_corr_threshold)
        fit_pooled_sample = fit_xi_from_curve(grid, pooled_mean, pooled_sem, fit_range[0], fit_range[1],
                                              min_corr_threshold)
        has_frame_pool = bool(pool_frame is not None and np.any(np.isfinite(pooled_mean_f)))
        fit_pooled_frame = (fit_xi_from_curve(grid, pooled_mean_f, pooled_sem_f, fit_range[0],
                                              fit_range[1], min_corr_threshold)
                            if has_frame_pool else fit_pooled_sample)
        fit_pooled = (fit_pooled_frame if (str(error_mode) == 'frame' and has_frame_pool)
                      else fit_pooled_sample)

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
                'pooled_mean_c_par': float(pp_par[0]),
                'pooled_sem_c_par': float(pp_par[1]),
                'pooled_mean_c_perp': float(pp_perp[0]),
                'pooled_sem_c_perp': float(pp_perp[1]),
                'pooled_mean_c_frame': float(pooled_mean_f[i]),
                'pooled_sem_c_frame': float(pooled_sem_f[i]),
                'pooled_n_frames': int(pooled_n_frames[i]),
                'pooled_mean_c_par_frame': float(pooled_par_f[i]),
                'pooled_sem_c_par_frame': float(pooled_sem_par_f[i]),
                'pooled_mean_c_perp_frame': float(pooled_perp_f[i]),
                'pooled_sem_c_perp_frame': float(pooled_sem_perp_f[i]),
                'error_mode': str(error_mode),
                'xi_bg_cond_fit_um': float(fit_cond['xi_um']),
                'xi_bg_cond_fit_err_um': float(fit_cond['xi_err_um']),
                'xi_bg_cond_fit_r2_log': float(fit_cond['r2_log']),
                'xi_bg_pooled_um': float(fit_pooled['xi_um']),
                'xi_bg_pooled_err_um': float(fit_pooled['xi_err_um']),
                'xi_bg_pooled_r2_log': float(fit_pooled['r2_log']),
                'xi_bg_pooled_sample_um': float(fit_pooled_sample['xi_um']),
                'xi_bg_pooled_sample_err_um': float(fit_pooled_sample['xi_err_um']),
                'xi_bg_pooled_frame_um': float(fit_pooled_frame['xi_um']),
                'xi_bg_pooled_frame_err_um': float(fit_pooled_frame['xi_err_um']),
            })

        xi_vals = sub_xi['xi_um'].to_numpy(dtype=float)
        xi_vals = xi_vals[np.isfinite(xi_vals)]
        xi_vals_sample = np.array([])
        xi_vals_frame = np.array([])
        if 'xi_um_sample_sem' in sub_xi.columns:
            xi_vals_sample = sub_xi['xi_um_sample_sem'].to_numpy(dtype=float)
            xi_vals_sample = xi_vals_sample[np.isfinite(xi_vals_sample)]
        if 'xi_um_frame_sem' in sub_xi.columns:
            xi_vals_frame = sub_xi['xi_um_frame_sem'].to_numpy(dtype=float)
            xi_vals_frame = xi_vals_frame[np.isfinite(xi_vals_frame)]
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
            'error_mode': str(error_mode),
            'n_frames_pooled': int(np.nanmax(pooled_n_frames)) if pooled_n_frames.size else 0,
            'xi_bg_mean_sample_sem_um': (float(np.mean(xi_vals_sample))
                                         if xi_vals_sample.size else np.nan),
            'xi_bg_sem_sample_sem_um': (float(np.std(xi_vals_sample, ddof=1)
                                              / np.sqrt(xi_vals_sample.size))
                                        if xi_vals_sample.size > 1 else 0.0),
            'xi_bg_mean_frame_sem_um': (float(np.mean(xi_vals_frame))
                                        if xi_vals_frame.size else np.nan),
            'xi_bg_sem_frame_sem_um': (float(np.std(xi_vals_frame, ddof=1)
                                             / np.sqrt(xi_vals_frame.size))
                                        if xi_vals_frame.size > 1 else 0.0),
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


def _log_positive_ylim(
    values: Sequence[float],
    default: Tuple[float, float] = (1e-3, 1.5),
    lower_factor: float = 0.5,
    upper_factor: float = 1.6,
) -> Tuple[float, float]:
    """
    対数軸に使う y 範囲を正の値のみから推定する（0 以下・非有限値は無視）。

    半対数表示（x = linear, y = log）では線形軸用の ylim（負値を含む）が使えないため、
    描画対象データに含まれる正の最小値・最大値に余白を付けて範囲を決める。
    正の値が 1 つも無い場合は default を返す。
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return default
    lo = float(np.min(arr)) * float(lower_factor)
    hi = float(np.max(arr)) * float(upper_factor)
    lo = max(lo, np.finfo(float).tiny)
    if hi <= lo:
        hi = lo * 10.0
    return lo, hi


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
    xscale: str = 'linear',
    yscale: str = 'log',
    max_dist: float = 12.0,
    ylim: Tuple[float, float] = (-0.2, 1.05),
    error_mode: str = 'frame',
) -> None:
    """
    条件（粒子径）ごとの C_bg(r) を 1 軸に描く。

    - 横軸 r は linear、縦軸 C_bg(r) は log（半対数表示）が既定。これにより
      指数減衰フィット ln C = ln a - r / xi（= フィットが実際に線形化しているモデル）が
      図上で直線として現れ、フィットと表示が一致する。--xscale / --yscale で変更可能。
    - 灰色細線: 実験別生カーブ
    - マーカー + エラーバー: 条件平均 ± 実験間 SEM（条件の再現性）
    - 帯: プールしたフレームブロック SEM（全フレーム・全仮想粒子・全実験をプールした
      推定精度。実効独立サンプル数 ≈ フレーム数）
    - 破線: 指数フィット（重み = --error_mode の SEM, 縦軸は ln C で線形フィット）
    """
    if df_curves.empty:
        print("[WARNING] C_bg(r) 曲線データが空のため図 1 をスキップします")
        return

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    exp_dirs = sorted(df_points['exp_dir'].unique())
    n_exp = len(exp_dirs)
    band_labeled = [False]
    fit_y_vals: List[np.ndarray] = []  # 対数 y 軸の範囲推定用に描画したフィット値も集める

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

        is_ctrl = binfo.get('is_control', False) or not np.isfinite(binfo.get('diameter_um', np.nan))
        display_name = binfo.get('display_name', "w/o Cargo" if is_ctrl else rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$")

        if np.isfinite(xi_mean) and np.isfinite(xi_sem) and xi_sem > 0:
            label = (rf"{display_name} "
                     rf"($\xi_{{\mathrm{{bg}}}} = {xi_mean:.1f} \pm {xi_sem:.1f}\,\mu\mathrm{{m}}$)")
        elif np.isfinite(xi_mean):
            label = (rf"{display_name} "
                     rf"($\xi_{{\mathrm{{bg}}}} = {xi_mean:.1f}\,\mu\mathrm{{m}}$)")
        else:
            label = display_name

        marker = binfo.get('marker')
        if marker:
            ax.errorbar(sub['distance_um'], sub['mean_c'], yerr=sub['sem_c'],
                        fmt=marker, ms=7.0, color=binfo['color'],
                        mfc=binfo['color'], mec='black', mew=0.8,
                        elinewidth=1.1, capsize=2.5, lw=1.5, alpha=0.95, zorder=4, label=label)
        else:
            ax.errorbar(sub['distance_um'], sub['mean_c'], yerr=sub['sem_c'],
                        fmt='none', color=binfo['color'],
                        elinewidth=1.1, capsize=2.5, alpha=0.95, zorder=4)
            ax.plot(sub['distance_um'], sub['mean_c'], ls='-', color=binfo['color'],
                    lw=1.6, alpha=0.95, zorder=4, label=label)

        # プールしたフレームブロック SEM の誤差帯（実効独立サンプル数 ≈ フレーム数）
        if 'pooled_sem_c_frame' in sub.columns:
            band = sub['pooled_sem_c_frame'].to_numpy(dtype=float)
            mid = sub['mean_c'].to_numpy(dtype=float)
            ok = np.isfinite(band) & np.isfinite(mid)
            if np.any(ok):
                ax.fill_between(sub['distance_um'].to_numpy(dtype=float)[ok],
                                mid[ok] - band[ok], mid[ok] + band[ok],
                                color=binfo['color'], alpha=0.18, lw=0, zorder=2,
                                label=("Frame-block SEM band (pooled, $N_{\\mathrm{eff}}"
                                       " = N_{\\mathrm{frames}}$)" if not band_labeled[0] else None))
                band_labeled[0] = True

        fit = fit_xi_from_curve(sub['distance_um'].to_numpy(dtype=float),
                                sub['mean_c'].to_numpy(dtype=float),
                                sub['sem_c'].to_numpy(dtype=float),
                                fit_range[0], fit_range[1], min_corr_threshold)
        fit_r = np.asarray(fit.get('fit_r', np.array([])), dtype=float)
        fit_c = np.asarray(fit.get('fit_c', np.array([])), dtype=float)
        if fit_r.size and np.isfinite(fit.get('xi_um', np.nan)):
            ax.plot(fit_r, fit_c, ls='--', color=binfo['color'], lw=1.5, alpha=0.9, zorder=3)
            fit_y_vals.append(fit_c)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if str(yscale) == 'log':
        # 半対数表示: 0 線（負値）は描けず、線形用 ylim（負値を含む）も使えないため、
        # 実際に描画する正の値（条件平均 ± 帯, 実験別生カーブ ± SEM, フィット線）から範囲を推定する。
        y_vals: List[np.ndarray] = list(fit_y_vals)
        for binfo in BEADS_INFO:
            s = df_curves[df_curves['bead_name'] == binfo['name']]
            if s.empty:
                continue
            mid = s['mean_c'].to_numpy(dtype=float)
            y_vals.append(mid)
            band = (s['pooled_sem_c_frame'].to_numpy(dtype=float)
                    if 'pooled_sem_c_frame' in s.columns else np.full_like(mid, np.nan))
            y_vals.append(mid - band)
            y_vals.append(mid + band)
            sem = (s['sem_c'].to_numpy(dtype=float)
                   if 'sem_c' in s.columns else np.full_like(mid, np.nan))
            y_vals.append(mid + sem)
        if not df_points.empty:
            raw = df_points['mean_c'].to_numpy(dtype=float)
            raw_sem = (df_points['sem_c'].to_numpy(dtype=float)
                       if 'sem_c' in df_points.columns else np.zeros_like(raw))
            y_vals.append(raw)
            y_vals.append(raw + raw_sem)
        y_flat = np.concatenate([np.ravel(v) for v in y_vals]) if y_vals else np.array([])
        ax.set_ylim(*_log_positive_ylim(y_flat))
    else:
        ax.axhline(0.0, color='gray', ls='--', lw=0.9, alpha=0.6)
        ax.set_ylim(*ylim)
    x_min = float(max(np.nanmin(df_curves['distance_um']) * 0.8, 1e-3)) if xscale == 'log' else 0.0
    ax.set_xlim(x_min, max_dist)
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
    ax.text(0.03, 0.03, info, transform=ax.transAxes, va='bottom', ha='left', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.88))

    weights_label = "frame-block SEM" if str(error_mode) == 'frame' else "sample SEM"
    ax.legend(fontsize=9.0, loc='upper right', framealpha=0.93,
              title=(r"Fit: $a\exp(-r/\xi_{\mathrm{bg}})$" + "\n"
                     + rf"($\sigma$ = {weights_label})"),
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
    error_mode: str = 'frame',
) -> None:
    """
    xi_bg vs 貨物直径 2R_c（実験点 + 条件平均 ± SEM + 全実験代表値 + プールフィット値）。

    実験点（白抜き）にはフィット誤差、塗りつぶしマーカーは条件平均 ± 実験間 SEM、
    白抜き四角はプール曲線のフィット値（重み = --error_mode の SEM）を示す。
    """
    if df_summary.empty:
        print("[WARNING] xi_bg 集計データが空のため図 2 をスキップします")
        return

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    rng = np.random.default_rng(7)
    plotted_any = False
    plotted_dias: List[float] = []

    for binfo in BEADS_INFO:
        sub = df_xi[(df_xi['bead_name'] == binfo['name']) & np.isfinite(df_xi['xi_um'])]
        row = df_summary[df_summary['bead_name'] == binfo['name']]
        if sub.empty or row.empty:
            continue
        dia = float(binfo.get('diameter_um', np.nan))
        if not np.isfinite(dia):
            # コントロール条件（w/o Cargo）は水平線・帯として描画
            ctrl_mean = float(row['xi_bg_mean_um'].iloc[0])
            ctrl_sem = float(row['xi_bg_sem_um'].iloc[0]) if np.isfinite(row['xi_bg_sem_um'].iloc[0]) else 0.0
            if np.isfinite(ctrl_mean):
                ax.axhspan(ctrl_mean - ctrl_sem, ctrl_mean + ctrl_sem, color='black', alpha=0.08, zorder=1)
                ax.axhline(ctrl_mean, color='black', ls=':', lw=1.4, zorder=2,
                           label=rf"w/o Cargo: $\xi_{{\mathrm{{bg}}}} = {ctrl_mean:.2f} \pm {ctrl_sem:.2f}\,\mu\mathrm{{m}}$ ($N = {len(sub)}$)")
            continue

        plotted_any = True
        plotted_dias.append(dia)
        if xscale == 'log':
            x_pts = dia * rng.uniform(0.90, 1.10, size=len(sub))
        else:
            x_pts = dia + rng.uniform(-0.08, 0.08, size=len(sub)) * dia
        ax.plot(x_pts, sub['xi_um'], marker=binfo['marker'], ls='none', ms=6.0,
                mfc='none', mec=binfo['color'], mew=1.0, alpha=0.8, zorder=3)

        # 実験ごとの xi_bg フィット誤差（回帰の重み = --error_mode の SEM）
        xi_err = sub['xi_err_um'].to_numpy(dtype=float)
        if np.any(np.isfinite(xi_err)):
            ax.errorbar(x_pts, sub['xi_um'], yerr=np.where(np.isfinite(xi_err), xi_err, 0.0),
                        fmt='none', ecolor=binfo['color'], elinewidth=0.9, capsize=2.0,
                        alpha=0.55, zorder=3)

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

    if len(set(plotted_dias)) < 2:
        # 単一条件のみの場合は log 軸だと目盛りラベルが重なって読めないため線形軸にする
        ax.set_xscale('linear')
        d0 = float(plotted_dias[0]) if plotted_dias else 1.0
        ax.set_xticks([d0])
        ax.set_xlim(max(d0 * 0.25, 1e-3), d0 * 1.75)
    else:
        ax.set_xscale(xscale)
    ax.set_xlabel(r"Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]", fontsize=13, fontweight='bold')
    ax.set_ylabel(r"Background Correlation Length $\xi_{\mathrm{bg}}$ [$\mu\mathrm{m}$]",
                  fontsize=13, fontweight='bold')
    ax.set_title(r"Background MT Flow Correlation Length vs Cargo Size",
                 fontsize=13.5, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.35)
    if ax.get_legend_handles_labels()[0]:
        pooled_label = ("open square: pooled fit (frame-block SEM weights)"
                        if str(error_mode) == 'frame'
                        else "open square: pooled fit (sample SEM weights)")
        ax.legend(fontsize=9.5, loc='best', framealpha=0.93,
                  title=("Filled: experiment mean $\\pm$ SEM" + "\n" + pooled_label),
                  title_fontsize=9.5)
    fig.tight_layout()
    save_figure_to_all(fig, 'bg_angular_correlation_xi_vs_diameter', out_dirs)
    plt.close(fig)


def plot_par_perp_panels(
    df_curves: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    xscale: str = 'linear',
    yscale: str = 'log',
    max_dist: float = 12.0,
    ylim: Tuple[float, float] = (-0.15, 1.05),
    error_mode: str = 'frame',
) -> None:
    """
    条件ごとのネマチック主軸分解（total / parallel / perpendicular）をパネル表示する。

    横軸 r は linear、縦軸 C_bg(r) は log（半対数表示）が既定（--xscale / --yscale で変更可能）。
    エラーバーは実験間 SEM、網掛け帯はプールしたフレームブロック SEM（実効独立サンプル数 ≈ フレーム数）
    を示す（error_mode は凡例のフィット重み表記に反映）。
    """
    conds = [b for b in BEADS_INFO if not df_curves[df_curves['bead_name'] == b['name']].empty]
    if not conds:
        print("[WARNING] par/perp データが空のため図 3 をスキップします")
        return

    # 半対数表示では線形用 ylim（負値を含む）が使えないため、全パネル共通の y 範囲を
    # 実際に描画する正の値（total / par / perp の平均 ± 帯, ± SEM）から推定する。
    if str(yscale) == 'log':
        pp_vals: List[np.ndarray] = []
        for binfo in conds:
            s = df_curves[df_curves['bead_name'] == binfo['name']]
            for mid_col, err_col, band_col in (
                ('mean_c', 'sem_c', 'pooled_sem_c_frame'),
                ('mean_c_par', 'sem_c_par', 'pooled_sem_c_par_frame'),
                ('mean_c_perp', 'sem_c_perp', 'pooled_sem_c_perp_frame'),
            ):
                if mid_col not in s.columns:
                    continue
                mid = s[mid_col].to_numpy(dtype=float)
                pp_vals.append(mid)
                band = (s[band_col].to_numpy(dtype=float)
                        if band_col in s.columns else np.full_like(mid, np.nan))
                pp_vals.append(mid - band)
                pp_vals.append(mid + band)
                sem = (s[err_col].to_numpy(dtype=float)
                       if err_col in s.columns else np.full_like(mid, np.nan))
                pp_vals.append(mid + sem)
        pp_flat = np.concatenate([np.ravel(v) for v in pp_vals]) if pp_vals else np.array([])
        panel_ylim = _log_positive_ylim(pp_flat)
    else:
        panel_ylim = tuple(ylim)

    ncols = 3
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows),
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

        # プールしたフレームブロック SEM の帯（total / parallel / perpendicular）
        r_plot = sub['distance_um'].to_numpy(dtype=float)
        for band_col, mid_col, bcolor in (('pooled_sem_c_frame', 'mean_c', '#333333'),
                                          ('pooled_sem_c_par_frame', 'mean_c_par', '#1b9e77'),
                                          ('pooled_sem_c_perp_frame', 'mean_c_perp', '#e7298a')):
            if band_col not in sub.columns or mid_col not in sub.columns:
                continue
            band = sub[band_col].to_numpy(dtype=float)
            mid = sub[mid_col].to_numpy(dtype=float)
            ok = np.isfinite(band) & np.isfinite(mid)
            if np.any(ok):
                ax.fill_between(r_plot[ok], mid[ok] - band[ok], mid[ok] + band[ok],
                                color=bcolor, alpha=0.16, lw=0, zorder=1)

        row = df_summary[df_summary['bead_name'] == binfo['name']]
        is_ctrl = binfo.get('is_control', False) or not np.isfinite(binfo.get('diameter_um', np.nan))
        display_name = binfo.get('display_name', "w/o Cargo" if is_ctrl else rf"$2R_c = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$")
        if not row.empty and np.isfinite(row['xi_bg_mean_um'].iloc[0]):
            title = (rf"{display_name} "
                     rf"($\xi_{{\mathrm{{bg}}}} = {row['xi_bg_mean_um'].iloc[0]:.1f}\,"
                     rf"\pm {row['xi_bg_sem_um'].iloc[0]:.1f}\,\mu\mathrm{{m}}$, "
                     rf"$N_{{\mathrm{{exp}}}} = {int(row['n_experiments'].iloc[0])}$)")
        else:
            title = display_name
        ax.set_title(title, fontsize=10.5, fontweight='bold')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if str(yscale) == 'log':
            ax.set_ylim(*panel_ylim)
        else:
            ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax.set_ylim(*ylim)
        ax.set_xlim(float(max(np.nanmin(df_curves['distance_um']) * 0.8, 1e-3)) if xscale == 'log' else 0.0,
                    max_dist)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel(r"$C_{\mathrm{bg}}(r)$ (nematic decomposition)", fontsize=11)
        if i == 0:
            ax.legend(fontsize=8.0, loc='upper right', framealpha=0.9)

    for j in range(len(conds), axes.size):
        axes[j].axis('off')

    weights_label = "frame-block SEM" if str(error_mode) == 'frame' else "sample SEM"
    fig.suptitle(r"Background MT Flow Spatial Correlation: Nematic Axis Decomposition "
                 r"(Cargo Vicinity Excluded)" + "\n"
                 + rf"(shaded: pooled frame-block SEM; fit weights: {weights_label})",
                 fontsize=13.0, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
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
    parser.add_argument('--n_virtual_points', type=int, default=100,
                        help="フレームあたりにサンプリングする仮想粒子（コントロール点）数（既定 100）. "
                             "キャッシュが要求以上の点を持つ場合は部分抽出して再利用する.")
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
    parser.add_argument('--n_workers', type=int, default=1,
                        help="仮想粒子サンプリング計算のプロセス並列数（実験ディレクトリ単位. 既定 1 = 逐次）. "
                             "全フレーム（--frame_stride 1）で計算する際に有効（例: 4）.")

    # --- 解析・作図 ---
    parser.add_argument('--error_mode', type=str, default='frame', choices=['frame', 'sample'],
                        help="xi_bg フィットの重みに使う誤差（既定 frame）. "
                             "'frame': フレーム平均間の SEM（実効独立サンプル数 ≈ フレーム数, 保守的）/ "
                             "'sample': 全サンプル間の SEM（従来通り, 誤差を過小評価しやすい）.")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--fit_range', type=float, nargs=2, default=[0.0, 20.0], metavar=('MIN', 'MAX'),
                        help="xi_bg フィットに使う距離範囲 [um]（既定 0 20）.")
    parser.add_argument('--min_corr_threshold', type=float, default=0.01,
                        help="対数をとる C_bg の下限閾値（既定 0.01）.")
    parser.add_argument('--max_dist', type=float, default=60.0, help="C_bg(r) 図の横軸上限 [um].")
    parser.add_argument('--xscale', type=str, default='linear', choices=['log', 'linear'],
                        help="C_bg(r) 図の横軸スケール（既定 linear）.")
    parser.add_argument('--yscale', type=str, default='linear', choices=['log', 'linear'],
                        help="C_bg(r) 図の縦軸スケール（既定 log = 半対数表示。フィットは ln C で線形化）.")
    parser.add_argument('--xi_xscale', type=str, default='log', choices=['log', 'linear'],
                        help="xi_bg vs 2R_c 図の横軸スケール（既定 log）.")
    parser.add_argument('--ylim', type=float, nargs=2, default=[-0.1, 1.05], metavar=('MIN', 'MAX'),
                        help="C_bg(r) 図の縦軸範囲（--yscale linear のときのみ適用。既定 -0.1 1.05。"
                             "log 軸では正のデータから自動決定）.")
    return parser


def _bg_cache_worker(payload: dict) -> dict:
    """
    並列ワーカー: 1 実験ディレクトリについてキャッシュの再利用可否を判定し、
    必要なら仮想粒子サンプリング相関を計算してキャッシュ zarr として保存する。

    重い GFP_flows.h5 の読み出し（NAS I/O）をプロセス並列化するため、
    トップレベル関数として定義する（ProcessPoolExecutor から pickle 可能にするため）。
    """
    exp_dir = Path(payload['exp_dir'])
    try:
        cache_name = payload['cache_name']
        cache_path = exp_dir / cache_name
        if cache_path.exists() and not payload.get('force_recompute', False):
            ds_c = open_virtual_point_dataset(cache_path)
            if ds_c is not None:
                plan = cache_reuse_plan(
                    ds_c, payload['distances'], payload['kernel_type'], payload['shell_width'],
                    int(payload['n_virtual_points']), payload['mask_px'], int(payload['border_px']),
                    payload['min_flow_mag'], int(payload['frame_stride']),
                )
                ds_c.close()
                if plan is not None:
                    return {'exp_dir': str(exp_dir), 'status': 'cache', 'source': f"cache:{cache_name}"}

        ds = compute_virtual_point_correlations(
            exp_dir, payload['distances'], int(payload['n_virtual_points']),
            payload['mask_px'], int(payload['border_px']),
            kernel_type=payload['kernel_type'], shell_width=payload['shell_width'],
            seed=int(payload['seed']), device=payload['device'],
            min_flow_mag=payload['min_flow_mag'], max_frames=payload['max_frames'],
            frame_stride=int(payload['frame_stride']), verbose=False,
        )
        if ds is None:
            return {'exp_dir': str(exp_dir), 'status': 'failed', 'source': 'no flow data'}
        if not payload.get('no_cache', False):
            save_dataset_cache(ds, cache_path)
        ds.close()
        return {'exp_dir': str(exp_dir), 'status': 'computed',
                'source': f"computed:{int(payload['n_virtual_points'])}pts"}
    except Exception as e:  # ワーカー内例外は呼び出し側で表示
        return {'exp_dir': str(exp_dir), 'status': 'error', 'source': f"{type(e).__name__}: {e}"}


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
    print(f"Frame stride        : {args.frame_stride}"
          + (f" (max_frames = {args.max_frames})" if args.max_frames is not None else ""))
    print(f"Mask radius         : max({args.min_mask_radius_px:.0f} px, "
          f"{args.mask_radius_factor:.1f} x R_c) / {args.scale} um/px")
    print(f"Source              : {args.source} (cache: {args.cache_name})")
    print(f"Fit range           : {fit_range[0]:.2f} - {fit_range[1]:.2f} um")
    print(f"Error mode          : {args.error_mode} SEM (fit weights & bands)")
    print(f"Workers             : {max(1, int(getattr(args, 'n_workers', 1)))}")
    print("-" * 78)

    # --- 対象実験ディレクトリの列挙 ---
    targets: List[Tuple[dict, Path]] = []
    for binfo in target_beads:
        exp_dirs = find_experiment_dirs(root_dir, binfo['name'], binfo.get('dir_name'))
        print(f"[{binfo['name']}] {len(exp_dirs)} experiment dir(s) with {FLOW_NAME}")
        targets.extend((binfo, ed) for ed in exp_dirs)
    if not targets:
        raise RuntimeError("No experiment directories found. Check --root_dir / --beads.")

    # --- 並列計算フェーズ: 重い GFP_flows.h5 読み出しをプロセス並列化（--n_workers > 1） ---
    n_workers = max(1, int(getattr(args, 'n_workers', 1)))
    if n_workers > 1 and args.source != 'existing' and not args.no_cache:
        payloads: List[dict] = []
        border_px_default = (int(args.border_margin_px) if args.border_margin_px is not None
                            else int(np.ceil(max(distances))))
        for binfo, exp_dir in targets:
            mask_px = mask_radius_px_for_bead(binfo['radius_um'], args.scale,
                                              args.mask_radius_factor, args.min_mask_radius_px)
            payloads.append({
                'exp_dir': str(exp_dir),
                'distances': [float(d) for d in distances],
                'n_virtual_points': int(args.n_virtual_points),
                'mask_px': float(mask_px),
                'border_px': int(border_px_default),
                'kernel_type': args.kernel_type,
                'shell_width': float(args.shell_width),
                'seed': int(args.seed),
                'device': args.device,
                'min_flow_mag': float(args.min_flow_mag),
                'max_frames': args.max_frames,
                'frame_stride': int(args.frame_stride),
                'cache_name': args.cache_name,
                'force_recompute': bool(args.force_recompute),
                'no_cache': bool(args.no_cache),
            })
        print(f"[parallel] {n_workers} worker processes で仮想粒子サンプリングを計算"
              f"（対象 {len(payloads)} 実験, cache = {args.cache_name}）...")
        # 注意: CUDA は fork 安全ではない（fork した子プロセスでは CUDA 初期化が失敗し、
        #       黙って CPU バックエンドへフォールバックして数倍遅くなる）。必ず spawn を使う。
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = {ex.submit(_bg_cache_worker, p): Path(p['exp_dir']) for p in payloads}
            n_done = 0
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {'exp_dir': str(futures[fut]), 'status': 'error',
                         'source': f"{type(e).__name__}: {e}"}
                n_done += 1
                print(f"    [{n_done}/{len(payloads)}] {Path(r['exp_dir']).name}: "
                      f"{r['status']} ({r['source']})", flush=True)
    elif n_workers > 1:
        print("[parallel] --no_cache / --source existing のため並列計算を行わず逐次処理します")

    # --- 集計フェーズ（キャッシュ読み込み / 必要なら逐次計算） ---
    exp_results: List[dict] = []
    for binfo, exp_dir in targets:
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
    grid_um = grid_px * float(args.scale)

    # --- 集計（実験 -> 条件） ---
    df_points, df_xi, pool, pool_frame = build_tables(exp_results, fit_range,
                                                      args.min_corr_threshold,
                                                      error_mode=args.error_mode)
    meta = {
        'n_exp': len(exp_results),
        'n_virtual_points': int(df_xi['n_virtual_points'].max()),
        'n_frames': int(df_xi['n_frames'].sum()),
        'n_samples': int((df_xi['n_frames'] * df_xi['n_virtual_points']).sum()),
        'mask_min_px': float(df_xi['mask_radius_px'].min()),
        'mask_max_px': float(df_xi['mask_radius_px'].max()),
        'error_mode': str(args.error_mode),
        'n_workers': int(n_workers),
    }
    df_curves, df_summary = summarize_conditions(df_points, df_xi, pool, target_beads,
                                                 grid_um, fit_range, args.min_corr_threshold,
                                                 pool_frame=pool_frame,
                                                 error_mode=args.error_mode)
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
                          xscale=args.xscale, yscale=args.yscale,
                          max_dist=args.max_dist, ylim=ylim,
                          error_mode=args.error_mode)
    plot_xi_vs_diameter(df_xi, df_summary, out_dirs, global_xi, xscale=args.xi_xscale,
                        error_mode=args.error_mode)
    plot_par_perp_panels(df_curves, df_summary, out_dirs,
                         xscale=args.xscale, yscale=args.yscale,
                         max_dist=args.max_dist,
                         error_mode=args.error_mode)

    # --- ログ出力 ---
    print("-" * 78)
    print(" Global statistics")
    print(f"   experiments          : {meta['n_exp']}")
    print(f"   frames (total)       : {meta['n_frames']}")
    print(f"   virtual points       : {meta['n_virtual_points']} / frame")
    print(f"   mask radius          : {meta['mask_min_px']:.1f} - {meta['mask_max_px']:.1f} px")
    print(f"   samples (total)      : {meta['n_samples']:,} (= frames x virtual points x experiments)")
    print(f"   fit error mode       : {meta['error_mode']} SEM")
    print(f"   weighted mean xi_bg  : {global_xi[0]:.2f} +/- {global_xi[1]:.2f} um (N = {global_xi[2]} experiments)")
    print()
    print(" Per-experiment xi_bg")
    cols = ['bead_name', 'exp_dir', 'source', 'n_frames', 'n_virtual_points', 'mask_radius_px',
            'n_samples_total', 'xi_um', 'xi_err_um', 'xi_um_sample_sem', 'xi_um_frame_sem',
            'xi_r2_log', 'n_fit_points']
    print(df_xi[[c for c in cols if c in df_xi.columns]].to_string(index=False))
    print()
    print(f" Per-condition summary (xi_bg: experiment-level mean +/- SEM; pooled fit = "
          f"{meta['error_mode']} SEM weights)")
    cols2 = ['bead_name', 'diameter_um', 'n_experiments', 'n_samples_total', 'n_frames_pooled',
             'xi_bg_mean_um', 'xi_bg_sem_um', 'xi_bg_pooled_um', 'xi_bg_pooled_sample_um',
             'xi_bg_pooled_frame_um', 'xi_bg_pooled_r2_log']
    print(df_summary[[c for c in cols2 if c in df_summary.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()
