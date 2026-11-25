# libs/experiment_cutoff.py
"""
実験動画ごとの安定フレーム数上限（カットオフ）定義およびフィルタリングユーティリティ。

背景:
- GFP蛍光チャネルの退色 (Photobleaching) や微小管活性の経時低下により、
  長時間動画の後半においてRAFTオプティカルフローが異常値・暴走（Artifact patches）を起こす。
- また、ビーズトラッキングにおいても終盤の画像劣化により局所的なドリフトや重心揺らぎが生じる。
- 本モジュールでは、動画ごとにフローおよびトラッキングが健全に維持されている
  「フレーム数上限 (cutoff_frame)」を一元管理し、全解析スクリプト（MSD, フロー相互作用, 輸送効率等）に適用可能とする。
"""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd

# 実験動画または特定粒子レベルの除外フィルタ（異常データ除外用・全解析共通）
# - beads1um_crop: 高速Run状態にも関わらず流れと完全無相関・相関長xiが異常に大きい動画
# - (beads1um001_crop, 7): オプティカルフロー異常スパイク (最大 ~80 px/frm)
EXCLUDED_EXPERIMENTS = frozenset({"beads1um_crop"})
EXCLUDED_PARTICLES = frozenset({("beads1um001_crop", 7)})


def _normalize_exp_name(exp_name_or_path: Union[str, Path]) -> str:
    return Path(str(exp_name_or_path).rstrip("/")).name


def is_excluded_experiment(exp_name_or_path: Union[str, Path]) -> bool:
    """実験動画名（またはパス）が除外対象なら True を返す。"""
    if exp_name_or_path is None:
        return False
    return _normalize_exp_name(exp_name_or_path) in EXCLUDED_EXPERIMENTS


def is_excluded_particle(
    exp_name_or_path: Optional[Union[str, Path]] = None,
    particle_id: Optional[Union[int, float, str]] = None,
    cargo_id: Optional[str] = None,
) -> bool:
    """
    実験単位または (実験名, 粒子ID) 単位で除外対象なら True を返す。
    cargo_id ('exp_name_pid' または 'exp_name:pid') のみ渡された場合も解釈する。
    """
    exp_name = _normalize_exp_name(exp_name_or_path) if exp_name_or_path is not None else ""
    pid = particle_id
    if pid is None and cargo_id is not None:
        s = str(cargo_id).strip()
        if ":" in s:
            exp_part, pid_part = s.rsplit(":", 1)
        elif "_" in s:
            exp_part, pid_part = s.rsplit("_", 1)
        else:
            exp_part, pid_part = s, None
        if not exp_name:
            exp_name = _normalize_exp_name(exp_part)
        if pid_part is not None:
            try:
                pid = int(float(pid_part))
            except (ValueError, TypeError):
                pid = None

    if exp_name and exp_name in EXCLUDED_EXPERIMENTS:
        return True
    if exp_name and pid is not None:
        try:
            if (exp_name, int(pid)) in EXCLUDED_PARTICLES:
                return True
        except (ValueError, TypeError):
            pass
    return False


def filter_excluded_data(
    df: pd.DataFrame,
    exp_col: Optional[str] = "exp_dir",
    particle_col: Optional[str] = "particle",
    cargo_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """DataFrame から除外対象の実験動画・粒子をすべて除去したコピーを返す。"""
    if df.empty:
        return df

    keep = pd.Series(True, index=df.index)
    resolved_exp_col = exp_col if (exp_col and exp_col in df.columns) else None
    if resolved_exp_col is None:
        for cand in ("exp_dir", "exp_name", "exp"):
            if cand in df.columns:
                resolved_exp_col = cand
                break

    resolved_pid_col = particle_col if (particle_col and particle_col in df.columns) else None
    if resolved_pid_col is None:
        for cand in ("particle", "orig_particle", "pid"):
            if cand in df.columns:
                resolved_pid_col = cand
                break

    if resolved_exp_col is not None:
        exp_names = df[resolved_exp_col].map(_normalize_exp_name)
        keep &= ~exp_names.isin(EXCLUDED_EXPERIMENTS)
        if resolved_pid_col is not None:
            pids_num = pd.to_numeric(df[resolved_pid_col], errors="coerce")
            pids_str = df[resolved_pid_col].astype(str)
            for ex_exp, ex_pid in EXCLUDED_PARTICLES:
                keep &= ~((exp_names == ex_exp) & ((pids_num == ex_pid) | (pids_str == f"{ex_exp}_{ex_pid}") | (pids_str == f"{ex_exp}:{ex_pid}")))

    resolved_cid_col = cargo_id_col if (cargo_id_col and cargo_id_col in df.columns) else None
    if resolved_cid_col is None and "cargo_id" in df.columns:
        resolved_cid_col = "cargo_id"
    if resolved_cid_col is None and "particle_id" in df.columns:
        resolved_cid_col = "particle_id"

    if resolved_cid_col is not None:
        keep &= ~df[resolved_cid_col].map(lambda v: is_excluded_particle(cargo_id=str(v)))

    return df[keep].copy()

# 実験ごとのフレーム上限 (0-indexed: frame <= cutoff_frame を保持)
DEFAULT_EXPERIMENT_CUTOFFS: Dict[str, int] = {
    # 20 µm
    'beads20um001': 450,  # 450以降退色とRAFT暴走パッチが発生
    'beads20um002': 750,  # 750以降微小管流失活およびトラッキングノイズ増加
    'beads20um_crop': 100, # 全長101フレーム、全域健全
    
    # 5 µm
    'beads5um002': 600,   # 1800フレーム中、600以降流速が初期の1/5以下に減速・退色
    'beads5um003': 600,   # 同上
    
    # 3 µm
    'beads3um003': 500,   # 796フレーム中、500以降流速が著しく低下
    
    # 1 µm
    'beads1um003': 450,   # 795フレーム中、450以降流速失活
    
    # 7 µm
    'beads7um003': 500,   # 743フレーム中、500以降流速減衰
}


def get_cutoff_frame(exp_name_or_path: Union[str, Path]) -> Optional[int]:
    """
    実験名またはパスからカットオフ上限フレーム数を取得する。
    未定義（短時間動画など全フレーム安定なもの）の場合は None を返す。
    """
    exp_str = str(exp_name_or_path).rstrip('/')
    exp_name = Path(exp_str).name
    return DEFAULT_EXPERIMENT_CUTOFFS.get(exp_name, None)


def filter_by_cutoff(
    df: pd.DataFrame,
    exp_col: str = 'exp_dir',
    frame_col: str = 'frame',
    cutoffs: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    データフレームに対して動画ごとのカットオフ上限を適用し、健全なフレームのみを抽出する。

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム（'frame' と実験名列を含む）
    exp_col : str
        実験名または実験パスが格納されているカラム名
    frame_col : str
        フレーム番号が格納されているカラム名
    cutoffs : dict, optional
        カスタムカットオフ辞書。省略時は DEFAULT_EXPERIMENT_CUTOFFS を使用。

    Returns
    -------
    pd.DataFrame
        フィルタリング後のデータフレーム
    """
    if df.empty or frame_col not in df.columns:
        return df

    cutoff_map = cutoffs if cutoffs is not None else DEFAULT_EXPERIMENT_CUTOFFS

    # exp_col が存在しない場合はフィルタリングせずにそのまま返す
    if exp_col not in df.columns:
        return df

    # 実験パスから末尾ディレクトリ名（exp_name）を抽出してマッピング
    def _extract_exp_name(val):
        return Path(str(val).rstrip('/')).name

    exp_names = df[exp_col].map(_extract_exp_name)
    cutoff_series = exp_names.map(cutoff_map)

    # カットオフ未指定（NaN）または frame <= cutoff の行を保持
    mask = cutoff_series.isna() | (df[frame_col] <= cutoff_series)
    filtered = df[mask].copy()
    return filter_excluded_data(filtered, exp_col=exp_col)


def load_strict_active_particles() -> set:
    """
    全解析で統一して使用する「厳格な活性貨物粒子」のセット {(exp_name, particle_id), ...} を返す。
    除外設定 (EXCLUDED_EXPERIMENTS, EXCLUDED_PARTICLES) を自動適用する。
    基準:
      - 軌道長 >= 5 frames
      - 活性輸送判定 iMSD alpha > 0.5
      - 微小管フローと粒子が重なり合う有効な相互作用フレーム >= 10 frames
      - 粒子直下フローが健全に取得可能
    """
    csv_candidates = [
        Path(__file__).resolve().parent.parent / "strict_active_particles.csv",
        Path(__file__).resolve().parent.parent / "figure" / "cos_decomposition" / "cos_i_cargo_individual_stats.csv",
        Path(__file__).resolve().parent.parent / "figure" / "chi_vs_radius" / "cos_vs_radius_particle_summary.csv",
    ]
    strict_set = set()
    for p in csv_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if 'exp_name' in df.columns and 'particle' in df.columns:
                    for _, r in df.iterrows():
                        exp = str(r['exp_name'])
                        pid = int(r['particle'])
                        if not is_excluded_particle(exp, pid):
                            strict_set.add((exp, pid))
                    return strict_set
                elif 'cargo_id' in df.columns:
                    for cid in df['cargo_id']:
                        exp, pid = str(cid).rsplit('_', 1)
                        pid_i = int(pid)
                        if not is_excluded_particle(exp, pid_i):
                            strict_set.add((exp, pid_i))
                    return strict_set
            except Exception:
                pass
    return strict_set


