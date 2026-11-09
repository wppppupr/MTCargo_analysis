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
    return df[mask].copy()


def load_strict_active_particles() -> set:
    """
    全解析で統一して使用する「厳格な活性貨物粒子 (N=70)」のセット {(exp_name, particle_id), ...} を返す。
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
                        strict_set.add((str(r['exp_name']), int(r['particle'])))
                    return strict_set
                elif 'cargo_id' in df.columns:
                    for cid in df['cargo_id']:
                        exp, pid = cid.rsplit('_', 1)
                        strict_set.add((exp, int(pid)))
                    return strict_set
            except Exception:
                pass
    return strict_set


