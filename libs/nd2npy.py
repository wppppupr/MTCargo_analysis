import nd2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
import os
from joblib import Parallel, delayed
from tqdm import tqdm

def process_single_frame(mt_frame_raw, beads_frame_raw, equalize, mt_sigma_2d, beads_sigma_2d, scale_factor):
    """
    1フレーム分の処理を行い、uint8型で返します。
    """
    # ---------------------------------------------------------
    # 1. MTチャンネルの処理
    # ---------------------------------------------------------
    # 生データをfloatに変換 (0-255スケール)
    mt_float = mt_frame_raw * scale_factor

    if equalize:
        # equalize_histは入力を正規化し、戻り値は float64 の [0, 1] になります
        # そのため、処理後に 255 を掛けてスケールを戻します
        mt_eq = exposure.equalize_hist(mt_float)
        mt_proc = mt_eq * 255.0
    else:
        mt_proc = mt_float

    # ガウシアンフィルタ (floatのまま適用して精度を維持)
    mt_smoothed = gaussian_filter(mt_proc, sigma=mt_sigma_2d)
    
    # 【重要】uint8への変換
    # 0-255の範囲を超えた値をクリップし、整数にキャストします
    mt_out = np.clip(mt_smoothed, 0, 255).astype(np.uint8)
    
    # ---------------------------------------------------------
    # 2. Beadsチャンネルの処理
    # ---------------------------------------------------------
    beads_float = beads_frame_raw * scale_factor
    if equalize:
        # equalize_histは入力を正規化し、戻り値は float64 の [0, 1] になります
        # そのため、処理後に 255 を掛けてスケールを戻します
        beads_eq = exposure.equalize_hist(beads_float)
        mt_proc = beads_eq * 255.0
    else:
        mt_proc = mt_float
    beads_smoothed = gaussian_filter(beads_float, sigma=beads_sigma_2d)
    beads_out = np.clip(beads_smoothed, 0, 255).astype(np.uint8)
    
    return mt_out, beads_out


def _save_array_big(path: str, arr: np.ndarray) -> None:
    """
    Save large numpy array to .npy safely using memmap to avoid large single-write errors.
    """
    # remove existing file if present
    if os.path.exists(path):
        os.remove(path)
    # write using memmap (writes directly to disk in chunks)
    mm = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype, shape=arr.shape)
    mm[:] = arr
    del mm


def process_nd2_file(file_path: str,
                     equalize: bool = True,
                     mt_sigma=(0, 1, 1),
                     beads_sigma=(0, 2, 2),
                     save: bool = True,
                     out_dir: str | None = None,
                     n_jobs: int = -1):
    """
    nd2ファイルを読み込み、OpenCV互換の uint8 npyファイルとして保存します。
    """

    print(f"Loading: {os.path.basename(file_path)} ...")
    vol = nd2.imread(file_path)
    
    # 生データ(uint16)のままスライスを取得 (メモリ節約)
    # スケーリング係数を計算 (12bit = 4095 -> 8bit = 255)
    scale_factor = 255.0 / 4095.0
    
    MTs_raw = vol[:, 0, :, :]
    beads_raw = vol[:, 1, :, :]
    n_frames = len(MTs_raw)

    # sigmaの次元調整
    mt_sigma_2d = mt_sigma[1:]
    beads_sigma_2d = beads_sigma[1:]

    print(f"Processing {n_frames} frames to uint8 with {n_jobs if n_jobs > 0 else 'all'} cores...")
    
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_generator = parallel(
            delayed(process_single_frame)(
                MTs_raw[i],       # Rawデータを渡す
                beads_raw[i],     # Rawデータを渡す
                equalize,
                mt_sigma_2d,
                beads_sigma_2d,
                scale_factor
            ) for i in range(n_frames)
        )
        
        results = [
            res for res in tqdm(results_generator, total=n_frames, unit="frame", desc="Smoothing")
        ]

    # 結果の結合 (uint8 なのでメモリ消費は極小です)
    MTs_result = np.array([r[0] for r in results], dtype=np.uint8)
    beads_result = np.array([r[1] for r in results], dtype=np.uint8)

    # 保存パス設定
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    if out_dir is None:
        save_dir = os.path.dirname(file_path)
    else:
        out_dir_expanded = os.path.abspath(os.path.expanduser(out_dir))
        # 絶対パスが渡されたらそのまま使用（NAS を /Volumes/... にマウントしておく）
        if os.path.isabs(out_dir_expanded):
            save_dir = out_dir_expanded
        else:
            # 以前の挙動を保持したい場合は experiment/ 配下に配置
            save_dir = os.path.abspath(os.path.join("experiment", out_dir))

    if save:
        os.makedirs(save_dir, exist_ok=True)
        if not os.access(save_dir, os.W_OK):
            raise PermissionError(f"No write permission to save_dir: {save_dir}")
        mt_save_path = os.path.join(save_dir, f"MTs.npy")
        beads_save_path = os.path.join(save_dir, f"beads.npy")
        
        print(f"Saving uint8 arrays to: {save_dir}")
        # Use a memmap-based safe saver to avoid 2GB write failures on some platforms
        _save_array_big(mt_save_path, MTs_result)
        _save_array_big(beads_save_path, beads_result)

    output_base_path = os.path.join(save_dir, base_name)
    return MTs_result, beads_result, output_base_path

if __name__ == "__main__":
    file_path = r'/Volumes/data/Sasaki/MTsingleBeads/20251226/MC00MT001.nd2'
    nas = "/Volumes/data/Sasaki/backup_git/MTCargo_analysis/experiment"
    # テスト実行
    if os.path.exists(file_path):
        process_nd2_file(
            file_path, 
            equalize=True, 
            save=True, 
            out_dir=f"{nas}/20251226/MC00MT001",
            n_jobs=-1
        )