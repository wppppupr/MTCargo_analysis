import nd2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
import os
from joblib import Parallel, delayed
from tqdm import tqdm # tqdmをインポート

def process_single_frame(mt_frame, beads_frame, equalize, mt_sigma_2d, beads_sigma_2d):
    """
    1フレーム分の処理を行う関数（並列化用）
    """
    # 1. MTチャンネルの処理
    if equalize:
        mt_processed = exposure.equalize_hist(mt_frame)
    else:
        mt_processed = mt_frame
    
    # 2. ガウシアンフィルタ (2D)
    mt_out = gaussian_filter(mt_processed, sigma=mt_sigma_2d)
    
    # 3. Beadsチャンネルの処理
    beads_out = gaussian_filter(beads_frame, sigma=beads_sigma_2d)
    
    return mt_out, beads_out

def process_nd2_file(file_path: str,
                     diameter: float,
                     scale: float = 0.11,
                     equalize: bool = True,
                     mt_sigma=(0, 1, 1),
                     beads_sigma=(0, 2, 2),
                     save: bool = True,
                     out_dir: str | None = None,
                     n_jobs: int = -1):
    """
    高速化版: nd2ファイルを読み込み、並列処理(tqdm付き)で平坦化・平滑化を行います。
    """

    print(f"Loading: {os.path.basename(file_path)} ...")
    # ファイル読み込み (RAMに余裕がある前提で一括読み込み)
    vol = nd2.imread(file_path)
    
    # float32へのキャスト (メモリ節約・高速化)
    comp = np.float32(255 / 4095)
    
    # チャンネル分離 (スライスのみ、コピーなし)
    MTs_raw = vol[:, 0, :, :]
    beads_raw = vol[:, 1, :, :]
    n_frames = len(MTs_raw)

    # sigmaの次元調整
    mt_sigma_2d = mt_sigma[1:]
    beads_sigma_2d = beads_sigma[1:]

    print(f"Processing {n_frames} frames with {n_jobs if n_jobs > 0 else 'all'} cores...")
    
    # 【tqdm実装ポイント】
    # return_as="generator" を使うことで、処理が終わった順に結果を取り出せるようになります。
    # これを tqdm で回すことで、正確な進捗状況が表示されます。
    # (joblib >= 1.2.0 推奨)
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_generator = parallel(
            delayed(process_single_frame)(
                MTs_raw[i] * comp,
                beads_raw[i] * comp,
                equalize,
                mt_sigma_2d,
                beads_sigma_2d
            ) for i in range(n_frames)
        )
        
        # ジェネレータから結果を順次取り出しつつ、tqdmで進捗表示
        results = [
            res for res in tqdm(results_generator, total=n_frames, unit="frame", desc="Smoothing")
        ]

    # 結果の結合
    MTs_smoothed = np.array([r[0] for r in results], dtype=np.float32)
    beads_smoothed = np.array([r[1] for r in results], dtype=np.float32)

    # 保存パスの構築
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    if out_dir is None:
        save_dir = os.path.dirname(file_path)
    else:
        save_dir = os.path.join("experiment", out_dir)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        mt_save_path = os.path.join(save_dir, f"{base_name}_MT_smoothed.npy")
        beads_save_path = os.path.join(save_dir, f"{base_name}_beads_smoothed.npy")
        
        print(f"Saving to: {save_dir}")
        np.save(mt_save_path, MTs_smoothed)
        np.save(beads_save_path, beads_smoothed)

    output_base_path = os.path.join(save_dir, base_name)
    return MTs_smoothed, beads_smoothed, output_base_path

if __name__ == "__main__":
    file_path = r'/Volumes/data/Sasaki/MTsingleBeads/20251210/MC03_4uM.nd2'
    diameter = 1.18

    if os.path.exists(file_path):
        process_nd2_file(
            file_path, 
            diameter, 
            equalize=True, 
            save=True, 
            out_dir="20251210",
            n_jobs=-1
        )
    else:
        print("File not found.")