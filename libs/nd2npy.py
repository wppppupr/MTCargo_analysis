import nd2
import numpy as np
import dask.array as da
import zarr
from numcodecs import Blosc
from skimage import exposure
from scipy.ndimage import gaussian_filter
import os

def process_chunk_wrapper(chunk, equalize, sigma, scale_factor):
    """
    Daskのブロック（チャンク）ごとに呼ばれる処理関数。
    入力: (Frames, Y, X) の形状を持つ3次元Numpy配列
    出力: (Frames, Y, X) の形状を持つuint8 Numpy配列
    """
    # 出力用配列の確保
    out_chunk = np.empty(chunk.shape, dtype=np.uint8)
    
    # チャンク内の各フレームに対して処理を行う
    # (Daskが各コアにこの関数を分散させるため、ここでのループは少量で高速です)
    for i in range(chunk.shape[0]):
        frame = chunk[i]
        
        # 1. Float変換 & スケーリング
        frame_float = frame * scale_factor

        # 2. ヒストグラム平坦化 (equalize)
        if equalize:
            # equalize_histは正規化された[0,1]を返すため、255倍する
            frame_eq = exposure.equalize_hist(frame_float)
            frame_proc = frame_eq * 255.0
        else:
            frame_proc = frame_float

        # 3. ガウシアンフィルタ
        if sigma is not None and any(s > 0 for s in sigma):
            frame_smoothed = gaussian_filter(frame_proc, sigma=sigma)
        else:
            frame_smoothed = frame_proc
            
        # 4. uint8へクリップ & キャスト
        out_chunk[i] = np.clip(frame_smoothed, 0, 255).astype(np.uint8)
        
    return out_chunk

def process_nd2_to_zarr(file_path: str,
                        equalize: bool = True,
                        mt_sigma=(0, 1, 1),
                        beads_sigma=(0, 2, 2),
                        save: bool = True,
                        out_dir: str | None = None):
    """
    nd2ファイルを読み込み、処理してZarr形式で保存します。
    """
    
    print(f"Loading (Lazy): {os.path.basename(file_path)} ...")
    
    # dask=Trueで遅延読み込みオブジェクトを取得
    vol = nd2.imread(file_path, dask=True)
    
    # nd2のdimsを確認 (例: {'T': 100, 'C': 2, 'Y': 1024, 'X': 1024})
    # 配列形状は (T, C, Y, X) であると仮定
    
    # チャンネルごとに分離 (まだデータは読み込まれません)
    # 0: MT, 1: Beads
    dask_mt_raw = vol[:, 0, :, :]
    dask_beads_raw = vol[:, 1, :, :]
    
    # スケーリング係数 (12bit -> 8bit)
    scale_factor = 255.0 / 4095.0
    
    # 2Dシグマの抽出 (T軸を除外)
    mt_sigma_2d = mt_sigma[1:]
    beads_sigma_2d = beads_sigma[1:]

    # チャンクサイズの調整
    # メモリ効率と処理速度のバランスが良いサイズに再分割します
    # 例: (時間方向10フレーム, Y全画素, X全画素) 単位で処理
    preferred_chunks = (10, -1, -1)
    dask_mt_raw = dask_mt_raw.rechunk(preferred_chunks)
    dask_beads_raw = dask_beads_raw.rechunk(preferred_chunks)

    # ---------------------------------------------------------
    # 計算グラフの構築 (map_blocks)
    # ---------------------------------------------------------
    
    # MTチャンネルの処理定義
    dask_mt_processed = dask_mt_raw.map_blocks(
        process_chunk_wrapper,
        equalize=equalize,
        sigma=mt_sigma_2d,
        scale_factor=scale_factor,
        dtype=np.uint8
    )
    
    # Beadsチャンネルの処理定義
    dask_beads_processed = dask_beads_raw.map_blocks(
        process_chunk_wrapper,
        equalize=equalize,  # Beadsにも適用するかは要件次第(元のコードは適用していた)
        sigma=beads_sigma_2d,
        scale_factor=scale_factor,
        dtype=np.uint8
    )

    # ---------------------------------------------------------
    # 保存処理 (Compute & Save)
    # ---------------------------------------------------------
    if save:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 保存先ディレクトリの決定
        if out_dir is None:
            save_dir = os.path.dirname(file_path)
        else:
            save_dir = os.path.abspath(os.path.expanduser(out_dir))
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(os.path.join("experiment", out_dir))
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Zarrの圧縮設定 (zstdは圧縮率と速度のバランスが良い)
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
        
        print(f"Processing and saving to Zarr: {save_dir}")
        
        # プログレスバーを表示するための設定（Dask標準）
        from dask.diagnostics import ProgressBar
        
        # MTの保存
        mt_zarr_path = os.path.join(save_dir, "MTs.zarr")
        print(f"  - Saving MTs -> {mt_zarr_path}")
        with ProgressBar():
            dask_mt_processed.to_zarr(
                mt_zarr_path, 
                compressor=compressor, 
                overwrite=True
            )
            
        # Beadsの保存
        beads_zarr_path = os.path.join(save_dir, "beads.zarr")
        print(f"  - Saving Beads -> {beads_zarr_path}")
        with ProgressBar():
            dask_beads_processed.to_zarr(
                beads_zarr_path, 
                compressor=compressor, 
                overwrite=True
            )

        return mt_zarr_path, beads_zarr_path

    else:
        # 保存しない場合は計算せずにDask配列自体を返す（後で計算可能）
        return dask_mt_processed, dask_beads_processed

if __name__ == "__main__":
    # 入力ファイルパス
    file_path = '/Volumes/data/Sasaki/MTsingleBeads/20260106/MT4uM_MC03001.nd2'
    nas_dir = "/Volumes/data/Sasaki/backup_git/MTCargo_analysis/experiment"

    print('checking file...')
    
    if os.path.exists(file_path):
        print('start processing nd2 to zarr...')
        # 処理実行
        process_nd2_to_zarr(
            file_path, 
            equalize=True, 
            out_dir=f"{nas_dir}/20260106/MT4uM_MC03001"
        )
        print('done.')