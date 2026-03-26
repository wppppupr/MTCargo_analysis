import nd2
import numpy as np
import dask.array as da
import zarr
from numcodecs import Blosc
from skimage import exposure
from scipy.ndimage import gaussian_filter
import os
import argparse

#####################################################

def process_chunk_wrapper(chunk, equalize, sigma, scale_factor, global_min=None, global_max=None):
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
        
        # 1. Float変換
        frame_float = frame.astype(np.float32)

        if global_min is not None and global_max is not None and global_max > global_min:
            # 全フレームの輝度最小・最大値を使って 0-255 にマッピング (ちらつき防止)
            frame_proc = (frame_float - global_min) / (global_max - global_min) * 255.0
        else:
            frame_float = frame_float * scale_factor
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
                        normalize_global: bool = True,
                        equalize: bool = False,
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
    # 0: MT red, 1: MT, 2: Beads
    dask_mt_red_raw = vol[:, 0, :, :]
    dask_mt_raw = vol[:, 1, :, :]
    dask_beads_raw = vol[:, 2, :, :]
    
    # スケーリング係数 (12bit -> 8bit)
    scale_factor = 255.0 / 4095.0
    
    # 2Dシグマの抽出 (T軸を除外)
    mt_sigma_2d = mt_sigma[1:]
    beads_sigma_2d = beads_sigma[1:]

    # チャンクサイズの調整
    # メモリ効率と処理速度のバランスが良いサイズに再分割します
    # 例: (時間方向10フレーム, Y全画素, X全画素) 単位で処理
    preferred_chunks = (10, -1, -1)
    dask_mt_red_raw = dask_mt_red_raw.rechunk(preferred_chunks)
    dask_mt_raw = dask_mt_raw.rechunk(preferred_chunks)
    dask_beads_raw = dask_beads_raw.rechunk(preferred_chunks)

    if normalize_global:
        import dask
        print("Calculating global min/max for all channels... (This prevents flickering)")
        red_min, red_max, mt_min, mt_max, b_min, b_max = dask.compute(
            dask_mt_red_raw.min(), dask_mt_red_raw.max(),
            dask_mt_raw.min(), dask_mt_raw.max(),
            dask_beads_raw.min(), dask_beads_raw.max()
        )
    else:
        red_min = red_max = mt_min = mt_max = b_min = b_max = None

    # ---------------------------------------------------------
    # 計算グラフの構築 (map_blocks)
    # ---------------------------------------------------------
    
    # redMTチャンネルの処理定義
    dask_mt_red_processed = dask_mt_red_raw.map_blocks(
        process_chunk_wrapper,
        equalize=equalize,
        sigma=mt_sigma_2d,
        scale_factor=scale_factor,
        global_min=red_min,
        global_max=red_max,
        dtype=np.uint8
    )

    # MTチャンネルの処理定義
    dask_mt_processed = dask_mt_raw.map_blocks(
        process_chunk_wrapper,
        equalize=equalize,
        sigma=mt_sigma_2d,
        scale_factor=scale_factor,
        global_min=mt_min,
        global_max=mt_max,
        dtype=np.uint8
    )
    
    # Beadsチャンネルの処理定義
    dask_beads_processed = dask_beads_raw.map_blocks(
        process_chunk_wrapper,
        equalize=equalize,  # Beadsにも適用するかは要件次第(元のコードは適用していた)
        sigma=beads_sigma_2d,
        scale_factor=scale_factor,
        global_min=b_min,
        global_max=b_max,
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
        if int(zarr.__version__.split('.')[0]) >= 3:
            from zarr.codecs import BloscCodec
            compressor_kwargs = {'compressors': [BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')]}
        else:
            compressor_kwargs = {'compressor': Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)}
        
        print(f"Processing and saving to Zarr: {save_dir}")
        
        # プログレスバーを表示するための設定（Dask標準）
        from dask.diagnostics import ProgressBar
        
        # MTの保存
        mt_red_zarr_path = os.path.join(save_dir, "MTs_red.zarr")
        print(f"  - Saving MTs red -> {mt_red_zarr_path}")
        with ProgressBar():
            dask_mt_red_processed.to_zarr(
                mt_red_zarr_path, 
                overwrite=True,
                **compressor_kwargs
            )

        # MTの保存
        mt_zarr_path = os.path.join(save_dir, "MTs.zarr")
        print(f"  - Saving MTs -> {mt_zarr_path}")
        with ProgressBar():
            dask_mt_processed.to_zarr(
                mt_zarr_path, 
                overwrite=True,
                **compressor_kwargs
            )
            
        # Beadsの保存
        beads_zarr_path = os.path.join(save_dir, "beads.zarr")
        print(f"  - Saving Beads -> {beads_zarr_path}")
        with ProgressBar():
            dask_beads_processed.to_zarr(
                beads_zarr_path, 
                overwrite=True,
                **compressor_kwargs
            )

        return mt_red_zarr_path, mt_zarr_path, beads_zarr_path

    else:
        # 保存しない場合は計算せずにDask配列自体を返す（後で計算可能）
        return dask_mt_red_processed, dask_mt_processed, dask_beads_processed
    

def process_nd2_to_zarrMT(file_path: str,
                        normalize_global: bool = True,
                        equalize: bool = False,
                        mt_sigma=(0, 1, 1),
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
    
    # スケーリング係数 (12bit -> 8bit)
    scale_factor = 255.0 / 4095.0
    
    # 2Dシグマの抽出 (T軸を除外)
    mt_sigma_2d = mt_sigma[1:]


    # チャンクサイズの調整
    # メモリ効率と処理速度のバランスが良いサイズに再分割します
    # 例: (時間方向10フレーム, Y全画素, X全画素) 単位で処理
    preferred_chunks = (10, -1, -1)
    dask_mt_raw = dask_mt_raw.rechunk(preferred_chunks)

    if normalize_global:
        import dask
        print("Calculating global min/max to prevent flickering...")
        mt_min, mt_max = dask.compute(
            dask_mt_raw.min(), dask_mt_raw.max()
        )
    else:
        mt_min = mt_max = None

    # ---------------------------------------------------------
    # 計算グラフの構築 (map_blocks)
    # ---------------------------------------------------------
    
    # MTチャンネルの処理定義
    dask_mt_processed = dask_mt_raw.map_blocks(
        process_chunk_wrapper,
        equalize=equalize,
        sigma=mt_sigma_2d,
        scale_factor=scale_factor,
        global_min=mt_min,
        global_max=mt_max,
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
        if int(zarr.__version__.split('.')[0]) >= 3:
            from zarr.codecs import BloscCodec
            compressor_kwargs = {'compressors': [BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')]}
        else:
            compressor_kwargs = {'compressor': Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)}
        
        print(f"Processing and saving to Zarr: {save_dir}")
        
        # プログレスバーを表示するための設定（Dask標準）
        from dask.diagnostics import ProgressBar
        
        # MTの保存
        mt_zarr_path = os.path.join(save_dir, "MTs_red.zarr")
        print(f"  - Saving MTs -> {mt_zarr_path}")
        with ProgressBar():
            dask_mt_processed.to_zarr(
                mt_zarr_path, 
                overwrite=True,
                **compressor_kwargs
            )

        return mt_zarr_path

    else:
        # 保存しない場合は計算せずにDask配列自体を返す（後で計算可能）
        return dask_mt_processed
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ND2 to Zarr with optional normalization and equalization.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input ND2 file.')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--normalize_global', type=bool, default=True, help='Normalize global min/max to prevent flickering.')
    parser.add_argument('--equalize', type=bool, default=False, help='Equalize histogram.')
    parser.add_argument('--mt_sigma', type=tuple, default=(0, 1, 1), help='Gaussian sigma for MT channels.')
    parser.add_argument('--beads_sigma', type=tuple, default=(0, 2, 2), help='Gaussian sigma for beads channel.')
    parser.add_argument('--save', type=bool, default=True, help='Save the processed data to Zarr.')
    args = parser.parse_args()
    
    # 入力ファイルパス

    print('checking file...')
    
    if os.path.exists(args.file_path):
        print('start processing nd2 to zarr...')
        # 処理実行
        process_nd2_to_zarr(
            args.file_path, 
            normalize_global=args.normalize_global,
            equalize=args.equalize, 
            out_dir=args.out_dir
        )
        print('done.')

    else:
        print('file not found.')