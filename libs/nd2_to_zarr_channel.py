import nd2
import numpy as np
import dask.array as da
import zarr
from numcodecs import Blosc
from skimage import exposure
from scipy.ndimage import gaussian_filter
import os
import argparse

def process_chunk_wrapper(chunk, equalize, sigma, scale_factor, global_min=None, global_max=None):
    """
    Daskのブロック（チャンク）ごとに呼ばれる処理関数。
    入力: (Frames, Y, X) の形状を持つ3次元Numpy配列
    出力: (Frames, Y, X) の形状を持つuint8 Numpy配列
    """
    # 出力用配列の確保
    out_chunk = np.empty(chunk.shape, dtype=np.uint8)
    
    # チャンク内の各フレームに対して処理を行う
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

def process_channel_to_zarr(file_path: str,
                            target_channel: int | str,
                            out_name: str | None = None,
                            normalize_global: bool = True,
                            equalize: bool = False,
                            sigma=(0, 1, 1),
                            save: bool = True,
                            out_dir: str | None = None):
    """
    nd2ファイルを読み込み、指定されたチャンネルを処理してZarr形式で保存します。
    """
    print(f"Loading (Lazy): {os.path.basename(file_path)} ...")
    
    with nd2.ND2File(file_path) as f:
        dask_arr = f.to_dask()
        
        channel_idx = None
        
        if target_channel is not None:
            if 'C' in f.sizes and f.sizes['C'] > 1:
                axis_c = list(f.sizes.keys()).index('C')
                
                # 数値指定かどうか
                if isinstance(target_channel, int) or (isinstance(target_channel, str) and target_channel.isdigit()):
                    channel_idx = int(target_channel)
                else:
                    # 文字列としてチャンネル名を検索
                    if hasattr(f, 'metadata') and hasattr(f.metadata, 'channels'):
                        for i, ch in enumerate(f.metadata.channels):
                            name = ""
                            if hasattr(ch, 'channel') and hasattr(ch.channel, 'name'):
                                name = ch.channel.name
                            elif hasattr(ch, 'name'):
                                name = ch.name
                            
                            if name and str(target_channel).lower() == name.lower():
                                channel_idx = i
                                break
                            elif name and str(target_channel).lower() in name.lower():
                                if channel_idx is None:
                                    channel_idx = i
                                    
                if channel_idx is None:
                    print(f"Error: Channel '{target_channel}' not found.")
                    return None
                    
                # スライスを作成して特定のチャンネルのみを抽出
                slices = [slice(None)] * len(f.sizes)
                slices[axis_c] = channel_idx
                dask_raw = dask_arr[tuple(slices)]
                print(f"Extracted channel index {channel_idx}. New shape: {dask_raw.shape}")
            else:
                print(f"Warning: Channel dimension 'C' not found or only 1 channel exists. Ignoring target_channel='{target_channel}'")
                dask_raw = dask_arr
        else:
            dask_raw = dask_arr

        # チャンクサイズとmap_blocksの前提として(Frames, Y, X)があると想定
        if len(dask_raw.shape) == 2:
            # 2次元の場合は時間軸(Frames=1)を追加
            dask_raw = dask_raw.reshape((1,) + dask_raw.shape)

        # スケーリング係数 (12bit -> 8bit) 
        scale_factor = 255.0 / 4095.0
        
        # 2Dシグマの抽出 (T軸を含む場合は除外)
        if isinstance(sigma, (tuple, list)) and len(sigma) >= 3:
            sigma_2d = sigma[-2:] # 最後の2次元 (Y, X)
        else:
            sigma_2d = sigma
            
        # 次元の形に合わせてrechunk
        rechunk_tuple = (10,) + (-1,) * (len(dask_raw.shape) - 1)
        dask_raw = dask_raw.rechunk(rechunk_tuple)

        if normalize_global:
            import dask
            print("Calculating global min/max for the channel... (This prevents flickering)")
            c_min, c_max = dask.compute(dask_raw.min(), dask_raw.max())
        else:
            c_min = c_max = None

        dask_processed = dask_raw.map_blocks(
            process_chunk_wrapper,
            equalize=equalize,
            sigma=sigma_2d,
            scale_factor=scale_factor,
            global_min=c_min,
            global_max=c_max,
            dtype=np.uint8
        )
        
        if save:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            if out_dir is None:
                save_dir = os.path.dirname(file_path)
            else:
                save_dir = os.path.abspath(os.path.expanduser(out_dir))
                if not os.path.isabs(save_dir):
                    save_dir = os.path.abspath(os.path.join("experiment", out_dir))
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Zarrの圧縮設定
            if int(zarr.__version__.split('.')[0]) >= 3:
                from zarr.codecs import BloscCodec
                compressor_kwargs = {'compressors': [BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')]}
            else:
                compressor_kwargs = {'compressor': Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)}
            
            if out_name:
                zarr_name = out_name
                if not zarr_name.endswith('.zarr'):
                    zarr_name += '.zarr'
            else:
                ch_suffix = f"_ch{channel_idx}" if channel_idx is not None else ""
                zarr_name = f"{base_name}{ch_suffix}.zarr"
                
            zarr_path = os.path.join(save_dir, zarr_name)
            print(f"Processing and saving to Zarr: {zarr_path}")
            
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                dask_processed.to_zarr(
                    zarr_path, 
                    overwrite=True,
                    **compressor_kwargs
                )

            return zarr_path
        else:
            return dask_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert single channel from ND2 to Zarr with optional normalization and equalization.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input ND2 file.')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--channel', type=str, required=True, help='Target channel name or index to extract (e.g., "0", "GFP").')
    parser.add_argument('--out_name', type=str, default=None, help='Output Zarr file name.')
    parser.add_argument('--normalize_global', type=bool, default=True, help='Normalize global min/max to prevent flickering.')
    parser.add_argument('--equalize', type=bool, default=False, help='Equalize histogram.')
    parser.add_argument('--sigma', type=tuple, default=(0, 1, 1), help='Gaussian sigma for spatial smoothing.')
    parser.add_argument('--save', type=bool, default=True, help='Save the processed data to Zarr.')
    args = parser.parse_args()
    
    print('checking file...')
    
    if os.path.exists(args.file_path):
        print('start processing nd2 to zarr...')
        process_channel_to_zarr(
            args.file_path, 
            target_channel=args.channel,
            out_name=args.out_name,
            normalize_global=args.normalize_global,
            equalize=args.equalize, 
            sigma=args.sigma,
            save=args.save,
            out_dir=args.out_dir
        )
        print('done.')
    else:
        print('file not found.')
