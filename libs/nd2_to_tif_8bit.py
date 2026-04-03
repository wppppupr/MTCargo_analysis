import argparse
import nd2
import tifffile
import numpy as np
import dask.array as da
from pathlib import Path
from tqdm import tqdm
import time
import os
from skimage.exposure import match_histograms

def convert_nd2_to_tif_8bit(nd2_path, tif_output_dir, mode='hist_match_prev', target_channel=None):
    """
    ND2ファイルを読み込み、8-bitに変換して、指定したディレクトリ内に個別のTIFF画像群として保存します。
    ターミナルから特定のチャンネルだけを抽出して変換することも可能です。
    
    引数:
        nd2_path: 入力ND2ファイルのパス
        tif_output_dir: 個別のTIFF画像を保存するディレクトリのパス
        mode: 8-bitへの変換手法
              1. 'minmax_global'- 全体の最小・最大値を計算して正規化。（デフォルト設定。画像のチラつきを防ぎます）
              2. 'minmax_frame' - フレームごとに最小・最大を計算して正規化。（高速ですが、フレーム間で輝度が変わります）
              3. 'shift'        - 単純なビットシフトを使用。（最速・元の輝度を保持。暗く映る場合があります）
              4. 'hist_match_prev' - 1つ前のフレームのヒストグラムに合わせる。（急激な輝度変化を抑えます）
        target_channel: 抽出するチャンネル名 (例: 'GFP')、またはインデックス (例: '0')。Noneなら全て。
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(tif_output_dir)
    
    # 出力先ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading: {nd2_path}")
    print(f"Output Directory: {output_dir}")
    start_time = time.time()

    with nd2.ND2File(str(nd2_path)) as f:
        print(f"Original shape: {f.shape}, Dtype: {f.dtype}, Dimensions: {f.sizes}")
        
        # dask arrayとして取得 (遅延評価による省メモリ化)
        dask_arr = f.to_dask()

        # --- 指定チャンネルの抽出処理 ---
        if target_channel is not None:
            if 'C' in f.sizes and f.sizes['C'] > 1:
                axis_c = list(f.sizes.keys()).index('C')
                channel_idx = None
                
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
                            
                            if name and target_channel.lower() == name.lower():
                                channel_idx = i
                                print(f"Found target channel '{target_channel}' at index {i} (Full name: {name})")
                                break
                            elif name and target_channel.lower() in name.lower():
                                # 大文字小文字を無視した部分一致も許容（完全一致がなければ）
                                if channel_idx is None:
                                    channel_idx = i
                                    print(f"Found partial match for '{target_channel}' at index {i} (Full name: {name})")
                    
                    if channel_idx is None:
                        av_channels = []
                        if hasattr(f, 'metadata') and hasattr(f.metadata, 'channels'):
                            for ch in f.metadata.channels:
                                if hasattr(ch, 'channel') and hasattr(ch.channel, 'name'):
                                    av_channels.append(ch.channel.name)
                                elif hasattr(ch, 'name'):
                                    av_channels.append(ch.name)
                        print(f"Error: Channel '{target_channel}' not found. Available channels: {av_channels}")
                        return
                
                # スライスを作成して特定のチャンネルのみを抽出
                slices = [slice(None)] * len(f.sizes)
                slices[axis_c] = channel_idx
                dask_arr = dask_arr[tuple(slices)]
                print(f"Extracted channel index {channel_idx}. New shape: {dask_arr.shape}")
            else:
                print(f"Warning: Channel dimension 'C' not found or only 1 channel exists. Ignoring target_channel='{target_channel}'")

        # フレーム数とゼロ埋めの桁数を決定
        num_frames = dask_arr.shape[0] if len(dask_arr.shape) >= 3 else 1
        zfill_len = max(3, len(str(num_frames)))

        # --- 変換パラメータの事前計算 ---
        shift = 0
        global_min = 0
        global_scale = 1.0

        if mode == 'shift':
            try:
                sig_bits = f.attributes.bitsPerComponentSignificant
                if sig_bits is None or sig_bits <= 0:
                    sig_bits = 16
            except Exception:
                sig_bits = 16
            shift = max(0, sig_bits - 8)
            print(f"Mode: 'shift' (Significant bits: {sig_bits}, right shift: >> {shift})")
        
        elif mode == 'minmax_global':
            print("Mode: 'minmax_global' (Computing global min/max... this may take a moment)")
            g_min = float(dask_arr.min().compute())
            g_max = float(dask_arr.max().compute())
            print(f"Global Min: {g_min}, Global Max: {g_max}")
            
            global_min = g_min
            if g_max > g_min:
                global_scale = 255.0 / (g_max - g_min)
            else:
                global_scale = 1.0
        
        elif mode == 'hist_match_prev':
            print("Mode: 'hist_match_prev' (Matching histogram to the previous frame)")
        
        else:
            print("Mode: 'minmax_frame' (Normalizing frame by frame)")

        # --- メインループ ---
        prev_frame_8bit = None
        
        if len(dask_arr.shape) >= 3:
            for i in tqdm(range(num_frames), desc="Converting and saving frames"):
                # compute()で1フレーム分だけメモリに展開（I/O負荷を軽減）
                frame = dask_arr[i].compute()
                
                if mode == 'shift':
                    if shift > 0:
                        frame_8bit = np.right_shift(frame, shift).astype(np.uint8)
                    else:
                        frame_8bit = frame.astype(np.uint8)
                
                elif mode == 'minmax_global':
                    f_float = (frame.astype(np.float32) - global_min) * global_scale
                    frame_8bit = np.clip(f_float, 0, 255).astype(np.uint8)    
                
                elif mode == 'hist_match_prev':
                    # 各フレームをまず自身のmin-maxで8-bit化
                    f_min, f_max = float(frame.min()), float(frame.max())
                    f_scale = 255.0 / (f_max - f_min) if f_max > f_min else 1.0
                    f_float = (frame.astype(np.float32) - f_min) * f_scale
                    frame_8bit = np.clip(f_float, 0, 255).astype(np.uint8)
                    
                    if prev_frame_8bit is not None:
                        # 1つ前のフレーム(8-bit)のヒストグラムに合わせる
                        frame_8bit = match_histograms(frame_8bit, prev_frame_8bit).astype(np.uint8)
                    
                    prev_frame_8bit = frame_8bit

                else: # 'minmax_frame'
                    f_min, f_max = float(frame.min()), float(frame.max())
                    f_scale = 255.0 / (f_max - f_min) if f_max > f_min else 1.0
                    f_float = (frame.astype(np.float32) - f_min) * f_scale
                    frame_8bit = np.clip(f_float, 0, 255).astype(np.uint8)

                # TIFF保存 (連番)
                filename = output_dir / f"frame_{str(i).zfill(zfill_len)}.tif"
                tifffile.imwrite(filename, frame_8bit, compression='zlib', photometric='minisblack')
        
        else: # 2D画像一枚のみの場合
            frame = dask_arr.compute()
            if mode == 'shift':
                if shift > 0:
                    frame_8bit = np.right_shift(frame, shift).astype(np.uint8)
                else:
                    frame_8bit = frame.astype(np.uint8)
            else: # 2D画像の場合は全体=フレーム計算
                f_min, f_max = float(frame.min()), float(frame.max())
                f_scale = 255.0 / (f_max - f_min) if f_max > f_min else 1.0
                f_float = (frame.astype(np.float32) - f_min) * f_scale
                frame_8bit = np.clip(f_float, 0, 255).astype(np.uint8)

            filename = output_dir / f"frame_{str(0).zfill(zfill_len)}.tif"
            tifffile.imwrite(filename, frame_8bit, compression='zlib', photometric='minisblack')

    print(f"Conversion Completed in {time.time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ND2ファイルを8-bit TIFF画像群(連番ファイル)に変換します。")
    parser.add_argument("input", type=str, help="入力ND2ファイルのパス")
    parser.add_argument("output_dir", type=str, help="連番TIFF画像を保存する出力ディレクトリのパス")
    parser.add_argument("--channel", type=str, default=None, help="抽出する特定のチャンネル名 (例: GFP) またはインデックス番号")
    parser.add_argument("--mode", type=str, default="hist_match_prev", choices=['minmax_global', 'minmax_frame', 'shift', 'hist_match_prev'],
                        help="8-bitへの圧縮モード。デフォルトは 'hist_match_prev'")
    
    args = parser.parse_args()
    
    convert_nd2_to_tif_8bit(
        nd2_path=args.input,
        tif_output_dir=args.output_dir,
        mode=args.mode,
        target_channel=args.channel
    )
