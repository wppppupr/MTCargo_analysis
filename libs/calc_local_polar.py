import argparse
import numpy as np
import zarr
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter

def main():
    parser = argparse.ArgumentParser(description='Calculate local polar order over multiple window sizes.')
    parser.add_argument('base_path', type=str, help='Path to the base directory containing zarr files')
    parser.add_argument('--start', type=int, default=3, help='Start window size (default: 3)')
    parser.add_argument('--stop', type=int, default=15, help='Stop window size (inclusive, default: 15)')
    parser.add_argument('--step', type=int, default=2, help='Step size for window sizes (default: 2)')
    parser.add_argument('--frame_offset', type=int, default=1, help='Frame offset for flow matching (e.g. flow[f-1] corresponds to frame f)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    print(f'Loading data from {base_path}...')

    try:
        im_theta = zarr.open_array(str(base_path / "im_theta.zarr"), read_only=True)
        im_eccentricity = zarr.open_array(str(base_path / "im_eccentricity.zarr"), read_only=True)
        green_flow = zarr.open_array(str(base_path / "green_flow.zarr"), read_only=True)
    except Exception as e:
        print(f"Error loading zarr files: {e}")
        return

    num_frames = im_theta.shape[0]
    rows, cols = im_theta.shape[1], im_theta.shape[2]
    
    # 範囲指定で探索するウィンドウサイズのリストを生成
    local_sizes = list(range(args.start, args.stop + 1, args.step))
    
    # 局所的なポーラー度を保存するための配列 (ウィンドウサイズごとに用意)
    local_polar_orders = {size: np.zeros((num_frames, rows, cols), dtype=np.float32) for size in local_sizes}

    print(f"im_theta shape: {im_theta.shape}")
    print(f"green_flow shape: {green_flow.shape}")
    print(f"Calculating local polar order for window sizes: {local_sizes}...")

    # green_flowが元の画像サイズか、すでにグリッドサイズにダウンサンプルされているかを判別
    flow_is_full_res = (green_flow.shape[1] != rows or green_flow.shape[2] != cols)

    if flow_is_full_res:
        # ダウンサンプルの比率を計算 (大まかな推定)
        # 注意: 正確なグリッド座標(target_X, target_Y)が必要な場合は、
        # AFTのパラメータから計算するか、保存されているXY座標を使用する必要があります。
        scale_y = green_flow.shape[1] / rows
        scale_x = green_flow.shape[2] / cols
        target_y = np.linspace(scale_y/2, green_flow.shape[1] - scale_y/2, rows).astype(int)
        target_x = np.linspace(scale_x/2, green_flow.shape[2] - scale_x/2, cols).astype(int)
        target_X, target_Y = np.meshgrid(target_x, target_y)

    # 比較対象となるデータの長さを決定
    n_frames_to_process = min(num_frames - args.frame_offset, green_flow.shape[0])
    
    if n_frames_to_process > 0:
        print("Loading data into memory for vectorized calculation...")
        # 必要なフレーム分のデータだけを一括でnumpy配列に読み込む
        im_theta_data = im_theta[args.frame_offset : args.frame_offset + n_frames_to_process]
        flow_data = green_flow[:n_frames_to_process]

        # im_theta から単位ベクトルを生成 (全フレーム一括)
        ui = np.cos(im_theta_data)
        vi = np.sin(im_theta_data)
        
        # flowの取得 (全フレーム一括)
        if flow_is_full_res:
            move_x = flow_data[:, target_Y, target_X, 0]
            move_y = flow_data[:, target_Y, target_X, 1]
        else:
            move_x = flow_data[:, :, :, 0]
            move_y = flow_data[:, :, :, 1]
            
        # 内積による極性判定 (全フレーム一括)
        dot_product = ui * move_x + vi * move_y
        mask = dot_product < 0
        
        u_polar = np.where(mask, -ui, ui)
        v_polar = np.where(mask, -vi, vi)
        
        # 偏心度（eccentricity）による重み付けやマスクを適用する場合
        
        u_polar[np.isnan(u_polar)] = 0
        v_polar[np.isnan(v_polar)] = 0
        
        # 指定サイズのウィンドウで局所的な平均ベクトルを計算
        print("Calculating spatial local average...")
        for size in tqdm(local_sizes, desc="Window Sizes"):
            # 時間方向(フレーム)は平均化せず、空間方向(Y, X)のみでフィルタを掛けるため、sizeを(1, size, size)とする
            local_mean_u = uniform_filter(u_polar, size=(1, size, size))
            local_mean_v = uniform_filter(v_polar, size=(1, size, size))
            
            # 局所的なポーラー度を一括計算
            local_p_val = np.sqrt(local_mean_u**2 + local_mean_v**2)
            
            # 結果を該当フレームにセット
            local_polar_orders[size][args.frame_offset : args.frame_offset + n_frames_to_process] = local_p_val

    # 保存処理とスウィープ結果の集計（生データの書き出し）
    summary_data = []
    
    # 全部入りの生データを保存するための辞書（長さが揃わないので後でNaN埋めする）
    all_raw_data = {}
    max_len = 0

    for size in local_sizes:
        output_zarr_path = base_path / f"local_polar_order_w{size}.zarr"
        zarr.save(str(output_zarr_path), local_polar_orders[size])
        
        # 0フレーム目はスキップ
        valid_orders = local_polar_orders[size][args.frame_offset:] 
        
        # 背景として追加された0やNaNを除外して1次元配列にする
        valid_1d = valid_orders[(valid_orders > 0) & (~np.isnan(valid_orders))].flatten()
        
        all_raw_data[f'w={size}'] = valid_1d
        if len(valid_1d) > max_len:
            max_len = len(valid_1d)
            
        mean_polar = np.nanmean(valid_1d)
        print(f"Saved w={size} successfully to {output_zarr_path} (Mean: {mean_polar:.4f}, Valid points: {len(valid_1d)})")

    # CSV出力用に不足分を NaN でパディングして DataFrame の列の長さを揃える
    print("Preparing raw data CSV... (This may take a moment for large data)")
    for key in all_raw_data:
        arr = all_raw_data[key]
        if len(arr) < max_len:
            pad_width = max_len - len(arr)
            all_raw_data[key] = np.pad(arr, (0, pad_width), constant_values=np.nan)

    # 生データのCSVファイルへの書き出し
    df_raw = pd.DataFrame(all_raw_data)
    csv_path = base_path / "polar_order_sweep_raw_data.csv"
    df_raw.to_csv(csv_path, index=False)
    print(f"\nSaved raw data of polar parameter sweep to {csv_path}")

if __name__ == "__main__":
    main()
