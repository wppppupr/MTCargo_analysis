import cv2
import numpy as np
import zarr
import dask.array as da
from dask import delayed, compute
import os
from tqdm import tqdm

####################################################

FILE_PATH = "/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp002/MTs_red.zarr"
####################################################

def _calc_flow_chunk(start_idx, end_idx, input_path, output_path):
    """
    指定された範囲のフレームについてオプティカルフローを計算し、
    結果を出力先Zarrに直接書き込む関数（ワーカープロセスで実行されます）
    """
    # 各プロセスでファイルを開く（Zarrは並列読み書きに強い）
    # mode='r' で読み込み専用
    in_zarr = zarr.open_array(input_path, mode='r')
    # mode='r+' で既存のファイルに書き込み
    out_zarr = zarr.open_array(output_path, mode='r+')

    # 最初のフレーム（前フレームとして使用）
    prvs = in_zarr[start_idx]
    
    # データ型等の変換が必要な場合
    if prvs.dtype != np.uint8:
        prvs = (prvs / 256).astype(np.uint8) if prvs.max() > 255 else prvs.astype(np.uint8)

    # 指定範囲をループ処理
    # flow[i] は frame[i] と frame[i+1] の間の動きを表すとする
    for i in range(start_idx, end_idx):
        # 次のフレーム
        next_frame_idx = i + 1
        if next_frame_idx >= in_zarr.shape[0]:
            break
            
        next_img = in_zarr[next_frame_idx]
        if next_img.dtype != np.uint8:
            next_img = (next_img / 256).astype(np.uint8) if next_img.max() > 255 else next_img.astype(np.uint8)

        # オプティカルフロー計算
        flow = cv2.calcOpticalFlowFarneback(
            prvs,
            next_img,
            None,
            pyr_scale=0.5, levels=3, winsize=10, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )

        # 結果をZarrに書き込み
        out_zarr[i] = flow
        
        # 更新
        prvs = next_img

    return True

def calculate_optical_flow_zarr(file_path, output_path=None, chunk_size=100):
    """
    Zarrファイルを入力とし、オプティカルフローを計算してZarrに保存します。
    Daskを用いて並列処理を行います。
    """
    # 読み込みもパスを直接指定でOKです
    input_zarr = zarr.open_array(file_path, mode='r')
    
    n_frames = input_zarr.shape[0]
    height, width = input_zarr.shape[1], input_zarr.shape[2]

    # 出力パスの設定
    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '_flow.zarr'

    print(f"Input: {file_path}")
    print(f"Output: {output_path}")
    print(f"Frames: {n_frames}, Resolution: {width}x{height}")
    print("Start parallel processing...")

    # 【修正点】DirectoryStoreを使わず、zarr.open に直接パスと設定を渡します
    # これで自動的にフォルダとして保存されます
    output_zarr = zarr.open(
        output_path, 
        mode='w', 
        shape=(n_frames, height, width, 2), 
        chunks=(10, height, width, 2), 
        dtype=np.float32
    )

    # タスクの分割
    tasks = []
    for i in range(0, n_frames - 1, chunk_size):
        end = min(i + chunk_size, n_frames - 1)
        if i >= end: break
        
        # delayedを使って関数を遅延評価オブジェクトにする
        task = delayed(_calc_flow_chunk)(i, end, file_path, output_path)
        tasks.append(task)

    # 並列実行 (プログレスバー付き)
    with tqdm(total=len(tasks)) as pbar:
        results = compute(*tasks)
        pbar.update(len(tasks))
        
    print("Calculation completed.")
    return output_path


def create_flow_movie(image_path, flow_path, output_video_name, scale=5, step=10):
    """
    計算済みの画像ZarrとフローZarrから動画を作成します。
    """
    print("Creating video...")
    img_zarr = zarr.open_array(image_path, mode='r')
    flow_zarr = zarr.open_array(flow_path, mode='r')
    
    n_frames = min(img_zarr.shape[0], flow_zarr.shape[0]) - 1
    height, width = img_zarr.shape[1], img_zarr.shape[2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, 10, (width*2, height))
    
    for i in tqdm(range(n_frames), desc="Writing frames"):
        # 画像読み込み
        frame = img_zarr[i+1] # flow[i]は i -> i+1 の動きなので、i+1の画像に重ねると自然
        if frame.dtype != np.uint8:
            frame = (frame / 256).astype(np.uint8) if frame.max() > 255 else frame.astype(np.uint8)
            
        flow = flow_zarr[i]
        
        # 左: 元画像（緑チャンネル）
        mt_img = np.zeros((height, width, 3), dtype=np.uint8)
        mt_img[..., 1] = frame 

        # 右: オプティカルフロー矢印
        flow_img = np.zeros((height, width, 3), dtype=np.uint8)
        flow_img[..., 1] = frame 

        # 矢印描画 (重い処理なので間引く)
        # ベクトル化して高速化も可能だが、cv2.arrowedLineは見やすいのでループで描画
        y_grid, x_grid = np.mgrid[0:height:step, 0:width:step]
        
        # stepごとの座標とフローを取得
        # インデックス配列を作成
        y_indices = range(0, height, step)
        x_indices = range(0, width, step)
        
        for y in y_indices:
            for x in x_indices:
                fx, fy = flow[y, x]
                # フローがある程度大きい場合のみ描画するなどのフィルタも可能
                if abs(fx) > 0.1 or abs(fy) > 0.1:
                    end_point = (int(x + fx * scale), int(y + fy * scale))
                    # 範囲外チェック
                    end_point = (max(0, min(width-1, end_point[0])), max(0, min(height-1, end_point[1])))
                    
                    cv2.arrowedLine(
                        flow_img,
                        (x, y),
                        end_point,
                        color=(0, 0, 255), 
                        thickness=1,
                        tipLength=0.3
                    )
        
        combined = np.hstack([mt_img, flow_img])
        out.write(combined)
        
    out.release()
    print(f"Video saved to {output_video_name}")

def getP(path):
    flow_path = os.path.join(path, "MTs_red_flow.zarr")
    flow_array = zarr.open_array(flow_path, mode='r')
    
    x = flow_array[:, :, :, 0]  # x成分
    y = flow_array[:, :, :, 1]  # y成分
    # ベクトルの大きさを計算
    magnitude = np.sqrt(x**2 + y**2)

    # --- 1. Polar Order Parameter (P) の計算 ---
    # 全ベクトルの平均を計算
    mean_x = np.mean(x, axis=(1,2))
    mean_y = np.mean(y, axis=(1,2))
        
    # 平均ベクトルの大きさを、全ベクトルの平均の大きさで割って正規化
    # P = |<v>| / <|v|>
    mean_magnitude_of_vectors = np.mean(magnitude, axis=(1,2))
    magnitude_of_mean_vector = np.sqrt(mean_x**2 + mean_y**2)

    P = magnitude_of_mean_vector / mean_magnitude_of_vectors
    n_nanind = np.where(~np.isnan(P))
    PNan = P[n_nanind[0]]

    polar_path = os.path.join(path, "MTs_red_polar.zarr")
    Polar_parameter = zarr.open(polar_path, mode = 'w', shape = P.shape, dtype = P.dtype)
    Polar_parameter[:] = PNan

    return PNan

if __name__ == "__main__":
    
    # 1. まず高速に計算だけ行う
    flow_zarr_path = calculate_optical_flow_zarr(FILE_PATH, chunk_size=1)
    
    # 2. 必要なら動画にする (計算結果のZarrを使う)
    video_name = FILE_PATH.replace('.zarr', '_opticalflow.mp4')
    create_flow_movie(FILE_PATH, flow_zarr_path, video_name, scale=5, step=10)


    # 3. Polar度の計算
    _ = getP(FILE_PATH)