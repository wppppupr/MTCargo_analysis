import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import h5py

from tqdm import tqdm

def evaluate_optical_flow(h5_path, csv_path, scale=0.11, interval=4, out_h5=None, plot=False):
    print(f"Reading dense flows from HDF5: {h5_path}...")
    
    print(f"Loading validation tracking data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. t と t+1 のフレーム間でグラウンドトゥルース(v*)を計算
    df_next = df.copy()
    df_next['frame'] = df_next['frame'] - 1
    merged = pd.merge(df, df_next, on=['frame', 'particle'], suffixes=('_t', '_t1'))
    
    merged['gt_vx'] = merged['x_t1'] - merged['x_t']
    merged['gt_vy'] = merged['y_t1'] - merged['y_t']
    
    pred_vx_list, pred_vy_list = [], []
    gt_vx_list, gt_vy_list = [], []
    
    # HDF5ファイルをオープン (メモリに全て乗せずに遅延読み込みを行う)
    with h5py.File(h5_path, 'r') as h5f:
        if 'flows' not in h5f:
            raise KeyError(f"HDF5ファイル内に 'flows' データセットが見つかりません。")
            
        flows = h5f['flows']  # Shape: (T-1, 2, H, W)
        num_frames = flows.shape[0]
        _, H, W = flows.shape[1:]
        
        # 2. HDF5の予測フロー(v)から、トラッキングポイントの速度をサンプリング
        for t, group in tqdm(merged.groupby('frame'), desc="Sampling flow vectors"):
            if t >= num_frames:
                continue  # フレーム数を超過した場合はスキップ
                
            # ディスク(HDF5)から該当フレームのフロー配列のみをメモリに引っ張り出す (RAM節約)
            flow_t = flows[t].astype(np.float32)  # [2, H, W]
            
            # ピクセル座標を整数化し、画像範囲内にクリップ
            xs = np.clip(np.round(group['x_t'].values).astype(int), 0, W - 1)
            ys = np.clip(np.round(group['y_t'].values).astype(int), 0, H - 1)
            
            # 抽出
            pred_vx = flow_t[0, ys, xs]
            pred_vy = flow_t[1, ys, xs]
            
            pred_vx_list.extend(pred_vx)
            pred_vy_list.extend(pred_vy)
            gt_vx_list.extend(group['gt_vx'].values)
            gt_vy_list.extend(group['gt_vy'].values)
        
    # NumPy配列に変換
    v_pred_x = np.array(pred_vx_list)
    v_pred_y = np.array(pred_vy_list)
    v_gt_x = np.array(gt_vx_list)
    v_gt_y = np.array(gt_vy_list)
    
    N = len(v_gt_x)
    print(f"Total valid tracking points evaluated: {N}")
    if N == 0:
        print("No matching frames found between HDF5 and CSV.")
        return

    # 3. 各指標の計算
    # ベクトルの大きさ(ノルム)
    norm_pred = np.hypot(v_pred_x, v_pred_y)
    norm_gt = np.hypot(v_gt_x, v_gt_y)
    
    # ゼロ除算回避のためのマスク
    valid_mask = norm_gt > 1e-5
    
    # --- (1) Relative Speed Error ---
    # abs(||v|| - ||v*||) / ||v*||
    moving_mask = norm_gt > 2.0 
    if np.sum(moving_mask) > 0:
        speed_error = np.abs(norm_pred[moving_mask] - norm_gt[moving_mask]) / norm_gt[moving_mask]
        mean_speed_error = np.mean(speed_error)
    else:
        mean_speed_error = float('nan')
    
    # --- (2) Orientation Error ---
    # cos(θ) = (v · v*) / (||v|| * ||v*||)
    valid_mask = (norm_gt > 1e-3) & (norm_pred > 1e-3)
    dot_product = (v_pred_x[valid_mask] * v_gt_x[valid_mask]) + (v_pred_y[valid_mask] * v_gt_y[valid_mask])
    cos_theta = dot_product / (norm_pred[valid_mask] * norm_gt[valid_mask])
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # 計算誤差によるarccosのNaNを防止
    orientation_error_rad = np.arccos(cos_theta)
    orientation_error_deg = np.degrees(orientation_error_rad)
    mean_orientation_error = np.mean(orientation_error_deg)
    
    # --- (3) Normalized Zero-lag Cross-correlation ---
    # G = sum(v · v*) / sum(||v*||^2)
    numerator = np.sum((v_pred_x * v_gt_x) + (v_pred_y * v_gt_y))
    denominator = np.sqrt(np.sum(norm_pred**2) * np.sum(norm_gt**2))
    cross_correlation = numerator / (denominator + 1e-8)
    
    # 4. 結果の出力
    print("=" * 40)
    print(" 📊 Evaluation Results")
    print("=" * 40)
    print(f"1. Relative Speed Error : {mean_speed_error:.4f} (Lower is better)")
    print(f"2. Orientation Error    : {mean_orientation_error:.2f}° (Lower is better)")
    print(f"3. Cross-correlation (G): {cross_correlation:.4f} (Closer to 1.0 is better)")
    print("=" * 40)

    if out_h5 is None:
        out_h5 = h5_path.replace('.h5', '_metrics.h5')
        if out_h5 == h5_path:
            out_h5 = h5_path + '_metrics.h5'
    
    print(f"Saving evaluation metrics to {out_h5}...")
    # HDF5形式での保存 (npzよりロード/アクセスが高速で大規模データにも耐えられます)
    with h5py.File(out_h5, 'w') as out_f:
        out_f.create_dataset('norm_pred', data=norm_pred)
        out_f.create_dataset('norm_gt', data=norm_gt)
        out_f.create_dataset('speed_error', data=speed_error)
        out_f.create_dataset('orientation_error_deg', data=orientation_error_deg)
        out_f.create_dataset('valid_mask', data=valid_mask)

    print(f"Metrics effectively saved into {out_h5}.")

    # 5. 可視化
    # plt.style.use('/home/sasaki/opticalflow-activenematics/raft_finetune/my_style.mplstyle')
    if plot:
        # 速度の比較
        fig, ax = plt.subplots()
        ax.scatter(norm_pred * scale / interval, norm_gt * scale / interval)
        ax.set_xlabel("Optical flow velocity [\u03bcm/s]")
        ax.set_ylabel("Tracking velocity [\u03bcm/s]")
        plt.show()

        fig, ax = plt.subplots()
        edge = np.histogram_bin_edges(speed_error, bins="sturges")
        ax.hist(speed_error, bins=edge, density=False)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Relative Speed Error")
        ax.set_ylabel("Counts")
        plt.show()

        fig, ax = plt.subplots()
        edge = np.histogram_bin_edges(orientation_error_deg, bins="sturges")
        ax.hist(orientation_error_deg, bins=edge, density=False)
        ax.set_xlim(0, 180)
        ax.set_xlabel("Orientation Error (degrees)")
        ax.set_ylabel("Counts")
        plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='./dense_flows.h5', help='Path to output h5 file from inference.py')
    parser.add_argument('--csv_path', type=str, default='./val_track.csv', help='Path to validation CSV file')
    parser.add_argument('--scale', type=float, default=0.11, help='Scale factor for converting pixels to micrometers')
    parser.add_argument('--interval', type=int, default=4, help='Time interval between frames in seconds')
    parser.add_argument('--out_h5', type=str, default=None, help='Path to output metrics h5 file')
    parser.add_argument('--plot', type=bool, default=False, help='Whether to plot the results')
    args = parser.parse_args()
    
    evaluate_optical_flow(args.h5_path, args.csv_path, args.scale, args.interval, args.out_h5, args.plot)