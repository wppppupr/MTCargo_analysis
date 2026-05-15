import tifffile
import cv2
import h5py
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

##########################################################################################
# 移動量 18px に最適化したパラメータ
params = dict(
    pyr_scale = 0.5, 
    levels = 7,          # 階層を増やす（18pxなら5層以上必要）
    winsize = 21,        # 移動量よりも大きいサイズを確保。2倍程度が安定
    iterations = 10, 
    poly_n = 7,          # 大きな動きにはより滑らかな近似（7）が有効
    poly_sigma = 1
)

##########################################################################################

def calc_optical_flow(tif_folder, output_h5):
    tif_folder = Path(tif_folder)
    output_h5 = Path(output_h5)

    print(f'Calculating optical flow for TIFF files in: {tif_folder}')

    # 1. 画像のリストアップと準備
    files = sorted(list(tif_folder.glob('*.tif')))
    if not files:
        print("No TIFF files found.")
        return

    # 最初の1枚を読んでサイズを取得
    sample_img = tifffile.imread(files[0])
    h, w = sample_img.shape[:2]
    num_pairs = len(files) - 1

    # 2. HDF5ファイルを開いてデータセットを準備
    with h5py.File(output_h5, 'w') as h5f:
        # 形状を計算結果に合わせて (T, H, W, 2) にするか、(T, 2, H, W) にするか統一する
        # ここでは後処理のしやすさを考え (T, 2, H, W) で作成
        dset = h5f.create_dataset('flows', 
                                  shape=(num_pairs, 2, h, w), 
                                  dtype=np.float16, 
                                  chunks=(1, 2, h, w), # 1フレーム単位でチャンク化
                                  compression='lzf')
        
        current_flow = np.zeros((h, w, 2), np.float32)
        
        # 3. 1フレームずつ読み込みながら計算して書き込む (メモリに優しい)
        prev_img = tifffile.imread(files[0])
        if prev_img.dtype != np.uint8:
            prev_img = cv2.normalize(prev_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        for i in tqdm(range(num_pairs), desc="Computing & Saving Flow"):
            next_img = tifffile.imread(files[i+1])
            if next_img.dtype != np.uint8:
                next_img = cv2.normalize(next_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            flow_flags = 0 if i == 0 else cv2.OPTFLOW_USE_INITIAL_FLOW
            
            current_flow = cv2.calcOpticalFlowFarneback(
                prev_img, 
                next_img, 
                current_flow, 
                **params, 
                flags=flow_flags
            )
            
            # 1フレーム分を HDF5 に書き込む（float32 -> float16 に自動変換）
            dset[i] = current_flow.transpose(2, 0, 1)
            
            # 次のループのために画像を入れ替え
            prev_img = next_img

    print(f'Done. Saved to {output_h5}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate optical flow and save as h5.')
    parser.add_argument('input_dir', type=str, help='Path to the input TIFF folder.')
    parser.add_argument('output_file', type=str, help='Path to the output H5 file.')
    args = parser.parse_args()

    calc_optical_flow(args.input_dir, args.output_file)