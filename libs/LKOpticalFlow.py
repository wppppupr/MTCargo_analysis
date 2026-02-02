import nd2
import zarr
import os
import numpy as np
import dask.array as da
from scipy.ndimage import convolve
from tqdm import tqdm
import gc # メモリ解放用

####################################################

#FOLDER_PATH= '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001'
#FILE_PATH = FOLDER_PATH + '/exp001.nd2'

velocity = 0.5  # μm/sec
time_interval = 4  # sec/frame
scale = 0.11  # μm/pixel

XY_SIGMA = 1.0  # 空間方向の勾配計算用シグマ
T_SIGMA = 1.0   # 時間方向の勾配計算用シグマ
W_SIGMA = int(velocity * time_interval / scale)   # 近傍サイズ

CHUNK_SIZE = 10  # フレームチャンクサイズ

####################################################

def lk_opt_flow_optimized(images, xy_sig1, t_sig, w_sig, chunk_size=10, output=None):
    n_frames, height, width = images.shape
    xy_sig2 = xy_sig1 / 4

    def get_gaussian_kernel(sigma, deriv=False):
        # 3シグマ範囲でカーネル作成
        r = int(np.ceil(3 * sigma))
        x = np.arange(-r, r + 1)
        f = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        if deriv:
            return (f * (x / (sigma**2))).astype(np.float32)
        return f.astype(np.float32)

    # 1. フィルタ作成 (全て float32 に)
    filt_ix_y = get_gaussian_kernel(xy_sig2).reshape(1, -1, 1)
    filt_ix_x = get_gaussian_kernel(xy_sig1, deriv=True).reshape(1, 1, -1)
    filt_iy_y = get_gaussian_kernel(xy_sig1, deriv=True).reshape(1, -1, 1)
    filt_iy_x = get_gaussian_kernel(xy_sig2).reshape(1, 1, -1)
    
    filt_it_y = get_gaussian_kernel(xy_sig1).reshape(1, -1, 1)
    filt_it_x = get_gaussian_kernel(xy_sig1).reshape(1, 1, -1)
    filt_it_t = get_gaussian_kernel(t_sig, deriv=True).reshape(-1, 1, 1)

    gw = get_gaussian_kernel(w_sig)
    filt_w_y = gw.reshape(1, -1, 1)
    filt_w_x = gw.reshape(1, 1, -1)

    t_pad = int(np.ceil(3 * t_sig))
    eps = np.float32(np.finfo(np.float32).eps)

    for t_start in tqdm(range(0, n_frames, chunk_size), desc="Calculating Optical Flow"):
        t_end = min(t_start + chunk_size, n_frames)
        load_start = max(0, t_start - t_pad)
        load_end = min(n_frames, t_end + t_pad)
        
        # 対策: float32 で計算（メモリ半分、速度向上）
        chunk = images[load_start:load_end].astype(np.float32)
        if isinstance(images, da.Array):
            chunk = chunk.compute()
            
        # 2. 勾配計算（中間変数を最小限に）
        ix = convolve(convolve(chunk, filt_ix_y), filt_ix_x)
        iy = convolve(convolve(chunk, filt_iy_y), filt_iy_x)
        it = convolve(convolve(convolve(chunk, filt_it_y), filt_it_x), filt_it_t)
        del chunk # 元データは不要なので消去
        
        # 3. 構造テンソルと流速計算
        # メモリ節約のため、wdx2などの計算と同時に処理を進める
        wdx2 = convolve(convolve(ix * ix, filt_w_y), filt_w_x)
        wdy2 = convolve(convolve(iy * iy, filt_w_y), filt_w_x)
        wdxy = convolve(convolve(ix * iy, filt_w_y), filt_w_x)
        wdtx = convolve(convolve(ix * it, filt_w_y), filt_w_x)
        wdty = convolve(convolve(iy * it, filt_w_y), filt_w_x)
        del ix, iy, it

        det = (wdx2 * wdy2) - (wdxy**2)
        inv_det = 1.0 / (det + eps)
        
        valid_s = t_start - load_start
        valid_e = valid_s + (t_end - t_start)

        # 出力への書き込み
        output[0, t_start:t_end] = (inv_det * (wdy2 * -wdtx + (-wdxy * -wdty)))[valid_s:valid_e]
        output[1, t_start:t_end] = (inv_det * (-wdxy * -wdtx + wdx2 * -wdty))[valid_s:valid_e]
        
        # 信頼性（最小固有値）
        trace = wdx2 + wdy2
        sqrt_disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
        output[2, t_start:t_end] = ((trace - sqrt_disc) / 2)[valid_s:valid_e]

        # 明示的なメモリ解放
        del wdx2, wdy2, wdxy, wdtx, wdty, det, inv_det, trace, sqrt_disc
        gc.collect()

if __name__ == "__main__":

    mypass = '/media/sasaki/My Passport/Sasaki/MTSingleBeads'

    path2 = os.path.join(mypass,"20260122/exp002")
    path3 = os.path.join(mypass,"20260121/beads_trans_crop_crop")
    path4 = os.path.join(mypass,"20260121/exp_crop1")
    file_paths = [
                  path2 + '/exp002.nd2',
                  path3 + '/beads_trans_crop_crop.nd2',
                  path4 + '/exp_crop1.nd2']
    
    folder_paths = [
                    path2,
                    path3,
                    path4]
    
    for file, folder in zip(file_paths, folder_paths):
        print(f"Processing file: {file}")
        # nd2ファイルの読み込み（Dask配列として）
        nd2_file = nd2.imread(file, dask=True)
        # 軸構成に注意：通常 nd2は (T, C, Y, X)
        images = nd2_file[:, 0, :, :] 
        
        n_frames, h, w = images.shape
        output_zarr_path = os.path.join(folder, 'optical_flow_output.zarr')
        
        # 対策: Zarrのchunkも計算単位に合わせる
        opt_zarr = zarr.open(output_zarr_path, mode='w', 
                            shape=(3, n_frames, h, w), 
                            dtype=np.float32, 
                            chunks=(1, CHUNK_SIZE, h, w))
        
        lk_opt_flow_optimized(images, xy_sig1=XY_SIGMA, t_sig=T_SIGMA, w_sig=W_SIGMA, 
                            chunk_size=CHUNK_SIZE, output=opt_zarr)