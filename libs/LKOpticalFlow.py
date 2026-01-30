import nd2
import zarr
import numpy as np
import dask.array as da
from scipy.ndimage import convolve

####################################################

FOLDER_PATH= '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001'
FILE_PATH = FOLDER_PATH + '/exp001.nd2'

velocity = 0.5  # μm/sec
time_interval = 4  # sec/frame
scale = 0.11  # μm/pixel

XY_SIGMA = 1.0  # 空間方向の勾配計算用シグマ
T_SIGMA = 1.0   # 時間方向の勾配計算用シグマ
W_SIGMA = int(velocity * time_interval / scale)   # 近傍サイズ

####################################################

def lk_opt_flow(images, xy_sig1, t_sig, w_sig):
    """
    Lucas-Kanade法によるオプティカルフロー計算のPython移植版
    
    Parameters:
    images : np.ndarray (H, W, T) - 入力画像シーケンス
    xy_sig1: float - 空間方向の勾配計算用シグマ
    t_sig  : float - 時間方向の勾配計算用シグマ
    w_sig  : float - 構造テンソルのスムージング用シグマ
    """
    images = images.astype(np.float64)
    
    # フィルタパラメータの計算
    xy_sig2 = xy_sig1 / 4
    
    def get_gaussian_kernels(sigma, range_val, deriv=False):
        x = np.arange(-np.ceil(3 * range_val), np.ceil(3 * range_val) + 1)
        # ガウス分布
        f = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        if deriv:
            # 1次微分
            g = x / (sigma**2)
            return f * g, x
        return f, x

    # 1. 空間・時間勾配フィルタの作成
    # x-gradient kernels
    fx1, x_range = get_gaussian_kernels(xy_sig1, xy_sig1)
    gx1 = x_range / (xy_sig1**2)
    x_fil1 = (fx1 * gx1).reshape(-1, 1, 1)
    
    fy1, _ = get_gaussian_kernels(xy_sig2, xy_sig2)
    y_fil1 = fy1.reshape(1, -1, 1)

    # y-gradient kernels
    fx2, _ = get_gaussian_kernels(xy_sig2, xy_sig2)
    x_fil2 = fx2.reshape(-1, 1, 1)
    
    fy2, y_range = get_gaussian_kernels(xy_sig1, xy_sig1)
    gy2 = y_range / (xy_sig1**2)
    y_fil2 = (fy2 * gy2).reshape(1, -1, 1)

    # t-gradient kernels
    fx3, _ = get_gaussian_kernels(xy_sig1, xy_sig1)
    fy3, _ = get_gaussian_kernels(xy_sig1, xy_sig1)
    ft3, t_range = get_gaussian_kernels(t_sig, t_sig)
    gt3 = t_range / (t_sig**2)
    
    x_fil3 = fx3.reshape(-1, 1, 1)
    y_fil3 = fy3.reshape(1, -1, 1)
    t_fil3 = (ft3 * gt3).reshape(1, 1, -1)

    # 2. 勾配の計算 (MATLABの 'replicate' は 'nearest' に相当)
    dx_i = convolve(convolve(images, x_fil1, mode='nearest'), y_fil1, mode='nearest')
    dy_i = convolve(convolve(images, x_fil2, mode='nearest'), y_fil2, mode='nearest')
    dt_i = convolve(convolve(convolve(images, x_fil3, mode='nearest'), y_fil3, mode='nearest'), t_fil3, mode='nearest')

    # 3. 構造テンソルの計算
    w_range = np.arange(-np.ceil(3 * w_sig), np.ceil(3 * w_sig) + 1)
    gw = np.exp(-w_range**2 / (2 * w_sig**2)) / (np.sqrt(2 * np.pi) * w_sig)
    x_fil4 = gw.reshape(-1, 1, 1)
    y_fil4 = gw.reshape(1, -1, 1)

    def apply_w_filter(data):
        return convolve(convolve(data, x_fil4, mode='nearest'), y_fil4, mode='nearest')

    wdx2 = apply_w_filter(dx_i * dx_i)
    wdxy = apply_w_filter(dx_i * dy_i)
    wdy2 = apply_w_filter(dy_i * dy_i)
    wdtx = apply_w_filter(dx_i * dt_i)
    wdty = apply_w_filter(dy_i * dt_i)

    # 4. オプティカルフロー(vx, vy)の算出
    # 論文の式(6)の解法に基づき、行列式を用いて最小二乗解を求める [cite: 586, 612]
    eps = np.finfo(float).eps
    det = (wdx2 * wdy2) - (wdxy**2)
    
    vx = (1.0 / (det + eps)) * (wdy2 * -wdtx + (-wdxy * -wdty))
    vy = (1.0 / (det + eps)) * (-wdxy * -wdtx + wdx2 * -wdty)

    # 5. 信頼性(reliability)の算出
    # 構造テンソル A^T w A の最小固有値を求める [cite: 586, 613]
    trace = wdx2 + wdy2
    # 2x2行列の固有値公式: (tr ± sqrt(tr^2 - 4*det)) / 2
    sqrt_disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
    e1 = (trace + sqrt_disc) / 2
    e2 = (trace - sqrt_disc) / 2
    rel = np.real(np.minimum(e1, e2))

    return vx, vy, rel


if __name__ == "__main__":
    nd2_file = nd2.imread(FILE_PATH, dask=True)
    images = nd2_file[:,0,:,:]  # チャネル0を選択してメモリに読み込み
    print(f"Loaded images : {FILE_PATH}")
    vx, vy, rel = lk_opt_flow(images, xy_sig1=XY_SIGMA, t_sig=T_SIGMA, w_sig=W_SIGMA)
    opt_flow = np.array([vx, vy, rel])
    opt_zarr = zarr.open('optical_flow_output.zarr', mode='w', shape=opt_flow.shape, dtype=opt_flow.dtype)
    opt_zarr[:] = opt_flow