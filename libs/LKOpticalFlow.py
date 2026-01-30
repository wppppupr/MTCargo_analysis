import nd2
import zarr
import numpy as np
import dask.array as da
from scipy.ndimage import convolve
from tqdm import tqdm

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

def lk_opt_flow(images, xy_sig1, t_sig, w_sig, chunk_size=100, output=None):
    """
    Lucas-Kanade法によるオプティカルフロー計算のPython移植版 (Optimized)
    
    Parameters:
    images : np.ndarray or dask.array (T, Y, X) - 入力画像シーケンス
    xy_sig1: float - 空間方向の勾配計算用シグマ
    t_sig  : float - 時間方向の勾配計算用シグマ
    w_sig  : float - 構造テンソルのスムージング用シグマ
    chunk_size: int - 時間方向のチャンクサイズ
    output : zarr.Array or None - 出力先
    """

    n_frames, height, width = images.shape
    
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

    # --- 1. フィルタ作成 (T, Y, X) 用 ---
    
    # Ix (Horizontal) Calculation: Smooth Y (Axis 1) * Deriv X (Axis 2)
    # Iy (Vertical) Calculation: Deriv Y (Axis 1) * Smooth X (Axis 2)
    
    # Common 1D kernels
    # For Deriv: Use xy_sig1
    # For Smooth: Use xy_sig2
    
    # X-kernels (Axis 2)
    fx_deriv, x_range = get_gaussian_kernels(xy_sig1, xy_sig1, deriv=True) # Deriv X
    fx_smooth, _ = get_gaussian_kernels(xy_sig2, xy_sig2, deriv=False)     # Smooth X

    # Y-kernels (Axis 1)
    fy_deriv, y_range = get_gaussian_kernels(xy_sig1, xy_sig1, deriv=True) # Deriv Y
    fy_smooth, _ = get_gaussian_kernels(xy_sig2, xy_sig2, deriv=False)     # Smooth Y

    # T-kernels (Axis 0)
    # For It: Smooth Y * Smooth X * Deriv T
    ft_deriv, t_range = get_gaussian_kernels(t_sig, t_sig, deriv=True)     # Deriv T

    # Also need Smooth Y and Smooth X with sig1 for It calculation?
    # Original code used:
    # dt_i = convolve(convolve(convolve(images, x_fil3), y_fil3), t_fil3)
    # x_fil3, y_fil3 used xy_sig1. t_fil3 used t_sig.
    # So It uses Smooth(sig1) on spatial axes.
    fx_smooth_sig1, _ = get_gaussian_kernels(xy_sig1, xy_sig1, deriv=False)
    fy_smooth_sig1, _ = get_gaussian_kernels(xy_sig1, xy_sig1, deriv=False)

    # Reshape kernels for (T, Y, X)
    # Axis 0: T, Axis 1: Y, Axis 2: X

    # Kernels for Ix (Horizontal gradient) -> usually denoted as fx in code, but output is dx
    # Original dx_i = convolve(convolve(images, x_fil1), y_fil1)
    # x_fil1 was "deriv" (mapped to Y in my hypothesis or T).
    # Let's stick to standard Optical Flow definitions:
    # Ix = dI/dx. Iy = dI/dy.

    # Ix: Smooth Y (sig2) * Deriv X (sig1)
    filt_ix_y = fy_smooth.reshape(1, -1, 1)
    filt_ix_x = fx_deriv.reshape(1, 1, -1)

    # Iy: Deriv Y (sig1) * Smooth X (sig2)
    filt_iy_y = fy_deriv.reshape(1, -1, 1)
    filt_iy_x = fx_smooth.reshape(1, 1, -1)

    # It: Smooth Y (sig1) * Smooth X (sig1) * Deriv T (sig)
    filt_it_y = fy_smooth_sig1.reshape(1, -1, 1)
    filt_it_x = fx_smooth_sig1.reshape(1, 1, -1)
    filt_it_t = ft_deriv.reshape(-1, 1, 1)

    # Structure tensor smoothing kernels (Spatial only)
    # w_sig
    w_range = np.arange(-np.ceil(3 * w_sig), np.ceil(3 * w_sig) + 1)
    gw = np.exp(-w_range**2 / (2 * w_sig**2)) / (np.sqrt(2 * np.pi) * w_sig)

    filt_w_y = gw.reshape(1, -1, 1)
    filt_w_x = gw.reshape(1, 1, -1)

    def apply_w_filter(data):
        return convolve(convolve(data, filt_w_y, mode='nearest'), filt_w_x, mode='nearest')

    # Prepare output
    if output is None:
        # 3 channels (vx, vy, rel)
        output = np.zeros((3, n_frames, height, width), dtype=np.float32)
        use_zarr = False
    else:
        use_zarr = True

    # Padding for temporal convolution
    t_pad = int(np.ceil(3 * t_sig))

    for t_start in tqdm(range(0, n_frames, chunk_size), desc="Calculating Optical Flow"):
        t_end = min(t_start + chunk_size, n_frames)

        # Determine load range including padding
        load_start = max(0, t_start - t_pad)
        load_end = min(n_frames, t_end + t_pad)

        # Load chunk (force float64 for precision)
        # Using slice on dask array automatically loads it to numpy
        chunk = images[load_start:load_end].astype(np.float64)

        if isinstance(images, da.Array):
            chunk = chunk.compute()

        # 2. Compute Gradients

        # Ix
        # Apply Y-smooth then X-deriv
        ix = convolve(convolve(chunk, filt_ix_y, mode='nearest'), filt_ix_x, mode='nearest')

        # Iy
        # Apply Y-deriv then X-smooth
        iy = convolve(convolve(chunk, filt_iy_y, mode='nearest'), filt_iy_x, mode='nearest')

        # It
        # Apply Y-smooth(sig1), X-smooth(sig1), then T-deriv
        # Note: T-deriv is valid only where we have enough temporal context
        it_temp = convolve(convolve(chunk, filt_it_y, mode='nearest'), filt_it_x, mode='nearest')
        it = convolve(it_temp, filt_it_t, mode='nearest')

        # 3. Structure Tensor
        wdx2 = apply_w_filter(ix * ix)
        wdxy = apply_w_filter(ix * iy)
        wdy2 = apply_w_filter(iy * iy)
        wdtx = apply_w_filter(ix * it)
        wdty = apply_w_filter(iy * it)

        # 4. Optical Flow (vx, vy)
        eps = np.finfo(float).eps
        det = (wdx2 * wdy2) - (wdxy**2)

        vx_chunk = (1.0 / (det + eps)) * (wdy2 * -wdtx + (-wdxy * -wdty))
        vy_chunk = (1.0 / (det + eps)) * (-wdxy * -wdtx + wdx2 * -wdty)

        # 5. Reliability
        trace = wdx2 + wdy2
        sqrt_disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
        e1 = (trace + sqrt_disc) / 2
        e2 = (trace - sqrt_disc) / 2
        rel_chunk = np.real(np.minimum(e1, e2))

        # Extract Valid Region (remove padding)
        # Coordinates in 'chunk' corresponding to [t_start, t_end)
        valid_start = t_start - load_start
        valid_end = valid_start + (t_end - t_start)

        vx_valid = vx_chunk[valid_start:valid_end]
        vy_valid = vy_chunk[valid_start:valid_end]
        rel_valid = rel_chunk[valid_start:valid_end]

        # Write to output
        if use_zarr:
            output[0, t_start:t_end] = vx_valid
            output[1, t_start:t_end] = vy_valid
            output[2, t_start:t_end] = rel_valid
        else:
            output[0, t_start:t_end] = vx_valid
            output[1, t_start:t_end] = vy_valid
            output[2, t_start:t_end] = rel_valid

    if not use_zarr:
        return output[0], output[1], output[2]
    else:
        return None # Output is written to zarr

if __name__ == "__main__":
    nd2_file = nd2.imread(FILE_PATH, dask=True)
    images = nd2_file[:,0,:,:]  # (T, Y, X)
    print(f"Loaded images : {FILE_PATH}, Shape: {images.shape}")

    # Output Zarr
    n_frames, h, w = images.shape
    output_zarr_path = 'optical_flow_output.zarr'
    opt_zarr = zarr.open(output_zarr_path, mode='w', shape=(3, n_frames, h, w), dtype=np.float32, chunks=(1, 10, h, w)) # Chunking T=10?

    lk_opt_flow(images, xy_sig1=XY_SIGMA, t_sig=T_SIGMA, w_sig=W_SIGMA, chunk_size=100, output=opt_zarr)

    print("Processing complete.")
