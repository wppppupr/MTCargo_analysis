import numpy as np
import scipy.fft
import os
import warnings

HAS_TORCH = False
TORCH_CUDA = False
try:
    import torch
    HAS_TORCH = True
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    pass


def create_spatial_kernel(r, kernel_type='disk', shell_width=2.0):
    """
    Create a 2D spatial convolution kernel for radius/distance r.
    """
    if r == 0:
        return np.ones((1, 1), dtype=np.float32)

    if kernel_type == 'disk':
        max_r = int(np.ceil(r))
        ksize = 2 * max_r + 1
        center = max_r
        ky, kx = np.ogrid[:ksize, :ksize]
        kmask = ((kx - center)**2 + (ky - center)**2) <= r**2
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[kmask] = 1.0
    elif kernel_type == 'ring':
        half_w = float(shell_width) / 2.0
        max_r = int(np.ceil(r + half_w))
        ksize = 2 * max_r + 1
        center = max_r
        ky, kx = np.ogrid[:ksize, :ksize]
        dist = np.hypot(kx - center, ky - center)
        kmask = (dist >= (r - half_w)) & (dist <= (r + half_w))
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if np.any(kmask):
            kernel[kmask] = 1.0
        else:
            closest_idx = np.unravel_index(np.argmin(np.abs(dist - r)), dist.shape)
            kernel[closest_idx] = 1.0
    elif kernel_type == 'gaussian':
        sigma = float(r)
        max_r = int(np.ceil(3 * sigma))
        ksize = 2 * max_r + 1
        center = max_r
        ky, kx = np.ogrid[:ksize, :ksize]
        dist_sq = (kx - center)**2 + (ky - center)**2
        kernel = np.exp(-dist_sq / (2.0 * sigma**2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    return kernel


class FFTConvolver:
    """
    High performance 2D FFT Convolution Engine.
    - Precomputes kernel FFTs on CPU RAM.
    - Uses CUDA if available and sufficient free memory, else uses optimized multithreaded SciPy CPU.
    """
    def __init__(self, shape, sizes, kernel_type='disk', shell_width=2.0, device=None):
        self.H, self.W = shape
        self.sizes = list(sizes)
        self.kernel_type = kernel_type
        self.shell_width = shell_width

        # Check GPU free memory
        if device is None:
            if HAS_TORCH and TORCH_CUDA:
                try:
                    # Check free memory (need at least 200MB free)
                    free_mem = torch.cuda.mem_get_info()[0] / (1024**2)
                    if free_mem > 200:
                        self.device_type = 'cuda'
                    else:
                        self.device_type = 'scipy'
                except Exception:
                    self.device_type = 'scipy'
            else:
                self.device_type = 'scipy'
        else:
            self.device_type = device

        self.device = torch.device(self.device_type) if (HAS_TORCH and self.device_type == 'cuda') else None

        self._precompute_kernels()

    def _precompute_kernels(self):
        """Precompute 2D Real-FFT of all kernels."""
        self.kernel_ffts_np = []
        self.kernel_ffts_torch = []

        for s in self.sizes:
            kernel = create_spatial_kernel(s, kernel_type=self.kernel_type, shell_width=self.shell_width)
            kh, kw = kernel.shape
            r_y = kh // 2
            r_x = kw // 2

            full_k = np.zeros((self.H, self.W), dtype=np.float32)
            full_k[:kh - r_y, :kw - r_x] = kernel[r_y:, r_x:]
            full_k[:kh - r_y, -(r_x):] = kernel[r_y:, :r_x]
            full_k[-(r_y):, :kw - r_x] = kernel[:r_y, r_x:]
            full_k[-(r_y):, -(r_x):] = kernel[:r_y, :r_x]

            k_fft_np = scipy.fft.rfft2(full_k, workers=4)
            self.kernel_ffts_np.append(k_fft_np)

            if HAS_TORCH:
                self.kernel_ffts_torch.append(torch.from_numpy(k_fft_np))

    def convolve_and_sample_polar(self, m_ux, m_uy, valid_mask=None, y_idx=None, x_idx=None, roi_slice=None):
        num_sizes = len(self.sizes)
        num_p = len(y_idx) if y_idx is not None else 0

        p_vals = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None
        roi_p_vals = np.empty((num_sizes,), dtype=np.float32) if roi_slice is not None else None

        if self.device_type == 'cuda':
            try:
                with torch.no_grad():
                    d_ux = torch.from_numpy(m_ux).to(self.device, non_blocking=True)
                    d_uy = torch.from_numpy(m_uy).to(self.device, non_blocking=True)
                    fft_ux = torch.fft.rfft2(d_ux)
                    fft_uy = torch.fft.rfft2(d_uy)

                    if valid_mask is not None:
                        d_mask = torch.from_numpy(valid_mask).to(self.device, non_blocking=True)
                        fft_mask = torch.fft.rfft2(d_mask)
                    else:
                        fft_mask = None

                    if num_p > 0:
                        t_y = torch.from_numpy(y_idx).to(self.device, non_blocking=True)
                        t_x = torch.from_numpy(x_idx).to(self.device, non_blocking=True)

                    for idx, k_fft_cpu in enumerate(self.kernel_ffts_torch):
                        k_fft = k_fft_cpu.to(self.device, non_blocking=True)
                        u_conv = torch.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W))
                        v_conv = torch.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W))

                        if fft_mask is not None:
                            v_mask = torch.fft.irfft2(fft_mask * k_fft, s=(self.H, self.W))
                            if num_p > 0:
                                mask_p = v_mask[t_y, t_x]
                                u_p = u_conv[t_y, t_x] / torch.where(mask_p > 1e-6, mask_p, torch.tensor(1.0, device=self.device))
                                v_p = v_conv[t_y, t_x] / torch.where(mask_p > 1e-6, mask_p, torch.tensor(1.0, device=self.device))
                                p_vals[idx, :] = torch.hypot(u_p, v_p).cpu().numpy()
                            if roi_slice is not None:
                                inv_m = torch.where(v_mask > 1e-6, 1.0 / v_mask, 0.0)
                                p_f = torch.hypot(u_conv * inv_m, v_conv * inv_m)
                                roi_p_vals[idx] = float(torch.nanmean(p_f[roi_slice[0], roi_slice[1]]).cpu().numpy())
                        else:
                            if num_p > 0:
                                p_vals[idx, :] = torch.hypot(u_conv[t_y, t_x], v_conv[t_y, t_x]).cpu().numpy()
                            if roi_slice is not None:
                                p_f = torch.hypot(u_conv, v_conv)
                                roi_p_vals[idx] = float(torch.nanmean(p_f[roi_slice[0], roi_slice[1]]).cpu().numpy())

                return p_vals, roi_p_vals
            except torch.OutOfMemoryError:
                warnings.warn("GPU Out of Memory. Falling back to multi-threaded CPU.")
                torch.cuda.empty_cache()
                self.device_type = 'scipy'

        # Fast Multi-threaded SciPy CPU
        fft_ux = scipy.fft.rfft2(m_ux, workers=4)
        fft_uy = scipy.fft.rfft2(m_uy, workers=4)
        fft_mask = scipy.fft.rfft2(valid_mask, workers=4) if valid_mask is not None else None

        for idx, k_fft in enumerate(self.kernel_ffts_np):
            u_conv = scipy.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W), workers=4)
            v_conv = scipy.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W), workers=4)

            if fft_mask is not None:
                v_mask = scipy.fft.irfft2(fft_mask * k_fft, s=(self.H, self.W), workers=4)
                if num_p > 0:
                    mask_p = v_mask[y_idx, x_idx]
                    denom = np.where(mask_p > 1e-6, mask_p, 1.0)
                    u_p = u_conv[y_idx, x_idx] / denom
                    v_p = v_conv[y_idx, x_idx] / denom
                    p_vals[idx, :] = np.hypot(u_p, v_p)
                if roi_slice is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        inv_m = np.where(v_mask > 1e-6, 1.0 / v_mask, 0.0)
                    p_f = np.hypot(u_conv * inv_m, v_conv * inv_m)
                    roi_p_vals[idx] = np.nanmean(p_f[roi_slice[0], roi_slice[1]])
            else:
                if num_p > 0:
                    p_vals[idx, :] = np.hypot(u_conv[y_idx, x_idx], v_conv[y_idx, x_idx])
                if roi_slice is not None:
                    p_f = np.hypot(u_conv, v_conv)
                    roi_p_vals[idx] = np.nanmean(p_f[roi_slice[0], roi_slice[1]])

        return p_vals, roi_p_vals

    def convolve_and_sample_bg_polar(self, m_ux, m_uy, safe_masks):
        num_sizes = len(self.sizes)
        p_vals = np.empty((num_sizes,), dtype=np.float32)

        if self.device_type == 'cuda':
            try:
                with torch.no_grad():
                    d_ux = torch.from_numpy(m_ux).to(self.device, non_blocking=True)
                    d_uy = torch.from_numpy(m_uy).to(self.device, non_blocking=True)
                    fft_ux = torch.fft.rfft2(d_ux)
                    fft_uy = torch.fft.rfft2(d_uy)

                    for idx, k_fft_cpu in enumerate(self.kernel_ffts_torch):
                        k_fft = k_fft_cpu.to(self.device, non_blocking=True)
                        u_conv = torch.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W))
                        v_conv = torch.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W))
                        p_field = torch.hypot(u_conv, v_conv)

                        mask = safe_masks[idx] if isinstance(safe_masks, list) else safe_masks
                        t_mask = torch.from_numpy(mask).to(self.device, non_blocking=True)
                        if torch.any(t_mask):
                            p_vals[idx] = float(torch.mean(p_field[t_mask]).cpu().numpy())
                        else:
                            p_vals[idx] = np.nan
                return p_vals
            except torch.OutOfMemoryError:
                warnings.warn("GPU Out of Memory. Falling back to multi-threaded CPU.")
                torch.cuda.empty_cache()
                self.device_type = 'scipy'

        fft_ux = scipy.fft.rfft2(m_ux, workers=4)
        fft_uy = scipy.fft.rfft2(m_uy, workers=4)
        for idx, k_fft in enumerate(self.kernel_ffts_np):
            u_conv = scipy.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W), workers=4)
            v_conv = scipy.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W), workers=4)
            p_field = np.hypot(u_conv, v_conv)
            mask = safe_masks[idx] if isinstance(safe_masks, list) else safe_masks
            if np.any(mask):
                p_vals[idx] = np.mean(p_field[mask])
            else:
                p_vals[idx] = np.nan

        return p_vals

    def convolve_and_sample_angular_correlation(
        self, m_ux, m_uy, valid_mask=None,
        y_idx=None, x_idx=None, center_flow_ux=None, center_flow_uy=None,
        b_ux=None, b_uy=None, theta=0.0, roi_slice=None
    ):
        """
        Convolve and sample angular spatial correlation with decomposition into
        1st principal component (Parallel to nematic axis theta) and
        2nd principal component (Perpendicular to nematic axis).
        """
        num_sizes = len(self.sizes)
        num_p = len(y_idx) if y_idx is not None else 0

        p_corr_flow = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None
        p_corr_flow_par = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None
        p_corr_flow_perp = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None

        p_corr_bead = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None
        p_corr_bead_par = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None
        p_corr_bead_perp = np.empty((num_sizes, num_p), dtype=np.float32) if num_p > 0 else None

        roi_corr_vals = np.empty((num_sizes,), dtype=np.float32) if roi_slice is not None else None
        roi_corr_par = np.empty((num_sizes,), dtype=np.float32) if roi_slice is not None else None
        roi_corr_perp = np.empty((num_sizes,), dtype=np.float32) if roi_slice is not None else None

        cos_th = float(np.cos(theta))
        sin_th = float(np.sin(theta))

        if self.device_type == 'cuda':
            try:
                with torch.no_grad():
                    d_ux = torch.from_numpy(m_ux).to(self.device, non_blocking=True)
                    d_uy = torch.from_numpy(m_uy).to(self.device, non_blocking=True)
                    fft_ux = torch.fft.rfft2(d_ux)
                    fft_uy = torch.fft.rfft2(d_uy)

                    if valid_mask is not None:
                        d_mask = torch.from_numpy(valid_mask).to(self.device, non_blocking=True)
                        fft_mask = torch.fft.rfft2(d_mask)
                    else:
                        fft_mask = None

                    t_cos = torch.tensor(cos_th, dtype=torch.float32, device=self.device)
                    t_sin = torch.tensor(sin_th, dtype=torch.float32, device=self.device)

                    if num_p > 0:
                        t_y = torch.from_numpy(y_idx).to(self.device, non_blocking=True)
                        t_x = torch.from_numpy(x_idx).to(self.device, non_blocking=True)
                        t_c_ux = torch.from_numpy(center_flow_ux).to(self.device, non_blocking=True)
                        t_c_uy = torch.from_numpy(center_flow_uy).to(self.device, non_blocking=True)
                        t_b_ux = torch.from_numpy(b_ux).to(self.device, non_blocking=True)
                        t_b_uy = torch.from_numpy(b_uy).to(self.device, non_blocking=True)

                        # Project center flow and bead directions onto parallel and perpendicular axes
                        c_flow_par = t_c_ux * t_cos + t_c_uy * t_sin
                        c_flow_perp = -t_c_ux * t_sin + t_c_uy * t_cos

                        b_par = t_b_ux * t_cos + t_b_uy * t_sin
                        b_perp = -t_b_ux * t_sin + t_b_uy * t_cos

                    for idx, k_fft_cpu in enumerate(self.kernel_ffts_torch):
                        k_fft = k_fft_cpu.to(self.device, non_blocking=True)
                        u_conv = torch.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W))
                        v_conv = torch.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W))

                        if fft_mask is not None:
                            v_mask = torch.fft.irfft2(fft_mask * k_fft, s=(self.H, self.W))
                            if num_p > 0:
                                mask_p = v_mask[t_y, t_x]
                                denom = torch.where(mask_p > 1e-6, mask_p, torch.tensor(1.0, device=self.device))
                                avg_ux_p = u_conv[t_y, t_x] / denom
                                avg_uy_p = v_conv[t_y, t_x] / denom

                                avg_u_par = avg_ux_p * t_cos + avg_uy_p * t_sin
                                avg_u_perp = -avg_ux_p * t_sin + avg_uy_p * t_cos

                                p_corr_flow[idx, :] = (t_c_ux * avg_ux_p + t_c_uy * avg_uy_p).cpu().numpy()
                                p_corr_flow_par[idx, :] = (c_flow_par * avg_u_par).cpu().numpy()
                                p_corr_flow_perp[idx, :] = (c_flow_perp * avg_u_perp).cpu().numpy()

                                p_corr_bead[idx, :] = (t_b_ux * avg_ux_p + t_b_uy * avg_uy_p).cpu().numpy()
                                p_corr_bead_par[idx, :] = (b_par * avg_u_par).cpu().numpy()
                                p_corr_bead_perp[idx, :] = (b_perp * avg_u_perp).cpu().numpy()

                            if roi_slice is not None:
                                inv_m = torch.where(v_mask > 1e-6, 1.0 / v_mask, 0.0)
                                u_conv_m = u_conv * inv_m
                                v_conv_m = v_conv * inv_m
                                corr_map = d_ux * u_conv_m + d_uy * v_conv_m
                                corr_map_par = (d_ux * t_cos + d_uy * t_sin) * (u_conv_m * t_cos + v_conv_m * t_sin)
                                corr_map_perp = (-d_ux * t_sin + d_uy * t_cos) * (-u_conv_m * t_sin + v_conv_m * t_cos)

                                roi_corr_vals[idx] = float(torch.nanmean(corr_map[roi_slice[0], roi_slice[1]]).cpu().numpy())
                                roi_corr_par[idx] = float(torch.nanmean(corr_map_par[roi_slice[0], roi_slice[1]]).cpu().numpy())
                                roi_corr_perp[idx] = float(torch.nanmean(corr_map_perp[roi_slice[0], roi_slice[1]]).cpu().numpy())
                        else:
                            if num_p > 0:
                                avg_ux_p = u_conv[t_y, t_x]
                                avg_uy_p = v_conv[t_y, t_x]

                                avg_u_par = avg_ux_p * t_cos + avg_uy_p * t_sin
                                avg_u_perp = -avg_ux_p * t_sin + avg_uy_p * t_cos

                                p_corr_flow[idx, :] = (t_c_ux * avg_ux_p + t_c_uy * avg_uy_p).cpu().numpy()
                                p_corr_flow_par[idx, :] = (c_flow_par * avg_u_par).cpu().numpy()
                                p_corr_flow_perp[idx, :] = (c_flow_perp * avg_u_perp).cpu().numpy()

                                p_corr_bead[idx, :] = (t_b_ux * avg_ux_p + t_b_uy * avg_uy_p).cpu().numpy()
                                p_corr_bead_par[idx, :] = (b_par * avg_u_par).cpu().numpy()
                                p_corr_bead_perp[idx, :] = (b_perp * avg_u_perp).cpu().numpy()

                            if roi_slice is not None:
                                corr_map = d_ux * u_conv + d_uy * v_conv
                                corr_map_par = (d_ux * t_cos + d_uy * t_sin) * (u_conv * t_cos + v_conv * t_sin)
                                corr_map_perp = (-d_ux * t_sin + d_uy * t_cos) * (-u_conv * t_sin + v_conv * t_cos)

                                roi_corr_vals[idx] = float(torch.nanmean(corr_map[roi_slice[0], roi_slice[1]]).cpu().numpy())
                                roi_corr_par[idx] = float(torch.nanmean(corr_map_par[roi_slice[0], roi_slice[1]]).cpu().numpy())
                                roi_corr_perp[idx] = float(torch.nanmean(corr_map_perp[roi_slice[0], roi_slice[1]]).cpu().numpy())

                return {
                    'flow_total': p_corr_flow, 'flow_par': p_corr_flow_par, 'flow_perp': p_corr_flow_perp,
                    'bead_total': p_corr_bead, 'bead_par': p_corr_bead_par, 'bead_perp': p_corr_bead_perp,
                    'roi_total': roi_corr_vals, 'roi_par': roi_corr_par, 'roi_perp': roi_corr_perp
                }
            except torch.OutOfMemoryError:
                warnings.warn("GPU Out of Memory. Falling back to multi-threaded CPU.")
                torch.cuda.empty_cache()
                self.device_type = 'scipy'

        # Fast Multi-threaded SciPy CPU
        fft_ux = scipy.fft.rfft2(m_ux, workers=4)
        fft_uy = scipy.fft.rfft2(m_uy, workers=4)
        fft_mask = scipy.fft.rfft2(valid_mask, workers=4) if valid_mask is not None else None

        if num_p > 0:
            c_flow_par = center_flow_ux * cos_th + center_flow_uy * sin_th
            c_flow_perp = -center_flow_ux * sin_th + center_flow_uy * cos_th

            b_par = b_ux * cos_th + b_uy * sin_th
            b_perp = -b_ux * sin_th + b_uy * cos_th

        for idx, k_fft in enumerate(self.kernel_ffts_np):
            u_conv = scipy.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W), workers=4)
            v_conv = scipy.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W), workers=4)

            if fft_mask is not None:
                v_mask = scipy.fft.irfft2(fft_mask * k_fft, s=(self.H, self.W), workers=4)
                if num_p > 0:
                    mask_p = v_mask[y_idx, x_idx]
                    denom = np.where(mask_p > 1e-6, mask_p, 1.0)
                    avg_ux_p = u_conv[y_idx, x_idx] / denom
                    avg_uy_p = v_conv[y_idx, x_idx] / denom

                    avg_u_par = avg_ux_p * cos_th + avg_uy_p * sin_th
                    avg_u_perp = -avg_ux_p * sin_th + avg_uy_p * cos_th

                    p_corr_flow[idx, :] = center_flow_ux * avg_ux_p + center_flow_uy * avg_uy_p
                    p_corr_flow_par[idx, :] = c_flow_par * avg_u_par
                    p_corr_flow_perp[idx, :] = c_flow_perp * avg_u_perp

                    p_corr_bead[idx, :] = b_ux * avg_ux_p + b_uy * avg_uy_p
                    p_corr_bead_par[idx, :] = b_par * avg_u_par
                    p_corr_bead_perp[idx, :] = b_perp * avg_u_perp

                if roi_slice is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        inv_m = np.where(v_mask > 1e-6, 1.0 / v_mask, 0.0)
                    u_conv_m = u_conv * inv_m
                    v_conv_m = v_conv * inv_m
                    corr_map = m_ux * u_conv_m + m_uy * v_conv_m
                    corr_map_par = (m_ux * cos_th + m_uy * sin_th) * (u_conv_m * cos_th + v_conv_m * sin_th)
                    corr_map_perp = (-m_ux * sin_th + m_uy * cos_th) * (-u_conv_m * sin_th + v_conv_m * cos_th)

                    roi_corr_vals[idx] = np.nanmean(corr_map[roi_slice[0], roi_slice[1]])
                    roi_corr_par[idx] = np.nanmean(corr_map_par[roi_slice[0], roi_slice[1]])
                    roi_corr_perp[idx] = np.nanmean(corr_map_perp[roi_slice[0], roi_slice[1]])
            else:
                if num_p > 0:
                    avg_ux_p = u_conv[y_idx, x_idx]
                    avg_uy_p = v_conv[y_idx, x_idx]

                    avg_u_par = avg_ux_p * cos_th + avg_uy_p * sin_th
                    avg_u_perp = -avg_ux_p * sin_th + avg_uy_p * cos_th

                    p_corr_flow[idx, :] = center_flow_ux * avg_ux_p + center_flow_uy * avg_uy_p
                    p_corr_flow_par[idx, :] = c_flow_par * avg_u_par
                    p_corr_flow_perp[idx, :] = c_flow_perp * avg_u_perp

                    p_corr_bead[idx, :] = b_ux * avg_ux_p + b_uy * avg_uy_p
                    p_corr_bead_par[idx, :] = b_par * avg_u_par
                    p_corr_bead_perp[idx, :] = b_perp * avg_u_perp

                if roi_slice is not None:
                    corr_map = m_ux * u_conv + m_uy * v_conv
                    corr_map_par = (m_ux * cos_th + m_uy * sin_th) * (u_conv * cos_th + v_conv * sin_th)
                    corr_map_perp = (-m_ux * sin_th + m_uy * cos_th) * (-u_conv * sin_th + v_conv * cos_th)

                    roi_corr_vals[idx] = np.nanmean(corr_map[roi_slice[0], roi_slice[1]])
                    roi_corr_par[idx] = np.nanmean(corr_map_par[roi_slice[0], roi_slice[1]])
                    roi_corr_perp[idx] = np.nanmean(corr_map_perp[roi_slice[0], roi_slice[1]])

        return {
            'flow_total': p_corr_flow, 'flow_par': p_corr_flow_par, 'flow_perp': p_corr_flow_perp,
            'bead_total': p_corr_bead, 'bead_par': p_corr_bead_par, 'bead_perp': p_corr_bead_perp,
            'roi_total': roi_corr_vals, 'roi_par': roi_corr_par, 'roi_perp': roi_corr_perp
        }

    def convolve_and_sample_bg_angular_correlation(self, m_ux, m_uy, roi_y, roi_x, theta=0.0):
        num_sizes = len(self.sizes)
        c_vals = np.empty((num_sizes,), dtype=np.float32)
        c_vals_par = np.empty((num_sizes,), dtype=np.float32)
        c_vals_perp = np.empty((num_sizes,), dtype=np.float32)

        center_ux = float(m_ux[roi_y, roi_x])
        center_uy = float(m_uy[roi_y, roi_x])

        cos_th = float(np.cos(theta))
        sin_th = float(np.sin(theta))

        c_par = center_ux * cos_th + center_uy * sin_th
        c_perp = -center_ux * sin_th + center_uy * cos_th

        if self.device_type == 'cuda':
            try:
                with torch.no_grad():
                    d_ux = torch.from_numpy(m_ux).to(self.device, non_blocking=True)
                    d_uy = torch.from_numpy(m_uy).to(self.device, non_blocking=True)
                    fft_ux = torch.fft.rfft2(d_ux)
                    fft_uy = torch.fft.rfft2(d_uy)

                    for idx, k_fft_cpu in enumerate(self.kernel_ffts_torch):
                        k_fft = k_fft_cpu.to(self.device, non_blocking=True)
                        u_conv = torch.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W))
                        v_conv = torch.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W))

                        avg_ux = float(u_conv[roi_y, roi_x].cpu().numpy())
                        avg_uy = float(v_conv[roi_y, roi_x].cpu().numpy())

                        avg_u_par = avg_ux * cos_th + avg_uy * sin_th
                        avg_u_perp = -avg_ux * sin_th + avg_uy * cos_th

                        c_vals[idx] = center_ux * avg_ux + center_uy * avg_uy
                        c_vals_par[idx] = c_par * avg_u_par
                        c_vals_perp[idx] = c_perp * avg_u_perp

                return {
                    'bg_total': c_vals,
                    'bg_par': c_vals_par,
                    'bg_perp': c_vals_perp
                }
            except torch.OutOfMemoryError:
                warnings.warn("GPU Out of Memory. Falling back to multi-threaded CPU.")
                torch.cuda.empty_cache()
                self.device_type = 'scipy'

        fft_ux = scipy.fft.rfft2(m_ux, workers=4)
        fft_uy = scipy.fft.rfft2(m_uy, workers=4)

        for idx, k_fft in enumerate(self.kernel_ffts_np):
            u_conv = scipy.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W), workers=4)
            v_conv = scipy.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W), workers=4)

            avg_ux = u_conv[roi_y, roi_x]
            avg_uy = v_conv[roi_y, roi_x]

            avg_u_par = avg_ux * cos_th + avg_uy * sin_th
            avg_u_perp = -avg_ux * sin_th + avg_uy * cos_th

            c_vals[idx] = center_ux * avg_ux + center_uy * avg_uy
            c_vals_par[idx] = c_par * avg_u_par
            c_vals_perp[idx] = c_perp * avg_u_perp

        return {
            'bg_total': c_vals,
            'bg_par': c_vals_par,
            'bg_perp': c_vals_perp
        }
