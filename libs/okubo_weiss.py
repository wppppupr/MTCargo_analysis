"""
libs/okubo_weiss.py

微小管流場（Optical Flow）から Okubo-Weiss パラメータ Q_OW を計算し、
背景（Eulerian）と貨物軌道上（Lagrangian）のサンプリング分布を解析するためのライブラリモジュール。

Okubo-Weiss パラメータの定義:
    s_n = ∂u/∂x - ∂v/∂y   (normal strain rate: 通常伸縮歪み)
    s_s = ∂v/∂x + ∂u/∂y   (shear strain rate: 剪断歪み)
    ω   = ∂v/∂x - ∂u/∂y   (vorticity: 渦度)
    s^2 = s_n^2 + s_s^2   (total strain rate squared: 総歪み率の二乗)
    Q_OW = s_n^2 + s_s^2 - ω^2 = s^2 - ω^2

物理的解釈:
    Q_OW < 0 : 渦度支配 / 回転支配 (vortex core / rotation dominated)
    Q_OW ≈ 0 : 中立境界 (neutral boundary)
    Q_OW > 0 : 歪み・剪断支配 / 流線レーン (strain/shear dominated, hyperbolic lanes)
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy import stats
from tqdm import tqdm


def compute_okubo_weiss(
    u: np.ndarray,
    v: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    edge_order: int = 2
) -> Dict[str, np.ndarray]:
    """
    2次元速度場 (u, v) から速度勾配、歪み率、渦度、および Okubo-Weiss パラメータ Q_OW を計算する。

    Parameters
    ----------
    u : 2D np.ndarray (shape: [H, W])
        x方向速度成分
    v : 2D np.ndarray (shape: [H, W])
        y方向速度成分
    dx : float
        x方向のグリッド間隔 (デフォルト: 1.0)
    dy : float
        y方向のグリッド間隔 (デフォルト: 1.0)
    edge_order : int
        境界での差分精度 (np.gradient の edge_order, デフォルト: 2)

    Returns
    -------
    dict:
        'Q_ow': Q_OW = s_n^2 + s_s^2 - omega^2
        's_sq': s_n^2 + s_s^2 (総歪み率の2乗)
        'omega_sq': omega^2 (渦度の2乗)
        'omega': omega (渦度)
        's_n': s_n (normal strain rate)
        's_s': s_s (shear strain rate)
    """
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # np.gradient: axis 0 is y, axis 1 is x
    u_y, u_x = np.gradient(u, dy, dx, edge_order=edge_order)
    v_y, v_x = np.gradient(v, dy, dx, edge_order=edge_order)

    s_n = u_x - v_y
    s_s = v_x + u_y
    omega = v_x - u_y

    s_sq = s_n**2 + s_s**2
    omega_sq = omega**2
    Q_ow = s_sq - omega_sq

    return {
        'Q_ow': Q_ow,
        's_sq': s_sq,
        'omega_sq': omega_sq,
        'omega': omega,
        's_n': s_n,
        's_s': s_s,
    }


def sample_field_at_coordinates(
    field_2d: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    method: str = 'bilinear'
) -> np.ndarray:
    """
    2次元場から連続座標 (xs, ys) の値をサンプリングする。
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    H, W = field_2d.shape

    # 画像外判定マスク
    valid_mask = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)

    order = 1 if method == 'bilinear' else 0
    coords = np.vstack([ys, xs])

    sampled = map_coordinates(
        field_2d,
        coords,
        order=order,
        mode='constant',
        cval=np.nan
    )

    sampled[~valid_mask] = np.nan
    return sampled


class OnlineStatsAccumulator:
    """
    大規模な全ピクセルデータをメモリに溜め込まずに、
    オンラインで平均・分散およびヒストグラムを正確に累積集計するクラス。
    """
    def __init__(self, bin_edges: np.ndarray):
        self.bin_edges = bin_edges
        self.hist_counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.vortex_count = 0  # Q < 0
        self.strain_count = 0  # Q > 0
        self.neutral_count = 0 # Q == 0
        self.min_val = np.inf
        self.max_val = -np.inf

    def update(self, values: np.ndarray):
        flat = values.ravel()
        valid = flat[np.isfinite(flat)]
        if len(valid) == 0:
            return

        n = len(valid)
        self.count += n
        s = np.sum(valid, dtype=np.float64)
        s_sq = np.sum(valid.astype(np.float64)**2, dtype=np.float64)
        self.sum += s
        self.sum_sq += s_sq

        self.vortex_count += np.count_nonzero(valid < 0)
        self.strain_count += np.count_nonzero(valid > 0)
        self.neutral_count += np.count_nonzero(valid == 0)

        c_min = np.min(valid)
        c_max = np.max(valid)
        if c_min < self.min_val:
            self.min_val = float(c_min)
        if c_max > self.max_val:
            self.max_val = float(c_max)

        counts, _ = np.histogram(valid, bins=self.bin_edges)
        self.hist_counts += counts

    @property
    def mean(self) -> float:
        if self.count == 0:
            return np.nan
        return float(self.sum / self.count)

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return np.nan
        var = (self.sum_sq - (self.sum**2 / self.count)) / (self.count - 1)
        return float(max(0.0, var))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def vortex_fraction(self) -> float:
        if self.count == 0:
            return np.nan
        return float(self.vortex_count / self.count)

    @property
    def strain_fraction(self) -> float:
        if self.count == 0:
            return np.nan
        return float(self.strain_count / self.count)

    def get_density_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """ビン中心座標と確率密度を返す"""
        bin_widths = np.diff(self.bin_edges)
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        total_in_bins = np.sum(self.hist_counts)
        if total_in_bins == 0:
            density = np.zeros_like(bin_centers)
        else:
            density = self.hist_counts / (total_in_bins * bin_widths)
        return bin_centers, density


def analyze_experiment_okubo_weiss(
    flows_h5_path: Union[str, Path],
    tracks_csv_path: Union[str, Path],
    scale: float = 0.11,
    frame_interval: float = 4.0,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    pixel_stride: int = 2,
    snapshot_frame: Optional[int] = None,
    bin_range_sigmas: float = 6.0,
    n_bins: int = 120,
    use_cache_if_available: bool = True,
    verbose: bool = True
) -> Dict:
    """
    微小管流場から Okubo-Weiss パラメータを計算し、
    背景流場 (Eulerian) と貨物軌道 (Lagrangian) のサンプリング分布を解析する。
    mt_flow_cache_s4_f5.h5 等のキャッシュが存在する場合は自動利用して超高速化。
    """
    flows_h5_path = Path(flows_h5_path)
    tracks_csv_path = Path(tracks_csv_path)
    exp_dir = flows_h5_path.parent

    # 1. トラッキングデータの読み込み
    df_tracks = pd.read_csv(tracks_csv_path)
    required_cols = {'frame', 'x', 'y', 'particle'}
    if not required_cols.issubset(df_tracks.columns):
        raise ValueError(f"tracks_csv must contain columns {required_cols}, found {df_tracks.columns}")

    # 2. キャッシュの探索
    cache_path = None
    if use_cache_if_available:
        for c_cand in [exp_dir / 'mt_flow_cache_s4_f5.h5', exp_dir / 'mt_flow_cache_s8_f5.h5']:
            if c_cand.exists():
                cache_path = c_cand
                break

    use_cache = cache_path is not None
    if verbose:
        if use_cache:
            print(f"Using flow cache: {cache_path.name}")
        else:
            print(f"Using full flow file: {flows_h5_path.name}")

    cargo_records = []
    bg_samples_for_ks = []

    if use_cache:
        # --- キャッシュを利用した超高速処理 ---
        with h5py.File(cache_path, 'r') as h5_c:
            mx_all = h5_c['mx'][:]  # shape: (n_frames, H_c, W_c)
            my_all = h5_c['my'][:]
            p_stride = int(h5_c.attrs.get('pixel_stride', 4))
            frame_ids = h5_c.attrs.get('frame_ids', None)

            n_c_frames, H_c, W_c = mx_all.shape
            if frame_ids is None:
                f_stride = int(h5_c.attrs.get('frame_stride', 5))
                frame_ids = np.arange(n_c_frames) * f_stride
            else:
                frame_ids = np.array(frame_ids)

        if max_frames is not None:
            valid_idx = np.where(frame_ids < max_frames)[0]
            mx_all = mx_all[valid_idx]
            my_all = my_all[valid_idx]
            frame_ids = frame_ids[valid_idx]
            n_c_frames = len(frame_ids)

        # 初期スケール推定
        q0 = compute_okubo_weiss(mx_all[0], my_all[0], dx=float(p_stride), dy=float(p_stride))['Q_ow']
        sigma_Q_est = float(np.std(q0))
        if sigma_Q_est <= 0 or not np.isfinite(sigma_Q_est):
            sigma_Q_est = 1.0

        fine_bins = np.linspace(-bin_range_sigmas * sigma_Q_est * 2, bin_range_sigmas * sigma_Q_est * 2, 401)
        euler_accumulator = OnlineStatsAccumulator(fine_bins)

        # トラッキングデータをキャッシュのフレーム番号と対応付け
        df_tracks_valid = df_tracks[df_tracks['frame'] <= frame_ids[-1]].copy()
        tracks_by_frame = {f: grp for f, grp in df_tracks_valid.groupby('frame')}

        # スナップショットフレームの決定
        if snapshot_frame is None:
            if len(df_tracks_valid) > 0:
                frame_counts = df_tracks_valid['frame'].value_counts()
                snapshot_frame = int(frame_counts.index[0])
            else:
                snapshot_frame = int(frame_ids[n_c_frames // 2])

        # 最も近いキャッシュフレーム
        closest_cache_idx = int(np.argmin(np.abs(frame_ids - snapshot_frame)))
        chosen_snapshot_frame = int(frame_ids[closest_cache_idx])

        # 全キャッシュフレームをループ
        iterator = range(n_c_frames)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Q_OW from cache")

        for i in iterator:
            f_orig = int(frame_ids[i])
            u_c = mx_all[i]
            v_c = my_all[i]

            # dx, dy = p_stride (元ピクセル単位に合わせる)
            q_res = compute_okubo_weiss(u_c, v_c, dx=float(p_stride), dy=float(p_stride))
            Q_map = q_res['Q_ow']
            s_sq_map = q_res['s_sq']
            omega_sq_map = q_res['omega_sq']

            euler_accumulator.update(Q_map)

            if len(bg_samples_for_ks) < 5000:
                bg_sub = Q_map[::2, ::2].ravel()
                bg_samples_for_ks.append(bg_sub[:min(200, len(bg_sub))])

            # 貨物サンプリング: 元のフレーム f_orig またはその近傍
            # 貨物データの中で f_orig に一致するフレーム (または ±f_stride//2 内)
            # キャッシュフレームに完全一致するフレームの貨物をサンプリング
            if f_orig in tracks_by_frame:
                grp = tracks_by_frame[f_orig]
                xs_c = grp['x'].values / p_stride
                ys_c = grp['y'].values / p_stride
                parts = grp['particle'].values

                q_sampled = sample_field_at_coordinates(Q_map, xs_c, ys_c, method='bilinear')
                s_sq_sampled = sample_field_at_coordinates(s_sq_map, xs_c, ys_c, method='bilinear')
                w_sq_sampled = sample_field_at_coordinates(omega_sq_map, xs_c, ys_c, method='bilinear')

                for p, ox, oy, q_val, s_val, w_val in zip(parts, grp['x'].values, grp['y'].values, q_sampled, s_sq_sampled, w_sq_sampled):
                    cargo_records.append({
                        'frame': f_orig,
                        'particle': p,
                        'x': ox,
                        'y': oy,
                        'Q_ow': q_val,
                        's_sq': s_val,
                        'omega_sq': w_val,
                    })

        # スナップショットは最高解像度で表示するため、GFP_flows.h5 から 1 フレームだけ読み込む
        snapshot_dict = None
        if flows_h5_path.exists():
            if verbose:
                print(f"Reading full-resolution flow for snapshot frame {chosen_snapshot_frame}...")
            with h5py.File(flows_h5_path, 'r') as h5_full:
                flow_snap = h5_full['flows'][chosen_snapshot_frame]
                u_snap = flow_snap[0]
                v_snap = flow_snap[1]
                q_snap_res = compute_okubo_weiss(u_snap, v_snap, dx=1.0, dy=1.0)
                snapshot_particles = tracks_by_frame.get(chosen_snapshot_frame, pd.DataFrame())
                snapshot_dict = {
                    'frame': chosen_snapshot_frame,
                    'Q_ow': q_snap_res['Q_ow'],
                    'u': u_snap,
                    'v': v_snap,
                    'particles': snapshot_particles
                }

    else:
        # --- キャッシュなし: GFP_flows.h5 からストリーミング ---
        with h5py.File(flows_h5_path, 'r', rdcc_nbytes=1024**2 * 50) as h5_in:
            flows_dset = h5_in['flows']
            total_frames, _, H, W = flows_dset.shape
            num_frames = total_frames if max_frames is None else min(total_frames, max_frames)

            df_tracks_valid = df_tracks[df_tracks['frame'] < num_frames].copy()
            tracks_by_frame = {f: grp for f, grp in df_tracks_valid.groupby('frame')}

            if snapshot_frame is None:
                if len(df_tracks_valid) > 0:
                    frame_counts = df_tracks_valid['frame'].value_counts()
                    snapshot_frame = int(frame_counts.index[0])
                else:
                    snapshot_frame = num_frames // 2
            chosen_snapshot_frame = min(max(0, snapshot_frame), num_frames - 1)

            flow_0 = flows_dset[0]
            q_init = compute_okubo_weiss(flow_0[0], flow_0[1])['Q_ow']
            sigma_Q_est = float(np.std(q_init[::4, ::4]))
            if sigma_Q_est <= 0 or not np.isfinite(sigma_Q_est):
                sigma_Q_est = 1.0

            fine_bins = np.linspace(-bin_range_sigmas * sigma_Q_est * 2, bin_range_sigmas * sigma_Q_est * 2, 401)
            euler_accumulator = OnlineStatsAccumulator(fine_bins)

            frames_to_process = list(range(0, num_frames, frame_stride))
            if chosen_snapshot_frame not in frames_to_process:
                frames_to_process.append(chosen_snapshot_frame)
                frames_to_process.sort()

            iterator = frames_to_process
            if verbose:
                iterator = tqdm(frames_to_process, desc="Computing Q_OW")

            snapshot_dict = None
            for f_idx in iterator:
                flow = flows_dset[f_idx]
                u_px = flow[0]
                v_px = flow[1]

                q_res = compute_okubo_weiss(u_px, v_px)
                Q_map = q_res['Q_ow']
                s_sq_map = q_res['s_sq']
                omega_sq_map = q_res['omega_sq']

                sub_Q = Q_map[::pixel_stride, ::pixel_stride]
                euler_accumulator.update(sub_Q)

                if len(bg_samples_for_ks) < 5000:
                    bg_sub = sub_Q[::4, ::4].ravel()
                    bg_samples_for_ks.append(bg_sub[:min(200, len(bg_sub))])

                if f_idx in tracks_by_frame:
                    grp = tracks_by_frame[f_idx]
                    xs = grp['x'].values
                    ys = grp['y'].values
                    parts = grp['particle'].values

                    q_sampled = sample_field_at_coordinates(Q_map, xs, ys, method='bilinear')
                    s_sq_sampled = sample_field_at_coordinates(s_sq_map, xs, ys, method='bilinear')
                    w_sq_sampled = sample_field_at_coordinates(omega_sq_map, xs, ys, method='bilinear')

                    for p, x, y, q_val, s_val, w_val in zip(parts, xs, ys, q_sampled, s_sq_sampled, w_sq_sampled):
                        cargo_records.append({
                            'frame': f_idx,
                            'particle': p,
                            'x': x,
                            'y': y,
                            'Q_ow': q_val,
                            's_sq': s_val,
                            'omega_sq': w_val,
                        })

                if f_idx == chosen_snapshot_frame:
                    snapshot_particles = tracks_by_frame.get(f_idx, pd.DataFrame())
                    snapshot_dict = {
                        'frame': f_idx,
                        'Q_ow': Q_map.copy(),
                        'u': u_px.copy(),
                        'v': v_px.copy(),
                        'particles': snapshot_particles.copy()
                    }

    # 3. 統計の集約と規格化
    df_cargo = pd.DataFrame(cargo_records)
    df_cargo = df_cargo.dropna(subset=['Q_ow']).copy()

    sigma_Q = euler_accumulator.std
    if sigma_Q == 0 or not np.isfinite(sigma_Q):
        sigma_Q = sigma_Q_est

    # 規格化
    df_cargo['Q_norm'] = df_cargo['Q_ow'] / sigma_Q

    if snapshot_dict is not None:
        snapshot_dict['Q_norm'] = snapshot_dict['Q_ow'] / sigma_Q

    # 共通の規格化ビン [-bin_range_sigmas, +bin_range_sigmas]
    norm_bins = np.linspace(-bin_range_sigmas, bin_range_sigmas, n_bins + 1)
    norm_bin_centers = 0.5 * (norm_bins[:-1] + norm_bins[1:])
    norm_bin_widths = np.diff(norm_bins)

    # 背景ヒストグラムを規格化ビンに射影
    raw_fine_centers, raw_fine_density = euler_accumulator.get_density_histogram()
    norm_fine_centers = raw_fine_centers / sigma_Q
    norm_fine_density = raw_fine_density * sigma_Q

    euler_density_norm = np.interp(norm_bin_centers, norm_fine_centers, norm_fine_density, left=0.0, right=0.0)
    total_area = np.sum(euler_density_norm * norm_bin_widths)
    if total_area > 0:
        euler_density_norm /= total_area

    # 貨物の規格化ヒストグラム
    q_cargo_norm = df_cargo['Q_norm'].values
    cargo_hist_counts, _ = np.histogram(q_cargo_norm, bins=norm_bins)
    total_cargo = len(q_cargo_norm)
    if total_cargo > 0:
        cargo_density_norm = cargo_hist_counts / (total_cargo * norm_bin_widths)
    else:
        cargo_density_norm = np.zeros_like(norm_bin_centers)

    cargo_vortex_frac = float(np.mean(q_cargo_norm < 0)) if total_cargo > 0 else np.nan
    cargo_strain_frac = float(np.mean(q_cargo_norm > 0)) if total_cargo > 0 else np.nan

    # KS検定
    bg_flat_samples = np.concatenate(bg_samples_for_ks) if bg_samples_for_ks else np.array([])
    bg_norm_samples = (bg_flat_samples / sigma_Q) if len(bg_flat_samples) > 0 else np.array([])
    if total_cargo > 0 and len(bg_norm_samples) > 0:
        ks_res = stats.ks_2samp(q_cargo_norm, bg_norm_samples)
        ks_stat = float(ks_res.statistic)
        ks_pvalue = float(ks_res.pvalue)
    else:
        ks_stat, ks_pvalue = np.nan, np.nan

    euler_stats = {
        'count': euler_accumulator.count,
        'mean_raw': euler_accumulator.mean,
        'std_raw': sigma_Q,
        'mean_norm': euler_accumulator.mean / sigma_Q,
        'vortex_fraction': euler_accumulator.vortex_fraction,
        'strain_fraction': euler_accumulator.strain_fraction,
    }

    cargo_stats = {
        'count': total_cargo,
        'mean_raw': float(np.mean(df_cargo['Q_ow'])) if total_cargo > 0 else np.nan,
        'std_raw': float(np.std(df_cargo['Q_ow'])) if total_cargo > 0 else np.nan,
        'mean_norm': float(np.mean(q_cargo_norm)) if total_cargo > 0 else np.nan,
        'median_norm': float(np.median(q_cargo_norm)) if total_cargo > 0 else np.nan,
        'vortex_fraction': cargo_vortex_frac,
        'strain_fraction': cargo_strain_frac,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
    }

    if verbose:
        print("\n=== Analysis Summary ===")
        print(f"Background σ_Q: {sigma_Q:.4e}")
        print(f"Eulerian (Background) : mean = {euler_stats['mean_norm']:.3f} σ_Q | vortex fraction = {euler_stats['vortex_fraction']*100:.1f}% | strain fraction = {euler_stats['strain_fraction']*100:.1f}%")
        print(f"Lagrangian (Cargo)    : mean = {cargo_stats['mean_norm']:.3f} σ_Q | vortex fraction = {cargo_stats['vortex_fraction']*100:.1f}% | strain fraction = {cargo_stats['strain_fraction']*100:.1f}%")
        print(f"KS-test: statistic = {ks_stat:.4f}, p-value = {ks_pvalue:.4e}")

    return {
        'cargo_df': df_cargo,
        'euler_stats': euler_stats,
        'cargo_stats': cargo_stats,
        'hist_euler': (norm_bin_centers, euler_density_norm),
        'hist_cargo': (norm_bin_centers, cargo_density_norm),
        'sigma_Q': sigma_Q,
        'snapshot': snapshot_dict,
        'norm_bins': norm_bins,
        'scale': scale,
        'frame_interval': frame_interval,
    }
