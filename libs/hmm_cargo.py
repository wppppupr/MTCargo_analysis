"""
libs/hmm_cargo.py

貨物微粒子の運動モード（Run / Tumble / 停滞・拡散等）を1次元ガウス放出隠れマルコフモデル（Gaussian HMM）
により自動同定・分類・定量化するためのモジュールです。

観測量:
    O_t = [ ln(v_t + \epsilon) ]
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import i0
import matplotlib.pyplot as plt
from hmmlearn import hmm


def extract_hmm_features(
    df_tracks: pd.DataFrame,
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
    min_track_len: int = 3,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    """
    粒子軌跡データから 1次元 HMM 用の対数速力観測量 O_t = [ln(v + eps)] を抽出する。

    Parameters
    ----------
    df_tracks : pd.DataFrame
        'particle', 'frame', 'x', 'y' カラムを含むトラッキング DataFrame
    tau : int, default 1
        ラグタイム（ステップ間隔）
    scale : float, default 0.11
        空間スケール (um/pixel)
    frame_interval : float, default 4.0
        フレーム時間間隔 (s)
    epsilon : float, default 1e-3
        ln(v + epsilon) の発散防止用微小定数 (um/s)
    min_track_len : int, default 3
        抽出に必要な最小連続フレーム数

    Returns
    -------
    X : np.ndarray, shape (N_total, 1)
        HMM 学習用観測量 [ln(v + eps)]
    lengths : List[int]
        各粒子の観測シーケンスの長さリスト
    df_obs : pd.DataFrame
        各観測点に対応する詳細データ
    """
    required_cols = {'particle', 'frame', 'x', 'y'}
    if not required_cols.issubset(df_tracks.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    df_sorted = df_tracks[['particle', 'frame', 'x', 'y']].sort_values(by=['particle', 'frame']).copy()

    obs_records = []
    lengths = []
    dt_sec = tau * frame_interval

    for particle_id, group in df_sorted.groupby('particle'):
        frames = group['frame'].to_numpy()
        x = group['x'].to_numpy() * scale
        y = group['y'].to_numpy() * scale

        n_pts = len(frames)
        if n_pts < tau + 1:
            continue

        # 連続フレーム判定 (t, t+tau)
        p0_idx = np.arange(0, n_pts - tau)
        p1_idx = p0_idx + tau

        f0 = frames[p0_idx]
        f1 = frames[p1_idx]

        valid = (f1 == f0 + tau)
        if not np.any(valid):
            continue

        valid_indices = np.where(valid)[0]
        current_seq_len = 0

        for i in range(len(valid_indices)):
            idx0 = valid_indices[i]
            idx1 = idx0 + tau

            dx = x[idx1] - x[idx0]
            dy = y[idx1] - y[idx0]
            dr_sq = dx**2 + dy**2

            if dr_sq < 1e-14:
                v = 0.0
            else:
                v = np.sqrt(dr_sq) / dt_sec

            log_v_eps = np.log(v + epsilon)

            # フレーム連続性チェック
            if i > 0 and valid_indices[i] != valid_indices[i - 1] + 1:
                if current_seq_len > 0:
                    lengths.append(current_seq_len)
                    current_seq_len = 0

            rec = {
                'particle': particle_id,
                'frame': frames[idx0],
                'x_um': x[idx0],
                'y_um': y[idx0],
                'dx_um': dx,
                'dy_um': dy,
                'v': v,
                'log_v_eps': log_v_eps,
                'obs_0': log_v_eps,
            }
            if 'exp_dir' in group.columns:
                rec['exp_dir'] = group['exp_dir'].iloc[0]
            obs_records.append(rec)
            current_seq_len += 1

        if current_seq_len > 0:
            lengths.append(current_seq_len)

    if not obs_records:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_obs = pd.DataFrame(obs_records)
    X = df_obs[['obs_0']].to_numpy()

    return X, lengths, df_obs


class CargoGaussianHMM:
    """
    貨物粒子の対数速力観測量 O_t = [ln(v + eps)] に対する 1次元 Gaussian HMM。

    状態パラメータを自動的に平均速度の昇順（State 0: Tumble/Slow, State 1: Run/Fast）に
    整列させ、一貫した物理的解釈を保証します。
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 150,
        tol: float = 1e-3,
        random_state: int = 42,
        epsilon: float = 1e-3,
        min_covar: float = 1e-3,
        init_means: Optional[np.ndarray] = None,
        init_covars: Optional[np.ndarray] = None,
        init_transmat: Optional[np.ndarray] = None,
        init_startprob: Optional[np.ndarray] = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.epsilon = epsilon
        self.min_covar = min_covar
        self.init_means = np.asarray(init_means, dtype=float) if init_means is not None else None
        self.init_covars = np.asarray(init_covars, dtype=float) if init_covars is not None else None
        self.init_transmat = np.asarray(init_transmat, dtype=float) if init_transmat is not None else None
        self.init_startprob = np.asarray(init_startprob, dtype=float) if init_startprob is not None else None

        init_chars = list("stmc")
        if self.init_means is not None and "m" in init_chars:
            init_chars.remove("m")
        if self.init_covars is not None and "c" in init_chars:
            init_chars.remove("c")
        if self.init_transmat is not None and "t" in init_chars:
            init_chars.remove("t")
        if self.init_startprob is not None and "s" in init_chars:
            init_chars.remove("s")

        self.init_params = "".join(init_chars)

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            min_covar=min_covar,
            init_params=self.init_params,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> "CargoGaussianHMM":
        """観測シーケンスから HMM パラメータを学習し、状態を速度昇順にソートする。"""
        if len(X) == 0:
            raise ValueError("X must not be empty.")

        if self.init_means is not None:
            self.model.means_ = self.init_means.copy()
        if self.init_covars is not None:
            self.model.covars_ = self.init_covars.copy()
        if self.init_transmat is not None:
            self.model.transmat_ = self.init_transmat.copy()
        if self.init_startprob is not None:
            self.model.startprob_ = self.init_startprob.copy()

        self.model.fit(X, lengths=lengths)
        self.is_fitted = True
        self._sort_states_by_speed()
        return self

    def _sort_states_by_speed(self):
        """状態を平均対数速度 (means_[:, 0]) の昇順に整列。"""
        order = np.argsort(self.model.means_[:, 0])
        if np.array_equal(order, np.arange(self.n_components)):
            return

        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_ = self.model.transmat_[order, :][:, order]
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]

    def predict(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> np.ndarray:
        """Viterbi アルゴリズムによる最尤隠れ状態系列の復号。"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X, lengths=lengths)

    def predict_proba(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> np.ndarray:
        """各時点における各隠れ状態の事後確率 P(S_t = k | O)。"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X, lengths=lengths)

    def score(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> float:
        """対数尤度 log P(O | lambda) を計算。"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.score(X, lengths=lengths)

    def compute_bic_aic(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> Tuple[float, float]:
        """
        1次元ガウスHMMの AIC / BIC を算出。
        パラメータ数: 初期確率 (k-1) + 遷移確率 k*(k-1) + 平均値 k + 分散 k
        """
        n_samples = len(X)
        log_likelihood = self.score(X, lengths=lengths)
        k = self.n_components
        n_params = (k - 1) + k * (k - 1) + k + k

        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + n_params * np.log(n_samples)

        return bic, aic

    def get_stationary_distribution(self) -> np.ndarray:
        """マルコフ連鎖の定常分布 π (pi * A = pi) を算出。"""
        A = self.model.transmat_
        k = self.n_components
        mat = np.vstack([A.T - np.eye(k), np.ones((1, k))])
        b = np.zeros(k + 1)
        b[-1] = 1.0
        pi, _, _, _ = np.linalg.lstsq(mat, b, rcond=None)
        return np.maximum(0.0, pi) / np.sum(np.maximum(0.0, pi))

    def get_state_summary(self, frame_interval: float = 4.0) -> pd.DataFrame:
        """各状態の統計パラメータサマリー DataFrame を作成。"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        pi_stat = self.get_stationary_distribution()
        A = self.model.transmat_

        records = []
        for i in range(self.n_components):
            mu_log_v = float(self.model.means_[i, 0])
            cov = self.model.covars_[i]
            var_log_v = float(cov[0, 0] if cov.ndim == 2 else (cov[0] if cov.ndim == 1 else cov))
            std_log_v = np.sqrt(max(var_log_v, 1e-8))

            v_geom = float(np.exp(mu_log_v) - self.epsilon)
            if v_geom < 0:
                v_geom = 0.0

            v_mean_model = float(np.exp(mu_log_v + 0.5 * var_log_v) - self.epsilon)

            p_stay = float(A[i, i])
            mean_dwell_time = frame_interval / (1.0 - p_stay + 1e-12) if p_stay < 1.0 else np.inf

            if self.n_components == 2:
                state_label = "Tumble / Pause" if i == 0 else "Run"
            elif self.n_components == 3:
                state_label = ["Tumble / Pause", "Intermediate", "Fast Run"][i]
            else:
                state_label = f"State {i}"

            records.append({
                'state': i,
                'label': state_label,
                'mean_log_v': mu_log_v,
                'std_log_v': std_log_v,
                'mean_speed_geom_um_s': v_geom,
                'mean_speed_model_um_s': v_mean_model,
                'self_trans_prob': p_stay,
                'stationary_prob': float(pi_stat[i]),
                'theoretical_dwell_time_s': mean_dwell_time,
            })

        return pd.DataFrame(records)


def filter_state_glitches(
    states: np.ndarray,
    lengths: List[int],
    min_duration_frames: int = 2,
) -> np.ndarray:
    """
    復号された状態系列から 1〜(min_duration_frames-1) フレームの孤立したスパイク・細切れ反転（グリッチ）を除去する。

    例: min_duration_frames=2 の場合、
    `[..., 0, 1, 0, ...]` -> `[..., 0, 0, 0, ...]` のように、前後に挟まれた 1 フレームのみの状態反転を
    前後の支配的な状態に統合します。

    Parameters
    ----------
    states : np.ndarray
        Viterbi 復号された隠れ状態系列 (1D array)
    lengths : List[int]
        各粒子の軌跡長さリスト
    min_duration_frames : int, default 2
        許容する最小持続フレーム数（2 の場合、1フレームの反転スパイクを除去）

    Returns
    -------
    np.ndarray
        細切れグリッチが除去された平滑化状態系列
    """
    if len(states) == 0 or min_duration_frames <= 1:
        return states.copy()

    filtered = states.copy()
    curr_idx = 0
    for length in lengths:
        seq = filtered[curr_idx:curr_idx + length].copy()
        if len(seq) >= 3:
            changed = True
            while changed:
                changed = False
                diffs = np.where(seq[1:] != seq[:-1])[0] + 1
                split_points = np.concatenate([[0], diffs, [len(seq)]])
                n_segs = len(split_points) - 1

                for seg_i in range(1, n_segs - 1):
                    start = split_points[seg_i]
                    end = split_points[seg_i + 1]
                    seg_len = end - start
                    prev_state = seq[split_points[seg_i - 1]]
                    next_state = seq[split_points[seg_i + 1]]

                    if seg_len < min_duration_frames and prev_state == next_state:
                        seq[start:end] = prev_state
                        changed = True
                        break

        filtered[curr_idx:curr_idx + length] = seq
        curr_idx += length

    return filtered


def calc_state_dwell_times(
    states: np.ndarray,
    lengths: List[int],
    frame_interval: float = 4.0,
    drop_edges: bool = True,
) -> Dict[int, List[float]]:
    """復号された状態系列から各状態の持続時間（Dwell time [s]）を抽出。"""
    n_states = int(np.max(states)) + 1 if len(states) > 0 else 0
    dwell_times = {s: [] for s in range(n_states)}

    curr_idx = 0
    for length in lengths:
        seq = states[curr_idx:curr_idx + length]
        curr_idx += length

        if len(seq) == 0:
            continue

        changes = np.where(seq[1:] != seq[:-1])[0] + 1
        split_points = np.concatenate([[0], changes, [len(seq)]])

        n_segments = len(split_points) - 1
        for seg_i in range(n_segments):
            if drop_edges and (seg_i == 0 or seg_i == n_segments - 1):
                continue
            s_val = seq[split_points[seg_i]]
            seg_len = split_points[seg_i + 1] - split_points[seg_i]
            dwell_times[s_val].append(seg_len * frame_interval)

    return dwell_times


def fit_exponential_pdf(
    durations: List[float],
    frame_interval: float = 4.0,
    min_val: Optional[float] = None,
    n_bins: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    持続時間 Dwell time の確率密度関数 PDF P(t) を算出し、
    y 軸を対数（ln P(t)）にとった上で指数分布 ln P(t) = ln(A) - t / tau
    を重み付き線形回帰でフィッティングする。

    Parameters
    ----------
    durations : List[float]
        持続時間リスト [s]
    frame_interval : float, default 4.0
        フレーム間隔 [s]
    min_val : Optional[float]
        フィッティングに使用する最小時間 [s]
    n_bins : Optional[int]
        ヒストグラムビン数（None の場合はデータ数に応じて自動調整）

    Returns
    -------
    dict:
        tau_fit_s, tau_err_s, amplitude_A, r2_log, r2_linear,
        t_centers, p_vals, p_errs, counts, t_fit, p_fit,
        mean_empirical_s, std_empirical_s, count
    """
    arr = np.asarray(durations)
    if min_val is not None:
        arr = arr[arr >= min_val]

    if len(arr) < 3:
        return {
            'tau_fit_s': np.nan, 'tau_err_s': np.nan, 'amplitude_A': np.nan,
            'r2_log': np.nan, 'r2_linear': np.nan,
            't_centers': np.array([]), 'p_vals': np.array([]), 'p_errs': np.array([]),
            'counts': np.array([]), 't_fit': np.array([]), 'p_fit': np.array([]),
            'mean_empirical_s': np.nan, 'std_empirical_s': np.nan, 'count': len(arr),
        }

    if n_bins is None:
        n_bins = min(15, max(5, int(np.sqrt(len(arr)) * 2)))

    min_t = max(frame_interval * 0.8, np.min(arr) * 0.9)
    max_t = np.max(arr) * 1.1

    bins = np.logspace(np.log10(min_t), np.log10(max_t), n_bins)
    hist, edges = np.histogram(arr, bins=bins, density=True)
    counts, _ = np.histogram(arr, bins=bins)

    valid = (counts > 0)
    if np.sum(valid) < 3:
        return {
            'tau_fit_s': float(np.mean(arr)), 'tau_err_s': float(np.mean(arr) / np.sqrt(len(arr))),
            'amplitude_A': 1.0 / float(np.mean(arr)), 'r2_log': np.nan, 'r2_linear': np.nan,
            't_centers': np.array([]), 'p_vals': np.array([]), 'p_errs': np.array([]),
            'counts': np.array([]), 't_fit': np.array([]), 'p_fit': np.array([]),
            'mean_empirical_s': float(np.mean(arr)), 'std_empirical_s': float(np.std(arr)), 'count': len(arr),
        }

    t_centers = np.sqrt(edges[:-1] * edges[1:])[valid]
    p_vals = hist[valid]
    counts_valid = counts[valid]
    p_errs = p_vals / np.sqrt(counts_valid)

    log_p = np.log(p_vals)
    w = np.sqrt(counts_valid)

    try:
        poly, cov = np.polyfit(t_centers, log_p, deg=1, w=w, cov=True)
        slope, intercept = poly[0], poly[1]

        if slope < 0:
            tau_fit = float(-1.0 / slope)
            tau_err = float((tau_fit**2) * np.sqrt(cov[0, 0])) if cov is not None else np.nan
            a_fit = float(np.exp(intercept))
        else:
            tau_fit = float(np.mean(arr))
            tau_err = float(tau_fit / np.sqrt(len(arr)))
            a_fit = float(np.exp(intercept))

        pred_log_p = intercept + slope * t_centers
        ss_res_log = np.sum(w * (log_p - pred_log_p)**2)
        ss_tot_log = np.sum(w * (log_p - np.average(log_p, weights=w))**2)
        r2_log = float(1.0 - ss_res_log / (ss_tot_log + 1e-12)) if ss_tot_log > 0 else np.nan

        pred_p = a_fit * np.exp(-t_centers / tau_fit)
        ss_res_lin = np.sum((p_vals - pred_p)**2)
        ss_tot_lin = np.sum((p_vals - np.mean(p_vals))**2)
        r2_lin = float(1.0 - ss_res_lin / (ss_tot_lin + 1e-12)) if ss_tot_lin > 0 else np.nan

        t_fit = np.linspace(np.min(t_centers) * 0.8, np.max(t_centers) * 1.1, 150)
        p_fit = a_fit * np.exp(-t_fit / tau_fit)

    except Exception:
        tau_fit = float(np.mean(arr))
        tau_err = float(tau_fit / np.sqrt(len(arr)))
        a_fit = 1.0 / tau_fit
        r2_log = np.nan
        r2_lin = np.nan
        t_fit = np.linspace(frame_interval, np.max(arr), 100)
        p_fit = a_fit * np.exp(-t_fit / tau_fit)

    return {
        'tau_fit_s': tau_fit,
        'tau_err_s': tau_err,
        'amplitude_A': a_fit,
        'r2_log': r2_log,
        'r2_linear': r2_lin,
        't_centers': t_centers,
        'p_vals': p_vals,
        'p_errs': p_errs,
        'counts': counts_valid,
        't_fit': t_fit,
        'p_fit': p_fit,
        'mean_empirical_s': float(np.mean(arr)),
        'std_empirical_s': float(np.std(arr)),
        'count': len(arr),
    }


def fit_exponential_ccdf(
    durations: List[float],
    min_val: Optional[float] = None,
    fit_mode: str = 'log',
    fix_amplitude_one: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    持続時間 Dwell time の相補累積分布関数 (CCDF: P(T >= t)) を算出し、
    純粋な指数減衰関数 C(t) = exp(-t / tau) (余計な係数のない形式: C(0)=1, ln C(t) = -t/tau)
    をフィッティングして緩和時間 tau を推定する。

    Parameters
    ----------
    durations : List[float]
        持続時間リスト [s]
    min_val : Optional[float]
        フィッティング対象とする最小時間 [s]
    fit_mode : str, default 'log'
        'log' (ln C(t) = -t/tau のゼロ切片線形回帰: 推奨)
        'linear' (非線形最小二乗法 curve_fit による C(t) = exp(-t/tau) 最適化)
    fix_amplitude_one : bool, default True
        True の場合、C(t) = exp(-t / tau) (A = 1.0 固定) としてフィッティング

    Returns
    -------
    dict:
        tau_fit_s, tau_err_s, amplitude_A, r2_log, r2_linear,
        t_unique, ccdf_unique, t_fit, ccdf_fit,
        mean_empirical_s, std_empirical_s, count
    """
    arr = np.asarray(durations, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if min_val is not None:
        arr = arr[arr >= min_val]

    n = len(arr)
    if n < 2:
        mean_val = float(np.mean(arr)) if n > 0 else np.nan
        return {
            'tau_fit_s': mean_val,
            'tau_err_s': mean_val / np.sqrt(max(1, n)) if not np.isnan(mean_val) else np.nan,
            'amplitude_A': 1.0,
            'r2_log': np.nan,
            'r2_linear': np.nan,
            't_unique': np.array([]),
            'ccdf_unique': np.array([]),
            't_fit': np.array([]),
            'ccdf_fit': np.array([]),
            'mean_empirical_s': mean_val,
            'std_empirical_s': float(np.std(arr)) if n > 0 else np.nan,
            'count': n,
        }

    sorted_d = np.sort(arr)
    unique_t, counts_at_t = np.unique(sorted_d, return_counts=True)
    cum_counts = np.cumsum(counts_at_t)
    ccdf_vals = (n - cum_counts + counts_at_t) / float(n)

    valid = (ccdf_vals > 0) & np.isfinite(ccdf_vals) & (unique_t > 0)
    x_data = unique_t[valid]
    y_data = ccdf_vals[valid]

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    if len(x_data) < 2:
        tau_fit = mean_val
        tau_err = mean_val / np.sqrt(n)
        a_fit = 1.0
        r2_log = np.nan
        r2_lin = np.nan
        t_fit = np.linspace(0, np.max(arr) * 1.15, 150)
        ccdf_fit = np.exp(-t_fit / tau_fit)
    else:
        try:
            if fit_mode == 'log':
                log_y = np.log(y_data)
                weights = np.sqrt(y_data * n)

                if fix_amplitude_one:
                    # ln C(t) = -t / tau (切片 0 固定)
                    # slope = sum(w^2 * x * log_y) / sum(w^2 * x^2)
                    denom = np.sum((weights * x_data)**2)
                    numer = np.sum((weights**2) * x_data * log_y)
                    slope = float(numer / denom) if denom > 0 else float(-1.0 / mean_val)
                    intercept = 0.0
                    a_fit = 1.0

                    if slope < 0:
                        tau_fit = float(-1.0 / slope)
                        # 残差分散から slope の標準誤差を計算
                        pred_log_y = slope * x_data
                        residuals = log_y - pred_log_y
                        df_resid = max(1, len(x_data) - 1)
                        s_sq = np.sum(weights * (residuals**2)) / (np.sum(weights) * (df_resid / len(x_data)) + 1e-12)
                        se_slope = np.sqrt(s_sq / (denom + 1e-12))
                        tau_err = float((tau_fit**2) * se_slope) if np.isfinite(se_slope) else float(tau_fit / np.sqrt(n))
                    else:
                        tau_fit = mean_val
                        tau_err = float(mean_val / np.sqrt(n))

                else:
                    poly, cov = np.polyfit(x_data, log_y, deg=1, w=weights, cov=True)
                    slope, intercept = poly[0], poly[1]
                    cov_slope = cov[0, 0] if cov is not None else 0.0

                    if slope < 0:
                        tau_fit = float(-1.0 / slope)
                        tau_err = float((tau_fit**2) * np.sqrt(cov_slope)) if cov_slope > 0 else float(tau_fit / np.sqrt(n))
                        a_fit = float(np.exp(intercept))
                    else:
                        tau_fit = mean_val
                        tau_err = float(mean_val / np.sqrt(n))
                        a_fit = 1.0

                # 決定係数の計算
                pred_log_y = intercept + slope * x_data
                ss_res_log = np.sum(weights * (log_y - pred_log_y)**2)
                ss_tot_log = np.sum(weights * (log_y - np.average(log_y, weights=weights))**2)
                r2_log = float(1.0 - ss_res_log / (ss_tot_log + 1e-12)) if ss_tot_log > 0 else np.nan

                pred_y = np.exp(-x_data / tau_fit) if fix_amplitude_one else a_fit * np.exp(-x_data / tau_fit)
                ss_res_lin = np.sum((y_data - pred_y)**2)
                ss_tot_lin = np.sum((y_data - np.mean(y_data))**2)
                r2_lin = float(1.0 - ss_res_lin / (ss_tot_lin + 1e-12)) if ss_tot_lin > 0 else np.nan

            else:
                from scipy.optimize import curve_fit
                if fix_amplitude_one:
                    def exp_ccdf_pure(t, tau):
                        return np.exp(-t / tau)
                    popt, pcov = curve_fit(exp_ccdf_pure, x_data, y_data, p0=[mean_val], bounds=([1e-3], [1e5]), maxfev=5000)
                    tau_fit = float(popt[0])
                    a_fit = 1.0
                    tau_err = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float(mean_val / np.sqrt(n))
                else:
                    def exp_ccdf_func(t, tau, a):
                        return a * np.exp(-t / tau)
                    popt, pcov = curve_fit(exp_ccdf_func, x_data, y_data, p0=[mean_val, 1.0], bounds=([1e-3, 0.1], [1e5, 5.0]), maxfev=5000)
                    tau_fit, a_fit = float(popt[0]), float(popt[1])
                    tau_err = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float(mean_val / np.sqrt(n))

                pred_y = np.exp(-x_data / tau_fit) if fix_amplitude_one else a_fit * np.exp(-x_data / tau_fit)
                ss_res_lin = np.sum((y_data - pred_y)**2)
                ss_tot_lin = np.sum((y_data - np.mean(y_data))**2)
                r2_lin = float(1.0 - ss_res_lin / (ss_tot_lin + 1e-12)) if ss_tot_lin > 0 else np.nan

                pred_log_y = -x_data / tau_fit if fix_amplitude_one else np.log(np.maximum(pred_y, 1e-12))
                log_y = np.log(y_data)
                ss_res_log = np.sum((log_y - pred_log_y)**2)
                ss_tot_log = np.sum((log_y - np.mean(log_y))**2)
                r2_log = float(1.0 - ss_res_log / (ss_tot_log + 1e-12)) if ss_tot_log > 0 else np.nan

            max_t = np.max(x_data)
            t_fit = np.linspace(0, max_t * 1.15, 200)
            ccdf_fit = np.exp(-t_fit / tau_fit) if fix_amplitude_one else a_fit * np.exp(-t_fit / tau_fit)

        except Exception:
            tau_fit = mean_val
            tau_err = float(mean_val / np.sqrt(n))
            a_fit = 1.0
            r2_log = np.nan
            r2_lin = np.nan
            t_fit = np.linspace(0, np.max(arr) * 1.15, 150)
            ccdf_fit = np.exp(-t_fit / tau_fit)

    return {
        'tau_fit_s': tau_fit,
        'tau_err_s': tau_err,
        'amplitude_A': 1.0 if fix_amplitude_one else a_fit,
        'r2_log': r2_log,
        'r2_linear': r2_lin,
        't_unique': x_data,
        'ccdf_unique': y_data,
        't_fit': t_fit,
        'ccdf_fit': ccdf_fit,
        'mean_empirical_s': mean_val,
        'std_empirical_s': std_val,
        'count': n,
    }


def fit_exponential_distribution(
    durations: List[float],
    min_val: Optional[float] = None,
    frame_interval: float = 4.0,
    fit_target: str = 'ccdf',
) -> Tuple[float, float, float]:
    """
    持続時間データに対して指数分布フィッティングを行い (tau, tau_err, r2) を返す。
    fit_target='ccdf' (デフォルト: CCDF 指数回帰) または 'pdf' (PDF 指数回帰)。
    """
    if fit_target.lower() == 'ccdf':
        res = fit_exponential_ccdf(durations, min_val=min_val)
    else:
        res = fit_exponential_pdf(durations, frame_interval=frame_interval, min_val=min_val)
    return res['tau_fit_s'], res['tau_err_s'], res['r2_log']


def calc_posterior_statistics(proba: np.ndarray, n_components: int = 2) -> dict:
    """各状態の事後確率行列 P(S_t = k | O) から信頼度・不確実性統計を算出。"""
    if len(proba) == 0:
        return {}

    max_proba = np.max(proba, axis=1)
    mean_conf = float(np.mean(max_proba))
    median_conf = float(np.median(max_proba))
    high_conf_ratio = float(np.mean(max_proba >= 0.80))
    very_high_conf_ratio = float(np.mean(max_proba >= 0.95))

    eps = 1e-12
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    mean_entropy = float(np.mean(entropy))
    max_entropy = np.log(n_components) if n_components > 1 else 1.0
    norm_entropy = float(mean_entropy / max_entropy)

    res = {
        'mean_confidence': mean_conf,
        'median_confidence': median_conf,
        'high_conf_ratio_80': high_conf_ratio,
        'high_conf_ratio_95': very_high_conf_ratio,
        'mean_entropy': mean_entropy,
        'norm_entropy': norm_entropy,
    }
    for k in range(n_components):
        res[f'mean_proba_s{k}'] = float(np.mean(proba[:, k]))
        res[f'std_proba_s{k}'] = float(np.std(proba[:, k]))

    return res


def check_markov_property(
    states: np.ndarray,
    lengths: List[int],
    transmat: np.ndarray,
    frame_interval: float = 4.0,
    max_lag_steps: int = 15,
    dwell_times: Optional[Dict[int, List[float]]] = None,
) -> dict:
    """
    復号された状態系列 S(t) に対してマルコフ性（1次マルコフ連鎖の妥当性）を包括的に検証する。

    1. Chapman-Kolmogorov (CK) テスト & 固有緩和時間（Implied Timescale）の不変性
       - 実測遷移確率行列 A_emp(n*tau) vs マルコフ予測 [A(1)]^n のフロベニウスノルム誤差
       - 固有値 lambda_2(n*tau) から算出される Implied Timescale tau_implied(n) = - n*dt / ln(lambda_2)
    2. 1次 vs 2次マルコフ性・記憶効果の独立性検定 (Conditional Independence: P(S_{t+1} | S_t, S_{t-1}) = P(S_{t+1} | S_t))
       - 各 S_t の下で過去 S_{t-1} と未来 S_{t+1} の独立性カイ二乗検定
    3. 状態系列の自己相関関数 C_S(k*tau) とマルコフ理論減衰 C_S^theo(k) = (lambda_2)^k の比較
    4. Dwell time の無記憶性検定 (KS 検定: Kolmogorov-Smirnov test against exponential)

    Returns
    -------
    dict:
        ck_df : pd.DataFrame
        autocorr_df : pd.DataFrame
        chi2_results : dict
        ks_results : dict
        tau_relax_s : float
        mean_ck_error : float
    """
    n_components = transmat.shape[0]
    if len(states) < 10 or n_components < 2:
        return {}

    # 理論第2固有値と理論緩和時間
    eig_1 = np.sort(np.linalg.eigvals(transmat))[::-1]
    lambda2_1 = float(eig_1[1]) if len(eig_1) > 1 else np.nan
    tau_relax_s = float(-frame_interval / np.log(max(1e-6, min(0.9999, lambda2_1)))) if lambda2_1 > 0 else np.nan

    # 1. Chapman-Kolmogorov テスト
    ck_records = []
    for n_lag in range(1, max_lag_steps + 1):
        n_trans = np.zeros((n_components, n_components))
        curr = 0
        for l in lengths:
            seq = states[curr:curr + l]
            curr += l
            if len(seq) > n_lag:
                s0 = seq[:-n_lag]
                sn = seq[n_lag:]
                for i in range(n_components):
                    for j in range(n_components):
                        n_trans[i, j] += np.sum((s0 == i) & (sn == j))

        row_sums = n_trans.sum(axis=1, keepdims=True)
        A_emp_n = n_trans / np.maximum(row_sums, 1)
        A_pred_n = np.linalg.matrix_power(transmat, n_lag)

        frob_err = float(np.linalg.norm(A_emp_n - A_pred_n, ord='fro'))
        max_abs_err = float(np.max(np.abs(A_emp_n - A_pred_n)))

        eig_emp = np.sort(np.linalg.eigvals(A_emp_n))[::-1]
        lambda2_emp = float(eig_emp[1]) if len(eig_emp) > 1 else np.nan

        if lambda2_emp > 0 and lambda2_emp < 0.9999:
            tau_implied = float(- (n_lag * frame_interval) / np.log(lambda2_emp))
        else:
            tau_implied = np.nan

        ck_records.append({
            'lag_step': n_lag,
            'lag_time_s': n_lag * frame_interval,
            'frobenius_error': frob_err,
            'max_abs_error': max_abs_err,
            'lambda2_emp': lambda2_emp,
            'lambda2_pred': float(lambda2_1 ** n_lag) if not np.isnan(lambda2_1) else np.nan,
            'tau_implied_s': tau_implied,
            'tau_pred_s': tau_relax_s,
        })

    df_ck = pd.DataFrame(ck_records)

    # 2. 状態系列の自己相関関数 C_S(k)
    mu_s = float(np.mean(states))
    var_s = float(np.var(states))
    autocorr_records = []

    for n_lag in range(max_lag_steps + 1):
        lag_time_s = n_lag * frame_interval
        if n_lag == 0:
            c_emp = 1.0
            c_theo = 1.0
        else:
            num = 0.0
            den = 0
            curr = 0
            for l in lengths:
                seq = states[curr:curr + l]
                curr += l
                if len(seq) > n_lag:
                    num += np.sum((seq[:-n_lag] - mu_s) * (seq[n_lag:] - mu_s))
                    den += (len(seq) - n_lag)
            c_emp = float(num / (den * var_s)) if (den > 0 and var_s > 0) else np.nan
            c_theo = float(lambda2_1 ** n_lag) if not np.isnan(lambda2_1) else np.nan

        autocorr_records.append({
            'lag_step': n_lag,
            'lag_time_s': lag_time_s,
            'autocorr_emp': c_emp,
            'autocorr_theo': c_theo,
        })

    df_autocorr = pd.DataFrame(autocorr_records)

    # 3. 1次 vs 2次マルコフ性 (高次記憶効果の独立性検定)
    # P(S_{t+1} | S_t, S_{t-1}) = P(S_{t+1} | S_t)
    triplets = np.zeros((n_components, n_components, n_components))
    curr = 0
    for l in lengths:
        seq = states[curr:curr + l]
        curr += l
        if len(seq) >= 3:
            s_p = seq[:-2]
            s_c = seq[1:-1]
            s_n = seq[2:]
            for ip in range(n_components):
                for ic in range(n_components):
                    for inn in range(n_components):
                        triplets[ip, ic, inn] += np.sum((s_p == ip) & (s_c == ic) & (s_n == inn))

    chi2_results = {}
    for ic in range(n_components):
        cont_table = triplets[:, ic, :]
        if cont_table.sum() >= 10:
            try:
                res_chi2 = stats.chi2_contingency(cont_table)
                chi2_results[ic] = {
                    'chi2_stat': float(res_chi2.statistic),
                    'p_value': float(res_chi2.pvalue),
                    'dof': int(res_chi2.dof),
                    'is_markovian_p05': bool(res_chi2.pvalue >= 0.05),
                }
            except Exception:
                chi2_results[ic] = {'chi2_stat': np.nan, 'p_value': np.nan, 'dof': 1, 'is_markovian_p05': np.nan}
        else:
            chi2_results[ic] = {'chi2_stat': np.nan, 'p_value': np.nan, 'dof': 1, 'is_markovian_p05': np.nan}

    # 4. Dwell Time の指数性検定 (KS 検定)
    ks_results = {}
    if dwell_times is not None:
        for s in range(n_components):
            arr_s = np.asarray(dwell_times.get(s, []))
            if len(arr_s) >= 8:
                tau_mle = float(np.mean(arr_s))
                # KS test against exponential CDF F(t) = 1 - exp(-t/tau)
                ks_stat, ks_pval = stats.kstest(arr_s, 'expon', args=(0, tau_mle))
                ks_results[s] = {
                    'ks_stat': float(ks_stat),
                    'p_value': float(ks_pval),
                    'is_exponential_p05': bool(ks_pval >= 0.05),
                }
            else:
                ks_results[s] = {'ks_stat': np.nan, 'p_value': np.nan, 'is_exponential_p05': np.nan}

    mean_ck_err = float(df_ck['frobenius_error'].mean()) if not df_ck.empty else np.nan

    return {
        'ck_df': df_ck,
        'autocorr_df': df_autocorr,
        'chi2_results': chi2_results,
        'ks_results': ks_results,
        'tau_relax_s': tau_relax_s,
        'lambda2_1': lambda2_1,
        'mean_ck_error': mean_ck_err,
    }


def calc_state_dependent_msd(
    df_obs: pd.DataFrame,
    max_tau: int = 25,
    frame_interval: float = 4.0,
    min_segment_len: int = 3,
    n_components: int = 2,
    fit_min_tau: int = 1,
    fit_max_tau: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_obs.empty or 'pred_state' not in df_obs.columns:
        return pd.DataFrame(), pd.DataFrame()

    # 状態ごとの変位二乗リスト: state -> tau -> list of dr^2
    state_dr2 = {s: {tau: [] for tau in range(1, max_tau + 1)} for s in range(n_components)}
    all_dr2 = {tau: [] for tau in range(1, max_tau + 1)}

    group_cols = ['exp_dir', 'particle'] if 'exp_dir' in df_obs.columns else ['particle']

    for _, group in df_obs.groupby(group_cols):
        df_p = group.sort_values(by='frame')
        frames = df_p['frame'].to_numpy()
        x = df_p['x_um'].to_numpy()
        y = df_p['y_um'].to_numpy()
        states = df_p['pred_state'].to_numpy()

        n = len(frames)
        if n < min_segment_len:
            continue

        # 1. 全体 (All States) の連続フレームセグメントを抽出
        frame_diffs = np.diff(frames)
        all_split_idx = np.where(frame_diffs != 1)[0] + 1
        all_starts = np.concatenate([[0], all_split_idx])
        all_ends = np.concatenate([all_split_idx, [n]])

        for a_start, a_end in zip(all_starts, all_ends):
            seg_len = a_end - a_start
            if seg_len < min_segment_len:
                continue
            x_seg = x[a_start:a_end]
            y_seg = y[a_start:a_end]
            for tau in range(1, min(max_tau + 1, seg_len)):
                dx = x_seg[tau:] - x_seg[:-tau]
                dy = y_seg[tau:] - y_seg[:-tau]
                all_dr2[tau].extend((dx**2 + dy**2).tolist())

        # 2. 状態ごとの連続フレームセグメントを抽出
        state_diffs = np.diff(states)
        split_idx = np.where((frame_diffs != 1) | (state_diffs != 0))[0] + 1
        seg_starts = np.concatenate([[0], split_idx])
        seg_ends = np.concatenate([split_idx, [n]])

        for s_start, s_end in zip(seg_starts, seg_ends):
            seg_len = s_end - s_start
            if seg_len < min_segment_len:
                continue

            st = states[s_start]
            x_seg = x[s_start:s_end]
            y_seg = y[s_start:s_end]

            for tau in range(1, min(max_tau + 1, seg_len)):
                dx = x_seg[tau:] - x_seg[:-tau]
                dy = y_seg[tau:] - y_seg[:-tau]
                state_dr2[st][tau].extend((dx**2 + dy**2).tolist())

    # 集計 DataFrame の作成
    msd_records = []
    # 各状態
    for s in range(n_components):
        s_lbl = "Tumble / Pause" if s == 0 else ("Run" if s == 1 else f"State {s}")
        for tau in range(1, max_tau + 1):
            vals = np.array(state_dr2[s][tau])
            if len(vals) > 0:
                mean_msd = float(np.mean(vals))
                std_msd = float(np.std(vals))
                sem_msd = float(std_msd / np.sqrt(len(vals)))
                n_count = len(vals)
            else:
                mean_msd, std_msd, sem_msd, n_count = np.nan, np.nan, np.nan, 0

            msd_records.append({
                'state': s,
                'state_label': s_lbl,
                'tau_step': tau,
                'lag_time_s': tau * frame_interval,
                'msd_um2': mean_msd,
                'msd_std_um2': std_msd,
                'msd_sem_um2': sem_msd,
                'sem_um2': sem_msd,
                'count': n_count,
            })

    # 全体 (All)
    for tau in range(1, max_tau + 1):
        vals = np.array(all_dr2[tau])
        if len(vals) > 0:
            mean_msd = float(np.mean(vals))
            std_msd = float(np.std(vals))
            sem_msd = float(std_msd / np.sqrt(len(vals)))
            n_count = len(vals)
        else:
            mean_msd, std_msd, sem_msd, n_count = np.nan, np.nan, np.nan, 0

        msd_records.append({
            'state': -1,
            'state_label': "All",
            'tau_step': tau,
            'lag_time_s': tau * frame_interval,
            'msd_um2': mean_msd,
            'msd_std_um2': std_msd,
            'msd_sem_um2': sem_msd,
            'sem_um2': sem_msd,
            'count': n_count,
        })

    df_msd = pd.DataFrame(msd_records)

    fit_records = []
    for s in list(range(n_components)) + [-1]:
        sub_df = df_msd[(df_msd['state'] == s) & 
                        (df_msd['tau_step'] >= fit_min_tau) & 
                        (df_msd['tau_step'] <= fit_max_tau) & 
                        (df_msd['msd_um2'] > 0) & 
                        (~df_msd['msd_um2'].isna())]

        s_label = "All" if s == -1 else ("Tumble / Pause" if s == 0 else ("Run" if s == 1 else f"State {s}"))

        if len(sub_df) < 3:
            fit_records.append({
                'state': s,
                'state_label': s_label,
                'alpha': np.nan,
                'alpha_err': np.nan,
                'D_apparent_um2_s': np.nan,
                'r_squared': np.nan,
                'fit_points': len(sub_df),
            })
            continue

        log_t = np.log10(sub_df['lag_time_s'].to_numpy())
        log_msd = np.log10(sub_df['msd_um2'].to_numpy())

        try:
            poly, cov = np.polyfit(log_t, log_msd, deg=1, cov=True)
            alpha = float(poly[0])
            alpha_err = float(np.sqrt(cov[0, 0]))
            intercept = poly[1]
            D_app = float((10.0**intercept) / 4.0)

            pred_log_msd = np.polyval(poly, log_t)
            ss_res = np.sum((log_msd - pred_log_msd)**2)
            ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        except Exception:
            alpha, alpha_err, D_app, r2 = np.nan, np.nan, np.nan, np.nan

        fit_records.append({
            'state': s,
            'state_label': s_label,
            'alpha': alpha,
            'alpha_err': alpha_err,
            'D_apparent_um2_s': D_app,
            'r_squared': r2,
            'fit_points': len(sub_df),
        })

    df_fits = pd.DataFrame(fit_records)
    return df_msd, df_fits


def plot_emission_1d_distribution_6panel(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    n_components: int = 2,
    epsilon: float = 1e-3,
    state_names: Optional[Dict[int, str]] = None,
    state_colors: Optional[Dict[int, str]] = None,
):
    """1次元観測空間 ln(v + eps) における放出確率密度分布の6パネル比較プロット。"""
    if state_names is None:
        state_names = {0: "Tumble / Pause", 1: "Run"}
    if state_colors is None:
        state_colors = {0: "#4477AA", 1: "#EE6677"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes_flat[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        X = res['X']
        model = res['model']

        if len(X) == 0:
            ax.set_visible(False)
            continue

        log_v_vals = X[:, 0]
        pi_stat = model.get_stationary_distribution()

        # ヒストグラム
        counts, bin_edges, _ = ax.hist(
            log_v_vals,
            bins=40,
            density=True,
            color='#999999',
            alpha=0.35,
            edgecolor='#777777',
            label='Observed Data',
            zorder=1,
        )

        # 混合ガウス曲線
        x_grid = np.linspace(np.min(log_v_vals) - 0.5, np.max(log_v_vals) + 0.5, 300)
        total_pdf = np.zeros_like(x_grid)

        for s in range(n_components):
            mu = float(model.model.means_[s, 0])
            cov = model.model.covars_[s]
            var = float(cov[0, 0] if cov.ndim == 2 else (cov[0] if cov.ndim == 1 else cov))
            sigma = np.sqrt(max(var, 1e-8))
            weight = float(pi_stat[s])

            pdf_s = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
            weighted_pdf_s = weight * pdf_s
            total_pdf += weighted_pdf_s

            v_geom = float(np.exp(mu) - epsilon)
            if v_geom < 0:
                v_geom = 0.0

            s_lbl = state_names.get(s, f"State {s}")
            col = state_colors.get(s, f"C{s}")

            ax.plot(
                x_grid,
                weighted_pdf_s,
                color=col,
                lw=2.2,
                label=f"{s_lbl}: $v_{{\\mathrm{{geom}}}}={v_geom:.3f}\\,\\mu\\mathrm{{m/s}}$ ({weight*100:.1f}%)",
                zorder=3,
            )
            ax.fill_between(x_grid, 0, weighted_pdf_s, color=col, alpha=0.18, zorder=2)
            ax.axvline(mu, color=col, linestyle=':', lw=1.5, alpha=0.8, zorder=3)

        ax.plot(x_grid, total_pdf, color='#111111', lw=1.8, linestyle='--', label=r'Mixture Fit $\sum \pi_k \mathcal{N}_k$', zorder=4)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (N={len(X):,})", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8.0, frameon=True, framealpha=0.92)

        if idx >= 3:
            ax.set_xlabel(f"$\\ln(v + \\epsilon)$  [$\\epsilon={epsilon}$]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel("Probability Density", fontsize=11)

    fig.suptitle(r"1D Speed Gaussian HMM Emission Distributions ($\ln(v+\epsilon)$)", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def calc_state_dependent_turning_angles(
    df_obs: pd.DataFrame,
    min_displacement: float = 1e-6,
    n_components: int = 2,
) -> Dict[int, np.ndarray]:
    """
    HMM の推定状態系列に基づいて、各状態における方向転換角 Δθ (rad, [-π, π]) を抽出する。

    Parameters
    ----------
    df_obs : pd.DataFrame
        ['exp_dir', 'particle', 'frame', 'dx_um', 'dy_um', 'pred_state'] を含む DataFrame
    min_displacement : float, default 1e-6
        微小変位（ゼロ除算）カットオフ
    n_components : int, default 2

    Returns
    -------
    Dict[int, np.ndarray]:
        {
            0: np.ndarray of Δθ_tumble (rad),
            1: np.ndarray of Δθ_run (rad),
            -1: np.ndarray of Δθ_all (rad),
        }
    """
    if df_obs.empty or 'pred_state' not in df_obs.columns:
        return {s: np.array([]) for s in list(range(n_components)) + [-1]}

    angles_by_state = {s: [] for s in list(range(n_components)) + [-1]}
    group_cols = ['exp_dir', 'particle'] if 'exp_dir' in df_obs.columns else ['particle']

    for _, group in df_obs.groupby(group_cols):
        df_p = group.sort_values(by='frame')
        frames = df_p['frame'].to_numpy()
        dx = df_p['dx_um'].to_numpy()
        dy = df_p['dy_um'].to_numpy()
        states = df_p['pred_state'].to_numpy()

        n = len(frames)
        if n < 2:
            continue

        frame_diffs = np.diff(frames)
        valid_pairs = np.where(frame_diffs == 1)[0]

        for i in valid_pairs:
            dx1, dy1 = dx[i], dy[i]
            dx2, dy2 = dx[i + 1], dy[i + 1]

            dr1_sq = dx1**2 + dy1**2
            dr2_sq = dx2**2 + dy2**2

            if dr1_sq < min_displacement**2 or dr2_sq < min_displacement**2:
                continue

            th1 = np.arctan2(dy1, dx1)
            th2 = np.arctan2(dy2, dx2)
            d_th = np.arctan2(np.sin(th2 - th1), np.cos(th2 - th1))

            st1 = states[i]
            st2 = states[i + 1]

            angles_by_state[-1].append(d_th)

            # 純粋な Run (Run -> Run)
            if st1 == 1 and st2 == 1:
                angles_by_state[1].append(d_th)
            # Tumble 状態を含むステップ (Tumble -> Tumble, Run -> Tumble, Tumble -> Run)
            if st1 == 0 or st2 == 0:
                angles_by_state[0].append(d_th)

    return {s: np.asarray(angles_by_state[s], dtype=float) for s in angles_by_state}


def calc_circular_stats_dict(angles_rad: np.ndarray) -> dict:
    """円統計（Circular Statistics）指標を算出する。"""
    arr = np.asarray(angles_rad, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            'count': 0,
            'mean_cos': np.nan,
            'mean_sin': np.nan,
            'mean_resultant_length_R': np.nan,
            'circular_mean_rad': np.nan,
            'circular_variance': np.nan,
            'circular_std_rad': np.nan,
            'mean_abs_angle_rad': np.nan,
            'mean_abs_angle_deg': np.nan,
        }

    c = float(np.mean(np.cos(arr)))
    s = float(np.mean(np.sin(arr)))
    R = float(np.hypot(c, s))
    mu = float(np.arctan2(s, c))
    circ_var = float(1.0 - R)
    circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(1e-12, R)))))
    mean_abs_rad = float(np.mean(np.abs(arr)))

    return {
        'count': len(arr),
        'mean_cos': c,
        'mean_sin': s,
        'mean_resultant_length_R': R,
        'circular_mean_rad': mu,
        'circular_variance': circ_var,
        'circular_std_rad': circ_std,
        'mean_abs_angle_rad': mean_abs_rad,
        'mean_abs_angle_deg': float(np.rad2deg(mean_abs_rad)),
    }


def fit_von_mises_distribution(angles_rad: np.ndarray, bins: int = 36) -> dict:
    """
    角度変化データに対して von Mises 分布 P(θ) = exp(kappa * cos(θ - mu)) / (2π I0(kappa)) をフィッティングする。
    """
    arr = np.asarray(angles_rad, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return {
            'kappa': np.nan,
            'kappa_err': np.nan,
            'mu': np.nan,
            'r_squared': np.nan,
            'bin_centers': np.array([]),
            'pdf_data': np.array([]),
            'fit_x': np.array([]),
            'fit_y': np.array([]),
        }

    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    counts, _ = np.histogram(arr, bins=bin_edges, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    def von_mises_func(x, kappa, mu):
        k = np.clip(kappa, 1e-4, 50.0)
        return np.exp(k * np.cos(x - mu)) / (2.0 * np.pi * i0(k))

    c_stats = calc_circular_stats_dict(arr)
    R = c_stats['mean_resultant_length_R']
    # kappa の初期推定値 (アプロキシメーション)
    if R < 0.53:
        k_init = 2 * R + R**3 + 5/6 * R**5
    elif R < 0.85:
        k_init = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        k_init = 1 / (R**3 - 4 * R**2 + 3 * R)
    k_init = float(np.clip(k_init, 0.01, 10.0))

    try:
        popt, pcov = curve_fit(
            von_mises_func,
            centers,
            counts,
            p0=[k_init, 0.0],
            bounds=([0.0, -np.pi], [50.0, np.pi]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0]
        kappa_fit, mu_fit = float(popt[0]), float(popt[1])
        kappa_err = float(perr[0])
    except Exception:
        kappa_fit, kappa_err, mu_fit = k_init, np.nan, 0.0

    fit_x = np.linspace(-np.pi, np.pi, 200)
    fit_y = von_mises_func(fit_x, kappa_fit, mu_fit)

    y_pred = von_mises_func(centers, kappa_fit, mu_fit)
    ss_res = np.sum((counts - y_pred)**2)
    ss_tot = np.sum((counts - np.mean(counts))**2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0

    return {
        'kappa': kappa_fit,
        'kappa_err': kappa_err,
        'mu': mu_fit,
        'r_squared': r2,
        'bin_centers': centers,
        'pdf_data': counts,
        'fit_x': fit_x,
        'fit_y': fit_y,
    }


def calc_state_dependent_autocorrelations(
    df_obs: pd.DataFrame,
    n_components: int = 2,
    frame_interval: float = 4.0,
    max_lag_frames: int = 25,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    HMM で同定された各運動状態 (State 0: Tumble/Pause, State 1: Run) の連続セグメントから
    1. 速度ベクトル自己相関 (VACF: Velocity Autocorrelation Function)
    2. 配向方向自己相関 (OACF: Orientation Autocorrelation Function)
    3. 速さスカラー自己相関 (SACF: Speed Autocorrelation Function, ゆらぎ & 正規化)
    を算出し、各ラグ時間 tau に対するアンサンブル平均・標準誤差 (SEM) を返す。

    Parameters
    ----------
    df_obs : pd.DataFrame
        'particle', 'frame', 'x', 'y' (または 'vx', 'vy'), 'pred_state' を含む観測データ
    n_components : int, default 2
        状態数
    frame_interval : float, default 4.0
        フレーム間隔 [s]
    max_lag_frames : int, default 25
        計算する最大ラグフレーム数

    Returns
    -------
    dict: {state_id: {'vacf': df_vacf, 'oacf': df_oacf, 'sacf': df_sacf}}
        各 DataFrame には 'lag_frames', 'lag_time_s', 'corr', 'sem', 'count' が含まれる。
    """
    df = df_obs.copy()
    if 'pred_state' not in df.columns:
        return {}

    # 速度ベクトル (vx, vy) の確認/計算
    if 'vx' not in df.columns or 'vy' not in df.columns:
        if 'dx_um' in df.columns and 'dy_um' in df.columns:
            df['vx'] = df['dx_um'] / frame_interval
            df['vy'] = df['dy_um'] / frame_interval
        elif 'x_um' in df.columns and 'y_um' in df.columns:
            df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
            diffs = df.groupby('particle')[['x_um', 'y_um']].diff()
            df['vx'] = diffs['x_um'] / frame_interval
            df['vy'] = diffs['y_um'] / frame_interval
        elif 'x' in df.columns and 'y' in df.columns:
            df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
            diffs = df.groupby('particle')[['x', 'y']].diff()
            df['vx'] = diffs['x'] / frame_interval
            df['vy'] = diffs['y'] / frame_interval

    df = df.dropna(subset=['vx', 'vy', 'pred_state']).copy()
    if 'v' in df.columns and not df['v'].isna().all():
        df['speed'] = df['v']
    else:
        df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    # 単位方向ベクトル (ex, ey)
    eps = 1e-12
    df['ex'] = df['vx'] / (df['speed'] + eps)
    df['ey'] = df['vy'] / (df['speed'] + eps)

    results = {}

    for s in range(n_components):
        # 状態 s の連続セグメントを抽出
        df_s_all = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
        
        # 状態 s における平均速さ (速さゆらぎ計算用)
        s_mask = (df_s_all['pred_state'] == s)
        mean_speed_s = float(np.mean(df_s_all.loc[s_mask, 'speed'])) if np.sum(s_mask) > 0 else 0.0

        # 各ラグ m における相関ペアの集計リスト
        vacf_pairs_by_lag = {m: [] for m in range(max_lag_frames + 1)}
        oacf_pairs_by_lag = {m: [] for m in range(max_lag_frames + 1)}
        sacf_fluc_pairs_by_lag = {m: [] for m in range(max_lag_frames + 1)}
        v_sq_list = []
        speed_fluc_sq_list = []

        for p_id, p_group in df_s_all.groupby('particle'):
            states = p_group['pred_state'].values
            vx = p_group['vx'].values
            vy = p_group['vy'].values
            ex = p_group['ex'].values
            ey = p_group['ey'].values
            spd = p_group['speed'].values

            n_p = len(states)
            if n_p < 2:
                continue

            # 状態 s の連続ブロックを特定
            is_s = (states == s)
            diff_s = np.diff(np.pad(is_s.astype(int), (1, 1), 'constant', constant_values=0))
            start_indices = np.where(diff_s == 1)[0]
            end_indices = np.where(diff_s == -1)[0]

            for st, ed in zip(start_indices, end_indices):
                seg_len = ed - st
                if seg_len < 2:
                    continue

                seg_vx = vx[st:ed]
                seg_vy = vy[st:ed]
                seg_ex = ex[st:ed]
                seg_ey = ey[st:ed]
                seg_spd = spd[st:ed]

                # 分母用
                v_sq_list.extend(seg_vx**2 + seg_vy**2)
                speed_fluc_sq_list.extend((seg_spd - mean_speed_s)**2)

                max_m = min(seg_len - 1, max_lag_frames)
                for m in range(max_m + 1):
                    if m == 0:
                        v_dot = seg_vx**2 + seg_vy**2
                        o_dot = np.ones(seg_len)
                        s_dot = (seg_spd - mean_speed_s)**2
                    else:
                        v_dot = seg_vx[:-m] * seg_vx[m:] + seg_vy[:-m] * seg_vy[m:]
                        o_dot = seg_ex[:-m] * seg_ex[m:] + seg_ey[:-m] * seg_ey[m:]
                        s_dot = (seg_spd[:-m] - mean_speed_s) * (seg_spd[m:] - mean_speed_s)

                    vacf_pairs_by_lag[m].extend(v_dot)
                    oacf_pairs_by_lag[m].extend(o_dot)
                    sacf_fluc_pairs_by_lag[m].extend(s_dot)

        mean_v_sq = float(np.mean(v_sq_list)) if len(v_sq_list) > 0 else 1.0
        mean_s_fluc_sq = float(np.mean(speed_fluc_sq_list)) if len(speed_fluc_sq_list) > 0 else 1.0

        # VACF DataFrame
        vacf_rows = []
        oacf_rows = []
        sacf_rows = []

        for m in range(max_lag_frames + 1):
            lag_t = m * frame_interval
            # VACF
            v_arr = np.asarray(vacf_pairs_by_lag[m])
            if len(v_arr) > 0 and mean_v_sq > 0:
                norm_v = v_arr / mean_v_sq
                vacf_rows.append({
                    'lag_frames': m,
                    'lag_time_s': lag_t,
                    'corr': float(np.mean(norm_v)),
                    'sem': float(np.std(norm_v) / np.sqrt(len(norm_v))) if len(norm_v) > 1 else 0.0,
                    'count': len(norm_v),
                })
            # OACF
            o_arr = np.asarray(oacf_pairs_by_lag[m])
            if len(o_arr) > 0:
                oacf_rows.append({
                    'lag_frames': m,
                    'lag_time_s': lag_t,
                    'corr': float(np.mean(o_arr)),
                    'sem': float(np.std(o_arr) / np.sqrt(len(o_arr))) if len(o_arr) > 1 else 0.0,
                    'count': len(o_arr),
                })
            # SACF (Fluctuation normalized)
            s_arr = np.asarray(sacf_fluc_pairs_by_lag[m])
            if len(s_arr) > 0 and mean_s_fluc_sq > 0:
                norm_s = s_arr / mean_s_fluc_sq
                sacf_rows.append({
                    'lag_frames': m,
                    'lag_time_s': lag_t,
                    'corr': float(np.mean(norm_s)),
                    'sem': float(np.std(norm_s) / np.sqrt(len(norm_s))) if len(norm_s) > 1 else 0.0,
                    'count': len(norm_s),
                })

        results[s] = {
            'vacf': pd.DataFrame(vacf_rows),
            'oacf': pd.DataFrame(oacf_rows),
            'sacf': pd.DataFrame(sacf_rows),
            'mean_speed': mean_speed_s,
        }

    return results


def fit_autocorrelation_exponential(
    df_corr: pd.DataFrame,
    max_lag_s: Optional[float] = None,
    min_points: int = 3,
    with_offset: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    自己相関関数 C(tau) に対し、
    with_offset=True の場合:  C(tau) = (1 - A) * exp(-tau / tau_corr) + A  (正規化 C(0)=1, 残存オフセット A)
    with_offset=False の場合: C(tau) = exp(-tau / tau_corr)
    をフィッティングして緩和時間 tau_corr および漸近オフセット A を推定する。

    Parameters
    ----------
    df_corr : pd.DataFrame
        'lag_time_s', 'corr', 'sem' (optional), 'count' (optional) を含む DataFrame
    max_lag_s : Optional[float]
        フィッティングに使用する最大ラグ時間 [s]
    min_points : int, default 3
        フィッティングに必要な最小点数
    with_offset : bool, default True
        True の場合、(1 - A) * exp(-tau / tau) + A モデルを使用

    Returns
    -------
    dict:
        tau_corr_s, tau_err_s, offset_A, offset_A_err, r_squared, fit_t, fit_corr, count
    """
    if df_corr is None or df_corr.empty or len(df_corr) < min_points:
        return {
            'tau_corr_s': np.nan, 'tau_err_s': np.nan,
            'offset_A': np.nan, 'offset_A_err': np.nan,
            'r_squared': np.nan, 'fit_t': np.array([]), 'fit_corr': np.array([]), 'count': 0,
        }

    df_sub = df_corr.copy()
    if max_lag_s is not None:
        df_sub = df_sub[df_sub['lag_time_s'] <= max_lag_s]

    t_data = df_sub['lag_time_s'].values
    c_data = df_sub['corr'].values

    valid_mask = np.isfinite(c_data) & np.isfinite(t_data)
    if np.sum(valid_mask) < min_points:
        return {
            'tau_corr_s': np.nan, 'tau_err_s': np.nan,
            'offset_A': np.nan, 'offset_A_err': np.nan,
            'r_squared': np.nan, 'fit_t': np.array([]), 'fit_corr': np.array([]), 'count': len(t_data),
        }

    t_val = t_data[valid_mask]
    c_val = c_data[valid_mask]

    sigma = None
    if 'sem' in df_sub.columns:
        s_val = df_sub.loc[valid_mask, 'sem'].values
        if np.all(s_val > 0) and np.all(np.isfinite(s_val)):
            sigma = np.maximum(s_val, 1e-4)

    if with_offset:
        # C(tau) = (1 - A) * exp(-tau / tau_corr) + A
        def exp_offset_func(t, tau, A):
            return (1.0 - A) * np.exp(-t / np.maximum(tau, 1e-6)) + A

        # 初期値推定
        tail_len = max(1, len(c_val) // 4)
        a_init = float(np.clip(np.mean(c_val[-tail_len:]), -0.1, 0.6))
        tau_init = 15.0

        try:
            popt, pcov = curve_fit(
                exp_offset_func,
                t_val,
                c_val,
                p0=[tau_init, a_init],
                bounds=([0.1, -0.3], [1000.0, 0.95]),
                sigma=sigma,
                absolute_sigma=False if sigma is not None else False,
                maxfev=5000,
            )
            tau_corr = float(popt[0])
            offset_A = float(popt[1])
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
            tau_err = float(perr[0]) if np.isfinite(perr[0]) else np.nan
            offset_A_err = float(perr[1]) if np.isfinite(perr[1]) else np.nan

            pred_c = exp_offset_func(t_val, tau_corr, offset_A)
            ss_res = np.sum((c_val - pred_c)**2)
            ss_tot = np.sum((c_val - np.mean(c_val))**2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan

            fit_t = np.linspace(0, np.max(t_data) * 1.1, 150)
            fit_corr = exp_offset_func(fit_t, tau_corr, offset_A)

        except Exception:
            tau_corr, tau_err, offset_A, offset_A_err, r2 = np.nan, np.nan, np.nan, np.nan, np.nan
            fit_t, fit_corr = np.array([]), np.array([])

    else:
        # C(tau) = exp(-tau / tau_corr) (ゼロ切片対数回帰)
        pos_mask = (c_data > 0) & (t_data > 0) & np.isfinite(c_data)
        if np.sum(pos_mask) < min_points - 1:
            return {
                'tau_corr_s': np.nan, 'tau_err_s': np.nan,
                'offset_A': 0.0, 'offset_A_err': 0.0,
                'r_squared': np.nan, 'fit_t': np.array([]), 'fit_corr': np.array([]), 'count': len(t_data),
            }

        t_pos = t_data[pos_mask]
        c_pos = c_data[pos_mask]

        if sigma is not None:
            weights = 1.0 / np.maximum(df_sub.loc[pos_mask, 'sem'].values, 1e-4)
        else:
            weights = np.sqrt(c_pos)

        log_c = np.log(c_pos)
        denom = np.sum((weights * t_pos)**2)
        numer = np.sum((weights**2) * t_pos * log_c)
        slope = float(numer / denom) if denom > 0 else -1.0

        if slope < 0:
            tau_corr = float(-1.0 / slope)
            pred_log_c = slope * t_pos
            residuals = log_c - pred_log_c
            df_resid = max(1, len(t_pos) - 1)
            s_sq = np.sum(weights * (residuals**2)) / (np.sum(weights) * (df_resid / len(t_pos)) + 1e-12)
            se_slope = np.sqrt(s_sq / (denom + 1e-12))
            tau_err = float((tau_corr**2) * se_slope) if np.isfinite(se_slope) else np.nan
        else:
            tau_corr = np.nan
            tau_err = np.nan

        offset_A = 0.0
        offset_A_err = 0.0

        if not np.isnan(tau_corr):
            pred_c = np.exp(-t_data / tau_corr)
            ss_res = np.sum((c_data - pred_c)**2)
            ss_tot = np.sum((c_data - np.mean(c_data))**2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan
            fit_t = np.linspace(0, np.max(t_data) * 1.1, 150)
            fit_corr = np.exp(-fit_t / tau_corr)
        else:
            r2 = np.nan
            fit_t = np.array([])
            fit_corr = np.array([])

    return {
        'tau_corr_s': tau_corr,
        'tau_err_s': tau_err,
        'offset_A': offset_A,
        'offset_A_err': offset_A_err,
        'r_squared': r2,
        'fit_t': fit_t,
        'fit_corr': fit_corr,
        'count': len(t_data),
    }


def fit_active_brownian_msd(
    lag_times: np.ndarray,
    msd_vals: np.ndarray,
    v0: float,
    sem_vals: Optional[np.ndarray] = None,
    min_points: int = 3,
) -> dict:
    """
    Run 状態の MSD 曲線に対し、アクティブブラウニアン粒子 (ABP) の理論 MSD:
        <Δr^2(t)> = 4 * D_t * t + 2 * v_0^2 * tau_r * [ t - tau_r * (1 - exp(-t / tau_r)) ]
    を非線形最小二乗法でフィッティングし、並進拡散係数 D_t および回転・配向緩和時間 tau_r を推定する。

    Parameters
    ----------
    lag_times : np.ndarray
        ラグ時間 t [s]
    msd_vals : np.ndarray
        MSD 実測値 <Δr^2(t)> [µm^2]
    v0 : float
        Run 状態の平均速度 [µm/s] (固定パラメータ)
    sem_vals : Optional[np.ndarray]
        MSD の標準誤差 (重み付け用)
    min_points : int, default 3

    Returns
    -------
    dict:
        Dt_um2_s, Dt_err_um2_s,
        tau_r_s, tau_r_err_s,
        v0_um_s,
        D_eff_um2_s,
        lambda_p_um,
        r_squared,
        fit_t, fit_msd,
        count
    """
    t_arr = np.asarray(lag_times, dtype=float)
    m_arr = np.asarray(msd_vals, dtype=float)

    valid = np.isfinite(t_arr) & np.isfinite(m_arr) & (t_arr > 0) & (m_arr > 0)
    t_data = t_arr[valid]
    m_data = m_arr[valid]

    if len(t_data) < min_points or v0 <= 0:
        return {
            'Dt_um2_s': np.nan, 'Dt_err_um2_s': np.nan,
            'tau_r_s': np.nan, 'tau_r_err_s': np.nan,
            'v0_um_s': v0,
            'D_eff_um2_s': np.nan,
            'lambda_p_um': np.nan,
            'r_squared': np.nan,
            'fit_t': np.array([]), 'fit_msd': np.array([]),
            'count': len(t_data),
        }

    sigma = None
    if sem_vals is not None:
        s_arr = np.asarray(sem_vals, dtype=float)[valid]
        if np.all(s_arr > 0) and np.all(np.isfinite(s_arr)):
            sigma = np.maximum(s_arr, 1e-4)

    def abp_msd_model(t, Dt, tau_r):
        tr = np.maximum(tau_r, 1e-6)
        # 4 * Dt * t + 2 * v0^2 * tau_r * [ t - tau_r * (1 - exp(-t / tau_r)) ]
        return 4.0 * Dt * t + 2.0 * (v0**2) * tr * (t - tr * (1.0 - np.exp(-t / tr)))

    # 初期値推定
    dt_init = max(1e-4, float(m_data[0] / (4.0 * t_data[0])))
    tau_r_init = 20.0

    try:
        popt, pcov = curve_fit(
            abp_msd_model,
            t_data,
            m_data,
            p0=[dt_init, tau_r_init],
            bounds=([0.0, 1e-3], [100.0, 1e4]),
            sigma=sigma,
            absolute_sigma=False if sigma is not None else False,
            maxfev=5000,
        )
        Dt_fit = float(popt[0])
        tau_r_fit = float(popt[1])
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        Dt_err = float(perr[0]) if np.isfinite(perr[0]) else np.nan
        tau_r_err = float(perr[1]) if np.isfinite(perr[1]) else np.nan

        pred_m = abp_msd_model(t_data, Dt_fit, tau_r_fit)
        ss_res = np.sum((m_data - pred_m)**2)
        ss_tot = np.sum((m_data - np.mean(m_data))**2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan

        # 有効拡散係数 D_eff = Dt + v0^2 * tau_r / 2
        D_eff = float(Dt_fit + 0.5 * (v0**2) * tau_r_fit)
        # 持続走行長 lambda_p = v0 * tau_r
        lambda_p = float(v0 * tau_r_fit)

        fit_t = np.logspace(np.log10(np.min(t_data) * 0.8), np.log10(np.max(t_data) * 1.5), 150)
        fit_msd = abp_msd_model(fit_t, Dt_fit, tau_r_fit)

    except Exception:
        Dt_fit, Dt_err = np.nan, np.nan
        tau_r_fit, tau_r_err = np.nan, np.nan
        D_eff, lambda_p, r2 = np.nan, np.nan, np.nan
        fit_t, fit_msd = np.array([]), np.array([])

    return {
        'Dt_um2_s': Dt_fit,
        'Dt_err_um2_s': Dt_err,
        'tau_r_s': tau_r_fit,
        'tau_r_err_s': tau_r_err,
        'v0_um_s': v0,
        'D_eff_um2_s': D_eff,
        'lambda_p_um': lambda_p,
        'r_squared': r2,
        'fit_t': fit_t,
        'fit_msd': fit_msd,
        'count': len(t_data),
    }


