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
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.epsilon = epsilon
        self.min_covar = min_covar

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            min_covar=min_covar,
            init_params="stmc",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> "CargoGaussianHMM":
        """観測シーケンスから HMM パラメータを学習し、状態を速度昇順にソートする。"""
        if len(X) == 0:
            raise ValueError("X must not be empty.")

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


def fit_exponential_distribution(
    durations: List[float],
    min_val: Optional[float] = None,
) -> Tuple[float, float, float]:
    """持続時間データに対して指数分布 P(t) = (1/tau) * exp(-t/tau) の最尤推定を行う。"""
    arr = np.asarray(durations)
    if min_val is not None:
        arr = arr[arr >= min_val]

    if len(arr) < 3:
        return np.nan, np.nan, np.nan

    tau_mle = float(np.mean(arr))
    tau_err = tau_mle / np.sqrt(len(arr))

    sorted_d = np.sort(arr)
    ccdf = 1.0 - (np.arange(1, len(sorted_d) + 1) - 0.5) / len(sorted_d)
    ccdf = np.clip(ccdf, 1e-6, 1.0)
    theo_ccdf = np.exp(-sorted_d / tau_mle)

    y_log = np.log(ccdf)
    y_theo_log = np.log(np.clip(theo_ccdf, 1e-6, 1.0))
    ss_res = np.sum((y_log - y_theo_log)**2)
    ss_tot = np.sum((y_log - np.mean(y_log))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    return tau_mle, tau_err, r2


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
