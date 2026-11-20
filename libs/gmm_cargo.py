"""
libs/gmm_cargo.py

小貨物（0.63 µm, 1.18 µm）の運動モード（Run / Tumble）を、
時間変化する遠方微小管流速 u_inf(t) で規格化した瞬時速力の対数
    O_t = ln(v(t) / u_inf(t))
に対する 1次元ガウス混合モデル（1D Gaussian Mixture Model: GMM）により
分類・定量化するためのモジュールです。

状態の定義:
    - State 0: Tumble（低速状態）
    - State 1: Run（高速状態）
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture

from libs import hmm_cargo as hc
from libs import hmm_flow_correlation as hfc

RAD_MAP: Dict[str, int] = {
    "beads06um": 3,
    "beads1um": 6,
    "beads3um": 15,
    "beads5um": 25,
    "beads7um": 35,
    "beads20um": 95,
}

BEST_MAP: Dict[str, int] = {
    "beads06um": 5,
    "beads1um": 10,
    "beads3um": 20,
    "beads5um": 30,
    "beads7um": 40,
    "beads20um": 100,
}

STATE_COLORS: Dict[int, str] = {
    0: "#d95f02",  # Tumble: オレンジ系
    1: "#1b9e77",  # Run: 青緑系
    2: "#7570b3",  # K=3 時の第3状態
}

STATE_NAMES: Dict[int, Dict[int, str]] = {
    1: {0: "Single"},
    2: {0: "Tumble", 1: "Run"},
    3: {0: "Tumble", 1: "Intermediate", 2: "Run"},
}


def get_state_label(state_id: int, n_components: int = 2) -> str:
    """状態番号に対応する標準ラベル（K=2 では 0: Tumble, 1: Run）を返す。"""
    if state_id == -1:
        return "All"
    return STATE_NAMES.get(n_components, {}).get(state_id, f"State {state_id}")


def relabel_state_dataframe(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """hmm_cargo 側の DataFrame 内ラベル（'Tumble / Pause' 等）を 'Tumble' / 'Run' に統一する。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "state" in out.columns:
        mapping = {s: get_state_label(int(s), n_components) for s in out["state"].dropna().unique()}
        if "state_label" in out.columns:
            out["state_label"] = out["state"].map(mapping)
        if "label" in out.columns:
            out["label"] = out["state"].map(mapping)
    return out


def compute_contiguous_lengths(df_obs: pd.DataFrame) -> List[int]:
    """
    観測 DataFrame から、同一粒子かつ連続フレーム (diff(frame) == 1) ごとの
    シーケンス長リストを算出する。
    """
    if df_obs.empty:
        return []
    lengths: List[int] = []
    group_cols = ["exp_dir", "particle"] if "exp_dir" in df_obs.columns else ["particle"]
    for _, grp in df_obs.groupby(group_cols, sort=False):
        frames = grp["frame"].to_numpy()
        if len(frames) == 0:
            continue
        splits = np.where(np.diff(frames) != 1)[0] + 1
        pts = np.concatenate([[0], splits, [len(frames)]])
        for start, end in zip(pts[:-1], pts[1:]):
            lengths.append(int(end - start))
    return lengths


def extract_gmm_features(
    df_tracks: pd.DataFrame,
    df_uinf: pd.DataFrame,
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 0.0,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    """
    粒子軌跡データと時間変化する遠方流速 u_inf(t) から、1D GMM 用の対数規格化速力
        O_t = ln(v(t) / u_inf(t) + epsilon)
    を抽出する。

    Parameters
    ----------
    df_tracks : pd.DataFrame
        'particle', 'frame', 'x', 'y', 'exp_dir' を含むトラッキング DataFrame
        （事前に experiment_cutoff と strict_active_particles でフィルタ済みを想定）
    df_uinf : pd.DataFrame
        'exp_dir', 'frame', 'u_inf_t' を含む遠方流速 DataFrame
    tau : int, default 1
        ラグステップ数
    scale : float, default 0.11
        空間スケール [um/pixel]
    frame_interval : float, default 4.0
        フレーム間隔 [s]
    epsilon : float, default 0.0
        ゼロ速度回避用微小定数（全ステップ v > 0 の場合は 0.0 で純粋な ln(v/u_inf(t))）

    Returns
    -------
    X : np.ndarray, shape (N, 1)
        GMM 入力用の対数規格化速力配列
    lengths : List[int]
        連続フレームセグメントごとの長さリスト
    df_obs : pd.DataFrame
        各観測フレームの詳細 DataFrame
    """
    required_cols = {"particle", "frame", "x", "y", "exp_dir"}
    if not required_cols.issubset(df_tracks.columns):
        raise ValueError(f"df_tracks must contain columns: {required_cols}")
    if not {"exp_dir", "frame", "u_inf_t"}.issubset(df_uinf.columns):
        raise ValueError("df_uinf must contain columns: {'exp_dir', 'frame', 'u_inf_t'}")

    extra_cols = [c for c in ["orig_particle", "bead_name"] if c in df_tracks.columns]
    cols = ["exp_dir", "particle", "frame", "x", "y"] + extra_cols
    df_sorted = df_tracks[cols].sort_values(by=["exp_dir", "particle", "frame"]).reset_index(drop=True)

    dt_sec = float(tau) * float(frame_interval)
    chunks: List[pd.DataFrame] = []

    for (exp_dir, particle_id), grp in df_sorted.groupby(["exp_dir", "particle"], sort=False):
        frames = grp["frame"].to_numpy(dtype=np.int64)
        x_um = grp["x"].to_numpy(dtype=float) * scale
        y_um = grp["y"].to_numpy(dtype=float) * scale

        if len(frames) < tau + 1:
            continue

        valid = (frames[tau:] == frames[:-tau] + tau)
        if not np.any(valid):
            continue

        idx0 = np.where(valid)[0]
        idx1 = idx0 + tau
        dx = x_um[idx1] - x_um[idx0]
        dy = y_um[idx1] - y_um[idx0]
        v = np.hypot(dx, dy) / dt_sec

        chunk = pd.DataFrame({
            "exp_dir": exp_dir,
            "particle": particle_id,
            "frame": frames[idx0],
            "x_um": x_um[idx0],
            "y_um": y_um[idx0],
            "dx_um": dx,
            "dy_um": dy,
            "v": v,
        })
        for col in extra_cols:
            chunk[col] = grp[col].iloc[0]
        chunks.append(chunk)

    if not chunks:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_obs = pd.concat(chunks, ignore_index=True)

    # 時間ごとの u_inf(t) を (exp_dir, frame) で結合
    uinf_clean = df_uinf[["exp_dir", "frame", "u_inf_t"]].drop_duplicates(subset=["exp_dir", "frame"])
    df_obs = df_obs.merge(uinf_clean, on=["exp_dir", "frame"], how="left")
    df_obs = df_obs[df_obs["u_inf_t"].notna() & np.isfinite(df_obs["u_inf_t"]) & (df_obs["u_inf_t"] > 0)].copy()

    df_obs["v_norm"] = df_obs["v"] / df_obs["u_inf_t"]
    valid_log = (df_obs["v_norm"] + epsilon) > 0
    df_obs = df_obs[valid_log].copy()
    if df_obs.empty:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_obs["log_v_norm"] = np.log(df_obs["v_norm"] + epsilon)
    df_obs = df_obs.sort_values(by=["exp_dir", "particle", "frame"]).reset_index(drop=True)

    lengths = compute_contiguous_lengths(df_obs)
    X = df_obs[["log_v_norm"]].to_numpy(dtype=float)
    return X, lengths, df_obs


class CargoGaussianGMM:
    """
    対数規格化速力 O_t = ln(v(t) / u_inf(t)) に対する 1次元 Gaussian Mixture Model。

    学習後にコンポーネントを平均値の昇順に整列し、
    State 0 = Tumble（遅い状態）、State 1 = Run（速い状態）を一貫して保証する。
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_init: int = 20,
        max_iter: int = 1000,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        random_state: int = 42,
        epsilon: float = 0.0,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.epsilon = epsilon

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        self.is_fitted = False
        self._order = np.arange(n_components)
        self._inv_order = np.arange(n_components)

    def fit(self, X: np.ndarray) -> "CargoGaussianGMM":
        """GMM を学習し、平均値昇順（0: Tumble, 1: Run）の並び順を確立する。"""
        if len(X) == 0:
            raise ValueError("X must not be empty.")
        self.model.fit(X)
        self._order = np.argsort(self.model.means_[:, 0])
        self._inv_order = np.argsort(self._order)
        self.is_fitted = True
        return self

    @property
    def means_(self) -> np.ndarray:
        """状態順に整列された平均値 (K, 1)。"""
        self._require_fitted()
        return self.model.means_[self._order]

    @property
    def weights_(self) -> np.ndarray:
        """状態順に整列された混合重み (K,)。"""
        self._require_fitted()
        return self.model.weights_[self._order]

    @property
    def variances_(self) -> np.ndarray:
        """状態順に整列された各コンポーネントの分散 (K,)。"""
        self._require_fitted()
        covs = self.model.covariances_
        if self.covariance_type == "full":
            raw_vars = covs[:, 0, 0]
        elif self.covariance_type == "diag":
            raw_vars = covs[:, 0]
        elif self.covariance_type == "spherical":
            raw_vars = covs
        elif self.covariance_type == "tied":
            raw_vars = np.full(self.n_components, float(covs[0, 0]))
        else:
            raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")
        return np.asarray(raw_vars, dtype=float)[self._order]

    @property
    def stds_(self) -> np.ndarray:
        """状態順に整列された各コンポーネントの標準偏差 (K,)。"""
        return np.sqrt(np.maximum(self.variances_, 1e-12))

    def _require_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """各観測点の事後確率 P(S_t = k | O_t) を状態順（0: Tumble, 1: Run）で返す。"""
        self._require_fitted()
        return self.model.predict_proba(X)[:, self._order]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """事後確率最大 (MAP) による状態ラベル（0: Tumble, 1: Run）を返す。"""
        self._require_fitted()
        raw_pred = self.model.predict(X)
        return self._inv_order[raw_pred]

    def score(self, X: np.ndarray) -> float:
        """総対数尤度 log P(O | theta) を返す。"""
        self._require_fitted()
        return float(self.model.score(X) * len(X))

    def compute_bic_aic_icl(self, X: np.ndarray) -> Tuple[float, float, float]:
        """
        BIC, AIC, および ICL (Integrated Completed Likelihood) を算出する。
        ICL = BIC + 2 * Entropy(posterior) により、重なりが大きい過剰分割を抑制する。
        """
        self._require_fitted()
        bic = float(self.model.bic(X))
        aic = float(self.model.aic(X))
        proba = self.predict_proba(X)
        entropy = float(-np.sum(proba * np.log(np.maximum(proba, 1e-12))))
        icl = bic + 2.0 * entropy
        return bic, aic, icl

    def get_separation_index(self) -> float:
        """K=2 の状態分離度 S_v = |mu_Run - mu_Tumble| / sqrt((sigma_0^2 + sigma_1^2)/2)。"""
        if self.n_components != 2:
            return np.nan
        mu = self.means_[:, 0]
        var = self.variances_
        pooled_sd = np.sqrt(0.5 * (var[0] + var[1]))
        return float(abs(mu[1] - mu[0]) / (pooled_sd + 1e-12))

    def get_decision_boundary(self) -> Optional[float]:
        """
        K=2 において P(Run | x*) = P(Tumble | x*) = 0.5 となる対数規格化速力 x* を求める。
        """
        if self.n_components != 2:
            return None
        mu0, mu1 = float(self.means_[0, 0]), float(self.means_[1, 0])
        v0, v1 = float(self.variances_[0]), float(self.variances_[1])
        s0, s1 = np.sqrt(max(v0, 1e-12)), np.sqrt(max(v1, 1e-12))
        w0, w1 = float(self.weights_[0]), float(self.weights_[1])

        a = 0.5 / v1 - 0.5 / v0
        b = mu0 / v0 - mu1 / v1
        c = 0.5 * (mu1 ** 2) / v1 - 0.5 * (mu0 ** 2) / v0 + np.log((w0 * s1) / (w1 * s0 + 1e-12) + 1e-12)

        if abs(a) < 1e-12:
            return float(-c / b) if abs(b) > 1e-12 else 0.5 * (mu0 + mu1)

        disc = b ** 2 - 4.0 * a * c
        if disc < 0:
            return 0.5 * (mu0 + mu1)
        roots = [(-b - np.sqrt(disc)) / (2.0 * a), (-b + np.sqrt(disc)) / (2.0 * a)]
        between = [r for r in roots if mu0 <= r <= mu1]
        if len(between) == 1:
            return float(between[0])
        midpoint = 0.5 * (mu0 + mu1)
        return float(min(roots, key=lambda r: abs(r - midpoint)))

    def get_state_summary(self) -> pd.DataFrame:
        """各状態（0: Tumble, 1: Run）のモデル統計サマリー DataFrame を返す。"""
        self._require_fitted()
        means = self.means_[:, 0]
        vars_ = self.variances_
        stds = self.stds_
        weights = self.weights_

        records = []
        for i in range(self.n_components):
            mu = float(means[i])
            var = float(vars_[i])
            sd = float(stds[i])
            v_norm_geom = max(0.0, float(np.exp(mu) - self.epsilon))
            v_norm_mean = max(0.0, float(np.exp(mu + 0.5 * var) - self.epsilon))
            records.append({
                "state": i,
                "label": get_state_label(i, self.n_components),
                "mean_log_v_norm": mu,
                "std_log_v_norm": sd,
                "v_norm_geom": v_norm_geom,
                "v_norm_model_mean": v_norm_mean,
                "mixture_weight": float(weights[i]),
            })
        return pd.DataFrame(records)


def calc_empirical_transitions(
    states: np.ndarray,
    lengths: List[int],
    n_components: int = 2,
    frame_interval: float = 4.0,
) -> dict:
    """
    連続フレームセグメント内の分類状態系列から経験的遷移回数・遷移確率・スイッチング率を算出する。
    """
    counts = np.zeros((n_components, n_components), dtype=int)
    state_frames = np.zeros(n_components, dtype=int)

    curr = 0
    for length in lengths:
        seq = states[curr:curr + length]
        curr += length
        if len(seq) == 0:
            continue
        for s in range(n_components):
            state_frames[s] += int(np.sum(seq == s))
        if len(seq) >= 2:
            for s_from, s_to in zip(seq[:-1], seq[1:]):
                counts[int(s_from), int(s_to)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    trans_prob = counts / np.maximum(row_sums, 1)

    rates = {}
    if n_components == 2:
        # Tumble (0) -> Run (1) and Run (1) -> Tumble (0)
        t_tumble_s = max(int(row_sums[0, 0]) * frame_interval, 1e-12)
        t_run_s = max(int(row_sums[1, 0]) * frame_interval, 1e-12)
        rates["n_T_to_R"] = int(counts[0, 1])
        rates["n_R_to_T"] = int(counts[1, 0])
        rates["k_T_to_R_per_s"] = float(counts[0, 1] / t_tumble_s) if row_sums[0, 0] > 0 else np.nan
        rates["k_R_to_T_per_s"] = float(counts[1, 0] / t_run_s) if row_sums[1, 0] > 0 else np.nan

    return {
        "counts": counts,
        "trans_prob": trans_prob,
        "state_frames": state_frames,
        "rates": rates,
    }


def calc_state_dependent_autocorrelations(
    df_obs: pd.DataFrame,
    n_components: int = 2,
    frame_interval: float = 4.0,
    max_lag_frames: int = 25,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    各運動状態（0: Tumble, 1: Run）の連続フレームセグメントから
    VACF, OACF, SACF（および正規化速さ v/u_inf(t) の SACF）を算出する。
    (exp_dir, particle) ごとにグループ化し、フレーム欠損 (diff(frame) != 1) でセグメントを分割する。
    """
    if df_obs.empty or "pred_state" not in df_obs.columns:
        return {}

    df = df_obs.copy()
    if "vx" not in df.columns or "vy" not in df.columns:
        df["vx"] = df["dx_um"] / frame_interval
        df["vy"] = df["dy_um"] / frame_interval

    df = df.dropna(subset=["vx", "vy", "pred_state"]).copy()
    df["speed"] = df["v_norm"] if "v_norm" in df.columns else np.hypot(df["vx"], df["vy"])
    raw_speed = np.hypot(df["vx"], df["vy"])
    eps = 1e-12
    df["ex"] = df["vx"] / (raw_speed + eps)
    df["ey"] = df["vy"] / (raw_speed + eps)

    group_cols = ["exp_dir", "particle"] if "exp_dir" in df.columns else ["particle"]
    results: Dict[int, Dict[str, pd.DataFrame]] = {}

    for s in range(n_components):
        s_mask = df["pred_state"] == s
        mean_speed_s = float(np.mean(df.loc[s_mask, "speed"])) if np.any(s_mask) else 0.0

        vacf_pairs = {m: [] for m in range(max_lag_frames + 1)}
        oacf_pairs = {m: [] for m in range(max_lag_frames + 1)}
        sacf_pairs = {m: [] for m in range(max_lag_frames + 1)}
        v_sq_list: List[float] = []
        s_fluc_sq_list: List[float] = []

        for _, p_group in df.groupby(group_cols, sort=False):
            p_sorted = p_group.sort_values(by="frame")
            frames = p_sorted["frame"].to_numpy()
            states = p_sorted["pred_state"].to_numpy()
            vx = p_sorted["vx"].to_numpy()
            vy = p_sorted["vy"].to_numpy()
            ex = p_sorted["ex"].to_numpy()
            ey = p_sorted["ey"].to_numpy()
            spd = p_sorted["speed"].to_numpy()

            n_p = len(frames)
            if n_p < 2:
                continue

            frame_diffs = np.diff(frames)
            state_diffs = np.diff(states)
            splits = np.where((frame_diffs != 1) | (state_diffs != 0))[0] + 1
            starts = np.concatenate([[0], splits])
            ends = np.concatenate([splits, [n_p]])

            for st, ed in zip(starts, ends):
                if states[st] != s:
                    continue
                seg_len = ed - st
                if seg_len < 2:
                    continue

                seg_vx, seg_vy = vx[st:ed], vy[st:ed]
                seg_ex, seg_ey = ex[st:ed], ey[st:ed]
                seg_spd = spd[st:ed]

                v_sq_list.extend((seg_vx ** 2 + seg_vy ** 2).tolist())
                s_fluc_sq_list.extend(((seg_spd - mean_speed_s) ** 2).tolist())

                max_m = min(seg_len - 1, max_lag_frames)
                for m in range(max_m + 1):
                    if m == 0:
                        v_dot = seg_vx ** 2 + seg_vy ** 2
                        o_dot = np.ones(seg_len)
                        s_dot = (seg_spd - mean_speed_s) ** 2
                    else:
                        v_dot = seg_vx[:-m] * seg_vx[m:] + seg_vy[:-m] * seg_vy[m:]
                        o_dot = seg_ex[:-m] * seg_ex[m:] + seg_ey[:-m] * seg_ey[m:]
                        s_dot = (seg_spd[:-m] - mean_speed_s) * (seg_spd[m:] - mean_speed_s)

                    vacf_pairs[m].extend(v_dot.tolist())
                    oacf_pairs[m].extend(o_dot.tolist())
                    sacf_pairs[m].extend(s_dot.tolist())

        mean_v_sq = float(np.mean(v_sq_list)) if v_sq_list else 1.0
        mean_s_fluc_sq = float(np.mean(s_fluc_sq_list)) if s_fluc_sq_list else 1.0

        vacf_rows, oacf_rows, sacf_rows = [], [], []
        for m in range(max_lag_frames + 1):
            lag_t = m * frame_interval
            v_arr = np.asarray(vacf_pairs[m], dtype=float)
            if len(v_arr) > 0 and mean_v_sq > 0:
                nv = v_arr / mean_v_sq
                vacf_rows.append({
                    "lag_frames": m,
                    "lag_time_s": lag_t,
                    "corr": float(np.mean(nv)),
                    "sem": float(np.std(nv) / np.sqrt(len(nv))) if len(nv) > 1 else 0.0,
                    "count": len(nv),
                })

            o_arr = np.asarray(oacf_pairs[m], dtype=float)
            if len(o_arr) > 0:
                oacf_rows.append({
                    "lag_frames": m,
                    "lag_time_s": lag_t,
                    "corr": float(np.mean(o_arr)),
                    "sem": float(np.std(o_arr) / np.sqrt(len(o_arr))) if len(o_arr) > 1 else 0.0,
                    "count": len(o_arr),
                })

            s_arr = np.asarray(sacf_pairs[m], dtype=float)
            if len(s_arr) > 0 and mean_s_fluc_sq > 0:
                ns = s_arr / mean_s_fluc_sq
                sacf_rows.append({
                    "lag_frames": m,
                    "lag_time_s": lag_t,
                    "corr": float(np.mean(ns)),
                    "sem": float(np.std(ns) / np.sqrt(len(ns))) if len(ns) > 1 else 0.0,
                    "count": len(ns),
                })

        results[s] = {
            "vacf": pd.DataFrame(vacf_rows),
            "oacf": pd.DataFrame(oacf_rows),
            "sacf": pd.DataFrame(sacf_rows),
            "mean_speed": mean_speed_s,
        }

    return results


def calc_per_particle_summary(df_obs: pd.DataFrame, bead_name: str, diameter_um: float) -> pd.DataFrame:
    """粒子ごとの Run / Tumble 占有率および速度統計を集計する。"""
    if df_obs.empty:
        return pd.DataFrame()

    p_col = "orig_particle" if "orig_particle" in df_obs.columns else "particle"
    records = []
    for (exp_dir, pid), grp in df_obs.groupby(["exp_dir", p_col], sort=False):
        st = grp["pred_state"].to_numpy()
        run_mask = st == 1
        tumble_mask = st == 0
        records.append({
            "bead_name": bead_name,
            "diameter_um": diameter_um,
            "exp_dir": exp_dir,
            "particle": int(pid),
            "n_frames": len(grp),
            "n_run_frames": int(np.sum(run_mask)),
            "n_tumble_frames": int(np.sum(tumble_mask)),
            "f_run": float(np.mean(run_mask)),
            "f_tumble": float(np.mean(tumble_mask)),
            "mean_proba_run": float(grp["proba_run"].mean()) if "proba_run" in grp.columns else np.nan,
            "mean_v_um_s": float(grp["v"].mean()),
            "mean_u_inf_t_um_s": float(grp["u_inf_t"].mean()),
            "median_v_norm": float(grp["v_norm"].median()),
            "geom_mean_v_norm": float(np.exp(grp["log_v_norm"].mean())),
            "geom_mean_v_norm_run": float(np.exp(grp.loc[run_mask, "log_v_norm"].mean())) if np.any(run_mask) else np.nan,
            "geom_mean_v_norm_tumble": float(np.exp(grp.loc[tumble_mask, "log_v_norm"].mean())) if np.any(tumble_mask) else np.nan,
        })
    return pd.DataFrame(records)


def compute_masked_annulus_polar_order(
    exp_dir: Path,
    df_exp_obs: pd.DataFrame,
    bead_name: str,
    scale: float = 0.11,
    frame_interval: float = 4.0,
) -> pd.DataFrame:
    """
    GFP_flows.h5 から、貨物内部 (r <= RAD_MAP[bead_name]) をマスクした足印アニュラス
        RAD_MAP[bead_name] < r <= BEST_MAP[bead_name]
    における局所ポーラーオーダー Phi_t および関連指標を直接計算する。
    (scratch/extract_footprint_heterogeneity.py と同一定義)
    """
    if bead_name not in RAD_MAP or bead_name not in BEST_MAP:
        raise ValueError(f"Unknown bead_name for footprint mask: {bead_name}")
    flow_p = Path(exp_dir) / "GFP_flows.h5"
    if not flow_p.exists() or df_exp_obs.empty:
        return pd.DataFrame()

    p_rad = RAD_MAP[bead_name]
    best_rad = BEST_MAP[bead_name]
    r_ceil = best_rad
    v_unit = scale / frame_interval
    p_col = "orig_particle" if "orig_particle" in df_exp_obs.columns else "particle"

    records = []
    with h5py.File(str(flow_p), "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        flow = f[list(f.keys())[0]]
        total_frms = flow.shape[0]
        h, w = flow.shape[2], flow.shape[3]

        for frm, grp in df_exp_obs.groupby("frame"):
            frm_int = int(frm)
            if frm_int >= total_frms:
                continue
            for _, r in grp.iterrows():
                bx = int(round(float(r["x_um"]) / scale))
                by = int(round(float(r["y_um"]) / scale))
                dx_px = float(r["dx_um"]) / scale
                dy_px = float(r["dy_um"]) / scale

                y_min, y_max = max(0, by - r_ceil), min(h, by + r_ceil + 1)
                x_min, x_max = max(0, bx - r_ceil), min(w, bx + r_ceil + 1)
                if y_max <= y_min or x_max <= x_min:
                    continue

                patch = flow[frm_int, :, y_min:y_max, x_min:x_max]
                u_box, v_box = patch[0], patch[1]
                yg, xg = np.ogrid[y_min - by:y_max - by, x_min - bx:x_max - bx]
                dsq = xg ** 2 + yg ** 2
                m = (dsq <= best_rad ** 2) & (dsq > p_rad ** 2)
                if not np.any(m):
                    continue

                u_in = u_box[m].astype(np.float32)
                v_in = v_box[m].astype(np.float32)
                norm_in = np.hypot(u_in, v_in)
                valid = norm_in > 0
                if not np.any(valid):
                    continue

                u_hat = u_in[valid] / norm_in[valid]
                v_hat = v_in[valid] / norm_in[valid]
                phi_t = float(np.hypot(np.mean(u_hat), np.mean(v_hat)))

                u_bar = float(np.mean(u_in))
                v_bar = float(np.mean(v_in))
                u_fp_px = float(np.hypot(u_bar, v_bar))
                if u_fp_px > 0:
                    f_opp = float(np.mean((u_in * u_bar + v_in * v_bar) < 0))
                    v_par_um = float((dx_px * (u_bar / u_fp_px) + dy_px * (v_bar / u_fp_px)) * v_unit)
                else:
                    f_opp = 0.0
                    v_par_um = 0.0

                records.append({
                    "exp_dir": Path(exp_dir).name,
                    p_col: int(r[p_col]),
                    "frame": frm_int,
                    "phi_t": phi_t,
                    "f_opp": f_opp,
                    "v_parallel": v_par_um,
                    "u_fp": float(u_fp_px * v_unit),
                    "u_local_scalar": float(np.mean(norm_in) * v_unit),
                    "v_cargo": float(np.hypot(dx_px, dy_px) * v_unit),
                })

    return pd.DataFrame(records)


def load_or_compute_masked_annulus_data(
    df_obs: pd.DataFrame,
    bead_name: str,
    exp_dirs: List[Path],
    cache_path: Optional[Path] = None,
    recompute: bool = False,
    scale: float = 0.11,
    frame_interval: float = 4.0,
) -> pd.DataFrame:
    """
    貨物マスク付き足印アニュラスのポーラーオーダー (phi_t 等) を取得し、df_obs と結合する。
    cache_path (scratch/footprint_heterogeneity_cache.pkl) が存在し全フレームをカバーしていれば高速読込し、
    未計算または recompute=True の場合は GFP_flows.h5 から直接算出する。
    """
    if df_obs.empty:
        return pd.DataFrame()

    p_col = "orig_particle" if "orig_particle" in df_obs.columns else "particle"
    df_fp = pd.DataFrame()

    if (not recompute) and cache_path is not None and Path(cache_path).exists():
        cached = pd.read_pickle(cache_path)
        sub = cached[cached["bead_name"] == bead_name].copy()
        if not sub.empty:
            sub[p_col] = sub["particle"].astype(int)
            sub["frame"] = sub["frame"].astype(int)
            df_fp = sub[["exp_dir", p_col, "frame", "phi_t", "f_opp", "v_parallel", "u_fp", "u_local_scalar", "v_cargo"]]

    if df_fp.empty:
        parts = []
        exp_map = {d.name: d for d in exp_dirs}
        for exp_name, grp in df_obs.groupby("exp_dir"):
            if exp_name in exp_map:
                res = compute_masked_annulus_polar_order(
                    exp_map[exp_name], grp, bead_name=bead_name, scale=scale, frame_interval=frame_interval
                )
                if not res.empty:
                    parts.append(res)
        if not parts:
            return pd.DataFrame()
        df_fp = pd.concat(parts, ignore_index=True)

    merged = df_obs.merge(df_fp.drop_duplicates(subset=["exp_dir", p_col, "frame"]), on=["exp_dir", p_col, "frame"], how="inner")
    if merged.empty:
        return merged

    merged["u_fp_norm"] = merged["u_fp"] / merged["u_inf_t"]
    merged["v_par_norm"] = merged["v_parallel"] / merged["u_inf_t"]
    merged["eta"] = np.where(merged["u_fp"] > 0, merged["v_parallel"] / merged["u_fp"], np.nan)
    merged["cos_psi"] = np.where(merged["v_cargo"] > 0, merged["v_parallel"] / merged["v_cargo"], np.nan)
    return merged


def extract_masked_radial_polar_order(
    exp_dirs: List[Path],
    df_obs: pd.DataFrame,
    scale: float = 0.11,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    local_polar_w.zarr（particle_radius による貨物マスク済み）および local_polar_bg.zarr から、
    Run / Tumble / All / Background ごとの窓半径依存ポーラーオーダー Phi(R) を抽出する。
    """
    if df_obs.empty:
        return pd.DataFrame(), pd.DataFrame()

    p_col = "orig_particle" if "orig_particle" in df_obs.columns else "particle"
    exp_map = {d.name: d for d in exp_dirs}

    sample_records = []
    bg_records = []

    for exp_name, grp in df_obs.groupby("exp_dir"):
        ed = exp_map.get(exp_name)
        if ed is None:
            continue
        zp = ed / "local_polar_w.zarr"
        if not zp.exists():
            continue

        ds_p = xr.open_zarr(str(zp), consolidated=False)
        w_dim = "window size" if "window size" in ds_p.dims else "window_size"
        windows_px = ds_p.coords[w_dim].values.astype(int)
        p_mask_rad = int(ds_p.attrs.get("particle_radius", 0))
        valid_w_mask = windows_px > p_mask_rad
        windows_px = windows_px[valid_w_mask]
        windows_um = windows_px * scale

        frames = ds_p.coords["frame"].values.astype(int)
        particles = ds_p.coords["particle"].values.astype(int)
        f_to_idx = {f: i for i, f in enumerate(frames)}
        p_to_idx = {p: i for i, p in enumerate(particles)}

        arr_p = ds_p["polar_order"].transpose(w_dim, "frame", "particle").values[valid_w_mask]

        used_frames = set()
        for _, r in grp.iterrows():
            f_val = int(r["frame"])
            p_val = int(r[p_col])
            st = int(r["pred_state"])
            fi = f_to_idx.get(f_val)
            pi = p_to_idx.get(p_val)
            if fi is None or pi is None:
                continue
            used_frames.add(f_val)
            vals = arr_p[:, fi, pi]
            if not np.any(np.isfinite(vals)):
                continue
            for w_px, w_um, phi_v in zip(windows_px, windows_um, vals):
                if np.isfinite(phi_v):
                    sample_records.append({
                        "exp_dir": exp_name,
                        "orig_particle": p_val,
                        "frame": f_val,
                        "pred_state": st,
                        "window_px": int(w_px),
                        "window_um": float(w_um),
                        "phi": float(phi_v),
                    })

        bg_zp = ed / "local_polar_bg.zarr"
        if bg_zp.exists() and used_frames:
            ds_bg = xr.open_zarr(str(bg_zp), consolidated=False)
            if "polar_order" in ds_bg:
                bg_w_dim = "window size" if "window size" in ds_bg.dims else "window_size"
                bg_w_px = ds_bg.coords[bg_w_dim].values.astype(int)
                bg_frames = ds_bg.coords["frame"].values.astype(int)
                bg_f_to_idx = {f: i for i, f in enumerate(bg_frames)}
                bg_w_to_idx = {w: i for i, w in enumerate(bg_w_px)}
                bg_arr = ds_bg["polar_order"].transpose(bg_w_dim, "frame").values

                f_indices = [bg_f_to_idx[f] for f in sorted(used_frames) if f in bg_f_to_idx]
                if f_indices:
                    for w_px, w_um in zip(windows_px, windows_um):
                        wi = bg_w_to_idx.get(int(w_px))
                        if wi is None:
                            continue
                        vals_bg = bg_arr[wi, f_indices]
                        vals_bg = vals_bg[np.isfinite(vals_bg)]
                        if len(vals_bg) > 0:
                            bg_records.append({
                                "exp_dir": exp_name,
                                "window_px": int(w_px),
                                "window_um": float(w_um),
                                "phi_mean": float(np.mean(vals_bg)),
                                "n_frames": int(len(vals_bg)),
                            })

    return pd.DataFrame(sample_records), pd.DataFrame(bg_records)


def hierarchical_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    実験動画 -> 貨物粒子の2段階階層ブートストラップにより、平均値・SEM・95% CI を算出する。
    """
    sub = df[df[value_col].notna() & np.isfinite(df[value_col])]
    if sub.empty:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "sem": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_obs": 0, "n_cargos": 0, "n_movies": 0}

    p_col = "orig_particle" if "orig_particle" in sub.columns else "particle"
    obs_mean = float(sub[value_col].mean())
    obs_med = float(sub[value_col].median())
    obs_std = float(sub[value_col].std(ddof=1)) if len(sub) > 1 else 0.0
    n_cargos = int(sub.groupby(["exp_dir", p_col]).ngroups)
    n_movies = int(sub["exp_dir"].nunique())

    if n_movies == 0 or n_cargos <= 1:
        return {"mean": obs_mean, "median": obs_med, "std": obs_std, "sem": 0.0, "ci_lo": obs_mean, "ci_hi": obs_mean, "n_obs": len(sub), "n_cargos": n_cargos, "n_movies": n_movies}

    mov = {}
    for ex, grp in sub.groupby("exp_dir"):
        sums = grp.groupby(p_col)[value_col].sum().to_numpy(dtype=float)
        cnts = grp.groupby(p_col)[value_col].count().to_numpy(dtype=float)
        mov[ex] = (sums, cnts)

    keys = list(mov.keys())
    n_m = len(keys)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        tot_s, tot_c = 0.0, 0.0
        for k in rng.choice(keys, size=n_m, replace=True):
            s_arr, c_arr = mov[k]
            idx = rng.integers(0, len(s_arr), size=len(s_arr))
            tot_s += s_arr[idx].sum()
            tot_c += c_arr[idx].sum()
        boots[b] = tot_s / tot_c if tot_c > 0 else np.nan

    return {
        "mean": obs_mean,
        "median": obs_med,
        "std": obs_std,
        "sem": float(np.nanstd(boots, ddof=1)),
        "ci_lo": float(np.nanpercentile(boots, 2.5)),
        "ci_hi": float(np.nanpercentile(boots, 97.5)),
        "n_obs": int(len(sub)),
        "n_cargos": n_cargos,
        "n_movies": n_movies,
    }


def extract_spatial_correlation_curves(
    exp_dirs: List[Path],
    df_obs: pd.DataFrame,
    scale: float = 0.11,
) -> Tuple[np.ndarray, Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]], pd.DataFrame]:
    """
    angular_correlation_w.zarr（distance_px > particle_radius）および angular_correlation_bg.zarr から、
    状態別（0: Tumble, 1: Run, -1: All）の貨物別相関曲線データと背景相関データを抽出する。

    Returns
    -------
    d_um : np.ndarray
        有効距離配列 [um]
    cargo_curves_by_state : dict
        {state: {exp_dir: [(mean_curve_for_cargo, count_for_cargo), ...]}}
    df_bg_curves : pd.DataFrame
        背景相関曲線 DataFrame
    """
    if df_obs.empty:
        return np.array([]), {}, pd.DataFrame()

    p_col = "orig_particle" if "orig_particle" in df_obs.columns else "particle"
    exp_map = {d.name: d for d in exp_dirs}

    d_um_ref = None
    cargo_curves_by_state: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {0: {}, 1: {}, -1: {}}
    bg_sums = None
    bg_cnts = None

    for exp_name, grp in df_obs.groupby("exp_dir"):
        ed = exp_map.get(exp_name)
        if ed is None:
            continue
        zp = ed / "angular_correlation_w.zarr"
        if not zp.exists():
            continue

        ds = xr.open_zarr(str(zp), consolidated=False)
        d_px = ds.coords["distance"].values.astype(int)
        p_mask_rad = int(ds.attrs.get("particle_radius", 0))
        valid_d = d_px > p_mask_rad
        d_um = d_px[valid_d].astype(float) * scale
        if d_um_ref is None:
            d_um_ref = d_um

        frames = ds.coords["frame"].values.astype(int)
        particles = ds.coords["particle"].values.astype(int)
        f_to_idx = {f: i for i, f in enumerate(frames)}
        p_to_idx = {p: i for i, p in enumerate(particles)}
        arr = ds["angular_correlation"].transpose("distance", "frame", "particle").values[valid_d]

        used_frames = set()
        for pid, p_grp in grp.groupby(p_col, sort=False):
            pi = p_to_idx.get(int(pid))
            if pi is None:
                continue
            for st in (0, 1, -1):
                st_grp = p_grp if st == -1 else p_grp[p_grp["pred_state"] == st]
                if st_grp.empty:
                    continue
                f_indices = [f_to_idx[int(f)] for f in st_grp["frame"].to_numpy() if int(f) in f_to_idx]
                if not f_indices:
                    continue
                if st == -1:
                    used_frames.update(int(f) for f in st_grp["frame"].to_numpy() if int(f) in f_to_idx)
                mat = arr[:, f_indices, pi]
                cnt = np.sum(np.isfinite(mat), axis=1).astype(float)
                if np.any(cnt > 0):
                    c_mean = np.nanmean(mat, axis=1)
                    cargo_curves_by_state[st].setdefault(exp_name, []).append((c_mean, cnt))

        bg_zp = ed / "angular_correlation_bg.zarr"
        if bg_zp.exists() and used_frames:
            ds_bg = xr.open_zarr(str(bg_zp), consolidated=False)
            if "angular_correlation" in ds_bg:
                bg_d_px = ds_bg.coords["distance"].values.astype(int)
                bg_d_map = {int(dp): idx for idx, dp in enumerate(bg_d_px)}
                sel_d_idx = [bg_d_map[int(dp)] for dp in d_px[valid_d] if int(dp) in bg_d_map]
                bg_frames = ds_bg.coords["frame"].values.astype(int)
                bg_f_map = {f: i for i, f in enumerate(bg_frames)}
                sel_f_idx = [bg_f_map[f] for f in sorted(used_frames) if f in bg_f_map]
                if len(sel_d_idx) == len(d_um) and sel_f_idx:
                    bg_da = ds_bg["angular_correlation"]
                    other_dims = [dim for dim in bg_da.dims if dim not in ("distance", "frame")]
                    bg_arr = bg_da.transpose("distance", "frame", *other_dims).values[sel_d_idx][:, sel_f_idx]
                    bg_flat = bg_arr.reshape(len(d_um), -1)
                    if bg_sums is None:
                        bg_sums = np.zeros(len(d_um), dtype=float)
                        bg_cnts = np.zeros(len(d_um), dtype=float)
                        bg_sq_sums = np.zeros(len(d_um), dtype=float)
                    finite_m = np.isfinite(bg_flat)
                    bg_sums += np.where(finite_m, bg_flat, 0.0).sum(axis=1)
                    bg_sq_sums += np.where(finite_m, bg_flat ** 2, 0.0).sum(axis=1)
                    bg_cnts += finite_m.sum(axis=1)

    df_bg = pd.DataFrame()
    if d_um_ref is not None and bg_sums is not None:
        valid_b = bg_cnts > 1
        mean_b = np.where(valid_b, bg_sums / np.maximum(bg_cnts, 1.0), np.nan)
        var_b = np.where(valid_b, np.maximum(0.0, (bg_sq_sums / np.maximum(bg_cnts, 1.0)) - mean_b ** 2), 0.0)
        sem_b = np.where(valid_b, np.sqrt(var_b / np.maximum(bg_cnts, 1.0)), np.nan)
        df_bg = pd.DataFrame({
            "distance_um": d_um_ref[valid_b],
            "mean_correlation": mean_b[valid_b],
            "sem_correlation": sem_b[valid_b],
            "n_samples": bg_cnts[valid_b].astype(int),
        })

    return (d_um_ref if d_um_ref is not None else np.array([])), cargo_curves_by_state, df_bg


def extract_masked_footprint_spatial_correlation(
    exp_dirs: List[Path],
    df_obs: pd.DataFrame,
    bead_name: str,
    scale: float = 0.11,
    r_max_um: float = 20.5,
) -> Tuple[np.ndarray, Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]]:
    """
    貨物マスク済み足印アニュラス (RAD_MAP < r <= BEST_MAP) の平均配向 u_hat_fp を基準とし、
    外側のリング (r > BEST_MAP) の流れとの空間配向相関 C_fp(r) = < u_hat_fp . u_hat(r_c + r) >
    を GFP_flows.h5 から直接算出する（貨物中心ピクセルを完全に除外したクロスチェック用）。
    """
    if df_obs.empty:
        return np.array([]), {}

    r_in_fp, r_out_fp = RAD_MAP[bead_name], BEST_MAP[bead_name]
    r_px_list = np.arange(r_out_fp + 1, int(np.ceil(r_max_um / scale)) + 1, 2)
    r_um_list = r_px_list.astype(float) * scale
    p_col = "orig_particle" if "orig_particle" in df_obs.columns else "particle"
    exp_map = {d.name: d for d in exp_dirs}

    cargo_curves_by_state: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {0: {}, 1: {}, -1: {}}
    r_max_ceil = int(r_px_list[-1]) + 2

    for exp_name, grp in df_obs.groupby("exp_dir"):
        ed = exp_map.get(exp_name)
        if ed is None or not (ed / "GFP_flows.h5").exists():
            continue

        # Accumulate per (orig_particle, pred_state)
        acc: Dict[Tuple[int, int], List[np.ndarray]] = {}
        with h5py.File(str(ed / "GFP_flows.h5"), "r", rdcc_nbytes=128 * 1024 * 1024) as f:
            fl = f[list(f.keys())[0]]
            total_frms, H, W = fl.shape[0], fl.shape[2], fl.shape[3]
            for frm, f_grp in grp.groupby("frame"):
                fi = int(frm)
                if fi >= total_frms:
                    continue
                uv = fl[fi]
                u_all, v_all = uv[0].astype(np.float32), uv[1].astype(np.float32)
                for _, r in f_grp.iterrows():
                    bx = int(round(float(r["x_um"]) / scale))
                    by = int(round(float(r["y_um"]) / scale))
                    y0, y1 = max(0, by - r_max_ceil), min(H, by + r_max_ceil + 1)
                    x0, x1 = max(0, bx - r_max_ceil), min(W, bx + r_max_ceil + 1)
                    ub, vb = u_all[y0:y1, x0:x1], v_all[y0:y1, x0:x1]
                    nrm = np.hypot(ub, vb)
                    ok = nrm > 0
                    yg, xg = np.ogrid[y0 - by:y1 - by, x0 - bx:x1 - bx]
                    dist = np.hypot(xg, yg)
                    m_fp = (dist > r_in_fp) & (dist <= r_out_fp) & ok
                    if not np.any(m_fp):
                        continue
                    uh_fp = float(np.mean(ub[m_fp] / nrm[m_fp]))
                    vh_fp = float(np.mean(vb[m_fp] / nrm[m_fp]))
                    n_fp = float(np.hypot(uh_fp, vh_fp))
                    if n_fp <= 0:
                        continue
                    uh_fp /= n_fp
                    vh_fp /= n_fp

                    vec = np.full(len(r_px_list), np.nan, dtype=float)
                    ux_n = np.where(ok, ub / np.where(ok, nrm, 1.0), 0.0)
                    uy_n = np.where(ok, vb / np.where(ok, nrm, 1.0), 0.0)
                    for i_r, rp in enumerate(r_px_list):
                        mr = (dist >= rp - 1.0) & (dist <= rp + 1.0) & ok
                        if np.any(mr):
                            vec[i_r] = float(np.mean(ux_n[mr] * uh_fp + uy_n[mr] * vh_fp))

                    pid = int(r[p_col])
                    st = int(r["pred_state"])
                    acc.setdefault((pid, st), []).append(vec)
                    acc.setdefault((pid, -1), []).append(vec)

        for (pid, st), vecs in acc.items():
            mat = np.asarray(vecs, dtype=float)
            cnt = np.sum(np.isfinite(mat), axis=0).astype(float)
            if np.any(cnt > 0):
                cargo_curves_by_state[st].setdefault(exp_name, []).append((np.nanmean(mat, axis=0), cnt))

    return r_um_list, cargo_curves_by_state


def fit_xi_with_hierarchical_bootstrap(
    cargo_curves_by_movie: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    d_um: np.ndarray,
    radius_um: float,
    min_fit_dist: float = 0.0,
    max_fit_dist: float = 20.0,
    min_corr_threshold: float = 0.01,
    cutoff_ratio: Optional[float] = 1.0 / np.e,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    貨物別の相関曲線から重み付きプール相関曲線 C(r) を構築して xi をフィッティングし、
    さらに実験動画 -> 貨物粒子の2段階階層ブートストラップで xi および R_c / xi の 95% CI を算出する。
    """
    if not cargo_curves_by_movie or len(d_um) < 3:
        return pd.DataFrame(), {"xi_um": np.nan, "rc_over_xi": np.nan}

    def _aggregate_curve(items: List[Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        num = np.zeros(len(d_um), dtype=float)
        den = np.zeros(len(d_um), dtype=float)
        sq = np.zeros(len(d_um), dtype=float)
        for c_mean, cnt in items:
            valid = np.isfinite(c_mean) & (cnt > 0)
            num[valid] += c_mean[valid] * cnt[valid]
            sq[valid] += (c_mean[valid] ** 2) * cnt[valid]
            den[valid] += cnt[valid]
        ok = den >= 3
        mean_c = np.where(ok, num / np.maximum(den, 1.0), np.nan)
        var_c = np.where(ok, np.maximum(0.0, sq / np.maximum(den, 1.0) - mean_c ** 2), 0.0)
        sem_c = np.where(ok, np.sqrt(var_c / np.maximum(den, 1.0)), np.nan)
        return pd.DataFrame({
            "distance_um": d_um[ok],
            "mean_correlation": mean_c[ok],
            "sem_correlation": sem_c[ok],
            "n_samples": den[ok].astype(int),
        })

    all_items = [item for items in cargo_curves_by_movie.values() for item in items]
    df_curve = _aggregate_curve(all_items)
    fit_res = hfc.fit_flow_correlation_length(
        df_curve,
        min_fit_dist=min_fit_dist,
        max_fit_dist=max_fit_dist,
        min_corr_threshold=min_corr_threshold,
        cutoff_ratio=cutoff_ratio,
    )
    xi_obs = float(fit_res.get("xi_um", np.nan))
    rc_xi_obs = float(radius_um / xi_obs) if (np.isfinite(xi_obs) and xi_obs > 0) else np.nan

    keys = [k for k, v in cargo_curves_by_movie.items() if len(v) > 0]
    n_m = len(keys)
    n_c = len(all_items)

    xi_boots = []
    if n_m >= 1 and n_c >= 2:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            sampled = []
            for mk in rng.choice(keys, size=n_m, replace=True):
                items = cargo_curves_by_movie[mk]
                idx = rng.integers(0, len(items), size=len(items))
                for i in idx:
                    sampled.append(items[int(i)])
            df_b = _aggregate_curve(sampled)
            rb = hfc.fit_flow_correlation_length(
                df_b,
                min_fit_dist=min_fit_dist,
                max_fit_dist=max_fit_dist,
                min_corr_threshold=min_corr_threshold,
                cutoff_ratio=cutoff_ratio,
            )
            xb = float(rb.get("xi_um", np.nan))
            if np.isfinite(xb) and xb > 0:
                xi_boots.append(xb)

    if len(xi_boots) >= 10:
        xi_arr = np.asarray(xi_boots, dtype=float)
        rc_xi_arr = radius_um / xi_arr
        xi_lo, xi_hi = float(np.percentile(xi_arr, 2.5)), float(np.percentile(xi_arr, 97.5))
        rc_lo, rc_hi = float(np.percentile(rc_xi_arr, 2.5)), float(np.percentile(rc_xi_arr, 97.5))
    else:
        xi_lo = xi_hi = xi_obs
        rc_lo = rc_hi = rc_xi_obs

    summary = {
        "xi_um": xi_obs,
        "xi_fit_err_um": float(fit_res.get("xi_err_um", np.nan)),
        "xi_ci_lo": xi_lo,
        "xi_ci_hi": xi_hi,
        "rc_over_xi": rc_xi_obs,
        "rc_xi_ci_lo": rc_lo,
        "rc_xi_ci_hi": rc_hi,
        "r2_log": float(fit_res.get("r2_log", np.nan)),
        "r2": float(fit_res.get("r2", np.nan)),
        "amplitude": float(fit_res.get("amplitude", np.nan)),
        "r_fit_min_um": float(fit_res.get("r_fit_min_um", np.nan)),
        "r_fit_max_um": float(fit_res.get("r_fit_max_um", np.nan)),
        "n_cargos": n_c,
        "n_movies": n_m,
        "fit_r": fit_res.get("fit_r", np.array([])),
        "fit_c": fit_res.get("fit_c", np.array([])),
    }
    return df_curve, summary


# 再利用する解析関数（hmm_cargo.py とロジックを重複させない）
filter_state_glitches = hc.filter_state_glitches
calc_state_dwell_times = hc.calc_state_dwell_times
fit_exponential_pdf = hc.fit_exponential_pdf
fit_exponential_ccdf = hc.fit_exponential_ccdf
calc_posterior_statistics = hc.calc_posterior_statistics
calc_state_dependent_msd = hc.calc_state_dependent_msd
calc_state_dependent_msad = hc.calc_state_dependent_msad
fit_msad_linear = hc.fit_msad_linear
calc_state_dependent_turning_angles = hc.calc_state_dependent_turning_angles
calc_circular_stats_dict = hc.calc_circular_stats_dict
fit_von_mises_distribution = hc.fit_von_mises_distribution
fit_autocorrelation_exponential = hc.fit_autocorrelation_exponential
fit_active_brownian_msd = hc.fit_active_brownian_msd
fit_tumble_brownian_msd = hc.fit_tumble_brownian_msd
