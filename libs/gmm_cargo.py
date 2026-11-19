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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from libs import hmm_cargo as hc

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
