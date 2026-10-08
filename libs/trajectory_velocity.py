"""
libs/trajectory_velocity.py

トラッキング結果（(t, x, y) 座標系列）から粒子の滑らかな軌跡および瞬時速度を推定するモジュール。

実装手法:
1. カルマンフィルター / RTS (Rauch-Tung-Striebel) スムーザー
2. 平滑化スプライン (Smoothing Splines: UnivariateSpline) + 解析的微分
3. （参考比較用）移動平均差分法 (w=3) / 単純中心差分法
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


@dataclass
class VelocityEstimateResult:
    """速度推定結果を格納するデータクラス"""
    t: np.ndarray          # 時間配列 [s]
    x_smooth: np.ndarray   # 平滑化された x 座標
    y_smooth: np.ndarray   # 平滑化された y 座標
    vx: np.ndarray         # x 方向の速度 [μm/s または unit/s]
    vy: np.ndarray         # y 方向の速度 [μm/s または unit/s]
    v: np.ndarray          # スカラー速度 (速さ: sqrt(vx^2 + vy^2))
    theta: np.ndarray      # 進行方向角度 (rad: atan2(vy, vx))
    vx_std: Optional[np.ndarray] = None  # vx の推定標準誤差 (Kalman RTS)
    vy_std: Optional[np.ndarray] = None  # vy の推定標準誤差 (Kalman RTS)
    v_std: Optional[np.ndarray] = None   # スカラー速さの推定標準誤差
    method: str = ""                     # 使用した手法名
    params: Optional[dict] = None        # 使用したハイパーパラメータ


# ==============================================================================
# 1. カルマンフィルター / RTS (Rauch-Tung-Striebel) スムーザー
# ==============================================================================

class KalmanRTSSmoother:
    """
    2D 等速運動 (Constant Velocity) 状態空間モデルに基づく
    カルマンフィルターおよび Rauch-Tung-Striebel (RTS) 固定区間スムーザー。

    状態ベクトル:
        s_k = [x_k, vx_k, y_k, vy_k]^T

    状態方程式:
        s_k = F_k * s_{k-1} + w_k,   w_k ~ N(0, Q_k)

    観測方程式:
        z_k = H * s_k + v_k,         v_k ~ N(0, R)
        where H = [[1, 0, 0, 0],
                   [0, 0, 1, 0]]
    """

    def __init__(
        self,
        process_noise_std: float = 0.05,   # 加速度ゆらぎの標準偏差 q (μm/s^2)
        obs_noise_std: float = 0.05,       # 観測位置ノイズの標準偏差 sigma_obs (μm)
        init_pos_std: float = 1.0,         # 初期位置の不確実性
        init_vel_std: float = 1.0,         # 初期速度の不確実性
    ):
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std
        self.init_pos_std = init_pos_std
        self.init_vel_std = init_vel_std

    def _build_Q(self, dt: float, q_var: float) -> np.ndarray:
        """連続時間ホワイトノイズ加速度モデルの離散化プロセスノイズ共分散 Q"""
        q_1d = q_var * np.array([
            [(dt ** 3) / 3.0, (dt ** 2) / 2.0],
            [(dt ** 2) / 2.0, dt]
        ])
        Q = np.zeros((4, 4))
        Q[0:2, 0:2] = q_1d
        Q[2:4, 2:4] = q_1d
        return Q

    def _build_F(self, dt: float) -> np.ndarray:
        """状態遷移行列 F"""
        F = np.eye(4)
        F[0, 1] = dt
        F[2, 3] = dt
        return F

    def estimate(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        scale: float = 1.0,
    ) -> VelocityEstimateResult:
        """
        カルマンフィルター & RTS スムーザーを実行して平滑化軌跡と速度を算出。
        """
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float) * scale
        y = np.asarray(y, dtype=float) * scale
        N = len(t)

        if N < 2:
            raise ValueError("Kalman smoothing requires at least 2 data points.")

        dt_array = np.diff(t)
        if np.any(dt_array <= 0):
            raise ValueError("Time array must be strictly monotonically increasing.")

        # 観測行列 H (2 x 4)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

        # 観測ノイズ共分散 R (2 x 2)
        R = (self.obs_noise_std ** 2) * np.eye(2)
        q_var = self.process_noise_std ** 2

        # 1. 前向きカルマンフィルター (Forward Pass)
        x_filt = np.zeros((N, 4))        # x_{k|k}
        P_filt = np.zeros((N, 4, 4))     # P_{k|k}
        x_pred = np.zeros((N, 4))        # x_{k|k-1}
        P_pred = np.zeros((N, 4, 4))     # P_{k|k-1}
        F_list = []

        # 初期状態推定 (k=0)
        dt0 = dt_array[0]
        vx0 = (x[1] - x[0]) / dt0
        vy0 = (y[1] - y[0]) / dt0

        x_init = np.array([x[0], vx0, y[0], vy0])
        P_init = np.diag([
            self.init_pos_std ** 2,
            self.init_vel_std ** 2,
            self.init_pos_std ** 2,
            self.init_vel_std ** 2,
        ])

        # k=0 の観測更新
        z0 = np.array([x[0], y[0]])
        y0_res = z0 - H @ x_init
        S0 = H @ P_init @ H.T + R
        K0 = P_init @ H.T @ np.linalg.inv(S0)
        x_filt[0] = x_init + K0 @ y0_res
        P_filt[0] = (np.eye(4) - K0 @ H) @ P_init

        for k in range(1, N):
            dt_k = dt_array[k - 1]
            F_k = self._build_F(dt_k)
            Q_k = self._build_Q(dt_k, q_var)
            F_list.append(F_k)

            # 予測 (Predict)
            xp = F_k @ x_filt[k - 1]
            Pp = F_k @ P_filt[k - 1] @ F_k.T + Q_k
            x_pred[k] = xp
            P_pred[k] = Pp

            # 観測更新 (Update)
            zk = np.array([x[k], y[k]])
            yk_res = zk - H @ xp
            Sk = H @ Pp @ H.T + R
            Kk = Pp @ H.T @ np.linalg.inv(Sk)

            x_filt[k] = xp + Kk @ yk_res
            P_filt[k] = (np.eye(4) - Kk @ H) @ Pp

        # 2. 後向き RTS スムーザー (Backward RTS Pass)
        x_smooth_state = np.zeros((N, 4))
        P_smooth = np.zeros((N, 4, 4))

        # 終端状態
        x_smooth_state[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for k in range(N - 2, -1, -1):
            F_next = F_list[k]
            P_p_next = P_pred[k + 1]
            P_f_k = P_filt[k]

            # スムーザーゲイン C_k = P_{k|k} * F_{k+1}^T * P_{k+1|k}^(-1)
            C_k = P_f_k @ F_next.T @ np.linalg.inv(P_p_next)

            x_smooth_state[k] = x_filt[k] + C_k @ (x_smooth_state[k + 1] - x_pred[k + 1])
            P_smooth[k] = P_f_k + C_k @ (P_smooth[k + 1] - P_p_next) @ C_k.T

        # 結果の抽出
        x_smooth = x_smooth_state[:, 0]
        vx = x_smooth_state[:, 1]
        y_smooth = x_smooth_state[:, 2]
        vy = x_smooth_state[:, 3]

        vx_std = np.sqrt(np.maximum(P_smooth[:, 1, 1], 0.0))
        vy_std = np.sqrt(np.maximum(P_smooth[:, 3, 3], 0.0))

        v = np.sqrt(vx ** 2 + vy ** 2)
        theta = np.arctan2(vy, vx)

        with np.errstate(divide='ignore', invalid='ignore'):
            v_std = np.where(v > 1e-8, np.sqrt((vx * vx_std) ** 2 + (vy * vy_std) ** 2) / v, 0.5 * (vx_std + vy_std))

        params = {
            "process_noise_std": float(self.process_noise_std),
            "obs_noise_std": float(self.obs_noise_std),
        }

        return VelocityEstimateResult(
            t=t,
            x_smooth=x_smooth,
            y_smooth=y_smooth,
            vx=vx,
            vy=vy,
            v=v,
            theta=theta,
            vx_std=vx_std,
            vy_std=vy_std,
            v_std=v_std,
            method="Kalman Filter / RTS Smoother",
            params=params,
        )


# ==============================================================================
# 2. 平滑化スプライン (Smoothing Splines) + 解析的微分
# ==============================================================================

class SplineVelocityEstimator:
    """
    scipy.interpolate.UnivariateSpline を用いたスプライン平滑化と
    解析的微分 (derivative) による速度推定。

    平滑化パラメータ s:
        sum((y - g(x))^2) <= s
        デフォルトでは s = N * (noise_std ^ 2) に設定。
    """

    def __init__(
        self,
        smoothing_factor: Optional[float] = None,  # 明示的な s (None の場合は noise_std から自動計算)
        noise_std: float = 0.05,                   # 観測ノイズ標準偏差 [μm]
        k: int = 3,                                # スプラインの次数 (3: 3次スプライン)
    ):
        self.smoothing_factor = smoothing_factor
        self.noise_std = noise_std
        self.k = k

    def estimate(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> VelocityEstimateResult:
        """
        平滑化スプラインにより平滑化軌跡および解析的微分による速度を算出。
        """
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float) * scale
        y = np.asarray(y, dtype=float) * scale
        N = len(t)

        if t_eval is None:
            t_eval = t
        else:
            t_eval = np.asarray(t_eval, dtype=float)

        s_val = self.smoothing_factor
        if s_val is None:
            s_val = N * (self.noise_std ** 2)

        # x(t) と y(t) の UnivariateSpline フィッティング
        spline_k = min(self.k, N - 1)
        if spline_k < 1:
            raise ValueError("Spline smoothing requires at least 2 data points.")

        spl_x = UnivariateSpline(t, x, k=spline_k, s=s_val)
        spl_y = UnivariateSpline(t, y, k=spline_k, s=s_val)

        # 解析的1階微分関数
        dspl_x = spl_x.derivative(n=1)
        dspl_y = spl_y.derivative(n=1)

        # 評価
        x_smooth = spl_x(t_eval)
        y_smooth = spl_y(t_eval)
        vx = dspl_x(t_eval)
        vy = dspl_y(t_eval)

        v = np.sqrt(vx ** 2 + vy ** 2)
        theta = np.arctan2(vy, vx)

        params = {
            "smoothing_factor_s": float(s_val),
            "k_degree": int(spline_k),
            "noise_std": float(self.noise_std),
        }

        return VelocityEstimateResult(
            t=t_eval,
            x_smooth=x_smooth,
            y_smooth=y_smooth,
            vx=vx,
            vy=vy,
            v=v,
            theta=theta,
            method=f"Smoothing Spline (k={spline_k})",
            params=params,
        )


# ==============================================================================
# 3. ベースライン: 差分法 (Finite Difference)
# ==============================================================================

class FiniteDifferenceVelocityEstimator:
    """
    単純差分法（中心差分）または移動平均平滑化後の差分法。
    """

    def __init__(self, window: int = 1):
        self.window = window

    def estimate(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        scale: float = 1.0,
    ) -> VelocityEstimateResult:
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float) * scale
        y = np.asarray(y, dtype=float) * scale

        if self.window > 1:
            # 端点でもゼロパディングしない安全な移動平均平滑化 (center=True, min_periods=1)
            x_s = pd.Series(x).rolling(window=self.window, center=True, min_periods=1).mean().values
            y_s = pd.Series(y).rolling(window=self.window, center=True, min_periods=1).mean().values
        else:
            x_s = x.copy()
            y_s = y.copy()

        # numpy.gradient による中心差分（端点は片側差分）
        vx = np.gradient(x_s, t)
        vy = np.gradient(y_s, t)

        v = np.sqrt(vx ** 2 + vy ** 2)
        theta = np.arctan2(vy, vx)

        return VelocityEstimateResult(
            t=t,
            x_smooth=x_s,
            y_smooth=y_s,
            vx=vx,
            vy=vy,
            v=v,
            theta=theta,
            method=f"Finite Difference (window={self.window})",
            params={"window": self.window},
        )


# ==============================================================================
# 4. DataFrame に対する統合適用関数
# ==============================================================================

def estimate_particle_velocities(
    tracking_df: pd.DataFrame,
    method: str = "kalman_rts",
    frame_interval: float = 1.0,
    scale: float = 1.0,
    time_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
    particle_col: str = "particle",
    **kwargs,
) -> pd.DataFrame:
    """
    トラッキング DataFrame（各 particle ごと）に対して指定した手法で平滑化と速度推定を行う。

    Parameters:
    - tracking_df: トラッキング結果の DataFrame
    - method: 'kalman_rts' (カルマン/RTS), 'spline' (平滑化スプライン), 'diff' (差分法)
    - frame_interval: フレーム間時間 [s]
    - scale: ピクセルから物理単位 (μm) への変換係数 [μm/pixel]
    - time_col, x_col, y_col, particle_col: 対応する列名
    - kwargs: 各推定器クラスに渡すオプション引数

    Returns:
    - result_df: 元の DataFrame に平滑化位置 (x_smooth, y_smooth)、速度成分 (vx, vy)、
      スカラー速さ (v)、方向角度 (theta)、推定標準誤差 (v_std など) を追加した DataFrame
    """
    method_lower = method.lower()
    df_out = tracking_df.copy()

    # 各列の初期化
    df_out['t'] = df_out[time_col] * frame_interval
    df_out['x_smooth'] = np.nan
    df_out['y_smooth'] = np.nan
    df_out['vx'] = np.nan
    df_out['vy'] = np.nan
    df_out['v'] = np.nan
    df_out['theta'] = np.nan
    df_out['v_std'] = np.nan

    grouped = df_out.groupby(particle_col)

    for pid, group in grouped:
        idx = group.index
        if len(group) < 3:
            continue

        t_arr = group['t'].values
        x_arr = group[x_col].values
        y_arr = group[y_col].values

        if method_lower in ["kalman", "kalman_rts", "rts"]:
            estimator = KalmanRTSSmoother(**kwargs)
        elif method_lower in ["spline", "smoothing_spline"]:
            estimator = SplineVelocityEstimator(**kwargs)
        elif method_lower in ["diff", "finite_diff"]:
            estimator = FiniteDifferenceVelocityEstimator(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'kalman_rts', 'spline', 'diff'.")

        try:
            res = estimator.estimate(t_arr, x_arr, y_arr, scale=scale)
            df_out.loc[idx, 'x_smooth'] = res.x_smooth
            df_out.loc[idx, 'y_smooth'] = res.y_smooth
            df_out.loc[idx, 'vx'] = res.vx
            df_out.loc[idx, 'vy'] = res.vy
            df_out.loc[idx, 'v'] = res.v
            df_out.loc[idx, 'theta'] = res.theta
            if res.v_std is not None:
                df_out.loc[idx, 'v_std'] = res.v_std
        except Exception as e:
            warnings.warn(f"Failed to estimate velocity for particle {pid}: {e}")

    return df_out
