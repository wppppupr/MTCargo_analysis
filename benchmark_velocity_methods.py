#!/usr/bin/env python3
"""
benchmark_velocity_methods.py

貨物トラッキング軌跡に対する速度推定手法
1. カルマンフィルター / RTS スムーザー
2. 平滑化スプライン (Smoothing Splines) + 解析的微分
3. 単純中心差分 (Raw Diff)

の性能比較・可視化ベンチマークスクリプト。
"""

# ==============================================================================
# 実験撮影パラメータ設定
# ==============================================================================
FRAME_INTERVAL = 4.0   # フレーム間インターバル時間 [s] (4秒ごとに1枚撮影)
SCALE = 0.11           # ピクセルから物理単位への変換 [μm/pixel]
NOISE_PIXELS = 0.5     # トラッキング局在化誤差目安 [pixels] (約 0.055 μm)

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libs.trajectory_velocity import (
    KalmanRTSSmoother,
    SplineVelocityEstimator,
    FiniteDifferenceVelocityEstimator,
    estimate_particle_velocities,
)

OUTPUT_DIR = Path("/home/sasaki/MTCargo_analysis/figure/velocity_methods_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_cargo_trajectory(
    n_points: int = 120,
    dt: float = FRAME_INTERVAL,
    scale: float = SCALE,
    noise_pixels: float = NOISE_PIXELS,
    seed: int = 42,
):
    """
    ピクセル座標系でのカーゴ軌跡を生成し、真の物理速度 [μm/s] と観測ピクセル座標を出力。
    """
    np.random.seed(seed)
    t = np.arange(n_points) * dt

    # 物理空間での真の速度 [μm/s]（微小管上の典型的な速度: 0.1 ~ 0.8 μm/s）
    total_time = n_points * dt
    omega = 2.0 * np.pi / (total_time * 0.5)
    
    # 滑らかな速度変化
    vx_true = 0.35 + 0.15 * np.cos(omega * t) + 0.05 * np.sin(2 * omega * t)  # [μm/s]
    vy_true = 0.20 + 0.10 * np.sin(omega * t) - 0.04 * np.cos(3 * omega * t)  # [μm/s]
    v_true = np.sqrt(vx_true ** 2 + vy_true ** 2)

    # 積分して真の物理位置 [μm]
    x_true_um = np.cumsum(vx_true) * dt
    y_true_um = np.cumsum(vy_true) * dt

    # ピクセル座標系へ変換
    x_true_pix = x_true_um / scale
    y_true_pix = y_true_um / scale

    # 観測ノイズ（画像認識・重心抽出誤差）を付加 [pixels]
    x_obs_pix = x_true_pix + np.random.normal(0, noise_pixels, size=n_points)
    y_obs_pix = y_true_pix + np.random.normal(0, noise_pixels, size=n_points)

    return {
        "t": t,
        "x_true_um": x_true_um,
        "y_true_um": y_true_um,
        "vx_true": vx_true,
        "vy_true": vy_true,
        "v_true": v_true,
        "x_obs_pix": x_obs_pix,
        "y_obs_pix": y_obs_pix,
        "noise_um": noise_pixels * scale,
    }


def run_benchmark():
    print("=" * 70)
    print(f"貨物速度推定手法の比較ベンチマーク (dt={FRAME_INTERVAL}s, scale={SCALE}μm/pix)")
    print("=" * 70)

    # 1. テストデータの生成
    data = generate_synthetic_cargo_trajectory(
        n_points=100,
        dt=FRAME_INTERVAL,
        scale=SCALE,
        noise_pixels=NOISE_PIXELS,
        seed=123
    )
    t = data["t"]
    x_obs = data["x_obs_pix"]
    y_obs = data["y_obs_pix"]
    v_true = data["v_true"]
    vx_true = data["vx_true"]
    vy_true = data["vy_true"]
    obs_noise_um = data["noise_um"]  # μm 単位のノイズ

    methods = {}

    # 手法1: カルマンフィルター / RTS スムーザー
    print("1. カルマンフィルター / RTS スムーザー を実行中...")
    t0 = time.perf_counter()
    kalman_est = KalmanRTSSmoother(
        process_noise_std=0.05,     # 加速度ゆらぎ [μm/s^2]
        obs_noise_std=obs_noise_um  # 局在化誤差 [μm]
    )
    res_kalman = kalman_est.estimate(t, x_obs, y_obs, scale=SCALE)
    t_kalman = (time.perf_counter() - t0) * 1000
    methods["Kalman / RTS Smoother"] = (res_kalman, t_kalman)

    # 手法2: 平滑化スプライン (Smoothing Spline)
    print("2. 平滑化スプライン (Smoothing Spline) を実行中...")
    t0 = time.perf_counter()
    spline_est = SplineVelocityEstimator(noise_std=obs_noise_um, k=3)
    res_spline = spline_est.estimate(t, x_obs, y_obs, scale=SCALE)
    t_spline = (time.perf_counter() - t0) * 1000
    methods["Smoothing Spline (k=3)"] = (res_spline, t_spline)

    # 手法3: 単純差分 (Finite Difference)
    t0 = time.perf_counter()
    diff_est = FiniteDifferenceVelocityEstimator(window=1)
    res_diff = diff_est.estimate(t, x_obs, y_obs, scale=SCALE)
    t_diff = (time.perf_counter() - t0) * 1000
    methods["Raw Finite Difference"] = (res_diff, t_diff)

    # 定量評価テーブルの作成
    eval_records = []
    for name, (res, el_time) in methods.items():
        rmse_v = np.sqrt(np.mean((res.v - v_true) ** 2))
        mae_v = np.mean(np.abs(res.v - v_true))
        rmse_vx = np.sqrt(np.mean((res.vx - vx_true) ** 2))
        rmse_vy = np.sqrt(np.mean((res.vy - vy_true) ** 2))
        corr_v = np.corrcoef(res.v, v_true)[0, 1]

        # 位置の平滑化誤差
        rmse_pos = np.sqrt(np.mean((res.x_smooth - data["x_true_um"]) ** 2 + (res.y_smooth - data["y_true_um"]) ** 2))

        eval_records.append({
            "Method": name,
            "Speed RMSE [μm/s]": rmse_v,
            "Speed MAE [μm/s]": mae_v,
            "Vx RMSE [μm/s]": rmse_vx,
            "Vy RMSE [μm/s]": rmse_vy,
            "Pos RMSE [μm]": rmse_pos,
            "Correlation (r)": corr_v,
            "Time [ms]": el_time,
        })

    eval_df = pd.DataFrame(eval_records)
    print("\n--- 速度推定の定量評価結果 ---")
    print(eval_df.to_string(index=False))

    eval_df.to_csv(OUTPUT_DIR / "method_comparison_summary.csv", index=False)

    # 4. 可視化プロットの作成
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # (1) 2D 軌跡の比較
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data["x_true_um"], data["y_true_um"], 'k--', label='True Trajectory', lw=2.5, zorder=5)
    ax1.scatter(x_obs * SCALE, y_obs * SCALE, color='gray', alpha=0.5, s=20, label='Observed (Noisy)', zorder=2)
    ax1.plot(res_kalman.x_smooth, res_kalman.y_smooth, label='Kalman/RTS', color='#2ca02c', lw=2)
    ax1.plot(res_spline.x_smooth, res_spline.y_smooth, label='Spline', color='#ff7f0e', lw=2, linestyle=':')
    ax1.set_title("A. 2D Trajectory Smoothing", fontsize=13, fontweight='bold')
    ax1.set_xlabel("X Position [μm]", fontsize=11)
    ax1.set_ylabel("Y Position [μm]", fontsize=11)
    ax1.legend(loc='best', frameon=True)

    # (2) スカラー速度 (Speed v) の時間変化比較
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, v_true, 'k--', label='True Speed', lw=2.5, zorder=5)
    ax2.plot(t, res_diff.v, color='lightgray', label='Raw Difference', lw=1, alpha=0.7)
    ax2.plot(t, res_kalman.v, label=f'Kalman/RTS (RMSE={eval_df.loc[0, "Speed RMSE [μm/s]"]:.4f})', color='#2ca02c', lw=2)
    ax2.plot(t, res_spline.v, label=f'Spline (RMSE={eval_df.loc[1, "Speed RMSE [μm/s]"]:.4f})', color='#ff7f0e', lw=2, linestyle=':')
    ax2.set_title("B. Speed $v(t) = \\sqrt{v_x^2 + v_y^2}$ Comparison", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Time [s]", fontsize=11)
    ax2.set_ylabel("Speed [μm/s]", fontsize=11)
    ax2.legend(loc='best', frameon=True, fontsize=9)

    # (3) x 方向速度 vx(t) と不確実性 (Kalman)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, vx_true, 'k--', label='True $v_x(t)$', lw=2)
    ax3.plot(t, res_diff.vx, color='lightgray', label='Raw Diff', lw=0.8, alpha=0.6)
    ax3.plot(t, res_kalman.vx, label='Kalman/RTS $v_x$', color='#2ca02c', lw=1.8)
    if res_kalman.vx_std is not None:
        ax3.fill_between(t, res_kalman.vx - 1.96 * res_kalman.vx_std, res_kalman.vx + 1.96 * res_kalman.vx_std, color='#2ca02c', alpha=0.15, label='Kalman 95% CI')
    ax3.plot(t, res_spline.vx, label='Spline $v_x$', color='#ff7f0e', lw=1.5, linestyle=':')
    ax3.set_title("C. Velocity Component $v_x(t)$", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Time [s]", fontsize=11)
    ax3.set_ylabel("$v_x$ [μm/s]", fontsize=11)
    ax3.legend(loc='best', frameon=True, fontsize=9)

    # (4) y 方向速度 vy(t)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, vy_true, 'k--', label='True $v_y(t)$', lw=2)
    ax4.plot(t, res_diff.vy, color='lightgray', label='Raw Diff', lw=0.8, alpha=0.6)
    ax4.plot(t, res_kalman.vy, label='Kalman/RTS $v_y$', color='#2ca02c', lw=1.8)
    if res_kalman.vy_std is not None:
        ax4.fill_between(t, res_kalman.vy - 1.96 * res_kalman.vy_std, res_kalman.vy + 1.96 * res_kalman.vy_std, color='#2ca02c', alpha=0.15, label='Kalman 95% CI')
    ax4.plot(t, res_spline.vy, label='Spline $v_y$', color='#ff7f0e', lw=1.5, linestyle=':')
    ax4.set_title("D. Velocity Component $v_y(t)$", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Time [s]", fontsize=11)
    ax4.set_ylabel("$v_y$ [μm/s]", fontsize=11)
    ax4.legend(loc='best', frameon=True, fontsize=9)

    # (5) 速度推定誤差の分布 (Boxplot)
    ax5 = fig.add_subplot(gs[2, 0])
    errors = [
        np.abs(res_kalman.v - v_true),
        np.abs(res_spline.v - v_true),
        np.abs(res_diff.v - v_true),
    ]
    box_labels = ["Kalman/RTS", "Spline", "Raw Diff"]
    colors = ['#2ca02c', '#ff7f0e', 'gray']
    bplot = ax5.boxplot(errors, tick_labels=box_labels, patch_artist=True)
    for patch, c in zip(bplot['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax5.set_title("E. Absolute Speed Estimation Error $|v_{est} - v_{true}|$", fontsize=13, fontweight='bold')
    ax5.set_ylabel("Absolute Error [μm/s]", fontsize=11)
    ax5.set_yscale('log')

    # (6) 進行方向角度 theta(t) (Raw Diff も含めて表示)
    ax6 = fig.add_subplot(gs[2, 1])
    theta_true = np.arctan2(vy_true, vx_true)
    raw_theta_unwrapped = np.unwrap(np.where(np.isnan(res_diff.theta), 0.0, res_diff.theta))
    ax6.plot(t, np.unwrap(theta_true), 'k--', label='True Angle', lw=2.0, zorder=5)
    ax6.plot(t, raw_theta_unwrapped, color='lightgray', label='Raw Diff Angle', lw=1.0, alpha=0.7)
    ax6.plot(t, np.unwrap(res_kalman.theta), label='Kalman Angle', color='#2ca02c', lw=1.8)
    ax6.plot(t, np.unwrap(res_spline.theta), label='Spline Angle', color='#ff7f0e', lw=1.5, linestyle=':')
    ax6.set_title("F. Direction Angle $\\theta(t)$ (Unwrapped [rad])", fontsize=13, fontweight='bold')
    ax6.set_xlabel("Time [s]", fontsize=11)
    ax6.set_ylabel("Angle [rad]", fontsize=11)
    ax6.legend(loc='best', frameon=True, fontsize=9)

    plt.suptitle("Benchmark: Velocity Estimation Methods for Cargo Tracking", fontsize=16, fontweight='bold')
    fig_path = OUTPUT_DIR / "velocity_methods_benchmark.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n比較グラフを保存しました: {fig_path}")


if __name__ == "__main__":
    run_benchmark()
