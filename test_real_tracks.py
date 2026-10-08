#!/usr/bin/env python3
"""
test_real_tracks.py

実際の実験トラッキングデータ (beads_tracks.csv) を用いて
1. カルマンフィルター / RTS スムーザー
2. 平滑化スプライン (Smoothing Splines)
3. 単純差分 (Raw Diff)
を実行・比較可視化するスクリプト。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libs.trajectory_velocity import (
    KalmanRTSSmoother,
    SplineVelocityEstimator,
    FiniteDifferenceVelocityEstimator,
    estimate_particle_velocities,
)
from libs.cal_vel import cal_advanced

# デフォルトパラメータ
DEFAULT_CSV_PATH = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um003/beads_tracks.csv")
DEFAULT_FRAME_INTERVAL = 4.0   # [s]
DEFAULT_SCALE = 0.11           # [μm/pixel]

OUTPUT_DIR = Path("/home/sasaki/MTCargo_analysis/figure/velocity_real_tracks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tracking_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        fallback = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um002/beads_tracks.csv")
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Tracking file not found: {csv_path}")

    print(f"Loading real tracking data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def analyze_real_particle(
    df: pd.DataFrame,
    particle_id: Optional[int] = None,
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    scale: float = DEFAULT_SCALE,
    obs_noise_pixels: float = 0.5,
):
    counts = df.groupby('particle').size().sort_values(ascending=False)
    print("\nParticle track lengths (top 5):")
    print(counts.head(5))

    if particle_id is None:
        particle_id = counts.index[0]
        print(f"\nAuto-selected longest track: Particle {particle_id} ({counts[particle_id]} frames)")
    else:
        print(f"\nAnalyzing user-selected: Particle {particle_id} ({counts.get(particle_id, 0)} frames)")

    p_df = df[df['particle'] == particle_id].sort_values('frame').copy()
    
    t = p_df['frame'].values * frame_interval
    x_pix = p_df['x'].values
    y_pix = p_df['y'].values
    obs_noise_um = obs_noise_pixels * scale

    print(f"Time span: {t[0]:.1f}s to {t[-1]:.1f}s (Total {len(t)} points, duration {(t[-1]-t[0])/60:.1f} min)")

    methods = {}

    # 1. Kalman / RTS Smoother
    print("\n1. カルマンフィルター / RTS スムーザー を実行中...")
    t0 = time.perf_counter()
    kalman = KalmanRTSSmoother(
        process_noise_std=0.05,     # 加速度変動 [μm/s^2]
        obs_noise_std=obs_noise_um  # 観測局在化ノイズ [μm]
    )
    res_kalman = kalman.estimate(t, x_pix, y_pix, scale=scale)
    t_kalman = (time.perf_counter() - t0) * 1000
    methods["Kalman / RTS Smoother"] = (res_kalman, t_kalman)

    # 2. Smoothing Spline
    print("2. 平滑化スプライン (Smoothing Splines) を実行中...")
    t0 = time.perf_counter()
    spline = SplineVelocityEstimator(noise_std=obs_noise_um, k=3)
    res_spline = spline.estimate(t, x_pix, y_pix, scale=scale)
    t_spline = (time.perf_counter() - t0) * 1000
    methods["Smoothing Spline (k=3)"] = (res_spline, t_spline)

    # 3. Raw Finite Difference
    t0 = time.perf_counter()
    diff_est = FiniteDifferenceVelocityEstimator(window=1)
    res_diff = diff_est.estimate(t, x_pix, y_pix, scale=scale)
    t_diff = (time.perf_counter() - t0) * 1000
    methods["Raw Finite Difference"] = (res_diff, t_diff)

    # サマリー統計の出力
    stats_records = []
    for name, (res, el_time) in methods.items():
        stats_records.append({
            "Method": name,
            "Mean Speed <v> [μm/s]": np.nanmean(res.v),
            "Median Speed [μm/s]": np.nanmedian(res.v),
            "Std Speed [μm/s]": np.nanstd(res.v),
            "Max Speed [μm/s]": np.nanmax(res.v),
            "Mean Vx [μm/s]": np.nanmean(res.vx),
            "Mean Vy [μm/s]": np.nanmean(res.vy),
            "Time [ms]": el_time,
        })

    stats_df = pd.DataFrame(stats_records)
    print(f"\n--- 実データ (Particle {particle_id}) の速度推定統計サマリー ---")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(OUTPUT_DIR / f"real_particle_{particle_id}_stats.csv", index=False)

    # 可視化プロットの作成
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.25)

    x_obs_um = x_pix * scale
    y_obs_um = y_pix * scale

    # (1) 2D 軌跡の平滑化結果
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x_obs_um, y_obs_um, color='gray', alpha=0.4, s=15, label='Raw Track (Noisy)', zorder=2)
    ax1.plot(res_kalman.x_smooth, res_kalman.y_smooth, label='Kalman/RTS Smooth', color='#2ca02c', lw=2)
    ax1.plot(res_spline.x_smooth, res_spline.y_smooth, label='Spline Smooth', color='#ff7f0e', lw=1.8, linestyle=':')
    ax1.set_title(f"A. 2D Trajectory (Particle {particle_id})", fontsize=13, fontweight='bold')
    ax1.set_xlabel("X Position [μm]", fontsize=11)
    ax1.set_ylabel("Y Position [μm]", fontsize=11)
    ax1.legend(loc='best', frameon=True)

    # (2) スカラー速さ v(t) の比較
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, res_diff.v, color='lightgray', label='Raw Diff (Noisy)', lw=1, alpha=0.7)
    ax2.plot(t, res_kalman.v, label=f'Kalman/RTS (mean={np.nanmean(res_kalman.v):.3f} μm/s)', color='#2ca02c', lw=1.8)
    ax2.plot(t, res_spline.v, label=f'Spline (mean={np.nanmean(res_spline.v):.3f} μm/s)', color='#ff7f0e', lw=1.8, linestyle=':')
    ax2.set_title("B. Instantaneous Speed $v(t) = \\sqrt{v_x^2 + v_y^2}$", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Time [s]", fontsize=11)
    ax2.set_ylabel("Speed [μm/s]", fontsize=11)
    ax2.legend(loc='best', frameon=True, fontsize=9)

    # (3) x 方向速度 vx(t)
    ax3 = fig.add_subplot(gs[1, 0])
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
    ax4.plot(t, res_diff.vy, color='lightgray', label='Raw Diff', lw=0.8, alpha=0.6)
    ax4.plot(t, res_kalman.vy, label='Kalman/RTS $v_y$', color='#2ca02c', lw=1.8)
    if res_kalman.vy_std is not None:
        ax4.fill_between(t, res_kalman.vy - 1.96 * res_kalman.vy_std, res_kalman.vy + 1.96 * res_kalman.vy_std, color='#2ca02c', alpha=0.15, label='Kalman 95% CI')
    ax4.plot(t, res_spline.vy, label='Spline $v_y$', color='#ff7f0e', lw=1.5, linestyle=':')
    ax4.set_title("D. Velocity Component $v_y(t)$", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Time [s]", fontsize=11)
    ax4.set_ylabel("$v_y$ [μm/s]", fontsize=11)
    ax4.legend(loc='best', frameon=True, fontsize=9)

    # (5) 速度の確率密度分布 (Histogram / KDE)
    ax5 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0, max(np.percentile(res_diff.v, 99), 0.5), 35)
    ax5.hist(res_diff.v, bins=bins, density=True, alpha=0.3, color='gray', label='Raw Diff')
    ax5.hist(res_kalman.v, bins=bins, density=True, alpha=0.5, color='#2ca02c', label='Kalman/RTS')
    ax5.hist(res_spline.v, bins=bins, density=True, alpha=0.5, color='#ff7f0e', label='Spline')
    ax5.set_title("E. Speed Distribution $P(v)$", fontsize=13, fontweight='bold')
    ax5.set_xlabel("Speed $v$ [μm/s]", fontsize=11)
    ax5.set_ylabel("Probability Density", fontsize=11)
    ax5.set_yscale('log')
    ax5.legend(loc='best', frameon=True, fontsize=9)

    # (6) 進行方向角度 theta(t) (Raw Diff も含めて表示)
    ax6 = fig.add_subplot(gs[2, 1])
    # Raw Diff の theta は角度の急変を unwrap
    raw_theta_unwrapped = np.unwrap(np.where(np.isnan(res_diff.theta), 0.0, res_diff.theta))
    ax6.plot(t, raw_theta_unwrapped, color='lightgray', label='Raw Diff Angle', lw=1.0, alpha=0.7)
    ax6.plot(t, np.unwrap(res_kalman.theta), label='Kalman Angle', color='#2ca02c', lw=1.8)
    ax6.plot(t, np.unwrap(res_spline.theta), label='Spline Angle', color='#ff7f0e', lw=1.5, linestyle=':')
    ax6.set_title("F. Direction Angle $\\theta(t)$ (Unwrapped [rad])", fontsize=13, fontweight='bold')
    ax6.set_xlabel("Time [s]", fontsize=11)
    ax6.set_ylabel("Angle [rad]", fontsize=11)
    ax6.legend(loc='best', frameon=True, fontsize=9)

    plt.suptitle(f"Real Tracking Data Velocity Estimation: Particle {particle_id} (dt={frame_interval}s, scale={scale}μm/pix)", fontsize=15, fontweight='bold')
    out_fig = OUTPUT_DIR / f"real_particle_{particle_id}_velocity_comparison.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n比較プロットを保存しました: {out_fig}")

    return stats_df


def main():
    parser = argparse.ArgumentParser(description="Test velocity estimation methods on real tracking CSV.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH), help="Path to beads_tracks.csv")
    parser.add_argument("--particle", type=int, default=None, help="Particle ID to analyze (default: longest track)")
    parser.add_argument("--dt", type=float, default=DEFAULT_FRAME_INTERVAL, help="Frame interval in seconds (default: 4.0)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Scale in um/pixel (default: 0.11)")
    args = parser.parse_args()

    df = load_tracking_data(Path(args.csv))
    analyze_real_particle(
        df=df,
        particle_id=args.particle,
        frame_interval=args.dt,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
