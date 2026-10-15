#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cargo_alignment_and_turning_fluctuation.py
================================================

貨物微粒子（蛍光ビーズ）の局所アライメントおよび軌道曲率（方向転換角の揺らぎ）と
微小管領域のイジング磁化 M との相関を解析・プロットするスクリプト。

【プロット案 1: 条件付きアライメント <cos phi | M> vs M】
- phi \equiv theta_cargo - theta_MT
- 各フレームで、カーゴの変位ベクトル Delta r_cargo (または速度ベクトル v_cargo) と、
  その位置における微小管流速ベクトル v_MT のなす角 phi を計算。
- M のビンごとに <cos phi> を平均してプロット。
  微小管が整列するほど ( |M| -> 1 )、カーゴが微小管流と強くアライメントするかを検証。

【プロット案 2: 方向転換角の揺らぎ <(Delta theta)^2 | M> vs M】
- 微小管流の方向を使わず、カーゴ自身の単位時間あたりの進行方向変化
  Delta theta(t) = wrap_to_pi( theta(t+Delta t) - theta(t) )
  およびその 2 階差分 ddot{theta}(t) = wrap_to_pi( Delta theta(t+Delta t) - Delta theta(t) )
  の二乗平均を M の関数としてプロット。
- 瞬時転換角 (Delta t = 4 s: 連続フレーム間方向変化) および
  ストライド転換角 (Delta t = 20 s: 5フレーム間方向変化) の両方を解析。
- 「微小管が整列するほど、カーゴの軌道曲率（ふらつき）が抑えられるか？」を直接テスト。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイルの適用
style_path = CURRENT_DIR / "libs" / "my_style.mplstyle"
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    except Exception:
        style_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
else:
    style_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ビーズ基本情報 (MSD.py 準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "label": r"$2R_c = 0.63\,\mu\mathrm{m}$", "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "label": r"$2R_c = 1.18\,\mu\mathrm{m}$", "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "label": r"$2R_c = 3.37\,\mu\mathrm{m}$", "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "label": r"$2R_c = 5.00\,\mu\mathrm{m}$", "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "label": r"$2R_c = 7.24\,\mu\mathrm{m}$", "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "label": r"$2R_c = 20.0\,\mu\mathrm{m}$", "marker": "s", "color": style_colors[5]},
]
BEAD_LOOKUP = {b["name"]: b for b in BEADS_INFO}

POSSIBLE_ROOTS = [
    Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads"),
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    Path("/mnt/NAS-Ebanaru/sasaki/MTSingleBeads"),
    Path("/Volumes/data-1/Sasaki/MTsingleBeads"),
    Path("/Volumes/data/Sasaki/MTsingleBeads"),
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def apply_custom_style():
    style_path = CURRENT_DIR / "libs" / "my_style.mplstyle"
    if style_path.exists():
        try:
            plt.style.use(str(style_path))
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["mathtext.fontset"] = "cm"


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """角度を [-pi, pi) の範囲に折り畳む"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300):
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / f"{basename}.png"
        svg_path = d / f"{basename}.svg"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {basename}.png / .svg -> {len(out_dirs)} dir(s)")


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]):
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / f"{basename}.csv"
        df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {basename}.csv -> {len(out_dirs)} dir(s)")


# =============================================================================
# データ読み込み & 微小管流速ベクトル & 角度変化の計算
# =============================================================================

def load_and_augment_data(
    points_csv: Path,
    data_root: Path,
    output_dir: Path,
    force_recompute: bool = False,
    scale: float = 0.11,
    dt: float = 4.0,
    pixel_stride: int = 4,
) -> pd.DataFrame:
    """
    cargo_spin_velocity_points.csv を読み込み、
    各点の微小管流速ベクトル (flow_vx, flow_vy) と
    カーゴ進行方向変化 Delta theta (4s & 20s), ddot{theta} を付加する。
    """
    augmented_csv = output_dir / "cargo_alignment_turning_points.csv"
    if augmented_csv.exists() and not force_recompute:
        df_aug = pd.read_csv(augmented_csv)
        if "delta_theta_4s_sq" in df_aug.columns and "flow_vx_um_s" in df_aug.columns:
            print(f"Loading cached augmented data from {augmented_csv}...")
            print(f"Loaded {len(df_aug)} augmented records.")
            return df_aug

    if not points_csv.exists():
        raise FileNotFoundError(f"Points CSV not found: {points_csv}")

    print(f"Loading base points CSV from {points_csv}...")
    df = pd.read_csv(points_csv)
    print(f"Loaded {len(df)} base points.")

    # 1. 微小管流速ベクトル (flow_vx, flow_vy) の抽出
    has_flow_vec = "flow_vx_um_s" in df.columns and "flow_vy_um_s" in df.columns and df["flow_vx_um_s"].notna().any()

    if not has_flow_vec:
        print("Extracting local microtubule flow vectors (flow_vx, flow_vy) from flow caches...")
        t0 = time.time()
        flow_vx_list = np.full(len(df), np.nan, dtype=np.float32)
        flow_vy_list = np.full(len(df), np.nan, dtype=np.float32)

        for exp_name, grp in df.groupby("exp_dir"):
            bead_name = grp["bead_name"].iloc[0]
            cache_candidates = list((data_root / bead_name).glob(f"*/{exp_name}/mt_flow_cache_s4_f5.h5"))
            if not cache_candidates:
                cache_candidates = list((data_root / bead_name).glob(f"{exp_name}/mt_flow_cache_s4_f5.h5"))

            if not cache_candidates or not cache_candidates[0].exists():
                print(f"  [WARNING] Cache not found for {exp_name} ({bead_name})")
                continue

            cache_path = cache_candidates[0]
            with h5py.File(cache_path, "r") as f:
                frame_ids = list(f.attrs.get("frame_ids", []))
                frame_to_idx = {fid: idx for idx, fid in enumerate(frame_ids)}
                H, W = f["mx"].shape[1], f["mx"].shape[2]

                for frame_val, frame_grp in grp.groupby("frame"):
                    if frame_val not in frame_to_idx:
                        continue
                    f_idx = frame_to_idx[frame_val]
                    mx = np.asarray(f["mx"][f_idx], dtype=np.float32)
                    my = np.asarray(f["my"][f_idx], dtype=np.float32)

                    for idx_row, row in frame_grp.iterrows():
                        cx = (row["x_um"] / scale) / pixel_stride
                        cy = (row["y_um"] / scale) / pixel_stride
                        r_grid = (row["region_radius_um"] / scale) / pixel_stride

                        y0 = max(int(np.floor(cy - r_grid)), 0)
                        y1 = min(int(np.ceil(cy + r_grid)) + 1, H)
                        x0 = max(int(np.floor(cx - r_grid)), 0)
                        x1 = min(int(np.ceil(cx + r_grid)) + 1, W)

                        if y1 <= y0 or x1 <= x0:
                            continue

                        yy, xx = np.ogrid[y0:y1, x0:x1]
                        mask = ((yy - cy)**2 + (xx - cx)**2) <= r_grid**2

                        sub_mx = mx[y0:y1, x0:x1][mask]
                        sub_my = my[y0:y1, x0:x1][mask]

                        valid = (sub_mx**2 + sub_my**2) > 0.0001**2
                        if np.count_nonzero(valid) > 0:
                            ux = float(np.mean(sub_mx[valid])) * scale / dt
                            uy = float(np.mean(sub_my[valid])) * scale / dt
                            flow_vx_list[idx_row] = ux
                            flow_vy_list[idx_row] = uy

        df["flow_vx_um_s"] = flow_vx_list
        df["flow_vy_um_s"] = flow_vy_list
        print(f"Extracted flow vectors in {time.time() - t0:.2f} s. Valid vectors: {np.isfinite(df['flow_vx_um_s']).sum()}/{len(df)}")

    # 2. プロット案 1 用の物理量計算:
    v_cargo_mag = np.hypot(df["vx_um_s"], df["vy_um_s"])
    v_flow_mag = np.hypot(df["flow_vx_um_s"], df["flow_vy_um_s"])
    df["v_cargo_mag_um_s"] = v_cargo_mag
    df["v_mt_mag_um_s"] = v_flow_mag

    theta_cargo = np.arctan2(df["vy_um_s"], df["vx_um_s"])
    theta_mt = np.arctan2(df["flow_vy_um_s"], df["flow_vx_um_s"])
    df["theta_cargo_rad"] = theta_cargo
    df["theta_mt_rad"] = theta_mt

    # なす角 phi = theta_cargo - theta_mt
    phi = wrap_to_pi(theta_cargo - theta_mt)
    df["phi_rad"] = phi
    df["phi_deg"] = np.degrees(phi)

    # cos phi
    dot_prod = df["vx_um_s"] * df["flow_vx_um_s"] + df["vy_um_s"] * df["flow_vy_um_s"]
    denom = v_cargo_mag * v_flow_mag
    cos_phi = np.where(denom > 0, dot_prod / denom, np.nan)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    df["cos_phi"] = cos_phi

    if "abs_m" not in df.columns:
        df["abs_m"] = df["m_ising"].abs()

    # 3. プロット案 2 用: 瞬時方向転換角 (Delta t = 4 s, 連続フレーム間) の計算
    print("Extracting instantaneous turning angle (Delta t = 4 s) from beads_tracks.csv...")
    tracks_turning_list = []
    for exp_name, grp in df.groupby("exp_dir"):
        bead_name = grp["bead_name"].iloc[0]
        track_candidates = list((data_root / bead_name).glob(f"*/{exp_name}/beads_tracks.csv"))
        if not track_candidates:
            track_candidates = list((data_root / bead_name).glob(f"{exp_name}/beads_tracks.csv"))
        if not track_candidates or not track_candidates[0].exists():
            continue

        df_tr = pd.read_csv(track_candidates[0]).sort_values(["particle", "frame"]).reset_index(drop=True)
        # 前後フレームの変位
        df_tr["dx_prev"] = df_tr.groupby("particle")["x"].diff(1)
        df_tr["dy_prev"] = df_tr.groupby("particle")["y"].diff(1)
        df_tr["frame_prev"] = df_tr.groupby("particle")["frame"].shift(1)

        df_tr["dx_next"] = df_tr.groupby("particle")["x"].diff(-1) * (-1)
        df_tr["dy_next"] = df_tr.groupby("particle")["y"].diff(-1) * (-1)
        df_tr["frame_next"] = df_tr.groupby("particle")["frame"].shift(-1)

        # 連続フレーム (frame - 1, frame, frame + 1)
        valid_step = (df_tr["frame"] - df_tr["frame_prev"] == 1) & (df_tr["frame_next"] - df_tr["frame"] == 1)
        th_prev = np.arctan2(df_tr["dy_prev"], df_tr["dx_prev"])
        th_next = np.arctan2(df_tr["dy_next"], df_tr["dx_next"])
        df_tr["turning_4s"] = np.where(valid_step, wrap_to_pi(th_next - th_prev), np.nan)

        # 2階差分 (4s)
        df_tr["turning_4s_next"] = df_tr.groupby("particle")["turning_4s"].shift(-1)
        df_tr["ddot_4s"] = np.where(
            df_tr["turning_4s"].notna() & df_tr["turning_4s_next"].notna(),
            wrap_to_pi(df_tr["turning_4s_next"] - df_tr["turning_4s"]),
            np.nan
        )

        df_tr["exp_dir"] = exp_name
        tracks_turning_list.append(df_tr[["exp_dir", "particle", "frame", "turning_4s", "ddot_4s"]])

    if tracks_turning_list:
        df_tracks_turning = pd.concat(tracks_turning_list, ignore_index=True)
        df = pd.merge(df, df_tracks_turning, on=["exp_dir", "particle", "frame"], how="left")
        df["delta_theta_4s_rad"] = df["turning_4s"]
        df["delta_theta_4s_deg"] = np.degrees(df["turning_4s"])
        df["delta_theta_4s_sq"] = df["turning_4s"]**2
        df["circ_disp_4s"] = 1.0 - np.cos(df["turning_4s"])
        df["ddot_theta_4s_rad"] = df["ddot_4s"]
        df["ddot_theta_4s_sq"] = df["ddot_4s"]**2
        df = df.drop(columns=["turning_4s", "ddot_4s"])
    else:
        df["delta_theta_4s_rad"] = np.nan
        df["delta_theta_4s_sq"] = np.nan
        df["circ_disp_4s"] = np.nan
        df["ddot_theta_4s_rad"] = np.nan
        df["ddot_theta_4s_sq"] = np.nan

    # 4. プロット案 2 用: 5フレーム間隔 (Delta t = 20 s) での進行方向変化
    print("Computing turning angle Delta theta and ddot{theta} at stride Delta t = 20 s...")
    df = df.sort_values(["exp_dir", "particle", "frame"]).reset_index(drop=True)

    df["next_frame"] = df.groupby(["exp_dir", "particle"])["frame"].shift(-1)
    df["next_theta"] = df.groupby(["exp_dir", "particle"])["theta_cargo_rad"].shift(-1)
    df["frame_diff"] = df["next_frame"] - df["frame"]

    is_consec = (df["frame_diff"] == 5)
    delta_theta_20s = np.where(is_consec, wrap_to_pi(df["next_theta"] - df["theta_cargo_rad"]), np.nan)
    df["delta_theta_20s_rad"] = delta_theta_20s
    df["delta_theta_20s_deg"] = np.degrees(delta_theta_20s)
    df["delta_theta_20s_sq"] = delta_theta_20s**2

    df["next_delta_theta"] = df.groupby(["exp_dir", "particle"])["delta_theta_20s_rad"].shift(-1)
    df["next_frame_diff"] = df.groupby(["exp_dir", "particle"])["frame_diff"].shift(-1)
    is_consec2 = is_consec & (df["next_frame_diff"] == 5)
    ddot_theta_20s = np.where(is_consec2, wrap_to_pi(df["next_delta_theta"] - df["delta_theta_20s_rad"]), np.nan)
    df["ddot_theta_20s_rad"] = ddot_theta_20s
    df["ddot_theta_20s_deg"] = np.degrees(ddot_theta_20s)
    df["ddot_theta_20s_sq"] = ddot_theta_20s**2
    df["circ_disp_20s"] = 1.0 - np.cos(delta_theta_20s)

    cols_to_drop = ["next_frame", "next_theta", "frame_diff", "next_delta_theta", "next_frame_diff"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # キャッシュ保存
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(augmented_csv, index=False)
    print(f"Saved augmented dataset to {augmented_csv}")

    return df


# =============================================================================
# ビン集計ユーティリティ
# =============================================================================

def compute_binned_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: np.ndarray,
    min_count: int = 5,
) -> pd.DataFrame:
    """指定されたビン境界で平均、SEM、中央値、IQR、サンプル数を集計"""
    valid = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    sub = df[valid].copy()
    sub["bin_idx"] = pd.cut(sub[x_col], bins=bins, include_lowest=True, labels=False)

    records = []
    n_bins = len(bins) - 1
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        bsub = sub[sub["bin_idx"] == i]
        n = len(bsub)
        if n >= min_count:
            vals = bsub[y_col].to_numpy()
            mean_val = float(np.mean(vals))
            sem_val = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            med_val = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            x_mean = float(bsub[x_col].mean())
            x_med = float(bsub[x_col].median())
        else:
            mean_val = sem_val = std_val = med_val = q25 = q75 = np.nan
            x_mean = (low + high) / 2.0
            x_med = (low + high) / 2.0

        records.append({
            "bin_idx": i,
            "x_low": low,
            "x_high": high,
            "x_center": (low + high) / 2.0,
            "x_mean": x_mean,
            "x_median": x_med,
            "count": n,
            "mean": mean_val,
            "sem": sem_val,
            "std": std_val,
            "median": med_val,
            "q25": q25,
            "q75": q75,
        })
    return pd.DataFrame(records)


# =============================================================================
# プロット案 1: 条件付きアライメント <cos phi | M> vs M
# =============================================================================

def generate_alignment_plots(
    df: pd.DataFrame,
    out_dirs: List[Path],
    n_bins_abs: int = 8,
    n_bins_signed: int = 10,
    min_count: int = 5,
):
    print("\n" + "=" * 70)
    print("Generating Plot 1: Conditional Alignment <cos phi | M> vs M")
    print("=" * 70)

    valid_mask = np.isfinite(df["cos_phi"]) & np.isfinite(df["abs_m"]) & (df["v_cargo_mag_um_s"] > 0.01)
    df_valid = df[valid_mask].copy()
    print(f"Valid points for alignment analysis: {len(df_valid)} / {len(df)}")

    bins_abs = np.linspace(0.0, 1.0, n_bins_abs + 1)
    bins_signed = np.linspace(-1.0, 1.0, n_bins_signed + 1)

    pool_stats_abs = compute_binned_stats(df_valid, "abs_m", "cos_phi", bins_abs, min_count=min_count)
    pool_stats_signed = compute_binned_stats(df_valid, "m_ising", "cos_phi", bins_signed, min_count=min_count)

    bead_stats_abs: Dict[str, pd.DataFrame] = {}
    bead_stats_signed: Dict[str, pd.DataFrame] = {}
    for bead in BEADS_INFO:
        bname = bead["name"]
        bsub = df_valid[df_valid["bead_name"] == bname]
        bead_stats_abs[bname] = compute_binned_stats(bsub, "abs_m", "cos_phi", bins_abs, min_count=min_count)
        bead_stats_signed[bname] = compute_binned_stats(bsub, "m_ising", "cos_phi", bins_signed, min_count=min_count)

    summary_rows = []
    for _, row in pool_stats_abs.iterrows():
        r = row.to_dict()
        r["condition"] = "Pooled (All)"
        r["mode"] = "abs_m"
        summary_rows.append(r)
    for bname, bdf in bead_stats_abs.items():
        for _, row in bdf.iterrows():
            r = row.to_dict()
            r["condition"] = bname
            r["mode"] = "abs_m"
            summary_rows.append(r)
    for _, row in pool_stats_signed.iterrows():
        r = row.to_dict()
        r["condition"] = "Pooled (All)"
        r["mode"] = "signed_m"
        summary_rows.append(r)
    for bname, bdf in bead_stats_signed.items():
        for _, row in bdf.iterrows():
            r = row.to_dict()
            r["condition"] = bname
            r["mode"] = "signed_m"
            summary_rows.append(r)

    df_summary = pd.DataFrame(summary_rows)
    save_csv_to_all(df_summary, "conditional_alignment_summary", out_dirs)

    # 1-A: <cos phi | |M|> vs |M|
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for bead in BEADS_INFO:
        bname = bead["name"]
        bdf = bead_stats_abs[bname]
        valid_b = bdf["mean"].notna()
        if valid_b.sum() > 0:
            x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
            y_vals = bdf.loc[valid_b, "mean"].to_numpy()
            err_vals = bdf.loc[valid_b, "sem"].to_numpy()
            ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.12, edgecolor="none")
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=err_vals,
                label=bead["label"],
                marker=bead["marker"],
                color=bead["color"],
                markersize=7.5,
                markeredgecolor="white",
                markeredgewidth=1.0,
                linewidth=1.8,
                capsize=3.5,
                elinewidth=1.4,
                alpha=0.90,
                zorder=4,
            )

    valid_pool = pool_stats_abs["mean"].notna()
    x_p = pool_stats_abs.loc[valid_pool, "x_mean"].to_numpy()
    y_p = pool_stats_abs.loc[valid_pool, "mean"].to_numpy()
    err_p = pool_stats_abs.loc[valid_pool, "sem"].to_numpy()
    ax.fill_between(x_p, y_p - err_p, y_p + err_p, color="#111111", alpha=0.10, edgecolor="none")
    ax.errorbar(
        x_p,
        y_p,
        yerr=err_p,
        label="Pooled (All Cargo)",
        marker="D",
        color="#111111",
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.4,
        linewidth=2.5,
        capsize=4.0,
        elinewidth=1.8,
        zorder=10,
    )

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Conditional Alignment $\langle \cos \phi \mid |M| \rangle$", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.2, 0.8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, framealpha=0.92, fontsize=9.5, loc="lower right")
    ax.set_title(r"Cargo–MT Flow Alignment vs $|M|$ ($\phi \equiv \theta_{\mathrm{cargo}} - \theta_{\mathrm{MT}}$)", fontsize=12, fontweight="bold", pad=10)
    save_figure_to_all(fig, "conditional_alignment_vs_abs_magnetization", out_dirs)

    # 1-B: <cos phi | M> vs M (Signed)
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    for bead in BEADS_INFO:
        bname = bead["name"]
        bdf = bead_stats_signed[bname]
        valid_b = bdf["mean"].notna()
        if valid_b.sum() > 0:
            x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
            y_vals = bdf.loc[valid_b, "mean"].to_numpy()
            err_vals = bdf.loc[valid_b, "sem"].to_numpy()
            ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.12, edgecolor="none")
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=err_vals,
                label=bead["label"],
                marker=bead["marker"],
                color=bead["color"],
                markersize=7.5,
                markeredgecolor="white",
                markeredgewidth=1.0,
                linewidth=1.8,
                capsize=3.5,
                elinewidth=1.4,
                alpha=0.90,
                zorder=4,
            )

    valid_pool_sgn = pool_stats_signed["mean"].notna()
    x_ps = pool_stats_signed.loc[valid_pool_sgn, "x_mean"].to_numpy()
    y_ps = pool_stats_signed.loc[valid_pool_sgn, "mean"].to_numpy()
    err_ps = pool_stats_signed.loc[valid_pool_sgn, "sem"].to_numpy()
    ax.fill_between(x_ps, y_ps - err_ps, y_ps + err_ps, color="#111111", alpha=0.10, edgecolor="none")
    ax.errorbar(
        x_ps,
        y_ps,
        yerr=err_ps,
        label="Pooled (All Cargo)",
        marker="D",
        color="#111111",
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.4,
        linewidth=2.5,
        capsize=4.0,
        elinewidth=1.8,
        zorder=10,
    )

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Cargo-Region Ising Magnetization $M$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Conditional Alignment $\langle \cos \phi \mid M \rangle$", fontsize=12, fontweight="bold")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.3, 0.8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, framealpha=0.92, fontsize=9.5, loc="lower right")
    ax.set_title(r"Cargo–MT Flow Alignment vs Signed $M$ ($\phi \equiv \theta_{\mathrm{cargo}} - \theta_{\mathrm{MT}}$)", fontsize=12, fontweight="bold", pad=10)
    save_figure_to_all(fig, "conditional_alignment_vs_signed_magnetization", out_dirs)

    # 1-C: 2-Panel
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.08})
    for bead in BEADS_INFO:
        bname = bead["name"]
        bdf = bead_stats_abs[bname]
        valid_b = bdf["mean"].notna()
        if valid_b.sum() > 0:
            x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
            y_vals = bdf.loc[valid_b, "mean"].to_numpy()
            err_vals = bdf.loc[valid_b, "sem"].to_numpy()
            ax_top.errorbar(
                x_vals,
                y_vals,
                yerr=err_vals,
                label=bead["label"],
                marker=bead["marker"],
                color=bead["color"],
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=1.5,
                capsize=3.0,
                elinewidth=1.2,
                alpha=0.85,
            )
    ax_top.errorbar(
        x_p,
        y_p,
        yerr=err_p,
        label="Pooled (All Cargo)",
        marker="D",
        color="#111111",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        linewidth=2.2,
        capsize=3.5,
        elinewidth=1.6,
        zorder=10,
    )
    ax_top.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_top.set_ylabel(r"Alignment $\langle \cos \phi \mid |M| \rangle$", fontsize=11, fontweight="bold")
    ax_top.set_ylim(-0.15, 0.8)
    ax_top.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_top.legend(frameon=True, framealpha=0.92, fontsize=8.5, loc="lower right", ncol=2)
    ax_top.set_title(r"Conditional Alignment $\langle \cos \phi \mid |M| \rangle$ vs $|M|$", fontsize=12, fontweight="bold", pad=8)

    width = (bins_abs[1] - bins_abs[0]) * 0.85
    ax_bot.bar(pool_stats_abs["x_center"], pool_stats_abs["count"], width=width, color=style_colors[0], alpha=0.75, edgecolor="#333333")
    ax_bot.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=11, fontweight="bold")
    ax_bot.set_ylabel("Count $N$", fontsize=11, fontweight="bold")
    ax_bot.set_yscale("log")
    ax_bot.set_xlim(-0.02, 1.02)
    ax_bot.grid(True, which="both", linestyle="--", alpha=0.4)
    save_figure_to_all(fig, "conditional_alignment_vs_abs_m_2panel", out_dirs)

    # 1-D: 6-Panel
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, bead in enumerate(BEADS_INFO):
        ax = axes[i]
        bname = bead["name"]
        bdf = bead_stats_abs[bname]
        valid_b = bdf["mean"].notna()
        bsub = df_valid[df_valid["bead_name"] == bname]
        if len(bsub) > 0:
            sample_sub = bsub.sample(min(len(bsub), 1200), random_state=42)
            ax.scatter(sample_sub["abs_m"], sample_sub["cos_phi"], color=bead["color"], alpha=0.18, s=16, edgecolors="none", zorder=2)

        ax.plot(x_p, y_p, color="#333333", linestyle="--", linewidth=1.8, label="Pooled Trend", alpha=0.7, zorder=3)

        if valid_b.sum() > 0:
            x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
            y_vals = bdf.loc[valid_b, "mean"].to_numpy()
            err_vals = bdf.loc[valid_b, "sem"].to_numpy()
            ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.20, edgecolor="none", zorder=4)
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=err_vals,
                label=f"{bead['label']} ($N={len(bsub)}$)",
                marker=bead["marker"],
                color=bead["color"],
                markersize=7.0,
                markeredgecolor="white",
                markeredgewidth=1.0,
                linewidth=2.0,
                capsize=3.5,
                elinewidth=1.4,
                zorder=5,
            )

        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_title(bead["label"], fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, framealpha=0.90, fontsize=8.5, loc="lower right")

    for ax in axes[3:]:
        ax.set_xlabel(r"Absolute Magnetization $|M|$", fontsize=10, fontweight="bold")
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel(r"Alignment $\langle \cos \phi \rangle$", fontsize=10, fontweight="bold")

    fig.suptitle(r"Cargo–MT Alignment $\langle \cos \phi \mid |M| \rangle$ per Cargo Size", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure_to_all(fig, "conditional_alignment_per_bead_panels", out_dirs)

    # 1-E: Distribution of phi
    fig, (ax_hist, ax_polar) = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1.3, 1.0]})
    sub_low_m = df_valid[df_valid["abs_m"] < 0.25]["phi_deg"]
    sub_high_m = df_valid[df_valid["abs_m"] > 0.75]["phi_deg"]

    bins_phi = np.linspace(-180, 180, 37)
    ax_hist.hist(df_valid["phi_deg"], bins=bins_phi, density=True, histtype="step", color="#111111", linewidth=2.0, label="All points")
    ax_hist.hist(sub_low_m, bins=bins_phi, density=True, histtype="stepfilled", color=style_colors[0], alpha=0.30, label=r"Disordered ($|M| < 0.25$)")
    ax_hist.hist(sub_high_m, bins=bins_phi, density=True, histtype="stepfilled", color=style_colors[3], alpha=0.45, label=r"Ordered ($|M| > 0.75$)")
    ax_hist.set_xlabel(r"Relative Angle $\phi = \theta_{\mathrm{cargo}} - \theta_{\mathrm{MT}}$ [deg]", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_hist.set_xlim(-180, 180)
    ax_hist.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_hist.legend(frameon=True, framealpha=0.92, fontsize=9.5)
    ax_hist.set_title(r"Angle Distribution $P(\phi)$", fontsize=12, fontweight="bold")

    ax_polar.remove()
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")
    phi_rad_all = df_valid["phi_rad"].to_numpy()
    phi_rad_high = df_valid[df_valid["abs_m"] > 0.75]["phi_rad"].to_numpy()

    counts_all, edges = np.histogram(phi_rad_all, bins=36, range=(-np.pi, np.pi), density=True)
    counts_high, _ = np.histogram(phi_rad_high, bins=36, range=(-np.pi, np.pi), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    ax_pol.plot(centers, counts_all, color="#111111", linewidth=1.8, label="All")
    ax_pol.plot(centers, counts_high, color=style_colors[3], linewidth=2.2, label=r"$|M| > 0.75$")
    ax_pol.fill_between(centers, 0, counts_high, color=style_colors[3], alpha=0.25)
    ax_pol.set_theta_zero_location("E")
    ax_pol.set_title(r"Polar Alignment Distribution $P(\phi)$", fontsize=12, fontweight="bold", pad=15)
    ax_pol.legend(frameon=True, framealpha=0.90, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.28, 0.0))

    fig.tight_layout()
    save_figure_to_all(fig, "cargo_mt_relative_angle_distribution", out_dirs)


# =============================================================================
# プロット案 2: 方向転換角の揺らぎ <(Delta theta)^2 | M> vs M
# =============================================================================

def generate_turning_fluctuation_plots(
    df: pd.DataFrame,
    out_dirs: List[Path],
    n_bins_abs: int = 8,
    n_bins_signed: int = 10,
    min_count: int = 5,
):
    print("\n" + "=" * 70)
    print("Generating Plot 2: Turning Angle Fluctuation <(Delta theta)^2 | M> vs M")
    print("=" * 70)

    # 4s 瞬時方向転換角
    has_4s = "delta_theta_4s_sq" in df.columns and df["delta_theta_4s_sq"].notna().any()
    # 20s ストライド方向転換角
    has_20s = "delta_theta_20s_sq" in df.columns and df["delta_theta_20s_sq"].notna().any()

    bins_abs = np.linspace(0.0, 1.0, n_bins_abs + 1)
    bins_signed = np.linspace(-1.0, 1.0, n_bins_signed + 1)

    summary_rows = []

    # -------------------------------------------------------------------------
    # 2-A: 瞬時方向転換角 (Delta t = 4 s) のプロット (主プロット)
    # -------------------------------------------------------------------------
    if has_4s:
        valid_4s = df[np.isfinite(df["delta_theta_4s_sq"]) & np.isfinite(df["abs_m"])].copy()
        print(f"Valid points for instantaneous 4s turning: {len(valid_4s)} / {len(df)}")

        pool_4s_abs = compute_binned_stats(valid_4s, "abs_m", "delta_theta_4s_sq", bins_abs, min_count=min_count)
        pool_4s_sgn = compute_binned_stats(valid_4s, "m_ising", "delta_theta_4s_sq", bins_signed, min_count=min_count)
        pool_4s_circ = compute_binned_stats(valid_4s, "abs_m", "circ_disp_4s", bins_abs, min_count=min_count)

        bead_4s_abs: Dict[str, pd.DataFrame] = {}
        bead_4s_sgn: Dict[str, pd.DataFrame] = {}
        bead_4s_circ: Dict[str, pd.DataFrame] = {}
        for bead in BEADS_INFO:
            bname = bead["name"]
            bsub = valid_4s[valid_4s["bead_name"] == bname]
            bead_4s_abs[bname] = compute_binned_stats(bsub, "abs_m", "delta_theta_4s_sq", bins_abs, min_count=min_count)
            bead_4s_sgn[bname] = compute_binned_stats(bsub, "m_ising", "delta_theta_4s_sq", bins_signed, min_count=min_count)
            bead_4s_circ[bname] = compute_binned_stats(bsub, "abs_m", "circ_disp_4s", bins_abs, min_count=min_count)

        # サマリーに追加
        for _, row in pool_4s_abs.iterrows():
            r = row.to_dict()
            r["variable"] = "delta_theta_4s_sq"
            r["condition"] = "Pooled (All)"
            summary_rows.append(r)
        for bname, bdf in bead_4s_abs.items():
            for _, row in bdf.iterrows():
                r = row.to_dict()
                r["variable"] = "delta_theta_4s_sq"
                r["condition"] = bname
                summary_rows.append(r)

        # 図: 主プロット <(Delta theta)^2 | |M|> vs |M| (Delta t = 4s)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for bead in BEADS_INFO:
            bname = bead["name"]
            bdf = bead_4s_abs[bname]
            valid_b = bdf["mean"].notna()
            if valid_b.sum() > 0:
                x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
                y_vals = bdf.loc[valid_b, "mean"].to_numpy()
                err_vals = bdf.loc[valid_b, "sem"].to_numpy()
                ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.12, edgecolor="none")
                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=err_vals,
                    label=bead["label"],
                    marker=bead["marker"],
                    color=bead["color"],
                    markersize=7.5,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    linewidth=1.8,
                    capsize=3.5,
                    elinewidth=1.4,
                    alpha=0.90,
                    zorder=4,
                )

        valid_pool = pool_4s_abs["mean"].notna()
        x_p4 = pool_4s_abs.loc[valid_pool, "x_mean"].to_numpy()
        y_p4 = pool_4s_abs.loc[valid_pool, "mean"].to_numpy()
        err_p4 = pool_4s_abs.loc[valid_pool, "sem"].to_numpy()
        ax.fill_between(x_p4, y_p4 - err_p4, y_p4 + err_p4, color="#111111", alpha=0.10, edgecolor="none")
        ax.errorbar(
            x_p4,
            y_p4,
            yerr=err_p4,
            label="Pooled (All Cargo)",
            marker="D",
            color="#111111",
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=1.4,
            linewidth=2.5,
            capsize=4.0,
            elinewidth=1.8,
            zorder=10,
        )

        ax.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Turning Angle Fluctuation $\langle (\Delta \theta)^2 \mid |M| \rangle$ [$\mathrm{rad}^2$]", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(1.5, 4.8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, framealpha=0.92, fontsize=9.5, loc="upper right")
        ax.set_title(r"Instantaneous Turning Fluctuation $\langle (\Delta \theta)^2 \rangle$ vs $|M|$ ($\Delta t = 4\,\mathrm{s}$)", fontsize=12, fontweight="bold", pad=10)
        save_figure_to_all(fig, "turning_angle_fluctuation_vs_abs_magnetization", out_dirs)

        # 2パネル図 (上: <(Delta theta)^2>, 下: サンプル数 N)
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.08})
        for bead in BEADS_INFO:
            bname = bead["name"]
            bdf = bead_4s_abs[bname]
            valid_b = bdf["mean"].notna()
            if valid_b.sum() > 0:
                x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
                y_vals = bdf.loc[valid_b, "mean"].to_numpy()
                err_vals = bdf.loc[valid_b, "sem"].to_numpy()
                ax_top.errorbar(
                    x_vals,
                    y_vals,
                    yerr=err_vals,
                    label=bead["label"],
                    marker=bead["marker"],
                    color=bead["color"],
                    markersize=6.5,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    linewidth=1.5,
                    capsize=3.0,
                    elinewidth=1.2,
                    alpha=0.85,
                )
        ax_top.errorbar(
            x_p4,
            y_p4,
            yerr=err_p4,
            label="Pooled (All Cargo)",
            marker="D",
            color="#111111",
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.2,
            capsize=3.5,
            elinewidth=1.6,
            zorder=10,
        )
        ax_top.set_ylabel(r"Fluctuation $\langle (\Delta \theta)^2 \rangle$ [$\mathrm{rad}^2$]", fontsize=11, fontweight="bold")
        ax_top.set_ylim(1.5, 4.8)
        ax_top.grid(True, which="both", linestyle="--", alpha=0.4)
        ax_top.legend(frameon=True, framealpha=0.92, fontsize=8.5, loc="upper right", ncol=2)
        ax_top.set_title(r"Instantaneous Turning Fluctuation $\langle (\Delta \theta)^2 \mid |M| \rangle$ vs $|M|$ ($\Delta t = 4\,\mathrm{s}$)", fontsize=12, fontweight="bold", pad=8)

        width = (bins_abs[1] - bins_abs[0]) * 0.85
        ax_bot.bar(pool_4s_abs["x_center"], pool_4s_abs["count"], width=width, color=style_colors[3], alpha=0.75, edgecolor="#333333")
        ax_bot.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=11, fontweight="bold")
        ax_bot.set_ylabel("Count $N$", fontsize=11, fontweight="bold")
        ax_bot.set_yscale("log")
        ax_bot.set_xlim(-0.02, 1.02)
        ax_bot.grid(True, which="both", linestyle="--", alpha=0.4)
        save_figure_to_all(fig, "turning_angle_fluctuation_vs_abs_m_2panel", out_dirs)

        # 6パネル条件別比較 (4s)
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, bead in enumerate(BEADS_INFO):
            ax = axes[i]
            bname = bead["name"]
            bdf = bead_4s_abs[bname]
            valid_b = bdf["mean"].notna()
            bsub = valid_4s[valid_4s["bead_name"] == bname]
            if len(bsub) > 0:
                sample_sub = bsub.sample(min(len(bsub), 1200), random_state=42)
                ax.scatter(sample_sub["abs_m"], sample_sub["delta_theta_4s_sq"], color=bead["color"], alpha=0.18, s=16, edgecolors="none", zorder=2)

            ax.plot(x_p4, y_p4, color="#333333", linestyle="--", linewidth=1.8, label="Pooled Trend", alpha=0.7, zorder=3)

            if valid_b.sum() > 0:
                x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
                y_vals = bdf.loc[valid_b, "mean"].to_numpy()
                err_vals = bdf.loc[valid_b, "sem"].to_numpy()
                ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.20, edgecolor="none", zorder=4)
                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=err_vals,
                    label=f"{bead['label']} ($N={len(bsub)}$)",
                    marker=bead["marker"],
                    color=bead["color"],
                    markersize=7.0,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    linewidth=2.0,
                    capsize=3.5,
                    elinewidth=1.4,
                    zorder=5,
                )

            ax.set_title(bead["label"], fontsize=11, fontweight="bold")
            ax.set_ylim(0.0, 10.5)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(frameon=True, framealpha=0.90, fontsize=8.5, loc="upper right")

        for ax in axes[3:]:
            ax.set_xlabel(r"Absolute Magnetization $|M|$", fontsize=10, fontweight="bold")
        for ax in [axes[0], axes[3]]:
            ax.set_ylabel(r"$\langle (\Delta \theta)^2 \rangle$ [$\mathrm{rad}^2$]", fontsize=10, fontweight="bold")

        fig.suptitle(r"Cargo Turning Fluctuation $\langle (\Delta \theta)^2 \mid |M| \rangle$ per Cargo Size ($\Delta t = 4\,\mathrm{s}$)", fontsize=13, fontweight="bold", y=0.99)
        fig.tight_layout()
        save_figure_to_all(fig, "turning_fluctuation_per_bead_panels", out_dirs)

        # 円分散 1 - <cos Delta theta> vs |M|
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for bead in BEADS_INFO:
            bname = bead["name"]
            bdf = bead_4s_circ[bname]
            valid_b = bdf["mean"].notna()
            if valid_b.sum() > 0:
                x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
                y_vals = bdf.loc[valid_b, "mean"].to_numpy()
                err_vals = bdf.loc[valid_b, "sem"].to_numpy()
                ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.12, edgecolor="none")
                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=err_vals,
                    label=bead["label"],
                    marker=bead["marker"],
                    color=bead["color"],
                    markersize=7.5,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    linewidth=1.8,
                    capsize=3.5,
                    elinewidth=1.4,
                    alpha=0.90,
                    zorder=4,
                )

        valid_pool_circ = pool_4s_circ["mean"].notna()
        x_pc = pool_4s_circ.loc[valid_pool_circ, "x_mean"].to_numpy()
        y_pc = pool_4s_circ.loc[valid_pool_circ, "mean"].to_numpy()
        err_pc = pool_4s_circ.loc[valid_pool_circ, "sem"].to_numpy()
        ax.fill_between(x_pc, y_pc - err_pc, y_pc + err_pc, color="#111111", alpha=0.10, edgecolor="none")
        ax.errorbar(
            x_pc,
            y_pc,
            yerr=err_pc,
            label="Pooled (All Cargo)",
            marker="D",
            color="#111111",
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=1.4,
            linewidth=2.5,
            capsize=4.0,
            elinewidth=1.8,
            zorder=10,
        )

        ax.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Circular Variance $1 - \langle \cos \Delta \theta \mid |M| \rangle$", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.2, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, framealpha=0.92, fontsize=9.5, loc="upper right")
        ax.set_title(r"Trajectory Angular Dispersion $1 - \langle \cos \Delta \theta \rangle$ vs $|M|$ ($\Delta t = 4\,\mathrm{s}$)", fontsize=12, fontweight="bold", pad=10)
        save_figure_to_all(fig, "circular_variance_vs_abs_magnetization", out_dirs)

    # -------------------------------------------------------------------------
    # 2-B: 2階差分 ddot{theta}(t) = Delta theta(t+Delta t) - Delta theta(t)
    # -------------------------------------------------------------------------
    if "ddot_theta_4s_sq" in df.columns and df["ddot_theta_4s_sq"].notna().any():
        valid_ddot = df[np.isfinite(df["ddot_theta_4s_sq"]) & np.isfinite(df["abs_m"])].copy()
        pool_ddot = compute_binned_stats(valid_ddot, "abs_m", "ddot_theta_4s_sq", bins_abs, min_count=min_count)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for bead in BEADS_INFO:
            bname = bead["name"]
            bsub = valid_ddot[valid_ddot["bead_name"] == bname]
            bdf = compute_binned_stats(bsub, "abs_m", "ddot_theta_4s_sq", bins_abs, min_count=min_count)
            valid_b = bdf["mean"].notna()
            if valid_b.sum() > 0:
                x_vals = bdf.loc[valid_b, "x_mean"].to_numpy()
                y_vals = bdf.loc[valid_b, "mean"].to_numpy()
                err_vals = bdf.loc[valid_b, "sem"].to_numpy()
                ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals, color=bead["color"], alpha=0.12, edgecolor="none")
                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=err_vals,
                    label=bead["label"],
                    marker=bead["marker"],
                    color=bead["color"],
                    markersize=7.5,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    linewidth=1.8,
                    capsize=3.5,
                    elinewidth=1.4,
                    alpha=0.90,
                    zorder=4,
                )

        valid_p = pool_ddot["mean"].notna()
        x_pdd = pool_ddot.loc[valid_p, "x_mean"].to_numpy()
        y_pdd = pool_ddot.loc[valid_p, "mean"].to_numpy()
        err_pdd = pool_ddot.loc[valid_p, "sem"].to_numpy()
        ax.fill_between(x_pdd, y_pdd - err_pdd, y_pdd + err_pdd, color="#111111", alpha=0.10, edgecolor="none")
        ax.errorbar(
            x_pdd,
            y_pdd,
            yerr=err_pdd,
            label="Pooled (All Cargo)",
            marker="D",
            color="#111111",
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=1.4,
            linewidth=2.5,
            capsize=4.0,
            elinewidth=1.8,
            zorder=10,
        )

        ax.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Heading Acceleration $\langle (\ddot{\theta})^2 \mid |M| \rangle$ [$\mathrm{rad}^2$]", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(1.5, 5.2)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, framealpha=0.92, fontsize=9.5, loc="upper right")
        ax.set_title(r"Heading Acceleration $\ddot{\theta} = \Delta\theta(t+\Delta t) - \Delta\theta(t)$ vs $|M|$", fontsize=12, fontweight="bold", pad=10)
        save_figure_to_all(fig, "heading_acceleration_fluctuation_vs_abs_m", out_dirs)

    # -------------------------------------------------------------------------
    # 2-C: タイムスケール比較 (Delta t = 4 s vs Delta t = 20 s)
    # -------------------------------------------------------------------------
    if has_4s and has_20s:
        valid_20s = df[np.isfinite(df["delta_theta_20s_sq"]) & np.isfinite(df["abs_m"])].copy()
        pool_20s_abs = compute_binned_stats(valid_20s, "abs_m", "delta_theta_20s_sq", bins_abs, min_count=min_count)

        fig, ax = plt.subplots(figsize=(7.8, 5.5))
        ax.fill_between(x_p4, y_p4 - err_p4, y_p4 + err_p4, color=style_colors[0], alpha=0.15, edgecolor="none")
        ax.errorbar(
            x_p4,
            y_p4,
            yerr=err_p4,
            label=r"Instantaneous $\Delta t = 4\,\mathrm{s}$ (Pooled)",
            marker="o",
            color=style_colors[0],
            markersize=8.0,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.4,
            capsize=4.0,
            elinewidth=1.6,
        )
        valid_p20 = pool_20s_abs["mean"].notna()
        x_p20 = pool_20s_abs.loc[valid_p20, "x_mean"].to_numpy()
        y_p20 = pool_20s_abs.loc[valid_p20, "mean"].to_numpy()
        err_p20 = pool_20s_abs.loc[valid_p20, "sem"].to_numpy()
        ax.fill_between(x_p20, y_p20 - err_p20, y_p20 + err_p20, color=style_colors[3], alpha=0.15, edgecolor="none")
        ax.errorbar(
            x_p20,
            y_p20,
            yerr=err_p20,
            label=r"Stride $\Delta t = 20\,\mathrm{s}$ (Pooled)",
            marker="s",
            color=style_colors[3],
            markersize=8.0,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.4,
            capsize=4.0,
            elinewidth=1.6,
        )

        ax.set_xlabel(r"Cargo-Region Absolute Magnetization $|M|$", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Turning Fluctuation $\langle (\Delta \theta)^2 \mid |M| \rangle$ [$\mathrm{rad}^2$]", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(1.5, 4.5)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, framealpha=0.92, fontsize=10.0, loc="upper right")
        ax.set_title(r"Timescale Comparison of Cargo Turning Fluctuation vs $|M|$", fontsize=12, fontweight="bold", pad=10)
        save_figure_to_all(fig, "turning_fluctuation_timescale_comparison", out_dirs)

    if summary_rows:
        df_summary2 = pd.DataFrame(summary_rows)
        save_csv_to_all(df_summary2, "cargo_turning_fluctuation_summary", out_dirs)


# =============================================================================
# メイン処理
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot cargo-MT conditional alignment <cos phi | M> and turning angle fluctuation <(Delta theta)^2 | M> vs M."
    )
    parser.add_argument(
        "--points_csv",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity" / "cargo_spin_velocity_points.csv",
        help="Path to cargo_spin_velocity_points.csv",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root directory of experiments (e.g. /mnt/NAS-Ebanaru/Sasaki/MTsingleBeads)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_alignment_turning",
        help="Output directory for figures and tables",
    )
    parser.add_argument(
        "--no_save_root",
        action="store_true",
        help="Do not save copies into data_root/figure",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of augmented vectors even if cached file exists",
    )
    parser.add_argument(
        "--n_bins_abs",
        type=int,
        default=8,
        help="Number of bins for absolute magnetization |M|",
    )
    parser.add_argument(
        "--n_bins_signed",
        type=int,
        default=10,
        help="Number of bins for signed magnetization M",
    )
    parser.add_argument(
        "--min_bin_count",
        type=int,
        default=5,
        help="Minimum count per bin to display",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", "alignment", "turning"],
        help="Which plots to generate: all, alignment (Plot 1), turning (Plot 2)",
    )

    args = parser.parse_args()
    apply_custom_style()

    data_root = args.data_root or find_default_root()
    out_dirs = [args.output_dir]
    if not args.no_save_root and data_root.exists():
        root_fig = data_root / "figure" / "cargo_alignment_turning"
        out_dirs.append(root_fig)

    print("=" * 70)
    print("Cargo Alignment and Turning Fluctuation Analysis")
    print(f"Data Root   : {data_root}")
    print(f"Points CSV  : {args.points_csv}")
    print(f"Output Dirs : {[str(d) for d in out_dirs]}")
    print("=" * 70)

    df_augmented = load_and_augment_data(
        points_csv=args.points_csv,
        data_root=data_root,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
    )

    do_all = "all" in args.plots
    if do_all or "alignment" in args.plots:
        generate_alignment_plots(
            df=df_augmented,
            out_dirs=out_dirs,
            n_bins_abs=args.n_bins_abs,
            n_bins_signed=args.n_bins_signed,
            min_count=args.min_bin_count,
        )

    if do_all or "turning" in args.plots:
        generate_turning_fluctuation_plots(
            df=df_augmented,
            out_dirs=out_dirs,
            n_bins_abs=args.n_bins_abs,
            n_bins_signed=args.n_bins_signed,
            min_count=args.min_bin_count,
        )

    print("\n[SUCCESS] All requested analyses and plots completed successfully.")


if __name__ == "__main__":
    main()
