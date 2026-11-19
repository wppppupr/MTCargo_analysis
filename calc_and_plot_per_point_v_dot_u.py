#!/usr/bin/env python3
"""
calc_and_plot_per_point_v_dot_u.py

貨物ディスク周囲（フットプリント内）の各流れ場ベクトル点 r ごとの内積
    c_pt(r, t) = v_hat_i(t) . u_hat(r, t)
および符号付き角度
    theta_pt(r, t) = atan2(u_hat x v_hat, u_hat . v_hat)
を全点分集計してヒストグラム P(v_hat_i . u_hat) と P(theta_pt) を算出し、
貨物が実際に感じている有効ポテンシャル U_eff を導出する。

- キャッシュ: scratch/per_point_v_dot_u_hist_cache.pkl
- 出力先:
    - figure/per_point_v_dot_u/
    - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/per_point_v_dot_u/
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
OUT_DIRS = [
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
    ROOT_DIR / "figure" / "per_point_v_dot_u",
]
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
HIST_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"

RAD_MAP = {
    "beads06um": 3,
    "beads1um": 6,
    "beads3um": 15,
    "beads5um": 25,
    "beads7um": 35,
    "beads20um": 95,
}
BEST_MAP = {
    "beads06um": 5,
    "beads1um": 10,
    "beads3um": 20,
    "beads5um": 30,
    "beads7um": 40,
    "beads20um": 100,
}

BINS_COS = np.linspace(-1.0, 1.0, 41)  # 40 bins
BINS_TH = np.linspace(-np.pi, np.pi, 37)  # 36 bins


def _process_experiment(args):
    exp_name, bname, exp_rows = args
    p_rad = RAD_MAP[bname]
    best_rad = BEST_MAP[bname]
    r_ceil = best_rad

    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    ed_list = [d for d in exp_dirs if d.name == exp_name]
    if not ed_list:
        return []
    flow_p = ed_list[0] / "GFP_flows.h5"
    if not flow_p.exists():
        return []

    # 粒子ごとにヒストグラムを蓄積
    p_data = {}
    frame_groups = exp_rows.groupby("frame")

    with h5py.File(flow_p, "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        flow = f[list(f.keys())[0]]
        total_frms = flow.shape[0]
        h, w = flow.shape[2], flow.shape[3]

        for frm, grp in frame_groups:
            frm_int = int(frm)
            if frm_int >= total_frms:
                continue

            for _, r in grp.iterrows():
                cdx = float(r["cargo_dx"])
                cdy = float(r["cargo_dy"])
                v_norm = np.hypot(cdx, cdy)
                if not np.isfinite(v_norm) or v_norm <= 0:
                    continue
                vx_hat = cdx / v_norm
                vy_hat = cdy / v_norm

                bx = int(round(r["x_px"]))
                by = int(round(r["y_px"]))

                y_min = max(0, by - r_ceil)
                y_max = min(h, by + r_ceil + 1)
                x_min = max(0, bx - r_ceil)
                x_max = min(w, bx + r_ceil + 1)

                patch = flow[frm_int, :, y_min:y_max, x_min:x_max]
                u_box = patch[0]
                v_box = patch[1]

                yg, xg = np.ogrid[y_min - by : y_max - by, x_min - bx : x_max - bx]
                dsq = xg**2 + yg**2
                m = (dsq <= best_rad**2) & (dsq > p_rad**2)
                if not np.any(m):
                    continue

                u_in = u_box[m].astype(np.float32)
                v_in = v_box[m].astype(np.float32)
                u_norm = np.hypot(u_in, v_in)
                valid = u_norm > 0
                if not np.any(valid):
                    continue

                ux_hat = u_in[valid] / u_norm[valid]
                uy_hat = v_in[valid] / u_norm[valid]

                c_pt = np.clip(vx_hat * ux_hat + vy_hat * uy_hat, -1.0, 1.0)
                cross_pt = ux_hat * vy_hat - uy_hat * vx_hat
                th_pt = np.arctan2(cross_pt, c_pt)

                hc, _ = np.histogram(c_pt, bins=BINS_COS)
                ht, _ = np.histogram(th_pt, bins=BINS_TH)

                pid = f"{exp_name}_{int(r['particle'])}"
                if pid not in p_data:
                    p_data[pid] = {
                        "bead_name": bname,
                        "exp_dir": exp_name,
                        "particle_id": pid,
                        "rc_over_xi": float(r["rc_over_xi"]),
                        "hist_c": np.zeros(len(BINS_COS) - 1, dtype=np.int64),
                        "hist_th": np.zeros(len(BINS_TH) - 1, dtype=np.int64),
                        "n_pts": 0,
                        "n_frames": 0,
                        "sum_c": 0.0,
                        "sum_c2": 0.0,
                    }
                d = p_data[pid]
                d["hist_c"] += hc
                d["hist_th"] += ht
                d["n_pts"] += int(len(c_pt))
                d["n_frames"] += 1
                d["sum_c"] += float(np.sum(c_pt))
                d["sum_c2"] += float(np.sum(c_pt**2))

    print(f"  [{bname}/{exp_name}] particles={len(p_data)}")
    return list(p_data.values())


def extract_or_load_cache(force_recompute: bool = False):
    if HIST_CACHE.exists() and not force_recompute:
        print(f"Loading cached per-point histograms from {HIST_CACHE}...")
        with open(HIST_CACHE, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    print(f"Loading {INST_CACHE}...")
    df_inst = pd.read_pickle(INST_CACHE)

    tasks = []
    for (bname, exp_name), grp in df_inst.groupby(["bead_name", "exp_dir"]):
        tasks.append((exp_name, bname, grp))

    print(f"Extracting per-point v_hat . u_hat across {len(tasks)} experiments using Pool(8)...")
    with Pool(processes=8) as pool:
        results_nested = pool.map(_process_experiment, tasks)

    records = [item for sublist in results_nested for item in sublist]
    with open(HIST_CACHE, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} particle records to {HIST_CACHE} in {time.time() - t0:.1f}s.")
    return records


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    records = extract_or_load_cache(force_recompute=False)
    df_rec = pd.DataFrame(records)

    c_centers = 0.5 * (BINS_COS[:-1] + BINS_COS[1:])
    dc = BINS_COS[1] - BINS_COS[0]
    # 厳密な角度ビン幅 Δθ_k = arccos(c_k) - arccos(c_{k+1}) によるヤコビアン (Δc / Δθ_k)
    dth_exact = np.arccos(BINS_COS[:-1]) - np.arccos(BINS_COS[1:])
    denom_jacob = dc / dth_exact
    idx_zero_c = int(np.argmin(np.abs(c_centers)))

    th_centers = 0.5 * (BINS_TH[:-1] + BINS_TH[1:])
    dth = BINS_TH[1] - BINS_TH[0]
    idx_zero_th = int(np.argmin(np.abs(th_centers)))
    idx_pi_th = 0  # edge near -pi

    B_BOOT = 1000
    rng = np.random.default_rng(42)

    boot_dict = {}
    summary_rows = []

    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df_rec[df_rec["bead_name"] == bname]
        if sub.empty:
            continue

        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        tot_pts = int(sub["n_pts"].sum())
        tot_frms = int(sub["n_frames"].sum())
        mean_c = float(sub["sum_c"].sum() / tot_pts)
        mean_c2 = float(sub["sum_c2"].sum() / tot_pts)

        hc_arr = np.vstack(sub["hist_c"].values)
        ht_arr = np.vstack(sub["hist_th"].values)

        hc_tot = hc_arr.sum(axis=0).astype(float)
        P_c = hc_tot / (hc_tot.sum() * dc)
        ht_tot = ht_arr.sum(axis=0).astype(float)
        P_th = ht_tot / (ht_tot.sum() * dth)

        # 厳密な角度ビン幅 Δθ_k による等方基準ポテンシャル
        # U_eff(c_k) = -ln( P(c_k) / P_iso,k ) = -ln( pi * (Δc / Δθ_k) * P(c_k) )
        p_iso_k = dth_exact / (np.pi * dc)
        with np.errstate(divide="ignore", invalid="ignore"):
            U_c = np.where(P_c > 0, -np.log(P_c / p_iso_k), np.nan)
            U_th = np.where(P_th > 0, -np.log(2.0 * np.pi * P_th), np.nan)

        # Cluster bootstrap over particles
        boot_Pc = np.zeros((B_BOOT, len(c_centers)))
        boot_Uc = np.full((B_BOOT, len(c_centers)), np.nan)
        boot_Pth = np.zeros((B_BOOT, len(th_centers)))
        boot_Uth = np.full((B_BOOT, len(th_centers)), np.nan)
        boot_mc = np.zeros(B_BOOT)

        pts_arr = sub["n_pts"].values.astype(float)
        sumc_arr = sub["sum_c"].values.astype(float)

        for bi in range(B_BOOT):
            idx = rng.integers(0, n_p, size=n_p)
            hc_b = hc_arr[idx].sum(axis=0).astype(float)
            pc_b = hc_b / (hc_b.sum() * dc)
            boot_Pc[bi] = pc_b
            with np.errstate(divide="ignore", invalid="ignore"):
                boot_Uc[bi] = np.where(pc_b > 0, -np.log(pc_b / p_iso_k), np.nan)

            ht_b = ht_arr[idx].sum(axis=0).astype(float)
            pth_b = ht_b / (ht_b.sum() * dth)
            boot_Pth[bi] = pth_b
            with np.errstate(divide="ignore", invalid="ignore"):
                boot_Uth[bi] = np.where(pth_b > 0, -np.log(2.0 * np.pi * pth_b), np.nan)

            boot_mc[bi] = sumc_arr[idx].sum() / pts_arr[idx].sum()

        delta_U_c = float(U_c[0] - U_c[-1]) if np.isfinite(U_c[0]) and np.isfinite(U_c[-1]) else np.nan
        delta_U_th = float(U_th[idx_pi_th] - U_th[idx_zero_th]) if np.isfinite(U_th[idx_pi_th]) and np.isfinite(U_th[idx_zero_th]) else np.nan

        boot_dict[bname] = {
            "rc_xi": rc_xi,
            "n_p": n_p,
            "tot_pts": tot_pts,
            "tot_frms": tot_frms,
            "mean_c": mean_c,
            "mean_c2": mean_c2,
            "P_c": P_c,
            "P_c_lo": np.percentile(boot_Pc, 2.5, axis=0),
            "P_c_hi": np.percentile(boot_Pc, 97.5, axis=0),
            "U_c": U_c,
            "U_c_lo": np.nanpercentile(boot_Uc, 2.5, axis=0),
            "U_c_hi": np.nanpercentile(boot_Uc, 97.5, axis=0),
            "P_th": P_th,
            "P_th_lo": np.percentile(boot_Pth, 2.5, axis=0),
            "P_th_hi": np.percentile(boot_Pth, 97.5, axis=0),
            "U_th": U_th,
            "U_th_lo": np.nanpercentile(boot_Uth, 2.5, axis=0),
            "U_th_hi": np.nanpercentile(boot_Uth, 97.5, axis=0),
        }

        summary_rows.append({
            "bead_name": bname,
            "label": b_item["label"],
            "rc_over_xi": rc_xi,
            "n_particles": n_p,
            "n_frames": tot_frms,
            "n_flow_points": tot_pts,
            "mean_v_dot_u": mean_c,
            "mean_v_dot_u_ci_low": float(np.percentile(boot_mc, 2.5)),
            "mean_v_dot_u_ci_high": float(np.percentile(boot_mc, 97.5)),
            "mean_v_dot_u_sq": mean_c2,
            "delta_U_cos_pi_vs_0": delta_U_c,
            "delta_U_theta_pi_vs_0": delta_U_th,
        })

    df_sum = pd.DataFrame(summary_rows).sort_values("rc_over_xi")
    for d in OUT_DIRS:
        df_sum.to_csv(d / "per_point_v_dot_u_summary.csv", index=False)

    print("\n=== Per-Point P(v_hat_i . u_hat) Summary ===")
    print(df_sum[["bead_name", "rc_over_xi", "n_particles", "n_flow_points", "mean_v_dot_u", "mean_v_dot_u_sq", "delta_U_theta_pi_vs_0"]].to_string(index=False))

    # --- Figure 1: P(v_hat_i . u_hat) 6パネル ---
    p_iso_bin = dth_exact / (np.pi * dc)

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]

        ax.bar(c_centers, res["P_c"], width=dc * 0.92, color=col, alpha=0.35, edgecolor=col, linewidth=0.8,
               label=rf"Histogram ($N_{{\mathrm{{pt}}}}={res['tot_pts']:,}$)", zorder=2)
        ax.plot(c_centers, res["P_c"], color=col, linewidth=2.1, marker=b_item["marker"], markersize=4.5, zorder=4)
        ax.fill_between(c_centers, res["P_c_lo"], res["P_c_hi"], color=col, alpha=0.20, label="95% CI (Bootstrap)", zorder=3)
        ax.plot(c_centers, p_iso_bin, color="#555555", linestyle=":", linewidth=1.6, label=r"Isotropic $\Delta\theta_k/(\pi\Delta c)$", zorder=1)
        ax.axvline(res["mean_c"], color=col, linestyle="--", linewidth=1.8, label=rf"Mean $= {res['mean_c']:.3f}$", zorder=5)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0.0, 3.0)
        ax.set_title(rf"{lbl}  ($R_c/\xi={res['rc_xi']:.2f}$, $\langle\hat{{v}}_i\cdot\hat{{u}}\rangle={res['mean_c']:.2f}$)",
                     fontsize=12.5, fontweight="bold")
        ax.set_xlabel(r"Per-Point Alignment $\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=12)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}})$", fontsize=12.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle(r"Per-Point Footprint Alignment Histogram $P(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}})$",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    vcd.save_figure(fig, "per_point_v_dot_u_histogram_6panels", OUT_DIRS)
    plt.close(fig)

    # --- Figure 2: 1パネル P(v_hat_i . u_hat) ---
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.plot(c_centers, p_iso_bin, color="#555555", linestyle="--", linewidth=1.6, label=r"Isotropic 2D $\Delta\theta_k/(\pi\Delta c)$", zorder=1)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        ax.plot(c_centers, res["P_c"], marker=b_item["marker"], markersize=5.5, color=col, linewidth=2.2,
                label=rf"{lbl} ($\langle c\rangle={res['mean_c']:.2f}$)", zorder=3 + idx)
        ax.fill_between(c_centers, res["P_c_lo"], res["P_c_hi"], color=col, alpha=0.14, zorder=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(r"Per-Point Alignment $\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=14)
    ax.set_ylabel(r"Probability Density $P(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}})$", fontsize=14)
    ax.set_title(r"Per-Point Footprint Alignment $P(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}})$ Across Cargo Sizes", fontsize=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    vcd.save_figure(fig, "per_point_v_dot_u_histogram_1panel", OUT_DIRS)
    plt.close(fig)

    # --- Figure 3: 有効ポテンシャル U_eff(theta_pt) = -ln(2*pi*P(theta_pt)) 6パネル & 1パネル ---
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        valid = np.isfinite(res["U_th"])
        ax.plot(th_centers[valid], res["U_th"][valid], color=col, marker=b_item["marker"], markersize=4.5, linewidth=2.1,
                label=r"Felt Potential $\tilde{U}_{\mathrm{eff}}(\theta_{\mathrm{pt}})$", zorder=4)
        ax.fill_between(th_centers[valid], res["U_th_lo"][valid], res["U_th_hi"][valid], color=col, alpha=0.22, zorder=2)
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.6, 1.2)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_title(rf"{lbl}  ($R_c/\xi={res['rc_xi']:.2f}$)", fontsize=13, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel(r"Per-Point Relative Angle $\theta_{\mathrm{pt}}$ (rad)", fontsize=12.5)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(\theta_{\mathrm{pt}})$", fontsize=12.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9.0, framealpha=0.92)

    fig.suptitle(r"Effective Potential Felt by Cargo from Per-Point Flow Vectors $\tilde{U}_{\mathrm{eff}}(\theta_{\mathrm{pt}}) = -\ln(2\pi P(\theta_{\mathrm{pt}}))$",
                 fontsize=15.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    vcd.save_figure(fig, "per_point_theta_effective_potential_6panels", OUT_DIRS)
    plt.close(fig)

    # 1パネル重ね描き U_eff(theta_pt)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, label="Isotropic flat potential")
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        valid = np.isfinite(res["U_th"])
        ax.plot(th_centers[valid], res["U_th"][valid], marker=b_item["marker"], markersize=5.5, color=col, linewidth=2.2,
                label=rf"{lbl} ($R_c/\xi={res['rc_xi']:.2f}$)", zorder=3 + idx)
        ax.fill_between(th_centers[valid], res["U_th_lo"][valid], res["U_th_hi"][valid], color=col, alpha=0.12, zorder=2)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.6, 1.2)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"Per-Point Relative Angle $\theta_{\mathrm{pt}}$ (rad)", fontsize=14)
    ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(\theta_{\mathrm{pt}}) = -\ln(2\pi P(\theta_{\mathrm{pt}}))$", fontsize=14)
    ax.set_title(r"Effective Potential Felt by Cargo (Per-Point $\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}$)", fontsize=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    vcd.save_figure(fig, "per_point_theta_effective_potential_1panel", OUT_DIRS)
    plt.close(fig)

    # --- Figure 4: 角度に戻す前の c = v_hat_i . u_hat の有効ポテンシャル U_eff(c) = -ln(sqrt(1-c^2)*P(c)/P(0)) ---
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        valid = np.isfinite(res["U_c"])
        ax.plot(c_centers[valid], res["U_c"][valid], color=col, marker=b_item["marker"], markersize=4.5, linewidth=2.1,
                label=r"$\tilde{U}_{\mathrm{eff}}(c) = -\ln(P(c)/P_{\mathrm{iso}}(c))$", zorder=4)
        ax.fill_between(c_centers[valid], res["U_c_lo"][valid], res["U_c_hi"][valid], color=col, alpha=0.22, zorder=2)
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.6, 1.4)
        ax.set_title(rf"{lbl}  ($R_c/\xi={res['rc_xi']:.2f}$)", fontsize=13, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel(r"Per-Point Alignment $c = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=12.5)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$", fontsize=12.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

    fig.suptitle(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c) = -\ln\left(P(c)/P_{\mathrm{iso}}(c)\right) = -\ln\left(\pi\sqrt{1-c^2}\,P(c)\right)$",
                 fontsize=14.0, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    vcd.save_figure(fig, "per_point_cos_effective_potential_6panels", OUT_DIRS)
    plt.close(fig)

    # 1パネル重ね描き U_eff(c)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, label="Isotropic flat potential")
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        valid = np.isfinite(res["U_c"])
        ax.plot(c_centers[valid], res["U_c"][valid], marker=b_item["marker"], markersize=5.5, color=col, linewidth=2.2,
                label=rf"{lbl} ($R_c/\xi={res['rc_xi']:.2f}$)", zorder=3 + idx)
        ax.fill_between(c_centers[valid], res["U_c_lo"][valid], res["U_c_hi"][valid], color=col, alpha=0.12, zorder=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.6, 1.4)
    ax.set_xlabel(r"Per-Point Alignment $c = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=14)
    ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c) = -\ln\left(P(c)/P_{\mathrm{iso}}(c)\right)$", fontsize=13.5)
    ax.set_title(r"Effective Potential in Cosine Space $\tilde{U}_{\mathrm{eff}}(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}})$ Across Cargo Sizes", fontsize=14.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    vcd.save_figure(fig, "per_point_cos_effective_potential_1panel", OUT_DIRS)
    plt.close(fig)

    print("All per-point P(v_hat_i . u_hat) figures and summary saved successfully!")


if __name__ == "__main__":
    main()
