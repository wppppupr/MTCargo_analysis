#!/usr/bin/env python3
"""
quantify_k1_and_excess_pinning.py

厳密な角度ビン幅 Δθ_k = arccos(c_k) - arccos(c_{k+1}) で補正した
有効ポテンシャル U_tilde_eff(c) に対して、2段階クロスオーバーを定量化する:
  1. 直線領域 c in [-0.8, 0.6] の勾配から大域整列パラメータ kappa_1(R_c/xi) を抽出
  2. 実測 U_tilde_eff(c -> 1) と直線外挿 (-kappa_1 * c_edge + b) の差から
     +1 での過剰落ち込み（レール捕捉深さ） Delta U_tilde_+1(R_c/xi) を算出
  3. 粒子単位クラスターブートストラップ (B=1000) により両指標の 95% 信頼区間を算出

出力先:
  - figure/per_point_v_dot_u/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/per_point_v_dot_u/
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import verify_cos_decomposition as vcd
from calc_and_plot_kl_decomposition import solve_kappa1_from_m1

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = vcd.ROOT_DIR
OUT_DIRS = [
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
    ROOT_DIR / "figure" / "per_point_v_dot_u",
]
HIST_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"

BINS_COS = np.linspace(-1.0, 1.0, 41)  # 40 bins, dc = 0.05
C_MIN_FIT = -0.8
C_MAX_FIT = 0.6


def compute_u_and_decompose(hc_counts: np.ndarray, c_centers: np.ndarray,
                            p_iso_k: np.ndarray, dc: float,
                            fit_mask: np.ndarray, k1_fixed: float):
    """1次モーメント一致で決めた kappa_1 を固定し、U_eff(c) の直線領域 [-0.8, 0.6] から切片 b と Delta U_+1 を抽出"""
    tot = float(hc_counts.sum())
    if tot <= 0:
        return None
    p_c = hc_counts / (tot * dc)

    with np.errstate(divide="ignore", invalid="ignore"):
        u_c = np.where(p_c > 0, -np.log(p_c / p_iso_k), np.nan)

    valid_fit = fit_mask & np.isfinite(u_c)
    if np.sum(valid_fit) < 4:
        return None

    x_fit = c_centers[valid_fit]
    y_fit = u_c[valid_fit]
    k1 = float(max(0.0, k1_fixed))
    b_int = float(np.mean(y_fit + k1 * x_fit))

    u_obs_edge = float(u_c[-1])
    u_lin_edge = float(-k1 * c_centers[-1] + b_int)
    u_lin_at_1 = float(-k1 * 1.0 + b_int)
    du_plus1_edge = float(u_lin_edge - u_obs_edge)
    du_plus1_at_1 = float(u_lin_at_1 - u_obs_edge)

    return {
        "u_c": u_c,
        "k1": k1,
        "b_int": b_int,
        "u_obs_edge": u_obs_edge,
        "u_lin_edge": u_lin_edge,
        "du_plus1": du_plus1_edge,
        "du_plus1_at_1": du_plus1_at_1,
    }


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    with open(HIST_CACHE, "rb") as f:
        df_rec = pd.DataFrame(pickle.load(f))

    c_centers = 0.5 * (BINS_COS[:-1] + BINS_COS[1:])
    dc = float(BINS_COS[1] - BINS_COS[0])
    dth_exact = np.arccos(BINS_COS[:-1]) - np.arccos(BINS_COS[1:])
    p_iso_k = dth_exact / (np.pi * dc)
    fit_mask = (c_centers >= C_MIN_FIT) & (c_centers <= C_MAX_FIT)

    B_BOOT = 1000
    rng = np.random.default_rng(42)

    rows = []
    panel_cache = {}

    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df_rec[df_rec["bead_name"] == bname]
        if sub.empty:
            continue

        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        rc_xi_sem = float(sub["rc_over_xi"].sem()) if n_p > 1 else 0.0
        hc_arr = np.vstack(sub["hist_c"].values).astype(float)
        sumc_arr = sub["sum_c"].values.astype(float)
        npts_arr = sub["n_pts"].values.astype(float)

        m1_obs = float(sumc_arr.sum() / npts_arr.sum())
        k1_obs = solve_kappa1_from_m1(m1_obs)
        obs = compute_u_and_decompose(hc_arr.sum(axis=0), c_centers, p_iso_k, dc, fit_mask, k1_fixed=k1_obs)

        boot_u = np.full((B_BOOT, len(c_centers)), np.nan)
        boot_k1, boot_du, boot_du1 = [], [], []
        for bi in range(B_BOOT):
            idx = rng.integers(0, n_p, size=n_p)
            tot_n_b = float(npts_arr[idx].sum())
            if tot_n_b <= 0:
                continue
            k1_b = solve_kappa1_from_m1(float(sumc_arr[idx].sum() / tot_n_b))
            res_b = compute_u_and_decompose(hc_arr[idx].sum(axis=0), c_centers, p_iso_k, dc, fit_mask, k1_fixed=k1_b)
            if res_b is None:
                continue
            boot_u[bi] = res_b["u_c"]
            boot_k1.append(res_b["k1"])
            boot_du.append(res_b["du_plus1"])
            boot_du1.append(res_b["du_plus1_at_1"])

        panel_cache[bname] = {
            **obs,
            "u_lo": np.nanpercentile(boot_u, 2.5, axis=0),
            "u_hi": np.nanpercentile(boot_u, 97.5, axis=0),
            "rc_xi": rc_xi,
        }

        rows.append({
            "bead_name": bname,
            "diameter_um": b_item["diameter_um"],
            "label": b_item["label"],
            "color": b_item["color"],
            "marker": b_item["marker"],
            "rc_over_xi": rc_xi,
            "rc_over_xi_sem": rc_xi_sem,
            "n_particles": n_p,
            "kappa_1": obs["k1"],
            "kappa_1_ci_low": float(np.percentile(boot_k1, 2.5)),
            "kappa_1_ci_high": float(np.percentile(boot_k1, 97.5)),
            "intercept_b": obs["b_int"],
            "U_lin_edge": obs["u_lin_edge"],
            "U_obs_edge": obs["u_obs_edge"],
            "delta_U_plus1": obs["du_plus1"],
            "delta_U_plus1_ci_low": float(np.percentile(boot_du, 2.5)),
            "delta_U_plus1_ci_high": float(np.percentile(boot_du, 97.5)),
            "delta_U_plus1_at_c1": obs["du_plus1_at_1"],
        })

    df_out = pd.DataFrame(rows).sort_values("rc_over_xi").reset_index(drop=True)
    for d in OUT_DIRS:
        df_out.drop(columns=["color", "marker"]).to_csv(d / "k1_and_excess_pinning_summary.csv", index=False)

    print("=== Extracted Bulk Alignment kappa_1 & Excess Pinning Drop Delta U_+1 ===")
    print(df_out[["bead_name", "diameter_um", "rc_over_xi", "kappa_1", "kappa_1_ci_low", "kappa_1_ci_high",
                  "delta_U_plus1", "delta_U_plus1_ci_low", "delta_U_plus1_ci_high"]].to_string(index=False))

    # --- Figure 1: 6パネル 直線フィット [-0.8, 0.6] と +1 での過剰落ち込み可視化 ---
    c_line = np.linspace(-1.0, 1.0, 100)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        pc = panel_cache[bname]

        ax.axvspan(C_MIN_FIT, C_MAX_FIT, color="#eeeeee", alpha=0.8, zorder=0,
                   label=r"Linear fit range $[-0.8, 0.6]$" if idx == 0 else None)
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
        ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.9, zorder=1)

        valid = np.isfinite(pc["u_c"])
        ax.plot(c_centers[valid], pc["u_c"][valid], color=col, marker=b_item["marker"],
                markersize=4.5, linewidth=2.1, label=r"Observed $\tilde{U}_{\mathrm{eff}}(c)$", zorder=3)
        ax.fill_between(c_centers[valid], pc["u_lo"][valid], pc["u_hi"][valid],
                        color=col, alpha=0.22, zorder=2)

        u_line = -pc["k1"] * c_line + pc["b_int"]
        ax.plot(c_line, u_line, color="#111111", linestyle="--", linewidth=2.0,
                label=rf"Linear $-\kappa_1 c + b$ ($\kappa_1={pc['k1']:.2f}$)", zorder=4)

        # c -> 1 での過剰落ち込み矢印
        x_arr = c_centers[-1]
        ax.annotate(
            "", xy=(x_arr, pc["u_obs_edge"]), xytext=(x_arr, pc["u_lin_edge"]),
            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2), zorder=6
        )
        ax.text(
            0.94, 0.5 * (pc["u_obs_edge"] + pc["u_lin_edge"]),
            rf"$\Delta\tilde{{U}}_{{+1}}={pc['du_plus1']:+.2f}$",
            color="#d62728", fontsize=9.8, fontweight="bold", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.88), zorder=7
        )

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.6, 1.0)
        ax.set_title(rf"{lbl}  ($R_c/\xi={pc['rc_xi']:.2f}$)", fontsize=13, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel(r"Per-Point Alignment $c = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=12.5)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$", fontsize=12.5)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="lower left", fontsize=8.3, framealpha=0.92)

    fig.suptitle(r"Linear Decomposition of $\tilde{U}_{\mathrm{eff}}(c)$: Bulk Slope $\kappa_1$ and Excess Pinning Drop $\Delta\tilde{U}_{+1}$",
                 fontsize=15.0, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    vcd.save_figure(fig, "per_point_cos_potential_linear_decomposition_6panels", OUT_DIRS)
    plt.close(fig)

    # --- Figure 2: 2段階クロスオーバー (kappa_1 & Delta U_+1 vs R_c / xi) ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.4))

    # (A) 生の値: kappa_1 vs Delta U_+1
    axL.plot(df_out["rc_over_xi"], df_out["delta_U_plus1"], "-", color="#d62728", linewidth=2.2, alpha=0.8,
             label=r"Excess rail pinning $\Delta\tilde{U}_{+1}$", zorder=3)
    axL.plot(df_out["rc_over_xi"], df_out["kappa_1"], "--", color="#1f77b4", linewidth=2.2, alpha=0.85,
             label=r"Bulk alignment slope $\kappa_1$", zorder=3)

    for _, r in df_out.iterrows():
        yerr_du = [[r["delta_U_plus1"] - r["delta_U_plus1_ci_low"]], [r["delta_U_plus1_ci_high"] - r["delta_U_plus1"]]]
        yerr_k1 = [[r["kappa_1"] - r["kappa_1_ci_low"]], [r["kappa_1_ci_high"] - r["kappa_1"]]]
        axL.errorbar(r["rc_over_xi"], r["delta_U_plus1"], yerr=yerr_du, xerr=r["rc_over_xi_sem"],
                     fmt=r["marker"], color="#d62728", markersize=8.5, capsize=4, elinewidth=1.6,
                     markeredgecolor="k", markeredgewidth=0.7, zorder=5)
        axL.errorbar(r["rc_over_xi"], r["kappa_1"], yerr=yerr_k1, xerr=r["rc_over_xi_sem"],
                     fmt=r["marker"], color="#1f77b4", markersize=8.5, capsize=4, elinewidth=1.6,
                     markeredgecolor="k", markeredgewidth=0.7, zorder=4)

    X_LOCK = 45.0 * np.pi / 512.0  # ~0.28 (Higher-order collapse)
    X_BULK = 45.0 * np.pi / 128.0  # ~1.10 (Bulk alignment decay)

    axL.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    axL.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")
    axL.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    axL.set_xscale("log")
    axL.set_xlim(0.04, 3.0)
    axL.set_ylim(-0.08, 1.18)
    axL.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    axL.set_ylabel(r"Coupling / Pinning Depth ($\kappa_1,\ \Delta\tilde{U}_{+1}$)", fontsize=12.5)
    axL.set_title(r"(A) Two-Stage Crossover: $\Delta\tilde{U}_{+1}$ & $\kappa_1$ vs $R_c/\xi$", fontsize=13, fontweight="bold")
    axL.grid(True, which="both", linestyle=":", alpha=0.5)
    axL.legend(loc="upper right", fontsize=8.4, framealpha=0.92)

    # (B) 最小粒子で規格化した相対減衰曲線 (2段階の分離を明瞭化)
    du0 = float(df_out["delta_U_plus1"].iloc[0])
    k10 = float(df_out["kappa_1"].iloc[0])
    axR.plot(df_out["rc_over_xi"], df_out["delta_U_plus1"] / du0, "o-", color="#d62728",
             linewidth=2.3, markersize=8.5, markeredgecolor="k", markeredgewidth=0.7,
             label=r"Normalized pinning $\Delta\tilde{U}_{+1} / \Delta\tilde{U}_{+1}^{(0)}$")
    axR.plot(df_out["rc_over_xi"], df_out["kappa_1"] / k10, "s--", color="#1f77b4",
             linewidth=2.3, markersize=8.0, markeredgecolor="k", markeredgewidth=0.7,
             label=r"Normalized bulk alignment $\kappa_1 / \kappa_1^{(0)}$")

    for _, r in df_out.iterrows():
        axR.annotate(f"{r['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(r["rc_over_xi"], r["kappa_1"] / k10),
                     xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.8, color="#1f77b4", fontweight="bold")

    axR.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    axR.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")
    axR.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    axR.set_xscale("log")
    axR.set_xlim(0.04, 3.0)
    axR.set_ylim(-0.08, 1.30)
    axR.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    axR.set_ylabel(r"Normalized Strength (relative to $0.63\,\mu\mathrm{m}$)", fontsize=12.5)
    axR.set_title(r"(B) Decoupled Crossover Scales (Normalized)", fontsize=13, fontweight="bold")
    axR.grid(True, which="both", linestyle=":", alpha=0.5)
    axR.legend(loc="upper right", fontsize=8.4, framealpha=0.92)

    fig.tight_layout()
    vcd.save_figure(fig, "two_stage_crossover_k1_and_dU_plus1_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    print("Saved all linear decomposition & two-stage crossover figures successfully!")


if __name__ == "__main__":
    main()
