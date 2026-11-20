#!/usr/bin/env python3
"""
plot_gvm_potential.py

1次モーメント m_1 = <cos theta> と 2次モーメント m_2 = <cos 2theta> = 2<c^2> - 1 を算出し、
一般化フォン・ミーゼス (GvM) 分布
    P_GvM(theta) = exp(kappa_1 cos(theta) + kappa_2 cos(2*theta)) / Z(kappa_1, kappa_2)
が実験の (m_1, m_2) を厳密に再現するように (kappa_1, kappa_2) を連立決定（フィッティングなし）。

得られた (kappa_1, kappa_2) から理論有効ポテンシャル
    U_tilde_GvM(c) = -2*kappa_2*c^2 - kappa_1*c + (kappa_2 + ln(Z(kappa_1, kappa_2) / (2*pi)))
を描画し、実測 U_eff(c) = -ln(P(c)/P_iso(c)) と比較する新しいグラフを作成する。

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
from scipy.optimize import minimize
from scipy.special import i0

import verify_cos_decomposition as vcd
from calc_and_plot_kl_decomposition import solve_kappa1_from_m1, compute_kl_terms

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = vcd.ROOT_DIR
OUT_DIRS = [
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
    ROOT_DIR / "figure" / "per_point_v_dot_u",
]
HIST_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"

BINS_COS = np.linspace(-1.0, 1.0, 41)  # 40 bins, dc = 0.05
TH_GRID = np.linspace(-np.pi, np.pi, 720, endpoint=False)
DTH_GRID = 2.0 * np.pi / len(TH_GRID)
COS1_GRID = np.cos(TH_GRID)
COS2_GRID = np.cos(2.0 * TH_GRID)

X_LOCK = 45.0 * np.pi / 512.0  # ~0.28
X_BULK = 45.0 * np.pi / 128.0  # ~1.10


def compute_log_z_gvm(k1: float, k2: float) -> float:
    arg = k1 * COS1_GRID + k2 * COS2_GRID
    amax = float(np.max(arg))
    return float(amax + np.log(np.sum(np.exp(arg - amax)) * DTH_GRID))


def solve_gvm_from_moments(m1: float, m2: float):
    """
    厳密な最大エントロピー双対凸最適化により、
      <cos theta>_GvM = m1,  <cos 2theta>_GvM = m2
    を満たす唯一の (kappa_1, kappa_2) と log Z(kappa_1, kappa_2) を求める。
    """
    def obj_grad(params):
        k1, k2 = float(params[0]), float(params[1])
        arg = k1 * COS1_GRID + k2 * COS2_GRID
        amax = float(np.max(arg))
        w = np.exp(arg - amax)
        z_scaled = float(np.sum(w) * DTH_GRID)
        log_z = amax + np.log(z_scaled)
        p = w * DTH_GRID / z_scaled
        e1 = float(np.sum(p * COS1_GRID))
        e2 = float(np.sum(p * COS2_GRID))
        val = log_z - k1 * m1 - k2 * m2
        grad = np.array([e1 - m1, e2 - m2], dtype=float)
        return val, grad

    res = minimize(obj_grad, x0=np.array([2.0 * m1, 2.0 * m2]), jac=True, method="BFGS", options={"gtol": 1e-11})
    k1, k2 = float(res.x[0]), float(res.x[1])
    log_z = compute_log_z_gvm(k1, k2)
    return k1, k2, log_z


def u_eff_from_hist(hc_counts: np.ndarray, p_iso_k: np.ndarray, dc: float) -> np.ndarray:
    tot = float(hc_counts.sum())
    if tot <= 0:
        return np.full_like(p_iso_k, np.nan)
    p_c = hc_counts / (tot * dc)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_c > 0, -np.log(p_c / p_iso_k), np.nan)


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

    n_boot = 1000
    rng = np.random.default_rng(42)

    rows = []
    panel_cache = {}
    c_fine = np.linspace(-0.99, 0.99, 400)

    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df_rec[df_rec["bead_name"] == bname]
        if sub.empty:
            continue

        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        rc_xi_sem = float(sub["rc_over_xi"].sem()) if n_p > 1 else 0.0

        hc_arr = np.vstack(sub["hist_c"].values).astype(float)
        ht_arr = np.vstack(sub["hist_th"].values).astype(float)
        sumc_arr = sub["sum_c"].values.astype(float)
        sumc2_arr = sub["sum_c2"].values.astype(float)
        npts_arr = sub["n_pts"].values.astype(float)

        tot_n = float(npts_arr.sum())
        tot_sc = float(sumc_arr.sum())
        m1_obs = float(tot_sc / tot_n)
        c2_obs = float(sumc2_arr.sum() / tot_n)
        m2_obs = float(2.0 * c2_obs - 1.0)

        k1_obs, k2_obs, log_z_obs = solve_gvm_from_moments(m1_obs, m2_obs)
        offset_obs = k2_obs + log_z_obs - np.log(2.0 * np.pi)
        u_gvm_fine = -2.0 * k2_obs * (c_fine ** 2) - k1_obs * c_fine + offset_obs
        u_gvm_bins = -2.0 * k2_obs * (c_centers ** 2) - k1_obs * c_centers + offset_obs

        u_obs = u_eff_from_hist(hc_arr.sum(axis=0), p_iso_k, dc)
        valid_u = np.isfinite(u_obs)
        rmse_u = float(np.sqrt(np.mean((u_obs[valid_u] - u_gvm_bins[valid_u]) ** 2)))

        # 3項 KL 分解: D_KL(P || P_iso) = D_KL^(1) + D_KL^(2) + Delta D_>=3
        kl_obs = compute_kl_terms(ht_arr.sum(axis=0), tot_sc, tot_n)
        d_tot_obs = float(kl_obs["D_KL"])
        d1_obs = float(kl_obs["D_KL_1"])
        d_gvm_obs = float(k1_obs * m1_obs + k2_obs * m2_obs - (log_z_obs - np.log(2.0 * np.pi)))
        d2_obs = max(0.0, d_gvm_obs - d1_obs)
        d3_obs = max(0.0, d_tot_obs - d_gvm_obs)

        # クラスターブートストラップ 95% CI
        boot_u = np.full((n_boot, len(c_centers)), np.nan)
        boot_m1, boot_m2, boot_k1, boot_k2 = [], [], [], []
        boot_d1, boot_d2, boot_d3, boot_dtot, boot_rmse = [], [], [], [], []
        for bi in range(n_boot):
            idx = rng.integers(0, n_p, size=n_p)
            nb = float(npts_arr[idx].sum())
            if nb <= 0:
                continue
            sc_b = float(sumc_arr[idx].sum())
            ub = u_eff_from_hist(hc_arr[idx].sum(axis=0), p_iso_k, dc)
            boot_u[bi] = ub
            m1_b = float(sc_b / nb)
            m2_b = float(2.0 * (sumc2_arr[idx].sum() / nb) - 1.0)
            k1_b, k2_b, lz_b = solve_gvm_from_moments(m1_b, m2_b)
            boot_m1.append(m1_b)
            boot_m2.append(m2_b)
            boot_k1.append(k1_b)
            boot_k2.append(k2_b)

            off_b = k2_b + lz_b - np.log(2.0 * np.pi)
            ug_b = -2.0 * k2_b * (c_centers ** 2) - k1_b * c_centers + off_b
            vu = np.isfinite(ub)
            if np.any(vu):
                boot_rmse.append(float(np.sqrt(np.mean((ub[vu] - ug_b[vu]) ** 2))))

            kl_b = compute_kl_terms(ht_arr[idx].sum(axis=0), sc_b, nb)
            if kl_b is not None:
                dt_b = float(kl_b["D_KL"])
                d1_b = float(kl_b["D_KL_1"])
                dg_b = float(k1_b * m1_b + k2_b * m2_b - (lz_b - np.log(2.0 * np.pi)))
                boot_dtot.append(dt_b)
                boot_d1.append(d1_b)
                boot_d2.append(max(0.0, dg_b - d1_b))
                boot_d3.append(max(0.0, dt_b - dg_b))

        panel_cache[bname] = {
            "u_obs": u_obs,
            "u_lo": np.nanpercentile(boot_u, 2.5, axis=0),
            "u_hi": np.nanpercentile(boot_u, 97.5, axis=0),
            "u_gvm_fine": u_gvm_fine,
            "m1": m1_obs,
            "m2": m2_obs,
            "k1": k1_obs,
            "k2": k2_obs,
            "offset": offset_obs,
            "rmse_u": rmse_u,
            "rc_xi": rc_xi,
        }

        rows.append({
            "bead_name": bname,
            "diameter_um": float(b_item["diameter_um"]),
            "label": b_item["label"],
            "rc_over_xi": rc_xi,
            "rc_over_xi_sem": rc_xi_sem,
            "n_particles": n_p,
            "m1_cos_theta": m1_obs,
            "m1_ci_low": float(np.percentile(boot_m1, 2.5)),
            "m1_ci_high": float(np.percentile(boot_m1, 97.5)),
            "m2_cos_2theta": m2_obs,
            "m2_ci_low": float(np.percentile(boot_m2, 2.5)),
            "m2_ci_high": float(np.percentile(boot_m2, 97.5)),
            "kappa_1_gvm": k1_obs,
            "kappa_1_ci_low": float(np.percentile(boot_k1, 2.5)),
            "kappa_1_ci_high": float(np.percentile(boot_k1, 97.5)),
            "kappa_2_gvm": k2_obs,
            "kappa_2_ci_low": float(np.percentile(boot_k2, 2.5)),
            "kappa_2_ci_high": float(np.percentile(boot_k2, 97.5)),
            "log_Z_over_2pi": float(log_z_obs - np.log(2.0 * np.pi)),
            "gvm_offset": float(offset_obs),
            "rmse_U_obs_vs_GvM": rmse_u,
            "rmse_ci_low": float(np.percentile(boot_rmse, 2.5)),
            "rmse_ci_high": float(np.percentile(boot_rmse, 97.5)),
            "D_KL_total": d_tot_obs,
            "D_KL_total_ci_low": float(np.percentile(boot_dtot, 2.5)),
            "D_KL_total_ci_high": float(np.percentile(boot_dtot, 97.5)),
            "D_KL_1_polar": d1_obs,
            "D_KL_1_ci_low": float(np.percentile(boot_d1, 2.5)),
            "D_KL_1_ci_high": float(np.percentile(boot_d1, 97.5)),
            "D_KL_2_nematic": d2_obs,
            "D_KL_2_ci_low": float(np.percentile(boot_d2, 2.5)),
            "D_KL_2_ci_high": float(np.percentile(boot_d2, 97.5)),
            "Delta_D_ge3_residual": d3_obs,
            "Delta_D_ge3_ci_low": float(np.percentile(boot_d3, 2.5)),
            "Delta_D_ge3_ci_high": float(np.percentile(boot_d3, 97.5)),
        })

    df_out = pd.DataFrame(rows).sort_values("rc_over_xi").reset_index(drop=True)
    for d in OUT_DIRS:
        df_out.to_csv(d / "gvm_moments_and_potential_summary.csv", index=False)

    print("=== Zero-Fitting GvM Moment Matching & 3-Term KL Decomposition ===")
    print(df_out[["bead_name", "diameter_um", "rc_over_xi", "rmse_U_obs_vs_GvM",
                  "D_KL_total", "D_KL_1_polar", "D_KL_2_nematic", "Delta_D_ge3_residual"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Fig 1: 6パネル 実測 U_eff(c) vs ゼロフィッティング理論線 U_tilde_GvM(c)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    for idx, b_item in enumerate(vcd.BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        pc = panel_cache[bname]

        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
        ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.9, zorder=1)

        valid = np.isfinite(pc["u_obs"])
        ax.plot(c_centers[valid], pc["u_obs"][valid], color=col, marker=b_item["marker"],
                markersize=5.0, linewidth=1.9, label=r"Observed $\tilde{U}_{\mathrm{eff}}(c)$", zorder=4)
        ax.fill_between(c_centers[valid], pc["u_lo"][valid], pc["u_hi"][valid],
                        color=col, alpha=0.20, label=r"95% CI (Bootstrap)", zorder=2)

        ax.plot(c_fine, pc["u_gvm_fine"], color="#111111", linestyle="--", linewidth=2.2,
                label=rf"GvM theory ($\kappa_1={pc['k1']:.2f}, \kappa_2={pc['k2']:.2f}$)", zorder=5)

        ax.text(
            0.03, 0.05,
            rf"$\langle\cos\theta\rangle={pc['m1']:.3f}$" + "\n" + rf"$\langle\cos 2\theta\rangle={pc['m2']:.3f}$",
            transform=ax.transAxes, fontsize=9.2, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            zorder=6,
        )

        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.45, 1.05)
        ax.set_title(rf"{lbl}  ($R_c/\xi={pc['rc_xi']:.2f}$, $\mathrm{{RMSE}}={pc['rmse_u']:.3f}$)",
                     fontsize=13.0, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel(r"Per-Point Alignment $c = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=12.5)
        if idx % 3 == 0:
            ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$", fontsize=12.5)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="upper right", fontsize=8.4, framealpha=0.92)

    fig.suptitle(r"Zero-Fitting GvM Prediction $\tilde{U}_{\mathrm{GvM}}(c) = -2\kappa_2 c^2 - \kappa_1 c + (\kappa_2 + \ln\frac{Z}{2\pi})$ from $(\langle\cos\theta\rangle, \langle\cos 2\theta\rangle)$",
                 fontsize=13.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    vcd.save_figure(fig, "per_point_cos_potential_gvm_6panels", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: 2パネル (A) <cos theta>, <cos 2theta> vs R_c/xi, (B) kappa_1, kappa_2 vs R_c/xi
    # ------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 5.5))

    # (A) m1 = <cos theta> and m2 = <cos 2theta>
    axL.plot(df_out["rc_over_xi"], df_out["m1_cos_theta"], "s-", color="#1f77b4", linewidth=2.2,
             markersize=8.2, markeredgecolor="k", markeredgewidth=0.7, label=r"1st moment $\langle \cos\theta \rangle$", zorder=4)
    axL.plot(df_out["rc_over_xi"], df_out["m2_cos_2theta"], "^-", color="#d62728", linewidth=2.2,
             markersize=8.5, markeredgecolor="k", markeredgewidth=0.7, label=r"2nd moment $\langle \cos 2\theta \rangle$", zorder=5)

    for _, r in df_out.iterrows():
        y1 = [[max(0.0, r["m1_cos_theta"] - r["m1_ci_low"])], [max(0.0, r["m1_ci_high"] - r["m1_cos_theta"])]]
        y2 = [[max(0.0, r["m2_cos_2theta"] - r["m2_ci_low"])], [max(0.0, r["m2_ci_high"] - r["m2_cos_2theta"])]]
        axL.errorbar(r["rc_over_xi"], r["m1_cos_theta"], yerr=y1, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#1f77b4", capsize=3.8, elinewidth=1.5, zorder=3)
        axL.errorbar(r["rc_over_xi"], r["m2_cos_2theta"], yerr=y2, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#d62728", capsize=3.8, elinewidth=1.5, zorder=3)
        axL.annotate(rf"{r['diameter_um']:.2g} $\mu$m", xy=(r["rc_over_xi"], r["m1_cos_theta"]),
                     xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.8, color="#1f77b4", fontweight="bold")

    axL.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.85,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$")
    axL.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.6, alpha=0.85,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$")
    axL.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    axL.set_xscale("log")
    axL.set_xlim(0.045, 3.2)
    axL.set_ylim(-0.20, 0.50)
    axL.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=12.5)
    axL.set_ylabel(r"Trigonometric Moments $\langle\cos\theta\rangle,\ \langle\cos 2\theta\rangle$", fontsize=12.5)
    axL.set_title(r"(A) 1st & 2nd Moments vs $R_c/\xi$", fontsize=13.0, fontweight="bold")
    axL.grid(True, which="both", linestyle=":", alpha=0.55)
    axL.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    # (B) GvM (kappa_1, kappa_2)
    axR.plot(df_out["rc_over_xi"], df_out["kappa_1_gvm"], "s-", color="#1f77b4", linewidth=2.2,
             markersize=8.2, markeredgecolor="k", markeredgewidth=0.7, label=r"Polar coupling $\kappa_1$ (GvM)", zorder=4)
    axR.plot(df_out["rc_over_xi"], df_out["kappa_2_gvm"], "^-", color="#d62728", linewidth=2.2,
             markersize=8.5, markeredgecolor="k", markeredgewidth=0.7, label=r"Nematic coupling $\kappa_2$ (GvM)", zorder=5)

    for _, r in df_out.iterrows():
        y1 = [[max(0.0, r["kappa_1_gvm"] - r["kappa_1_ci_low"])], [max(0.0, r["kappa_1_ci_high"] - r["kappa_1_gvm"])]]
        y2 = [[max(0.0, r["kappa_2_gvm"] - r["kappa_2_ci_low"])], [max(0.0, r["kappa_2_ci_high"] - r["kappa_2_gvm"])]]
        axR.errorbar(r["rc_over_xi"], r["kappa_1_gvm"], yerr=y1, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#1f77b4", capsize=3.8, elinewidth=1.5, zorder=3)
        axR.errorbar(r["rc_over_xi"], r["kappa_2_gvm"], yerr=y2, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#d62728", capsize=3.8, elinewidth=1.5, zorder=3)
        axR.annotate(rf"{r['diameter_um']:.2g} $\mu$m", xy=(r["rc_over_xi"], r["kappa_1_gvm"]),
                     xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.8, color="#1f77b4", fontweight="bold")

    axR.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.85,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$")
    axR.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.6, alpha=0.85,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$")
    axR.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    axR.set_xscale("log")
    axR.set_xlim(0.045, 3.2)
    axR.set_ylim(-0.35, 1.05)
    axR.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=12.5)
    axR.set_ylabel(r"Moment-Matched GvM Parameters $\kappa_1,\ \kappa_2$", fontsize=12.5)
    axR.set_title(r"(B) GvM Couplings $(\kappa_1, \kappa_2)$ from Moments vs $R_c/\xi$", fontsize=13.0, fontweight="bold")
    axR.grid(True, which="both", linestyle=":", alpha=0.55)
    axR.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    fig.suptitle(r"Two-Moment Maximum-Entropy (GvM) Characterization across Cargo Sizes (95% Cluster Bootstrap CI)",
                 fontsize=14.2, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    vcd.save_figure(fig, "cos2theta_and_gvm_k1_k2_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3: 貨物サイズごとのゼロフィッティング GvM ポテンシャル RMSE vs R_c / xi
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    ax.plot(df_out["rc_over_xi"], df_out["rmse_U_obs_vs_GvM"], "-", color="#444444", linewidth=1.8, alpha=0.65, zorder=2)

    offsets_rmse = {
        "beads06um": (10, 4, "left"),
        "beads1um":  (10, 4, "left"),
        "beads3um":  (10, 6, "left"),
        "beads7um":  (10, -8, "left"),
        "beads5um":  (-10, -10, "right"),
        "beads20um": (0, 18, "center"),
    }
    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        r = df_out[df_out["bead_name"] == bname].iloc[0]
        yv = float(r["rmse_U_obs_vs_GvM"])
        yerr = [[max(0.0, yv - r["rmse_ci_low"])], [max(0.0, r["rmse_ci_high"] - yv)]]
        ax.errorbar(r["rc_over_xi"], yv, yerr=yerr, xerr=r["rc_over_xi_sem"],
                    fmt=b_item["marker"], color=b_item["color"], markersize=9.2,
                    markeredgecolor="#222222", markeredgewidth=0.9,
                    ecolor=b_item["color"], elinewidth=1.7, capsize=3.8, zorder=4)
        dx, dy, ha = offsets_rmse[bname]
        ax.annotate(rf"{r['diameter_um']:.2g} $\mu$m", xy=(r["rc_over_xi"], yv),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=10.2, fontweight="bold", color=b_item["color"], zorder=5)

    ax.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85,
               label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    ax.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85,
               label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")
    ax.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.045, 3.2)
    ax.set_ylim(-0.01, 0.27)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13.0)
    ax.set_ylabel(r"Potential Residual $\mathrm{RMSE}(\tilde{U}_{\mathrm{obs}} - \tilde{U}_{\mathrm{GvM}})$", fontsize=12.5)
    ax.set_title(r"Zero-Fitting GvM Potential Residual ($\mathrm{RMSE}$) vs $R_c / \xi$ (95% Bootstrap CI)",
                 fontsize=13.0, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=9.2, framealpha=0.92)
    fig.tight_layout()
    vcd.save_figure(fig, "gvm_potential_rmse_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 4: 3項 KL ダイバージェンス分解
    #   D_KL(P||P_iso) = D_KL^(1) + D_KL^(2) + Delta D_>=3
    # ------------------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.6, 5.7))

    # (A) R_c / xi に対する 3 成分の減衰
    axA.plot(df_out["rc_over_xi"], df_out["D_KL_total"], "o:", color="#333333", linewidth=1.8, markersize=6.5,
             label=r"Total $D_{\mathrm{KL}}(P \parallel P_{\mathrm{iso}})$", zorder=2)
    axA.plot(df_out["rc_over_xi"], df_out["D_KL_1_polar"], "s-", color="#1f77b4", linewidth=2.2, markersize=8.2,
             markeredgecolor="k", markeredgewidth=0.7,
             label=r"$D_{\mathrm{KL}}^{(1)} = D_{\mathrm{KL}}(P_{\mathrm{vM}} \parallel P_{\mathrm{iso}})$ (1st polar)", zorder=4)
    axA.plot(df_out["rc_over_xi"], df_out["D_KL_2_nematic"], "^-", color="#d95f02", linewidth=2.2, markersize=8.5,
             markeredgecolor="k", markeredgewidth=0.7,
             label=r"$D_{\mathrm{KL}}^{(2)} = D_{\mathrm{KL}}(P_{\mathrm{GvM}} \parallel P_{\mathrm{vM}})$ (2nd nematic)", zorder=5)
    axA.plot(df_out["rc_over_xi"], df_out["Delta_D_ge3_residual"], "D-", color="#7570b3", linewidth=2.0, markersize=7.5,
             markeredgecolor="k", markeredgewidth=0.7,
             label=r"$\Delta D_{\geq 3} = D_{\mathrm{KL}}(P \parallel P_{\mathrm{GvM}})$ ($\geq 3$rd residual)", zorder=6)

    for _, r in df_out.iterrows():
        y1 = [[max(0.0, r["D_KL_1_polar"] - r["D_KL_1_ci_low"])], [max(0.0, r["D_KL_1_ci_high"] - r["D_KL_1_polar"])]]
        y2 = [[max(0.0, r["D_KL_2_nematic"] - r["D_KL_2_ci_low"])], [max(0.0, r["D_KL_2_ci_high"] - r["D_KL_2_nematic"])]]
        y3 = [[max(0.0, r["Delta_D_ge3_residual"] - r["Delta_D_ge3_ci_low"])], [max(0.0, r["Delta_D_ge3_ci_high"] - r["Delta_D_ge3_residual"])]]
        axA.errorbar(r["rc_over_xi"], r["D_KL_1_polar"], yerr=y1, fmt="none", ecolor="#1f77b4", capsize=3.5, elinewidth=1.4, zorder=3)
        axA.errorbar(r["rc_over_xi"], r["D_KL_2_nematic"], yerr=y2, fmt="none", ecolor="#d95f02", capsize=3.5, elinewidth=1.4, zorder=3)
        axA.errorbar(r["rc_over_xi"], r["Delta_D_ge3_residual"], yerr=y3, fmt="none", ecolor="#7570b3", capsize=3.5, elinewidth=1.4, zorder=3)
        axA.annotate(rf"{r['diameter_um']:.2g} $\mu$m", xy=(r["rc_over_xi"], r["D_KL_total"]),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.8, fontweight="bold")

    axA.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.75,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$")
    axA.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.6, alpha=0.8,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$")
    axA.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    axA.set_xscale("log")
    axA.set_xlim(0.045, 3.2)
    axA.set_ylim(-0.008, 0.22)
    axA.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=12.5)
    axA.set_ylabel(r"KL Divergence [nats]", fontsize=12.5)
    axA.set_title(r"(A) 3-Term Orthogonal KL Decomposition vs $R_c/\xi$", fontsize=13.0, fontweight="bold")
    axA.grid(True, which="both", linestyle=":", alpha=0.55)
    axA.legend(loc="upper right", fontsize=8.1, framealpha=0.92)

    # (B) 積み上げ棒グラフ (粒子サイズ別内訳)
    x_pos = np.arange(len(df_out))
    d1_vals = df_out["D_KL_1_polar"].values
    d2_vals = df_out["D_KL_2_nematic"].values
    d3_vals = df_out["Delta_D_ge3_residual"].values
    axB.bar(x_pos, d1_vals, width=0.58, color="#1f77b4", edgecolor="k", linewidth=0.7,
            label=r"$D_{\mathrm{KL}}^{(1)}$ (1st polar)")
    axB.bar(x_pos, d2_vals, width=0.58, bottom=d1_vals, color="#d95f02", edgecolor="k", linewidth=0.7,
            label=r"$D_{\mathrm{KL}}^{(2)}$ (2nd nematic)")
    axB.bar(x_pos, d3_vals, width=0.58, bottom=d1_vals + d2_vals, color="#7570b3", edgecolor="k", linewidth=0.7,
            label=r"$\Delta D_{\geq 3}$ ($\geq 3$rd residual)")

    for i, r in df_out.iterrows():
        tot_i = float(r["D_KL_total"])
        if tot_i > 0.015:
            p1 = 100.0 * r["D_KL_1_polar"] / tot_i
            p2 = 100.0 * r["D_KL_2_nematic"] / tot_i
            p3 = 100.0 * r["Delta_D_ge3_residual"] / tot_i
            axB.text(i, tot_i + 0.004, f"{p1:.0f}/{p2:.0f}/{p3:.0f}%",
                     ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#222222")

    x_labels = [rf"{r['diameter_um']:.2g} $\mu$m" + "\n" + rf"({r['rc_over_xi']:.2f})" for _, r in df_out.iterrows()]
    axB.set_xticks(x_pos)
    axB.set_xticklabels(x_labels, fontsize=9.8)
    axB.set_ylim(0.0, 0.20)
    axB.set_xlabel(r"Cargo Diameter ($R_c/\xi$)", fontsize=12.5)
    axB.set_ylabel(r"KL Divergence Contributions [nats]", fontsize=12.5)
    axB.set_title(r"(B) Mode-by-Mode Information Budget ($D_{\mathrm{KL}}^{(1)} / D_{\mathrm{KL}}^{(2)} / \Delta D_{\geq 3}$)",
                  fontsize=12.8, fontweight="bold")
    axB.grid(True, axis="y", linestyle=":", alpha=0.55)
    axB.legend(loc="upper right", fontsize=9.2, framealpha=0.92)

    fig.suptitle(r"Pythagorean 3-Term KL Decomposition: $D_{\mathrm{KL}}(P\parallel P_{\mathrm{iso}}) = D_{\mathrm{KL}}^{(1)} + D_{\mathrm{KL}}^{(2)} + \Delta D_{\geq 3}$ ($B=36$)",
                 fontsize=13.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    vcd.save_figure(fig, "kl_3term_decomposition_gvm_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    print("Saved all GvM figures (6panels, moments/k1/k2, RMSE, 3-term KL decomposition) and CSV!")


if __name__ == "__main__":
    main()
