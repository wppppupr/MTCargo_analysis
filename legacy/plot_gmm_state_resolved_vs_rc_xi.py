#!/usr/bin/env python3
"""
plot_gmm_state_resolved_vs_rc_xi.py

Run / Tumble 状態別の相関長 xi (gmm_xi_and_rc_over_xi_summary_k2.csv) から求めた
状態別 R_c / xi を横軸にとり、
  - 0.63 um Run
  - 0.63 um Tumble
  - 1.18 um Run
  - 1.18 um Tumble
をそれぞれ独立した系列として、大粒子 (3.37, 7.24, 5.00, 20.0 um) とともにプロットする
新しいグラフを作成する（既存グラフは一切変更しない）。

出力先:
  - figure/gmm_1d/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/gmm_1d/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import verify_cos_decomposition as vcd
from gmm_cargo_analysis import find_default_root


def extract_stratonovich_segments(df_bead: pd.DataFrame):
    particles = []
    for pid, pg in df_bead.groupby("pid"):
        pg = pg.sort_values("frame")
        frm = pg["frame"].values
        ux, uy = pg["ux"].values, pg["uy"].values
        dx, dy = pg["dx_um"].values, pg["dy_um"].values
        splits = np.where(np.diff(frm) != 1)[0] + 1
        segs = []
        for idx_arr in np.split(np.arange(len(frm)), splits):
            if len(idx_arr) >= 2:
                i0, i1 = idx_arr[:-1], idx_arr[1:]
                dJ = 0.5 * (ux[i0] + ux[i1]) * dx[i0] + 0.5 * (uy[i0] + uy[i1]) * dy[i0]
                segs.append(dJ)
        if segs:
            particles.append({"pid": pid, "xi": float(pg["xi"].iloc[0]), "segs": segs})
    return particles


def calc_q_and_boot(particles, k_frames: int = 2, n_boot: int = 1000, seed: int = 44):
    tau_s = k_frames * 4.0
    p_arrs = []
    for p in particles:
        vals = [np.cumsum(np.concatenate([[0.0], s]))[k_frames:] - np.cumsum(np.concatenate([[0.0], s]))[:-k_frames]
                for s in p["segs"] if len(s) >= k_frames]
        p_arrs.append(np.concatenate(vals) if vals else np.array([], dtype=float))
    valid_p = [a for a in p_arrs if len(a) > 0]
    j_all = np.concatenate(valid_p)
    q_obs = float(2.0 * (np.mean(j_all) ** 2) / np.var(j_all, ddof=1))
    rate_obs = q_obs / tau_s
    rng = np.random.default_rng(seed)
    n_p = len(p_arrs)
    bq, br = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        parts = [p_arrs[i] for i in idx if len(p_arrs[i]) > 0]
        if not parts:
            continue
        jb = np.concatenate(parts)
        vb = float(np.var(jb, ddof=1)) if len(jb) > 1 else 0.0
        if vb > 0:
            qb = float(2.0 * (float(np.mean(jb)) ** 2) / vb)
            bq.append(qb)
            br.append(qb / tau_s)
    return {
        "Q": q_obs, "Q_ci_low": float(np.percentile(bq, 2.5)), "Q_ci_high": float(np.percentile(bq, 97.5)),
        "rate_s_inv": rate_obs, "rate_ci_low": float(np.percentile(br, 2.5)), "rate_ci_high": float(np.percentile(br, 97.5)),
    }

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = find_default_root()
OUT_DIRS = [CURRENT_DIR / "figure" / "gmm_1d"]
if ROOT_DIR != CURRENT_DIR and ROOT_DIR.exists():
    OUT_DIRS.append(ROOT_DIR / "figure" / "gmm_1d")

XI_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "gmm_xi_and_rc_over_xi_summary_k2.csv"
GMM_GVM_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "gmm_run_tumble_gvm_kl_summary.csv"
GMM_TUR_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "gmm_run_tumble_tur_summary.csv"
ALL_GVM_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "gvm_moments_and_potential_summary.csv"
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"

X_LOCK = 45.0 * np.pi / 512.0  # ~0.28
X_BULK = 45.0 * np.pi / 128.0  # ~1.10

SERIES_STYLE = {
    ("beads06um", "Run"):    {"label": r"$0.63\,\mu\mathrm{m}$ Run",    "color": "#00796b", "marker": "^"},
    ("beads06um", "Tumble"): {"label": r"$0.63\,\mu\mathrm{m}$ Tumble", "color": "#e65100", "marker": "v"},
    ("beads1um",  "Run"):    {"label": r"$1.18\,\mu\mathrm{m}$ Run",    "color": "#1565c0", "marker": "o"},
    ("beads1um",  "Tumble"): {"label": r"$1.18\,\mu\mathrm{m}$ Tumble", "color": "#c62828", "marker": "s"},
}


def build_combined_table() -> pd.DataFrame:
    df_xi = pd.read_csv(XI_CSV)
    df_xi_w = df_xi[df_xi["source"] == "angular_correlation_w"].copy()

    df_gvm_st = pd.read_csv(GMM_GVM_CSV)
    df_tur_st = pd.read_csv(GMM_TUR_CSV)
    df_tur_8 = df_tur_st[df_tur_st["tau_s"] == 8.0].copy()

    df_gvm_all = pd.read_csv(ALL_GVM_CSV)

    # 大粒子 TUR を contrast_inst_cache.pkl から直接算出 (既存ファイル非変更)
    df_inst = pd.read_pickle(INST_CACHE)
    df_inst["pid"] = df_inst["exp_dir"].astype(str) + "_" + df_inst["particle"].astype(str)
    f_norm = np.hypot(df_inst["flow_u"].values, df_inst["flow_v"].values)
    vmask = (f_norm > 0) & np.isfinite(df_inst["cargo_dx"].values) & np.isfinite(df_inst["cargo_dy"].values)
    df_inst = df_inst[vmask].copy()
    f_norm = np.hypot(df_inst["flow_u"].values, df_inst["flow_v"].values)
    df_inst["ux"] = df_inst["flow_u"].values / f_norm
    df_inst["uy"] = df_inst["flow_v"].values / f_norm
    df_inst["dx_um"] = df_inst["cargo_dx"].values * 0.11
    df_inst["dy_um"] = df_inst["cargo_dy"].values * 0.11

    tur_large_map = {}
    for bname in ["beads3um", "beads7um", "beads5um", "beads20um"]:
        sub_b = df_inst[df_inst["bead_name"] == bname]
        ps = extract_stratonovich_segments(sub_b)
        res8 = calc_q_and_boot(ps, k_frames=2, n_boot=1000, seed=44)
        tur_large_map[bname] = {
            "xi_mean_um": float(np.mean([p["xi"] for p in ps])),
            **res8,
        }

    rows = []
    # 1. 小粒子 4 系列 (0.63 Run/Tumble, 1.18 Run/Tumble)
    for (bname, mode), st_info in SERIES_STYLE.items():
        rx = df_xi_w[(df_xi_w["bead_name"] == bname) & (df_xi_w["mode"] == mode)].iloc[0]
        rg = df_gvm_st[(df_gvm_st["bead_name"] == bname) & (df_gvm_st["state_label"] == mode)].iloc[0]
        rt = df_tur_8[(df_tur_8["bead_name"] == bname) & (df_tur_8["state_label"] == mode)].iloc[0]

        rc_xi = float(rx["rc_over_xi"])
        rows.append({
            "series_key": f"{bname}_{mode}",
            "series_label": st_info["label"],
            "is_small_state": True,
            "bead_name": bname,
            "diameter_um": float(rg["diameter_um"]),
            "mode": mode,
            "xi_um": float(rx["xi_um"]),
            "rc_over_xi": rc_xi,
            "rc_xi_err_lo": max(0.0, rc_xi - float(rx["rc_xi_ci_lo"])),
            "rc_xi_err_hi": max(0.0, float(rx["rc_xi_ci_hi"]) - rc_xi),
            "m1": float(rg["m1_cos_theta"]),
            "m1_lo": float(rg["m1_ci_low"]),
            "m1_hi": float(rg["m1_ci_high"]),
            "m2": float(rg["m2_cos_2theta"]),
            "m2_lo": float(rg["m2_ci_low"]),
            "m2_hi": float(rg["m2_ci_high"]),
            "k1": float(rg["kappa_1_gvm"]),
            "k1_lo": float(rg["kappa_1_ci_low"]),
            "k1_hi": float(rg["kappa_1_ci_high"]),
            "k2": float(rg["kappa_2_gvm"]),
            "k2_lo": float(rg["kappa_2_ci_low"]),
            "k2_hi": float(rg["kappa_2_ci_high"]),
            "rmse": float(rg["rmse_U_obs_vs_GvM"]),
            "rmse_lo": float(rg["rmse_ci_low"]),
            "rmse_hi": float(rg["rmse_ci_high"]),
            "d1": float(rg["D_KL_1_polar"]),
            "d1_lo": float(rg["D_KL_1_ci_low"]),
            "d1_hi": float(rg["D_KL_1_ci_high"]),
            "d2": float(rg["D_KL_2_nematic"]),
            "d2_lo": float(rg["D_KL_2_ci_low"]),
            "d2_hi": float(rg["D_KL_2_ci_high"]),
            "d3": float(rg["Delta_D_ge3_residual"]),
            "d3_lo": float(rg["Delta_D_ge3_ci_low"]),
            "d3_hi": float(rg["Delta_D_ge3_ci_high"]),
            "Q_8s": float(rt["Q"]),
            "Q_8s_lo": float(rt["Q_ci_low"]),
            "Q_8s_hi": float(rt["Q_ci_high"]),
            "rate_8s": float(rt["rate_TUR_s_inv"]),
            "rate_8s_lo": float(rt["Q_ci_low"]) / 8.0,
            "rate_8s_hi": float(rt["Q_ci_high"]) / 8.0,
        })

    # 2. 大粒子参照系列 (3.37, 7.24, 5.00, 20.0 um All)
    for bname in ["beads3um", "beads7um", "beads5um", "beads20um"]:
        rg = df_gvm_all[df_gvm_all["bead_name"] == bname].iloc[0]
        rt = tur_large_map[bname]
        rc_xi = float(rg["rc_over_xi"])
        rc_sem = float(rg["rc_over_xi_sem"])
        rows.append({
            "series_key": f"{bname}_All",
            "series_label": "Large cargos (All)",
            "is_small_state": False,
            "bead_name": bname,
            "diameter_um": float(rg["diameter_um"]),
            "mode": "All",
            "xi_um": float(rt["xi_mean_um"]),
            "rc_over_xi": rc_xi,
            "rc_xi_err_lo": rc_sem,
            "rc_xi_err_hi": rc_sem,
            "m1": float(rg["m1_cos_theta"]),
            "m1_lo": float(rg["m1_ci_low"]),
            "m1_hi": float(rg["m1_ci_high"]),
            "m2": float(rg["m2_cos_2theta"]),
            "m2_lo": float(rg["m2_ci_low"]),
            "m2_hi": float(rg["m2_ci_high"]),
            "k1": float(rg["kappa_1_gvm"]),
            "k1_lo": float(rg["kappa_1_ci_low"]),
            "k1_hi": float(rg["kappa_1_ci_high"]),
            "k2": float(rg["kappa_2_gvm"]),
            "k2_lo": float(rg["kappa_2_ci_low"]),
            "k2_hi": float(rg["kappa_2_ci_high"]),
            "rmse": float(rg["rmse_U_obs_vs_GvM"]),
            "rmse_lo": float(rg["rmse_ci_low"]),
            "rmse_hi": float(rg["rmse_ci_high"]),
            "d1": float(rg["D_KL_1_polar"]),
            "d1_lo": float(rg["D_KL_1_ci_low"]),
            "d1_hi": float(rg["D_KL_1_ci_high"]),
            "d2": float(rg["D_KL_2_nematic"]),
            "d2_lo": float(rg["D_KL_2_ci_low"]),
            "d2_hi": float(rg["D_KL_2_ci_high"]),
            "d3": float(rg["Delta_D_ge3_residual"]),
            "d3_lo": float(rg["Delta_D_ge3_ci_low"]),
            "d3_hi": float(rg["Delta_D_ge3_ci_high"]),
            "Q_8s": float(rt["Q"]),
            "Q_8s_lo": float(rt["Q_ci_low"]),
            "Q_8s_hi": float(rt["Q_ci_high"]),
            "rate_8s": float(rt["rate_s_inv"]),
            "rate_8s_lo": float(rt["rate_ci_low"]),
            "rate_8s_hi": float(rt["rate_ci_high"]),
        })

    return pd.DataFrame(rows)


def plot_panel(
    ax,
    df_all: pd.DataFrame,
    y_col: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    title: str,
    ylim: tuple,
    logy: bool = False,
    legend_loc: str = "upper right",
    show_legend: bool = True,
):
    df_large = df_all[~df_all["is_small_state"]].sort_values("rc_over_xi")
    ax.plot(df_large["rc_over_xi"], df_large[y_col], "D--", color="#607d8b",
            linewidth=1.7, markersize=7.5, markeredgecolor="#222222", markeredgewidth=0.7,
            alpha=0.85, label="Large cargos (All)", zorder=3)
    for _, r in df_large.iterrows():
        yv = float(r[y_col])
        lo_v = max(1e-5, float(r[lo_col])) if logy else float(r[lo_col])
        hi_v = float(r[hi_col])
        yerr = [[max(0.0, yv - lo_v)], [max(0.0, hi_v - yv)]]
        xerr = [[float(r["rc_xi_err_lo"])], [float(r["rc_xi_err_hi"])]]
        ax.errorbar(r["rc_over_xi"], yv, yerr=yerr, xerr=xerr, fmt="none",
                    ecolor="#607d8b", elinewidth=1.4, capsize=3.2, alpha=0.8, zorder=2)
        dx, dy = (0, 10) if r["bead_name"] != "beads5um" else (0, -14)
        ax.annotate(rf"{r['diameter_um']:.2g} $\mu$m", xy=(r["rc_over_xi"], yv),
                    xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                    fontsize=8.3, color="#455a64", fontweight="bold", zorder=6)

    # 4 つの Run/Tumble 系列
    df_small = df_all[df_all["is_small_state"]]
    for (bname, mode), st_info in SERIES_STYLE.items():
        r = df_small[(df_small["bead_name"] == bname) & (df_small["mode"] == mode)].iloc[0]
        yv = float(r[y_col])
        lo_v = max(1e-5, float(r[lo_col])) if logy else float(r[lo_col])
        hi_v = float(r[hi_col])
        yerr = [[max(0.0, yv - lo_v)], [max(0.0, hi_v - yv)]]
        xerr = [[float(r["rc_xi_err_lo"])], [float(r["rc_xi_err_hi"])]]
        ax.errorbar(
            r["rc_over_xi"], yv, yerr=yerr, xerr=xerr,
            fmt=st_info["marker"], color=st_info["color"], markersize=9.5,
            markeredgecolor="#111111", markeredgewidth=0.9,
            ecolor=st_info["color"], elinewidth=1.8, capsize=3.8,
            label=st_info["label"], zorder=5,
        )

    ax.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.80,
               label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$")
    ax.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.5, alpha=0.85,
               label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$")
    if not logy:
        ax.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9, zorder=1)
    else:
        ax.set_yscale("log")

    ax.set_xscale("log")
    ax.set_xlim(0.022, 3.2)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"State-Resolved Dimensionless Radius $R_c / \xi_{\mathrm{state}}$", fontsize=12.0)
    ax.set_ylabel(ylabel, fontsize=11.8)
    ax.set_title(title, fontsize=12.8, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=8.3, framealpha=0.93)


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    df_all = build_combined_table()
    for d in OUT_DIRS:
        df_all.to_csv(d / "gmm_state_resolved_vs_rc_xi_summary.csv", index=False)

    print("=== State-Resolved Metrics vs R_c / xi_state ===")
    print(df_all[["series_key", "xi_um", "rc_over_xi", "m1", "m2", "k1", "k2",
                  "rmse", "d1", "d2", "d3", "Q_8s"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Fig 1: 2x2 Moments & GvM Couplings vs R_c / xi_state
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.2))
    plot_panel(axes[0, 0], df_all, "m1", "m1_lo", "m1_hi",
               r"1st Moment $\langle\cos\theta\rangle$",
               r"(A) 1st Moment $\langle\cos\theta\rangle$ vs $R_c/\xi_{\mathrm{state}}$",
               ylim=(-0.20, 0.90), show_legend=True)
    plot_panel(axes[0, 1], df_all, "m2", "m2_lo", "m2_hi",
               r"2nd Moment $\langle\cos 2\theta\rangle$",
               r"(B) 2nd Moment $\langle\cos 2\theta\rangle$ vs $R_c/\xi_{\mathrm{state}}$",
               ylim=(-0.10, 0.78), show_legend=False)
    plot_panel(axes[1, 0], df_all, "k1", "k1_lo", "k1_hi",
               r"Polar Coupling $\kappa_1$ (GvM)",
               r"(C) GvM Polar Coupling $\kappa_1$ vs $R_c/\xi_{\mathrm{state}}$",
               ylim=(-0.25, 1.90), show_legend=False)
    plot_panel(axes[1, 1], df_all, "k2", "k2_lo", "k2_hi",
               r"Nematic Coupling $\kappa_2$ (GvM)",
               r"(D) GvM Nematic Coupling $\kappa_2$ vs $R_c/\xi_{\mathrm{state}}$",
               ylim=(-0.25, 1.50), show_legend=False)

    fig.suptitle(r"State-Resolved Alignment Moments & GvM Couplings vs $R_c/\xi_{\mathrm{state}}$ ($0.63\,\mu\mathrm{m},\ 1.18\,\mu\mathrm{m}$ Run/Tumble)",
                 fontsize=13.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    vcd.save_figure(fig, "gmm_state_moments_and_gvm_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: 2x2 RMSE & 3-Term KL Decomposition vs R_c / xi_state
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.2))
    plot_panel(axes[0, 0], df_all, "rmse", "rmse_lo", "rmse_hi",
               r"Potential Residual $\mathrm{RMSE}(\tilde{U}_{\mathrm{obs}} - \tilde{U}_{\mathrm{GvM}})$",
               r"(A) Zero-Fitting GvM Potential Residual ($\mathrm{RMSE}$)",
               ylim=(-0.02, 0.95), show_legend=True)
    plot_panel(axes[0, 1], df_all, "d1", "d1_lo", "d1_hi",
               r"1st-Order Polar KL $D_{\mathrm{KL}}^{(1)}$ [nats]",
               r"(B) 1st-Order Polar Propulsion $D_{\mathrm{KL}}^{(1)}$",
               ylim=(-0.03, 1.10), show_legend=False)
    plot_panel(axes[1, 0], df_all, "d2", "d2_lo", "d2_hi",
               r"2nd-Order Nematic KL $D_{\mathrm{KL}}^{(2)}$ [nats]",
               r"(C) 2nd-Order Nematic Alignment $D_{\mathrm{KL}}^{(2)}$",
               ylim=(-0.01, 0.28), show_legend=False)
    plot_panel(axes[1, 1], df_all, "d3", "d3_lo", "d3_hi",
               r"Higher-Order Residual $\Delta D_{\geq 3}$ [nats]",
               r"(D) Higher-Order Residual $\Delta D_{\geq 3}$",
               ylim=(-0.01, 0.25), show_legend=False)

    fig.suptitle(r"State-Resolved GvM Residual & 3-Term KL Decomposition vs $R_c/\xi_{\mathrm{state}}$",
                 fontsize=14.0, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    vcd.save_figure(fig, "gmm_state_kl_and_rmse_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3: 2-Panel Stratonovich TUR Q(8s) & Entropy Production Rate vs R_c / xi_state
    # ------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 5.6))
    plot_panel(axL, df_all, "Q_8s", "Q_8s_lo", "Q_8s_hi",
               r"TUR Precision Index $\mathcal{Q}(\tau = 8\,\mathrm{s})$",
               r"(A) Stratonovich TUR $\mathcal{Q}(\tau=8\,\mathrm{s})$ vs $R_c/\xi_{\mathrm{state}}$",
               ylim=(1e-3, 50.0), logy=True, legend_loc="lower left", show_legend=True)
    plot_panel(axR, df_all, "rate_8s", "rate_8s_lo", "rate_8s_hi",
               r"Entropy Production Lower Bound $\mathcal{Q}/8\,\mathrm{s}$ [$\mathrm{s}^{-1}$]",
               r"(B) TUR Lower Bound Rate $\mathcal{Q}/\tau$ vs $R_c/\xi_{\mathrm{state}}$",
               ylim=(1e-4, 6.0), logy=True, legend_loc="lower left", show_legend=False)

    fig.suptitle(r"State-Resolved Stratonovich TUR vs $R_c/\xi_{\mathrm{state}}$ ($0.63\,\mu\mathrm{m},\ 1.18\,\mu\mathrm{m}$ Run/Tumble)",
                 fontsize=13.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    vcd.save_figure(fig, "gmm_state_tur_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    print("Saved gmm_state_moments_and_gvm_vs_rc_xi, gmm_state_kl_and_rmse_vs_rc_xi, gmm_state_tur_vs_rc_xi, and summary CSV!")


if __name__ == "__main__":
    main()
