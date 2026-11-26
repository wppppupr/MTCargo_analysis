#!/usr/bin/env python3
"""
calc_and_plot_alpha_tau_tilde.py

1フレーム変位 (Delta t = 4 s) の無次元速度異方性
    A = (<v_tilde_par^2> - <v_tilde_perp^2>) / (<v_tilde_par^2> + <v_tilde_perp^2>)
および無次元1フレーム時間刻み
    Delta t_tilde = <u_inf(t) / xi> * Delta t
を用いて、局所対数微分としての有効異常拡散指数:
    alpha(tau_tilde) = 1 + [ (A / (1 - A)) * (tau_tilde / Delta t_tilde) ] / [ 1 + (A / (1 - A)) * (tau_tilde / Delta t_tilde) ]
を各ビーズサイズについて計算・可視化する。
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "translational_potential_2d",
    ROOT_DIR / "figure" / "translational_potential_2d",
]
CACHE_PATH = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
U_INF_CACHE_PATH = CURRENT_DIR / "scratch" / "u_inf_per_frame_cache.pkl"

style_path = CURRENT_DIR / "libs" / "my_style.mplstyle"
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    except Exception:
        style_colors = ["#882255", "#CC6677", "#DDCC77", "#999933", "#117733", "#44AA99"]
else:
    style_colors = ["#882255", "#CC6677", "#DDCC77", "#999933", "#117733", "#44AA99"]

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

SCALE = 0.11
DT = 4.0  # Delta t = 4 s
V_UNIT = SCALE / DT


def alpha_model(tau_tilde: np.ndarray, A: float, dt_tilde: float) -> np.ndarray:
    ratio = (A / (1.0 - A)) * (tau_tilde / dt_tilde)
    return 1.0 + ratio / (1.0 + ratio)


def main():
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    df_uinf = pd.read_pickle(U_INF_CACHE_PATH)
    df = df.merge(df_uinf, on=["exp_dir", "frame"], how="left")

    vx = df["cargo_dx"] * V_UNIT
    vy = df["cargo_dy"] * V_UNIT
    ux_raw = df["flow_u"] * V_UNIT
    uy_raw = df["flow_v"] * V_UNIT
    norm_u = np.sqrt(ux_raw**2 + uy_raw**2)

    valid = (norm_u > 0) & df["u_inf_t"].notna() & (df["u_inf_t"] > 0) & (df["xi"] > 0)
    df = df[valid].copy()
    norm_u = norm_u[valid]
    ux_hat = ux_raw[valid] / norm_u
    uy_hat = uy_raw[valid] / norm_u

    df["vp_t2"] = ((vx[valid] * ux_hat + vy[valid] * uy_hat) / df["u_inf_t"]) ** 2
    df["vo_t2"] = ((-vx[valid] * uy_hat + vy[valid] * ux_hat) / df["u_inf_t"]) ** 2
    df["dt_tilde"] = (df["u_inf_t"] / df["xi"]) * DT

    tau_grid = np.geomspace(0.05, 80.0, 200)
    eval_taus = [0.5, 1.0, 3.0, 10.0, 30.0]

    summary_rows = []
    curve_rows = []

    for info in BEADS_INFO:
        b = info["name"]
        sub = df[df["bead_name"] == b]
        if sub.empty:
            continue

        rc_xi = float(sub["rc_over_xi"].mean())
        dt_tilde = float(sub["dt_tilde"].mean())
        vp2 = float(sub["vp_t2"].mean())
        vo2 = float(sub["vo_t2"].mean())
        A = float((vp2 - vo2) / (vp2 + vo2))
        ratio_A = float(A / (1.0 - A))
        tau_cross = float(((1.0 - A) / A) * dt_tilde) if A > 0 else np.inf

        row = {
            "bead_name": b,
            "diameter_um": info["diameter_um"],
            "rc_over_xi": rc_xi,
            "dt_sec": DT,
            "dt_tilde": dt_tilde,
            "A_aniso": A,
            "A_over_1_minus_A": ratio_A,
            "tau_tilde_crossover_alpha_1_5": tau_cross,
            "alpha_at_dt_tilde": float(alpha_model(np.array([dt_tilde]), A, dt_tilde)[0]),
        }
        for t_val in eval_taus:
            row[f"alpha_tau_{t_val:g}"] = float(alpha_model(np.array([t_val]), A, dt_tilde)[0])
        summary_rows.append(row)

        a_curve = alpha_model(tau_grid, A, dt_tilde)
        for t_g, a_g in zip(tau_grid, a_curve):
            curve_rows.append({
                "bead_name": b,
                "diameter_um": info["diameter_um"],
                "rc_over_xi": rc_xi,
                "tau_tilde": float(t_g),
                "alpha": float(a_g),
            })

    df_sum = pd.DataFrame(summary_rows).sort_values("rc_over_xi").reset_index(drop=True)
    df_curves = pd.DataFrame(curve_rows)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(d / "alpha_tau_tilde_summary.csv", index=False)
        df_curves.to_csv(d / "alpha_tau_tilde_curves.csv", index=False)

    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    for info in BEADS_INFO:
        sub_c = df_curves[df_curves["bead_name"] == info["name"]]
        rc_xi = sub_c["rc_over_xi"].iloc[0]
        ax1.plot(sub_c["tau_tilde"], sub_c["alpha"], color=info["color"], linewidth=2.3,
                 label=rf"{info['label']} ($R_c/\xi \approx {rc_xi:.2f}$)")

    ax1.axhline(1.5, color="gray", linestyle="--", linewidth=1.1, label=r"Crossover $\alpha = 1.5$")
    ax1.axhline(2.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Dimensionless lag time $\tilde{\tau}$", fontsize=12.5)
    ax1.set_ylabel(r"Local diffusion exponent $\alpha(\tilde{\tau})$", fontsize=12.5)
    ax1.set_title(r"(A) $\alpha(\tilde{\tau})$ vs $\tilde{\tau}$ ($\Delta t = 4\,\mathrm{s}$)", fontsize=12.5, weight="bold")
    ax1.set_xlim(0.05, 80.0)
    ax1.set_ylim(0.95, 2.05)
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=9.2, framealpha=0.9)

    tau_styles = [(0.5, "#CC6677", "o-"), (1.0, "#882255", "s-"), (3.0, "#332288", "^-"), (10.0, "#117733", "d--")]
    for t_val, col, fmt in tau_styles:
        ax2.plot(df_sum["rc_over_xi"], df_sum[f"alpha_tau_{t_val:g}"], fmt, color=col,
                 linewidth=2.1, markersize=7.5, label=rf"$\tilde{{\tau}} = {t_val:g}$")

    ax2.axvline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax2.set_ylabel(r"$\alpha(\tilde{\tau})$", fontsize=12.5)
    ax2.set_title(r"(B) $\alpha(\tilde{\tau})$ vs $R_c/\xi$ at Fixed $\tilde{\tau}$", fontsize=12.5, weight="bold")
    ax2.set_ylim(1.15, 2.05)
    ax2.grid(True, which="both", linestyle=":", alpha=0.45)
    ax2.legend(loc="lower left", fontsize=9.8, framealpha=0.9)

    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "alpha_vs_tau_tilde.png", dpi=300)
        fig.savefig(d / "alpha_vs_tau_tilde.svg")
    plt.close(fig)

    cols = ["bead_name", "diameter_um", "rc_over_xi", "dt_tilde", "A_aniso", "tau_tilde_crossover_alpha_1_5",
            "alpha_at_dt_tilde", "alpha_tau_0.5", "alpha_tau_1", "alpha_tau_3", "alpha_tau_10", "alpha_tau_30"]
    print(df_sum[cols].to_string(index=False))


if __name__ == "__main__":
    main()
