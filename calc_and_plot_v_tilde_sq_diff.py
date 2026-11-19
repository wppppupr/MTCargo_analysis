#!/usr/bin/env python3
"""
calc_and_plot_v_tilde_sq_diff.py

局所微小管流（ネマチック／極性流向）座標系における無次元速度二乗差
    Delta_tilde_v2 = <v_tilde_parallel^2> - <v_tilde_perp^2>
および無次元ネマチック速度異方性
    S_v = (<v_tilde_parallel^2> - <v_tilde_perp^2>) / (<v_tilde_parallel^2> + <v_tilde_perp^2>)
を計算し、クラスターブートストラップ (粒子単位, B=2000) による 95% CI 付きで可視化する。

ここで v_tilde_parallel = v_parallel / u_rms_exp, v_tilde_perp = v_perp / u_rms_exp。
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
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2]},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4]},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3]},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5]},
]

SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT


def main():
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)

    vx = df["cargo_dx"] * V_UNIT
    vy = df["cargo_dy"] * V_UNIT
    ux_raw = df["flow_u"] * V_UNIT
    uy_raw = df["flow_v"] * V_UNIT

    norm_u = np.sqrt(ux_raw**2 + uy_raw**2)
    valid = norm_u > 0
    df = df[valid].copy()
    norm_u = norm_u[valid]

    ux_hat = ux_raw[valid] / norm_u
    uy_hat = uy_raw[valid] / norm_u

    df["v_par"] = vx[valid] * ux_hat + vy[valid] * uy_hat
    df["v_perp"] = -vx[valid] * uy_hat + vy[valid] * ux_hat
    df["u2"] = ux_raw[valid]**2 + uy_raw[valid]**2

    df_uinf = pd.read_pickle(U_INF_CACHE_PATH)
    df = df.merge(df_uinf, on=["exp_dir", "frame"], how="left")
    df = df[df["u_inf_t"].notna() & (df["u_inf_t"] > 0)].copy()

    df["vp_t2"] = (df["v_par"] / df["u_inf_t"]) ** 2
    df["vo_t2"] = (df["v_perp"] / df["u_inf_t"]) ** 2
    df["diff_t2"] = df["vp_t2"] - df["vo_t2"]

    rng = np.random.default_rng(42)
    records = []

    for info in BEADS_INFO:
        b = info["name"]
        sub = df[df["bead_name"] == b]
        if sub.empty:
            continue

        rc_xi = float(sub["rc_over_xi"].mean())
        vp2 = float(sub["vp_t2"].mean())
        vo2 = float(sub["vo_t2"].mean())
        diff = float(vp2 - vo2)
        nem_aniso = float(diff / (vp2 + vo2))

        p_agg = sub.groupby("particle").agg({
            "vp_t2": "mean",
            "vo_t2": "mean",
            "diff_t2": "mean",
            "frame": "count"
        })
        n_p = len(p_agg)
        if n_p > 1:
            w = p_agg["frame"].values.astype(float)
            diffs_p = p_agg["diff_t2"].values
            vp2_p = p_agg["vp_t2"].values
            vo2_p = p_agg["vo_t2"].values
            boot_diff, boot_aniso = [], []
            for _ in range(2000):
                idx = rng.choice(n_p, size=n_p, replace=True)
                w_s = w[idx]
                w_sum = np.sum(w_s)
                b_vp2 = np.sum(vp2_p[idx] * w_s) / w_sum
                b_vo2 = np.sum(vo2_p[idx] * w_s) / w_sum
                boot_diff.append(b_vp2 - b_vo2)
                boot_aniso.append((b_vp2 - b_vo2) / (b_vp2 + b_vo2))
            diff_sem = float(np.std(boot_diff))
            diff_ci_low, diff_ci_high = [float(x) for x in np.percentile(boot_diff, [2.5, 97.5])]
            aniso_sem = float(np.std(boot_aniso))
            aniso_ci_low, aniso_ci_high = [float(x) for x in np.percentile(boot_aniso, [2.5, 97.5])]
        else:
            diff_sem = 0.0
            diff_ci_low = diff_ci_high = diff
            aniso_sem = 0.0
            aniso_ci_low = aniso_ci_high = nem_aniso

        records.append({
            "bead_name": b,
            "diameter_um": info["diameter_um"],
            "rc_over_xi": rc_xi,
            "n_particles": n_p,
            "n_frames": len(sub),
            "mean_v_tilde_par2": vp2,
            "mean_v_tilde_perp2": vo2,
            "diff_v_tilde_sq": diff,
            "diff_boot_sem": diff_sem,
            "diff_ci_low": diff_ci_low,
            "diff_ci_high": diff_ci_high,
            "nematic_vel_aniso": nem_aniso,
            "aniso_boot_sem": aniso_sem,
            "aniso_ci_low": aniso_ci_low,
            "aniso_ci_high": aniso_ci_high,
            "marker": info["marker"],
            "color": info["color"],
        })

    df_res = pd.DataFrame(records).sort_values("rc_over_xi").reset_index(drop=True)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_res.drop(columns=["marker", "color"]).to_csv(d / "v_tilde_sq_diff_summary.csv", index=False)

    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # (A) <v_tilde_par^2> - <v_tilde_perp^2>
    ax1.plot(df_res["rc_over_xi"], df_res["diff_v_tilde_sq"], "-", color="#882255", linewidth=2.0, alpha=0.7)
    for _, row in df_res.iterrows():
        yerr = [[row["diff_v_tilde_sq"] - row["diff_ci_low"]], [row["diff_ci_high"] - row["diff_v_tilde_sq"]]]
        ax1.errorbar(row["rc_over_xi"], row["diff_v_tilde_sq"], yerr=yerr,
                     fmt=row["marker"], color=row["color"], markersize=9.5,
                     markeredgecolor="black", markeredgewidth=1.2, capsize=4, capthick=1.4, zorder=5)
        ax1.annotate(f"{row['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(row["rc_over_xi"], row["diff_v_tilde_sq"]),
                     xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    ax1.axhline(0, color="gray", linestyle="--", linewidth=1.1)
    ax1.axvline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax1.set_ylabel(r"$\langle \tilde{v}_\parallel^2 \rangle - \langle \tilde{v}_\perp^2 \rangle$", fontsize=13)
    ax1.set_title(r"(A) Dimensionless Energy Excess $\langle \tilde{v}_\parallel^2 \rangle - \langle \tilde{v}_\perp^2 \rangle$", fontsize=12.5, weight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=10)

    # (B) Normalized nematic velocity anisotropy
    ax2.plot(df_res["rc_over_xi"], df_res["nematic_vel_aniso"], "-", color="#332288", linewidth=2.0, alpha=0.7)
    for _, row in df_res.iterrows():
        yerr = [[row["nematic_vel_aniso"] - row["aniso_ci_low"]], [row["aniso_ci_high"] - row["nematic_vel_aniso"]]]
        ax2.errorbar(row["rc_over_xi"], row["nematic_vel_aniso"], yerr=yerr,
                     fmt=row["marker"], color=row["color"], markersize=9.5,
                     markeredgecolor="black", markeredgewidth=1.2, capsize=4, capthick=1.4, zorder=5)
        ax2.annotate(f"{row['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(row["rc_over_xi"], row["nematic_vel_aniso"]),
                     xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.1, label="Isotropic limit (0)")
    ax2.axvline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax2.set_ylabel(r"$(\langle \tilde{v}_\parallel^2 \rangle - \langle \tilde{v}_\perp^2 \rangle) / (\langle \tilde{v}_\parallel^2 \rangle + \langle \tilde{v}_\perp^2 \rangle)$", fontsize=11.5)
    ax2.set_title(r"(B) Normalized Velocity Anisotropy vs $R_c / \xi$", fontsize=12.5, weight="bold")
    ax2.set_ylim(-0.05, 0.60)
    ax2.grid(True, which="both", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "v_tilde_sq_diff_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "v_tilde_sq_diff_vs_rc_xi.svg")
    plt.close(fig)

    print(df_res[["bead_name", "diameter_um", "rc_over_xi", "mean_v_tilde_par2", "mean_v_tilde_perp2", "diff_v_tilde_sq", "diff_boot_sem", "nematic_vel_aniso"]].to_string(index=False))


if __name__ == "__main__":
    main()
