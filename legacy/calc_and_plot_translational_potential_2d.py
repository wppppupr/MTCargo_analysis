#!/usr/bin/env python3
"""
calc_and_plot_translational_potential_2d.py

局所ネマチック（微小管流）座標系での 2 次元有効並進ポテンシャル面:
    U_trans(Delta r_tilde_parallel, Delta r_tilde_perp; R_c) = - ln P(Delta r_tilde_parallel, Delta r_tilde_perp)

無次元化ステップ変位（＝無次元化速度）の定義:
    Delta r_tilde_parallel = Delta r_parallel / (u_rms_exp * dt) = v_parallel / u_rms_exp
    Delta r_tilde_perp     = Delta r_perp     / (u_rms_exp * dt) = v_perp     / u_rms_exp
ここで u_rms_exp は各実験動画における背景微小管流の RMS 速度、dt = 4.0 s。
Delta r_tilde_parallel = 1 は「背景微小管流と完全に同じ速さで移流されている状態」に対応する。

出力先:
    - figure/translational_potential_2d/
    - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/translational_potential_2d/
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ワークスペースおよび保存先設定
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

# スタイル適用
style_path = CURRENT_DIR / "libs" / "my_style.mplstyle"
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    except Exception:
        style_colors = ["#882255", "#CC6677", "#DDCC77", "#999933", "#117733", "#44AA99"]
else:
    style_colors = ["#882255", "#CC6677", "#DDCC77", "#999933", "#117733", "#44AA99"]

# ビーズ基本情報 (MSD.py 準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

SCALE = 0.11  # um / pixel
DT = 4.0      # s / frame
V_UNIT = SCALE / DT  # 0.0275 um/s
R_MAX_INTEGRAL = 1.5  # 無次元変位の偶奇分解積分範囲 [0, 1.5]


def load_dimensionless_displacements():
    """キャッシュを読み込み、局所流向座標系での無次元ステップ変位を算出"""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_PATH}")

    print(f"Loading instantaneous cache from {CACHE_PATH}...")
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
    vx = vx[valid]
    vy = vy[valid]

    # 局所流向（ネマチック/極性座標軸）に沿った速度 [um/s]
    df["v_par"] = vx * ux_hat + vy * uy_hat
    df["v_perp"] = -vx * uy_hat + vy * ux_hat
    df["u2"] = ux_raw[valid]**2 + uy_raw[valid]**2

    # フレームごとの遠方背景流速 u_inf(t) で無次元化
    # Delta r_tilde = Delta r / (u_inf(t) * dt) = v / u_inf(t)
    if not U_INF_CACHE_PATH.exists():
        raise FileNotFoundError(f"u_inf(t) cache file not found: {U_INF_CACHE_PATH}")
    df_uinf = pd.read_pickle(U_INF_CACHE_PATH)
    df = df.merge(df_uinf, on=["exp_dir", "frame"], how="left")
    df = df[df["u_inf_t"].notna() & (df["u_inf_t"] > 0)].copy()

    df["dr_par_tilde"] = df["v_par"] / df["u_inf_t"]
    df["dr_perp_tilde"] = df["v_perp"] / df["u_inf_t"]

    return df


def compute_translational_potentials(df, grid_limit=2.0, n_grid=121, r_max_int=R_MAX_INTEGRAL):
    """
    各ビーズサイズについて P(Delta r_tilde_par, Delta r_tilde_perp) を 2D ガウス KDE で推定し、
    ボルツマン反転 U_trans = -ln P (および最小値ゼロ基準 dU_trans = -ln(P/P_max)) を算出。
    """
    grid_1d = np.linspace(-grid_limit, grid_limit, n_grid)
    d_grid = grid_1d[1] - grid_1d[0]
    DR_PAR, DR_PERP = np.meshgrid(grid_1d, grid_1d)
    pts = np.vstack([DR_PAR.ravel(), DR_PERP.ravel()])

    center_idx = n_grid // 2
    r_mask = (grid_1d >= 0) & (grid_1d <= r_max_int)
    pos_r = grid_1d[r_mask]
    n_pts_pos = len(pos_r)

    results = {}
    summary_records = []

    for info in BEADS_INFO:
        b_name = info["name"]
        sub = df[df["bead_name"] == b_name]
        if sub.empty:
            continue

        rp = sub["dr_par_tilde"].values
        ro = sub["dr_perp_tilde"].values
        rc_xi = float(sub["rc_over_xi"].mean())

        kde = gaussian_kde(np.vstack([rp, ro]), bw_method="silverman")
        P_2d = kde(pts).reshape(DR_PAR.shape)
        p_max = float(P_2d.max())

        # U_trans = -ln P および 最小値基準 dU_trans = -ln(P / P_max)
        U_raw = -np.log(np.maximum(P_2d, 1e-12))
        U_min = float(-np.log(p_max))
        dU_trans = U_raw - U_min

        # ポテンシャル最小点 (最頻ステップ変位)
        min_idx = np.unravel_index(np.argmin(dU_trans), dU_trans.shape)
        min_dr_par = float(grid_1d[min_idx[1]])
        min_dr_perp = float(grid_1d[min_idx[0]])

        # 2D 平均ステップ変位
        mean_dr_par = float(np.mean(rp))
        mean_dr_perp = float(np.mean(ro))
        std_dr_par = float(np.std(rp))
        std_dr_perp = float(np.std(ro))
        anisotropy = float(std_dr_par / std_dr_perp)

        # 最小値近傍での実効曲率（バネ定数 k = d^2 U / dr^2）
        j0, i0 = min_idx
        if 1 <= i0 < n_grid - 1 and 1 <= j0 < n_grid - 1:
            k_par = float((dU_trans[j0, i0 + 1] - 2.0 * dU_trans[j0, i0] + dU_trans[j0, i0 - 1]) / (d_grid**2))
            k_perp = float((dU_trans[j0 + 1, i0] - 2.0 * dU_trans[j0, i0] + dU_trans[j0 - 1, i0]) / (d_grid**2))
        else:
            k_par, k_perp = np.nan, np.nan

        # Delta r_tilde_perp = 0 に沿った縦断面スライスと偶奇分解
        slice_par = dU_trans[center_idx, :]
        U_pos = slice_par[center_idx : center_idx + n_pts_pos]
        U_neg = slice_par[center_idx - n_pts_pos + 1 : center_idx + 1][::-1]
        U_even = 0.5 * (U_pos + U_neg)
        U_odd = 0.5 * (U_pos - U_neg)

        A_odd = float(trapezoid(np.abs(U_odd), pos_r))
        I_even = float(trapezoid(U_even, pos_r))
        eta_asym = float(A_odd / I_even if I_even > 0 else 0.0)

        results[b_name] = {
            "info": info,
            "rc_over_xi": rc_xi,
            "DR_PAR": DR_PAR,
            "DR_PERP": DR_PERP,
            "grid_1d": grid_1d,
            "P_2d": P_2d,
            "U_raw": U_raw,
            "U_min": U_min,
            "dU_trans": dU_trans,
            "min_dr_par": min_dr_par,
            "min_dr_perp": min_dr_perp,
            "mean_dr_par": mean_dr_par,
            "mean_dr_perp": mean_dr_perp,
            "std_dr_par": std_dr_par,
            "std_dr_perp": std_dr_perp,
            "anisotropy": anisotropy,
            "k_par": k_par,
            "k_perp": k_perp,
            "slice_par": slice_par,
            "slice_perp": dU_trans[:, center_idx],
            "pos_r": pos_r,
            "U_even": U_even,
            "U_odd": U_odd,
            "A_odd": A_odd,
            "I_even": I_even,
            "eta_asym": eta_asym,
            "n_frames": len(sub),
        }

        summary_records.append({
            "bead_name": b_name,
            "diameter_um": info["diameter_um"],
            "radius_um": info["radius_um"],
            "rc_over_xi": rc_xi,
            "n_frames": len(sub),
            "U_trans_min_raw": U_min,
            "min_dr_par_tilde": min_dr_par,
            "min_dr_perp_tilde": min_dr_perp,
            "mean_dr_par_tilde": mean_dr_par,
            "mean_dr_perp_tilde": mean_dr_perp,
            "std_dr_par_tilde": std_dr_par,
            "std_dr_perp_tilde": std_dr_perp,
            "anisotropy_std_ratio": anisotropy,
            "k_par_curv": k_par,
            "k_perp_curv": k_perp,
            "A_odd_integral": A_odd,
            "I_even_integral": I_even,
            "eta_asym_ratio": eta_asym,
        })

    df_summary = pd.DataFrame(summary_records)
    return results, df_summary


def plot_2d_translational_potential_grid(results):
    """6サイズの 2D 有効並進ポテンシャル面 dU_trans(Delta r_tilde_par, Delta r_tilde_perp) の等高線図"""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), sharex=True, sharey=True)
    axes = axes.flatten()

    contour_levels = np.linspace(0.0, 5.0, 11)
    last_cf = None

    for idx, info in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = info["name"]
        data = results[b_name]

        cf = ax.contourf(
            data["DR_PAR"], data["DR_PERP"], data["dU_trans"],
            levels=contour_levels, cmap="viridis_r", extend="max"
        )
        ax.contour(
            data["DR_PAR"], data["DR_PERP"], data["dU_trans"],
            levels=contour_levels, colors="white", linewidths=0.7, alpha=0.65
        )
        last_cf = cf

        ax.axvline(0, color="gray", linestyle=":", linewidth=1.1, alpha=0.8)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.1, alpha=0.8)
        ax.axvline(1.0, color="cyan", linestyle="--", linewidth=0.9, alpha=0.65)

        # 最頻点 (ポテンシャル最小) と 平均変位点
        ax.plot(data["min_dr_par"], data["min_dr_perp"], marker="+", markersize=10,
                markeredgewidth=2.2, color="red", label="Min")
        ax.plot(data["mean_dr_par"], data["mean_dr_perp"], marker="o", markersize=6,
                color="orange", markeredgecolor="black", markeredgewidth=0.9, label="Mean")

        ax.set_title(rf"{info['label']}  ($R_c/\xi \approx {data['rc_over_xi']:.2f}$)",
                     fontsize=13, weight="bold", pad=8)

        ax.text(
            0.05, 0.93,
            rf"$\langle \Delta\tilde{{r}}_\parallel \rangle = {data['mean_dr_par']:+.2f}$" + "\n" +
            rf"$\sigma_\parallel/\sigma_\perp = {data['anisotropy']:.2f}$" + "\n" +
            rf"$\eta_{{\mathrm{{asym}}}} = {data['eta_asym']*100:.1f}\%$",
            transform=ax.transAxes, fontsize=9.5, weight="bold",
            color="white", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6)
        )

        ax.set_xlim(-1.6, 1.8)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Longitudinal step $\Delta\tilde{r}_\parallel = \Delta r_\parallel / (u_\infty(t)\Delta t)$", fontsize=10.5)
        ax.set_ylabel(r"Transverse step $\Delta\tilde{r}_\perp = \Delta r_\perp / (u_\infty(t)\Delta t)$", fontsize=10.5)
        ax.grid(False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_cf, cax=cbar_ax)
    cbar.set_label(r"Effective Translational Potential $\tilde{U}_{\mathrm{trans}} - \tilde{U}_{\min} = -\ln(P/P_{\max})$",
                   fontsize=12, weight="bold")
    cbar.set_ticks(contour_levels[::2])

    fig.suptitle(
        r"2D Effective Translational Potential $\tilde{U}_{\mathrm{trans}}(\Delta\tilde{r}_\parallel, \Delta\tilde{r}_\perp; R_c)$ in Local Flow Frame",
        fontsize=14.5, weight="bold", y=0.97
    )
    plt.subplots_adjust(right=0.90, top=0.91, bottom=0.08, left=0.07, hspace=0.25, wspace=0.22)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "translational_potential_2d_contour_grid.png", dpi=300)
        fig.savefig(d / "translational_potential_2d_contour_grid.svg")
    plt.close(fig)
    print("Saved Figure 1: translational_potential_2d_contour_grid.png / .svg")


def plot_1d_slices(results):
    """1次元断面プロット: (A) 縦方向断面 (Delta r_perp = 0) と (B) 横方向断面 (Delta r_par = 0)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    for info in BEADS_INFO:
        data = results[info["name"]]
        g = data["grid_1d"]
        label = rf"{info['label']} ($R_c/\xi \approx {data['rc_over_xi']:.2f}$)"
        ax1.plot(g, data["slice_par"], label=label, color=info["color"], linewidth=2.2)
        ax2.plot(g, data["slice_perp"], label=label, color=info["color"], linewidth=2.2)

    ax1.axvline(0, color="gray", linestyle=":", linewidth=1.2)
    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label=r"Flow speed ($\Delta\tilde{r}_\parallel=1$)")
    ax1.set_xlabel(r"Dimensionless longitudinal step $\Delta\tilde{r}_\parallel$", fontsize=12)
    ax1.set_ylabel(r"$\tilde{U}_{\mathrm{trans}}(\Delta\tilde{r}_\parallel, 0) - \tilde{U}_{\min}$", fontsize=12)
    ax1.set_title(r"(A) Longitudinal Slice ($\Delta\tilde{r}_\perp = 0$)", fontsize=12.5, weight="bold")
    ax1.set_xlim(-1.6, 1.8)
    ax1.set_ylim(-0.15, 4.5)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper center", fontsize=9.0, framealpha=0.88)

    ax2.axvline(0, color="gray", linestyle=":", linewidth=1.2)
    ax2.set_xlabel(r"Dimensionless transverse step $\Delta\tilde{r}_\perp$", fontsize=12)
    ax2.set_ylabel(r"$\tilde{U}_{\mathrm{trans}}(0, \Delta\tilde{r}_\perp) - \tilde{U}_{\min}$", fontsize=12)
    ax2.set_title(r"(B) Transverse Slice ($\Delta\tilde{r}_\parallel = 0$)", fontsize=12.5, weight="bold")
    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-0.15, 4.5)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper center", fontsize=9.0, framealpha=0.88)

    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "translational_potential_1d_slices.png", dpi=300)
        fig.savefig(d / "translational_potential_1d_slices.svg")
    plt.close(fig)
    print("Saved Figure 2: translational_potential_1d_slices.png / .svg")


def plot_topology_metrics_vs_rc_xi(df_summary):
    """R_c / xi に対する無次元ドリフト・非対称寄与率・ポテンシャル異方性の変化"""
    df_plot = df_summary.sort_values(by="rc_over_xi").copy()
    x = df_plot["rc_over_xi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # (A) Mean dimensionless drift & odd-asymmetry ratio
    ax1.plot(x, df_plot["mean_dr_par_tilde"], "o-", color="#CC6677", linewidth=2.3, markersize=8.5,
             label=r"Mean drift $\langle \Delta\tilde{r}_\parallel \rangle$")
    ax1.plot(x, df_plot["eta_asym_ratio"], "s--", color="#332288", linewidth=2.1, markersize=8.0,
             label=r"Odd asymmetry ratio $\eta_{\mathrm{asym}}$")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax1.axvline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax1.set_ylabel(r"Dimensionless Drift / Asymmetry", fontsize=12)
    ax1.set_title(r"(A) Propulsion Bias & Parity Restoration vs $R_c/\xi$", fontsize=12.5, weight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=9.8, framealpha=0.9)

    for _, row in df_plot.iterrows():
        ax1.annotate(f"{row['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(row["rc_over_xi"], row["mean_dr_par_tilde"]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color="#CC6677", weight="bold")

    # (B) Anisotropy sigma_par / sigma_perp -> 1
    ax2.plot(x, df_plot["anisotropy_std_ratio"], "D-", color="#117733", linewidth=2.3, markersize=8.5,
             label=r"Well aspect ratio $\sigma_\parallel / \sigma_\perp$")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label=r"Isotropic 2D limit ($\sigma_\parallel / \sigma_\perp = 1$)")
    ax2.axvline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.6, label=r"$R_c = \xi$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=12.5)
    ax2.set_ylabel(r"Translational Well Anisotropy $\sigma_\parallel / \sigma_\perp$", fontsize=12)
    ax2.set_title(r"(B) Restoration of Rotational Isotropy vs $R_c/\xi$", fontsize=12.5, weight="bold")
    ax2.set_ylim(0.95, 1.6)
    ax2.grid(True, which="both", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=9.8, framealpha=0.9)

    for _, row in df_plot.iterrows():
        ax2.annotate(f"{row['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(row["rc_over_xi"], row["anisotropy_std_ratio"]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color="#117733", weight="bold")

    plt.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "translational_potential_topology_vs_rc_xi.png", dpi=300)
        fig.savefig(d / "translational_potential_topology_vs_rc_xi.svg")
    plt.close(fig)
    print("Saved Figure 3: translational_potential_topology_vs_rc_xi.png / .svg")


def main():
    print("=== 2D Effective Translational Potential Surface U_trans Analysis ===")
    df = load_dimensionless_displacements()
    results, df_summary = compute_translational_potentials(df)

    csv_filename = "translational_potential_2d_summary.csv"
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(d / csv_filename, index=False)
    print(f"Saved summary CSV to {OUT_DIRS[0] / csv_filename}")

    plot_2d_translational_potential_grid(results)
    plot_1d_slices(results)
    plot_topology_metrics_vs_rc_xi(df_summary)

    print("\n--- 2D Effective Translational Potential Summary ---")
    cols = [
        "bead_name", "diameter_um", "rc_over_xi",
        "mean_dr_par_tilde", "std_dr_par_tilde", "std_dr_perp_tilde",
        "anisotropy_std_ratio", "A_odd_integral", "eta_asym_ratio"
    ]
    print(df_summary[cols].to_string(index=False))
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
