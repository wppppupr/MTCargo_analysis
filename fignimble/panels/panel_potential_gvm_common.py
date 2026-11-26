from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

ROOT = Path(__file__).resolve().parent.parent.parent
U_CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "per_point_cos_effective_potential_summary.csv",
    ROOT / "figure" / "per_point_cos_effective_potential_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/per_point_cos_effective_potential_summary.csv"),
]
GVM_CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "gvm_moments_and_potential_summary.csv",
    ROOT / "figure" / "gvm_moments_and_potential_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/gvm_moments_and_potential_summary.csv"),
]


def draw_bead_potential(fig, bead_name: str):
    """Draw observed effective potential U_eff(c) + GvM theory curve for one cargo size."""
    ax = fig.add_subplot(111)
    u_path = next((p for p in U_CSV_CANDIDATES if p.exists()), None)
    gvm_path = next((p for p in GVM_CSV_CANDIDATES if p.exists()), None)
    if u_path is None or gvm_path is None:
        ax.text(0.5, 0.5, "Missing summary CSVs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df_u = pd.read_csv(u_path)
    df_g = pd.read_csv(gvm_path)
    sub = df_u[df_u["bead_name"] == bead_name].sort_values("c_center")
    g_row = df_g[df_g["bead_name"] == bead_name]
    if sub.empty or g_row.empty:
        ax.text(0.5, 0.5, f"No data for {bead_name}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d_um = float(sub["diameter_um"].iloc[0])
    rc_xi = float(g_row["rc_over_xi"].iloc[0])
    k1 = float(g_row["kappa_1_gvm"].iloc[0])
    k2 = float(g_row["kappa_2_gvm"].iloc[0])
    offset = float(g_row["gvm_offset"].iloc[0])
    color, marker = get_bead_style(d_um=d_um, bead_name=bead_name)

    x = sub["c_center"].to_numpy(dtype=float)
    u = sub["u_eff"].to_numpy(dtype=float)
    valid = np.isfinite(u)

    ax.axhline(0.0, color="#666666", linestyle="--", alpha=0.6, zorder=1)
    ax.axvline(0.0, color="#888888", linestyle=":", alpha=0.5, zorder=1)

    if "u_eff_ci_low" in sub.columns and "u_eff_ci_high" in sub.columns:
        u_lo = sub["u_eff_ci_low"].to_numpy(dtype=float)
        u_hi = sub["u_eff_ci_high"].to_numpy(dtype=float)
        v_ci = valid & np.isfinite(u_lo) & np.isfinite(u_hi)
        ax.fill_between(x[v_ci], u_lo[v_ci], u_hi[v_ci], color=color, alpha=0.18, edgecolor="none", zorder=2)

    ax.plot(x[valid], u[valid], marker=marker, linestyle="-", markersize=3.0, color=color,
            label=rf"{d_um:.2f} µm ($R_c/\xi={rc_xi:.2f}$)", zorder=4)

    c_grid = np.linspace(-1.0, 1.0, 200)
    u_gvm_at_x = -k1 * x[valid] - k2 * (2.0 * x[valid]**2 - 1.0)
    offset = float(np.mean(u[valid] - u_gvm_at_x))
    u_gvm = -k1 * c_grid - k2 * (2.0 * c_grid**2 - 1.0) + offset
    ax.plot(c_grid, u_gvm, linestyle="--", color="#222222", label="GvM theory", zorder=5)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.5, 1.0)
    ax.set_xlabel(r"$c = \cos\theta_i$")
    ax.set_ylabel(r"$\widetilde{U}_{\mathrm{eff}}(c)$")
    #ax.legend(loc="lower left", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, linestyle="--", alpha=0.35)
