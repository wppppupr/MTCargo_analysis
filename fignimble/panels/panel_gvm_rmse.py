from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "gvm_moments_and_potential_summary.csv",
    ROOT / "gvm_moments_and_potential_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/gvm_moments_and_potential_summary.csv")
]


def draw(fig):
    """Draw zero-fitting GvM potential residual (RMSE) vs Rc / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\ngvm_moments_and_potential_summary.csv\nRun analysis/calc_gvm_potential.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df_out = pd.read_csv(csv_path).sort_values("rc_over_xi")
    ax.plot(df_out["rc_over_xi"], df_out["rmse_U_obs_vs_GvM"], "-", color="#444444", alpha=0.65, zorder=2)

    offsets = {
        "beads06um": (8, 3, "left"), "beads1um": (8, 3, "left"), "beads3um": (8, 5, "left"),
        "beads7um": (8, -6, "left"), "beads5um": (-8, -8, "right"), "beads20um": (0, 12, "center"),
    }
    for _, r in df_out.iterrows():
        bname = r["bead_name"]
        color, marker = get_bead_style(bead_name=bname)
        yv = float(r["rmse_U_obs_vs_GvM"])
        yerr = [[max(0.0, yv - float(r["rmse_ci_low"]))], [max(0.0, float(r["rmse_ci_high"]) - yv)]]
        ax.errorbar(r["rc_over_xi"], yv, yerr=yerr, xerr=r["rc_over_xi_sem"],
                    fmt=marker, color=color, markersize=4.5, markeredgecolor="#222222",
                    ecolor=color, capsize=3.0, zorder=4)
        dx, dy, ha = offsets.get(bname, (6, 4, "left"))
        ax.annotate(rf"{r['diameter_um']:.2g} $\mu$m", xy=(r["rc_over_xi"], yv),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=color, zorder=5)

    x_lock = 45.0 * np.pi / 512.0
    x_bulk = 45.0 * np.pi / 128.0
    ax.axvline(x_lock, color="#d95f02", linestyle="--", alpha=0.85, label=r"$x_{\mathrm{lock}} \approx 0.28$")
    ax.axvline(x_bulk, color="#222222", linestyle=":", alpha=0.85, label=r"$x_{\mathrm{bulk}} \approx 1.10$")
    ax.axhline(0.0, color="#888888", linestyle="-")
    ax.set_xscale("log")
    ax.set_xlim(0.045, 3.2)
    ax.set_ylim(-0.01, 0.27)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax.set_ylabel(r"Potential Residual $\mathrm{RMSE}(\tilde{U}_{\mathrm{obs}} - \tilde{U}_{\mathrm{GvM}})$")
    #ax.grid(True, which="both", linestyle=":", alpha=0.55)
    #ax.legend(loc="upper right", framealpha=0.92)
