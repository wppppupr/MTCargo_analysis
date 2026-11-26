from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "lagrangian_flow_autocorr" / "lagrangian_u_bar_autocorr_curves.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/lagrangian_flow_autocorr/lagrangian_u_bar_autocorr_curves.csv"),
]


def draw(fig):
    """Draw Lagrangian autocorrelation of footprint flow C_u(tau)."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nlagrangian_u_bar_autocorr_curves.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    if "mode" in df.columns and (df["mode"] == "vec_raw").any():
        df = df[df["mode"] == "vec_raw"]

    ax.axhline(0.0, color="#666666", linestyle="--", alpha=0.7, zorder=1)
    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp.sort_values("lag_s")
        bname = str(grp["bead_name"].iloc[0]) if "bead_name" in grp.columns else None
        color, marker = get_bead_style(d_um=d_um, bead_name=bname)
        t = grp["lag_s"].to_numpy(dtype=float)
        c = grp["C"].to_numpy(dtype=float)
        if "C_ci_lo" in grp.columns and "C_ci_hi" in grp.columns:
            ax.fill_between(t, grp["C_ci_lo"].to_numpy(dtype=float), grp["C_ci_hi"].to_numpy(dtype=float),
                            color=color, alpha=0.16, edgecolor="none", zorder=2)
        ax.plot(t, c, marker=marker, linestyle="-", markersize=3.0, markevery=3, color=color,
                label=f"{d_um:.2f} µm", zorder=4)

    ax.set_xlim(0.0, 240.0)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel(r"Lag time $\tau$ [s]")
    ax.set_ylabel(r"Flow Lagrangian ACF $C_{\bar{\mathbf{u}}}(\tau)$")
    ax.legend(loc="upper right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, linestyle="--", alpha=0.35)
