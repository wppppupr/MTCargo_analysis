from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "per_point_c_histogram_summary.csv",
    ROOT / "figure" / "per_point_c_histogram_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/per_point_c_histogram_summary.csv"),
]


def draw(fig):
    """Draw P(c) histogram (c = cos theta_i) with bootstrap 95% CI across cargo sizes."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nper_point_c_histogram_summary.csv\nRun analysis/calc_per_point_v_dot_u.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    if "p_iso" in df.columns:
        iso_grp = df.groupby("c_center", as_index=False)["p_iso"].first().sort_values("c_center")
        ax.plot(iso_grp["c_center"].to_numpy(dtype=float), iso_grp["p_iso"].to_numpy(dtype=float),
                color="#555555", linestyle="--", label=r"Isotropic 2D", zorder=1)
    else:
        ax.axhline(0.5, color="#888888", linestyle=":", alpha=0.7, label="Uniform (0.5)")

    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp.sort_values("c_center")
        bname = str(grp["bead_name"].iloc[0]) if "bead_name" in grp.columns else None
        color, marker = get_bead_style(d_um=d_um, bead_name=bname)
        x = grp["c_center"].to_numpy(dtype=float)
        y = grp["pdf"].to_numpy(dtype=float)
        valid = np.isfinite(y)
        if not np.any(valid):
            continue

        if "pdf_ci_low" in grp.columns and "pdf_ci_high" in grp.columns:
            y_lo = grp["pdf_ci_low"].to_numpy(dtype=float)
            y_hi = grp["pdf_ci_high"].to_numpy(dtype=float)
            valid_ci = valid & np.isfinite(y_lo) & np.isfinite(y_hi)
            ax.fill_between(x[valid_ci], y_lo[valid_ci], y_hi[valid_ci], color=color, alpha=0.16, edgecolor="none", zorder=2)

        ax.plot(x[valid], y[valid], marker=marker, linestyle="-", markersize=3.0, color=color, label=f"{d_um:.2f} µm", zorder=4)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(r"$c = \cos\theta_i$")
    ax.set_ylabel(r"Probability density $P(c)$")
    ax.legend(loc="upper left", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, linestyle="--", alpha=0.35)
