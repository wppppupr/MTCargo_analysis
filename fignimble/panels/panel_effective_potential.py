from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "per_point_cos_effective_potential_summary.csv",
    ROOT / "figure" / "per_point_cos_effective_potential_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/per_point_cos_effective_potential_summary.csv"),
]


def draw(fig):
    """Draw 1-panel effective potential U_eff(c) = -ln(P(c)/P_iso(c)) with bootstrap 95% CI."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nper_point_cos_effective_potential_summary.csv\nRun analysis/calc_per_point_v_dot_u.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.axhline(0.0, color="#666666", linestyle="--", alpha=0.7, zorder=1, label="Isotropic flat potential")
    ax.axvline(0.0, color="#888888", linestyle=":", alpha=0.6, zorder=1)

    df = pd.read_csv(csv_path)
    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp.sort_values("c_center")
        bname = str(grp["bead_name"].iloc[0]) if "bead_name" in grp.columns else None
        color, marker = get_bead_style(d_um=d_um, bead_name=bname)
        x = grp["c_center"].to_numpy(dtype=float)
        u = grp["u_eff"].to_numpy(dtype=float)
        valid = np.isfinite(u)
        if not np.any(valid):
            continue

        if "u_eff_ci_low" in grp.columns and "u_eff_ci_high" in grp.columns:
            u_lo = grp["u_eff_ci_low"].to_numpy(dtype=float)
            u_hi = grp["u_eff_ci_high"].to_numpy(dtype=float)
            valid_ci = valid & np.isfinite(u_lo) & np.isfinite(u_hi)
            ax.fill_between(x[valid_ci], u_lo[valid_ci], u_hi[valid_ci], color=color, alpha=0.18, edgecolor="none", zorder=2)

        ax.plot(x[valid], u[valid], marker=marker, linestyle="-", markersize=3.0, color=color, label=f"{d_um:.2f} µm", zorder=4)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.6, 1.4)
    ax.set_xlabel(r"$c = \cos\theta_i$")
    ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c) = -\ln(P(c)/P_{\mathrm{iso}}(c))$")
    #ax.legend(loc="lower left", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, linestyle="--", alpha=0.35)
