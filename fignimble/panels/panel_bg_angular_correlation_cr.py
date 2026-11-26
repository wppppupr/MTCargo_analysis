from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "bg_angular_correlation" / "bg_angular_correlation_curves.csv",
    ROOT / "figure" / "bg_angular_correlation_curves.csv",
    ROOT / "bg_angular_correlation_curves.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_curves.csv"),
]


def draw(fig):
    """Draw background MT spatial orientation correlation C_bg(r) vs distance r."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nbg_angular_correlation_curves.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    for bname, grp in df.groupby("bead_name", sort=False):
        grp = grp.sort_values("distance_um")
        if grp.empty:
            continue

        d_val = grp["diameter_um"].iloc[0]
        if bname == "control" or pd.isna(d_val):
            color = "white" if is_dark() else "#222222"
            marker = "x"
            label_prefix = "Control (no cargo)"
        else:
            d_um = float(d_val)
            color, marker = get_bead_style(d_um=d_um, bead_name=str(bname))
            label_prefix = f"{d_um:.2f} µm"

        r = grp["distance_um"].to_numpy(dtype=float)
        c = grp["mean_c"].to_numpy(dtype=float)
        sem = grp["sem_c"].to_numpy(dtype=float)
        xi_fit = grp["xi_bg_cond_fit_um"].iloc[0] if "xi_bg_cond_fit_um" in grp.columns else np.nan
        label = f"{label_prefix} ($\\xi={xi_fit:.1f}\\,\\mu\\mathrm{{m}}$)" if np.isfinite(xi_fit) else label_prefix

        ax.fill_between(r, c - sem, c + sem, color=color, alpha=0.16, edgecolor="none")
        ax.errorbar(r, c, yerr=sem, fmt="none", ecolor=color, capsize=2.0, errorevery=3, alpha=0.8, zorder=3)
        ax.plot(r, c, marker=marker, linestyle="-", markersize=3.0, markevery=3, color=color, label=label, zorder=4)

    ax.axhline(0.0, color="#888888", linestyle=":", alpha=0.7)
    ax.set_xlim(0.0, 25.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Orientation correlation $C_{\mathrm{bg}}(r)$")
    #ax.legend(loc="upper right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, linestyle="--", alpha=0.35)
