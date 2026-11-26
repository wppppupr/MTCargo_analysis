from pathlib import Path
import pandas as pd
import matplotlib.ticker as ticker
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv",
    ROOT / "figure" / "bg_angular_correlation_length_summary.csv",
    ROOT / "bg_angular_correlation_length_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_summary.csv"),
]


def draw(fig):
    """Draw correlation length xi_bg vs cargo radius Rc."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nbg_angular_correlation_length_summary.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    ctrl = df[df["bead_name"] == "control"]
    beads = df[df["bead_name"] != "control"].sort_values("radius_um")

    if not ctrl.empty:
        xi_c = float(ctrl["xi_bg_mean_um"].iloc[0])
        sem_c = float(ctrl["xi_bg_sem_um"].iloc[0])
        ref_color = "white" if is_dark() else "#333333"
        #ax.axhline(xi_c, color=ref_color, linestyle="--", alpha=0.8, label=rf"Control ($\xi={xi_c:.2f}\,\mu\mathrm{{m}}$)")
        #ax.axhspan(xi_c - sem_c, xi_c + sem_c, color="#888888", alpha=0.18)

    if not beads.empty:
        r_vals = beads["radius_um"].to_numpy(dtype=float)
        xi_vals = beads["xi_bg_mean_um"].to_numpy(dtype=float)
        ax.plot(r_vals, xi_vals, color="#666666", linestyle="-", alpha=0.55, zorder=2)

        for _, row in beads.iterrows():
            bname = str(row["bead_name"])
            d_um = float(row["diameter_um"])
            color, marker = get_bead_style(d_um=d_um, bead_name=bname)
            ax.errorbar(
                row["radius_um"],
                row["xi_bg_mean_um"],
                yerr=row["xi_bg_sem_um"],
                fmt=marker,
                color=color,
                ecolor=color,
                markersize=4.5,
                capsize=3.0,
                label=f"{d_um:.2f} µm",
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xlim(0.25, 15.0)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"Cargo Radius $R_c$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Correlation length $\xi$ [$\mu\mathrm{m}$]")
    #ax.legend(loc="best", framealpha=0.88, handlelength=1.4)
    #ax.grid(True, which="both", linestyle="--", alpha=0.35)
