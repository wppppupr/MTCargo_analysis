from pathlib import Path
import pandas as pd
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "local_polar_variance" / "local_polar_variance_summary.csv",
    ROOT / "figure" / "local_polar_variance_summary.csv",
    ROOT / "local_polar_variance_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/local_polar_variance/local_polar_variance_summary.csv"),
]


def draw(fig):
    """Draw local polar order variance chi_Phi(R) vs R / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nlocal_polar_variance_summary.csv\nRun analysis/calc_local_polar_variance.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    for bead_name, sub in df.groupby("bead_name", sort=False):
        sub = sub.sort_values("radius_um")
        d_um = float(sub["diameter_um"].iloc[0]) if "diameter_um" in sub.columns else 1.0
        xi = float(sub["xi_um"].iloc[0]) if "xi_um" in sub.columns else 5.0
        color, marker = get_bead_style(d_um=d_um, bead_name=bead_name)

        r_norm = sub["radius_um"].to_numpy(dtype=float) / xi
        mean_v = sub["chi_phi_mean"].to_numpy(dtype=float)
        sem_v = sub["chi_phi_sem"].to_numpy(dtype=float)
        q25 = sub["chi_phi_q25"].to_numpy(dtype=float)
        q75 = sub["chi_phi_q75"].to_numpy(dtype=float)

        ax.fill_between(r_norm, q25, q75, color=color, alpha=0.18, edgecolor="none")
        ax.errorbar(r_norm, mean_v, yerr=sem_v, marker=marker, linestyle="-", markersize=3.5,
                    label=f"{d_um:.2f} µm ($\\xi={xi:.2f}\\,\\mu\\mathrm{{m}}$)", color=color, capsize=2.0)

    line_color = "white" if is_dark() else "#222222"
    ax.axvline(1.0, color=line_color, linestyle="--", zorder=2, label=r"$R / \xi = 1$")
    ax.set_xlabel(r"Normalized Window Radius $R / \xi$")
    ax.set_ylabel(r"Variance $\chi_\Phi(R)$")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=0.85, handlelength=1.5)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(bottom=0)
