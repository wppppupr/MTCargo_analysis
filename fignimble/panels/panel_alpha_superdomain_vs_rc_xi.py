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
    ROOT / "alpha_superdomain_vs_delta_u_prop_summary.csv",
    ROOT / "figure" / "alpha_superdomain_vs_delta_u_prop_summary.csv",
]


def draw(fig):
    """Draw super-domain diffusion exponent alpha_WLS vs Rc / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nalpha_superdomain_vs_delta_u_prop_summary.csv\nRun analysis/calc_alpha_vs_delta_u_prop.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df_out = pd.read_csv(csv_path)
    rx_min, rx_max, rx_split = 0.045, 4.0, 0.28

    ax.axvspan(rx_min, rx_split, color="#fcedea", alpha=0.65, zorder=0)
    ax.axvspan(rx_split, rx_max, color="#e9eff6", alpha=0.65, zorder=0)
    ax.axhline(1.0, color="#555555", linestyle="--", zorder=1, label=r"Normal diffusion ($\alpha = 1$)")
    ax.axhline(2.0, color="#333333", linestyle="-.", zorder=1, label=r"Ballistic limit ($\alpha = 2$)")
    ax.axvline(rx_split, color="#444444", linestyle=":", zorder=1, label=r"Crossover ($R_c / \xi \approx 0.28$)")

    bg = "black" if is_dark() else "white"
    ax.text(np.sqrt(rx_min * rx_split), 2.24, "Superdiffusive\n" + r"($R_c / \xi < 0.28$)",
            ha="center", va="top", color="#8c2d38",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=bg, edgecolor="#e0b4b8", alpha=0.88), zorder=3)
    ax.text(np.sqrt(rx_split * rx_max), 2.24, "Diffusive\n" + r"($R_c / \xi > 0.28$)",
            ha="center", va="top", color="#3b526b",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=bg, edgecolor="#b8c7d9", alpha=0.88), zorder=3)

    offsets = {
        "beads06um": (8, 5, "left"), "beads1um": (8, -4, "left"), "beads3um": (8, 5, "left"),
        "beads7um": (8, -8, "left"), "beads5um": (8, 6, "left"), "beads20um": (-8, 8, "right"),
    }

    for _, r in df_out.iterrows():
        bname = r["bead_name"]
        color, marker = get_bead_style(bead_name=bname)
        yerr_lo = max(0.0, float(r["alpha_wls"]) - float(r["alpha_ci_lo"]))
        yerr_hi = max(0.0, float(r["alpha_ci_hi"]) - float(r["alpha_wls"]))
        ax.errorbar(r["rc_over_xi"], r["alpha_wls"], yerr=[[yerr_lo], [yerr_hi]],
                    fmt=marker, color=color, markersize=4.5, markeredgecolor="#222222",
                    ecolor=color, capsize=3.0, alpha=0.92, zorder=4)
        dx, dy, ha = offsets.get(bname, (6, 5, "left"))
        d_um = float(r["diameter_um"])
        ax.annotate(rf"{d_um:.2f} $\mu$m" if d_um < 10 else rf"{d_um:.1f} $\mu$m",
                    xy=(r["rc_over_xi"], r["alpha_wls"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", color=color, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(rx_min, rx_max)
    ax.set_ylim(0.7, 2.35)
    ax.set_xlabel(r"Dimensionless cargo radius $R_c / \xi$")
    ax.set_ylabel(r"Super-domain exponent $\alpha_{\mathrm{WLS}}$")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="lower left", framealpha=0.94)
