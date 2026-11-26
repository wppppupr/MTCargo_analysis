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
    ROOT / "figure" / "dimensionless_local_polar" / "dimensionless_local_polar_data.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/dimensionless_local_polar/dimensionless_local_polar_data.csv"),
]


def draw(fig):
    """Draw dimensionless local polar order Phi(R) vs R / xi (wo_cargo)."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\ndimensionless_local_polar_data.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    if "type" in df.columns and (df["type"] == "wo_cargo").any():
        df = df[df["type"] == "wo_cargo"]

    for cond, grp in df.groupby("condition", sort=False):
        d_um = float(grp["diameter_um"].iloc[0])
        color, marker = get_bead_style(d_um=d_um, bead_name=str(cond))
        stats = grp.groupby("window_size_px").agg(
            r_norm=("window_radius_norm", "mean"),
            phi_mean=("polar_order", "mean"),
            phi_sem=("polar_order", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
        ).sort_values("r_norm")

        r_norm = stats["r_norm"].to_numpy(dtype=float)
        phi_m = stats["phi_mean"].to_numpy(dtype=float)
        phi_s = stats["phi_sem"].to_numpy(dtype=float)

        ax.fill_between(r_norm, phi_m - phi_s, phi_m + phi_s, color=color, alpha=0.16, edgecolor="none")
        ax.errorbar(r_norm, phi_m, yerr=phi_s, marker=marker, linestyle="-", markersize=3.5,
                    capsize=2.0, color=color, label=f"{d_um:.2f} µm")

    ref_color = "white" if is_dark() else "#222222"
    ax.axvline(1.0, color=ref_color, linestyle="--", alpha=0.7, label=r"$R / \xi = 1$")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Normalized Window Radius $R / \xi$")
    ax.set_ylabel(r"Local Polar Order $\Phi(R)$")
    #ax.legend(loc="upper right", framealpha=0.9, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, linestyle="--", alpha=0.35)
