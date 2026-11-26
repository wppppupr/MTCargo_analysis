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
    ROOT / "figure" / "displacement" / "displacement_pdf_norm_tau100s_summary.csv",
    ROOT / "figure" / "displacement_pdf_norm_tau100s_summary.csv",
    ROOT / "displacement_pdf_norm_tau100s_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/displacement/displacement_pdf_norm_tau100s_summary.csv"),
]


def draw(fig):
    """Draw displacement PDF at lag time 100 s."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(
            0.5,
            0.5,
            "Missing:\ndisplacement_pdf_norm_tau100s_summary.csv\nRun analysis/calc_displacement_pdf.py",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp.sort_values("center_um")
        valid = grp["mean_pdf"] > 0
        grp = grp[valid]
        if grp.empty:
            continue

        color, marker = get_bead_style(d_um=d_um)
        x = grp["center_um"].to_numpy(dtype=float)
        y = grp["mean_pdf"].to_numpy(dtype=float)

        if "boot_ci_low" in grp.columns and "boot_ci_high" in grp.columns:
            ci_lo = grp["boot_ci_low"].to_numpy(dtype=float)
            ci_hi = grp["boot_ci_high"].to_numpy(dtype=float)
            ax.fill_between(x, np.clip(ci_lo, 1e-8, None), ci_hi, color=color, alpha=0.18, edgecolor="none")

        n_exps = int(grp["n_exps"].iloc[0]) if "n_exps" in grp.columns else 0
        label = f"{d_um:.2f} µm ($N={n_exps}$)" if n_exps > 0 else f"{d_um:.2f} µm"

        ax.plot(x, y, marker=marker, linestyle="-", markersize=3.5, color=color, label=label, alpha=0.9)

        if "fit_y" in grp.columns and grp["fit_y"].notna().any():
            ax.plot(x, grp["fit_y"].to_numpy(dtype=float), linestyle="--", color=color, alpha=0.8)

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"Displacement $|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"PDF $P(|\Delta \boldsymbol{r}|)$ [$\mu\mathrm{m}^{-1}$]")
    #ax.legend(loc="upper right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, which="both", linestyle="--", alpha=0.35)
