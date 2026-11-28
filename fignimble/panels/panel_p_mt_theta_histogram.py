from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    def is_dark():
        return False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "kl_angle_histograms_B36.csv",
    ROOT / "kl_angle_histograms_B36.csv",
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/per_point_v_dot_u/kl_angle_histograms_B36.csv"),
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/kl_angle_histograms_B36.csv"),
]


def _fmt_diam(d_um: float) -> str:
    v = float(d_um)
    if abs(v - 5.0) < 0.05:
        return "5.0"
    if abs(v - 20.0) < 0.5:
        return "20"
    return f"{v:.2f}"


def source_data(yscale="linear"):
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def draw(fig, yscale="linear"):
    """Draw MT disk relative-angle distribution P_MT(theta) used for D_KL^MT (B = 36 bins)."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(
            0.5, 0.5,
            "Missing:\nkl_angle_histograms_B36.csv\nRun analysis/calc_kl_angle_histograms.py",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    iso_col = "#bbbbbb" if is_dark() else "#555555"
    p_iso = float(df["p_iso"].iloc[0]) if "p_iso" in df.columns else 1.0 / (2.0 * np.pi)
    ax.axhline(p_iso, color=iso_col, linestyle="--", label=r"Uniform $1/(2\pi)$", zorder=1)

    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp.sort_values("theta_rad")
        bname = str(grp["bead_name"].iloc[0]) if "bead_name" in grp.columns else None
        color, marker = get_bead_style(d_um=d_um, bead_name=bname)
        x = grp["theta_rad"].to_numpy(dtype=float)
        y = grp["pdf_mt"].to_numpy(dtype=float)
        valid = np.isfinite(y)
        if not np.any(valid):
            continue

        y_plot = np.clip(y[valid], 1e-4, None) if yscale == "log" else y[valid]
        ax.plot(
            x[valid], y_plot, marker=marker, linestyle="-", markersize=3.0,
            color=color, label=f"{_fmt_diam(d_um)} µm", zorder=4,
        )

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 5.0)
    else:
        ax.set_ylim(0.0, 3.1)
    ax.set_xlabel(r"Relative angle $\theta = \theta_{\mathrm{MT}} - \theta_{\mathrm{bulk}}$ [rad]")
    ax.set_ylabel(r"Probability density $P_{\mathrm{MT}}(\theta)$ [$\mathrm{rad}^{-1}$]")
    ax.legend(loc="upper right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
