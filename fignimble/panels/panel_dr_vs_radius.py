from pathlib import Path
import pandas as pd
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "figure" / "msad" / "dr_vs_cargo_radius_summary.csv"


def draw(fig):
    """Draw rotational diffusion Dr vs cargo radius Rc."""
    ax = fig.add_subplot(111)
    if not CSV_PATH.exists():
        ax.text(0.5, 0.5, f"Missing:\n{CSV_PATH.name}\nRun analysis/calc_msad.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(CSV_PATH).sort_values("radius_um")
    rc_arr = df["radius_um"].to_numpy(dtype=float)
    dr_arr = df["Dr_4_40s_ens_rad2_s"].to_numpy(dtype=float)

    for _, row in df.iterrows():
        ax.errorbar(row["radius_um"], row["Dr_4_40s_ens_rad2_s"], yerr=row["Dr_4_40s_part_sem_rad2_s"],
                    fmt=row["marker"], color=row["color"], ecolor=row["color"], markersize=4.5, capsize=3.0,
                    label=f"{row['diameter_um']:.2f} µm", zorder=4)

    ax.plot(rc_arr, dr_arr, color="#555555", linestyle="--", alpha=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Cargo Radius $R_c$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Rotational Diffusion $D_r$ [$\mathrm{rad}^2/\mathrm{s}$]")
    ax.set_xlim(0.25, 15.0)
    ax.set_ylim(0.0, 0.65)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", framealpha=0.9, handlelength=1.4)
