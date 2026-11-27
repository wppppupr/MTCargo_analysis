#!/usr/bin/env python3
"""Batch-export individual Fignimble panel modules (fignimble/panels/panel_*.py) to PDF/SVG/PNG."""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGNIMBLE_DIR = REPO_ROOT / "fignimble"
if str(FIGNIMBLE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGNIMBLE_DIR))


def configure_style(font_size: float, linewidth: float) -> None:
    """Configure matplotlib rcParams to match fignimble/style.py with editable vector fonts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": max(font_size - 1.0, 5.0),
        "ytick.labelsize": max(font_size - 1.0, 5.0),
        "legend.fontsize": max(font_size - 1.5, 5.0),
        "lines.linewidth": linewidth,
        "axes.linewidth": linewidth * 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def despine_figure(fig: plt.Figure) -> None:
    """Hide top spine on all axes, and right spine on single-axis plots (preserving twinx)."""
    is_single_axis = len(fig.axes) == 1
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        if is_single_axis:
            ax.spines["right"].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-export Fignimble panels to PDF/SVG/PNG.")
    parser.add_argument("--out-dir", type=Path, default=FIGNIMBLE_DIR / "exports" / "panels",
                        help="Output directory (default: fignimble/exports/panels)")
    parser.add_argument("--width", type=float, default=90.0, help="Panel width in mm (default: 90.0)")
    parser.add_argument("--height", type=float, default=70.0, help="Panel height in mm (default: 70.0)")
    parser.add_argument("--font-size", type=float, default=9.0, help="Base font size in pt (default: 9.0)")
    parser.add_argument("--linewidth", type=float, default=1.0, help="Line width in pt (default: 1.0)")
    parser.add_argument("--format", choices=["pdf", "svg", "png"], default="pdf", help="Output format (default: pdf)")
    parser.add_argument("--only", type=str, default=None, help="Optional substring filter for panel filename")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output (default: 300)")
    args = parser.parse_args()

    configure_style(args.font_size, args.linewidth)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel_dir = FIGNIMBLE_DIR / "panels"
    panel_files = sorted(panel_dir.glob("panel_*.py"))
    if args.only:
        panel_files = [p for p in panel_files if args.only in p.stem]

    n_ok, n_fail, n_skip = 0, 0, 0
    figsize = (args.width / 25.4, args.height / 25.4)

    for p_path in panel_files:
        stem = p_path.stem
        try:
            mod = importlib.import_module(f"panels.{stem}")
        except Exception as exc:
            print(f"[FAIL] {stem}: import error ({exc})")
            n_fail += 1
            continue

        draw_fn = getattr(mod, "draw", None)
        if not callable(draw_fn):
            n_skip += 1
            continue

        fig = plt.figure(figsize=figsize)
        try:
            draw_fn(fig)
            despine_figure(fig)
            fig.tight_layout()
            out_file = args.out_dir / f"{stem}.{args.format}"
            fig.savefig(out_file, dpi=args.dpi, bbox_inches="tight")
            print(f"[OK]   {out_file.relative_to(REPO_ROOT)}")
            n_ok += 1
        except Exception as exc:
            print(f"[FAIL] {stem}: {exc}")
            n_fail += 1
        finally:
            plt.close(fig)

    print(f"\nExported {n_ok} panel(s) to {args.out_dir} (skipped: {n_skip}, failed: {n_fail})")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
