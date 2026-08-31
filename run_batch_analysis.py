import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path

BEADS_CONFIG = {
    'beads06um': {'radius': 3, 'diameter_um': 0.63},
    'beads1um':  {'radius': 6, 'diameter_um': 1.0},
    'beads3um':  {'radius': 15, 'diameter_um': 3.0},
    'beads5um':  {'radius': 25, 'diameter_um': 5.0},
    'beads7um':  {'radius': 35, 'diameter_um': 7.0},
    'beads20um': {'radius': 95, 'diameter_um': 20.0},
}

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]

def find_default_root():
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]

def find_target_directories(root_dir, bead_name):
    """
    Search for valid experiment directories under root_dir / bead_name / * / *
    or root_dir / bead_name / *
    A directory is valid if it contains 'GFP_flows.h5' and 'beads_tracks.csv'.
    """
    candidates = []
    base = root_dir / bead_name
    if not base.exists():
        return []

    # Search depth 1 and 2
    for p in base.glob("*/*"):
        if p.is_dir() and (p / "GFP_flows.h5").exists() and (p / "beads_tracks.csv").exists():
            candidates.append(p)

    if not candidates:
        for p in base.glob("*"):
            if p.is_dir() and (p / "GFP_flows.h5").exists() and (p / "beads_tracks.csv").exists():
                candidates.append(p)

    return sorted(candidates)

def run_cmd(cmd):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[WARNING] Command failed with return code {ret.returncode}")
    return ret.returncode

def main():
    parser = argparse.ArgumentParser(description="Batch runner for local polar order and angular spatial correlation analysis.")
    parser.add_argument('--beads', type=str, nargs='+', default=['all'],
                        help="Beads conditions to analyze (e.g. beads06um beads1um beads3um beads5um beads7um beads20um, or 'all')")
    parser.add_argument('--mode', type=str, default='polar', choices=['polar', 'corr', 'all'],
                        help="Analysis mode: 'polar' (calc_local_polar + bg), 'corr' (calc_angular_spatial_correlation + bg), or 'all'")
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing beads data. If None, checks default NAS paths.")
    parser.add_argument('--target_dir', type=str, default=None,
                        help="Explicit target directory to process directly (ignores --beads and --root_dir).")
    parser.add_argument('--particle_radius', type=int, default=None,
                        help="Override particle radius (pixels). If None, uses default per bead size.")
    args = parser.parse_args()

    python_exec = sys.executable

    # Single directory mode
    if args.target_dir is not None:
        target_path = Path(args.target_dir)
        p_radius = args.particle_radius if args.particle_radius is not None else 6
        print(f"Processing explicit directory: {target_path} (particle_radius={p_radius})")

        if args.mode in ['polar', 'all']:
            run_cmd([python_exec, "libs/calc_bg_polar.py", str(target_path)])
            run_cmd([python_exec, "libs/calc_local_polar.py", str(target_path), "--particle_radius", str(p_radius)])

        if args.mode in ['corr', 'all']:
            run_cmd([python_exec, "libs/calc_bg_angular_correlation.py", str(target_path)])
            run_cmd([python_exec, "libs/calc_angular_spatial_correlation.py", str(target_path), "--particle_radius", str(p_radius)])
        return

    # Root directory resolution
    if args.root_dir is not None:
        root_dir = Path(args.root_dir)
    else:
        root_dir = find_default_root()

    print(f"Using root directory: {root_dir}")

    # Determine which bead sizes to process
    if 'all' in args.beads:
        selected_beads = list(BEADS_CONFIG.keys())
    else:
        selected_beads = []
        for b in args.beads:
            for item in b.split(','):
                item = item.strip()
                if item == 'all':
                    selected_beads = list(BEADS_CONFIG.keys())
                    break
                if item in BEADS_CONFIG:
                    selected_beads.append(item)
                else:
                    print(f"[WARNING] Unknown bead size '{item}'. Choose from {list(BEADS_CONFIG.keys())}")

    print(f"Target bead conditions: {selected_beads}")

    total_dirs = 0
    for bead_name in selected_beads:
        cfg = BEADS_CONFIG[bead_name]
        p_radius = args.particle_radius if args.particle_radius is not None else cfg['radius']
        dirs = find_target_directories(root_dir, bead_name)
        print(f"\n========================================================")
        print(f"Condition: {bead_name} | Found {len(dirs)} experiments | particle_radius={p_radius}")
        print(f"========================================================")

        for d in dirs:
            total_dirs += 1
            print(f"\n---> Processing: {d.relative_to(root_dir)}")

            if args.mode in ['polar', 'all']:
                run_cmd([python_exec, "libs/calc_bg_polar.py", str(d)])
                run_cmd([python_exec, "libs/calc_local_polar.py", str(d), "--particle_radius", str(p_radius)])

            if args.mode in ['corr', 'all']:
                run_cmd([python_exec, "libs/calc_bg_angular_correlation.py", str(d)])
                run_cmd([python_exec, "libs/calc_angular_spatial_correlation.py", str(d), "--particle_radius", str(p_radius)])

    print(f"\n[DONE] Batch processing completed. Total directories processed: {total_dirs}")

if __name__ == "__main__":
    main()
