#!/usr/bin/env python3
"""
effective_diffusion_analysis.py

貨物微粒子の速度自己相関関数 (VACF) のGreen-Kubo積分および
Gaussian HMM パラメータに基づくRun-and-Tumble Particle (RTP) 理論モデルから
有効拡散係数 D_eff を一括算出・比較・可視化するスクリプトです。
"""

import argparse
import sys
from pathlib import Path

# libsディレクトリから親ディレクトリをパスに追加
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs.effective_diffusion import run_effective_diffusion_analysis
from libs.vacf_analysis import find_default_root


def main():
    parser = argparse.ArgumentParser(
        description="Cargo particle effective diffusion coefficient analysis: Green-Kubo integration vs microscopic HMM RTP theory."
    )
    parser.add_argument(
        '--root_dir', type=str, default=None,
        help="Root directory containing bead conditions (e.g. /Volumes/data/Sasaki/MTsingleBeads)."
    )
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help="Directory to save output figures and CSV summary (default: <workspace>/figure/effective_diffusion)."
    )
    parser.add_argument(
        '--hmm_summary', type=str, default=None,
        help="Path to HMM state parameters CSV (default: figure/hmm_1d/hmm_state_parameters_summary_k2.csv)."
    )
    parser.add_argument(
        '--angle_summary', type=str, default=None,
        help="Path to turning angle summary CSV (default: figure/hmm_1d/hmm_turning_angle_summary_k2.csv)."
    )
    parser.add_argument(
        '--abp_summary', type=str, default=None,
        help="Path to Run ABP MSD fits summary CSV (default: figure/hmm_1d/hmm_run_abp_fits_summary_k2.csv)."
    )
    parser.add_argument(
        '--max_lag', type=int, default=50,
        help="Maximum lag time in frames for VACF calculation (default: 50 frames = 200s)."
    )
    parser.add_argument(
        '--frame_interval', type=float, default=4.0,
        help="Frame interval in seconds (default: 4.0)."
    )
    parser.add_argument(
        '--scale', type=float, default=0.11,
        help="Spatial scale in um/pixel (default: 0.11)."
    )
    parser.add_argument(
        '--max_lag_time', type=float, default=200.0,
        help="Upper integration limit for Green-Kubo integral in seconds (default: 200.0)."
    )

    args = parser.parse_args()

    root_dir = args.root_dir if args.root_dir is not None else find_default_root()

    run_effective_diffusion_analysis(
        root_dir=root_dir,
        out_dir=args.out_dir,
        hmm_summary_path=args.hmm_summary,
        angle_summary_path=args.angle_summary,
        abp_summary_path=args.abp_summary,
        max_timeshift_frames=args.max_lag,
        frame_interval=args.frame_interval,
        scale=args.scale,
        max_lag_time=args.max_lag_time
    )



if __name__ == "__main__":
    main()
