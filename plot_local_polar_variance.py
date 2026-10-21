#!/usr/bin/env python3
"""
plot_local_polar_variance.py
============================

半径 R の窓内における局所ポーラーオーダーの空間分散（感受率）:
    chi_Phi(R) = <Phi(R)^2>_r - <Phi(R)>_r^2

を光学フロー場 u(r, t) 全体から算出し、横軸 R [um]、縦軸 chi_Phi(R) のグラフを作成するスクリプト。
見た目（マーカー、配色、スタイル）は MSD.py に準拠。
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch (GPU CUDA) の確認
HAS_TORCH = False
TORCH_CUDA = False
try:
    import torch
    HAS_TORCH = True
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    pass

# スタイルの適用
CURRENT_DIR = Path(__file__).parent.resolve()
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# データルートディレクトリの候補 (MSD.py 準拠)
POSSIBLE_ROOTS = [
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

DEFAULT_RADII_PX = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(glob.glob(str(r / b / '*' / '*' / 'GFP_flows.h5'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return CURRENT_DIR


class DiskPolarVarianceCalculator:
    """
    2D FFT 畳み込みによる局所ポーラーオーダーの空間平均と空間分散の高速計算クラス。
    PyTorch CUDA が利用可能な場合は GPU を用い、極めて高速に計算。
    """
    def __init__(self, shape: Tuple[int, int], radii_px: List[int], device: Optional[str] = None):
        self.H, self.W = shape
        self.radii_px = sorted(list(radii_px))
        
        if device is None:
            self.use_cuda = HAS_TORCH and TORCH_CUDA
        else:
            self.use_cuda = (device == 'cuda') and HAS_TORCH and TORCH_CUDA
            
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu') if HAS_TORCH else None
        self._precompute_kernels()

    def _precompute_kernels(self):
        """円盤カーネルの 2D FFT を事前計算してキャッシュ"""
        self.kernel_ffts = []
        if self.use_cuda:
            for r in self.radii_px:
                ksize = int(2 * r + 1)
                ky, kx = torch.meshgrid(
                    torch.arange(-r, r + 1, device=self.device),
                    torch.arange(-r, r + 1, device=self.device),
                    indexing='ij'
                )
                kmask = (kx**2 + ky**2 <= r**2).float()
                kmask /= kmask.sum()

                pad_kernel = torch.zeros((self.H, self.W), device=self.device)
                pad_kernel[:r+1, :r+1] = kmask[r:, r:]
                pad_kernel[:r+1, -(r):] = kmask[r:, :r]
                pad_kernel[-(r):, :r+1] = kmask[:r, r:]
                pad_kernel[-(r):, -(r):] = kmask[:r, :r]

                k_fft = torch.fft.rfft2(pad_kernel)
                self.kernel_ffts.append(k_fft)
        else:
            import scipy.fft
            self.scipy_kernels = []
            for r in self.radii_px:
                ksize = int(2 * r + 1)
                ky, kx = np.ogrid[-r:r+1, -r:r+1]
                kmask = (kx**2 + ky**2 <= r**2).astype(np.float32)
                kmask /= kmask.sum()

                pad_kernel = np.zeros((self.H, self.W), dtype=np.float32)
                pad_kernel[:r+1, :r+1] = kmask[r:, r:]
                pad_kernel[:r+1, -(r):] = kmask[r:, :r]
                pad_kernel[-(r):, :r+1] = kmask[:r, r:]
                pad_kernel[-(r):, -(r):] = kmask[:r, :r]

                k_fft = scipy.fft.rfft2(pad_kernel, workers=4)
                self.scipy_kernels.append(k_fft)

    def compute_frame(self, m_x: np.ndarray, m_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        1フレームのフロー場から、各半径 R における局所ポーラーオーダーの空間平均 <Phi(R)>_r
        および空間分散 chi_Phi(R) = <Phi(R)^2>_r - <Phi(R)>_r^2 を算出。
        
        Returns:
            means: ndarray, shape (len(radii_px),)
            vars: ndarray, shape (len(radii_px),)
        """
        mag = np.hypot(m_x, m_y)
        with np.errstate(divide='ignore', invalid='ignore'):
            ux = np.where(mag > 0, m_x / mag, 0.0).astype(np.float32)
            uy = np.where(mag > 0, m_y / mag, 0.0).astype(np.float32)
        valid = (mag > 0).astype(np.float32)

        means = np.full(len(self.radii_px), np.nan, dtype=np.float64)
        variances = np.full(len(self.radii_px), np.nan, dtype=np.float64)

        if self.use_cuda:
            t_ux = torch.from_numpy(ux).to(self.device, non_blocking=True)
            t_uy = torch.from_numpy(uy).to(self.device, non_blocking=True)
            t_val = torch.from_numpy(valid).to(self.device, non_blocking=True)

            fft_ux = torch.fft.rfft2(t_ux)
            fft_uy = torch.fft.rfft2(t_uy)
            fft_val = torch.fft.rfft2(t_val)

            for idx, (r, k_fft) in enumerate(zip(self.radii_px, self.kernel_ffts)):
                conv_ux = torch.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W))
                conv_uy = torch.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W))
                conv_val = torch.fft.irfft2(fft_val * k_fft, s=(self.H, self.W))

                # 境界マージン & 有効画素率 > 0.5 の領域のみで評価
                mask_ok = (conv_val > 0.5)
                # 境界から r 画素内側
                if r < min(self.H, self.W) // 2:
                    mask_ok[:r, :] = False
                    mask_ok[-r:, :] = False
                    mask_ok[:, :r] = False
                    mask_ok[:, -r:] = False

                if torch.sum(mask_ok) > 50:
                    u_norm = conv_ux / torch.where(mask_ok, conv_val, torch.ones_like(conv_val))
                    v_norm = conv_uy / torch.where(mask_ok, conv_val, torch.ones_like(conv_val))
                    phi = torch.hypot(u_norm, v_norm)
                    phi_v = phi[mask_ok]
                    m1 = torch.mean(phi_v).item()
                    m2 = torch.mean(phi_v**2).item()
                    means[idx] = m1
                    variances[idx] = max(0.0, m2 - m1**2)
        else:
            import scipy.fft
            fft_ux = scipy.fft.rfft2(ux, workers=4)
            fft_uy = scipy.fft.rfft2(uy, workers=4)
            fft_val = scipy.fft.rfft2(valid, workers=4)

            for idx, (r, k_fft) in enumerate(zip(self.radii_px, self.scipy_kernels)):
                conv_ux = scipy.fft.irfft2(fft_ux * k_fft, s=(self.H, self.W), workers=4)
                conv_uy = scipy.fft.irfft2(fft_uy * k_fft, s=(self.H, self.W), workers=4)
                conv_val = scipy.fft.irfft2(fft_val * k_fft, s=(self.H, self.W), workers=4)

                mask_ok = (conv_val > 0.5)
                if r < min(self.H, self.W) // 2:
                    mask_ok[:r, :] = False
                    mask_ok[-r:, :] = False
                    mask_ok[:, :r] = False
                    mask_ok[:, -r:] = False

                if np.sum(mask_ok) > 50:
                    denom = np.where(mask_ok, conv_val, 1.0)
                    u_norm = conv_ux / denom
                    v_norm = conv_uy / denom
                    phi = np.hypot(u_norm, v_norm)
                    phi_v = phi[mask_ok]
                    m1 = float(np.mean(phi_v))
                    m2 = float(np.mean(phi_v**2))
                    means[idx] = m1
                    variances[idx] = max(0.0, m2 - m1**2)

        return means, variances


def process_experiment(
    flow_path: Path,
    radii_px: List[int],
    n_frames: int = 25,
    calculator: Optional[DiskPolarVarianceCalculator] = None
) -> Tuple[np.ndarray, np.ndarray, DiskPolarVarianceCalculator]:
    """
    1つの実験の GFP_flows.h5 からサンプリングフレームにわたり空間平均と分散を計算し時間平均を返す。
    """
    with h5py.File(str(flow_path), 'r') as f:
        key = list(f.keys())[0]
        flow_data = f[key]
        shape = flow_data.shape
        num_frames = shape[0]

        if shape[-1] == 2:
            H, W = shape[1], shape[2]
            channel_first = False
        else:
            H, W = shape[2], shape[3]
            channel_first = True

        if calculator is None or calculator.H != H or calculator.W != W:
            calculator = DiskPolarVarianceCalculator(shape=(H, W), radii_px=radii_px)

        frame_indices = np.linspace(0, num_frames - 1, min(n_frames, num_frames), dtype=int)

        all_means = []
        all_vars = []

        for t in frame_indices:
            if channel_first:
                m_x = flow_data[t, 0, ...].astype(np.float32)
                m_y = flow_data[t, 1, ...].astype(np.float32)
            else:
                m_x = flow_data[t, ..., 0].astype(np.float32)
                m_y = flow_data[t, ..., 1].astype(np.float32)

            m, v = calculator.compute_frame(m_x, m_y)
            all_means.append(m)
            all_vars.append(v)

    mean_phi_time = np.nanmean(np.array(all_means), axis=0)
    chi_phi_time = np.nanmean(np.array(all_vars), axis=0)

    return mean_phi_time, chi_phi_time, calculator


def main():
    parser = argparse.ArgumentParser(description="Calculate and plot spatial variance of local polar order chi_Phi(R).")
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing beads datasets.")
    parser.add_argument("--scale", type=float, default=0.11, help="Spatial scale [um/pixel] (default: 0.11)")
    parser.add_argument("--n_frames", type=int, default=25, help="Number of frames sampled per video (default: 25)")
    parser.add_argument("--radii", type=str, default=None, help="Comma-separated radii in pixels (default: pre-defined list)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures and csv.")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation even if cached CSV exists.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    scale = args.scale
    n_frames = args.n_frames

    if args.radii:
        radii_px = [int(r.strip()) for r in args.radii.split(",") if r.strip()]
    else:
        radii_px = DEFAULT_RADII_PX
    radii_um = np.array(radii_px, dtype=float) * scale

    # 出力先の設定
    out_dirs = [CURRENT_DIR / "figure" / "local_polar_variance"]
    if root_dir.exists() and (root_dir / "figure").exists():
        out_dirs.append(root_dir / "figure" / "local_polar_variance")
    if args.output_dir:
        out_dirs.insert(0, Path(args.output_dir))

    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)

    summary_csv_path = out_dirs[0] / "local_polar_variance_summary.csv"
    per_exp_csv_path = out_dirs[0] / "local_polar_variance_per_experiment.csv"

    print("==================================================================")
    print(" Local Polar Order Variance Analysis: chi_Phi(R)")
    print("   chi_Phi(R) = <Phi(R)^2>_r - <Phi(R)>_r^2")
    print(f" Root dir       : {root_dir}")
    print(f" Pixel scale    : {scale} um/pixel")
    print(f" Radii (pixels) : {radii_px}")
    print(f" Radii (um)     : {[round(r, 2) for r in radii_um]}")
    print(f" Sample frames  : {n_frames} frames / video")
    print(f" Output dir     : {out_dirs[0]}")
    print("==================================================================")

    # 実験データの検索
    all_experiments = []
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        flow_files = sorted(glob.glob(str(root_dir / b_name / "*" / "*" / "GFP_flows.h5")))
        for f in flow_files:
            all_experiments.append({
                "bead_name": b_name,
                "diameter_um": b_item["diameter_um"],
                "exp_name": Path(f).parent.name,
                "flow_path": Path(f)
            })

    print(f"Found {len(all_experiments)} experiments across {len(BEADS_INFO)} bead conditions.\n")

    calculator = None
    exp_results = []

    # 計算実行またはキャッシュ読込
    if summary_csv_path.exists() and per_exp_csv_path.exists() and not args.recompute:
        print(f"Loading existing results from {per_exp_csv_path}...")
        df_per_exp = pd.read_csv(per_exp_csv_path)
    else:
        print("Starting computation of local polar order variance...")
        t_start = time.time()
        for exp in tqdm(all_experiments, desc="Processing experiments"):
            flow_path = exp["flow_path"]
            try:
                mean_p, chi_p, calculator = process_experiment(
                    flow_path, radii_px=radii_px, n_frames=n_frames, calculator=calculator
                )
                for r_px, r_u, m, c in zip(radii_px, radii_um, mean_p, chi_p):
                    exp_results.append({
                        "bead_name": exp["bead_name"],
                        "diameter_um": exp["diameter_um"],
                        "exp_name": exp["exp_name"],
                        "radius_px": r_px,
                        "radius_um": r_u,
                        "mean_phi": m,
                        "chi_phi": c
                    })
            except Exception as e:
                print(f"\nWarning: Failed processing {flow_path}: {e}")

        df_per_exp = pd.DataFrame(exp_results)
        print(f"\nComputation completed in {time.time() - t_start:.2f} s.")
        for od in out_dirs:
            df_per_exp.to_csv(od / "local_polar_variance_per_experiment.csv", index=False)

    # -------------------------------------------------------------
    # 条件ごとの統計集約（Mean, SEM, STD, IQR）
    # -------------------------------------------------------------
    summary_rows = []
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        sub = df_per_exp[df_per_exp["bead_name"] == b_name]
        if sub.empty:
            continue

        for r_u, grp in sub.groupby("radius_um"):
            r_px = grp["radius_px"].iloc[0]
            vals = grp["chi_phi"].dropna().values
            n = len(vals)
            if n > 0:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                sem_v = std_v / np.sqrt(n) if n > 1 else 0.0
                q25_v = float(np.percentile(vals, 25))
                q75_v = float(np.percentile(vals, 75))
                median_v = float(np.median(vals))
            else:
                mean_v = sem_v = std_v = q25_v = q75_v = median_v = np.nan

            summary_rows.append({
                "bead_name": b_name,
                "diameter_um": d_um,
                "radius_px": r_px,
                "radius_um": r_u,
                "chi_phi_mean": mean_v,
                "chi_phi_sem": sem_v,
                "chi_phi_std": std_v,
                "chi_phi_median": median_v,
                "chi_phi_q25": q25_v,
                "chi_phi_q75": q75_v,
                "n_experiments": n
            })

    df_summary = pd.DataFrame(summary_rows)
    for od in out_dirs:
        df_summary.to_csv(od / "local_polar_variance_summary.csv", index=False)
    print(f"Saved summary CSV to {out_dirs[0] / 'local_polar_variance_summary.csv'}")

    # -------------------------------------------------------------
    # グラフ描画（MSD.py スタイル準拠）
    # -------------------------------------------------------------
    
    # 1. メイン図: 線形プロット chi_Phi(R) vs R
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.2))
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]
        sub = df_summary[df_summary["bead_name"] == b_name]
        if sub.empty:
            continue

        r = sub["radius_um"].values
        mean_v = sub["chi_phi_mean"].values
        sem_v = sub["chi_phi_sem"].values
        q25 = sub["chi_phi_q25"].values
        q75 = sub["chi_phi_q75"].values
        n_exp = sub["n_experiments"].iloc[0]

        # 四分位範囲 (IQR) の帯
        ax1.fill_between(r, q25, q75, color=color, alpha=0.18, edgecolor='none')
        # エラーバー付き主線 (Mean +- SEM)
        ax1.errorbar(
            r, mean_v, yerr=sem_v,
            marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
            label=f'{d_um:.2f} \u03bcm ($N={n_exp}$)', color=color,
            capsize=3.0, elinewidth=1.2
        )

    ax1.set_xlabel(r'Window Radius $R$ [$\mu\mathrm{m}$]', fontsize=13)
    ax1.set_ylabel(r'Variance $\chi_\Phi(R) \equiv \langle \Phi(R)^2 \rangle_{\mathbf{r}} - \langle \Phi(R) \rangle_{\mathbf{r}}^2$', fontsize=13)
    ax1.set_title(r'Local Polar Order Variance $\chi_\Phi(R)$ vs Window Radius $R$', fontsize=14, pad=12)
    ax1.legend(fontsize=10.5, loc='upper right', framealpha=0.9, markerscale=0.85, handlelength=1.5, borderpad=0.4)
    ax1.set_xlim(0, max(radii_um) * 1.05)
    ax1.set_ylim(bottom=0)

    for od in out_dirs:
        fig1.savefig(od / "local_polar_variance_vs_R.png", dpi=300, bbox_inches='tight')
        fig1.savefig(od / "local_polar_variance_vs_R.pdf", bbox_inches='tight')
    print(f"Saved Figure 1 (Linear) to {out_dirs[0] / 'local_polar_variance_vs_R.png'}")

    # 2. サブ図: 両対数プロット (Log-Log)
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.2))
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]
        sub = df_summary[df_summary["bead_name"] == b_name]
        if sub.empty:
            continue

        r = sub["radius_um"].values
        mean_v = sub["chi_phi_mean"].values
        sem_v = sub["chi_phi_sem"].values
        n_exp = sub["n_experiments"].iloc[0]

        valid = mean_v > 0
        ax2.errorbar(
            r[valid], mean_v[valid], yerr=sem_v[valid],
            marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
            label=f'{d_um:.2f} \u03bcm ($N={n_exp}$)', color=color,
            capsize=3.0, elinewidth=1.2
        )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Window Radius $R$ [$\mu\mathrm{m}$]', fontsize=13)
    ax2.set_ylabel(r'Variance $\chi_\Phi(R)$', fontsize=13)
    ax2.set_title(r'Local Polar Order Variance $\chi_\Phi(R)$ (Log-Log)', fontsize=14, pad=12)
    ax2.legend(fontsize=10.5, loc='lower left', framealpha=0.9, markerscale=0.85, handlelength=1.5, borderpad=0.4)

    for od in out_dirs:
        fig2.savefig(od / "local_polar_variance_vs_R_loglog.png", dpi=300, bbox_inches='tight')
        fig2.savefig(od / "local_polar_variance_vs_R_loglog.pdf", bbox_inches='tight')
    print(f"Saved Figure 2 (Log-Log) to {out_dirs[0] / 'local_polar_variance_vs_R_loglog.png'}")

    # 3. 6パネル個別プロット (各実験ごとの再現性確認)
    fig3, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]

        sub_exp = df_per_exp[df_per_exp["bead_name"] == b_name]
        for exp_name, grp in sub_exp.groupby("exp_name"):
            ax.plot(grp["radius_um"], grp["chi_phi"], color=color, alpha=0.35, linewidth=1.2, linestyle='--')

        sub_sum = df_summary[df_summary["bead_name"] == b_name]
        if not sub_sum.empty:
            ax.errorbar(
                sub_sum["radius_um"], sub_sum["chi_phi_mean"], yerr=sub_sum["chi_phi_sem"],
                marker=marker, color=color, linewidth=2.2, markersize=6.5,
                label=f'Mean $\pm$ SEM ($N={sub_sum["n_experiments"].iloc[0]}$)'
            )

        ax.set_title(f'{d_um:.2f} \u03bcm ({b_name})', fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(radii_um) * 1.05)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc='upper right')
        if idx >= 3:
            ax.set_xlabel(r'$R$ [$\mu\mathrm{m}$]', fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r'$\chi_\Phi(R)$', fontsize=11)

    fig3.suptitle(r'Local Polar Order Variance $\chi_\Phi(R)$ per Experiment', fontsize=14, y=0.99)
    plt.tight_layout()

    for od in out_dirs:
        fig3.savefig(od / "local_polar_variance_per_experiment.png", dpi=300, bbox_inches='tight')
        fig3.savefig(od / "local_polar_variance_per_experiment.pdf", bbox_inches='tight')
    print(f"Saved Figure 3 (6 Panels) to {out_dirs[0] / 'local_polar_variance_per_experiment.png'}")

    # =============================================================
    # 相関長 xi による規格化プロット (R / xi)
    # =============================================================
    print("\n--- Generating normalized plots: chi_Phi(R) vs R / xi ---")
    
    # 1. 相関長 xi の読み込み (実験別 CSV & 条件別 xi.csv)
    xi_per_exp = {}
    csv_per_exp_path = CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"
    if not csv_per_exp_path.exists():
        csv_per_exp_path = root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"
    if csv_per_exp_path.exists():
        try:
            df_xi_exp = pd.read_csv(csv_per_exp_path)
            for _, row in df_xi_exp.iterrows():
                e_name = str(row['exp_dir']).strip()
                if 'xi_um' in row and np.isfinite(row['xi_um']) and row['xi_um'] > 0:
                    xi_per_exp[e_name] = float(row['xi_um'])
        except Exception as e:
            print(f"Warning: Failed loading {csv_per_exp_path}: {e}")

    # 条件ごとの平均相関長
    cond_mean_xi = {}
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        matched_xis = [xi_per_exp[e["exp_name"]] for e in all_experiments if e["bead_name"] == b_name and e["exp_name"] in xi_per_exp]
        if matched_xis:
            cond_mean_xi[b_name] = float(np.mean(matched_xis))
        else:
            cond_mean_xi[b_name] = 5.0  # デフォルトフォールバック

    print("Correlation lengths xi [um] per condition:")
    for b_name, x_val in cond_mean_xi.items():
        print(f"  {b_name}: xi = {x_val:.3f} um")

    # 4. 規格化図: 線形プロット chi_Phi(R) vs R / xi
    fig4, ax4 = plt.subplots(figsize=(8.5, 6.2))
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]
        sub = df_summary[df_summary["bead_name"] == b_name]
        if sub.empty:
            continue

        xi = cond_mean_xi.get(b_name, 5.0)
        r_norm = sub["radius_um"].values / xi
        mean_v = sub["chi_phi_mean"].values
        sem_v = sub["chi_phi_sem"].values
        q25 = sub["chi_phi_q25"].values
        q75 = sub["chi_phi_q75"].values
        n_exp = sub["n_experiments"].iloc[0]

        # 四分位範囲 (IQR) の帯
        ax4.fill_between(r_norm, q25, q75, color=color, alpha=0.18, edgecolor='none')
        # エラーバー付き主線 (Mean +- SEM)
        ax4.errorbar(
            r_norm, mean_v, yerr=sem_v,
            marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
            label=f'{d_um:.2f} \u03bcm ($\\xi={xi:.2f}\\,\\mu\\mathrm{{m}}$)', color=color,
            capsize=3.0, elinewidth=1.2
        )

    # R / xi = 1 の縦線
    ax4.axvline(1.0, color='#222222', linestyle='--', linewidth=1.8, zorder=2, label=r'$R / \xi = 1$')

    ax4.set_xlabel(r'Normalized Window Radius $R / \xi$', fontsize=13)
    ax4.set_ylabel(r'Variance $\chi_\Phi(R) \equiv \langle \Phi(R)^2 \rangle_{\mathbf{r}} - \langle \Phi(R) \rangle_{\mathbf{r}}^2$', fontsize=13)
    ax4.set_title(r'Local Polar Order Variance $\chi_\Phi(R)$ vs $R / \xi$', fontsize=14, pad=12)
    ax4.legend(fontsize=10.0, loc='upper right', framealpha=0.9, markerscale=0.85, handlelength=1.5, borderpad=0.4)
    ax4.set_xlim(0, 10.0)
    ax4.set_ylim(bottom=0)

    for od in out_dirs:
        fig4.savefig(od / "local_polar_variance_vs_R_over_xi.png", dpi=300, bbox_inches='tight')
        fig4.savefig(od / "local_polar_variance_vs_R_over_xi.pdf", bbox_inches='tight')
    print(f"Saved Figure 4 (Normalized Linear) to {out_dirs[0] / 'local_polar_variance_vs_R_over_xi.png'}")

    # 5. 規格化図: 両対数プロット (Log-Log) chi_Phi(R) vs R / xi
    fig5, ax5 = plt.subplots(figsize=(8.5, 6.2))
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]
        sub = df_summary[df_summary["bead_name"] == b_name]
        if sub.empty:
            continue

        xi = cond_mean_xi.get(b_name, 5.0)
        r_norm = sub["radius_um"].values / xi
        mean_v = sub["chi_phi_mean"].values
        sem_v = sub["chi_phi_sem"].values
        n_exp = sub["n_experiments"].iloc[0]

        valid = (mean_v > 0) & (r_norm > 0)
        ax5.errorbar(
            r_norm[valid], mean_v[valid], yerr=sem_v[valid],
            marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
            label=f'{d_um:.2f} \u03bcm ($\\xi={xi:.2f}\\,\\mu\\mathrm{{m}}$)', color=color,
            capsize=3.0, elinewidth=1.2
        )

    # R / xi = 1 の縦線
    ax5.axvline(1.0, color='#222222', linestyle='--', linewidth=1.8, zorder=2, label=r'$R / \xi = 1$')

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel(r'Normalized Window Radius $R / \xi$', fontsize=13)
    ax5.set_ylabel(r'Variance $\chi_\Phi(R)$', fontsize=13)
    ax5.set_title(r'Local Polar Order Variance $\chi_\Phi(R)$ vs $R / \xi$ (Log-Log)', fontsize=14, pad=12)
    ax5.legend(fontsize=10.0, loc='lower left', framealpha=0.9, markerscale=0.85, handlelength=1.5, borderpad=0.4)
    ax5.set_xlim(left=0.08, right=15.0)

    for od in out_dirs:
        fig5.savefig(od / "local_polar_variance_vs_R_over_xi_loglog.png", dpi=300, bbox_inches='tight')
        fig5.savefig(od / "local_polar_variance_vs_R_over_xi_loglog.pdf", bbox_inches='tight')
    print(f"Saved Figure 5 (Normalized Log-Log) to {out_dirs[0] / 'local_polar_variance_vs_R_over_xi_loglog.png'}")

    print("\nAll processing and plotting successfully finished!")


if __name__ == "__main__":
    main()
