r"""
plot_relative_modulation_heatmap.py

横軸 x = R_c / \xi_{i,t}、縦軸 M (局所磁化・配向秩序度) に対する
相対感度変調比:
    R(x, M) = \frac{\beta(x) M}{y_0(x)}   (または \frac{\beta(x) M}{|y_0(x)|})
のヒートマップおよび理論的マスター相図を作成するスクリプト。

【作成するグラフ】
1. 離散タイル型マトリクスヒートマップ (Discrete Matrix Heatmap)
2. 連続 2D 等高線ヒートマップ (Continuous 2D Contour Heatmap)
3. 理論幾何解 <M^2(x)> を重ね合わせた「理論的マスター相図」 (Universal Master Phase Diagram)
4. 対数スケール横軸 (log(x) vs M) マスター相図
5. 4パネル統合エグゼクティブ図 (Summary Phase Diagrams)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import integrate
import scipy.special as sp
from scipy.interpolate import interp1d, PchipInterpolator

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Paul Tol の公式 BuRd (Blue-Red) カラーマップ定義
paul_tol_burd_data = [
    (0.129, 0.400, 0.675),  # 濃い青
    (0.263, 0.576, 0.765),  # 青
    (0.573, 0.773, 0.871),  # 明るい青
    (0.819, 0.898, 0.941),  # 極めて淡い青
    (0.968, 0.968, 0.968),  # ニュートラル白 (ゼロ境界)
    (0.992, 0.859, 0.780),  # 極めて淡い赤
    (0.957, 0.647, 0.509),  # 明るい赤
    (0.839, 0.376, 0.302),  # 赤
    (0.690, 0.149, 0.173),  # 濃い赤
]
cmap_burd = LinearSegmentedColormap.from_list(
    'PaulTol_BuRd', paul_tol_burd_data, N=256
)

CMAP=cmap_burd

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        style_colors = [f"#{c}" if not c.startswith('#') else c for c in style_colors]
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']

# ビーズ基本情報 (全6サイズ)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "label": "0.63 μm", "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "label": "1.18 μm", "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "label": "3.37 μm", "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "label": "5.00 μm", "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "label": "7.24 μm", "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "label": "20.0 μm", "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]


def get_default_root_dir() -> Path:
    for p in POSSIBLE_ROOTS:
        if p.exists():
            return p
    return CURRENT_DIR


def save_figure_to_all(fig: plt.Figure, name_base: str, out_dirs: List[Path], dpi: int = 300):
    """PNG と SVG 形式で複数ディレクトリに図を保存"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / f"{name_base}.png"
        svg_path = d / f"{name_base}.svg"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {name_base}.png / .svg -> {len(out_dirs)} dir(s)")


# =============================================================================
# 理論幾何解 <M^2(x)> の厳密解
# =============================================================================

_THEORY_GRID_X = np.logspace(-3, 3, 300)
_THEORY_GRID_M2 = None


def _calc_theoretical_m2_disk(x_val: float) -> float:
    r"""
    2次元円形領域（半径 R）における指数相関場 C(r) = exp(-r/xi) の
    磁化2乗平均 <M^2>(x) (x = R / xi) の厳密解:
        <M^2(x)> = (2 / x^2) * { 1 + 2 * [I_0(2x) - L_0(2x)] - (3 / x) * [I_1(2x) - L_1(2x)] }
    """
    x = float(x_val)
    if x <= 0:
        return 1.0
    if x < 1e-4:
        return float(1.0 - (128.0 / (45.0 * np.pi)) * x)
    if x > 15.0:
        val, _ = integrate.quad(
            lambda w: (16.0 / np.pi) * w * (np.arccos(w) - w * np.sqrt(np.maximum(0.0, 1.0 - w**2))) * np.exp(-2.0 * x * w),
            0.0, 1.0, limit=100
        )
        return float(val)

    z = 2.0 * x
    term0 = sp.i0(z) - sp.modstruve(0, z)
    term1 = sp.i1(z) - sp.modstruve(1, z)
    val = (2.0 / (x**2)) * (1.0 + 2.0 * term0 - (3.0 / x) * term1)
    return float(np.clip(val, 0.0, 1.0))


def theoretical_m_rms_curve(x_array: np.ndarray) -> np.ndarray:
    """理論普遍関数 M_rms(x) = sqrt(<M^2>(x)) を算出"""
    global _THEORY_GRID_M2
    if _THEORY_GRID_M2 is None:
        _THEORY_GRID_M2 = np.array([_calc_theoretical_m2_disk(x) for x in _THEORY_GRID_X])

    xs = np.asarray(x_array, dtype=float)
    log_x = np.log10(np.clip(xs, 1e-4, 1e4))
    log_grid_x = np.log10(_THEORY_GRID_X)
    log_grid_m2 = np.log10(np.maximum(_THEORY_GRID_M2, 1e-12))
    log_m2_interp = np.interp(log_x, log_grid_x, log_grid_m2)
    m2_val = np.power(10.0, log_m2_interp)
    return np.sqrt(m2_val)


# =============================================================================
# データ読み出し・補間モデル
# =============================================================================

def load_fit_summary_data() -> pd.DataFrame:
    """線形回帰サマリー CSV を読み出す"""
    csv_path = CURRENT_DIR / "figure" / "conditional_log_velocity" / "conditional_log_velocity_linear_fit_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fit summary CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df_sub = df[df["bead_name"] != "overall_pooled"].dropna(subset=["rc_over_xi_mean"]).copy()
    df_sub = df_sub.sort_values(by="rc_over_xi_mean").reset_index(drop=True)

    # CSV に直接記録された rc_over_xi_std を使用（ない場合はフォールバック）
    if "rc_over_xi_std" not in df_sub.columns:
        if "rc_over_xi_sem" in df_sub.columns:
            df_sub["rc_over_xi_std"] = df_sub["rc_over_xi_sem"]
        else:
            df_sub["rc_over_xi_std"] = 0.0
    return df_sub


def build_interpolators(df_fit: pd.DataFrame):
    """x = Rc / xi に対する beta(x) と y0(x) の滑らかな補間関数 (PCHIP) を作成"""
    x_nodes = df_fit["rc_over_xi_mean"].values
    beta_nodes = df_fit["ln_beta_slope"].values
    y0_nodes = df_fit["ln_y0_intercept"].values

    interp_beta = PchipInterpolator(x_nodes, beta_nodes, extrapolate=True)
    interp_y0 = PchipInterpolator(x_nodes, y0_nodes, extrapolate=True)

    # クロスオーバー点 beta(x_c) = 0 の推定
    x_dense_cross = np.linspace(x_nodes.min(), x_nodes.max(), 2000)
    beta_dense_cross = interp_beta(x_dense_cross)
    idx_cross = np.where(np.diff(np.sign(beta_dense_cross)))[0]
    if len(idx_cross) > 0:
        x_crossover = float(x_dense_cross[idx_cross[0]])
    else:
        x_crossover = 0.85

    return interp_beta, interp_y0, x_crossover


# =============================================================================
# 作図 1: 離散タイル型マトリクスヒートマップ (Discrete Matrix Heatmap)
# =============================================================================

def plot_discrete_matrix_heatmap(df_fit: pd.DataFrame, out_dirs: List[Path]):
    """
    各ビーズサイズ（離散列）× M のビン（離散行）における
    相対変調比 Pi = (beta * M) / y0 のタイル型ヒートマップ
    """
    m_bins = np.linspace(0.05, 0.95, 10)  # M の代表点 (0.0~1.0, 10刻み中心)
    n_m = len(m_bins)
    n_beads = len(df_fit)

    ratio_matrix = np.zeros((n_m, n_beads))

    for col_idx, (_, row) in enumerate(df_fit.iterrows()):
        beta = row["ln_beta_slope"]
        y0 = row["ln_y0_intercept"]
        ratio_matrix[:, col_idx] = (beta * m_bins) / y0

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    vmax = np.max(np.abs(ratio_matrix))
    vmax = max(1.5, vmax)

    im = ax.imshow(
        ratio_matrix,
        cmap=CMAP,
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        extent=[-0.5, n_beads - 0.5, 0.0, 1.0]
    )

    # セル内数値アノテーション
    for i in range(n_m):
        for j in range(n_beads):
            val = ratio_matrix[i, j]
            text_color = "white" if np.abs(val) > vmax * 0.55 else "black"
            ax.text(
                j, m_bins[i], f"{val:+.2f}",
                ha="center", va="center", fontsize=9.0, fontweight="bold",
                color=text_color
            )

    # 軸ラベル・目盛り
    x_tick_labels = [
        f"{row['label']}\n($x={row['rc_over_xi_mean']:.2f}$)"
        for _, row in df_fit.iterrows()
    ]
    ax.set_xticks(range(n_beads))
    ax.set_xticklabels(x_tick_labels, fontsize=10.5)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.set_yticklabels([f"{m:.1f}" for m in np.linspace(0.0, 1.0, 11)], fontsize=11.0)

    ax.set_xlabel(r"Cargo Diameter $d$ and Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=12.5, fontweight='bold')
    ax.set_ylabel(r"Magnetization $M$ (Orientation Order)", fontsize=12.5, fontweight='bold')
    ax.set_title(
        r"Relative Velocity Modulation Ratio $\Pi(x, M) = \frac{\beta(x) M}{y_0(x)}$ [Matrix Heatmap]" + "\n" +
        r"($\ln\tilde{v} = y_0(x)[1 + \Pi]$, Linear Fit with QQ Filter)",
        fontsize=13.0, fontweight='bold', pad=12
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Velocity Modulation Ratio $\Pi = \frac{\beta M}{y_0}$", fontsize=12.0)

    plt.tight_layout()
    save_figure_to_all(fig, "heatmap_relative_modulation_matrix_discrete", out_dirs)


# =============================================================================
# 作図 2: 連続 2D 等高線ヒートマップ (Continuous 2D Contour Heatmap)
# =============================================================================

def plot_continuous_contour_heatmap(
    df_fit: pd.DataFrame,
    interp_beta,
    interp_y0,
    x_crossover: float,
    out_dirs: List[Path]
):
    r"""
    (x, M) 連続平面における \Pi(x, M) = \frac{\beta(x) M}{y_0(x)} のカラーマップ＋等高線
    """
    x_min, x_max = 0.01, 1.15
    m_min, m_max = 0.0, 1.0

    X, M = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(m_min, m_max, 200))
    Beta_grid = interp_beta(X)
    Y0_grid = interp_y0(X)
    Ratio_grid = (Beta_grid * M) / Y0_grid

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    vlim = 1.5

    cp = ax.contourf(
        X, M, Ratio_grid,
        levels=np.linspace(-vlim, vlim, 61),
        cmap=CMAP,
        extend="both"
    )

    # 等値線 (Contours)
    levels_lines = np.array([-1.2, -0.8, -0.5, -0.3, -0.1, 0.0, 0.2, 0.5, 0.8, 1.2])
    cs = ax.contour(X, M, Ratio_grid, levels=levels_lines, colors="black", linewidths=0.9, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=8.5, fmt="%+.1f")

    # クロスオーバー境界線 (beta = 0 -> ratio = 0)
    ax.axvline(x_crossover, color="darkgreen", ls="--", lw=2.2, label=rf"Crossover $x_c \approx {x_crossover:.2f}$ ($\Pi=0$)")

    # 実験データ点のプロット (M の平均値と STD エラーバー, x 方向 STD エラーバー)
    for _, row in df_fit.iterrows():
        b_name = row["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        x_pt = row["rc_over_xi_mean"]
        x_std = row["rc_over_xi_std"] if "rc_over_xi_std" in row else row["rc_over_xi_sem"]
        m_mean = row["m_mean"] if "m_mean" in row and not np.isnan(row["m_mean"]) else 0.5
        m_std = row["m_std"] if "m_std" in row and not np.isnan(row["m_std"]) else 0.3

        ax.errorbar(
            x_pt, m_mean,
            xerr=x_std, yerr=m_std,
            fmt=b_info["marker"], color=b_info["color"], ecolor="black",
            elinewidth=1.5, capsize=4.0, capthick=1.2, markersize=9.0,
            markeredgecolor="black", markeredgewidth=1.2, zorder=6,
            label=f"$d = {b_info['label']}$"
        )

    ax.set_xlim(0.0, 1.20)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13.0, fontweight='bold')
    ax.set_ylabel(r"Magnetization $M$", fontsize=13.0, fontweight='bold')
    ax.set_title(
        r"2D Continuous Phase Diagram: $\Pi(x, M) = \frac{\beta(x) M}{y_0(x)}$ vs $(x, M)$" + "\n" +
        r"(Blue: Acceleration / Sensitivity Enhancement, Red: Steric Deceleration)",
        fontsize=13.0, fontweight='bold', pad=14
    )
    ax.grid(True, which='major', ls=':', alpha=0.5, color='gray')
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

    cbar = fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Velocity Modulation Ratio $\Pi(x, M) = \frac{\beta(x) M}{y_0(x)}$", fontsize=12.0)

    plt.tight_layout()
    save_figure_to_all(fig, "heatmap_relative_modulation_continuous_2d", out_dirs)


# =============================================================================
# 作図 3: 理論幾何解 <M^2(x)> を重ね合わせた「理論的マスター相図」
# =============================================================================

def plot_universal_master_phase_diagram(
    df_fit: pd.DataFrame,
    interp_beta,
    interp_y0,
    x_crossover: float,
    out_dirs: List[Path],
    log_x: bool = False
):
    """
    理論幾何解 <M^2(x)> の RMS 磁化 M_rms(x) = sqrt(<M^2>(x)) を重ね合わせた
    『普遍的マスター相図』 (Universal Master Phase Diagram)
    理論曲線 M_rms(x) 上に横軸 STD エラーバー付きで実験データをオーバーレイ
    """
    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    if log_x:
        x_grid = np.logspace(-2.0, 0.4, 250)
    else:
        x_grid = np.linspace(0.01, 1.20, 250)

    m_grid = np.linspace(0.0, 1.0, 200)
    X, M = np.meshgrid(x_grid, m_grid)

    Beta_grid = interp_beta(X)
    Y0_grid = interp_y0(X)
    Ratio_grid = (Beta_grid * M) / Y0_grid

    vlim = 1.5
    cp = ax.contourf(
        X, M, Ratio_grid,
        levels=np.linspace(-vlim, vlim, 61),
        cmap=CMAP,
        extend="both"
    )

    levels_lines = np.array([-1.0, -0.6, -0.3, 0.0, 0.4, 0.8, 1.2])
    cs = ax.contour(X, M, Ratio_grid, levels=levels_lines, colors="#444444", linewidths=0.8, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8.0, fmt="%+.1f")

    # 1. 理論幾何解 M_rms(x) = sqrt(<M^2>(x))
    x_th = np.logspace(-2.5, 1.0, 400)
    m_rms_th = theoretical_m_rms_curve(x_th)
    ax.plot(
        x_th, m_rms_th,
        color="black", lw=3.0, ls="-", zorder=7,
        label=r"Geometric Theory: $M_{\rm rms}(x) = \sqrt{\langle M^2(x) \rangle}$"
    )

    # 2. クロスオーバー線 x = x_c (beta = 0)
    ax.axvline(
        x_crossover, color="darkgreen", lw=2.5, ls="--", zorder=7,
        label=rf"Crossover $x_c \approx {x_crossover:.2f}$ ($\Pi=0$)"
    )

    # 3. 相の領域注釈 (3領域)
    if not log_x:
        ax.text(
            0.38, 0.92,
            r"$\mathbf{Phase\ I:\ Coordinated\ Flow}$" + "\n" + r"($M > M_{\rm rms},\ \Pi < 0$)" + "\nVelocity Boost",
            ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#8888cc", alpha=0.9)
        )
        ax.text(
            0.20, 0.10,
            r"$\mathbf{Phase\ II:\ Fluctuation}$" + "\n" + r"($M < M_{\rm rms},\ \Pi < 0$)" + "\nWeak Acceleration",
            ha="center", va="center", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.85)
        )
        ax.text(
            1.05, 0.20,
            r"$\mathbf{Phase\ III:\ Jamming}$" + "\n" + r"($x > x_c,\ \Pi > 0$)" + "\nSevere Deceleration",
            ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cc8888", alpha=0.9)
        )
    else:
        ax.text(
            0.05, 0.88,
            r"$\mathbf{Phase\ I:\ Flow\ Boost}$" + "\n" + r"($M > M_{\rm rms},\ \Pi < 0$)",
            ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#8888cc", alpha=0.9)
        )
        ax.text(
            0.05, 0.12,
            r"$\mathbf{Phase\ II:\ Fluctuation}$" + "\n" + r"($M < M_{\rm rms},\ \Pi < 0$)",
            ha="center", va="center", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.85)
        )
        ax.text(
            1.05, 0.25,
            r"$\mathbf{Phase\ III:\ Jamming}$" + "\n" + r"($x > x_c,\ \Pi > 0$)",
            ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cc8888", alpha=0.9)
        )

    # 実験ビーズ位置のプロット (理論曲線 M_rms(x) 上にプロット, 横軸 STD エラーバー)
    for _, row in df_fit.iterrows():
        b_name = row["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        x_val = row["rc_over_xi_mean"]
        x_std = row["rc_over_xi_std"] if "rc_over_xi_std" in row else row["rc_over_xi_sem"]
        m_th_at_x = float(theoretical_m_rms_curve(np.array([x_val]))[0])

        ax.errorbar(
            x_val, m_th_at_x,
            xerr=x_std,
            fmt=b_info["marker"], color=b_info["color"], ecolor="black",
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
            markeredgecolor="black", markeredgewidth=1.2, zorder=10,
            label=f"$d = {b_info['label']}$ ($x={x_val:.2f}$)"
        )

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(0.015, 1.5)
        suffix = "logx"
    else:
        ax.set_xscale("linear")
        ax.set_xlim(0.0, 1.22)
        suffix = "linear"

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13.0, fontweight='bold')
    ax.set_ylabel(r"Magnetization $M$ (Orientation Order)", fontsize=13.0, fontweight='bold')

    scale_label = "Log Scale" if log_x else "Linear Scale"
    ax.set_title(
        rf"Universal Master Phase Diagram: $\Pi(x, M) = \frac{{\beta(x) M}}{{y_0(x)}}$ with Theory $\langle M^2(x) \rangle$ ({scale_label})" + "\n" +
        r"$\ln\tilde{v} = y_0(x)\left[1 + \Pi(x, M)\right]$, Blue: Flow Boost, Red: Jamming Drag",
        fontsize=12.5, fontweight='bold', pad=12
    )

    ax.grid(True, which='both' if log_x else 'major', ls=':', alpha=0.55, color='gray')
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, ncol=2)

    cbar = fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Velocity Modulation Ratio $\Pi(x, M) = \frac{\beta(x) M}{y_0(x)}$", fontsize=12.0)

    plt.tight_layout()
    save_figure_to_all(fig, f"universal_master_phase_diagram_m2_theory_{suffix}", out_dirs)


# =============================================================================
# 作図 4: 新規グラフ - 実効輸送速度 \ln\tilde{v}(x, \Pi) の等高線相図
# =============================================================================

def plot_effective_velocity_phase_diagram(
    df_fit: pd.DataFrame,
    interp_y0,
    x_crossover: float,
    out_dirs: List[Path],
    log_x: bool = False
):
    r"""
    新規グラフ: 実効輸送速度 \ln \tilde{v}(x, \Pi) の等高線相図
    数式:
        \ln \tilde{v}(x, \Pi) = y_0(x) [ 1 + \Pi ]
    横軸: スケール半径 x = R_c / \xi_{i,t} (エラーバー: STD)
    縦軸: 変調比 \Pi = \frac{\beta M}{y_0} (エラーバー: STD)
    背景カラー: 実効輸送速度 \ln \tilde{v} (-3.0 ~ 0.0)
    オーバーレイ: 各ビーズの実験データ点 (x_d, \Pi_d) とエラーバー
    """
    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    if log_x:
        x_grid = np.logspace(-2.0, 0.4, 250)
    else:
        x_grid = np.linspace(0.01, 1.20, 250)

    pi_grid = np.linspace(-1.2, 1.2, 200)
    X, PI = np.meshgrid(x_grid, pi_grid)

    Y0_grid = interp_y0(X)
    # 実効対数速度 ln(v_tilde) = y0(x) * (1 + Pi)
    LnV_grid = Y0_grid * (1.0 + PI)

    # カラーマップ: 速度グラデーション
    v_min, v_max = -3.5, 0.0
    cp = ax.contourf(
        X, PI, LnV_grid,
        levels=np.linspace(v_min, v_max, 71),
        cmap="viridis",
        extend="both"
    )

    # 等値線 (等速度線)
    levels_lines = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
    cs = ax.contour(X, PI, LnV_grid, levels=levels_lines, colors="white", linewidths=1.0, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=9.0, fmt=r"$\ln\tilde{v} = %.1f$", colors="white")

    # 1. 基準線: 変調なし線 Pi = 0 (基底速度 ln(v) = y0(x))
    ax.axhline(0, color="white", lw=1.8, ls=":", zorder=5, label=r"Baseline $\Pi = 0$ ($\ln\tilde{v} = y_0(x)$)")

    # 2. クロスオーバー線 x = x_c
    ax.axvline(x_crossover, color="gold", lw=2.2, ls="--", zorder=5, label=rf"Crossover $x_c \approx {x_crossover:.2f}$")

    # 3. 実験データ点のオーバーレイ (x 方向 STD, Pi 方向 STD)
    for _, row in df_fit.iterrows():
        b_name = row["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        x_val = row["rc_over_xi_mean"]
        x_std = row["rc_over_xi_std"] if "rc_over_xi_std" in row else row["rc_over_xi_sem"]

        beta = row["ln_beta_slope"]
        y0 = row["ln_y0_intercept"]
        m_mean = row["m_mean"] if "m_mean" in row and not np.isnan(row["m_mean"]) else 0.5
        m_std = row["m_std"] if "m_std" in row and not np.isnan(row["m_std"]) else 0.3

        # 変調比 Pi の代表値と誤差 (M の STD による伝播 STD)
        pi_val = (beta * m_mean) / y0
        pi_err = np.abs(beta / y0) * m_std

        ax.errorbar(
            x_val, pi_val,
            xerr=x_std, yerr=pi_err,
            fmt=b_info["marker"], color=b_info["color"], ecolor="white",
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
            markeredgecolor="white", markeredgewidth=1.3, zorder=10,
            label=f"$d = {b_info['label']}$"
        )

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(0.015, 1.5)
        suffix = "logx"
    else:
        ax.set_xscale("linear")
        ax.set_xlim(0.0, 1.22)
        suffix = "linear"

    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13.0, fontweight='bold')
    ax.set_ylabel(r"Modulation Ratio $\Pi = \frac{\beta M}{y_0}$", fontsize=13.0, fontweight='bold')

    scale_label = "Log Scale" if log_x else "Linear Scale"
    ax.set_title(
        rf"Effective Transport Velocity Phase Diagram: $\ln\tilde{{v}}(x, \Pi) = y_0(x)[1 + \Pi]$ ({scale_label})" + "\n" +
        r"Contour: Effective Velocity $\ln\tilde{v}$ (Yellow: High Velocity, Purple: Low Velocity)",
        fontsize=12.5, fontweight='bold', pad=12
    )

    ax.grid(True, which='both' if log_x else 'major', ls=':', alpha=0.45, color='gray')
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, ncol=2)

    cbar = fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Effective Log Velocity $\ln\tilde{v}(x, \Pi)$", fontsize=12.0)

    plt.tight_layout()
    save_figure_to_all(fig, f"effective_velocity_contour_phase_diagram_{suffix}", out_dirs)


# =============================================================================
# 作図 5: 4パネル統合エグゼクティブ図 (Summary 4-Panel Executive Phase Diagram)
# =============================================================================

def plot_4panel_executive_phase_diagram(
    df_fit: pd.DataFrame,
    interp_beta,
    interp_y0,
    x_crossover: float,
    out_dirs: List[Path]
):
    """
    (a) 離散マトリクス
    (b) 連続2Dカラーマップ (Linear x)
    (c) 理論幾何マスター相図 (Linear x, on Theory Curve)
    (d) 新規 実効輸送速度相図 ln(v_tilde)(x, Pi)
    の4パネル統合エグゼクティブ図 (全エラーバー STD)
    """
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 13.5))

    # (a) 離散マトリクス
    ax_a = axes[0, 0]
    m_bins = np.linspace(0.05, 0.95, 10)
    n_m = len(m_bins)
    n_beads = len(df_fit)
    ratio_matrix = np.zeros((n_m, n_beads))
    for col_idx, (_, row) in enumerate(df_fit.iterrows()):
        ratio_matrix[:, col_idx] = (row["ln_beta_slope"] * m_bins) / row["ln_y0_intercept"]

    vmax = 1.5
    im_a = ax_a.imshow(
        ratio_matrix, cmap=CMAP, aspect="auto", origin="lower",
        vmin=-vmax, vmax=vmax, extent=[-0.5, n_beads - 0.5, 0.0, 1.0]
    )
    for i in range(n_m):
        for j in range(n_beads):
            val = ratio_matrix[i, j]
            tc = "white" if np.abs(val) > vmax * 0.55 else "black"
            ax_a.text(j, m_bins[i], f"{val:+.2f}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=tc)
    ax_a.set_xticks(range(n_beads))
    ax_a.set_xticklabels([f"{r['label']}\n($x={r['rc_over_xi_mean']:.2f}$)" for _, r in df_fit.iterrows()], fontsize=9.5)
    ax_a.set_ylabel(r"Magnetization $M$", fontsize=11.5, fontweight='bold')
    ax_a.set_title(r"(a) Discrete Matrix Heatmap $\Pi = \frac{\beta M}{y_0}$", fontsize=12.5, fontweight='bold')
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.03, label=r"$\Pi = \frac{\beta M}{y_0}$")

    # (b) 連続2Dカラーマップ
    ax_b = axes[0, 1]
    X_lin, M_lin = np.meshgrid(np.linspace(0.01, 1.18, 200), np.linspace(0.0, 1.0, 180))
    Ratio_lin = (interp_beta(X_lin) * M_lin) / interp_y0(X_lin)
    cp_b = ax_b.contourf(X_lin, M_lin, Ratio_lin, levels=np.linspace(-vmax, vmax, 51), cmap=CMAP, extend="both")
    cs_b = ax_b.contour(X_lin, M_lin, Ratio_lin, levels=[-1.0, -0.5, 0.0, 0.5, 1.0], colors="black", linewidths=0.8, alpha=0.7)
    ax_b.clabel(cs_b, inline=True, fontsize=8.0, fmt="%+.1f")
    ax_b.axvline(x_crossover, color="darkgreen", ls="--", lw=2.0, label=rf"$x_c \approx {x_crossover:.2f}$")
    for _, row in df_fit.iterrows():
        b_info = next(b for b in BEADS_INFO if b["name"] == row["bead_name"])
        x_pt = row["rc_over_xi_mean"]
        x_std = row["rc_over_xi_std"] if "rc_over_xi_std" in row else row["rc_over_xi_sem"]
        m_mean = row["m_mean"] if "m_mean" in row and not np.isnan(row["m_mean"]) else 0.5
        m_std = row["m_std"] if "m_std" in row and not np.isnan(row["m_std"]) else 0.3
        ax_b.errorbar(x_pt, m_mean, xerr=x_std, yerr=m_std,
                      fmt=b_info["marker"], color=b_info["color"], ecolor="black", markersize=8.5, markeredgecolor="black", markeredgewidth=1.1)
    ax_b.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=11.5, fontweight='bold')
    ax_b.set_ylabel(r"Magnetization $M$", fontsize=11.5, fontweight='bold')
    ax_b.set_title(r"(b) Continuous 2D Phase Diagram $\Pi(x, M)$", fontsize=12.5, fontweight='bold')
    ax_b.legend(loc="upper left", fontsize=9.0)
    fig.colorbar(cp_b, ax=ax_b, fraction=0.046, pad=0.03, label=r"$\Pi(x, M)$")

    # (c) 理論幾何マスター相図 (Linear, on Theory Curve)
    ax_c = axes[1, 0]
    cp_c = ax_c.contourf(X_lin, M_lin, Ratio_lin, levels=np.linspace(-vmax, vmax, 51), cmap=CMAP, extend="both")
    x_th = np.linspace(0.005, 1.20, 250)
    ax_c.plot(x_th, theoretical_m_rms_curve(x_th), 'k-', lw=2.5, label=r"Theory $M_{\rm rms}(x) = \sqrt{\langle M^2 \rangle}$")
    ax_c.axvline(x_crossover, color="darkgreen", ls="--", lw=2.0, label=rf"Crossover $x_c \approx {x_crossover:.2f}$")
    for _, row in df_fit.iterrows():
        b_info = next(b for b in BEADS_INFO if b["name"] == row["bead_name"])
        x_pt = row["rc_over_xi_mean"]
        x_std = row["rc_over_xi_std"] if "rc_over_xi_std" in row else row["rc_over_xi_sem"]
        m_th = float(theoretical_m_rms_curve(np.array([x_pt]))[0])
        ax_c.errorbar(x_pt, m_th, xerr=x_std,
                      fmt=b_info["marker"], color=b_info["color"], ecolor="black", markersize=8.5, markeredgecolor="black", markeredgewidth=1.1,
                      label=f"$d = {b_info['label']}$")
    ax_c.set_xlim(0.0, 1.20)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=11.5, fontweight='bold')
    ax_c.set_ylabel(r"Magnetization $M$", fontsize=11.5, fontweight='bold')
    ax_c.set_title(r"(c) Theoretical Master Phase Diagram (Linear Scale)", fontsize=12.5, fontweight='bold')
    ax_c.legend(loc="upper right", fontsize=8.0, ncol=2)
    fig.colorbar(cp_c, ax=ax_c, fraction=0.046, pad=0.03, label=r"$\Pi = \frac{\beta M}{y_0}$")

    # (d) 新規 実効輸送速度相図 ln(v_tilde)(x, Pi)
    ax_d = axes[1, 1]
    pi_grid_d = np.linspace(-1.2, 1.2, 180)
    X_d, PI_d = np.meshgrid(np.linspace(0.01, 1.18, 200), pi_grid_d)
    LnV_d = interp_y0(X_d) * (1.0 + PI_d)
    cp_d = ax_d.contourf(X_d, PI_d, LnV_d, levels=np.linspace(-3.5, 0.0, 51), cmap="viridis", extend="both")
    cs_d = ax_d.contour(X_d, PI_d, LnV_d, levels=[-3.0, -2.5, -2.0, -1.5, -1.0, -0.5], colors="white", linewidths=0.8, alpha=0.8)
    ax_d.clabel(cs_d, inline=True, fontsize=7.5, fmt="%.1f", colors="white")
    ax_d.axhline(0, color="white", ls=":", lw=1.5, label=r"Baseline $\Pi=0$")
    ax_d.axvline(x_crossover, color="gold", ls="--", lw=1.8, label=rf"$x_c \approx {x_crossover:.2f}$")
    for _, row in df_fit.iterrows():
        b_info = next(b for b in BEADS_INFO if b["name"] == row["bead_name"])
        x_pt = row["rc_over_xi_mean"]
        x_std = row["rc_over_xi_std"] if "rc_over_xi_std" in row else row["rc_over_xi_sem"]
        beta = row["ln_beta_slope"]
        y0 = row["ln_y0_intercept"]
        m_mean = row["m_mean"] if "m_mean" in row and not np.isnan(row["m_mean"]) else 0.5
        m_std = row["m_std"] if "m_std" in row and not np.isnan(row["m_std"]) else 0.3
        pi_val = (beta * m_mean) / y0
        pi_err = np.abs(beta / y0) * m_std
        ax_d.errorbar(x_pt, pi_val, xerr=x_std, yerr=pi_err,
                      fmt=b_info["marker"], color=b_info["color"], ecolor="white", markersize=8.5, markeredgecolor="white", markeredgewidth=1.1)
    ax_d.set_xlim(0.0, 1.20)
    ax_d.set_ylim(-1.2, 1.2)
    ax_d.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=11.5, fontweight='bold')
    ax_d.set_ylabel(r"Modulation Ratio $\Pi = \frac{\beta M}{y_0}$", fontsize=11.5, fontweight='bold')
    ax_d.set_title(r"(d) Effective Transport Velocity $\ln\tilde{v}(x, \Pi) = y_0(x)[1 + \Pi]$", fontsize=12.5, fontweight='bold')
    ax_d.legend(loc="upper right", fontsize=8.0)
    fig.colorbar(cp_d, ax=ax_d, fraction=0.046, pad=0.03, label=r"$\ln\tilde{v}$")

    plt.suptitle(
        r"Executive Summary: Scaling Phase Diagrams of Velocity Modulation $\Pi(x, M) = \frac{\beta(x) M}{y_0(x)}$",
        fontsize=14.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure_to_all(fig, "heatmap_relative_modulation_4panel_executive", out_dirs)


# =============================================================================
# メイン実行関数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot relative modulation ratio heatmap and theoretical master phase diagrams.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Root directory of dataset")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "figure" / "conditional_log_velocity", help="Output directory")
    args = parser.parse_args()

    root_dir = args.root_dir or get_default_root_dir()

    out_dirs = [args.output_dir]
    nas_out_dir = root_dir / "figure" / "conditional_log_velocity"
    if nas_out_dir.exists() or nas_out_dir.parent.exists():
        out_dirs.append(nas_out_dir)

    print(f"Output directories: {[str(d) for d in out_dirs]}")

    df_fit = load_fit_summary_data()
    print("\n--- Loaded Linear Fit Parameters vs Scaled Radius ---")
    cols_to_print = ["bead_name", "diameter_um", "rc_over_xi_mean", "ln_y0_intercept", "ln_beta_slope"]
    if "m_mean" in df_fit.columns:
        cols_to_print.extend(["m_mean", "m_std"])
    print(df_fit[cols_to_print].to_string(index=False))

    interp_beta, interp_y0, x_crossover = build_interpolators(df_fit)
    print(f"\nEstimated Crossover Point (beta = 0): x_c = {x_crossover:.3f}")

    plot_discrete_matrix_heatmap(df_fit, out_dirs)
    plot_continuous_contour_heatmap(df_fit, interp_beta, interp_y0, x_crossover, out_dirs)
    plot_universal_master_phase_diagram(df_fit, interp_beta, interp_y0, x_crossover, out_dirs, log_x=False)
    plot_universal_master_phase_diagram(df_fit, interp_beta, interp_y0, x_crossover, out_dirs, log_x=True)
    plot_effective_velocity_phase_diagram(df_fit, interp_y0, x_crossover, out_dirs, log_x=False)
    plot_effective_velocity_phase_diagram(df_fit, interp_y0, x_crossover, out_dirs, log_x=True)
    plot_4panel_executive_phase_diagram(df_fit, interp_beta, interp_y0, x_crossover, out_dirs)

    print("\nAll relative modulation heatmaps, master phase diagrams, and effective velocity diagrams generated successfully!")


if __name__ == "__main__":
    main()

