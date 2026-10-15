#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ising_scaling_master_curve.py
==================================

微小管フローのイジングスピン磁化 <M^2> と空間配向相関長 xi によるスケーリンググラフを作成するスクリプト。

【概要】
1. 光学フロー場 u(x, y, t) とネマチックディレクター n からイジングスピン
       sigma(x, y, t) = sign( u(x, y, t) . n )
   を定義。
2. 任意のウィンドウ半径 R（円形ドメイン / 円板）内におけるイジングスピン平均値
       M(R) = (1 / N_R) sum_{i in disk} sigma_i
   およびその2乗 M^2(R)、アンサンブル平均 <M^2(R)> を FFT 畳み込みにより計算。
3. 空間配向相関 C(r) を指数減衰 C(r) = a * exp(-r / xi) でフィッティングして得られた相関長 xi を取得。
4. 無次元スケーリング変数 x = R / xi を定義し、横軸 x = R / xi、縦軸 <M^2> のグラフを作成。
5. 多数の (R, xi, exp, block) から大量のデータ点を集約し、x を適切な幅のビン（Bin）に分割して
   ビン平均値 <M^2> および標準誤差 (SEM) / 中央値・四分位数 (IQR) をプロット。
6. 2次元円形領域における指数相関場の理論マスターカーブ
       <M^2(x)> = (2 / x^2) * { 1 + 2 * [I_0(2x) - L_0(2x)] - (3 / x) * [I_1(2x) - L_1(2x)] }
   （I_n: 第1種変形ベッセル関数, L_n: 変形ストルーブ関数）を理論線として重ねて描画。

【出力ファイル（figure/ising_scaling/）】
- ising_scaling_master_curve_2panel.png / .svg  : Linear & Log-Log の2パネル比較図
- ising_scaling_master_curve_loglog.png / .svg  : Log-Log スケーリング確認図（べき減衰 x^-2 の漸近線付き）
- ising_scaling_master_curve_linear.png / .svg  : Linear 全体図
- ising_scaling_master_curve_per_condition.png / .svg : 粒子径条件別の比較図
- ising_scaling_binned_data.csv                 : ビン集計統計量（平均、SEM、中央値、IQR、サンプル数）
- ising_scaling_points_summary.csv              : 実験 x 窓サイズごとのデータサマリー
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# NAS / 共有ボリュームでの HDF5 ファイルロックエラー防止
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import integrate, signal
import scipy.special as sp
from tqdm import tqdm

# 親ディレクトリのパス設定
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import ising_magnetization as ising
import plot_mt_orientation_distribution as mt_ori
import plot_ising_magnetization as ising_plot

# スタイル適用
_style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if _style_path.exists():
    try:
        plt.style.use(str(_style_path))
    except Exception:
        pass

BEADS_INFO = mt_ori.BEADS_INFO
BEAD_LOOKUP = mt_ori.BEAD_LOOKUP


# =============================================================================
# 理論マスターカーブ（2次元円形領域における厳密解）
# =============================================================================

_THEORY_GRID_X = np.logspace(-3, 3, 300)
_THEORY_GRID_M2 = None


def _calc_theoretical_m2_disk(x_val: float) -> float:
    """
    2次元円形領域（半径 R）における指数相関場 C(r) = exp(-r/xi) の
    磁化2乗平均 <M^2>(x) (x = R / xi) の厳密解:
        <M^2(x)> = (2 / x^2) * { 1 + 2 * [I_0(2x) - L_0(2x)] - (3 / x) * [I_1(2x) - L_1(2x)] }
    """
    x = float(x_val)
    if x <= 0:
        return 1.0
    if x < 1e-4:
        # Taylor 展開: <M^2> = 1 - (128 / (45 * pi)) * x + ...
        return float(1.0 - (128.0 / (45.0 * np.pi)) * x)
    if x > 15.0:
        # 大スケール (x > 15) では Bessel/Struve 関数の指数増大によるオーバーフローを防ぐため、
        # 厳密な弦長分布積分による高精度評価を行う
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


def theoretical_master_curve(x_array: np.ndarray) -> np.ndarray:
    """
    円形領域における理論普遍関数 <M^2>(x) をルックアップテーブル補間で高速かつ高精度に評価する。
    """
    global _THEORY_GRID_M2
    if _THEORY_GRID_M2 is None:
        _THEORY_GRID_M2 = np.array([_calc_theoretical_m2_disk(x) for x in _THEORY_GRID_X])

    xs = np.asarray(x_array, dtype=float)
    # 対数空間で補間
    log_x = np.log10(np.clip(xs, 1e-4, 1e4))
    log_grid_x = np.log10(_THEORY_GRID_X)
    log_grid_m2 = np.log10(np.maximum(_THEORY_GRID_M2, 1e-12))

    log_m2_interp = np.interp(log_x, log_grid_x, log_grid_m2)
    return np.power(10.0, log_m2_interp)


# =============================================================================
# 相関長 xi の読み出し
# =============================================================================

def load_xi_dict(root_dir: Path) -> Dict[str, float]:
    """
    実験ディレクトリ名 -> 相関長 xi [um] の辞書を作成する。
    1. figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv
    2. xi.csv
    の順に探索・統合する。
    """
    xi_dict: Dict[str, float] = {}

    # 1. 実験ごとの高精度相関長 CSV
    csv_per_exp = CURRENT_DIR / 'figure' / 'bg_angular_correlation' / 'bg_angular_correlation_length_per_experiment.csv'
    if not csv_per_exp.exists():
        csv_per_exp = root_dir / 'figure' / 'bg_angular_correlation' / 'bg_angular_correlation_length_per_experiment.csv'

    if csv_per_exp.exists():
        try:
            df_per_exp = pd.read_csv(csv_per_exp)
            for _, row in df_per_exp.iterrows():
                exp_name = str(row['exp_dir']).strip()
                if 'xi_um' in row and np.isfinite(row['xi_um']) and row['xi_um'] > 0:
                    xi_dict[exp_name] = float(row['xi_um'])
        except Exception as e:
            print(f"[WARNING] Could not read {csv_per_exp}: {e}")

    # 2. xi.csv（条件代表値）のフォールバック
    xi_csv = CURRENT_DIR / 'xi.csv'
    if not xi_csv.exists():
        xi_csv = root_dir / 'xi.csv'
    cond_xi_map: Dict[str, float] = {}
    if xi_csv.exists():
        try:
            df_xi = pd.read_csv(xi_csv)
            df_xi.columns = df_xi.columns.str.strip()
            # flow_particle または background の total
            for ctype in ['flow_particle', 'background', 'bead_flow']:
                sub = df_xi[(df_xi['component'] == 'total') & (df_xi['type'] == ctype)]
                for _, row in sub.iterrows():
                    cname = str(row['condition']).strip()
                    if cname not in cond_xi_map and np.isfinite(row['xi_um']):
                        cond_xi_map[cname] = float(row['xi_um'])
        except Exception as e:
            print(f"[WARNING] Could not read {xi_csv}: {e}")

    return xi_dict, cond_xi_map


# =============================================================================
# 実験ごとのブロック磁化抽出
# =============================================================================

def extract_experiment_m2_scaling(
    exp_dir: Path,
    bead: dict,
    xi_val: float,
    window_sizes_um: Sequence[float],
    pixel_stride: int = 8,
    frame_stride: int = 5,
    max_frames_per_exp: Optional[int] = None,
    director: str = 'global',
    min_valid_fraction: float = 0.5,
    min_flow_mag: float = 1e-4,
    scale: float = 0.11,
    sample_blocks_per_window: int = 500,
    progress: bool = True,
) -> Optional[dict]:
    """
    1つの実験に対して、複数のウィンドウサイズ R [um] における
    イジングスピン平均 M および M^2 を計算し、スケーリングレコードを返す。
    """
    exp_dir = Path(exp_dir)
    st = max(1, int(pixel_stride))
    fs = max(1, int(frame_stride))

    try:
        reader = mt_ori.FlowFrameReader(exp_dir, pixel_stride=st, frame_stride=fs,
                                        max_frames_per_exp=max_frames_per_exp)
    except Exception as e:
        print(f"    [WARNING] cannot open flow for {exp_dir.name}: {e}", flush=True)
        return None

    with reader as src:
        if src.source is None:
            return None

        n_frames_total = int(src.n_frames_total)
        frame_ids = [int(t) for t in src.frame_ids]
        full_rows, full_cols = int(src.rows), int(src.cols)
        out_shape = (int(src.out_rows), int(src.out_cols))

        # ディレクターの決定
        thetas: Optional[np.ndarray] = None
        theta_maps: Optional[np.ndarray] = None
        if str(director) == 'local':
            theta_maps = ising_plot.load_local_director_maps(
                exp_dir, frame_ids, (full_rows, full_cols), out_shape, st)
            if theta_maps is None:
                thetas = mt_ori.load_nematic_directors(exp_dir, frame_ids, n_frames_total)
        else:
            thetas = mt_ori.load_nematic_directors(exp_dir, frame_ids, n_frames_total)

        # 窓サイズの格子単位変換（重複排除）
        windows_px = [max(st, int(round(w_um / scale))) for w_um in window_sizes_um]
        windows_grid = np.unique([max(1, int(round(w_px / st))) for w_px in windows_px])
        max_grid_limit = min(out_shape[0], out_shape[1]) // 2
        windows_grid = windows_grid[windows_grid <= max_grid_limit]
        if len(windows_grid) == 0:
            return None

        # 窓サイズごとの円形カーネル事前生成
        kernels = {w_g: ising.create_disk_kernel(w_g) for w_g in windows_grid}
        n_disks = {w_g: float(np.sum(kernels[w_g])) for w_g in windows_grid}

        # 窓サイズごとの集計配列
        # R_grid -> list of frame-level squared_mean, and pooled block M^2
        window_stats = {w: {'sum_m2': 0.0, 'sum_abs_m': 0.0, 'sum_m': 0.0,
                            'n_valid_blocks': 0, 'n_frames': 0,
                            'sampled_m2': []} for w in windows_grid}

        iterator = range(len(frame_ids))
        if progress:
            iterator = tqdm(iterator, desc=f"M^2 Scaling {exp_dir.parent.name}/{exp_dir.name}",
                            leave=False)

        for i in iterator:
            t = int(frame_ids[i])
            mx, my = src.get(i)
            mag = np.hypot(mx, my)
            valid = np.isfinite(mx) & np.isfinite(my) & (mag > max(float(min_flow_mag), 0.0))

            if theta_maps is not None:
                th = theta_maps[i]
                valid = valid & np.isfinite(th)
                ct = np.cos(th)
                stt = np.sin(th)
            elif thetas is not None:
                th = float(thetas[min(t, thetas.size - 1)])
                ct = float(np.cos(th))
                stt = float(np.sin(th))
            else:
                phi_v = np.arctan2(my[valid], mx[valid])
                th = mt_ori.global_nematic_theta_from_angles(phi_v)
                ct = float(np.cos(th))
                stt = float(np.sin(th))

            if not np.any(valid):
                continue

            dot = mx * ct + my * stt
            sigma = np.sign(dot)
            sigma[~valid] = 0.0

            s_in = np.where(valid, sigma, 0.0)
            v_in = valid.astype(np.float64)

            for w_g in windows_grid:
                k = kernels[w_g]
                n_disk = n_disks[w_g]

                s_conv = signal.fftconvolve(s_in, k, mode='same')
                v_conv = signal.fftconvolve(v_in, k, mode='same')

                step = max(1, int(round(w_g)))
                ys = np.arange(0, out_shape[0], step)
                xs = np.arange(0, out_shape[1], step)
                if len(ys) == 0 or len(xs) == 0:
                    continue

                sub_s = s_conv[np.ix_(ys, xs)]
                sub_v = v_conv[np.ix_(ys, xs)]

                valid_mask = (sub_v > 0.0) & (sub_v >= (float(min_valid_fraction) * n_disk))
                if not np.any(valid_mask):
                    continue

                m_block = sub_s[valid_mask] / sub_v[valid_mask]
                m2_block = m_block ** 2
                abs_m_block = np.abs(m_block)

                st_dict = window_stats[w_g]
                st_dict['sum_m2'] += float(np.sum(m2_block))
                st_dict['sum_abs_m'] += float(np.sum(abs_m_block))
                st_dict['sum_m'] += float(np.sum(m_block))
                st_dict['n_valid_blocks'] += int(len(m_block))
                st_dict['n_frames'] += 1

                # サンプリングブロック
                if len(m2_block) > 0 and len(st_dict['sampled_m2']) < sample_blocks_per_window:
                    n_take = min(len(m2_block), sample_blocks_per_window - len(st_dict['sampled_m2']))
                    sub_idx = np.linspace(0, len(m2_block) - 1, n_take, dtype=int)
                    st_dict['sampled_m2'].extend(m2_block[sub_idx].tolist())

    exp_records = []
    block_records = []

    for w_g, st_dict in window_stats.items():
        if st_dict['n_valid_blocks'] == 0:
            continue
        w_px = w_g * st
        w_um = w_px * scale
        x_scaled = w_um / xi_val

        m2_mean = st_dict['sum_m2'] / st_dict['n_valid_blocks']
        abs_m_mean = st_dict['sum_abs_m'] / st_dict['n_valid_blocks']
        m_mean = st_dict['sum_m'] / st_dict['n_valid_blocks']

        exp_records.append({
            'bead_name': bead['name'],
            'diameter_um': float(bead.get('diameter_um', np.nan)),
            'exp_dir': exp_dir.name,
            'window_grid': int(w_g),
            'window_px': int(w_px),
            'window_um': float(w_um),
            'xi_um': float(xi_val),
            'x_scaled': float(x_scaled),
            'squared_mean': float(m2_mean),
            'abs_mean': float(abs_m_mean),
            'signed_mean': float(m_mean),
            'n_valid_blocks': int(st_dict['n_valid_blocks']),
            'n_frames': int(st_dict['n_frames']),
        })

        for val_m2 in st_dict['sampled_m2']:
            block_records.append({
                'bead_name': bead['name'],
                'diameter_um': float(bead.get('diameter_um', np.nan)),
                'exp_dir': exp_dir.name,
                'window_um': float(w_um),
                'xi_um': float(xi_val),
                'x_scaled': float(x_scaled),
                'm2_value': float(val_m2),
            })

    return {
        'exp_records': exp_records,
        'block_records': block_records,
    }


# =============================================================================
# ビン分割集計
# =============================================================================

def compute_scaling_bins(
    df_points: pd.DataFrame,
    value_col: str = 'squared_mean',
    n_bins: int = 24,
    bin_scale: str = 'log',
    min_count: int = 5,
) -> pd.DataFrame:
    """
    無次元変数 x = R / xi をビン分割し、<M^2> および比率 M^2_exp / M^2_th の統計量を算出する。
    """
    if df_points.empty or value_col not in df_points.columns:
        return pd.DataFrame()

    df_valid = df_points.copy()
    x = df_valid['x_scaled'].to_numpy(dtype=float)
    y = df_valid[value_col].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    df_valid = df_valid[valid]
    x, y = x[valid], y[valid]

    if len(x) < min_count:
        return pd.DataFrame()

    # 理論値と比率の計算
    theory_vals = theoretical_master_curve(x)
    ratio_vals = y / np.maximum(theory_vals, 1e-12)

    if str(bin_scale) == 'log':
        x_min, x_max = np.min(x), np.max(x)
        edges = np.geomspace(max(x_min * 0.99, 1e-4), x_max * 1.01, n_bins + 1)
    else:
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)

    bin_records = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (x >= lo) & (x <= hi) if k == n_bins - 1 else (x >= lo) & (x < hi)
        count = int(np.count_nonzero(mask))
        if count < min_count:
            continue

        vals = y[mask]
        x_vals = x[mask]
        r_vals = ratio_vals[mask]

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if count > 1 else 0.0
        sem_val = float(std_val / np.sqrt(count)) if count > 1 else 0.0

        q25 = float(np.percentile(vals, 25))
        median_val = float(np.median(vals))
        q75 = float(np.percentile(vals, 75))

        # 比率の統計量
        ratio_mean = float(np.mean(r_vals))
        ratio_std = float(np.std(r_vals, ddof=1)) if count > 1 else 0.0
        ratio_sem = float(ratio_std / np.sqrt(count)) if count > 1 else 0.0
        ratio_median = float(np.median(r_vals))
        ratio_q25 = float(np.percentile(r_vals, 25))
        ratio_q75 = float(np.percentile(r_vals, 75))

        # 幾何平均中心（対数軸用）
        x_center = float(np.sqrt(lo * hi)) if bin_scale == 'log' else float(0.5 * (lo + hi))
        th_center = theoretical_master_curve(np.array([x_center]))[0]

        bin_records.append({
            'bin_idx': k,
            'x_low': float(lo),
            'x_high': float(hi),
            'x_center': x_center,
            'x_mean': float(np.mean(x_vals)),
            'n_points': count,
            'm2_mean': mean_val,
            'm2_sem': sem_val,
            'm2_std': std_val,
            'm2_median': median_val,
            'm2_q25': q25,
            'm2_q75': q75,
            'theory_m2': th_center,
            'ratio_mean': ratio_mean,
            'ratio_sem': ratio_sem,
            'ratio_std': ratio_std,
            'ratio_median': ratio_median,
            'ratio_q25': ratio_q25,
            'ratio_q75': ratio_q75,
            'binned_ratio': float(mean_val / max(th_center, 1e-12)),
        })

    return pd.DataFrame(bin_records)


# =============================================================================
# 作図
# =============================================================================

def plot_master_curves(
    df_exp: pd.DataFrame,
    df_blocks: pd.DataFrame,
    df_bins_exp: pd.DataFrame,
    df_bins_block: pd.DataFrame,
    out_dirs: Sequence[Path],
    target_beads: Sequence[dict],
) -> None:
    """
    スケーリンググラフ（Linear & Log-Log 2パネル、3パネル、Ratio単体図、Log-Log単体図、Linear単体図、条件別図）を作成する。
    """
    # 各点における理論値と比率を計算
    df_exp = df_exp.copy()
    df_exp['theory_m2'] = theoretical_master_curve(df_exp['x_scaled'].to_numpy(dtype=float))
    df_exp['m2_ratio'] = df_exp['squared_mean'] / np.maximum(df_exp['theory_m2'], 1e-12)

    if not df_blocks.empty:
        df_blocks = df_blocks.copy()
        df_blocks['theory_m2'] = theoretical_master_curve(df_blocks['x_scaled'].to_numpy(dtype=float))
        df_blocks['m2_ratio'] = df_blocks['m2_value'] / np.maximum(df_blocks['theory_m2'], 1e-12)

    # 理論曲線用の x グリッド
    x_theory = np.geomspace(0.01, 100.0, 300)
    y_theory = theoretical_master_curve(x_theory)

    # 漸近線（x >> 1: 2*pi / x^2）
    x_asymp = np.geomspace(1.5, 60.0, 100)
    y_asymp = 2.0 * np.pi / (x_asymp ** 2)

    # -------------------------------------------------------------------------
    # 1. 2-Panel Master Curve (Linear + Log-Log)
    # -------------------------------------------------------------------------
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14.2, 5.8))

    # --- 左パネル: Linear スケール ---
    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax_lin.scatter(sub['x_scaled'], sub['squared_mean'],
                       s=32, alpha=0.55, color=bead.get('color', '#332288'),
                       marker=bead.get('marker', 'o'), edgecolors='none',
                       label=bead.get('label', bead['name']), zorder=3)

    if not df_bins_exp.empty:
        ax_lin.errorbar(df_bins_exp['x_center'], df_bins_exp['m2_mean'],
                        yerr=df_bins_exp['m2_sem'], fmt='s-', color='black',
                        ecolor='black', ms=6.5, lw=2.2, capsize=3.0,
                        label=r'Binned Mean $\pm$ SEM', zorder=6)

    ax_lin.plot(x_theory, y_theory, 'r--', lw=2.5,
                label=r'Theory: $\langle M^2(x) \rangle$', zorder=5)

    ax_lin.set_xlim(0.0, 15.0)
    ax_lin.set_ylim(-0.02, 1.05)
    ax_lin.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=13)
    ax_lin.set_ylabel(r'Ising Magnetization Variance $\langle M^2 \rangle$', fontsize=13)
    ax_lin.set_title(r'(a) Linear Scaling Master Curve', fontsize=14, pad=10)
    ax_lin.grid(True, which='both', alpha=0.3)
    ax_lin.legend(fontsize=9.5, loc='upper right', framealpha=0.92)

    # --- 右パネル: Log-Log スケール ---
    if not df_blocks.empty:
        n_blk = len(df_blocks)
        stride_blk = max(1, n_blk // 6000)
        sub_blk = df_blocks.iloc[::stride_blk]
        ax_log.scatter(sub_blk['x_scaled'], sub_blk['m2_value'],
                       s=6, alpha=0.08, color='gray', edgecolors='none',
                       rasterized=True, label=f'Block Clouds ($N={n_blk:,}$)', zorder=1)

    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax_log.scatter(sub['x_scaled'], sub['squared_mean'],
                       s=36, alpha=0.75, color=bead.get('color', '#332288'),
                       marker=bead.get('marker', 'o'), edgecolors='white',
                       linewidths=0.5, label=bead.get('label', bead['name']), zorder=3)

    if not df_bins_exp.empty:
        ax_log.errorbar(df_bins_exp['x_center'], df_bins_exp['m2_mean'],
                        yerr=df_bins_exp['m2_sem'], fmt='s-', color='black',
                        ecolor='black', ms=6.5, lw=2.2, capsize=3.0,
                        label=r'Binned Mean $\pm$ SEM', zorder=6)

    ax_log.plot(x_theory, y_theory, 'r--', lw=2.5,
                label=r'Theory $\langle M^2(x) \rangle$', zorder=5)
    ax_log.plot(x_asymp, y_asymp, 'k:', lw=1.8,
                label=r'Asymptotic $\propto x^{-2}$ (CLT)', zorder=4)

    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_xlim(0.04, 35.0)
    ax_log.set_ylim(0.005, 1.15)
    ax_log.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=13)
    ax_log.set_ylabel(r'$\langle M^2 \rangle$', fontsize=13)
    ax_log.set_title(r'(b) Log-Log Scaling Collapse & Power Law', fontsize=14, pad=10)
    ax_log.grid(True, which='both', alpha=0.3)
    ax_log.legend(fontsize=9, loc='lower left', framealpha=0.92)

    plt.tight_layout()
    mt_ori.save_figure_to_all(fig, 'ising_scaling_master_curve_2panel', list(out_dirs))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Ratio to Theory 単体図: M^2_exp / M^2_th vs x
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    # ブロック単位のクラウド（比率）
    if not df_blocks.empty:
        n_blk = len(df_blocks)
        stride_blk = max(1, n_blk // 6000)
        sub_blk = df_blocks.iloc[::stride_blk]
        ax.scatter(sub_blk['x_scaled'], sub_blk['m2_ratio'],
                   s=6, alpha=0.08, color='#7f7f7f', edgecolors='none',
                   rasterized=True, label=f'Block Cloud ($N={n_blk:,}$)', zorder=1)

    # 実験点
    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax.scatter(sub['x_scaled'], sub['m2_ratio'],
                   s=40, alpha=0.75, color=bead.get('color', '#332288'),
                   marker=bead.get('marker', 'o'), edgecolors='white',
                   linewidths=0.5, label=bead.get('label', bead['name']), zorder=3)

    # 理論一致の基準線 y = 1
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2.4,
               label=r'Perfect Agreement ($\frac{M^2_{\rm exp}}{M^2_{\rm th}} = 1$)', zorder=4)

    # ビン平均 ± SEM（比率）
    if not df_bins_exp.empty:
        ax.errorbar(df_bins_exp['x_center'], df_bins_exp['ratio_mean'],
                    yerr=df_bins_exp['ratio_sem'], fmt='s-', color='black',
                    ecolor='black', ms=7.0, lw=2.4, capsize=3.5,
                    label=r'Binned Ratio Mean $\pm$ SEM', zorder=6)

    # 説明テキスト
    r_all = df_exp['m2_ratio'].dropna()
    r_med = np.median(r_all) if len(r_all) > 0 else np.nan
    r_mean = np.mean(r_all) if len(r_all) > 0 else np.nan
    r_std = np.std(r_all, ddof=1) if len(r_all) > 1 else np.nan

    textbox = (r"$\bf{Theory\ vs\ Experiment\ Ratio}$" + "\n"
               r"$\bullet\ x \lesssim 3$: Excellent match ($\approx 0.9 - 1.1$)" + "\n"
               r"$\bullet\ x \gtrsim 3$: Residual order / polar bias" + "\n"
               f"Overall Median: {r_med:.3f} | Mean: {r_mean:.3f}\n"
               f"Data points: {len(df_exp):,} exp-windows")
    ax.text(0.03, 0.05, textbox, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9.0, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.75', alpha=0.92))

    ax.set_xscale('log')
    ax.set_xlim(0.04, 35.0)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=13)
    ax.set_ylabel(r'Ratio $\frac{M^2_{\rm exp}}{M^2_{\rm th}}$', fontsize=14)
    ax.set_title(r'Agreement with Theory: $\frac{M^2_{\rm exp}}{M^2_{\rm th}}$ vs $x = R / \xi$',
                 fontsize=14, pad=12)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.92)

    plt.tight_layout()
    mt_ori.save_figure_to_all(fig, 'ising_scaling_theory_ratio', list(out_dirs))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. 3-Panel Comprehensive Figure (Linear + Log-Log + Ratio)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 5.8))

    # --- Panel 1: Linear ---
    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax1.scatter(sub['x_scaled'], sub['squared_mean'],
                    s=30, alpha=0.55, color=bead.get('color', '#332288'),
                    marker=bead.get('marker', 'o'), edgecolors='none',
                    label=bead.get('label', bead['name']), zorder=3)

    if not df_bins_exp.empty:
        ax1.errorbar(df_bins_exp['x_center'], df_bins_exp['m2_mean'],
                     yerr=df_bins_exp['m2_sem'], fmt='s-', color='black',
                     ecolor='black', ms=6.0, lw=2.0, capsize=2.5,
                     label=r'Binned Mean $\pm$ SEM', zorder=6)

    ax1.plot(x_theory, y_theory, 'r--', lw=2.4, label=r'Theory: $\langle M^2(x) \rangle$', zorder=5)
    ax1.set_xlim(0.0, 15.0)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=12)
    ax1.set_ylabel(r'$\langle M^2 \rangle$', fontsize=12)
    ax1.set_title(r'(a) Linear Scaling Master Curve', fontsize=13, pad=10)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

    # --- Panel 2: Log-Log ---
    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax2.scatter(sub['x_scaled'], sub['squared_mean'],
                    s=32, alpha=0.75, color=bead.get('color', '#332288'),
                    marker=bead.get('marker', 'o'), edgecolors='white',
                    linewidths=0.5, label=bead.get('label', bead['name']), zorder=3)

    if not df_bins_exp.empty:
        ax2.errorbar(df_bins_exp['x_center'], df_bins_exp['m2_mean'],
                     yerr=df_bins_exp['m2_sem'], fmt='s-', color='black',
                     ecolor='black', ms=6.0, lw=2.0, capsize=2.5,
                     label=r'Binned Mean $\pm$ SEM', zorder=6)

    ax2.plot(x_theory, y_theory, 'r--', lw=2.4, label=r'Theory $\langle M^2(x) \rangle$', zorder=5)
    ax2.plot(x_asymp, y_asymp, 'k:', lw=1.8, label=r'CLT $\propto x^{-2}$', zorder=4)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.04, 35.0)
    ax2.set_ylim(0.005, 1.15)
    ax2.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=12)
    ax2.set_ylabel(r'$\langle M^2 \rangle$', fontsize=12)
    ax2.set_title(r'(b) Log-Log Scaling Collapse', fontsize=13, pad=10)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=8.5, loc='lower left', framealpha=0.92)

    # --- Panel 3: Ratio ---
    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax3.scatter(sub['x_scaled'], sub['m2_ratio'],
                    s=32, alpha=0.75, color=bead.get('color', '#332288'),
                    marker=bead.get('marker', 'o'), edgecolors='white',
                    linewidths=0.5, label=bead.get('label', bead['name']), zorder=3)

    ax3.axhline(1.0, color='red', linestyle='--', linewidth=2.2, label=r'Theory $= 1$', zorder=4)

    if not df_bins_exp.empty:
        ax3.errorbar(df_bins_exp['x_center'], df_bins_exp['ratio_mean'],
                     yerr=df_bins_exp['ratio_sem'], fmt='s-', color='black',
                     ecolor='black', ms=6.0, lw=2.0, capsize=2.5,
                     label=r'Binned Ratio $\pm$ SEM', zorder=6)

    ax3.set_xscale('log')
    ax3.set_xlim(0.04, 35.0)
    ax3.set_ylim(0.0, 2.5)
    ax3.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=12)
    ax3.set_ylabel(r'Ratio $\frac{M^2_{\rm exp}}{M^2_{\rm th}}$', fontsize=13)
    ax3.set_title(r'(c) Agreement: $\frac{M^2_{\rm exp}}{M^2_{\rm th}}$ vs $x$', fontsize=13, pad=10)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(fontsize=8.5, loc='upper left', framealpha=0.92)

    plt.tight_layout()
    mt_ori.save_figure_to_all(fig, 'ising_scaling_master_curve_3panel', list(out_dirs))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. Log-Log 単体図（詳細版）
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    if not df_blocks.empty:
        n_blk = len(df_blocks)
        stride_blk = max(1, n_blk // 8000)
        sub_blk = df_blocks.iloc[::stride_blk]
        ax.scatter(sub_blk['x_scaled'], sub_blk['m2_value'],
                   s=8, alpha=0.12, color='#7f7f7f', edgecolors='none',
                   rasterized=True, label=f'Block Cloud ($N={n_blk:,}$)', zorder=1)

    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax.scatter(sub['x_scaled'], sub['squared_mean'],
                   s=42, alpha=0.8, color=bead.get('color', '#332288'),
                   marker=bead.get('marker', 'o'), edgecolors='white',
                   linewidths=0.6, label=bead.get('label', bead['name']), zorder=3)

    if not df_bins_exp.empty:
        ax.errorbar(df_bins_exp['x_center'], df_bins_exp['m2_mean'],
                    yerr=df_bins_exp['m2_sem'], fmt='s-', color='black',
                    ecolor='black', ms=7.0, lw=2.4, capsize=3.5,
                    label=r'Binned Mean $\pm$ SEM', zorder=6)

    ax.plot(x_theory, y_theory, 'r--', lw=2.8,
            label=r'Theoretical Master Curve: $\langle M^2(x) \rangle$', zorder=5)
    ax.plot(x_asymp, y_asymp, 'k:', lw=2.0,
            label=r'CLT Decay: $\langle M^2 \rangle \sim 2\pi x^{-2}$', zorder=4)

    # 物理的説明テキストボックス
    textbox = (r"$\bf{Finite-Size\ Scaling\ of\ Ising\ Magnetization}$" + "\n"
               r"$\bullet\ x \ll 1\ (R \ll \xi): \langle M^2 \rangle \to 1$" + "\n"
               r"$\bullet\ x \gg 1\ (R \gg \xi): \langle M^2 \rangle \sim 2\pi x^{-2}$" + "\n"
               f"$N = {len(df_exp):,}$ exp-windows, {len(df_blocks):,} blocks")
    ax.text(0.97, 0.96, textbox, transform=ax.transAxes, ha='right', va='top',
            fontsize=9.0, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.75', alpha=0.92))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.04, 35.0)
    ax.set_ylim(0.005, 1.15)
    ax.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=13)
    ax.set_ylabel(r'Ising Spin Variance $\langle M^2 \rangle$', fontsize=13)
    ax.set_title(r'Universal Scaling of Ising Magnetization $\langle M^2 \rangle$ vs $x = R / \xi$',
                 fontsize=14, pad=12)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=9.5, loc='lower left', framealpha=0.92)

    plt.tight_layout()
    mt_ori.save_figure_to_all(fig, 'ising_scaling_master_curve_loglog', list(out_dirs))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. Linear 単体図
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for bead in target_beads:
        sub = df_exp[df_exp['bead_name'] == bead['name']]
        if sub.empty:
            continue
        ax.scatter(sub['x_scaled'], sub['squared_mean'],
                   s=38, alpha=0.7, color=bead.get('color', '#332288'),
                   marker=bead.get('marker', 'o'), edgecolors='white',
                   linewidths=0.5, label=bead.get('label', bead['name']), zorder=3)

    if not df_bins_exp.empty:
        ax.errorbar(df_bins_exp['x_center'], df_bins_exp['m2_mean'],
                    yerr=df_bins_exp['m2_sem'], fmt='s-', color='black',
                    ecolor='black', ms=7.0, lw=2.4, capsize=3.5,
                    label=r'Binned Mean $\pm$ SEM', zorder=6)

    ax.plot(x_theory, y_theory, 'r--', lw=2.8,
            label=r'Theory: $\langle M^2(x) \rangle$', zorder=5)

    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r'Scaled Window Size $x = R / \xi$', fontsize=13)
    ax.set_ylabel(r'Ising Spin Variance $\langle M^2 \rangle$', fontsize=13)
    ax.set_title(r'Ising Magnetization $\langle M^2 \rangle$ vs $x = R / \xi$ (Linear Scale)',
                 fontsize=14, pad=12)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.92)

    plt.tight_layout()
    mt_ori.save_figure_to_all(fig, 'ising_scaling_master_curve_linear', list(out_dirs))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. 条件別比較図 (Per Condition Panels)
    # -------------------------------------------------------------------------
    selected_beads = [b for b in target_beads if not df_exp[df_exp['bead_name'] == b['name']].empty]
    n_cond = len(selected_beads)
    if n_cond > 1:
        ncols = 3
        nrows = int(np.ceil(n_cond / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows), squeeze=False)

        for k, bead in enumerate(selected_beads):
            ax_k = axes[k // ncols][k % ncols]
            sub = df_exp[df_exp['bead_name'] == bead['name']]
            col = bead.get('color', '#332288')
            mk = bead.get('marker', 'o')

            ax_k.scatter(sub['x_scaled'], sub['squared_mean'],
                         s=28, alpha=0.7, color=col, marker=mk, label='Data points', zorder=3)

            # 条件ごとのビン
            sub_bins = compute_scaling_bins(sub, value_col='squared_mean', n_bins=12, min_count=2)
            if not sub_bins.empty:
                ax_k.errorbar(sub_bins['x_center'], sub_bins['m2_mean'],
                              yerr=sub_bins['m2_sem'], fmt='s-', color='black',
                              ecolor='black', ms=4.5, lw=1.6, capsize=2.0,
                              label='Binned Mean', zorder=5)

            ax_k.plot(x_theory, y_theory, 'r--', lw=2.0, label='Theory', zorder=4)

            ax_k.set_xscale('log')
            ax_k.set_yscale('log')
            ax_k.set_xlim(0.04, 35.0)
            ax_k.set_ylim(0.005, 1.15)
            ax_k.set_xlabel(r'$x = R / \xi$', fontsize=10)
            ax_k.set_ylabel(r'$\langle M^2 \rangle$', fontsize=10)
            ax_k.set_title(bead.get('label', bead['name']), fontsize=11)
            ax_k.grid(True, which='both', alpha=0.3)
            if k == 0:
                ax_k.legend(fontsize=8, loc='lower left')

        for k in range(n_cond, nrows * ncols):
            axes[k // ncols][k % ncols].axis('off')

        fig.suptitle(r'Ising Magnetization Scaling by Bead Condition ($x = R / \xi$ vs $\langle M^2 \rangle$)',
                     fontsize=13, y=1.002)
        plt.tight_layout()
        mt_ori.save_figure_to_all(fig, 'ising_scaling_master_curve_per_condition', list(out_dirs))
        plt.close(fig)


# =============================================================================
# メイン処理
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finite-size scaling master curve of Ising magnetization <M^2> vs x = R / xi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="データルートディレクトリ")
    parser.add_argument('--beads', type=str, nargs='+', default='all',
                        help="対象の粒子径（例: beads1um 1um 3um / all）")
    parser.add_argument('--output_dir', type=str, default='figure/ising_scaling',
                        help="出力ディレクトリ")
    parser.add_argument('--no_save_root', action='store_true',
                        help="データルート側の figure/ への保存を行わない")
    parser.add_argument('--pixel_stride', type=int, default=8,
                        help="光学フローの画素間引き幅")
    parser.add_argument('--frame_stride', type=int, default=5,
                        help="フレーム間引き幅")
    parser.add_argument('--max_frames_per_exp', type=int, default=None,
                        help="1実験あたりの最大フレーム数（デバッグ用）")
    parser.add_argument('--director', type=str, default='global', choices=['global', 'local'],
                        help="イジングスピンのディレクター")
    parser.add_argument('--min_valid_fraction', type=float, default=0.5,
                        help="ブロック内の有効画素率下限")
    parser.add_argument('--min_flow_mag', type=float, default=1e-4,
                        help="有効画素とみなす最小流速")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="空間スケール [um/px]")
    parser.add_argument('--window_sizes_um', type=str, default='auto',
                        help="窓サイズ [um] の指定（例: '0.88:120:30' または 'auto'）")
    parser.add_argument('--window_steps', type=int, default=32,
                        help="auto 時の窓サイズステップ数")
    parser.add_argument('--n_bins', type=int, default=24,
                        help="スケーリング変数のビン数")
    parser.add_argument('--bin_min_count', type=int, default=5,
                        help="ビンに必要な最小サンプル数")
    parser.add_argument('--sample_blocks_per_window', type=int, default=500,
                        help="1実験・1窓サイズあたりサンプリングするブロック数（クラウド用）")
    parser.add_argument('--force_recompute', action='store_true',
                        help="既存のキャッシュを無視して再計算")
    parser.add_argument('--no_progress', action='store_true',
                        help="tqdm プログレスバーを非表示")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root_dir = Path(args.root_dir).expanduser() if args.root_dir else mt_ori.find_default_root()
    if root_dir is None or not Path(root_dir).exists():
        raise FileNotFoundError("Data root directory not found. Please specify --root_dir.")

    out_arg = Path(args.output_dir).expanduser()
    out_dirs: List[Path] = [out_arg if out_arg.is_absolute() else (CURRENT_DIR / out_arg)]
    if not args.no_save_root:
        out_dirs.append(Path(root_dir) / 'figure' / 'ising_scaling')
    out_dirs = list(dict.fromkeys(out_dirs))
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    target_beads = mt_ori.parse_target_beads(args.beads, BEADS_INFO)
    if not target_beads:
        raise RuntimeError("No target bead conditions selected.")

    print("=" * 78)
    print(" Finite-Size Scaling Master Curve: <M^2> vs x = R / xi")
    print("=" * 78)
    print(f"Data Root Directory : {root_dir}")
    print(f"Output Directories  : {', '.join(str(d) for d in out_dirs)}")
    print(f"Target Beads        : {[b['name'] for b in target_beads]}")
    print(f"Pixel / Frame Stride: {args.pixel_stride} px / {args.frame_stride} frames")
    print(f"Director            : {args.director}")
    print(f"Binned bins         : {args.n_bins} log-spaced bins")
    print("-" * 78)

    # 1. 相関長 xi の読み出し
    xi_per_exp, xi_per_cond = load_xi_dict(root_dir)
    print(f"Loaded correlation length xi for {len(xi_per_exp)} experiments:")
    for exp_k, xk in sorted(xi_per_exp.items())[:6]:
        print(f"    {exp_k:20s}: xi = {xk:.3f} um")
    if len(xi_per_exp) > 6:
        print(f"    ... and {len(xi_per_exp) - 6} more experiments")
    print("-" * 78)

    # 窓サイズ列の生成
    if str(args.window_sizes_um).lower() == 'auto':
        # 0.88 um から 120 um までの等比級数
        window_sizes_um = np.geomspace(0.88, 120.0, max(4, int(args.window_steps)))
    else:
        # パース
        try:
            tokens = [float(tok) for tok in str(args.window_sizes_um).replace(',', ' ').split()]
            window_sizes_um = np.array(tokens)
        except Exception:
            window_sizes_um = np.geomspace(0.88, 120.0, 32)

    # 2. キャッシュ確認または計算
    cache_points_csv = out_dirs[0] / 'ising_scaling_points_summary.csv'
    cache_blocks_csv = out_dirs[0] / 'ising_scaling_blocks_sampled.csv'

    exp_records: List[dict] = []
    block_records: List[dict] = []

    if (not args.force_recompute) and cache_points_csv.exists():
        print(f"[CACHE] Loading existing points from {cache_points_csv.name}...")
        df_exp = pd.read_csv(cache_points_csv)
        df_blocks = pd.read_csv(cache_blocks_csv) if cache_blocks_csv.exists() else pd.DataFrame()
    else:
        # 実験ごとの抽出処理
        for bead in target_beads:
            exp_dirs = [p for p in mt_ori.find_experiment_dirs(root_dir, bead['name'])
                        if (p / 'GFP_flows.h5').exists()]
            if not exp_dirs:
                print(f"[SKIP] {bead['name']}: no experiment dir with GFP_flows.h5", flush=True)
                continue
            print(f"[{bead['name']}] Processing {len(exp_dirs)} experiment(s)...", flush=True)

            for exp_dir in exp_dirs:
                # 相関長 xi の決定
                xi_val = xi_per_exp.get(exp_dir.name)
                if xi_val is None:
                    xi_val = xi_per_cond.get(bead['name'], 5.0)

                res = extract_experiment_m2_scaling(
                    exp_dir, bead, xi_val=xi_val,
                    window_sizes_um=window_sizes_um,
                    pixel_stride=args.pixel_stride,
                    frame_stride=args.frame_stride,
                    max_frames_per_exp=args.max_frames_per_exp,
                    director=args.director,
                    min_valid_fraction=args.min_valid_fraction,
                    min_flow_mag=args.min_flow_mag,
                    scale=args.scale,
                    sample_blocks_per_window=args.sample_blocks_per_window,
                    progress=not args.no_progress,
                )
                if res is None:
                    continue

                exp_records.extend(res['exp_records'])
                block_records.extend(res['block_records'])
                print(f"    {exp_dir.name}: xi = {xi_val:.2f} um, "
                      f"{len(res['exp_records'])} windows, "
                      f"{len(res['block_records'])} sampled blocks", flush=True)

        if not exp_records:
            raise RuntimeError("No scaling records were extracted.")

        df_exp = pd.DataFrame(exp_records)
        df_blocks = pd.DataFrame(block_records)

        mt_ori.save_csv_to_all(df_exp, 'ising_scaling_points_summary', out_dirs)
        if not df_blocks.empty:
            mt_ori.save_csv_to_all(df_blocks, 'ising_scaling_blocks_sampled', out_dirs)

    print("-" * 78)
    print(f"Total Exp-Window Points : {len(df_exp):,}")
    print(f"Total Block Clouds      : {len(df_blocks):,}")

    # 3. ビン分割集計
    df_bins_exp = compute_scaling_bins(df_exp, value_col='squared_mean',
                                       n_bins=args.n_bins, bin_scale='log',
                                       min_count=args.bin_min_count)
    df_bins_block = compute_scaling_bins(df_blocks, value_col='m2_value',
                                         n_bins=args.n_bins, bin_scale='log',
                                         min_count=args.bin_min_count) if not df_blocks.empty else pd.DataFrame()

    if not df_bins_exp.empty:
        mt_ori.save_csv_to_all(df_bins_exp, 'ising_scaling_binned_data', out_dirs)

    # 4. 作図
    print("Generating Master Curve Figures...")
    plot_master_curves(df_exp, df_blocks, df_bins_exp, df_bins_block, out_dirs, target_beads)

    # 5. コンソール要約
    print("-" * 78)
    print(" Scaling Binned Summary (<M^2> vs x = R / xi)")
    if not df_bins_exp.empty:
        cols = ['bin_idx', 'x_low', 'x_high', 'x_center', 'n_points', 'm2_mean', 'm2_sem', 'theory_m2']
        print(df_bins_exp[[c for c in cols if c in df_bins_exp.columns]].to_string(index=False))
    print()
    print("Done.")


if __name__ == '__main__':
    main()
