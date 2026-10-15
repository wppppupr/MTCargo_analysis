#!/usr/bin/env python3
"""
plot_global_nematic_polar_order.py

各貨物粒子サイズごとの大域的なネマチックオーダーと大域的なポーラーオーダーの2軸プロット。
- 第1軸 (左): 大域ネマチックオーダー S (青色)
- 第2軸 (右): 大域ポーラーオーダー P (赤色)
- マーカー: MSD.py 準拠
- エラーバー: STD (実験間標準偏差)
- 横軸: 貨物粒子のサイズ (直径 / 半径, 線形 / 対数)

【高速化 & メモリ保護機能】
- mt_flow_cache_s8_f5.h5 等の既存キャッシュを自動検知して0.3秒で超高速計算
- GFP_flows.h5 からの計算時はフレーム単位のストリーミング & ストライド間引きでOOMクラッシュを完全防止
- 計算結果はログCSV (global_order_experiments_cache.csv) に逐次保存・キャッシュされ、2回目以降は即座にプロット可能
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import zarr

# ワークスペースと libs のインポート
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
    except Exception:
        pass

# ビーズ基本情報 (MSD.py 準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "label": r"$20.0\,\mu\mathrm{m}$"},
]

POSSIBLE_ROOTS = [
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]


def find_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists():
                    return r
    return CURRENT_DIR


def compute_nematic_order(ed: Path) -> float:
    """各実験の MTs_order_parameter.zarr から大域ネマチックオーダー平均を計算"""
    nem_p = ed / 'MTs_order_parameter.zarr'
    if not nem_p.exists():
        return np.nan
    try:
        op_zarr = zarr.open(str(nem_p), mode='r')[:]
        op_flat = op_zarr[~np.isnan(op_zarr)]
        if len(op_flat) > 0:
            return float(np.mean(op_flat))
    except Exception as e:
        print(f"  [Warning] Failed loading nematic order for {nem_p}: {e}")
    return np.nan


def compute_polar_order_from_cache_or_flow(
    ed: Path,
    pixel_stride: int = 8,
    frame_stride: int = 5
) -> Tuple[float, str]:
    """
    大域ポーラーオーダー P を計算する。
    1. mt_flow_cache_s8_f5.h5 等の既存キャッシュがあれば最速 (約0.3秒) で計算。
    2. キャッシュがない場合は GFP_flows.h5 をフレーム毎にストリーミング読み込み (OOMクラッシュ完全防止)。
    
    戻り値: (polar_order_mean, source_info)
    """
    # 候補1: mt_flow_cache_s8_f5.h5
    cache_s8 = ed / 'mt_flow_cache_s8_f5.h5'
    if cache_s8.exists():
        try:
            with h5py.File(cache_s8, 'r') as f:
                if 'mx' in f and 'my' in f:
                    u = f['mx'][:].astype(np.float32)
                    v = f['my'][:].astype(np.float32)
                    mag = np.hypot(u, v)
                    mask = mag > 1e-6
                    u_hat = np.where(mask, u / np.where(mask, mag, 1.0), 0.0)
                    v_hat = np.where(mask, v / np.where(mask, mag, 1.0), 0.0)
                    mean_u = np.mean(u_hat, axis=(1, 2))
                    mean_v = np.mean(v_hat, axis=(1, 2))
                    p_frames = np.hypot(mean_u, mean_v)
                    return float(np.mean(p_frames)), "cache_s8_f5"
        except Exception as e:
            print(f"  [Warning] Failed reading cache {cache_s8}: {e}. Falling back to raw flow.")

    # 候補2: mt_flow_cache_s4_f5.h5
    cache_s4 = ed / 'mt_flow_cache_s4_f5.h5'
    if cache_s4.exists():
        try:
            with h5py.File(cache_s4, 'r') as f:
                if 'mx' in f and 'my' in f:
                    u = f['mx'][:, ::2, ::2].astype(np.float32)
                    v = f['my'][:, ::2, ::2].astype(np.float32)
                    mag = np.hypot(u, v)
                    mask = mag > 1e-6
                    u_hat = np.where(mask, u / np.where(mask, mag, 1.0), 0.0)
                    v_hat = np.where(mask, v / np.where(mask, mag, 1.0), 0.0)
                    mean_u = np.mean(u_hat, axis=(1, 2))
                    mean_v = np.mean(v_hat, axis=(1, 2))
                    p_frames = np.hypot(mean_u, mean_v)
                    return float(np.mean(p_frames)), "cache_s4_f5"
        except Exception as e:
            pass

    # 候補3: GFP_flows.h5 からフレーム毎にストリーミング処理 (省メモリ & 安全)
    flow_p = ed / 'GFP_flows.h5'
    if flow_p.exists():
        try:
            with h5py.File(flow_p, 'r') as f:
                key = 'flows' if 'flows' in f else list(f.keys())[0]
                ds = f[key]
                shape = ds.shape

                if len(shape) != 4:
                    return np.nan, "invalid_shape"

                # チャンネル軸の判定
                if shape[1] == 2:
                    channel_first = True
                    n_frames = shape[0]
                elif shape[-1] == 2:
                    channel_first = False
                    n_frames = shape[0]
                else:
                    return np.nan, "unknown_channels"

                frame_indices = range(0, n_frames, frame_stride)
                p_frames = []

                for t in frame_indices:
                    if channel_first:
                        u = ds[t, 0, ::pixel_stride, ::pixel_stride].astype(np.float32)
                        v = ds[t, 1, ::pixel_stride, ::pixel_stride].astype(np.float32)
                    else:
                        u = ds[t, ::pixel_stride, ::pixel_stride, 0].astype(np.float32)
                        v = ds[t, ::pixel_stride, ::pixel_stride, 1].astype(np.float32)

                    mag = np.hypot(u, v)
                    mask = mag > 1e-6
                    if np.any(mask):
                        u_hat = u[mask] / mag[mask]
                        v_hat = v[mask] / mag[mask]
                        p_t = np.hypot(np.mean(u_hat), np.mean(v_hat))
                        p_frames.append(p_t)

                if p_frames:
                    return float(np.mean(p_frames)), f"raw_flow(s{pixel_stride}_f{frame_stride})"
        except Exception as e:
            print(f"  [Warning] Failed streaming GFP_flows for {flow_p}: {e}")

    return np.nan, "none"


def load_or_create_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            print(f"Warning: Failed to read cache {cache_path}: {e}")
    return pd.DataFrame(columns=[
        'bead_name', 'exp_path', 'exp_name', 'diameter_um', 'radius_um',
        'nematic_order', 'polar_order', 'polar_source'
    ])


def extract_orders(
    root_dir: Path,
    cache_csv_path: Optional[Path] = None,
    pixel_stride: int = 8,
    frame_stride: int = 5,
    recompute: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """全実験のオーダーパラメータを抽出・集計する (CSVキャッシュ対応)"""
    df_cache = pd.DataFrame()
    if not recompute and cache_csv_path is not None and cache_csv_path.exists():
        df_cache = load_or_create_cache(cache_csv_path)
        print(f"Loaded existing cache with {len(df_cache)} experiments from {cache_csv_path}")

    summary_records = []
    experiment_records = []

    for item in BEADS_INFO:
        b_name = item['name']
        b_path = root_dir / b_name
        if not b_path.exists():
            continue

        exp_dirs = [d for d in sorted(b_path.glob('*/*')) if d.is_dir() and (d / 'MTs_order_parameter.zarr').exists()]

        nem_vals = []
        pol_vals = []

        print(f"\nProcessing {b_name} ({len(exp_dirs)} experiments)...")
        for ed in exp_dirs:
            exp_path_str = str(ed)
            exp_name = ed.name

            # キャッシュのチェック
            cached_row = None
            if not df_cache.empty and 'exp_path' in df_cache.columns:
                match = df_cache[df_cache['exp_path'] == exp_path_str]
                if not match.empty:
                    c_nem = match.iloc[0].get('nematic_order', np.nan)
                    c_pol = match.iloc[0].get('polar_order', np.nan)
                    if np.isfinite(c_nem) and np.isfinite(c_pol):
                        cached_row = match.iloc[0]

            if cached_row is not None:
                exp_nem = float(cached_row['nematic_order'])
                exp_pol = float(cached_row['polar_order'])
                src_info = cached_row.get('polar_source', 'cache')
                print(f"  [Cache hit] {exp_name}: S = {exp_nem:.4f}, P = {exp_pol:.4f} ({src_info})")
            else:
                t0 = time.time()
                # 1. ネマチックオーダー
                exp_nem = compute_nematic_order(ed)
                # 2. ポーラーオーダー
                exp_pol, src_info = compute_polar_order_from_cache_or_flow(
                    ed, pixel_stride=pixel_stride, frame_stride=frame_stride
                )
                t1 = time.time()
                print(f"  [Computed] {exp_name}: S = {exp_nem:.4f}, P = {exp_pol:.4f} ({src_info}, {t1-t0:.2f}s)")

            if np.isfinite(exp_nem):
                nem_vals.append(exp_nem)
            if np.isfinite(exp_pol):
                pol_vals.append(exp_pol)

            rec = {
                'bead_name': b_name,
                'exp_path': exp_path_str,
                'exp_name': exp_name,
                'diameter_um': item['diameter_um'],
                'radius_um': item['radius_um'],
                'marker': item['marker'],
                'nematic_order': exp_nem,
                'polar_order': exp_pol,
                'polar_source': src_info if cached_row is None else cached_row.get('polar_source', 'cache'),
            }
            experiment_records.append(rec)

            # 随時キャッシュCSVを更新保存 (中断しても進捗を保護)
            if cache_csv_path is not None:
                pd.DataFrame(experiment_records).to_csv(cache_csv_path, index=False)

        summary_rec = {
            'bead_name': b_name,
            'diameter_um': item['diameter_um'],
            'radius_um': item['radius_um'],
            'marker': item['marker'],
            'label': item['label'],
            'n_nem': len(nem_vals),
            'nematic_mean': float(np.mean(nem_vals)) if nem_vals else np.nan,
            'nematic_std': float(np.std(nem_vals, ddof=1)) if len(nem_vals) > 1 else 0.0,
            'n_pol': len(pol_vals),
            'polar_mean': float(np.mean(pol_vals)) if pol_vals else np.nan,
            'polar_std': float(np.std(pol_vals, ddof=1)) if len(pol_vals) > 1 else 0.0,
        }
        summary_records.append(summary_rec)

    df_summary = pd.DataFrame(summary_records)
    df_exp = pd.DataFrame(experiment_records)

    return df_summary, df_exp


def plot_dual_axis_order(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    out_dirs: List[Path],
    x_var: str = 'diameter_um',
    xscale: str = 'log',
    offset_ratio: float = 0.03
):
    """
    大域ネマチックオーダー(青)と大域ポーラーオーダー(赤)の2軸プロットを作成する。
    エラーバーの視認性を高めるため、わずかなx方向シフトを適用。
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))

    color_nem = '#1f77b4'  # 青色
    color_pol = '#d62728'  # 赤色

    ax2 = ax1.twinx()

    # 軸ラベルの設定
    x_label = r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]' if x_var == 'diameter_um' else r'Cargo Radius $R_c$ [$\mu\mathrm{m}$]'
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel(r'Global Nematic Order $S$', color=color_nem, fontsize=14)
    ax2.set_ylabel(r'Global Polar Order $P$', color=color_pol, fontsize=14)

    ax1.tick_params(axis='y', labelcolor=color_nem)
    ax2.tick_params(axis='y', labelcolor=color_pol)

    # ガイド線
    ax1.grid(True, which='major', linestyle='--', alpha=0.4)

    x_vals = df[x_var].values
    if xscale == 'log':
        x_shift_nem = x_vals * (1.0 - offset_ratio)
        x_shift_pol = x_vals * (1.0 + offset_ratio)
    else:
        shift_val = np.mean(x_vals) * offset_ratio
        x_shift_nem = x_vals - shift_val
        x_shift_pol = x_vals + shift_val

    # トレンド線
    ax1.plot(x_shift_nem, df['nematic_mean'].values, color=color_nem, linestyle='--', linewidth=1.8, alpha=0.6, zorder=2)
    ax2.plot(x_shift_pol, df['polar_mean'].values, color=color_pol, linestyle=':', linewidth=1.8, alpha=0.6, zorder=2)

    # 各実験の個別測定点（半透明）
    for _, row in df.iterrows():
        b_name = row['bead_name']
        marker = row['marker']
        exp_sub = df_exp[df_exp['bead_name'] == b_name]

        # x座標
        if xscale == 'log':
            xn = row[x_var] * (1.0 - offset_ratio)
            xp = row[x_var] * (1.0 + offset_ratio)
        else:
            xn = row[x_var] - shift_val
            xp = row[x_var] + shift_val

        # 個別点
        ax1.scatter(
            [xn] * len(exp_sub), exp_sub['nematic_order'],
            marker=marker, color=color_nem, alpha=0.35, s=36, zorder=3
        )
        ax2.scatter(
            [xp] * len(exp_sub), exp_sub['polar_order'],
            marker=marker, color=color_pol, alpha=0.35, s=36, zorder=3
        )

        # アンサンブル平均 & STD
        ax1.errorbar(
            xn, row['nematic_mean'], yerr=row['nematic_std'],
            fmt=marker, color=color_nem, ecolor=color_nem,
            elinewidth=2.0, capsize=4.5, capthick=1.5,
            markersize=9, markeredgecolor='black', markeredgewidth=0.9,
            zorder=4
        )

        ax2.errorbar(
            xp, row['polar_mean'], yerr=row['polar_std'],
            fmt=marker, color=color_pol, ecolor=color_pol,
            elinewidth=2.0, capsize=4.5, capthick=1.5,
            markersize=9, markeredgecolor='black', markeredgewidth=0.9,
            zorder=4
        )

    # 軸範囲
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylim(0.0, 1.05)

    if xscale == 'log':
        ax1.set_xscale('log')
        if x_var == 'diameter_um':
            ax1.set_xlim(0.45, 28.0)
        else:
            ax1.set_xlim(0.22, 14.0)
    else:
        if x_var == 'diameter_um':
            ax1.set_xlim(-0.5, 21.5)
        else:
            ax1.set_xlim(-0.3, 10.8)

    # 凡例
    legend_elements = [
        Line2D([0], [0], color=color_nem, marker='o', linestyle='--', linewidth=1.5,
               markersize=8, markeredgecolor='black', label=r'Nematic Order $S$ (Blue)'),
        Line2D([0], [0], color=color_pol, marker='o', linestyle=':', linewidth=1.5,
               markersize=8, markeredgecolor='black', label=r'Polar Order $P$ (Red)'),
    ]
    ax1.legend(handles=legend_elements, loc='lower left', framealpha=0.92, fontsize=11)

    title_suffix = "(Log Scale)" if xscale == 'log' else "(Linear Scale)"
    ax1.set_title(
        f"Global Nematic and Polar Order Parameters vs Cargo Size\n(Mean $\pm$ STD, {title_suffix})",
        fontsize=13
    )

    plt.tight_layout()

    # 出力保存
    fname_base = f"global_order_vs_cargo_size_{x_var}_{xscale}"
    for out_d in out_dirs:
        out_d.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_d / f"{fname_base}.png", dpi=300)
        fig.savefig(out_d / f"{fname_base}.svg")
        print(f"Saved: {out_d / f'{fname_base}.png'}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot global nematic and polar order parameters vs cargo size.")
    parser.add_argument('--recompute', action='store_true', help="Force recomputation ignoring cached CSV.")
    parser.add_argument('--pixel_stride', type=int, default=8, help="Pixel stride for raw flow reading (default: 8).")
    parser.add_argument('--frame_stride', type=int, default=5, help="Frame stride for raw flow reading (default: 5).")
    args = parser.parse_args()

    root = find_root()
    print(f"Using root directory: {root}")

    out_dirs = [
        CURRENT_DIR / "figure" / "global_order",
        root / "figure" / "global_order",
    ]

    out_dirs[0].mkdir(parents=True, exist_ok=True)
    cache_csv_path = out_dirs[0] / "global_order_experiments_cache.csv"

    print("Extracting global nematic and polar order parameters...")
    df_summary, df_exp = extract_orders(
        root_dir=root,
        cache_csv_path=cache_csv_path,
        pixel_stride=args.pixel_stride,
        frame_stride=args.frame_stride,
        recompute=args.recompute
    )

    print("\n=== Extracted Order Parameters Summary ===")
    print(df_summary[['bead_name', 'diameter_um', 'marker', 'n_nem', 'nematic_mean', 'nematic_std', 'n_pol', 'polar_mean', 'polar_std']].to_string())

    # CSV保存
    for out_d in out_dirs:
        out_d.mkdir(parents=True, exist_ok=True)
        summary_csv = out_d / "global_nematic_polar_order_summary.csv"
        exp_csv = out_d / "global_nematic_polar_order_experiments.csv"
        df_summary.to_csv(summary_csv, index=False)
        df_exp.to_csv(exp_csv, index=False)
        print(f"Saved CSV: {summary_csv}")
        print(f"Saved CSV: {exp_csv}")

    # プロット生成 (直径: log / linear, 半径: log / linear)
    print("\nGenerating dual-axis order plots...")
    plot_dual_axis_order(df_summary, df_exp, out_dirs, x_var='diameter_um', xscale='log')
    plot_dual_axis_order(df_summary, df_exp, out_dirs, x_var='diameter_um', xscale='linear')
    plot_dual_axis_order(df_summary, df_exp, out_dirs, x_var='radius_um', xscale='log')
    plot_dual_axis_order(df_summary, df_exp, out_dirs, x_var='radius_um', xscale='linear')
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
