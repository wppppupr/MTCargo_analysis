"""
vacf_analysis.py

貨物微粒子（蛍光ビーズ）の自己相関関数を一括解析・可視化するスクリプトです。
全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）を対象に、
1. 速度ベクトル自己相関 (VACF: Velocity Autocorrelation Function)
2. 配向方向自己相関 (OACF: Orientation Autocorrelation Function)
3. 速さスカラー自己相関 (SACF: Speed Autocorrelation Function)
を一括で計算・プロット・CSV保存します。
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# libsディレクトリから親ディレクトリ(MTCargo_analysis)をパスに追加
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent if current_dir.name == 'libs' else current_dir
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from libs import vacf

# スタイルの適用
style_path = parent_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ビーズ条件設定 (名前, 直径μm, マーカー, ラベル, カラー)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "label": "0.63 μm", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "label": "1.18 μm", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "label": "3.37 μm", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": 10,  "label": "5.00 μm", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": 11,  "label": "7.24 μm", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "label": "20.0 μm", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root():
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def exp_decay_model(x, xi, A):
    """
    自己相関関数のフィッティングモデル: f(x) = (1 - A) * exp(-x / xi) + A
    x=0 で f(0) = 1 となる。
    """
    return (1.0 - A) * np.exp(-x / xi) + A


def fit_acf_curve(x_data, y_data, y_err=None, p0=(5.0, 0.0), bounds=((1e-4, -2.0), (1000.0, 2.0))):
    """
    自己相関曲線に対して f(x) = (1 - A) * exp(-x / xi) + A をフィッティングする。
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return None

    sigma = None
    if y_err is not None:
        err = np.asarray(y_err)[valid]
        if np.all(err > 0) and np.all(np.isfinite(err)):
            sigma = err

    try:
        p0_init = [max(p0[0], 0.1), float(np.clip(y[-1] if len(y) > 0 else 0.0, -0.5, 0.5))]
        popt, pcov = curve_fit(
            exp_decay_model,
            x,
            y,
            p0=p0_init,
            bounds=bounds,
            sigma=sigma,
            maxfev=5000
        )
        xi_fit, A_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0]

        # 決定係数 R^2
        y_pred = exp_decay_model(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        x_fit_eval = np.linspace(0, np.max(x), 200)
        y_fit_eval = exp_decay_model(x_fit_eval, *popt)

        return {
            'xi': float(xi_fit),
            'xi_err': float(perr[0]),
            'A': float(A_fit),
            'A_err': float(perr[1]),
            'r_squared': float(r2),
            'fit_x': x_fit_eval,
            'fit_y': y_fit_eval
        }
    except Exception as e:
        print(f"    [WARNING] フィッティング失敗: {e}")
        return None


def vacf_plot(track_path, max_timeshift_frames=50, frame_interval=4.0, scale=0.11,
              mode='velocity', normalize=True, display=False, ax=None):
    """
    1つの実験データ(beads_tracks.csv)から自己相関関数を計算する。
    """
    track_df = pd.read_csv(track_path)

    IVACF = vacf.ivacf(
        track_df,
        max_timeshift_frames=max_timeshift_frames,
        frame_interval=frame_interval,
        scale=scale,
        mode=mode,
        normalize=normalize,
        display=display,
        ax=ax
    )

    EVACF, EVACF_err = vacf.evacf(
        IVACF,
        mode=mode,
        normalize=normalize,
        display=False,
        ax=ax
    )

    return IVACF, EVACF, EVACF_err


def vacf_dir(directory, max_timeshift_frames=50, frame_interval=4.0, scale=0.11,
             mode='velocity', normalize=True):
    """
    指定ビーズディレクトリ以下の全実験データについて自己相関を計算し、
    実験ごとの結果を結合したDataFrameを返す。
    """
    evacf_list = []
    directory = Path(directory)

    # 2階層下、または1階層下のディレクトリを探索 (例: directory / "date" / "exp_name" or directory / "exp_name")
    exp_dirs = [Path(p) for p in glob.glob(str(directory / "*" / "*"))]
    exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]
    if not exp_dirs:
        exp_dirs = [Path(p) for p in glob.glob(str(directory / "*"))]
        exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]

    for i, input_dir in enumerate(exp_dirs):
        track_path = input_dir / "beads_tracks.csv"
        try:
            ivacf, evacf_s, evacf_err = vacf_plot(
                track_path,
                max_timeshift_frames=max_timeshift_frames,
                frame_interval=frame_interval,
                scale=scale,
                mode=mode,
                normalize=normalize,
                display=False
            )
            df = pd.DataFrame({
                "exp": i,
                "exp_name": input_dir.name,
                "lag time": evacf_s.index,
                "VACF": evacf_s.values
            })
            evacf_list.append(df)
        except Exception as e:
            print(f"  [WARNING] エラー発生 ({input_dir.name}): {e}", flush=True)

    if not evacf_list:
        return pd.DataFrame()

    evacf_df = pd.concat(evacf_list, ignore_index=True)
    return evacf_df


def run_all_beads_analysis(root_dir, out_dir=None, modes=None, max_timeshift_frames=50,
                           frame_interval=4.0, scale=0.11, normalize=True, xlim=(0, 30),
                           yscale='linear', fit=False):
    """
    全ビーズサイズ・全モードについて自己相関関数を一括計算し、個別図・比較図・CSVを出力する。
    """
    ALL_STANDARD_MODES = [
        'velocity', 'orientation', 'speed',
        'velocity_fluctuation', 'speed_fluctuation',
        'angle_change', 'angle_change_fluctuation'
    ]
    
    if modes is None:
        modes = ['velocity', 'orientation', 'speed']
    elif isinstance(modes, str):
        if modes.lower() == 'all':
            modes = ['velocity', 'orientation', 'speed']
        elif modes.lower() in ['all_modes', 'all_with_fluc', 'full']:
            modes = ALL_STANDARD_MODES
        else:
            modes = [modes]
    elif any(m.lower() == 'all' for m in modes):
        modes = ['velocity', 'orientation', 'speed']
    elif any(m.lower() in ['all_modes', 'all_with_fluc', 'full'] for m in modes):
        modes = ALL_STANDARD_MODES

    root = Path(root_dir)
    if out_dir is None:
        out = root / 'figure' / 'autocorrelation'
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_mode_results = {}
    all_fit_results = []

    print(f"\n{'='*70}")
    print(f"ビーズ自己相関関数 (ACF) 一括解析開始")
    print(f"ルートディレクトリ: {root}")
    print(f"出力先ディレクトリ: {out}")
    print(f"解析モード: {modes}")
    print(f"パラメータ: max_lag={max_timeshift_frames} frames, interval={frame_interval}s, scale={scale} um/px, yscale={yscale}, fit={fit}")
    if fit:
        print(f"フィッティングモデル: f(t) = (1 - A) * exp(-t / xi) + A")
    print(f"{'='*70}\n")

    for mode in modes:
        mode_str = mode.lower()
        print(f"\n--- [モード: {mode_str}] の解析中 ---")
        mode_results = {}
        summary_rows = []

        fig, ax = plt.subplots(figsize=(6.5, 5.0))

        for item in BEADS_INFO:
            b_name = item["name"]
            target_dir = root / b_name
            if not target_dir.exists():
                print(f"  ディレクトリが見つかりません: {target_dir}")
                continue

            print(f"  処理中: {item['label']} ({b_name}) ...", flush=True)
            evacf_df = vacf_dir(
                target_dir,
                max_timeshift_frames=max_timeshift_frames,
                frame_interval=frame_interval,
                scale=scale,
                mode=mode_str,
                normalize=normalize
            )

            if evacf_df.empty:
                print(f"    -> データがありません: {b_name}")
                continue

            # アンサンブル集計 (lag time ごとの平均と標準偏差)
            evacf_mean = evacf_df.groupby('lag time')['VACF'].mean()
            evacf_std = evacf_df.groupby('lag time')['VACF'].std().fillna(0.0)
            n_exps = evacf_df['exp'].nunique()

            # 指数減衰フィッティング
            fit_res = None
            if fit and normalize:
                fit_res = fit_acf_curve(
                    evacf_mean.index.to_numpy(),
                    evacf_mean.values,
                    y_err=evacf_std.values
                )
                if fit_res is not None:
                    all_fit_results.append({
                        'mode': mode_str,
                        'bead_name': b_name,
                        'diameter_um': item["diameter_um"],
                        'xi_s': fit_res['xi'],
                        'xi_err_s': fit_res['xi_err'],
                        'A': fit_res['A'],
                        'A_err': fit_res['A_err'],
                        'r_squared': fit_res['r_squared'],
                        'n_experiments': n_exps
                    })
                    print(f"    -> フィッティング結果: xi = {fit_res['xi']:.2f} s, A = {fit_res['A']:.3f}, R^2 = {fit_res['r_squared']:.3f}")

            mode_results[b_name] = {
                "mean": evacf_mean,
                "std": evacf_std,
                "n_exps": n_exps,
                "raw_df": evacf_df,
                "fit_res": fit_res
            }

            # CSVサマリー用データの集約
            for lag_t, mean_val, std_val in zip(evacf_mean.index, evacf_mean.values, evacf_std.values):
                summary_rows.append({
                    "mode": mode_str,
                    "bead_name": b_name,
                    "diameter_um": item["diameter_um"],
                    "lag_time_s": lag_t,
                    "mean_acf": mean_val,
                    "std_acf": std_val,
                    "n_experiments": n_exps
                })

            # 単体プロットへの描画
            label_text = f'{item["label"]}'
            if fit_res is not None:
                label_text += f' ($\\xi={fit_res["xi"]:.1f}\\mathrm{{s}}$, $R^2={fit_res["r_squared"]:.2f}$)'
            else:
                label_text += f' ($N={n_exps}$)'

            ax.errorbar(
                evacf_mean.index,
                evacf_mean.values,
                yerr=evacf_std.values,
                marker=item["marker"],
                color=item["color"],
                label=label_text,
                capsize=3,
                elinewidth=1.0,
                markersize=6,
                alpha=0.9,
                linestyle='none' if fit_res is not None else '-'
            )

            # フィッティング曲線の描画（破線）
            if fit_res is not None:
                ax.plot(
                    fit_res['fit_x'],
                    fit_res['fit_y'],
                    linestyle='--',
                    color=item["color"],
                    alpha=0.85,
                    linewidth=1.5
                )

        ax.legend(frameon=True, fontsize=8)
        max_x = xlim[1] if xlim is not None else max_timeshift_frames * frame_interval
        if yscale == 'linear':
            ax.hlines(0, 0, max_x, colors='#333333', linestyles='dashed', alpha=0.6, zorder=-1)

        ylabel, title = vacf._get_ylabel_and_title(mode_str, normalize, is_ensemble=True)
        ax.set(
            xlim=xlim if xlim is not None else (0, max_x),
            ylim=(-0.3, 1.05) if mode_str == 'orientation' else ((-0.3, 1.05) if normalize else None) if yscale == 'linear' else (1e-1, 1.05) if normalize and yscale == 'log' else None,
            yscale=yscale,
            xlabel=r'lag time $\Delta t$ [s]',
            ylabel=ylabel,
        )
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        save_svg = out / f"{mode_str.upper()}_ACF.svg"
        save_png = out / f"{mode_str.upper()}_ACF.png"
        fig.savefig(save_svg, bbox_inches='tight')
        fig.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [保存完了] {save_svg}")

        # CSVの保存
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            csv_path = out / f"{mode_str.upper()}_summary.csv"
            df_summary.to_csv(csv_path, index=False)
            print(f"  [保存完了] {csv_path}")

        all_mode_results[mode_str] = mode_results

    # フィッティングサマリー CSV の保存
    if all_fit_results:
        df_fits = pd.DataFrame(all_fit_results)
        fit_csv_path = out / "ACF_fits_summary.csv"
        df_fits.to_csv(fit_csv_path, index=False)
        print(f"  [保存完了] フィッティング結果サマリー: {fit_csv_path}")

    # 複数モードが存在する場合、横並び比較図も生成
    if len(all_mode_results) > 1:
        _plot_combined_comparison(all_mode_results, out, normalize, xlim, frame_interval, max_timeshift_frames, yscale, fit)

    print(f"\n{'='*70}")
    print(f"全ビーズの自己相関解析が完了しました。出力先: {out}")
    print(f"{'='*70}\n")
    return all_mode_results


def _plot_combined_comparison(all_mode_results, out_dir, normalize, xlim, frame_interval, max_timeshift_frames, yscale, fit=False):
    """
    複数モードを並べた総合比較プロットを作成・保存する。
    """
    mode_keys = list(all_mode_results.keys())
    n_modes = len(mode_keys)
    if n_modes == 0:
        return

    fig, axes = plt.subplots(1, n_modes, figsize=(5.5 * n_modes, 4.8), squeeze=False)
    max_x = xlim[1] if xlim is not None else max_timeshift_frames * frame_interval

    for col_idx, m_key in enumerate(mode_keys):
        ax = axes[0, col_idx]
        mode_data = all_mode_results[m_key]

        for item in BEADS_INFO:
            b_name = item["name"]
            if b_name not in mode_data:
                continue

            res = mode_data[b_name]
            mean_s = res["mean"]
            std_s = res["std"]
            n_exps = res["n_exps"]
            fit_res = res.get("fit_res", None)

            label_text = f"{item['label']}"
            if fit and fit_res is not None:
                label_text += f" ($\\xi={fit_res['xi']:.1f}\\mathrm{{s}}$)"

            ax.errorbar(
                mean_s.index,
                mean_s.values,
                yerr=std_s.values,
                marker=item["marker"],
                color=item["color"],
                label=label_text,
                capsize=2,
                elinewidth=0.8,
                markersize=5,
                alpha=0.9,
                linestyle='none' if (fit and fit_res is not None) else '-'
            )

            if fit and fit_res is not None:
                ax.plot(
                    fit_res['fit_x'],
                    fit_res['fit_y'],
                    linestyle='--',
                    color=item["color"],
                    alpha=0.85,
                    linewidth=1.2
                )
         
        if yscale == 'linear':
            ax.hlines(0, 0, max_x, colors='#333333', linestyles='dashed', alpha=0.6, zorder=-1)

        ylabel, title = vacf._get_ylabel_and_title(m_key, normalize, is_ensemble=True)
        ax.set(
            xlim=xlim if xlim is not None else (0, max_x),
            ylim=(-0.3, 1.05) if normalize and yscale == 'linear' else (1e-3, 1.05) if normalize and yscale == 'log' else None,
            xlabel=r'lag time $\Delta t$ [s]',
            ylabel=ylabel,
            yscale=yscale,
        )
        ax.grid(True, linestyle='--', alpha=0.4)
        if col_idx == 0:
            ax.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    comb_svg = out_dir / "all_ACF_comparison.svg"
    comb_png = out_dir / "all_ACF_comparison.png"
    fig.savefig(comb_svg, bbox_inches='tight')
    fig.savefig(comb_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 比較プロット: {comb_svg}")


def main():
    parser = argparse.ArgumentParser(
        description="Cargo particle velocity, orientation, speed, and fluctuation autocorrelation function (ACF) batch analysis."
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions (e.g. /Volumes/data/Sasaki/MTsingleBeads).")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Directory to save figures and CSVs (default: root_dir/figure/autocorrelation).")
    parser.add_argument('--mode', type=str, nargs='+', default=['all'],
                        choices=['all', 'all_with_fluc', 'velocity', 'orientation', 'speed',
                                 'velocity_fluctuation', 'speed_fluctuation',
                                 'angle_change', 'angle_change_fluctuation'],
                        help="Autocorrelation mode(s): 'all', 'all_with_fluc', 'velocity', 'orientation', 'speed', 'velocity_fluctuation', 'speed_fluctuation', 'angle_change', 'angle_change_fluctuation'.")
    parser.add_argument('--max_lag', type=int, default=50,
                        help="Maximum lag time in frames (default: 50).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames in seconds (default: 4.0).")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale in um/pixel (default: 0.11).")
    parser.add_argument('--no_norm', dest='normalize', action='store_false',
                        help="Disable normalization by tau=0 value.")
    parser.add_argument('--xlim_max', type=float, default=30.0,
                        help="Maximum lag time for x-axis in plots (default: 30.0 s).")
    parser.add_argument('--yscale', type=str, default='linear',
                        choices=['linear', 'log'],
                        help="Y-axis scale for plots (default: 'linear').")
    parser.add_argument('--fit', action='store_true', default=False,
                        help="Fit autocorrelation curves with model f(t) = (1 - A) * exp(-t / xi) + A.")

    args = parser.parse_args()

    # root_dir の解決
    root_dir = args.root_dir
    if root_dir is None:
        root_dir = find_default_root()
    root_dir = Path(root_dir)

    modes = args.mode
    if 'all' in modes:
        modes = ['velocity', 'orientation', 'speed']
    elif 'all_with_fluc' in modes:
        modes = ['velocity', 'orientation', 'speed', 'velocity_fluctuation', 'speed_fluctuation']

    run_all_beads_analysis(
        root_dir=root_dir,
        out_dir=args.out_dir,
        modes=modes,
        max_timeshift_frames=args.max_lag,
        frame_interval=args.frame_interval,
        scale=args.scale,
        normalize=args.normalize,
        xlim=(0, args.xlim_max) if args.xlim_max > 0 else None,
        yscale=args.yscale,
        fit=args.fit
    )


if __name__ == "__main__":
    main()
