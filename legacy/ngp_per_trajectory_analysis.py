#!/usr/bin/env python3
"""
ngp_per_trajectory_analysis.py

プールした NGP は「軌道ごとの拡散の速さ（MSD）の違い」だけでも O(1) の値になる。
例えば各軌道がガウス分布で分散 s_i だけが違うとき（2D）、
    1 + α_pool = <s_i^2> / <s_i>^2   →   α_pool = CV^2(s_i)
となる。この寄与を取り除くため、以下の 3 種類を比較する。

  (A) pooled       : 従来どおり全軌道の変位をまとめて α_2 を計算
  (B) per-traj     : 軌道ごとに α_2,i(τ) を計算し、軌道について平均（重み = 変位サンプル数）
  (C) normalized   : 各軌道の変位を、その軌道自身の sqrt(<r^2>_i(τ)) で割ってからまとめて α_2 を計算
  (H) hetero       : 軌道間 MSD のばらつき由来の寄与  CV_w^2(s_i) = <s_i^2>_w/<s_i>_w^2 - 1

厳密な分解:  1 + α_pool = Σ w_i (1+α_i) s_i^2 / (Σ w_i s_i)^2   （w_i = 軌道 i のサンプル数割合）

軌道ごとの尖度推定は短い軌道ほど負に偏る（有限サンプル・重なりのある変位）。
そのため、同じ軌道長・同じ欠損パターンの 2D ブラウン運動を生成し、
同じ推定量を適用した「ガウス帰無基準線」も描く。

誤差: 軌道単位のクラスタブートストラップ（軌道を復元抽出）で 95% CI。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from libs.ngp_analysis import BEADS_INFO, find_default_root, load_theta_array  # noqa: E402
from libs.experiment_cutoff import get_cutoff_frame, load_strict_active_particles  # noqa: E402
from libs.ngp_analysis import glob  # noqa: E402

style_path = ROOT / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))


# ----------------------------------------------------------------------------
# データ読み込み
# ----------------------------------------------------------------------------
def load_dense_tracks(target_dir, scale=0.11, component='2d'):
    """1 ビーズ条件の全軌道を (key, x_dense, y_dense) のリストで返す（欠損は NaN）。
    component が parallel/perpendicular の場合は (d_par, d_perp) 用に theta も返す。"""
    target_dir = Path(target_dir)
    exp_dirs = [Path(p) for p in glob.glob(str(target_dir / "*" / "*"))]
    exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]
    if not exp_dirs:
        exp_dirs = [Path(p) for p in glob.glob(str(target_dir / "*"))]
        exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]
    strict_set = load_strict_active_particles()
    tracks = []
    for exp_dir in sorted(exp_dirs):
        df = pd.read_csv(exp_dir / "beads_tracks.csv")
        cut = get_cutoff_frame(exp_dir.name)
        if cut is not None:
            df = df[df['frame'] <= cut]
        theta = None
        if component in ('parallel', 'perpendicular'):
            theta = load_theta_array(exp_dir)
            if theta is None:
                continue
        for pid, g in df.groupby('particle'):
            if strict_set and (exp_dir.name, int(pid)) not in strict_set:
                continue
            g = g.sort_values('frame')
            fr = g['frame'].to_numpy().astype(int)
            n = fr[-1] - fr[0] + 1
            xd = np.full(n, np.nan)
            yd = np.full(n, np.nan)
            xd[fr - fr[0]] = scale * g['x'].to_numpy()
            yd[fr - fr[0]] = scale * g['y'].to_numpy()
            th = None
            if theta is not None:
                idx = np.arange(fr[0], fr[0] + n)
                th = np.full(n, np.nan)
                ok = idx < len(theta)
                th[ok] = theta[idx[ok]]
            tracks.append({'key': (exp_dir.name, int(pid)), 'x': xd, 'y': yd, 'th': th})
    return tracks


def track_displacements(tr, tau, component='2d'):
    x, y = tr['x'], tr['y']
    if len(x) <= tau:
        return np.empty(0)
    dx = x[tau:] - x[:-tau]
    dy = y[tau:] - y[:-tau]
    if component == '2d':
        r = np.sqrt(dx**2 + dy**2)
    elif component == 'x':
        r = dx
    elif component == 'y':
        r = dy
    else:
        th = tr['th'][:-tau]
        if component == 'parallel':
            r = dx * np.cos(th) + dy * np.sin(th)
        else:
            r = -dx * np.sin(th) + dy * np.cos(th)
    return r[np.isfinite(r)]


# ----------------------------------------------------------------------------
# NGP 推定量
# ----------------------------------------------------------------------------
def per_track_moments(tracks, tau, component, min_n):
    """各軌道の (n_i, s_i=<r^2>_i, q_i=<r^4>_i) を返す。"""
    out = []
    for tr in tracks:
        r = track_displacements(tr, tau, component)
        if len(r) < min_n:
            continue
        r2 = r**2
        s = r2.mean()
        if s <= 0:
            continue
        out.append((len(r), s, (r2**2).mean()))
    return np.array(out) if out else np.empty((0, 3))


def ngp_estimators(M, factor):
    """M: (n_i, s_i, q_i) の配列 → 各推定量の dict。"""
    n, s, q = M[:, 0], M[:, 1], M[:, 2]
    w = n / n.sum()
    pooled = factor * (w @ q) / (w @ s)**2 - 1.0
    a_i = factor * q / s**2 - 1.0
    per_traj = w @ a_i                        # サンプル数重み付き軌道平均
    per_traj_unw = a_i.mean()                 # 単純軌道平均
    normalized = factor * (w @ (q / s**2)) - 1.0   # 規格化してプール（= per_traj と数学的に一致）
    hetero = (w @ s**2) / (w @ s)**2 - 1.0    # MSD の軌道間ばらつき
    return {'pooled': pooled, 'per_traj': per_traj, 'per_traj_unw': per_traj_unw,
            'normalized': normalized, 'hetero': hetero}


def cluster_bootstrap(M, factor, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    K = len(M)
    idx = rng.integers(0, K, size=(n_boot, K))
    keys = ['pooled', 'per_traj', 'per_traj_unw', 'hetero']
    res = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        e = ngp_estimators(M[idx[b]], factor)
        for k in keys:
            res[k][b] = e[k]
    return {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in res.items()}


def brownian_surrogate(tracks, rng):
    """同じ長さ・同じ欠損パターンの等方 2D ブラウン運動（D 共通）。"""
    sur = []
    for tr in tracks:
        n = len(tr['x'])
        x = np.cumsum(rng.normal(size=n))
        y = np.cumsum(rng.normal(size=n))
        mask = ~np.isfinite(tr['x'])
        x[mask] = np.nan
        y[mask] = np.nan
        th = None if tr['th'] is None else np.zeros(n)
        sur.append({'key': tr['key'], 'x': x, 'y': y, 'th': th})
    return sur


# ----------------------------------------------------------------------------
def analyze_condition(tracks, taus, component, min_n, n_boot, n_null):
    factor = 0.5 if component == '2d' else 1.0 / 3.0
    rng = np.random.default_rng(42)
    rows = []
    for tau in taus:
        M = per_track_moments(tracks, tau, component, min_n)
        if len(M) < 3:
            continue
        est = ngp_estimators(M, factor)
        ci = cluster_bootstrap(M, factor, n_boot, rng)
        row = {'tau': tau, 'n_tracks': len(M), 'n_disp': int(M[:, 0].sum()),
               'median_n_per_track': float(np.median(M[:, 0]))}
        for k, v in est.items():
            row[k] = v
        for k, (lo, hi) in ci.items():
            row[f'{k}_lo'] = lo
            row[f'{k}_hi'] = hi
        rows.append(row)
    df = pd.DataFrame(rows)
    # ガウス帰無基準（同じ推定量をブラウン運動サロゲートへ適用）
    null = {t: [] for t in df['tau']}
    for _ in range(n_null):
        sur = brownian_surrogate(tracks, rng)
        for tau in df['tau']:
            M = per_track_moments(sur, tau, component, min_n)
            null[tau].append(ngp_estimators(M, factor)['per_traj'])
    df['null_per_traj'] = [np.mean(null[t]) for t in df['tau']]
    df['null_per_traj_lo'] = [np.percentile(null[t], 2.5) for t in df['tau']]
    df['null_per_traj_hi'] = [np.percentile(null[t], 97.5) for t in df['tau']]
    df['per_traj_excess'] = df['per_traj'] - df['null_per_traj']
    return df


def plot_panels(results, out_dir, component, frame_interval):
    sizes = [b for b in BEADS_INFO if b['name'] in results]
    nb = len(sizes)
    ncol = 3
    nrow = int(np.ceil(nb / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.8 * nrow), sharex=True, squeeze=False)
    series = [('pooled', 'pooled (conventional)', '#332288', '-'),
              ('hetero', 'MSD heterogeneity  CV²(s_i)', '#999933', '--'),
              ('per_traj', 'per-trajectory / normalized', '#CC6677', '-')]
    for ax, b in zip(axes.flat, sizes):
        df = results[b['name']]
        t = df['tau'] * frame_interval
        for k, lab, c, ls in series:
            ax.plot(t, df[k], color=c, ls=ls, lw=2, label=lab)
            ax.fill_between(t, df[f'{k}_lo'], df[f'{k}_hi'], color=c, alpha=0.18, lw=0)
        ax.plot(t, df['null_per_traj'], color='0.4', ls=':', lw=1.8, label='Brownian null (per-traj)')
        ax.fill_between(t, df['null_per_traj_lo'], df['null_per_traj_hi'], color='0.6', alpha=0.2, lw=0)
        ax.axhline(0, color='0.75', lw=0.8)
        ax.set_title(f"{b['label']}  (N={int(df['n_tracks'].iloc[0])})", fontsize=16)
        ax.set_xscale('log')
        ax.tick_params(labelsize=14)
    for ax in axes.flat[nb:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Lag time Δt [s]', fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\alpha_2(\Delta t)$', fontsize=16)
    axes.flat[0].legend(fontsize=11, frameon=False, loc='upper left')
    fig.suptitle(f'NGP decomposition: pooled vs per-trajectory ({component})', fontsize=18)
    fig.tight_layout()
    p = out_dir / f'NGP_per_trajectory_panels_{component}.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    return p


def plot_size_summary(results, out_dir, component, frame_interval, tau_list):
    sizes = [b for b in BEADS_INFO if b['name'] in results]
    fig, axes = plt.subplots(1, len(tau_list), figsize=(6.2 * len(tau_list), 5.0), squeeze=False)
    keys = [('pooled', 'pooled', '#332288', 'o'),
            ('hetero', 'CV²(s_i)', '#999933', 's'),
            ('per_traj_excess', 'per-traj − Brownian null', '#CC6677', '^')]
    for ax, tau in zip(axes[0], tau_list):
        d = np.array([b['diameter_um'] for b in sizes])
        for j, (k, lab, c, m) in enumerate(keys):
            y, lo, hi = [], [], []
            for b in sizes:
                df = results[b['name']]
                r = df[df['tau'] == tau]
                if r.empty:
                    y.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                r = r.iloc[0]
                if k == 'per_traj_excess':
                    y.append(r['per_traj_excess'])
                    lo.append(r['per_traj_lo'] - r['null_per_traj'])
                    hi.append(r['per_traj_hi'] - r['null_per_traj'])
                else:
                    y.append(r[k]); lo.append(r[f'{k}_lo']); hi.append(r[f'{k}_hi'])
            y, lo, hi = map(np.array, (y, lo, hi))
            xs = d * (1 + 0.06 * (j - 1))
            ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt=m + '-', color=c, ms=8, lw=1.5,
                        capsize=3, label=lab)
        ax.axhline(0, color='0.7', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Bead diameter [μm]', fontsize=16)
        ax.set_title(f'Δt = {tau * frame_interval:.0f} s', fontsize=16)
        ax.tick_params(labelsize=14)
    axes[0, 0].set_ylabel(r'$\alpha_2$', fontsize=16)
    axes[0, 0].legend(fontsize=12, frameon=False)
    fig.tight_layout()
    p = out_dir / f'NGP_per_trajectory_vs_size_{component}.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root_dir', default=None)
    ap.add_argument('--out_dir', default=str(ROOT / 'figure' / 'ngp'))
    ap.add_argument('--component', default='2d', choices=['2d', 'x', 'y', 'parallel', 'perpendicular'])
    ap.add_argument('--max_tau', type=int, default=50)
    ap.add_argument('--min_n', type=int, default=10, help='軌道ごとに必要な最小変位サンプル数')
    ap.add_argument('--n_boot', type=int, default=500)
    ap.add_argument('--n_null', type=int, default=20)
    ap.add_argument('--frame_interval', type=float, default=4.0)
    ap.add_argument('--scale', type=float, default=0.11)
    a = ap.parse_args()

    root = Path(a.root_dir) if a.root_dir else find_default_root()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    taus = np.unique(np.round(np.logspace(0, np.log10(a.max_tau), 18)).astype(int))

    results = {}
    for b in BEADS_INFO:
        d = root / b['name']
        if not d.exists():
            continue
        tracks = load_dense_tracks(d, a.scale, a.component)
        if not tracks:
            continue
        lens = [np.isfinite(t['x']).sum() for t in tracks]
        print(f"{b['label']}: {len(tracks)} tracks, length median={np.median(lens):.0f} "
              f"[{np.min(lens)}, {np.max(lens)}] frames", flush=True)
        df = analyze_condition(tracks, taus, a.component, a.min_n, a.n_boot, a.n_null)
        df.insert(0, 'bead', b['name'])
        df.insert(2, 'lag_time', df['tau'] * a.frame_interval)
        results[b['name']] = df

    allres = pd.concat(results.values(), ignore_index=True)
    csv = out / f'NGP_per_trajectory_{a.component}.csv'
    allres.to_csv(csv, index=False)
    p1 = plot_panels(results, out, a.component, a.frame_interval)
    tl = [t for t in (1, 5, 25) if t in set(taus)]
    p2 = plot_size_summary(results, out, a.component, a.frame_interval, tl)
    print('saved:', csv, p1, p2)
    cols = ['bead', 'lag_time', 'n_tracks', 'median_n_per_track', 'pooled', 'hetero',
            'per_traj', 'per_traj_lo', 'per_traj_hi', 'null_per_traj', 'per_traj_excess']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(allres[allres['tau'].isin(tl)][cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
