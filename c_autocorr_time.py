#!/usr/bin/env python3
"""
c_autocorr_time.py

c(t) = cos θ_i（verify_cos_decomposition.py の瞬時値 cos_i、粒子直下窓）の
時間自己相関関数 C(τ) と自己相関時間をビーズサイズごとに求める。

C(τ) = < δc(t) δc(t+τ) > / < δc² >   （軌道内で frame 差がちょうど τ のペアのみ使う）
  - "ens"  : δc = c − <c>_size（サイズ全体の平均を引く。軌道ごとの平均の違い＝静的不均一も相関に残る）
  - "traj" : δc = c − <c>_traj（軌道ごとの平均を引く。軌道内のゆらぎだけ。短い軌道では負のバイアス）

相関時間（Δt = 4 s/frame）:
  τ_1/e : C が初めて 1/e を下回る時刻（線形補間）
  τ_int : C(0)/2 + Σ_{τ≥1} C(τ) を最初の C ≤ 0 の手前まで足して Δt を掛けたもの
信頼区間は軌道（cargo_id）単位のクラスタブートストラップ（既定 500 回）。

出力: figure/c_autocorr/
キャッシュ: figure/c_autocorr/c_instant_cache.pkl（--rebuild で再作成）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR))
import verify_cos_decomposition as vcd  # noqa: E402

OUT = CUR / "figure" / "c_autocorr"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "c_instant_cache.pkl"
THETA_CACHE = CUR / "figure" / "theta_fit_ci" / "theta_instant_cache.pkl"
DT = 4.0          # s / frame
MAX_LAG = 60      # frames (240 s)
MIN_PAIRS = 30


def load(rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_pickle(CACHE)
    active = vcd.load_active_particles(vcd.ROOT_DIR)
    df_i, _ = vcd.load_all_data(vcd.ROOT_DIR, active)
    df_i['cargo_id'] = df_i['exp_dir'].astype(str) + '_' + df_i['particle'].astype(str)
    df = df_i[['bead_name', 'cargo_id', 'frame', 'cos_i', 'cos_bulk']].copy()
    df = df.drop_duplicates(['cargo_id', 'frame']).sort_values(['cargo_id', 'frame'])
    df.to_pickle(CACHE)
    return df


def per_traj_sums(groups, col, mean_mode, mu_size):
    """各軌道について lag ごとの Σδcδc とペア数を返す（ブートストラップで再利用）。"""
    S = np.zeros((len(groups), MAX_LAG + 1))
    N = np.zeros((len(groups), MAX_LAG + 1))
    for k, g in enumerate(groups):
        f = g['frame'].values.astype(int)
        x = g[col].values.astype(float)
        mu = x.mean() if mean_mode == 'traj' else mu_size
        grid = np.full(f.max() - f.min() + 1, np.nan)
        grid[f - f.min()] = x - mu
        ok = np.isfinite(grid)
        g0 = np.where(ok, grid, 0.0)
        L = len(grid)
        for lag in range(min(MAX_LAG, L - 1) + 1):
            m = ok[:L - lag] & ok[lag:]
            N[k, lag] = m.sum()
            S[k, lag] = (g0[:L - lag] * g0[lag:])[m].sum()
    return S, N


def acf_from(S, N):
    s, n = S.sum(0), N.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = s / n
    C[n < MIN_PAIRS] = np.nan
    return C / C[0], n


def times(C):
    """τ_1/e と τ_int（秒）"""
    lag = np.arange(len(C))
    t1e = np.nan
    below = np.where(C < np.exp(-1))[0]
    if len(below):
        j = below[0]
        c0, c1 = C[j - 1], C[j]
        t1e = (j - 1 + (c0 - np.exp(-1)) / (c0 - c1)) * DT
    neg = np.where(~(C > 0))[0]
    jend = neg[0] if len(neg) else len(C)
    tint = (0.5 * C[0] + np.nansum(C[1:jend])) * DT
    if len(neg) == 0:
        tint = np.nan  # 打ち切り内で 0 に届かない → 定まらない（下限のみ）
    return t1e, tint, (jend if len(neg) else None)


def slow_fit(C):
    """τ ≥ Δt の部分に C = a exp(−τ/τ_s) + b をフィット。返り値 (a, τ_s[s], b)。
    τ = 0 → Δt の落ち（1 − C(Δt)）は Δt より速い成分（分解できない）とみなす。"""
    lag = np.arange(len(C)) * DT
    m = (lag >= DT) & np.isfinite(C)
    if m.sum() < 5:
        return np.nan, np.nan, np.nan
    try:
        p, _ = curve_fit(lambda t, a, ts, b: a * np.exp(-t / ts) + b, lag[m], C[m],
                         p0=(0.15, 20.0, 0.02), bounds=([0, DT / 4, -0.2], [1.0, 2000.0, 0.5]), maxfev=8000)
        return tuple(float(v) for v in p)
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    df = load(a.rebuild)
    rc = {}
    if THETA_CACHE.exists():
        t = pd.read_pickle(THETA_CACHE)
        rc = t.groupby('bead_name')['rc_over_xi_bead'].first().to_dict()
    rng = np.random.default_rng(a.seed)

    rows, curves = [], []
    for b in vcd.BEADS_INFO:
        sub = df[df['bead_name'] == b['name']].dropna(subset=['cos_i'])
        if sub.empty:
            continue
        groups = [g for _, g in sub.groupby('cargo_id')]
        lens = np.array([len(g) for g in groups])
        spans = np.array([g['frame'].max() - g['frame'].min() + 1 for g in groups])
        for col in ('cos_i', 'cos_bulk'):
            for mode in ('ens', 'traj'):
                S, N = per_traj_sums(groups, col, mode, float(sub[col].mean()))
                C, n = acf_from(S, N)
                t1e, tint, jzero = times(C)
                fa, fts, fb = slow_fit(C)
                bt1, bti, bC, bfit = [], [], [], []
                for _ in range(a.n_boot):
                    sel = rng.integers(0, len(groups), len(groups))
                    Cb, _ = acf_from(S[sel], N[sel])
                    x, y, _ = times(Cb)
                    bt1.append(x); bti.append(y); bC.append(Cb); bfit.append(slow_fit(Cb))
                bt1, bti, bC, bfit = np.array(bt1), np.array(bti), np.array(bC), np.array(bfit)
                bfast = 1.0 - bC[:, 1]
                q = lambda v, p: float(np.nanpercentile(v, p)) if np.isfinite(v).any() else np.nan
                rows.append(dict(
                    bead_name=b['name'], diameter_um=b['diameter_um'], rc_over_xi=rc.get(b['name'], np.nan),
                    signal=col, mean_sub=mode, n_cargo=len(groups), n_points=len(sub),
                    traj_len_median_frames=float(np.median(lens)), traj_span_median_s=float(np.median(spans)) * DT,
                    c_mean=float(sub[col].mean()), c_var=float(sub[col].var()),
                    C_lag1=float(C[1]), C_lag5=float(C[5]) if len(C) > 5 else np.nan,
                    tau_1e_s=t1e, tau_1e_lo=q(bt1, 2.5), tau_1e_hi=q(bt1, 97.5),
                    tau_int_s=tint, tau_int_lo=q(bti, 2.5), tau_int_hi=q(bti, 97.5),
                    first_zero_s=(jzero * DT if jzero is not None else np.nan),
                    tau_1e_resolved=bool(np.isfinite(t1e) and t1e > DT),
                    frac_fast=float(1 - C[1]), frac_fast_lo=q(bfast, 2.5), frac_fast_hi=q(bfast, 97.5),
                    slow_amp=fa, slow_amp_lo=q(bfit[:, 0], 2.5), slow_amp_hi=q(bfit[:, 0], 97.5),
                    tau_slow_s=fts, tau_slow_lo=q(bfit[:, 1], 2.5), tau_slow_hi=q(bfit[:, 1], 97.5),
                    plateau=fb, plateau_lo=q(bfit[:, 2], 2.5), plateau_hi=q(bfit[:, 2], 97.5)))
                curves.append(pd.DataFrame(dict(
                    bead_name=b['name'], signal=col, mean_sub=mode, lag_s=np.arange(len(C)) * DT,
                    C=C, C_lo=np.nanpercentile(bC, 2.5, axis=0), C_hi=np.nanpercentile(bC, 97.5, axis=0),
                    n_pairs=n)))
                r = rows[-1]
                print(f"{b['name']:10s} {col:8s} {mode:4s} N={len(groups):3d} C(4s)={C[1]:.3f} "
                      f"τ1/e={t1e:6.1f}[{r['tau_1e_lo']:.1f},{r['tau_1e_hi']:.1f}] s "
                      f"τint={tint:6.1f}[{r['tau_int_lo']:.1f},{r['tau_int_hi']:.1f}] s | fast={r['frac_fast']:.2f} "
                      f"a={fa:.3f}[{r['slow_amp_lo']:.3f},{r['slow_amp_hi']:.3f}] "
                      f"τs={fts:.0f}[{r['tau_slow_lo']:.0f},{r['tau_slow_hi']:.0f}] s "
                      f"b={fb:.3f}[{r['plateau_lo']:.3f},{r['plateau_hi']:.3f}]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "c_autocorr_time_by_size.csv", index=False)
    cur = pd.concat(curves)
    cur.to_csv(OUT / "c_autocorr_curves.csv", index=False)
    plot(res, cur)


def plot(res, cur):
    info = {b['name']: b for b in vcd.BEADS_INFO}
    for col in ('cos_i', 'cos_bulk'):
        fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
        for mode, ax in zip(('ens', 'traj'), axs[0, :2]):
            r = res[(res.signal == col) & (res.mean_sub == mode)].set_index('bead_name')
            for bn, g in cur[(cur.signal == col) & (cur.mean_sub == mode)].groupby('bead_name', sort=False):
                b = info[bn]
                ax.plot(g.lag_s, g.C, color=b['color'], marker=b['marker'], ms=3.5, lw=1.3,
                        label=f"{b['diameter_um']:.2f} µm (N={r.loc[bn, 'n_cargo']})")
                ax.fill_between(g.lag_s, g.C_lo, g.C_hi, color=b['color'], alpha=0.12, lw=0)
                x = r.loc[bn]
                if np.isfinite(x.tau_slow_s):
                    t = np.linspace(DT, g.lag_s.max(), 200)
                    ax.plot(t, x.slow_amp * np.exp(-t / x.tau_slow_s) + x.plateau, color=b['color'], ls='--', lw=1)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xlim(0, MAX_LAG * DT); ax.set_ylim(-0.2, 0.45)
            ax.set_xlabel(r"lag $\tau$ [s]"); ax.set_ylabel(r"$C(\tau)$  (C(0)=1, off-scale)"); ax.grid(alpha=0.3)
            ax.set_title({'ens': "(A) δc = c − ⟨c⟩_size   (dashed: a e^{−τ/τ_s}+b, τ≥4 s)",
                          'traj': "(B) δc = c − ⟨c⟩_traj"}[mode], fontsize=10)
        axs[0, 0].legend(fontsize=8)
        # (C) 短ラグ拡大
        ax = axs[0, 2]
        for bn, g in cur[(cur.signal == col) & (cur.mean_sub == 'ens')].groupby('bead_name', sort=False):
            b = info[bn]; g = g[g.lag_s <= 40]
            ax.plot(g.lag_s, g.C, color=b['color'], marker=b['marker'], ms=5, lw=1.3)
        ax.axhline(np.exp(-1), color='gray', ls=':', lw=1); ax.text(30, np.exp(-1) + 0.02, '1/e', color='gray')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel(r"lag $\tau$ [s]"); ax.set_ylabel(r"$C(\tau)$"); ax.grid(alpha=0.3)
        ax.set_title("(C) short lags (ens): C drops below 1/e within Δt = 4 s", fontsize=10)
        # (D)(E)(F) パラメータ vs R_c/ξ
        specs = [('frac_fast', r"fast fraction $1-C(\Delta t)$", "(D) variance decorrelating within Δt", False),
                 ('tau_slow_s', r"$\tau_s$ [s]", "(E) slow time τ_s", True),
                 ('slow_amp', r"slow amplitude $a$ / plateau $b$", "(F) a (filled), b (open), ens", False)]
        for (key, yl, tt, logy), ax in zip(specs, axs[1]):
            for mode, off in (('ens', 0.95), ('traj', 1.05)):
                if key == 'slow_amp' and mode == 'traj':
                    continue
                r = res[(res.signal == col) & (res.mean_sub == mode)]
                for _, x in r.iterrows():
                    b = info[x.bead_name]
                    base = 'tau_slow' if key == 'tau_slow_s' else key
                    lo, hi = x[base + '_lo'], x[base + '_hi']
                    ax.errorbar(x.rc_over_xi * off, x[key], yerr=[[max(x[key] - lo, 0)], [max(hi - x[key], 0)]],
                                color=b['color'], marker=b['marker'], ms=8, capsize=3, mec='k', mew=0.6,
                                mfc=b['color'] if mode == 'ens' else 'white', ls='none')
                    if key == 'slow_amp':
                        ax.errorbar(x.rc_over_xi * 1.05, x.plateau,
                                    yerr=[[max(x.plateau - x.plateau_lo, 0)], [max(x.plateau_hi - x.plateau, 0)]],
                                    color=b['color'], marker=b['marker'], ms=8, capsize=3, mec=b['color'],
                                    mfc='white', ls='none')
            ax.set_xscale('log')
            if logy:
                ax.set_yscale('log'); ax.axhline(DT, color='gray', ls='--', lw=1)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel(r"$R_c/\xi$"); ax.set_ylabel(yl); ax.set_title(tt, fontsize=10); ax.grid(alpha=0.3, which='both')
        axs[1, 0].plot([], [], 'ko', label='ens'); axs[1, 0].plot([], [], 'ko', mfc='white', label='traj')
        axs[1, 0].legend(fontsize=8)
        fig.suptitle(f"Autocorrelation of c(t) = {'cos θ_i' if col == 'cos_i' else 'cos Θ_bulk'}"
                     "  [95% cluster bootstrap CI, Δt = 4 s]", y=1.0)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(OUT / f"c_autocorr_{col}.{ext}", dpi=200, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
