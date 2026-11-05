#!/usr/bin/env python3
"""
plot_velocity_decomposition.py

恒等式分解に基づく速度伝達効率・幾何平均化係数の解析スクリプト:
  <v_parallel> / <|u|> = [ <v_parallel> / <|<u_Rc>|> ] x [ <|<u_Rc>|> / <|u|> ]
                       = eta (伝達効率: 平均化以外) x A (幾何平均化による足元の流れの減少)

- v_parallel = v_c . u_hat_fp (貨物速度の足元の流れ方向成分: ノイズで偏らない量)
- <|u|>: ピクセル単位の流れの速さ (velocities_mean.csv より取得)
- フィッティング一切不要 (平均値の比のみ)
- 誤差は粒子単位のブートストラップ (Bootstrap N=2000)
- Rc/xi に対する A と eta (および eta*A) を 1 枚の高品位図に描画
- 20 um は別記号 (オープンマーカー) で明示
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Tuple
from scipy.integrate import quad

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

# ROOT_DIR がローカルにフォールバックした場合に同じ場所へ二重保存しないよう重複除去
OUT_DIRS = list(dict.fromkeys([
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis",
]))

SCALE = 0.11  # um/px
DT = 4.0      # s/frame
V_UNIT = SCALE / DT  # um/s per (px/frame)
N_BOOT = 2000
SEED = 42


def setup_matplotlib():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def save_fig(fig, base_name):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


def calc_disk_polar_order(rc_over_xi):
    """
    円盤領域 (半径 Rc) 内で相関関数 C(r) = exp(-r/xi) を持つ
    2次元等方性ベクトル場を平均したときの局所ポーラーオーダー比 Phi(Rc/xi).
    Phi^2(y) = (2/pi) * int_0^2 x * [2*arccos(x/2) - (x/2)*sqrt(4 - x^2)] * exp(-y*x) dx
    (括弧内は単位円 2 枚の重なり面積。Phi(0) = 1 に正規化される。
     verify_cos_decomposition.calc_disk_polar_order と同一の式)
    """
    def integrand(x, y):
        if x >= 2.0:
            return 0.0
        k = 2.0 * np.arccos(x / 2.0) - 0.5 * x * np.sqrt(4.0 - x**2)
        return (2.0 / np.pi) * x * k * np.exp(-y * x)

    if np.isscalar(rc_over_xi):
        val, _ = quad(integrand, 0.0, 2.0, args=(rc_over_xi,))
        return np.sqrt(max(0.0, val))
    else:
        res = []
        for y in rc_over_xi:
            val, _ = quad(integrand, 0.0, 2.0, args=(y,))
            res.append(np.sqrt(max(0.0, val)))
        return np.array(res)


def load_and_preprocess_data(use_local_scalar=True):
    inst_file = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
    if not inst_file.exists():
        raise FileNotFoundError(f"{inst_file} がありません。先に contrast_analysis.py を実行してキャッシュを生成してください。")
    df_inst = pd.read_pickle(inst_file)
    df_inst['unique_pid'] = df_inst['exp_dir'] + '_' + df_inst['particle'].astype(str)

    local_spd_file = CURRENT_DIR / "scratch" / "particle_local_scalar_speed_summary.pkl"
    has_local = False
    if use_local_scalar and local_spd_file.exists():
        df_p_speeds = pd.read_pickle(local_spd_file)
        df_p_speeds['local_scalar_spd_um'] = df_p_speeds['local_scalar_spd_px'] * V_UNIT
        n_pid_before = df_inst['unique_pid'].nunique()
        df_inst = pd.merge(df_inst, df_p_speeds[['unique_pid', 'local_scalar_spd_um']], on='unique_pid', how='inner')
        n_drop = n_pid_before - df_inst['unique_pid'].nunique()
        if n_drop > 0:
            print(f"[warn] {n_drop}/{n_pid_before} particles lack local scalar speed and were dropped")
        has_local = True
        print(f"Loaded particle-level local scalar speeds from {local_spd_file.name} (N={len(df_p_speeds)})")

    # Load velocities_mean.csv
    vm_list = []
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
        for ed in exp_dirs:
            vm_file = ed / "velocities_mean.csv"
            if vm_file.exists():
                df_vm = pd.read_csv(vm_file)
                df_vm['exp_dir'] = ed.name
                vm_list.append(df_vm[['exp_dir', 'frame', 'mean_velocity']])

    if not vm_list:
        raise FileNotFoundError(f"velocities_mean.csv not found under {ROOT_DIR}")
    df_vm_all = pd.concat(vm_list, ignore_index=True)
    df_merged = pd.merge(df_inst, df_vm_all, on=['exp_dir', 'frame'], how='inner')

    flow_norm_px = np.hypot(df_merged['flow_u'], df_merged['flow_v'])
    valid = flow_norm_px > 0
    df_merged = df_merged[valid].copy()

    u_hat_x = df_merged['flow_u'] / flow_norm_px
    u_hat_y = df_merged['flow_v'] / flow_norm_px

    # Variables in physical units [um/s]
    df_merged['v_parallel'] = (df_merged['cargo_dx'] * u_hat_x + df_merged['cargo_dy'] * u_hat_y) * V_UNIT
    df_merged['u_fp'] = flow_norm_px * V_UNIT
    if has_local:
        df_merged['u_px'] = df_merged['local_scalar_spd_um']
    else:
        df_merged['u_px'] = df_merged['mean_velocity']

    # NaN/inf はブートストラップの粒子和を NaN 汚染するため除外し、件数を報告
    finite = np.isfinite(df_merged[['v_parallel', 'u_fp', 'u_px']].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        print(f"[warn] dropped {int((~finite).sum())} rows with non-finite v_parallel/u_fp/u_px")
        df_merged = df_merged[finite].copy()
    if df_merged.empty:
        raise ValueError("前処理後に有効な測定が残りませんでした (入力キャッシュ/velocities_mean.csv を確認)")

    return df_merged


_QTY_COLS = ['v_parallel', 'u_fp', 'u_px']


def _ratio_stats(num_boot: np.ndarray, den_boot: np.ndarray, point: float, prefix: str) -> Dict:
    # 分母ゼロのリサンプル (極少粒子数で起こり得る) は NaN として除外
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num_boot / den_boot
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {prefix: point, f'{prefix}_sem': np.nan, f'{prefix}_ci95': (np.nan, np.nan)}
    return {
        prefix: point,
        f'{prefix}_sem': float(np.std(r)),
        f'{prefix}_ci95': (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))),
    }


def compute_decomposition_statistics(df_merged: pd.DataFrame, n_boot: int = N_BOOT,
                                     min_particles: int = 5):
    """各ビーズについて A, eta, eta*A を粒子ブートストラップで評価 (ベクトル化版).

    Method 1: 粒子ごとの時間平均 -> 粒子を等重みでリサンプル
    Method 2: クラスターブートストラップ (リサンプルした粒子の全フレームをプール).
              プール平均 = sum(粒子ごとの和) / sum(粒子ごとのフレーム数) なので
              concat 不要で (n_boot, n_p) の整数インデックスのみで計算できる.
    """
    rng = np.random.default_rng(SEED)
    results = []

    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        sub = df_merged[df_merged['bead_name'] == bname]
        if sub.empty:
            print(f"[warn] {bname}: no valid frames, skipped")
            continue

        grp = sub.groupby('unique_pid')[_QTY_COLS]
        p_means = grp.mean().values          # (n_p, 3)
        p_sums = grp.sum().values            # (n_p, 3)
        p_counts = grp.size().values.astype(float)  # (n_p,)
        n_p = len(p_counts)

        idx = rng.integers(0, n_p, size=(n_boot, n_p))

        # Method 1: equal weight per particle
        m1 = p_means.mean(axis=0)
        b1 = p_means[idx].mean(axis=1)       # (n_boot, 3)

        # Method 2: frame-pooled cluster bootstrap
        m2 = sub[_QTY_COLS].mean().values
        b2 = p_sums[idx].sum(axis=1) / p_counts[idx].sum(axis=1)[:, None]

        rec = {
            'bead_name': bname,
            'diameter_um': b_item['diameter_um'],
            'radius_um': b_item['radius_um'],
            'xi_um': float(sub['xi'].mean()),
            'rc_over_xi': float(sub['rc_over_xi'].mean()),
            'n_particles': n_p,
            'n_frames': len(sub),
            'low_n': bool(n_p < min_particles),
        }
        if n_p < min_particles:
            print(f"[warn] {bname}: n_particles={n_p} < {min_particles}; "
                  f"bootstrap CI is unreliable (low_n=True)")
        for tag, pt, bt in (('m1', m1, b1), ('m2', m2, b2)):
            v, f, u = pt
            rec.update({f'{tag}_v_par': v, f'{tag}_u_fp': f, f'{tag}_u_px': u})
            rec.update(_ratio_stats(bt[:, 1], bt[:, 2], f / u, f'{tag}_A'))
            rec.update(_ratio_stats(bt[:, 0], bt[:, 1], v / f, f'{tag}_eta'))
            rec.update(_ratio_stats(bt[:, 0], bt[:, 2], v / u, f'{tag}_tot'))
        results.append(rec)

    df_res = pd.DataFrame(results).sort_values('rc_over_xi').reset_index(drop=True)
    return df_res


def _errorbar_pair(ax, df: pd.DataFrame, std_mask, sp_mask, key: str, fmt: str, color: str,
                   label: str, zorder: int, ms: float = 9, lw: float = 2.2, elw: float = 1.6,
                   cap: float = 4.5, mew_open: float = 2.5, alpha: float = 1.0):
    """標準ビーズ (塗り, 線で結ぶ) と大径ビーズ (白抜き) を同じ系列として描画. 95% CI 誤差棒."""
    marker = fmt.rstrip('-:')
    for mask, is_open in ((std_mask, False), (sp_mask, True)):
        if not np.any(mask):
            continue
        d = df.loc[mask]
        y = d[key].values
        ci = np.array(d[f'{key}_ci95'].tolist())
        yerr = np.vstack([y - ci[:, 0], ci[:, 1] - y])
        kw = dict(color=color, ecolor=color, elinewidth=elw, capsize=cap, capthick=elw * 0.85,
                  alpha=alpha)
        if is_open:
            ax.errorbar(d['rc_over_xi'], y, yerr=yerr, fmt=marker, mfc='white', mec=color,
                        mew=mew_open, markersize=ms + 1, zorder=zorder + 1, **kw)
        else:
            ax.errorbar(d['rc_over_xi'], y, yerr=yerr, fmt=fmt, markersize=ms, linewidth=lw,
                        label=label, zorder=zorder, **kw)


def _auto_ylim(df: pd.DataFrame, keys: List[str], default: Tuple[float, float],
               log: bool = False, pad: float = 0.08) -> Tuple[float, float]:
    """既定範囲を基本に, 点推定・95% CI がはみ出す場合のみ範囲を拡張 (無言のクリップ防止).

    log=True の場合は正値のみを対象とし, 乗法的に余白を取る.
    """
    vals = []
    for k in keys:
        vals.append(df[k].values.astype(float))
        ci = np.array(df[f'{k}_ci95'].tolist(), dtype=float).reshape(-1, 2)
        vals.append(ci.ravel())
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    lo, hi = default
    if v.size == 0:
        return lo, hi
    if log:
        lo = min(lo, v.min() / (1 + 3 * pad))
        hi = max(hi, v.max() * (1 + 3 * pad))
    else:
        span = max(hi - lo, 1e-12)
        if v.min() < lo:
            lo = v.min() - pad * span
        if v.max() > hi:
            hi = v.max() + pad * span
    if (lo, hi) != tuple(default):
        print(f"[info] y-range expanded {default} -> ({lo:.3g}, {hi:.3g}) to include all CIs")
    return lo, hi


def _mark_low_n(ax, df: pd.DataFrame, keys: List[str], log: bool = False):
    """low_n (粒子数不足) のビーズに灰色の破線リングと N 注記を重ね, CI が信頼できないことを明示."""
    if 'low_n' not in df or not df['low_n'].any():
        return False
    d = df[df['low_n'].astype(bool)]
    for k in keys:
        y = d[k].values.astype(float)
        ok = np.isfinite(y) & ((y > 0) if log else True)
        ax.scatter(d['rc_over_xi'].values[ok], y[ok], s=260, facecolors='none',
                   edgecolors='#555555', linewidths=1.2, linestyles='--', zorder=9)
    for _, r in d.iterrows():
        ys = [r[k] for k in keys if np.isfinite(r[k]) and (not log or r[k] > 0)]
        if ys:
            ax.annotate(f"N={int(r['n_particles'])}", xy=(r['rc_over_xi'], max(ys)),
                        xytext=(0, 12), textcoords='offset points', ha='center',
                        fontsize=8.5, color='#555555', zorder=10)
    return True


def plot_identity_decomposition(df_res: pd.DataFrame, large_d_um: float = 15.0, method: str = 'm2',
                                mark_low_n: bool = True):
    """method: 'm2' (フレームプール, 既定) または 'm1' (粒子等重み). m1 の図はファイル名に _m1 を付与.
    mark_low_n: low_n (粒子数 < min_particles) の点を破線リング + N 注記で強調."""
    if method not in ('m1', 'm2'):
        raise ValueError(f"method must be 'm1' or 'm2', got {method!r}")
    suffix = '' if method == 'm2' else f'_{method}'
    setup_matplotlib()
    from matplotlib.lines import Line2D

    c_eta = '#D95F02'     # Mechanical coupling efficiency
    c_A   = '#1B9E77'     # Footprint geometric averaging
    c_tot = '#7570B3'     # Overall ratio eta * A

    df_sorted = df_res[np.isfinite(df_res['rc_over_xi']) & (df_res['rc_over_xi'] > 0)]
    if len(df_sorted) < len(df_res):
        print(f"[warn] {len(df_res) - len(df_sorted)} bead(s) with invalid rc_over_xi dropped from plot")
    if df_sorted.empty:
        print('[warn] no valid rc_over_xi; decomposition figures skipped')
        return
    df_sorted = df_sorted.sort_values('rc_over_xi').reset_index(drop=True)
    std_mask = (df_sorted['diameter_um'] < large_d_um).values
    sp_mask = ~std_mask

    # x 範囲・理論曲線はデータから決定
    rc_all = df_sorted['rc_over_xi'].values
    x_lo, x_hi = rc_all.min() / 1.8, rc_all.max() * 1.8
    y_fine = np.logspace(np.log10(x_lo), np.log10(x_hi), 200)
    phi_fine = calc_disk_polar_order(y_fine)
    keys_all = [f'{method}_{k}' for k in ('A', 'eta', 'tot')]
    ylim_lin = _auto_ylim(df_sorted, keys_all, (-0.10, 1.45))
    ylim_log = _auto_ylim(df_sorted, keys_all, (0.015, 1.8), log=True)

    # =========================================================================
    # Figure 9: Master Single-Panel Decomposition Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.plot(y_fine, phi_fine, color=c_A, linestyle='--', linewidth=2.0, alpha=0.8,
            label=r'Theoretical spatial cancellation $\Phi(R_c/\xi)$', zorder=2)
    _errorbar_pair(ax, df_sorted, std_mask, sp_mask, f'{method}_A', 's-', c_A,
                   r'$A = \frac{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}{\langle|\mathbf{u}|\rangle_{R_c}}$ (Footprint Averaging / Polar Order)', 4)
    _errorbar_pair(ax, df_sorted, std_mask, sp_mask, f'{method}_eta', 'o-', c_eta,
                   r'$\eta = \frac{\langle v_\parallel\rangle}{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}$ (Transfer Efficiency: Non-averaging)', 6)
    _errorbar_pair(ax, df_sorted, std_mask, sp_mask, f'{method}_tot', 'd--', c_tot,
                   r'$\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle} = \eta \times A$ (Overall Transport Ratio)', 3,
                   ms=8, lw=1.8, elw=1.4, cap=4.0, alpha=0.85)

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, label='Ideal Perfect Coupling ($y = 1$)')
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)

    # 大径ビーズの注記 (値はデータから算出; 以前はハードコード)
    if sp_mask.any():
        r = df_sorted.loc[sp_mask].iloc[-1]
        ax.annotate(
            f"{r['diameter_um']:g} " + r'$\mu\mathrm{m}$' +
            f"\n($R_c/\\xi={r['rc_over_xi']:.2f}$, $N={int(r['n_particles'])}$)\n" +
            f"$\\eta = {r[f'{method}_eta']:.2f}$",
            xy=(r['rc_over_xi'], r[f'{method}_eta']),
            xytext=(r['rc_over_xi'] * 0.45, 0.22),
            arrowprops=dict(arrowstyle='->', lw=1.3, color='#333333', connectionstyle='arc3,rad=-0.15'),
            fontsize=10.5, fontweight='semibold',
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8EE', ec=c_eta, alpha=0.9, lw=1.2)
        )

    ax.text(
        0.05, 0.92,
        'Exact Identity (No Fitting Required):\n'
        r'$\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$' + '\n\n'
        r'$\bullet\ A \to 1$ at small $R_c/\xi$, follows theory $\Phi$' + '\n'
        r'$\bullet\ \eta \to 0$ as $R_c/\xi > 1$ (Decoupling)',
        transform=ax.transAxes, fontsize=10.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='#F4F6F9', ec='#334155', alpha=0.95, lw=1.2)
    )

    ax.set_xscale('log')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(*ylim_lin)
    ax.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=14, labelpad=8)
    ax.set_ylabel('Velocity Ratio [-]', fontsize=14, labelpad=8)
    ax.set_title(r'Exact Velocity Decomposition: $\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$',
                 fontsize=15, pad=12, fontweight='bold')

    has_low = mark_low_n and _mark_low_n(ax, df_sorted, keys_all)
    handles, labels = ax.get_legend_handles_labels()
    if has_low:
        low_lbl = 'Dashed ring: low N (CI unreliable)'
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                              markeredgecolor='#555555', linestyle='none', markersize=12, label=low_lbl))
        labels.append(low_lbl)
    if sp_mask.any():
        open_lbl = rf'Open: $\geq{large_d_um:g}\,\mu\mathrm{{m}}$ Cargo ($R_c > \xi$)'
        handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                              markeredgewidth=2.0, markersize=9, label=open_lbl))
        labels.append(open_lbl)
    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True,
              framealpha=0.95, facecolor='white', edgecolor='#CCCCCC', fontsize=10.0)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, f"contrast_fig9_velocity_decomposition_identity{suffix}")
    plt.close(fig)

    # =========================================================================
    # Figure 9b: 2-Panel (Linear vs Log-Log)
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    small = dict(ms=8, lw=2.0, elw=1.6, cap=4, mew_open=2.2)
    small_tot = dict(ms=7, lw=1.6, elw=1.4, cap=3.5, mew_open=2.2, alpha=0.85)

    ax1.plot(y_fine, phi_fine, color=c_A, linestyle='--', linewidth=1.8, alpha=0.8,
             label=r'Theory $\Phi(R_c/\xi)$', zorder=2)
    _errorbar_pair(ax1, df_sorted, std_mask, sp_mask, f'{method}_A', 's-', c_A,
                   r'$A = \langle|\langle\mathbf{u}\rangle_{R_c}|\rangle / \langle|\mathbf{u}|\rangle_{R_c}$', 4, **small)
    _errorbar_pair(ax1, df_sorted, std_mask, sp_mask, f'{method}_eta', 'o-', c_eta,
                   r'$\eta = \langle v_\parallel\rangle / \langle|\langle\mathbf{u}\rangle_{R_c}|\rangle$ (Efficiency)', 6, **small)
    _errorbar_pair(ax1, df_sorted, std_mask, sp_mask, f'{method}_tot', 'd--', c_tot,
                   r'$\eta \times A$ (Total Transport Ratio)', 3, **small_tot)
    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax1.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylim(*ylim_lin)
    ax1.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=13)
    ax1.set_ylabel('Velocity Ratio [-]', fontsize=13)
    ax1.set_title('(a) Exact Identity Decomposition (Linear Scale)', fontsize=14, fontweight='bold')
    if mark_low_n:
        _mark_low_n(ax1, df_sorted, keys_all)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Panel 2: log-log — 系列ごとに正値のみ (以前は eta>0 のマスクを全系列に流用していた)
    ax2.plot(y_fine, phi_fine, color=c_A, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
    for key, fmt, c, lbl, z, kw in (
            (f'{method}_A', 's-', c_A, r'Averaging Factor $A$', 4, small),
            (f'{method}_eta', 'o-', c_eta, r'Efficiency $\eta$', 6, small),
            (f'{method}_tot', 'd--', c_tot, r'Total Ratio $\eta \times A$', 3, small_tot)):
        pos = (df_sorted[key] > 0).values
        _errorbar_pair(ax2, df_sorted, std_mask & pos, sp_mask & pos, key, fmt, c, lbl, z, **kw)
    if mark_low_n:
        _mark_low_n(ax2, df_sorted, keys_all, log=True)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(*ylim_log)
    ax2.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=13)
    ax2.set_ylabel('Velocity Ratio [-] (Log Scale)', fontsize=13)
    ax2.set_title('(b) Scaling Comparison (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9.5)
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_fig(fig2, f"contrast_fig9b_velocity_decomposition_2panels{suffix}")
    plt.close(fig2)


COMPARE_BEADS_DEFAULT = ('beads06um', 'beads20um')


def _particle_boot_paired(p_means: pd.DataFrame, cols: List[str], rng: np.random.Generator,
                          n_boot: int = N_BOOT) -> Tuple[np.ndarray, np.ndarray]:
    """全列に同一の粒子リサンプルを用いる (対応ありブートストラップ).

    比 (eta, A) の CI を正しく求めるには分子・分母で同じインデックスが必要.
    戻り値: (点推定 (n_cols,), ブートストラップ平均 (n_boot, n_cols))
    """
    vals = p_means[cols].values
    n = len(vals)
    boots = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return vals.mean(axis=0), boots


def plot_size_comparison(df_merged: pd.DataFrame, bead_names=COMPARE_BEADS_DEFAULT,
                         n_boot: int = N_BOOT):
    """2 サイズ (既定: beads06um vs beads20um) の速度量を比較する 3 パネル図.

    (a) 粒子平均速度 <v_par>, <|<u_Rc>|>, <|u|> の棒グラフ (粒子ブートストラップ 95% CI)
    (b) 瞬時 v_parallel の分布 (密度ヒストグラム)
    (c) 粒子平均 v_parallel vs |<u_Rc>| 散布図 (y=x 参照線)
    """
    setup_matplotlib()
    info = {b['name']: b for b in vcd.BEADS_INFO}
    present = set(df_merged['bead_name'].unique())
    bad = [b for b in bead_names if b not in info or b not in present]
    if bad:
        print(f"[warn] skipped unknown/empty beads: {bad} (available: {sorted(present & set(info))})")
    bead_names = tuple(b for b in bead_names if b in info and b in present)
    if not bead_names:
        print('[warn] no valid beads to compare; size comparison skipped')
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    qty = [('v_parallel', r'$\langle v_\parallel\rangle$'),
           ('u_fp', r'$\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle$'),
           ('u_px', r'$\langle|\mathbf{u}|\rangle$')]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    width = 0.8 / len(bead_names)
    rows = []
    all_vpar = np.concatenate([df_merged.loc[df_merged['bead_name'] == b, 'v_parallel'].values
                               for b in bead_names])
    lo, hi = np.percentile(all_vpar, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax_max = 0.0

    for k, bname in enumerate(bead_names):
        sub = df_merged[df_merged['bead_name'] == bname]
        if sub.empty:
            print(f"[warn] no data for {bname}")
            continue
        b = info[bname]
        p_means = sub.groupby('unique_pid')[['v_parallel', 'u_fp', 'u_px']].mean()
        label_base = f"{b['label']} (N={len(p_means)} particles)"

        # (a) bar chart of means (paired bootstrap: 列間で同じ粒子リサンプル)
        pt, bt = _particle_boot_paired(p_means, [c for c, _ in qty], rng, n_boot)
        lo_q, hi_q = np.percentile(bt, [2.5, 97.5], axis=0)
        stats = list(zip(pt, lo_q, hi_q))
        with np.errstate(divide='ignore', invalid='ignore'):
            eta_pt, a_pt = pt[0] / pt[1], pt[1] / pt[2]
        r_eta = _ratio_stats(bt[:, 0], bt[:, 1], eta_pt, 'eta')
        r_A = _ratio_stats(bt[:, 1], bt[:, 2], a_pt, 'A')
        x = np.arange(len(qty)) + (k - (len(bead_names) - 1) / 2) * width
        m = np.array([s[0] for s in stats])
        err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
        label = (f"{label_base}\n"
                 rf"$\eta$={r_eta['eta']:.2f} [{r_eta['eta_ci95'][0]:.2f}, {r_eta['eta_ci95'][1]:.2f}], "
                 rf"$A$={r_A['A']:.2f} [{r_A['A_ci95'][0]:.2f}, {r_A['A_ci95'][1]:.2f}]")
        axes[0].bar(x, m, width=width, color=b['color'], yerr=err, capsize=4,
                    edgecolor='black', linewidth=0.8, label=label)
        rows.append({'bead_name': bname, 'n_particles': len(p_means), 'n_frames': len(sub),
                     **{f'{c}_mean': s[0] for (c, _), s in zip(qty, stats)},
                     **{f'{c}_ci95': (s[1], s[2]) for (c, _), s in zip(qty, stats)},
                     **r_eta, **r_A})

        # (b) distribution of instantaneous v_parallel
        axes[1].hist(sub['v_parallel'], bins=bins, density=True, histtype='stepfilled',
                     alpha=0.35, color=b['color'])
        axes[1].hist(sub['v_parallel'], bins=bins, density=True, histtype='step',
                     linewidth=1.8, color=b['color'], label=b['label'])
        axes[1].axvline(sub['v_parallel'].mean(), color=b['color'], linestyle='--', linewidth=1.5)

        # (c) particle-level scatter
        axes[2].scatter(p_means['u_fp'], p_means['v_parallel'], s=28, alpha=0.75,
                        color=b['color'], marker=b['marker'], edgecolor='black', linewidth=0.4,
                        label=b['label'])
        ax_max = max(ax_max, p_means['u_fp'].max(), p_means['v_parallel'].max())

    axes[0].set_xticks(np.arange(len(qty)))
    axes[0].set_xticklabels([q[1] for q in qty])
    axes[0].set_ylabel(r'Speed [$\mu$m/s]')
    axes[0].set_title('(a) Mean speeds (particle bootstrap 95% CI)')
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.3)  # 凡例(eta, A 注記)と誤差棒の重なり回避
    axes[0].legend(fontsize=9, loc='upper left')

    axes[1].axvline(0, color='gray', linewidth=0.8)
    axes[1].set_xlabel(r'$v_\parallel$ [$\mu$m/s] (instantaneous)')
    axes[1].set_ylabel('Probability density')
    axes[1].set_title(r'(b) Distribution of $v_\parallel$ (dashed: mean)')
    axes[1].legend(fontsize=9)

    lim = ax_max * 1.05 if ax_max > 0 else 1.0
    axes[2].plot([0, lim], [0, lim], color='gray', linestyle=':', label=r'$y=x$ ($\eta=1$)')
    axes[2].axhline(0, color='gray', linewidth=0.8)
    axes[2].set_xlim(0, lim)
    axes[2].set_xlabel(r'$\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle_t$ per particle [$\mu$m/s]')
    axes[2].set_ylabel(r'$\langle v_\parallel\rangle_t$ per particle [$\mu$m/s]')
    axes[2].set_title('(c) Particle-level cargo vs footprint flow')
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    tag = '_vs_'.join(bead_names)
    save_fig(fig, f"contrast_fig9c_speed_comparison_{tag}")
    plt.close(fig)

    df_cmp = pd.DataFrame(rows)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_cmp.to_csv(d / f"speed_comparison_{tag}.csv", index=False)
    print(f"\n=== Speed comparison: {' vs '.join(bead_names)} [um/s] ===")
    print(df_cmp[['bead_name', 'n_particles', 'n_frames', 'v_parallel_mean', 'u_fp_mean',
                  'u_px_mean', 'eta', 'eta_ci95', 'A', 'A_ci95']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df_cmp


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compare', nargs='+', default=list(COMPARE_BEADS_DEFAULT),
                    help='比較するビーズ名 (既定: beads06um beads20um)')
    ap.add_argument('--n-boot', type=int, default=N_BOOT, help=f'ブートストラップ回数 (既定: {N_BOOT})')
    ap.add_argument('--large-d', type=float, default=15.0,
                    help='白抜き記号で示す大径ビーズの直径閾値 [um] (既定: 15)')
    ap.add_argument('--min-particles', type=int, default=5,
                    help='粒子数がこれ未満のビーズを low_n として警告・表に記録 (既定: 5)')
    ap.add_argument('--no-local-scalar', action='store_true',
                    help='<|u|> に粒子局所スカラー速さではなく velocities_mean.csv の全体平均を使う')
    ap.add_argument('--method', choices=['m1', 'm2'], default='m2',
                    help='図に用いる平均法: m2=フレームプール(既定), m1=粒子等重み (出力名に _m1)')
    ap.add_argument('--no-mark-low-n', action='store_true',
                    help='low_n ビーズの破線リング・N 注記を描かない')
    ap.add_argument('--compare-only', action='store_true',
                    help='サイズ比較図のみ出力 (全サイズの分解統計を省略)')
    args = ap.parse_args()
    if args.n_boot < 1:
        ap.error('--n-boot must be >= 1')
    if args.min_particles < 1:
        ap.error('--min-particles must be >= 1')

    print("Loading data...")
    df_merged = load_and_preprocess_data(use_local_scalar=not args.no_local_scalar)

    if args.compare_only:
        plot_size_comparison(df_merged, tuple(args.compare), n_boot=args.n_boot)
        print("All tasks completed successfully!")
        return
    print(f"Loaded {len(df_merged)} valid instantaneous measurements across "
          f"{df_merged['exp_dir'].nunique()} experiments.")

    print(f"Computing decomposition statistics & particle bootstrap (N={args.n_boot})...")
    df_res = compute_decomposition_statistics(df_merged, n_boot=args.n_boot,
                                              min_particles=args.min_particles)

    # Save summary tables
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(d / "velocity_decomposition_identity_summary.csv", index=False)
        df_res.to_pickle(d / "velocity_decomposition_identity_summary.pkl")
    (CURRENT_DIR / "scratch").mkdir(exist_ok=True)
    (CURRENT_DIR / "scratch" / "velocity_decomposition_identity_summary.pkl").write_bytes(
        (OUT_DIRS[0] / "velocity_decomposition_identity_summary.pkl").read_bytes()
    )

    print(f"\n=== Velocity Decomposition Summary (method={args.method}) ===")
    cols_show = ['diameter_um', 'rc_over_xi', 'n_particles', 'low_n'] + [f'{args.method}_{k}' for k in ('A', 'eta', 'tot')]
    print(df_res[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nGenerating figures...")
    plot_identity_decomposition(df_res, large_d_um=args.large_d, method=args.method,
                                mark_low_n=not args.no_mark_low_n)
    plot_size_comparison(df_merged, tuple(args.compare), n_boot=args.n_boot)
    print("All tasks completed successfully!")


if __name__ == '__main__':
    main()

