#!/usr/bin/env python3
"""
contrast_analysis.py

アクティブ微小管流動場における貨物微粒子の「必須対照解析 (Essential Contrast Analysis)」スクリプト:
1. 正規化結合度 cos(psi_i) = v_hat . <u_fp> / |<u_fp>| の解析 (Rc/xi 依存性)
2. 下限側基準 (Null Models): 時間シャッフル & 粒子軌道スワップ
3. 上限側基準 (Virtual Cargo): 実測足印平均流速での移流モデル
   - MSD の直接比較 (6パネル & 比率)
   - 変位 PDF の直接比較 (非ガウス性・尖度)
   - 速度伝達係数 kappa = vc / <u_fp>
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit

# ワークスペースおよびNASのパス設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR_CANDIDATES = [
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    CURRENT_DIR,
]
ROOT_DIR = next((p for p in ROOT_DIR_CANDIDATES if p.exists()), CURRENT_DIR)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis"
]

BEADS_INFO = [
    {'name': 'beads06um', 'label': r'0.63 $\mu\mathrm{m}$', 'diameter_um': 0.63, 'radius_um': 0.315,
     'color': '#882255', 'marker': '^'},
    {'name': 'beads1um',  'label': r'1.18 $\mu\mathrm{m}$', 'diameter_um': 1.18, 'radius_um': 0.590,
     'color': '#CC6677', 'marker': 'o'},
    {'name': 'beads3um',  'label': r'3.37 $\mu\mathrm{m}$', 'diameter_um': 3.37, 'radius_um': 1.685,
     'color': '#DDCC77', 'marker': 'd'},
    {'name': 'beads7um',  'label': r'7.24 $\mu\mathrm{m}$', 'diameter_um': 7.24, 'radius_um': 3.620,
     'color': '#117733', 'marker': 'h'},
    {'name': 'beads5um',  'label': r'5.00 $\mu\mathrm{m}$', 'diameter_um': 5.00, 'radius_um': 2.500,
     'color': '#999933', 'marker': 'p'},
    {'name': 'beads20um', 'label': r'20.0 $\mu\mathrm{m}$', 'diameter_um': 20.0, 'radius_um': 10.000,
     'color': '#44AA99', 'marker': 's'},
]
# パネル配置・Rc/xi 方向の折れ線が粒径順になるよう常に直径で昇順ソート
BEADS_INFO.sort(key=lambda b: b['diameter_um'])

SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0


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
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def save_figure(fig, base_name: str, out_dirs=OUT_DIRS):
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


# キャッシュファイル (scratch/compute_contrast_data.py が生成, 本スクリプトが読み込み)
CACHE_DIR = CURRENT_DIR / "scratch"
CACHE_PATHS = {
    'inst': CACHE_DIR / "contrast_inst_cache.pkl",
    'imsd': CACHE_DIR / "contrast_imsd_cache.pkl",
    'disp': CACHE_DIR / "contrast_disp_cache.npy",
}


def _atomic_write(final_path: Path, writer):
    """一時ファイルに書いてから置換する (中断時に壊れたキャッシュを残さない)。"""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.stem + ".tmp" + final_path.suffix)
    try:
        writer(tmp_path)
        tmp_path.replace(final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _portable_df(df: pd.DataFrame) -> pd.DataFrame:
    """pandas 3 の StringDtype 列は pandas 2 で unpickle できないため object 列に戻す。"""
    df = df.copy()
    df.columns = pd.Index(list(df.columns), dtype=object)
    if isinstance(df.index.dtype, pd.StringDtype):
        df.index = df.index.astype(object)
    str_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)]
    return df.astype({c: object for c in str_cols}) if str_cols else df


def save_cached_data(df_inst: pd.DataFrame, df_imsd: pd.DataFrame, disp_data: dict):
    """3種のキャッシュを CACHE_DIR に保存する (実行ディレクトリに依存しない)。"""
    df_inst, df_imsd = _portable_df(df_inst), _portable_df(df_imsd)
    _atomic_write(CACHE_PATHS['inst'], lambda p: df_inst.to_pickle(p))
    _atomic_write(CACHE_PATHS['imsd'], lambda p: df_imsd.to_pickle(p))
    # np.save は拡張子 .npy を自動付与するため、ファイルハンドル経由で書く
    def _save_npy(p):
        with open(p, 'wb') as f:
            np.save(f, np.array(disp_data, dtype=object), allow_pickle=True)
    _atomic_write(CACHE_PATHS['disp'], _save_npy)
    print(f"Saved caches to {CACHE_DIR}")


def load_cached_data():
    missing = [str(p) for p in CACHE_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cache files not found:\n  " + "\n  ".join(missing)
            + "\nRun `python3 scratch/compute_contrast_data.py` first.")

    try:
        df_inst = pd.read_pickle(CACHE_PATHS['inst'])
        df_imsd = pd.read_pickle(CACHE_PATHS['imsd'])
        disp_data = np.load(CACHE_PATHS['disp'], allow_pickle=True).item()
    except Exception as e:
        raise RuntimeError(
            f"Failed to read contrast caches in {CACHE_DIR} ({type(e).__name__}: {e}). "
            "Re-run `python3 scratch/compute_contrast_data.py` to regenerate them.") from e
    return df_inst, df_imsd, disp_data


_NULL_COLS = ['cargo_dx', 'cargo_dy', 'cargo_spd', 'flow_u', 'flow_v', 'flow_spd']


def _cos_psi(cdx, cdy, cspd, u, v, fspd):
    """cos(psi) = (v_c . u) / (|v_c||u|)。ゼロ速度による inf/NaN は呼び出し側で除外する。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return (cdx * u + cdy * v) / (cspd * fspd)


def _finite_mean(chunks):
    if not chunks:
        return np.nan
    arr = np.concatenate(chunks)
    arr = arr[np.isfinite(arr)]
    return arr.mean() if arr.size else np.nan


def _summarize(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return vals.mean(), vals.std()


def compute_null_models(df_inst: pd.DataFrame, n_iter: int = 100, seed: int = 42):
    """粒子ごとの時間シャッフル & 同一実験内粒子スワップによる下限ヌルモデルの算出

    DataFrame のグループ分割・列抽出をループ外で1回だけ行い、反復内は numpy 演算のみ。
    乱数呼び出し順は旧実装 (np.random.seed(seed) + np.random.permutation) と同一なので結果は再現する。
    ゼロ速度フレームによる inf/NaN は平均から除外する。
    """
    rng = np.random.RandomState(seed)
    null_results = {}

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname]

        # --- 前処理: 時間シャッフル用 (粒子ごとの列配列; len<2 は除外) ---
        particle_arrays = [
            tuple(grp[c].to_numpy() for c in _NULL_COLS)
            for _, grp in sub.groupby(['exp_dir', 'particle'])
            if len(grp) >= 2
        ]

        # --- 前処理: スワップ用 (実験ごとに pid -> 列配列) ---
        exp_entries = []
        for _, e_grp in sub.groupby('exp_dir'):
            pids = e_grp['particle'].unique()
            whole = tuple(e_grp[c].to_numpy() for c in _NULL_COLS)
            per_pid = {}
            if len(pids) >= 2:
                pcol = e_grp['particle'].to_numpy()
                per_pid = {pid: tuple(a[pcol == pid] for a in whole) for pid in pids}
            exp_entries.append((pids, whole, per_pid))

        # 1. 時間シャッフル
        shuf_means = []
        for _ in range(n_iter):
            chunks = []
            for cdx, cdy, cspd, u, v, fspd in particle_arrays:
                idx = rng.permutation(len(cdx))
                chunks.append(_cos_psi(cdx[idx], cdy[idx], cspd[idx], u, v, fspd))
            if chunks:
                shuf_means.append(_finite_mean(chunks))

        # 2. 粒子スワップ
        swap_means = []
        for _ in range(n_iter):
            chunks = []
            for pids, whole, per_pid in exp_entries:
                if len(pids) < 2:
                    cdx, cdy, cspd, u, v, fspd = whole
                    idx = rng.permutation(len(cdx))
                    chunks.append(_cos_psi(cdx[idx], cdy[idx], cspd[idx], u, v, fspd))
                    continue
                p_map = dict(zip(pids, rng.permutation(pids)))
                for pid in pids:
                    a = per_pid[pid]
                    b = per_pid[p_map[pid]]
                    n = min(len(a[0]), len(b[0]))
                    if n < 2:
                        continue
                    chunks.append(_cos_psi(b[0][:n], b[1][:n], b[2][:n],
                                           a[3][:n], a[4][:n], a[5][:n]))
            if chunks:
                swap_means.append(_finite_mean(chunks))

        t_mean, t_std = _summarize(shuf_means)
        s_mean, s_std = _summarize(swap_means)
        null_results[bname] = {
            'time_shuf_mean': t_mean,
            'time_shuf_std': t_std,
            'swap_mean': s_mean,
            'swap_std': s_std,
        }

    return null_results


def plot_fig1_coupling_spectrum(df_inst: pd.DataFrame, null_results: dict):
    """
    Figure 1: 結合度スペクトル解析
    - (a) 正規化結合度 <cos psi> vs Rc/xi (上限仮想貨物・下限ヌルモデル併記)
    - (b) 従来点平均一致度 <cos theta_i> vs Rc/xi
    - (c) cos(psi) の確率密度分布 P(cos psi) (サイズ別 KDE)
    - (d) シグナル有意性 (Z-score) vs Rc/xi
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    rc_xi_list = []
    cos_psi_means = []
    cos_psi_sems = []
    cos_i_means = []
    cos_i_sems = []
    null_time_means = []
    null_time_stds = []
    null_swap_means = []
    z_scores = []
    labels = []
    colors = []
    markers = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname]
        rc_xi = sub['rc_over_xi'].mean()
        cb = sub['cos_bulk'].values
        ci = sub['cos_i'].values
        
        rc_xi_list.append(rc_xi)
        cos_psi_means.append(np.mean(cb))
        cos_psi_sems.append(stats.sem(cb))
        cos_i_means.append(np.mean(ci))
        cos_i_sems.append(stats.sem(ci))
        
        nr = null_results[bname]
        null_time_means.append(nr['time_shuf_mean'])
        null_time_stds.append(nr['time_shuf_std'])
        null_swap_means.append(nr['swap_mean'])
        
        z = (np.mean(cb) - nr['time_shuf_mean']) / stats.sem(cb)
        z_scores.append(z)
        
        labels.append(b_item['label'])
        colors.append(b_item['color'])
        markers.append(b_item['marker'])

    # ソート (Rc/xi 順)
    sort_idx = np.argsort(rc_xi_list)
    rc_xi_arr = np.array(rc_xi_list)[sort_idx]
    cos_psi_arr = np.array(cos_psi_means)[sort_idx]
    cos_psi_sem_arr = np.array(cos_psi_sems)[sort_idx]
    cos_i_arr = np.array(cos_i_means)[sort_idx]
    cos_i_sem_arr = np.array(cos_i_sems)[sort_idx]
    null_t_arr = np.array(null_time_means)[sort_idx]
    null_t_std_arr = np.array(null_time_stds)[sort_idx]
    null_s_arr = np.array(null_swap_means)[sort_idx]
    z_arr = np.array(z_scores)[sort_idx]

    # -------------------------------------------------------------
    # Panel (a): 正規化結合度 <cos psi> vs Rc/xi
    # -------------------------------------------------------------
    # 上限: 仮想貨物 理論線
    ax_a.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=2.0, label='Upper Bound: Virtual Cargo ($\\cos\\psi \\equiv 1$)')
    # 下限: ヌルモデル帯
    ax_a.fill_between([0.02, 5.0], 0, 0.05, color='gray', alpha=0.18, label='Lower Bound: Global Drift / Noise Baseline')
    ax_a.plot(rc_xi_arr, null_t_arr, color='#d7191c', linestyle=':', marker='x', markersize=7, label='Null Model (Time Shuffle)')
    ax_a.plot(rc_xi_arr, null_s_arr, color='#fdae61', linestyle='-.', marker='+', markersize=7, label='Null Model (Trajectory Swap)')

    # 実測値プロット
    for i in sort_idx:
        ax_a.errorbar(rc_xi_list[i], cos_psi_means[i], yerr=cos_psi_sems[i],
                      fmt=markers[i], color=colors[i], markersize=10, capsize=5, elinewidth=1.8,
                      label=f"{labels[i]} ($R_c/\\xi={rc_xi_list[i]:.2f}$)", zorder=5)
    ax_a.plot(rc_xi_arr, cos_psi_arr, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=4)

    # 臨界境界 Rc = xi
    ax_a.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_a.text(1.05, 0.85, r'$R_c = \xi$', fontsize=13, color='gray', fontweight='bold')

    ax_a.set_xscale('log')
    ax_a.set_xlim(0.04, 4.0)
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_a.set_ylabel(r'Normalized orientation coupling $\langle\cos\psi\rangle$')
    ax_a.set_title(r'(a) Normalized Coupling $\hat{v} \cdot \frac{\langle u\rangle_{R_c}}{|\langle u\rangle_{R_c}|}$ vs $R_c/\xi$', fontweight='bold')
    # 凡例を「基準線 (上限/下限/Null)」と「実測データ」に分割し、データの無い上部領域に配置
    # (データ点は <cos psi> < 0.4 に集中するため、下部配置だと 0.63 µm 点や Null 線と重なる)
    h_a, l_a = ax_a.get_legend_handles_labels()
    is_ref = [('Bound' in l) or ('Null' in l) for l in l_a]
    ref_h = [h for h, r in zip(h_a, is_ref) if r]
    ref_l = [l for l, r in zip(l_a, is_ref) if r]
    dat_h = [h for h, r in zip(h_a, is_ref) if not r]
    dat_l = [l for l, r in zip(l_a, is_ref) if not r]
    leg_ref = ax_a.legend(ref_h, ref_l, loc='upper left', bbox_to_anchor=(0.01, 0.94),
                          fontsize=9.0, framealpha=0.9)
    ax_a.add_artist(leg_ref)
    ax_a.legend(dat_h, dat_l, loc='upper left', bbox_to_anchor=(0.01, 0.70), ncol=2,
                fontsize=8.5, framealpha=0.9, columnspacing=0.8, handletextpad=0.4)
    ax_a.grid(True)

    # -------------------------------------------------------------
    # Panel (b): 従来の点平均一致度 <cos theta_i> vs Rc/xi
    # -------------------------------------------------------------
    for i in sort_idx:
        ax_b.errorbar(rc_xi_list[i], cos_i_means[i], yerr=cos_i_sems[i],
                      fmt=markers[i], color=colors[i], markersize=10, capsize=5, elinewidth=1.8,
                      label=labels[i], zorder=5)
    ax_b.plot(rc_xi_arr, cos_i_arr, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=4)

    ax_b.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    # 右上は凡例が占有するため、ラベルは臨界線の左側 (データの無い上部領域) に右寄せで配置
    ax_b.text(0.95, 0.40, r'$R_c = \xi$', fontsize=13, color='gray', fontweight='bold', ha='right')

    ax_b.set_xscale('log')
    ax_b.set_xlim(0.04, 4.0)
    ax_b.set_ylim(-0.02, 0.50)
    ax_b.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_b.set_ylabel(r'Raw point-averaged alignment $\langle\cos\theta_i\rangle$')
    ax_b.set_title(r'(b) Point-Averaged Alignment $\langle\hat{v}\cdot\hat{u}\rangle$ vs $R_c/\xi$', fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_b.grid(True)

    # -------------------------------------------------------------
    # Panel (c): cos(psi) の確率密度分布 P(cos psi) (KDE)
    # -------------------------------------------------------------
    x_grid = np.linspace(-1.0, 1.0, 300)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_inst[df_inst['bead_name'] == bname]
        cb = sub['cos_bulk'].values
        kde = stats.gaussian_kde(cb, bw_method=0.15)
        dens = kde(x_grid)
        ax_c.plot(x_grid, dens, color=b_item['color'], linewidth=2.2,
                  label=f"{b_item['label']} ($R_c/\\xi={sub['rc_over_xi'].mean():.2f}$)")

    # 等方分布基準線 (一様分布: 2次元単位ベクトルの内積分布 1 / (pi * sqrt(1 - x^2)))
    # 3次元なら一様分布 0.5
    ax_c.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Uniform isotropic reference')
    ax_c.set_xlim(-1.0, 1.0)
    ax_c.set_ylim(0, 1.5)
    ax_c.set_xlabel(r'Cosine of coupling angle $\cos\psi = \hat{v} \cdot \hat{u}_{\mathrm{fp}}$')
    ax_c.set_ylabel(r'Probability density $P(\cos\psi)$')
    ax_c.set_title(r'(c) Coupling Angle Distribution $P(\cos\psi)$', fontweight='bold')
    ax_c.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax_c.grid(True)

    # -------------------------------------------------------------
    # Panel (d): シグナル有意性 (Z-score) vs Rc/xi
    # -------------------------------------------------------------
    for i in sort_idx:
        ax_d.scatter(rc_xi_list[i], z_scores[i], color=colors[i], marker=markers[i], s=120, zorder=5, label=labels[i])
    ax_d.plot(rc_xi_arr, z_arr, color='#882255', linewidth=2.0, linestyle='-', alpha=0.7, zorder=4)

    # 有意水準線 (Z = 3: p = 0.0013, Z = 2: p = 0.05)
    ax_d.axhline(3.0, color='#cc2222', linestyle='--', linewidth=1.5, label=r'Significance threshold ($Z=3, p < 0.001$)')
    ax_d.axhline(0.0, color='gray', linestyle='-', linewidth=0.8)
    ax_d.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    # アノテーション: 各点の Z と両側 p 値をデータから算出 (旧版のハードコード値を廃止)
    # 隣接点 (Rc/xi 昇順) でラベルを上下交互に配置し、隣のマーカーとの重なりを防ぐ
    p_arr = 2.0 * stats.norm.sf(np.abs(z_arr))
    k_lab = 0
    for x, z, pv in zip(rc_xi_arr, z_arr, p_arr):
        if not np.isfinite(z):
            continue
        above = (k_lab % 2 == 0)
        k_lab += 1
        if pv < 1e-3:
            mant, expo = f"{pv:.1e}".split('e')
            p_txt = rf"$p={mant}\times10^{{{int(expo)}}}$"
        else:
            p_txt = f"$p={pv:.2f}$"
        ax_d.annotate(f"$Z={z:.1f}$\n{p_txt}",
                      (x, z), textcoords='offset points',
                      xytext=(-4, 14) if above else (-4, -16),
                      ha='center', va='bottom' if above else 'top', fontsize=8.5, zorder=6,
                      arrowprops=dict(arrowstyle='-', color='gray', linewidth=0.6, shrinkA=0, shrinkB=6),
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6f5d0' if z > 3 else '#fde0dd',
                                edgecolor='none', alpha=0.85))

    ax_d.set_xscale('log')
    ax_d.set_xlim(0.04, 4.0)
    z_fin = z_arr[np.isfinite(z_arr)]
    z_top = float(z_fin.max()) if z_fin.size else 5.0
    ax_d.set_ylim(min(-2.0, float(z_fin.min()) - 0.12 * z_top - 1.0) if z_fin.size else -2.0, max(5.0, z_top * 1.25))  # 下向きラベル用の余白
    ax_d.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_d.set_ylabel(r'Signal Significance $Z = (\langle\cos\psi\rangle - \mu_{\mathrm{null}}) / \mathrm{SEM}$')
    ax_d.set_title(r'(d) Statistical Decoupling Test ($Z$-score vs Null)', fontweight='bold')
    ax_d.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax_d.grid(True)

    save_figure(fig, "contrast_fig1_coupling_spectrum")
    plt.close(fig)

    order = list(sort_idx)
    return pd.DataFrame({
        'bead_name': [BEADS_INFO[i]['name'] for i in order],
        'rc_over_xi': rc_xi_arr,
        'cos_psi_mean': cos_psi_arr, 'cos_psi_sem': cos_psi_sem_arr,
        'cos_i_mean': cos_i_arr, 'cos_i_sem': cos_i_sem_arr,
        'null_time_shuf_mean': null_t_arr, 'null_time_shuf_std': null_t_std_arr,
        'null_swap_mean': null_s_arr,
        'z_score': z_arr, 'p_two_sided': p_arr,
    })


def plot_fig2_msd_real_vs_virtual(df_imsd: pd.DataFrame):
    """
    Figure 2: MSD 直接比較 (6パネル: 各ビーズサイズ)
    - 実測貨物のアンサンブル平均 MSD (実線)
    - 仮想貨物 (Footprint Advection) のアンサンブル平均 MSD (破線)
    - 比率 MSD_real / MSD_virt
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = b_item['name']
        sub = df_imsd[df_imsd['bead_name'] == bname].copy()
        
        rc_xi = sub['rc_over_xi'].mean()
        d_um = b_item['diameter_um']
        color = b_item['color']
        
        # ラグタイムごとの平均と標準誤差
        grp = sub.groupby('lag_time_s').agg({
            'msd_real': ['mean', stats.sem, 'count'],
            'msd_virt': ['mean', stats.sem]
        })
        
        lags = grp.index.values
        m_real = grp['msd_real']['mean'].values
        s_real = grp['msd_real']['sem'].values
        m_virt = grp['msd_virt']['mean'].values
        s_virt = grp['msd_virt']['sem'].values
        
        # 1. 仮想貨物 MSD (上限・理想追従トレーサー)
        ax.plot(lags, m_virt, color='#332288', linestyle='--', linewidth=2.4, label='Upper Bound: Virtual Cargo (Footprint Tracer)')
        ax.fill_between(lags, np.maximum(1e-4, m_virt - s_virt), m_virt + s_virt, color='#332288', alpha=0.15)
        
        # 2. 実測貨物 MSD
        ax.plot(lags, m_real, color=color, linestyle='-', linewidth=2.4, marker=b_item['marker'],
                markersize=6, markevery=5, label=f'Real Cargo ({b_item["label"]})')
        ax.fill_between(lags, np.maximum(1e-4, m_real - s_real), m_real + s_real, color=color, alpha=0.2)
        
        # 傾き参照線 (alpha = 1: 正常拡散, alpha = 2: 弾道運動)
        t_ref = np.array([4.0, 40.0])
        ref_scale = m_real[0] / t_ref[0]
        ax.plot(t_ref, ref_scale * t_ref, color='gray', linestyle=':', linewidth=1.2, label=r'$\sim \tau^1$ (Diffusion)')
        ax.plot(t_ref, (ref_scale / 10.0) * (t_ref**2), color='dimgray', linestyle='-.', linewidth=1.2, label=r'$\sim \tau^2$ (Ballistic)')
        
        # 比率インセット (Inset: Ratio = MSD_real / MSD_virt)
        ax_ins = ax.inset_axes([0.54, 0.12, 0.42, 0.32])
        ratio = m_real / np.where(m_virt > 0, m_virt, np.nan)
        ax_ins.plot(lags, ratio, color='black', linewidth=1.6)
        ax_ins.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
        ax_ins.set_xscale('log')
        ax_ins.set_xlim(4, 300)
        ax_ins.set_ylim(0, max(1.5, float(np.nanmax(ratio[:15])) * 1.1))
        ax_ins.set_title(r'$\mathrm{MSD}_{\mathrm{real}} / \mathrm{MSD}_{\mathrm{virt}}$', fontsize=9.5, pad=3)
        ax_ins.tick_params(labelsize=8)
        ax_ins.grid(True, linestyle=':', alpha=0.6)
        
        n_p = sub['particle'].nunique()
        ax.set_title(f"{b_item['label']}  ($R_c/\\xi = {rc_xi:.2f}, N={n_p}$)", fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3.5, 350)
        ax.set_ylim(0.01, 1000)
        ax.grid(True)
        
        if idx % 3 == 0:
            ax.set_ylabel(r'MSD $\langle \Delta r^2(\tau) \rangle$ [$\mu\mathrm{m}^2$]')
        if idx >= 3:
            ax.set_xlabel(r'Lag time $\tau$ [s]')
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9.0, framealpha=0.9)

    fig.suptitle(r'Ensemble Mean Squared Displacement: Real Cargo vs Virtual Cargo (Footprint Tracer)',
                 fontsize=16, fontweight='bold', y=0.995)
    save_figure(fig, "contrast_fig2_msd_real_vs_virtual_6panels")
    plt.close(fig)


def _finite_disp(disp_data: dict, kind: str, bname: str, lag: int) -> np.ndarray:
    """disp_data[kind][bname][lag] を有限値の 1D 配列で返す (欠損時は空配列)。"""
    vals = disp_data.get(kind, {}).get(bname, {}).get(lag, [])
    arr = np.asarray(vals, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def plot_fig3_displacement_pdf(disp_data: dict, lag: int = 1, bw: float = 0.2):
    """
    Figure 3: 変位確率密度関数 (Displacement PDF) 直接比較
    - tau = lag * DT_SEC での実測 vs 仮想貨物 (lag はキャッシュ内のフレーム数)
    - 規格化変位 Delta x / sigma, K = 過剰尖度 (Fig4 の伝達係数 kappa とは別物)
    """
    tau_sec = lag * DT_SEC
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    x_grid = np.linspace(-6, 6, 200)
    gauss_ref = stats.norm.pdf(x_grid, 0, 1)

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = b_item['name']
        color = b_item['color']
        
        r_disp = _finite_disp(disp_data, 'real', bname, lag)
        v_disp = _finite_disp(disp_data, 'virt', bname, lag)
        if r_disp.size < 3 or v_disp.size < 3:
            print(f"  [fig3] skip {bname}: insufficient displacements at lag={lag} "
                  f"(real={r_disp.size}, virt={v_disp.size})")
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(b_item['label'], fontsize=13, fontweight='bold')
            continue
        
        r_std = np.std(r_disp)
        v_std = np.std(v_disp)
        r_norm = r_disp / r_std if r_std > 0 else r_disp
        v_norm = v_disp / v_std if v_std > 0 else v_disp
        
        r_kurt = stats.kurtosis(r_disp)
        v_kurt = stats.kurtosis(v_disp)
        
        # ガウス基準線
        ax.plot(x_grid, gauss_ref, color='gray', linestyle='--', linewidth=1.5, label='Gaussian $\\mathcal{N}(0, 1)$')
        
        # KDE プロット
        kde_r = stats.gaussian_kde(r_norm, bw_method=bw)
        kde_v = stats.gaussian_kde(v_norm, bw_method=bw)
        
        ax.plot(x_grid, kde_v(x_grid), color='#332288', linestyle='--', linewidth=2.2,
                label=f'Virtual Cargo ($K={v_kurt:.1f}$)')
        ax.plot(x_grid, kde_r(x_grid), color=color, linestyle='-', linewidth=2.4,
                label=f'Real Cargo ($K={r_kurt:.1f}$)')
        
        ax.set_yscale('log')
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(1e-4, 1.2)
        ratio_val = r_std / v_std if v_std > 0 else np.nan
        title_str = f"{b_item['label']}  (" + r"$\sigma_{\mathrm{real}}/\sigma_{\mathrm{virt}} = " + f"{ratio_val:.2f}$)"
        ax.set_title(title_str, fontsize=13, fontweight='bold')
        ax.grid(True)
        
        if idx % 3 == 0:
            ax.set_ylabel(r'Normalized PDF $P(\Delta x / \sigma)$')
        if idx >= 3:
            ax.set_xlabel(r'Normalized displacement $\Delta x / \sigma$')
        if idx == 0:
            ax.legend(loc='lower center', fontsize=9.2, framealpha=0.9)

    fig.suptitle(r'Displacement Probability Density: Real Cargo vs Virtual Cargo '
                 + rf'($\tau = {tau_sec:g}\ \mathrm{{s}}$, $K$: excess kurtosis)',
                 fontsize=16, fontweight='bold', y=0.995)
    suffix = "" if lag == 1 else f"_lag{lag}"
    save_figure(fig, f"contrast_fig3_displacement_pdf_real_vs_virtual{suffix}")
    plt.close(fig)


def _transfer_stats(sub: pd.DataFrame, n_boot: int = 0, ci: float = 95.0, seed: int = 42) -> dict:
    """kappa = sum(vc . uf) / sum(|uf|^2) と速度比 <|vc|>/<|uf|> を算出し、
    軌跡 (exp_dir, particle) 単位のブートストラップで CI を付ける。
    フレーム間相関を保つため行ではなく軌跡を再標本化する。非有限値の行は分子・分母とも一貫して除外。"""
    cols = ['cargo_dx', 'cargo_dy', 'cargo_spd', 'flow_u', 'flow_v', 'flow_spd']
    sub = sub[np.isfinite(sub[cols].to_numpy(dtype=float)).all(axis=1)]
    out = dict(rc_xi=np.nan, kappa=np.nan, speed_ratio=np.nan, kappa_lo=np.nan, kappa_hi=np.nan,
               ratio_lo=np.nan, ratio_hi=np.nan, n_traj=0)
    if sub.empty:
        return out
    # 軌跡ごとの十分統計量: [sum(vc.uf), sum|uf|^2, sum|vc|, sum|uf|]
    g = sub.assign(dot=sub['cargo_dx'] * sub['flow_u'] + sub['cargo_dy'] * sub['flow_v'],
                   fsq=sub['flow_spd'] ** 2).groupby(['exp_dir', 'particle'])[
        ['dot', 'fsq', 'cargo_spd', 'flow_spd']].sum().to_numpy()

    def _est(a):
        return (a[..., 0] / np.where(a[..., 1] > 0, a[..., 1], np.nan),
                a[..., 2] / np.where(a[..., 3] > 0, a[..., 3], np.nan))

    out['kappa'], out['speed_ratio'] = (float(v) for v in _est(g.sum(axis=0)))
    out['rc_xi'] = float(sub['rc_over_xi'].mean())
    out['n_traj'] = len(g)
    if n_boot > 0 and len(g) > 1:
        rng = np.random.default_rng(seed)
        w = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=n_boot)  # (n_boot, n_traj)
        k_b, r_b = _est(w @ g)
        q = [(100 - ci) / 2, 100 - (100 - ci) / 2]
        out['kappa_lo'], out['kappa_hi'] = np.nanpercentile(k_b, q)
        out['ratio_lo'], out['ratio_hi'] = np.nanpercentile(r_b, q)
    return out


def _yerr(val, lo, hi) -> np.ndarray:
    """CI から非対称エラーバー配列 (2, n) を作る。NaN (CI 無し) は 0 扱い。"""
    val, lo, hi = (np.asarray(x, dtype=float) for x in (val, lo, hi))
    return np.nan_to_num(np.vstack([val - lo, hi - val])).clip(min=0)


def _errpoint(ax, x, y, yerr, color, marker, label, n, min_traj):
    """誤差棒付き1点を描画。n < min_traj なら白抜きマーカー、点の右に軌跡数 n を注記。"""
    if not (np.isfinite(x) and np.isfinite(y)):
        return
    low = n < min_traj
    ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, ms=11, capsize=4, linestyle='none', zorder=5,
                mfc='white' if low else color, mew=2.0 if low else 1.0,
                label=label + (f' (n<{min_traj})' if low else ''))
    ax.annotate(f'n={n}', (x, y), xytext=(8, -12), textcoords='offset points', fontsize=9, color=color)


def plot_fig4_velocity_transfer(df_inst: pd.DataFrame, n_boot: int = 500, ci: float = 95.0, seed: int = 42,
                                min_traj: int = 5):
    """
    min_traj: 軌跡数がこれ未満の粒径は白抜きマーカーで描き、CI が信頼できないことを示す。
              各点の横に軌跡数 n を注記する。
    Figure 4: 速度伝達係数 kappa = vc / <u_fp> と速度比
    - (a) 伝達係数 kappa vs Rc/xi
    - (b) 速度ノルム比 <|vc|> / <|u_fp|> vs Rc/xi
    - (c) 代表的な散布図 (0.63 um vs 20.0 um)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])

    st = pd.DataFrame([_transfer_stats(df_inst[df_inst['bead_name'] == b['name']], n_boot, ci, seed)
                       for b in BEADS_INFO])
    rc_xi_list = st['rc_xi'].tolist()
    kappa_list = st['kappa'].tolist()
    ratio_list = st['speed_ratio'].tolist()
    labels = [b['label'] for b in BEADS_INFO]
    colors = [b['color'] for b in BEADS_INFO]
    markers = [b['marker'] for b in BEADS_INFO]
    kappa_err = _yerr(st['kappa'], st['kappa_lo'], st['kappa_hi'])
    ratio_err = _yerr(st['speed_ratio'], st['ratio_lo'], st['ratio_hi'])
    n_traj = st['n_traj'].astype(int).tolist()
    low_n = [labels[i] for i in range(len(n_traj)) if 0 < n_traj[i] < min_traj]
    if low_n:
        print(f"  [fig4] n_traj < {min_traj} (open markers, CI unreliable): {', '.join(low_n)}")

    sort_idx = np.argsort(rc_xi_list)
    rc_xi_arr = np.array(rc_xi_list)[sort_idx]
    kappa_arr = np.array(kappa_list)[sort_idx]
    ratio_arr = np.array(ratio_list)[sort_idx]

    # Panel (a): kappa vs Rc/xi
    ax_a.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=1.8, label='Perfect Transfer ($\\kappa = 1.0$)')
    for i in sort_idx:
        _errpoint(ax_a, rc_xi_list[i], kappa_list[i], kappa_err[:, [i]], colors[i], markers[i], labels[i],
                  n_traj[i], min_traj)
    ax_a.plot(rc_xi_arr, kappa_arr, color='#882255', linewidth=2.0, linestyle='-', alpha=0.7, zorder=4)
    ax_a.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_a.set_xscale('log')
    ax_a.set_xlim(0.04, 4.0)
    ax_a.set_ylim(-0.05, 1.1)
    ax_a.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_a.set_ylabel(r'Velocity transfer coefficient $\kappa$' + (f' ({ci:g}% CI)' if n_boot else ''))
    ax_a.set_title(r'(a) Velocity Transfer Coefficient $\mathbf{v}_c = \kappa \langle\mathbf{u}\rangle_{R_c} + \boldsymbol{\eta}$', fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_a.grid(True)

    # Panel (b): Speed ratio vs Rc/xi
    ax_b.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=1.8, label='Equal Speed Ratio')
    for i in sort_idx:
        _errpoint(ax_b, rc_xi_list[i], ratio_list[i], ratio_err[:, [i]], colors[i], markers[i], labels[i],
                  n_traj[i], min_traj)
    ax_b.plot(rc_xi_arr, ratio_arr, color='#44AA99', linewidth=2.0, linestyle='-', alpha=0.7, zorder=4)
    ax_b.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_b.set_xscale('log')
    ax_b.set_xlim(0.04, 4.0)
    ax_b.set_ylim(0.0, 1.2)
    ax_b.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_b.set_ylabel(r'Speed ratio $\langle |\mathbf{v}_c| \rangle / \langle |\langle\mathbf{u}\rangle_{R_c}| \rangle$')
    ax_b.set_title(r'(b) Cargo-to-Flow Speed Ratio vs $R_c/\xi$', fontweight='bold')
    ax_b.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax_b.grid(True)

    # Panel (c1)/(c2): 最小・最大粒径の散布 (Rc/xi・kappa はデータから算出)
    rc_by_name = {BEADS_INFO[i]['name']: (rc_xi_list[i], kappa_list[i]) for i in range(len(BEADS_INFO))}
    max_v = 0.35
    for ax_c, b_item, cmap, tag in ((ax_c1, BEADS_INFO[0], 'Blues', 'c1'),
                                    (ax_c2, BEADS_INFO[-1], 'Greens', 'c2')):
        sub_c = df_inst[df_inst['bead_name'] == b_item['name']]
        if len(sub_c):
            ax_c.hexbin(sub_c['flow_spd'] * SCALE_UM_PER_PX / DT_SEC,
                        sub_c['cargo_spd'] * SCALE_UM_PER_PX / DT_SEC,
                        gridsize=35, cmap=cmap, mincnt=1, bins='log', extent=(0, max_v, 0, max_v))
        ax_c.plot([0, max_v], [0, max_v], color='#cc2222', linestyle='--', linewidth=1.8, label='1:1 Line')
        ax_c.set_xlim(0, max_v)
        ax_c.set_ylim(0, max_v)
        ax_c.set_xlabel(r'Footprint flow speed $|\langle\mathbf{u}\rangle_{R_c}|$ [$\mu\mathrm{m/s}$]')
        ax_c.set_ylabel(r'Cargo speed $|\mathbf{v}_c|$ [$\mu\mathrm{m/s}$]')
        rc, kp = rc_by_name[b_item['name']]
        ax_c.set_title(f"({tag}) {b_item['label']} ($R_c/\\xi={rc:.2f}$, $\\kappa={kp:.2f}$)", fontweight='bold')
        ax_c.legend(loc='upper left', fontsize=10)
        ax_c.grid(True)

    save_figure(fig, "contrast_fig4_velocity_transfer")
    plt.close(fig)

    return st.drop(columns='rc_xi').assign(bead_name=[b['name'] for b in BEADS_INFO])


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1] if __doc__ else None)
    p.add_argument('--n-iter', type=int, default=100, help='ヌルモデルの反復回数 (default: 100)')
    p.add_argument('--seed', type=int, default=42, help='ヌルモデルの乱数シード (default: 42)')
    p.add_argument('--pdf-lag', type=int, nargs='+', default=[1],
                   help='Fig3 の変位ラグ [frame] (キャッシュ内の値, 例: 1 4 16; default: 1)')
    p.add_argument('--pdf-bw', type=float, default=0.2, help='Fig3 KDE バンド幅 (default: 0.2)')
    p.add_argument('--kappa-boot', type=int, default=500,
                   help='Fig4 kappa/速度比の軌跡ブートストラップ回数 (0 で CI 無効; default: 500)')
    p.add_argument('--kappa-ci', type=float, default=95.0, help='Fig4 信頼区間 [%%] (default: 95)')
    p.add_argument('--kappa-min-traj', type=int, default=5,
                   help='Fig4 で軌跡数がこれ未満の粒径を白抜き表示 (default: 5)')
    args = p.parse_args(argv)
    if args.pdf_bw <= 0:
        p.error('--pdf-bw must be > 0')
    if args.kappa_boot < 0:
        p.error('--kappa-boot must be >= 0')
    if not 0 < args.kappa_ci < 100:
        p.error('--kappa-ci must be in (0, 100)')
    if args.kappa_min_traj < 1:
        p.error('--kappa-min-traj must be >= 1')
    if args.n_iter < 1:
        p.error('--n-iter must be >= 1')
    return args


def main(argv=None):
    args = parse_args(argv)
    setup_matplotlib()
    print("Loading cached data...")
    df_inst, df_imsd, disp_data = load_cached_data()
    print(f"Data loaded: {len(df_inst)} inst, {len(df_imsd)} imsd")

    print("\nComputing Null Models (Time Shuffle & Trajectory Swap)...")
    null_results = compute_null_models(df_inst, n_iter=args.n_iter, seed=args.seed)

    print("\nGenerating Figure 1: Coupling Spectrum & Null Models...")
    summary = plot_fig1_coupling_spectrum(df_inst, null_results)

    print("\nGenerating Figure 2: MSD Real vs Virtual Cargo (6 panels)...")
    plot_fig2_msd_real_vs_virtual(df_imsd)

    print("\nGenerating Figure 3: Displacement PDF Real vs Virtual...")
    avail = sorted(next(iter(disp_data.get('real', {}).values()), {}).keys())
    for lag in args.pdf_lag:
        if lag not in avail:
            print(f"  [fig3] lag={lag} not in cache (available: {avail}); skipped")
            continue
        plot_fig3_displacement_pdf(disp_data, lag=lag, bw=args.pdf_bw)

    print("\nGenerating Figure 4: Velocity Transfer & Speed Ratio...")
    summary = summary.merge(plot_fig4_velocity_transfer(
        df_inst, n_boot=args.kappa_boot, ci=args.kappa_ci, seed=args.seed,
        min_traj=args.kappa_min_traj), on='bead_name', how='left')
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "contrast_summary.csv", index=False)
    print(f"Saved summary table: contrast_summary.csv\n{summary.to_string(index=False, float_format='%.4g')}")

    print("\nAll contrast analysis figures generated successfully!")


if __name__ == '__main__':
    main()
