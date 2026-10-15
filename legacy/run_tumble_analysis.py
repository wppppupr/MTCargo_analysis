"""
run_tumble_analysis.py

雋ｨ迚ｩ蠕ｮ邊貞ｭ撰ｼ郁寫蜈峨ン繝ｼ繧ｺ�峨�Run (閭ｽ蜍戊ｼｸ騾�/襍ｰ陦�) 縺ｨ Tumble (蛛懈ｻ�/譁ｹ蜷題ｻ｢謠�) 縺ｮ
繧ｻ繧ｰ繝｡繝ｳ繝��繧ｷ繝ｧ繝ｳ縺翫ｈ縺ｳ謖∫ｶ壽凾髢灘�蟶�ｒ荳諡ｬ隗｣譫舌�蜿ｯ隕門喧縺吶ｋ繧ｹ繧ｯ繝ｪ繝励ヨ縺ｧ縺吶�

蜈ｨ繝薙�繧ｺ繧ｵ繧､繧ｺ��0.63ﾎｼm, 1.18ﾎｼm, 3.37ﾎｼm, 5.0ﾎｼm, 7.24ﾎｼm, 20ﾎｼm�峨↓縺翫＞縺ｦ縲�
1. 邊貞ｭ舌�蟷ｳ蝮�溷ｺｦ繧医ｊ騾溘＞繧ゅ�繧坦un, 驕�＞繧ゅ�繧探umble縺ｨ縺励※繧ｻ繧ｰ繝｡繝ｳ繝��繧ｷ繝ｧ繝ｳ
2. 蜈ｨ邊貞ｭ仙ｾ�� Run 譎る俣蛻�ｸ� (PDF & CCDF) 縺ｮ豈碑ｼ��繝ｭ繝�ヨ
3. 蜈ｨ邊貞ｭ仙ｾ�� Tumble 譎る俣蛻�ｸ� (PDF & CCDF) 縺ｮ豈碑ｼ��繝ｭ繝�ヨ
4. 蜷�ｲ貞ｭ仙ｾ�＃縺ｨ縺ｮ Run/Tumble 蛻�ｸ� 6繝代ロ繝ｫ隧ｳ邏ｰ繝励Ο繝�ヨ�域欠謨ｰ繝輔ぅ繝�ユ繧｣繝ｳ繧ｰ莉倥″��
5. 邊貞ｭ仙ｾ� vs 蟷ｳ蝮⑲un譎る俣 / 蟷ｳ蝮Ⅰumble譎る俣 / Run豈皮紫縺ｮ繧ｵ繝槭Μ繝ｼ繝励Ο繝�ヨ
6. 螳滄圀縺ｮ邊貞ｭ占ｻ瑚ｷ｡縺ｫ縺翫￠繧� Run / Tumble 繧ｻ繧ｰ繝｡繝ｳ繝��繧ｷ繝ｧ繝ｳ縺ｮ蜿ｯ隕門喧繧ｵ繝ｳ繝励Ν
7. 邨ｱ險医し繝槭Μ繝ｼ CSV 縺ｮ蜃ｺ蜉�
繧定｡後＞縺ｾ縺吶�
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# 隕ｪ繝�ぅ繝ｬ繧ｯ繝医Μ縺ｮ繝代せ險ｭ螳�
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import run_tumble as rt

# 繧ｹ繧ｿ繧､繝ｫ縺ｮ驕ｩ逕ｨ
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 繝薙�繧ｺ譚｡莉ｶ險ｭ螳�
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
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


def find_experiment_dirs(root_dir, bead_name):
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


def safe_save_csv(df, target_path, max_retries=5):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            df.to_csv(target_path, index=False)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    csv_text = df.to_csv(index=False)
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(csv_text)
                    return
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    return
            time.sleep(0.5)


def safe_savefig(fig, out_path_base, dpi=300):
    """PNG 縺ｨ PDF 縺ｮ荳｡譁ｹ繧貞ｮ牙�縺ｫ菫晏ｭ倥☆繧�"""
    out_base = Path(out_path_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_base.with_suffix('.png')
    pdf_path = out_base.with_suffix('.svg')

    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path} & {pdf_path}")


def load_and_process_bead_condition(exp_dirs, scale=0.11, frame_interval=4.0, drop_edges=False, threshold_mode='bead_mean'):
    """
    1縺､縺ｮ邊貞ｭ仙ｾ�擅莉ｶ縺ｮ螳滄ｨ薙ョ繧｣繝ｬ繧ｯ繝医Μ鄒､縺九ｉ蜈ｨ霆瑚ｷ｡繧定ｪｭ縺ｿ霎ｼ縺ｿ縲�溷ｺｦ險育ｮ励♀繧医�Run/Tumble謚ｽ蜃ｺ繧定｡後≧縲�
    """
    all_dfs = []
    for d in exp_dirs:
        csv_p = d / "beads_tracks.csv"
        try:
            df = pd.read_csv(csv_p)
            df['exp_dir'] = d.name
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Error reading {csv_p}: {e}")

    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # particle ID 縺ｮ驥崎､�ｒ蝗樣∩
    if 'exp_dir' in combined_df.columns:
        combined_df['particle_unique'] = combined_df['exp_dir'].astype(str) + "_" + combined_df['particle'].astype(str)
        df_for_calc = combined_df.rename(columns={'particle': 'particle_orig', 'particle_unique': 'particle'})
    else:
        df_for_calc = combined_df

    # 迸ｬ譎る溷ｺｦ縺ｮ險育ｮ�
    df_with_v = rt.calc_instantaneous_speeds(df_for_calc, scale=scale, frame_interval=frame_interval)

    valid_speeds = df_with_v['v'].dropna().values
    if len(valid_speeds) == 0:
        return None

    mean_v = float(np.mean(valid_speeds))
    std_v = float(np.std(valid_speeds))
    median_v = float(np.median(valid_speeds))

    # 髢ｾ蛟､縺ｮ豎ｺ螳�
    if threshold_mode == 'bead_mean':
        threshold = mean_v
    elif threshold_mode == 'bead_median':
        threshold = median_v
    else:
        threshold = mean_v

    # Run / Tumble 縺ｮ謚ｽ蜃ｺ
    run_durs, tumble_durs, states_dict = rt.extract_durations_from_df(
        df_with_v, threshold=threshold, frame_interval=frame_interval, drop_edges=drop_edges
    )

    return {
        'mean_v': mean_v,
        'std_v': std_v,
        'median_v': median_v,
        'threshold': threshold,
        'run_durations': run_durs,
        'tumble_durations': tumble_durs,
        'df_with_v': df_with_v,
        'states_dict': states_dict,
        'n_speeds': len(valid_speeds),
    }


def plot_duration_distributions(beads_results, out_dir, root_dir, frame_interval=4.0):
    """
    全粒子径の Unbound 譎る俣蛻�ｸ�ｒ繝励Ο繝�ヨ�育ｷ壼ｽ｢ & 迚�ｯｾ謨ｰ & CCDF��
    """
    # 1. Bound 時間分布 (PDF, 片対数)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax_lin, ax_log = axes[0], axes[1]
    ax_lin.set_title(r"Bound Duration PDF $P(\tau_{\mathrm{bound}})$ (Linear)", fontsize=14)
    ax_lin.set_xlabel(r"Bound Duration $\tau_{\mathrm{bound}}$ [s]", fontsize=13)
    ax_lin.set_ylabel(r"Probability Density [$\mathrm{s}^{-1}$]", fontsize=13)

    ax_log.set_title(r"Bound Duration PDF $P(\tau_{\mathrm{bound}})$ (Semi-log)", fontsize=14)
    ax_log.set_xlabel(r"Bound Duration $\tau_{\mathrm{bound}}$ [s]", fontsize=13)
    ax_log.set_ylabel(r"Probability Density [$\mathrm{s}^{-1}$]", fontsize=13)
    ax_log.set_yscale('log')

    max_t = 120.0
    bins = np.arange(frame_interval, max_t + 2 * frame_interval, frame_interval)

    for item in BEADS_INFO:
        b_name = item['name']
        if b_name not in beads_results or beads_results[b_name] is None:
            continue
        res = beads_results[b_name]
        durs = res['run_durations']
        if len(durs) == 0:
            continue

        centers, pdf, _ = rt.calc_duration_pdf(durs, bins=bins, density=True)
        valid = (pdf > 0) & (centers <= max_t)
        if not np.any(valid):
            continue

        lbl = f"{item['diameter_um']:.2f} $\\mu\\mathrm{{m}}$ ($\\langle\\tau_{{\\mathrm{{bound}}}}\\rangle={np.mean(durs):.1f}\\mathrm{{s}}$, N={len(durs)})"
        ax_lin.plot(centers[valid], pdf[valid], marker=item['marker'], color=item['color'], label=lbl, linewidth=1.8, markersize=6)
        ax_log.plot(centers[valid], pdf[valid], marker=item['marker'], color=item['color'], label=lbl, linewidth=1.8, markersize=6)

    ax_lin.legend(frameon=True, fontsize=10)
    ax_log.legend(frameon=True, fontsize=10)
    ax_lin.grid(True, linestyle='--', alpha=0.5)
    ax_log.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    safe_savefig(fig, root_dir/out_dir / "run_duration_distribution_pdf")
    plt.close(fig)

    # 2. Unbound 時間分布 (PDF, 迚�ｯｾ謨ｰ)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax_lin, ax_log = axes[0], axes[1]
    ax_lin.set_title(r"Unbound Duration PDF $P(\tau_{\mathrm{unbound}})$ (Linear)", fontsize=14)
    ax_lin.set_xlabel(r"Unbound Duration $\tau_{\mathrm{unbound}}$ [s]", fontsize=13)
    ax_lin.set_ylabel(r"Probability Density [$\mathrm{s}^{-1}$]", fontsize=13)

    ax_log.set_title(r"Unbound Duration PDF $P(\tau_{\mathrm{unbound}})$ (Semi-log)", fontsize=14)
    ax_log.set_xlabel(r"Unbound Duration $\tau_{\mathrm{unbound}}$ [s]", fontsize=13)
    ax_log.set_ylabel(r"Probability Density [$\mathrm{s}^{-1}$]", fontsize=13)
    ax_log.set_yscale('log')

    for item in BEADS_INFO:
        b_name = item['name']
        if b_name not in beads_results or beads_results[b_name] is None:
            continue
        res = beads_results[b_name]
        durs = res['tumble_durations']
        if len(durs) == 0:
            continue

        centers, pdf, _ = rt.calc_duration_pdf(durs, bins=bins, density=True)
        valid = (pdf > 0) & (centers <= max_t)
        if not np.any(valid):
            continue

        lbl = f"{item['diameter_um']:.2f} $\\mu\\mathrm{{m}}$ ($\\langle\\tau_{{\\mathrm{{unbound}}}}\\rangle={np.mean(durs):.1f}\\mathrm{{s}}$, N={len(durs)})"
        ax_lin.plot(centers[valid], pdf[valid], marker=item['marker'], color=item['color'], label=lbl, linewidth=1.8, markersize=6)
        ax_log.plot(centers[valid], pdf[valid], marker=item['marker'], color=item['color'], label=lbl, linewidth=1.8, markersize=6)

    ax_lin.legend(frameon=True, fontsize=10)
    ax_log.legend(frameon=True, fontsize=10)
    ax_lin.grid(True, linestyle='--', alpha=0.5)
    ax_log.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    safe_savefig(fig, root_dir/out_dir / "tumble_duration_distribution_pdf")
    plt.close(fig)

    # 3. CCDF (逶ｸ陬懃ｴｯ遨榊�蟶�未謨ｰ) 豈碑ｼ��繝ｭ繝�ヨ (Bound & Unbound)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax_run, ax_tum = axes[0], axes[1]
    ax_run.set_title(r"Bound Duration CCDF $P(T \geq \tau_{\mathrm{bound}})$", fontsize=14)
    ax_run.set_xlabel(r"Bound Duration $\tau_{\mathrm{bound}}$ [s]", fontsize=13)
    ax_run.set_ylabel(r"CCDF $P(T \geq t)$", fontsize=13)
    ax_run.set_yscale('log')

    ax_tum.set_title(r"Unbound Duration CCDF $P(T \geq \tau_{\mathrm{unbound}})$", fontsize=14)
    ax_tum.set_xlabel(r"Unbound Duration $\tau_{\mathrm{unbound}}$ [s]", fontsize=13)
    ax_tum.set_ylabel(r"CCDF $P(T \geq t)$", fontsize=13)
    ax_tum.set_yscale('log')

    for item in BEADS_INFO:
        b_name = item['name']
        if b_name not in beads_results or beads_results[b_name] is None:
            continue
        res = beads_results[b_name]

        # Bound CCDF
        r_durs = res['run_durations']
        if len(r_durs) > 0:
            s_t, ccdf = rt.calc_duration_ccdf(r_durs)
            ax_run.step(s_t, ccdf, where='post', color=item['color'], label=rf"{item['diameter_um']:.2f} $\mu\mathrm{{m}}$", linewidth=1.8)

        # Unbound CCDF
        t_durs = res['tumble_durations']
        if len(t_durs) > 0:
            s_t, ccdf = rt.calc_duration_ccdf(t_durs)
            ax_tum.step(s_t, ccdf, where='post', color=item['color'], label=rf"{item['diameter_um']:.2f} $\mu\mathrm{{m}}$", linewidth=1.8)

    ax_run.legend(frameon=True, fontsize=10)
    ax_tum.legend(frameon=True, fontsize=10)
    ax_run.grid(True, linestyle='--', alpha=0.5)
    ax_tum.grid(True, linestyle='--', alpha=0.5)
    ax_run.set_xlim(left=0, right=140)
    ax_tum.set_xlim(left=0, right=140)
    fig.tight_layout()
    safe_savefig(fig, root_dir/out_dir / "run_tumble_duration_ccdf")
    plt.close(fig)


def plot_6panels_comparison(beads_results, out_dir, root_dir, frame_interval=4.0):
    """
    6つの粒子径ごとに Bound と Unbound の分布を並べて表示する 6 パネル図（指数フィット曲線付き）
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    max_t = 120.0
    bins = np.arange(frame_interval, max_t + 2 * frame_interval, frame_interval)
    t_eval = np.linspace(frame_interval, max_t, 200)

    for idx, item in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = item['name']
        ax.set_title(rf"{item['diameter_um']:.2f} $\mu\mathrm{{m}}$ ({b_name})", fontsize=13, fontweight='bold')

        if b_name not in beads_results or beads_results[b_name] is None:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        res = beads_results[b_name]
        r_durs = res['run_durations']
        t_durs = res['tumble_durations']

        # Bound (Run)
        if len(r_durs) > 0:
            r_c, r_pdf, _ = rt.calc_duration_pdf(r_durs, bins=bins, density=True)
            r_valid = r_pdf > 0
            ax.scatter(r_c[r_valid], r_pdf[r_valid], color='#d62728', marker='o', s=35, label=f"Bound ($\\langle\\tau_{{\\mathrm{{bound}}}}\\rangle={np.mean(r_durs):.1f}\\mathrm{{s}}$)", alpha=0.85)

            # Fit
            r_fit = rt.fit_exponential(r_c, r_pdf)
            if np.isfinite(r_fit['tau']):
                ax.plot(t_eval, rt.exp_decay_func(t_eval, r_fit['tau'], r_fit['a']), '--', color='#d62728', linewidth=1.5,
                        label=rf"Bound fit ($\tau_{{0,\mathrm{{bound}}}}={r_fit['tau']:.1f}\mathrm{{s}}, R^2={r_fit['r_squared']:.2f}$)")

        # Unbound (Tumble)
        if len(t_durs) > 0:
            t_c, t_pdf, _ = rt.calc_duration_pdf(t_durs, bins=bins, density=True)
            t_valid = t_pdf > 0
            ax.scatter(t_c[t_valid], t_pdf[t_valid], color='#1f77b4', marker='s', s=35, label=rf"Unbound ($\langle\tau_{{\mathrm{{unbound}}}}\rangle={np.mean(t_durs):.1f}\mathrm{{s}}$)", alpha=0.85)

            # Fit
            t_fit = rt.fit_exponential(t_c, t_pdf)
            if np.isfinite(t_fit['tau']):
                ax.plot(t_eval, rt.exp_decay_func(t_eval, t_fit['tau'], t_fit['a']), ':', color='#1f77b4', linewidth=1.5,
                        label=rf"Unbound fit ($\tau_{{0,\mathrm{{unbound}}}}={t_fit['tau']:.1f}\mathrm{{s}}, R^2={t_fit['r_squared']:.2f}$)")

        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-4, top=1.0)
        ax.set_xlim(left=0, right=max_t)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(frameon=True, fontsize=8.5, loc='upper right')

        if idx in [0, 3]:
            ax.set_ylabel(r"Probability Density [$\mathrm{s}^{-1}$]", fontsize=12)
        if idx in [3, 4, 5]:
            ax.set_xlabel(r"Duration $\tau$ [s]", fontsize=12)

    fig.tight_layout()
    safe_savefig(fig, root_dir/out_dir / "run_tumble_6panels")
    plt.close(fig)


def plot_summary_vs_diameter(summary_df, out_dir, root_dir):
    """
    邊貞ｭ仙ｾ� vs 蟷ｳ蝮�戟邯壽凾髢薙�謖�焚繝輔ぅ繝�ユ繧｣繝ｳ繧ｰ譎ょｮ壽焚繝ｻRun譎る俣豈皮紫縺ｮ繧ｵ繝槭Μ繝ｼ繝励Ο繝�ヨ
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax_dur, ax_tau, ax_duty = axes[0], axes[1], axes[2]

    d_um = summary_df['diameter_um'].values

    # 1. 蟷ｳ蝮�戟邯壽凾髢� <tau>
    ax_dur.plot(d_um, summary_df['mean_run_dur_sec'], marker='o', color='#d62728', linewidth=2, markersize=8, label=r'Mean Run Duration $\langle \tau_{\mathrm{run}} \rangle$')
    ax_dur.plot(d_um, summary_df['mean_tumble_dur_sec'], marker='s', color='#1f77b4', linewidth=2, markersize=8, label=r'Mean Tumble Duration $\langle \tau_{\mathrm{tumble}} \rangle$')
    ax_dur.set_xscale('log')
    ax_dur.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=13)
    ax_dur.set_ylabel(r'Mean Duration [s]', fontsize=13)
    ax_dur.set_title('Mean Duration vs Diameter', fontsize=14)
    ax_dur.legend(frameon=True, fontsize=11)
    ax_dur.grid(True, linestyle='--', alpha=0.5)

    # 2. 謖�焚繝輔ぅ繝�ユ繧｣繝ｳ繧ｰ譎ょｮ壽焚 tau_0
    ax_tau.plot(d_um, summary_df['fit_tau_run_sec'], marker='o', color='#d62728', linestyle='--', linewidth=2, markersize=8, label=r'Run Lifetime $\tau_{0,\mathrm{run}}$')
    ax_tau.plot(d_um, summary_df['fit_tau_tumble_sec'], marker='s', color='#1f77b4', linestyle='--', linewidth=2, markersize=8, label=r'Tumble Lifetime $\tau_{0,\mathrm{tumble}}$')
    ax_tau.set_xscale('log')
    ax_tau.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=13)
    ax_tau.set_ylabel(r'Decay Constant $\tau_0$ [s]', fontsize=13)
    ax_tau.set_title(r'Exponential Decay Constant $\tau_0$', fontsize=14)
    ax_tau.legend(frameon=True, fontsize=11)
    ax_tau.grid(True, linestyle='--', alpha=0.5)

    # 3. Duty cycle (Run 豈皮紫) & 蟷ｳ蝮�溷ｺｦ
    ax_duty_v = ax_duty.twinx()
    l1 = ax_duty.plot(d_um, summary_df['run_duty_ratio'] * 100, marker='^', color='#2ca02c', linewidth=2, markersize=8, label='Run Fraction [%]')
    l2 = ax_duty_v.plot(d_um, summary_df['mean_speed_ums'], marker='d', color='#9467bd', linewidth=2, linestyle=':', markersize=8, label=r'Mean Speed $\langle v \rangle$ [$\mu\mathrm{m/s}$]')

    ax_duty.set_xscale('log')
    ax_duty.set_xlabel(r'Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=13)
    ax_duty.set_ylabel(r'Run Time Fraction [%]', fontsize=13, color='#2ca02c')
    ax_duty_v.set_ylabel(r'Mean Speed [$\mu\mathrm{m/s}$]', fontsize=13, color='#9467bd')
    ax_duty.set_title('Run Fraction & Mean Speed', fontsize=14)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_duty.legend(lines, labels, frameon=True, fontsize=11, loc='best')
    ax_duty.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    safe_savefig(fig, root_dir/out_dir / "run_tumble_summary_vs_diameter")
    plt.close(fig)


def plot_segmentation_sample(beads_results, out_dir, root_dir):
    """
    莉｣陦ｨ逧�↑邊貞ｭ占ｻ瑚ｷ｡縺ｨ騾溷ｺｦ譎らｳｻ蛻励↓縺翫￠繧� Run / Tumble 繧ｻ繧ｰ繝｡繝ｳ繝��繧ｷ繝ｧ繝ｳ縺ｮ萓九ｒ謠冗判
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, item in enumerate(BEADS_INFO):
        r_idx, c_idx = idx // 3, idx % 3
        ax = axes[r_idx, c_idx]
        b_name = item['name']

        if b_name not in beads_results or beads_results[b_name] is None:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        res = beads_results[b_name]
        states_dict = res['states_dict']
        if not states_dict:
            continue

        # 譛繧る聞縺�ヨ繝ｩ繝�け繧�1縺､驕ｸ謚�
        longest_p = max(states_dict.keys(), key=lambda p: len(states_dict[p]['v']))
        p_data = states_dict[longest_p]

        t_axis = p_data['frame'] * 4.0
        speeds = p_data['v']
        states = p_data['state']
        threshold = res['threshold']

        ax.plot(t_axis, speeds, color='gray', alpha=0.6, linewidth=1.2, label='Speed $v(t)$')
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1.2, label=rf'Threshold $v_{{\mathrm{{th}}}}={threshold:.3f}\mu\mathrm{{m/s}}$')

        # Run 蛹ｺ髢薙ｒ繝上う繝ｩ繧､繝�
        run_mask = states == 1
        tum_mask = states == 0
        ax.scatter(t_axis[run_mask], speeds[run_mask], color='#d62728', s=25, label='Run', zorder=4)
        ax.scatter(t_axis[tum_mask], speeds[tum_mask], color='#1f77b4', s=25, label='Tumble', zorder=4)

        ax.set_title(rf"{item['diameter_um']:.2f} $\mu\mathrm{{m}}$ (Track #{longest_p})", fontsize=12, fontweight='bold')
        ax.set_xlabel('Time $t$ [s]', fontsize=11)
        ax.set_ylabel(r'Speed $v$ [$\mu\mathrm{m/s}$]', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9, loc='upper right')

    fig.tight_layout()
    safe_savefig(fig, root_dir/ out_dir / "segmentation_sample_trajectories")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run and Tumble segmentation and duration distribution analysis.")
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads experiment folders.")
    parser.add_argument('--output_dir', type=str, default=f"figure/run_tumble", help="Output directory for figures and CSVs.")
    parser.add_argument('--scale', type=float, default=0.11, help="Pixel to micron scale (default: 0.11 um/px).")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="Time interval between frames (default: 4.0 s).")
    parser.add_argument('--threshold_mode', type=str, default='bead_mean', choices=['bead_mean', 'bead_median'], help="Threshold criterion.")
    parser.add_argument('--drop_edges', action='store_true', help="Drop edge segments to avoid censoring bias.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Run and Tumble Segmentation Analysis ===")
    print(f"Root Directory: {root_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Scale: {args.scale} um/px, Frame Interval: {args.frame_interval} s")
    print(f"Threshold Mode: {args.threshold_mode}, Drop Edges: {args.drop_edges}")

    beads_results = {}
    summary_rows = []
    all_events_rows = []

    for item in BEADS_INFO:
        b_name = item['name']
        d_um = item['diameter_um']
        exp_dirs = find_experiment_dirs(root_dir, b_name)
        print(f"\nProcessing {b_name} ({d_um} um): Found {len(exp_dirs)} experiments...")

        if not exp_dirs:
            print(f"[WARNING] No experiment directories found for {b_name}")
            continue

        res = load_and_process_bead_condition(
            exp_dirs, scale=args.scale, frame_interval=args.frame_interval,
            drop_edges=args.drop_edges, threshold_mode=args.threshold_mode
        )
        beads_results[b_name] = res

        if res is None:
            continue

        r_durs = res['run_durations']
        t_durs = res['tumble_durations']

        # 謖�焚繝輔ぅ繝�ユ繧｣繝ｳ繧ｰ
        bins = np.arange(args.frame_interval, 120.0 + 2 * args.frame_interval, args.frame_interval)
        r_c, r_pdf, _ = rt.calc_duration_pdf(r_durs, bins=bins, density=True)
        t_c, t_pdf, _ = rt.calc_duration_pdf(t_durs, bins=bins, density=True)

        r_fit = rt.fit_exponential(r_c, r_pdf)
        t_fit = rt.fit_exponential(t_c, t_pdf)

        mean_r = float(np.mean(r_durs)) if len(r_durs) > 0 else np.nan
        std_r = float(np.std(r_durs)) if len(r_durs) > 0 else np.nan
        mean_t = float(np.mean(t_durs)) if len(t_durs) > 0 else np.nan
        std_t = float(np.std(t_durs)) if len(t_durs) > 0 else np.nan

        tot_r_time = np.sum(r_durs) if len(r_durs) > 0 else 0
        tot_t_time = np.sum(t_durs) if len(t_durs) > 0 else 0
        duty_ratio = tot_r_time / (tot_r_time + tot_t_time) if (tot_r_time + tot_t_time) > 0 else np.nan

        print(f"  Speed: mean={res['mean_v']:.4f} um/s, threshold={res['threshold']:.4f} um/s")
        print(f"  Run events: N={len(r_durs)}, <tau>={mean_r:.2f} s, tau_fit={r_fit['tau']:.2f} s (R^2={r_fit['r_squared']:.2f})")
        print(f"  Tumble events: N={len(t_durs)}, <tau>={mean_t:.2f} s, tau_fit={t_fit['tau']:.2f} s (R^2={t_fit['r_squared']:.2f})")
        print(f"  Run fraction (duty ratio): {duty_ratio * 100:.1f}%")

        summary_rows.append({
            'bead_name': b_name,
            'diameter_um': d_um,
            'n_experiments': len(exp_dirs),
            'n_speed_points': res['n_speeds'],
            'mean_speed_ums': res['mean_v'],
            'std_speed_ums': res['std_v'],
            'threshold_ums': res['threshold'],
            'n_run_events': len(r_durs),
            'mean_run_dur_sec': mean_r,
            'std_run_dur_sec': std_r,
            'fit_tau_run_sec': r_fit['tau'],
            'fit_tau_run_err': r_fit['tau_err'],
            'fit_r_squared_run': r_fit['r_squared'],
            'n_tumble_events': len(t_durs),
            'mean_tumble_dur_sec': mean_t,
            'std_tumble_dur_sec': std_t,
            'fit_tau_tumble_sec': t_fit['tau'],
            'fit_tau_tumble_err': t_fit['tau_err'],
            'fit_r_squared_tumble': t_fit['r_squared'],
            'run_duty_ratio': duty_ratio
        })

        for d_val in r_durs:
            all_events_rows.append({'bead_name': b_name, 'diameter_um': d_um, 'state': 'Run', 'duration_sec': d_val})
        for d_val in t_durs:
            all_events_rows.append({'bead_name': b_name, 'diameter_um': d_um, 'state': 'Tumble', 'duration_sec': d_val})

    summary_df = pd.DataFrame(summary_rows)
    all_events_df = pd.DataFrame(all_events_rows)

    # 菫晏ｭ�
    safe_save_csv(summary_df, root_dir / output_dir / "run_tumble_summary.csv")
    safe_save_csv(all_events_df, root_dir / output_dir / "run_tumble_durations_all.csv")
    print(f"\nSaved CSV summaries to {output_dir}")

    # 繝励Ο繝�ヨ逕滓�
    print("\nGenerating plots...")
    plot_duration_distributions(beads_results, output_dir, root_dir, frame_interval=args.frame_interval)
    plot_6panels_comparison(beads_results, output_dir, root_dir, frame_interval=args.frame_interval)
    plot_summary_vs_diameter(summary_df, output_dir, root_dir)
    plot_segmentation_sample(beads_results, output_dir, root_dir)

    print("\n[DONE] All Run and Tumble analyses completed successfully!")


if __name__ == "__main__":
    main()
