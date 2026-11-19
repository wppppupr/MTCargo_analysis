Ran check cutoff usage in verify_cos_decomposition.py
Analyzed verify_cos_decomposition.py:L170-250
Analyzed verify_cos_decomposition.py:L295-335
Analyzed verify_cos_decomposition.py:L2280-2385
Ran check MORNING_REPORT.md and comment.md
Ran check exact A, kappa_2, Lambda from verify_cos_decomposition.py logic
Ran compare cos_i vs cos_bulk using exact verify_cos_decomposition pipeline
Checked task Compare cos_i vs cos_bulk using exact verify_cos_decomposition pipeline
Checked task Compare cos_i vs cos_bulk using exact verify_cos_decomposition pipeline
Ran verify Phi_mag * cos_bulk vs cos_i on exact pipeline
Checked task Verify Phi_mag * cos_bulk vs cos_i on exact pipeline
Checked task Verify Phi_mag * cos_bulk vs cos_i on exact pipeline
Edited verify_coarse_grained_velocity_noise.py
Ran run rigorous coarse-grained velocity analysis
Checked task Run rigorous coarse-grained velocity analysis
Ran check stuck beads in tracks for direct sigma estimate

ご指摘いただいた問題点をすべて修正し、**本論（[`verify_cos_decomposition.py`](file:///home/sasaki/MTCargo_analysis/verify_cos_decomposition.py)）と完全に同一のデータ抽出・解析パイプライン**上で再計算をやり直しました。

### 1. 本論（$n=1$）の厳密な再現確認

カットオフ適用（`get_cutoff_frame`）、`active_set`、同一解析半径（`best_rad`）、`lstsq` 復元変位ベクトル、`cos_i` 定義、およびヤコビアン $\sqrt{1-c^2}$・偶奇比率法をそのまま用いて再実装した結果、**$n=1$ のデータ点数とフィット値が本論の図と小数第2位まで完全に一致**することを確認しました：

| サイズ | データ点数 ($N$) | $A$ | $\kappa_2$ | $\Lambda = A/(4\kappa_2)$ | 95% CI ($\Lambda$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$0.63\,\mu\mathrm{m}$** | 3450 | 0.37 | 0.00 | $\infty$ | $> 1.65$（下限） |
| **$1.18\,\mu\mathrm{m}$** | 3556 | 0.40 | 0.10 | **1.02** | $> 0.31$（下限） |
| **$3.37\,\mu\mathrm{m}$** | 5019 | 0.59 | 0.72 | **0.21** | $[0.15, \, 0.27]$ |
| **$5.00\,\mu\mathrm{m}$** | 3368 | 0.28 | 0.99 | **0.07** | $[0.05, \, 0.10]$ |
| **$7.24\,\mu\mathrm{m}$** | 3203 | 0.41 | 1.22 | **0.08** | $[0.06, \, 0.11]$ |
| **$20.0\,\mu\mathrm{m}$** | 1303 | 0.12 | 0.13 | **0.23** | $[0.00, \infty)$ |

---

### 2. 粗視化ステップ $n = 1, 2, 3, 5$ に対する安定性（95% CI 付き）

連続フレームの変位を積算した粗視化速度 $\mathbf{v}_n(t) = [\mathbf{r}(t+n\Delta t) - \mathbf{r}(t)]/(n\Delta t)$ に対して、$A, \kappa_2, \Lambda$（粒子単位クラスターブートストラップ $B=500$ による 95% CI 付き、$\kappa_2=0$ に張り付く場合は $\Lambda$ を下限扱い）と、**重ならない区間（$\tau \geq n\Delta t$）** での自己相関関数 $C(\tau)$ を算出しました：

| サイズ | $n=1$ ($\kappa_2 / \Lambda$) | $n=2$ ($\kappa_2 / \Lambda$) | $n=3$ ($\kappa_2 / \Lambda$) | $n=5$ ($\kappa_2 / \Lambda$) |
| :--- | :---: | :---: | :---: | :---: |
| **$0.63\,\mu\mathrm{m}$** | 0.00 / $\infty$ | 0.03 / 2.66 | 0.00 / $\infty$ | 0.02 / 4.36 |
| **$1.18\,\mu\mathrm{m}$** | 0.10 / 1.02 | 0.00 / $\infty$ | 0.11 / 0.64 | 0.05 / 0.94 |
| **$3.37\,\mu\mathrm{m}$** | 0.72 / 0.21 | 0.66 / 0.19 | 0.68 / 0.16 | 0.71 / 0.14 |
| **$5.00\,\mu\mathrm{m}$** | 0.99 / 0.07 | 0.94 / 0.08 | 0.90 / 0.08 | 1.00 / 0.06 |
| **$7.24\,\mu\mathrm{m}$** | 1.22 / 0.08 | 1.06 / 0.07 | 1.05 / 0.07 | 1.07 / 0.06 |
| **$20.0\,\mu\mathrm{m}$** | 0.13 / 0.23 | 0.19 / 0.15 | 0.21 / 0.21 | 0.32 / 0.19 |

- **全サイズにおいて $n=1 \to 5$ でサイズ依存性とサドルノード分岐（$\Lambda_c \approx 0.385$）を挟む構造が安定して保たれています**。
  - 小粒子（$0.63, 1.18\,\mu\mathrm{m}$）はすべての $n$ で $\Lambda > 0.385$（弾道領域）。
  - $3.37, 5.00, 7.24\,\mu\mathrm{m}$ はすべての $n$ で $\Lambda < 0.385$（安定直交ケージング領域、$\kappa_2 \approx 0.66 \sim 1.22$）。
  - $20.0\,\mu\mathrm{m}$ も $\kappa_2 \approx 0.13 \sim 0.32$ と弱いまま安定しています。

---

### 3. 位置決め誤差 $\sigma$ の見積もり（変位共分散による下限）

静止ビーズが `beads_tracks.csv` に含まれていなかったため、連続する1フレーム変位の共分散 $\langle \Delta x_k \Delta x_{k+1} \rangle, \langle \Delta y_k \Delta y_{k+1} \rangle$ を全粒子・全有効フレームで直接計算しました：

- **全サイズで共分散は正（運動の持続性が $\sigma^2$ を上回っている）**：
  - $0.63\,\mu\mathrm{m}$: $\langle \Delta r_k \cdot \Delta r_{k+1} \rangle_{\mathrm{iso}} = +0.0968\,\mu\mathrm{m}^2$ （RMS 1フレーム変位 $0.640\,\mu\mathrm{m}$）
  - $1.18\,\mu\mathrm{m}$: $+0.1313\,\mu\mathrm{m}^2$ （RMS $0.712\,\mu\mathrm{m}$）
  - $3.37\,\mu\mathrm{m}$: $+0.0615\,\mu\mathrm{m}^2$ （RMS $0.657\,\mu\mathrm{m}$）
  - $5.00\,\mu\mathrm{m}$: $+0.0496\,\mu\mathrm{m}^2$ （RMS $0.563\,\mu\mathrm{m}$）
  - $7.24\,\mu\mathrm{m}$: $+0.0621\,\mu\mathrm{m}^2$ （RMS $0.572\,\mu\mathrm{m}$）
  - $20.0\,\mu\mathrm{m}$: $+0.0053\,\mu\mathrm{m}^2$ （RMS $0.249\,\mu\mathrm{m}$）
- これにより $\sigma^2 \ge -\langle \Delta x_k \Delta x_{k+1} \rangle$ から得られる下限は全サイズで $0.0$ となりました（ただし $20.0\,\mu\mathrm{m}$ は共分散が $+0.0053\,\mu\mathrm{m}^2$ と小さく、RMS 変位 $0.249\,\mu\mathrm{m}$ に対して数十 nm の誤差でも相対的な寄与が大きくなりうるため、1フレーム評価でのノイズ影響を受けやすい点とも整合します）。

---

### 4. 更新・出力されたファイル

- スクリプト: [`verify_coarse_grained_velocity_noise.py`](file:///home/sasaki/MTCargo_analysis/verify_coarse_grained_velocity_noise.py)
- SI 用 4パネル統合図: [`si_coarse_grained_velocity_noise_robustness.png`](file:///home/sasaki/MTCargo_analysis/figure/cos_decomposition/si_coarse_grained_velocity_noise_robustness.png) ([.svg](file:///home/sasaki/MTCargo_analysis/figure/cos_decomposition/si_coarse_grained_velocity_noise_robustness.svg))
- 粗視化パラメータサマリー CSV: [`si_coarse_grained_velocity_stability_summary.csv`](file:///home/sasaki/MTCargo_analysis/figure/cos_decomposition/si_coarse_grained_velocity_stability_summary.csv)
- 変位共分散・位置決め誤差下限 CSV: [`si_localization_error_covariance.csv`](file:///home/sasaki/MTCargo_analysis/figure/cos_decomposition/si_localization_error_covariance.csv)

