# 論文・解析コード統一記法ガイドライン (Unified Physical Notation)

本ドキュメントは、アクティブ微小管（MT）流動場およびカーゴ微粒子輸送解析（HMM, MSD, ACF, 空間配向相関, 有効拡散解析等）における**数式・グラフ軸ラベル・凡例・変数記法**の統一基準を定めたものです（`letter.md` および論文原稿 `draft1.md` 準拠）。

---

## 1. 物理量と統一記号対応表

| 物理量 / 概念 | 旧表記 | **新・統一推奨表記** | 英語表記 / 説明 |
| :--- | :--- | :--- | :--- |
| **粒子サイズ（半径）** | $a, r$ | **$R_c$** | Cargo particle radius（主表記） |
| **粒子サイズ（直径）** | $d, D_c, D_C$ | **$2R_c$** | Cargo particle diameter |
| **足場相関長 / ネマチック相関長** | $R_0, R_{0,\mathrm{vel}}, \xi_{\mathrm{director}}$ | **$\xi$** | MT flow / nematic spatial correlation length |
| **配向緩和時間 / 持続時間** | $\tau_{\mathrm{OACF}}, \tau_{\mathrm{OACF}}^{\mathrm{int}}$ | **$\tau_{\mathrm{p}}, \tau_{\mathrm{p}}^{\mathrm{int}}$** | Orientation persistence / relaxation time |
| **配向緩和速度** | $\tau_{\mathrm{OACF}}^{-1}$ | **$\tau_{\mathrm{p}}^{-1}$** | Orientation relaxation rate |
| **Bound 滞在時間 (Run 時間)** | $\tau_{\mathrm{dwell}}, \tau_{\mathrm{fast}}, \tau_{\mathrm{run}}$ | **$\tau_{\mathrm{bound}}$** | Bound / Run state dwell time |
| **Unbound 滞在時間 (Tumble 時間)**| $\tau_{\mathrm{pause}}, \tau_{\mathrm{slow}}, \tau_{\mathrm{tumble}}$| **$\tau_{\mathrm{unbound}}$** | Unbound / Tumbling state dwell time |
| **有効相関時間 (合成緩和時間)** | $\tau_{\mathrm{eff}} = (\tau_{\mathrm{OACF}}^{-1} + \tau_{\mathrm{dwell}}^{-1})^{-1}$ | **$\tau_{\mathrm{eff}} = (\tau_{\mathrm{p}}^{-1} + \tau_{\mathrm{bound}}^{-1})^{-1}$** | Effective persistence timescale |
| **Run 状態速度** | $v_{\mathrm{fast}}, v_R$ | **$v_{\mathrm{run}}$** | Bound / Run state active velocity |
| **Tumble 状態速度** | $v_{\mathrm{slow}}, v_{\mathrm{pause}}$ | **$v_{\mathrm{tum}}$** | Unbound / Tumbling state velocity |
| **垂直（横）速度揺らぎ** | $\delta v_{c,\perp}$ | **$\delta v_\perp$** | Perpendicular velocity fluctuation amplitude |
| **白色雑音** | $\xi(t)$ | **$\eta(t), \boldsymbol{\eta}(t)$** | Thermal / stochastic Gaussian white noise |
| **HMM 隠れ状態変数** | $\sigma_t, S_t$ | **$I_t \in \{0, 1\}$** | 0: Unbound (Tum), 1: Bound (Run) |
| **配向場スピン / イジング場** | $\sigma(\mathbf{r}) \in \{+1, -1\}$ | **$s(\mathbf{r}) \in \{+1, -1\}$** | Microtubule orientational spin field |
| **状態別速度分布幅 (標準偏差)** | $\sigma_{\mathrm{fast}}, \sigma_{\mathrm{slow}}$ | **$\sigma_{\mathrm{run}}, \sigma_{\mathrm{tum}}$** | Gaussian emission standard deviations |
| **ストークス・アインシュタイン拡散**| $D_0$ | **$D_{\mathrm{SE}}$** | $D_{\mathrm{SE}} = \frac{k_B T}{6\pi \eta R_c}$ |
| **有効拡散係数** | $D_{\mathrm{eff}}$ | **$D_{\mathrm{eff}}$** | $D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2} f_{\mathrm{bound}} v_{\mathrm{run}}^2 \tau_{\mathrm{eff}}$ |

---

## 2. 理論モデルの統一数式

### 2.1 粒子サイズ依存性の指数減衰理論
粒子半径 $R_c$（または直径 $2R_c$）とアクティブ流動相関長 $\xi$ の関係：

- **Run 速度のサイズ依存性**:
  $$v_{\mathrm{run}}(R_c) = v_0 \exp\left( -\frac{4 R_c}{3\xi} \right) = v_0 \exp\left( -\frac{2 (2R_c)}{3\xi} \right)$$
  対数線形フィッティング:
  $$\ln(v_{\mathrm{run}}) = -\frac{4 R_c}{3\xi} + \ln v_0$$

- **配向持続時間 $\tau_{\mathrm{p}}$ のサイズ依存性**:
  $$\tau_{\mathrm{p}}(R_c) = \tau_0 \exp\left( -\frac{4 R_c}{3\xi} \right)$$
  配向緩和速度:
  $$\frac{1}{\tau_{\mathrm{p}}(R_c)} = \frac{1}{\tau_0} \exp\left( \frac{4 R_c}{3\xi} \right)$$

### 2.2 有効持続時間 $\tau_{\mathrm{eff}}$ と輸送ダイナミクス
- **有効緩和時間**:
  $$\tau_{\mathrm{eff}} = \left( \frac{1}{\tau_{\mathrm{p}}} + \frac{1}{\tau_{\mathrm{bound}}} \right)^{-1} = \frac{\tau_{\mathrm{p}} \tau_{\mathrm{bound}}}{\tau_{\mathrm{p}} + \tau_{\mathrm{bound}}}$$

- **ランジュバン方程式**:
  $$\frac{\mathrm{d}\mathbf{r}(t)}{\mathrm{d}t} = I_t v_{\mathrm{run}} \hat{\mathbf{e}}(t) + \delta v_\perp \hat{\mathbf{e}}_\perp(t) + \boldsymbol{\eta}(t)$$
  $$\langle \eta_\alpha(t)\eta_\beta(t') \rangle = 2 D_{\mathrm{SE}} \delta_{\alpha\beta}\delta(t-t')$$

- **有効拡散係数 (Microscopic RTP Model & Green-Kubo Integral)**:
  $$D_{\mathrm{eff}} = D_{\mathrm{SE}} + \frac{1}{2} f_{\mathrm{bound}} v_{\mathrm{run}}^2 \tau_{\mathrm{eff}} \quad \left( f_{\mathrm{bound}} = \frac{\tau_{\mathrm{bound}}}{\tau_{\mathrm{bound}} + \tau_{\mathrm{unbound}}} \right)$$
  $$D_{\mathrm{GK}} = \frac{1}{2} \int_0^\infty \langle \mathbf{v}(t) \cdot \mathbf{v}(t+\tau) \rangle \mathrm{d}\tau$$

### 2.3 空間配向相関関数
- **微小管アクティブフローの空間相関**:
  $$C(r) = \langle \hat{\mathbf{u}}(\mathbf{r}_0) \cdot \hat{\mathbf{u}}(\mathbf{r}_0 + \mathbf{r}) \rangle = a \exp\left( -\frac{r}{\xi} \right)$$
  - 相関長 $\xi$ は縦軸の対数をとってからフィッティングする（$\ln C(r) = \ln a - r/\xi$ の線形回帰より $\xi = -1/\mathrm{slope}$）。

### 2.4 長時間輸送特性のスケーリング（MSD, 変位減衰長）
- **変位 PDF の指数減衰長** $\lambda(\Delta t)$:
  $$P(|\Delta \mathbf{r}|) = A \exp\left( -\frac{|\Delta \mathbf{r}|}{\lambda(\Delta t)} \right), \qquad
  \ln P(|\Delta \mathbf{r}|) = \ln A - \frac{|\Delta \mathbf{r}|}{\lambda(\Delta t)}$$
  - 指数分布では $\lambda(\Delta t) \simeq \langle |\Delta \mathbf{r}| \rangle$ が成り立つため、フィット値が平均変位から大きく外れる場合は統計不足として棄却する。
- **スケール半径**: $x = R_c / \xi$（$\xi$ は各実験の $\xi_{i,t}$ 中央値、または固定値 $2.78\,\mu\mathrm{m}$）。
- **長時間輸送特性のサイズ依存性**:
  $$\mathrm{MSD}(\Delta t = 300\,\mathrm{s}) = \langle \Delta r^2(\Delta t) \rangle, \qquad \lambda(\Delta t = 100\,\mathrm{s})$$
  いずれも $x$ の増加とともに単調減少する（大粒子ほど流動配向の空間平均化により駆動がキャンセルされる）。

---

## 3. グラフ・プロットにおける表記ガイドライン

| 対象 | 推奨 xlabel / ylabel | 推奨 凡例 (Legend) |
| :--- | :--- | :--- |
| **粒子サイズ横軸** | `Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]` または `Cargo Radius $R_c$ [$\mu\mathrm{m}$]` | `$2R_c = 0.63\,\mu\mathrm{m}$`, `$R_c = 0.32\,\mu\mathrm{m}$` |
| **配向緩和時間** | `Orientation Persistence Time $\tau_{\mathrm{p}}$ [s]` | `$\tau_{\mathrm{p}}^{\mathrm{int}}$`, `$\tau_{\mathrm{p}}^{\mathrm{fit}}$` |
| **配向緩和速度** | `Orientation Relaxation Rate $\tau_{\mathrm{p}}^{-1}$ [$\mathrm{s}^{-1}$]` | `Theory: $\frac{1}{\tau_0} \exp\left(\frac{4 R_c}{3\xi}\right)$` |
| **状態滞在時間** | `Dwell Duration $\tau_{\mathrm{bound}}, \tau_{\mathrm{unbound}}$ [s]` | `Bound $\tau_{\mathrm{bound}}$`, `Unbound $\tau_{\mathrm{unbound}}$` |
| **Run速度** | `Run Velocity $v_{\mathrm{run}}$ [$\mu\mathrm{m/s}$]` | `Fit: $\ln(v_{\mathrm{run}}) = -\frac{4 R_c}{3\xi} + B$` |
| **空間距離横軸** | `Distance $r$ [$\mu\mathrm{m}$]` | `$r = 2\,\mu\mathrm{m}$`, `$r = 16\,\mu\mathrm{m}$` |
| **空間相関縦軸** | `Spatial Correlation $C(r)$` | `Fit: $a \exp(-r/\xi)$ ($\xi = 2.78\,\mu\mathrm{m}$)` |
| **局所相関長 vs 速度** | `Cargo Velocity $v_{i,t}$ [$\mu\mathrm{m/s}$]` / `MT Correlation Length $\xi_{i,t}$ [$\mu\mathrm{m}$]` | `All $\xi_{i,t}$ ($N=...$)`, `Binned median ($\pm$IQR)` |
| **MSD 縦軸・横軸** | `MSD $\langle \Delta r^2 \rangle$ [$\mu\mathrm{m}^2$]` / `Lag time $\Delta t$ [s]` | `Bound ($\alpha=1.65$)`, `Unbound ($\alpha=0.98$)` |
| **MSD・変位減衰長 vs スケール半径** | (左軸) `MSD $\langle \Delta r^2(\Delta t = 300\,\mathrm{s}) \rangle$ [$\mu\mathrm{m}^2$]` / (右軸) `Displacement Decay Length $\lambda(\Delta t = 100\,\mathrm{s})$ [$\mu\mathrm{m}$]` / (横軸) `Scaled Cargo Radius $x = R_c / \xi$` | `Individual experiments ($N=...$)`, `$2R_c = 3.37\,\mu\mathrm{m}$ ($x = 0.18$)` |
| **MSD・変位減衰長 vs 貨物半径** | (左軸) `MSD $\langle \Delta r^2(\Delta t = 300\,\mathrm{s}) \rangle$ [$\mu\mathrm{m}^2$]`（黒, log） / (右軸) `Displacement Decay Length $\lambda(\Delta t = 100\,\mathrm{s})$ [$\mu\mathrm{m}$]`（赤, linear） / (横軸) `Cargo Radius $R_c$ [$\mu\mathrm{m}$]`（linear） | `Individual experiments ($N=...$)`, `$2R_c = 3.37\,\mu\mathrm{m}$ ($R_c = 1.69\,\mu\mathrm{m}$)` |
| **有効拡散係数** | `Diffusion Coefficient $D$ [$\mu\mathrm{m}^2/\mathrm{s}$]` | `Model $D_{\mathrm{eff}}$`, `Green-Kubo Median` |

---

## 4. 各解析スクリプトと対応機能

- **[plot_tau_oacf_theory.py](file:///home/sasaki/MTCargo_analysis/plot_tau_oacf_theory.py)**: $\tau_{\mathrm{p}}$ および $\tau_{\mathrm{p}}^{-1}$ の粒子サイズ依存性と理論カーブ可視化
- **[MSD.py](file:///home/sasaki/MTCargo_analysis/MSD.py)**: アンサンブル MSD, 各種緩和時間比較（$\tau_{\mathrm{p}}, \tau_{\mathrm{bound}}, \tau_{\mathrm{eff}}$ vs $2R_c$）
- **[plot_hmm_acf.py](file:///home/sasaki/MTCargo_analysis/plot_hmm_acf.py)**: HMM 状態別自己相関（VACF, OACF, SACF）および相関時間比率比較
- **[plot_run_velocity.py](file:///home/sasaki/MTCargo_analysis/plot_run_velocity.py)**: Run 速度 $v_{\mathrm{run}}$ vs $2R_c$（$\xi$ フィッティング）
- **[run_tumble_analysis.py](file:///home/sasaki/MTCargo_analysis/run_tumble_analysis.py)**: Bound/Unbound dwell 時間分布（PDF, CCDF）
- **[plot_xi_summary.py](file:///home/sasaki/MTCargo_analysis/plot_xi_summary.py)**: 空間相関長 $\xi$ vs $2R_c$
- **[angular_correlation.py](file:///home/sasaki/MTCargo_analysis/angular_correlation.py)**: 2D-FFT 角度空間相関 $C(r), C_\parallel(r), C_\perp(r)$
- **[plot_mt_spatial_correlation_histograms.py](file:///home/sasaki/MTCargo_analysis/plot_mt_spatial_correlation_histograms.py)**: 微小管フロー空間配向相関ヒストグラム
- **[plot_xi_vs_velocity.py](file:///home/sasaki/MTCargo_analysis/plot_xi_vs_velocity.py)**: 各粒子 $i$・各フレーム $t$ の局所相関長 $\xi_{i,t}$ vs 貨物粒子速度 $v_{i,t}$（粒子径ごとの散布図）
- **[plot_msd_lambda_vs_scaled_radius.py](file:///home/sasaki/MTCargo_analysis/plot_msd_lambda_vs_scaled_radius.py)**: MSD($\Delta t = 300\,\mathrm{s}$)（第1軸, 黒）と変位 PDF の指数減衰長 $\lambda(\Delta t = 100\,\mathrm{s})$（第2軸, 赤）の 2軸図。横軸は $x = R_c/\xi$（実験ごとの $\xi_{i,t}$ 中央値を使用）版と `Cargo Radius $R_c$`（linear）版を出力（出力先: `figure/scaling`, `figure`, `<root_dir>/figure/scaling`）

- **[libs/effective_diffusion.py](file:///home/sasaki/MTCargo_analysis/libs/effective_diffusion.py)**: Green-Kubo 積分および RTP 理論モデル有効拡散解析
