# KL ダイバージェンス・情報伝達効率・ラグランジュ緩和時間による巨視的 MSD の統一的議論

本ドキュメントは、異常データ（`beads1um_crop` および `beads1um001_crop` の `pid=7`）を除外したクレンジング済みデータセット（全 65 粒子・27,135 フレーム・21 貨物動画＋3 コントロール動画）に基づき、**ミクロな 1 ステップ配向情報量 $D_{\mathrm{KL}}$・情報伝達効率 $\eta \equiv D_{\mathrm{KL}}/D_{\mathrm{MT}}$・ラグランジュ速度自己相関 $\tau$・巨視的平均二乗変位 $\mathrm{MSD}(\Delta t)$** の相互関係をまとめたものである。

---

## 1. 全体像：Green–Kubo 分解による MSD の振幅と指数の分離

時間刻み $\Delta t = 4.0\,\mathrm{s}$ の離散速度自己相関 $G_{\mathbf{v}}(k) = \langle \Delta\mathbf{r}_0 \cdot \Delta\mathbf{r}_k \rangle$ を用いると、任意時間 $t = M\Delta t$ における $\mathrm{MSD}$ は厳密に次の 2 項に分解される：

$$\mathrm{MSD}(M\Delta t) \;=\; \underbrace{M\sigma_1^2}_{\mathrm{MSD}_{\mathrm{zero}}\text{（無相関ブラウニアン項）}} \;+\; \underbrace{2\sum_{k=1}^{M-1}(M-k)\,G_{\mathbf{v}}(k)}_{\mathrm{MSD}_{\mathrm{tail}}\text{（時間相関テール項）}}$$

ここで $\sigma_1^2 = G_{\mathbf{v}}(0) = \langle |\Delta\mathbf{r}|^2 \rangle$ である。実測値（$\Delta t = 240\,\mathrm{s}, M = 60$）は次の通り：

| 貨物直径 $2R_c$ | $N$ | $x = R_c/\xi$ | $D_{\mathrm{KL}}$ [nats] | $\eta = D_{\mathrm{KL}}/D_{\mathrm{MT}}$ | $\eta_{\mathrm{mech}} = D_{\mathrm{KL}}/D_{\mathrm{KL}}^{\mathrm{conv}}$ | $\mathrm{MSD}_{\mathrm{zero}}(240\,\mathrm{s})$ [$\mu\mathrm{m}^2$] | $\mathrm{MSD}_{\mathrm{tail}}(240\,\mathrm{s})$ [$\mu\mathrm{m}^2$] | $\mathrm{MSD}_{\mathrm{tot}}(240\,\mathrm{s})$ [$\mu\mathrm{m}^2$] | テール寄与率 | $\alpha_{4\text{--}240\,\mathrm{s}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.63 um` | 13 | `0.066` | `0.1615` | `0.0661` (`6.61%`) | `1.108` | `24.54` | **`403.55`** | `428.09` | `94.3%` | `1.698` |
| `1.18 um` | 12 | `0.136` | `0.1217` | `0.0577` (`5.77%`) | `0.891` | `23.59` | **`208.42`** | `232.01` | `89.8%` | `1.558` |
| `3.37 um` | 13 | `0.299` | `0.0634` | `0.0375` (`3.75%`) | `0.556` | `26.98` | **`56.31`** | `83.29` | `67.6%` | `1.275` |
| `7.24 um` | 12 | `0.406` | `0.0238` | `0.0157` (`1.57%`) | `0.222` | `18.02` | **`36.72`** | `54.74` | `67.1%` | `1.271` |
| `5.0 um` | 13 | `0.740` | `0.0100` | `0.0097` (`0.97%`) | `0.123` | `11.74` | **`47.35`** | `59.09` | `80.1%` | `1.395` |
| `20 um` | 3 | `1.964` | `0.0023` | `0.0050` (`0.50%`) | `0.057` | `12.46` | **`4.17`** | `16.63` | `25.1%` | `1.070` |

* 無相関項 $\mathrm{MSD}_{\mathrm{zero}}$ はサイズ全体で高々 **約 2 倍**（`11.7–27.0 um^2`）しか変化しないのに対し、相関テール $\mathrm{MSD}_{\mathrm{tail}}$ は **`403.6 → 4.2 um^2`（約 100 倍低下）** と劇的に消失する。
* すなわち、**サイズ増大に伴う MSD の激減と超拡散 $\to$ ブラウニアン拡散転移は、「貨物がその場で動けなくなるから」ではなく、「流れの向き情報を運動記憶として引き継ぐ相関テールが消えるから」** 生じている。

---

## 2. 1 ステップ配向情報量 $D_{\mathrm{KL}}$・効率 $\eta$ と MSD の 1 乗スケーリング

弱〜中整列領域において、1 ステップ（$\Delta t = 4\,\mathrm{s}$）の相対角分布から得られる KL ダイバージェンスは 1 次モーメントの 2 乗でスケールする：

$$D_{\mathrm{KL}} \;\simeq\; m_1^2 \;\propto\; \langle\cos\psi\rangle^2 \;\propto\; \frac{v_{\mathrm{drift}}^2}{\langle|\mathbf{v}_c|^2\rangle}$$

一方、Ornstein–Uhlenbeck / Green–Kubo 相関テールも同様に $v_{\mathrm{drift}}^2$ に比例するため、両者を組み合わせると次の **1 乗スケーリング則** が導かれる：

$$\mathrm{MSD}_{\mathrm{tail}}(t) \;\propto\; \langle|\mathbf{v}_c|^2\rangle\, D_{\mathrm{KL}} \cdot \tau_p(t)$$

### 実測スケーリング則（全 6 サイズ間の回帰・順位相関）
| 目的変数（巨視的 MSD 指標） | 説明変数（情報理論指標） | Spearman $\rho$ ($p$ 値) | 両対数傾き | 決定係数 $R^2$ ($p$ 値) |
| :--- | :--- | :---: | :---: | :---: |
| $\mathrm{MSD}_{\mathrm{tail}}(240\,\mathrm{s})$ | $D_{\mathrm{KL}}$ | `+0.943` (`0.0048`) | **`0.918`** | **`0.884`** (`0.0053`) |
| $\mathrm{MSD}_{\mathrm{tot}}(240\,\mathrm{s})$ | $D_{\mathrm{KL}}$ | `+0.943` (`0.0048`) | `0.667` | **`0.888`** (`0.0049`) |
| $\mathrm{MSD}(300\,\mathrm{s})$（`MSD.py`） | $\eta = D_{\mathrm{KL}}/D_{\mathrm{MT}}$ | **`+1.000`** (`< 1e-6`) | **`1.001`** | **`0.836`** (`0.0107`) |
| $\mathrm{MSD}_{\mathrm{tot}}(240\,\mathrm{s})$ | $\eta = D_{\mathrm{KL}}/D_{\mathrm{MT}}$ | `+0.943` (`0.0048`) | **`1.039`** | **`0.879`** (`0.0058`) |
| 無次元 $\widetilde{\mathrm{MSD}}(\tilde{t}=20)$ | $\eta = D_{\mathrm{KL}}/D_{\mathrm{MT}}$ | `+0.829` (`0.0416`) | **`1.031`** | `0.702` (`0.0373`) |
| $\mathrm{MSD}_{\mathrm{tot}}(240\,\mathrm{s})$ | $\eta_{\mathrm{mech}} = D_{\mathrm{KL}}/D_{\mathrm{KL}}^{\mathrm{conv}}$ | `+0.943` (`0.0048`) | **`0.920`** | **`0.884`** (`0.0052`) |

### 効率の因数分解 $\eta(x) = \eta_{\mathrm{geom}}(x)\cdot\eta_{\mathrm{mech}}(x)$ が示す駆動ボトルネック
最小粒子応答核 $P_0$（`0.63 um`）と背景微小管パッチ分布 $P_{\mathrm{MT}}$ の巡回畳み込み上限 $D_{\mathrm{KL}}^{\mathrm{conv}} = D_{\mathrm{KL}}(P_0 * P_{\mathrm{MT}})$ を用いると、情報伝達効率は次のように因数分解できる：

$$\eta(x) \;\equiv\; \frac{D_{\mathrm{KL}}(x)}{D_{\mathrm{MT}}(x)} \;=\; \underbrace{\frac{D_{\mathrm{KL}}^{\mathrm{conv}}(x)}{D_{\mathrm{MT}}(x)}}_{\eta_{\mathrm{geom}}(x)} \;\times\; \underbrace{\frac{D_{\mathrm{KL}}(x)}{D_{\mathrm{KL}}^{\mathrm{conv}}(x)}}_{\eta_{\mathrm{mech}}(x)}$$

* **$\eta_{\mathrm{geom}}(x)$（線形な幾何学的ぼやけ上限）** は全サイズを通じて `0.060–0.088` とほぼ一定である。
* 対照的に **$\eta_{\mathrm{mech}}(x)$（非線形・力学結合効率）** は `1.108`（`0.63 um`）$\to$ `0.891`（`1.18 um`）$\to$ `0.556`（`3.37 um`）$\to$ `0.222`（`7.24 um`）$\to$ `0.123`（`5.0 um`）$\to$ `0.057`（`20 um`）と 1 桁以上急落し、$\eta(x)$ と $\mathrm{MSD}$ の低下を支配する。
* $\eta_{\mathrm{mech}}$ の低下の中身を足元極性秩序度 $S_1$ で層別分解すると、2 段階の機構が現れる：
  1. **$x \lesssim 0.4$（`1.18–7.24 um`）**：足元が揃った瞬間（$S_1 > 0.95$）の配向一致度 $\langle\cos\psi\rangle$ は `0.418 → 0.441 → 0.426 → 0.365` とほとんど落ちない一方、複数ドメインをまたぐことによる綱引き状態（$S_1 \le 0.70$）の時間割合が `5.2% → 16.2% → 37.9% → 54.7%` へ急増し、非線形失速を引き起こす。
  2. **$x \gtrsim 0.7$（`5.0, 20 um`）**：$S_1 > 0.95$ の瞬間でも $\langle\cos\psi\rangle$ が `0.212 → 0.067` へ低下し、スリップ・ピン止め・抵抗増による真の機械的デカップリングが加わる。

---

## 3. ラグランジュ自己相関との統合：緩和時間と律速機構の転移

MSD の積分メモリ時間 $\tau_{\mathrm{GK}} = \frac{\Delta t}{2}\frac{\mathrm{MSD}_{\mathrm{tail}}(240\,\mathrm{s})}{\mathrm{MSD}_{\mathrm{zero}}(240\,\mathrm{s})}$ は、結合振幅 $A_{\mathrm{coup}}$ と遅い緩和時間 $\tau_{\mathbf{v}}^{\mathrm{slow}}$ の積に分解される：

$$\tau_{\mathrm{GK}} \;\approx\; A_{\mathrm{coup}} \cdot \tau_{\mathbf{v}}^{\mathrm{slow}}, \qquad \frac{1}{\tau_{\mathbf{v}}^{\mathrm{slow}}} \;=\; \frac{1}{\tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}} + \frac{1}{\tau_{\mathrm{decoup}}}$$

ここで、$\tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}} = \int_0^{240} C_{\bar{\mathbf{u}}}(\tau)\,d\tau$ は足元流れ自身のラグランジュ寿命、$\tau_{\mathbf{v}}^{\mathrm{slow}} = \frac{1}{C_{\mathbf{v}}(4\,\mathrm{s})}\int_4^{240} C_{\mathbf{v}}(\tau)\,d\tau$ は貨物の遅い相関緩和時間である。

| サイズ ($2R_c$) | $x = R_c/\xi$ | $\tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$ [s] | $\tau_{\mathbf{v}}^{\mathrm{slow}}$ [s] | $\tau_{\mathrm{GK}}$ [s] | 推定 $\tau_{\mathrm{decoup}}$ [s] | 律速レジーム |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.63 um` | `0.066` | `81.4` | **`119.9`** | `32.9` | $\gg \tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$ | **流れ寿命律速（Flow-lifetime limited）** |
| `1.18 um` | `0.136` | `53.0` | **`49.0`** | `17.7` | `~650` | **流れ寿命律速（Flow-lifetime limited）** |
| `3.37 um` | `0.299` | `39.0` | **`20.5`** | `4.2` | `~43` | 遷移域（両者が拮抗） |
| `7.24 um` | `0.406` | `130.3` | **`10.4`** | `4.1` | `~11` | **脱結合律速（Decoupling limited）** |
| `5.0 um` | `0.740` | `156.9` | **`41.0`** | `8.1` | `~55` | **脱結合律速（Decoupling limited）** |
| `20 um` | `1.964` | `223.2` | **`< 4`** | `0.7` | `< 4` | **即座に脱結合（Elastic/caging dominated）** |

* **小粒子（$x \lesssim 0.14$）**：$\tau_{\mathbf{v}}^{\mathrm{slow}} \approx \tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$ であり、足元の流れが向きを変える寿命いっぱいまで流れに乗り続ける。したがって超拡散からブラウニアンへの折れ曲がり時間 $t_\times \approx 2\tau_{\mathbf{v}}^{\mathrm{slow}}$ は `100–240 s` 以上となる。
* **大粒子（$x \gtrsim 0.3$）**：足元の流れ自体のラグランジュ寿命は `130–223 s` と長いにもかかわらず、貨物は $\tau_{\mathrm{decoup}} \sim 11\text{--}55\,\mathrm{s}$（`20 um` では `< 4 s`）で流れから脱結合するため、短時間（$t_\times \sim 8\text{--}40\,\mathrm{s}$）でブラウニアン拡散へ折れ曲がる。

---

## 4. なぜ $0.63\,\mu\mathrm{m}$ だけ $1000\,\mathrm{s}$ まで超拡散に見えるのか

アンサンブル平均 MSD と中央値・個別粒子 MSD を対比すると、$0.63\,\mu\mathrm{m}$ 特有の統計構造と物理が浮かび上がる：

* **$0.63\,\mu\mathrm{m}$ の中央値（典型粒子）自体は長時間でブラウニアン**（$\alpha_{\mathrm{med}}(40\text{--}200\,\mathrm{s}) = 1.24 \to \alpha_{\mathrm{med}}(200\text{--}600\,\mathrm{s}) = 1.06$）。
* しかし最小粒子（$x = 0.066$）は GMM Run 状態での効率が $\eta_{\mathrm{Run}} = 0.325$（有効ポテンシャル井戸 $3.80\,k_B T$）と突出して高く、交差フィラメントにも引っかかりにくいため、**13 粒子中の 2 粒子（`beads06um001_0`・`beads06um003_18`）が微小管フローに同期したまま（$\langle\cos\psi\rangle = 0.695, 0.584$）$120\text{--}186\,\mu\mathrm{m}$ をほぼ直進**する。
* MSD は変位の 2 乗平均であるため、この 2 粒子が $0.63\,\mu\mathrm{m}$ 全体の $\mathrm{MSD}(300\,\mathrm{s})$ の **`78.1%`**、$\mathrm{MSD}(600\,\mathrm{s})$ の **`94.8%`** を占め、アンサンブル平均 MSD の傾きを $\alpha \approx 1.9\text{--}2.0$ に保つ（この 2 粒子を除いた残り 11 粒子平均は $\alpha(200\text{--}600\,\mathrm{s}) = 0.29$ と折れ曲がる）。
* $1.18\,\mu\mathrm{m}$ 以上ではサイズ増大に伴う脱線・干渉確率の上昇により数百秒・$100\,\mu\mathrm{m}$ 級の超長距離 Run テールが断ち切られるため（$1.18\,\mu\mathrm{m}$ の最大正味変位は `57.1 um`、Run 時 $C_{\mathbf{v}}(120\,\mathrm{s}) = 0.225$ vs `0.63 um` の `0.549`）、アンサンブル平均でも $\Delta t \gtrsim 200\,\mathrm{s}$ でブラウニアン拡散へ収束する。

---

## 5. $D_{\mathrm{KL}}$ と緩和時間による MSD の定量再構成とサイズ依存の乖離機構

1 ステップ分散 $\sigma_1^2$ と、$D_{\mathrm{KL}}$・緩和時間のみを入力とする離散 Ornstein–Uhlenbeck 式

$$\mathrm{MSD}_{\mathrm{pred}}(M\Delta t) \;=\; \sigma_1^2 \left[\, M \;+\; 2\,A \sum_{k=1}^{M-1} (M - k)\,\exp\!\left(-\frac{k\Delta t}{\tau_p}\right) \right]$$

で全 6 サイズ $\times$ 75 ラグ（450 点, $\Delta t = 4\text{--}300\,\mathrm{s}$）を再構成した結果は次の通り：

* **ゼロパラメータモデル（$A = \sqrt{D_{\mathrm{KL}}},\; \tau_p = \tau_{\mathbf{v}}^{\mathrm{slow}}$）**：全体 $R^2_{\log_{10}} = \mathbf{0.9457}$、$\mathrm{RMSE}_{\log_{10}} = 0.1612\,\mathrm{dex}$（1.45 倍精度）。
* **流れ時間のみモデル（$A = \min(1, c\,D_{\mathrm{KL}}),\; c = 3.62,\; \tau_p = \tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$、貨物 VACF 不使用）**：全体 $R^2_{\log_{10}} = \mathbf{0.9490}$、$\mathrm{RMSE}_{\log_{10}} = 0.1561\,\mathrm{dex}$（1.43 倍精度）。
* **OU 基準線（$A = \max(0, C_{\mathbf{v}}(4\,\mathrm{s})),\; \tau_p = \tau_{\mathbf{v}}^{\mathrm{slow}}$）**：全体 $R^2_{\log_{10}} = \mathbf{0.9911}$、$\mathrm{RMSE}_{\log_{10}} = 0.0651\,\mathrm{dex}$（1.16 倍精度）。

### 小粒子と大粒子で一致度が変わる物理的原因（速度自己相関の平行・直交成分分解）
各ペア $(t, t+\tau)$ の速度内積 $\mathbf{v}(t)\cdot\mathbf{v}(t+\tau)$ を時刻 $t$ の足元平均流 $\hat{\bar{\mathbf{u}}}(t)$ の基底で厳密分解（$G_{\mathbf{v}} \equiv G_{\parallel} + G_{\perp}$）すると、$D_{\mathrm{KL}}$ 予測の精度差の理由が明快に説明できる：

| サイズ ($2R_c$) | $\sqrt{D_{\mathrm{KL}}}$ | 実測 $C_{\mathbf{v}}(4\,\mathrm{s})$ | 平行 $C_{\mathbf{v},\parallel}(4\,\mathrm{s})$ | 直交 $C_{\mathbf{v},\perp}(4\,\mathrm{s})$ | $\mathrm{MSD}_{\mathrm{tail}}$ 中の直交成分比 | $R^2_{\log}(D_{\mathrm{KL}}+\tau_{\mathbf{v}}^{\mathrm{slow}})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.63 um` | `0.402` | `0.473` | `0.430` | `0.044` | **`3.1%`** | **`0.991`** |
| `1.18 um` | `0.349` | `0.497` | `0.422` | `0.074` | **`10.0%`** | **`0.975`** |
| `3.37 um` | `0.252` | `0.222` | `0.172` | `0.050` | **`16.4%`** | **`1.000`** |
| `7.24 um` | **`0.154`** | **`0.361`** | `0.224` | **`0.138`** | **`21.9%`** | `0.700` |
| `5.0 um` | **`0.100`** | **`0.224`** | `0.165` | **`0.060`** | **`13.5%`** | `0.794` |
| `20 um` | `0.048` | **`-0.345`** | `-0.207` | `-0.137` | **`78.8%`** | `0.944` |

1. **小粒子（`0.63–3.37 um`）で高精度に一致する理由**：相関テールの **`84–97%`** が足元平均流に平行な成分（$C_{\mathbf{v},\parallel}$）で占められており、「貨物の運動記憶 $\approx$ 流れから受け取った配向情報 $D_{\mathrm{KL}}$」という純粋な流れ駆動レジームにあるため。
2. **`7.24 um`・`5.0 um` で $D_{\mathrm{KL}}$ 予測が過小評価（約半分）になる理由**：
   * 相関テールの **`13.5–21.9%`** が平均流に直交する局所運動の記憶（$C_{\mathbf{v},\perp}$）であり、原理的に平均流方向への整列度 $D_{\mathrm{KL}}$ に現れないこと。
   * 平行成分 $C_{\mathbf{v},\parallel}(4\,\mathrm{s})$ 自体も $\sqrt{D_{\mathrm{KL}}}$ の約 `1.5–1.6 倍` あり、全ピクセル角 $\theta_{\mathrm{pt}}$ で評価する極性指標 $\sqrt{D_{\mathrm{KL}}}$ が足元角度ばらつき $\Phi_{\mathrm{MT}}$ やネマチック往復運動によって目減りすること。
3. **`20 um` の短時間（`4–40 s`）だけ過大評価になる理由**：強い負の自己相関 $C_{\mathbf{v}}(4\,\mathrm{s}) = -0.345$（ネットワークによる弾性ケージング・押し戻し）により実測 MSD がブラウニアン直線 $M\sigma_1^2$ の下側に沈むのに対し、$D_{\mathrm{KL}} \ge 0$ モデルは下限がブラウニアン $M\sigma_1^2$ となるため。

---

## 6. 統一的結論（論文ストーリーの要点）
長時間の有効拡散係数 $D_{\mathrm{eff}}$ は次の積構造として一貫して理解できる：

$$D_{\mathrm{eff}} \;\simeq\; D_0 \;+\; \frac{1}{2}\langle|\mathbf{v}_c|^2\rangle\,\underbrace{A_{\mathrm{coup}}(D_{\mathrm{KL}}, \eta_{\mathrm{mech}})}_{\substack{\text{情報理論的結合振幅}\\ (\mathrm{MSD}_{\mathrm{tail}}\propto D_{\mathrm{KL}}^1\propto \eta^1)}} \;\times\; \underbrace{\left(\frac{1}{\tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}} + \frac{1}{\tau_{\mathrm{decoup}}}\right)^{-1}}_{\substack{\text{ラグランジュ記憶との比較で決まる緩和時間 } \tau_{\mathbf{v}}^{\mathrm{slow}}\\ \text{（流れ寿命律速 } \to \text{ 脱結合律速の転移）}}}$$

* **$D_{\mathrm{KL}}$（および $\eta_{\mathrm{mech}}$）が相関テールの結合振幅を 1 乗則で規定**し、**ラグランジュ自己相関との時間スケール比が「流れに乗って走り続けられる時間（流れ寿命律速 $\to$ 脱結合律速のクロスオーバー）」を決める**。
* 小粒子では両者の結合が強固で 100% 近くが流れ平行成分であるため 1 ステップ情報量と緩和時間から巨視的 MSD をパラメータ 0 個で定量予言でき、大粒子では脱結合・直交局所揺らぎ・弾性ケージングが支配的になってブラウニアン拡散へ回帰する。

---

## 関連スクリプト・CSV・図
* MSD 再構成スクリプト: [plot_msd_from_kl_and_relaxation.py](file:///home/sasaki/MTCargo_analysis/plot_msd_from_kl_and_relaxation.py)
* 情報伝達効率 $\eta = \eta_{\mathrm{geom}}\cdot\eta_{\mathrm{mech}}$ 算出スクリプト: [verify_data_processing_inequality.py](file:///home/sasaki/MTCargo_analysis/verify_data_processing_inequality.py)
* 数値サマリー CSV:
  * [msd_kl_relaxation_reconstruction_summary.csv](file:///home/sasaki/MTCargo_analysis/figure/msd/msd_kl_relaxation_reconstruction_summary.csv)
  * [data_processing_inequality_summary.csv](file:///home/sasaki/MTCargo_analysis/figure/per_point_v_dot_u/data_processing_inequality_summary.csv)
  * [green_kubo_msd_tail_summary.csv](file:///home/sasaki/MTCargo_analysis/figure/lagrangian_flow_autocorr/green_kubo_msd_tail_summary.csv)
* 図:
  * [msd_reconstructed_from_kl_and_tau.png](file:///home/sasaki/MTCargo_analysis/figure/msd/msd_reconstructed_from_kl_and_tau.png)
  * [msd_kl_relaxation_synthesis.png](file:///home/sasaki/MTCargo_analysis/figure/msd/msd_kl_relaxation_synthesis.png)
  * [info_efficiency_eta_dkl_over_dmt.png](file:///home/sasaki/MTCargo_analysis/figure/per_point_v_dot_u/info_efficiency_eta_dkl_over_dmt.png)
  * [green_kubo_msd_tail_decomposition.png](file:///home/sasaki/MTCargo_analysis/figure/lagrangian_flow_autocorr/green_kubo_msd_tail_decomposition.png)
  * [gmm_run_tumble_lagrangian_autocorr.png](file:///home/sasaki/MTCargo_analysis/figure/lagrangian_flow_autocorr/gmm_run_tumble_lagrangian_autocorr.png)


![](figure/msd/msd_reconstructed_from_kl_and_tau.png)