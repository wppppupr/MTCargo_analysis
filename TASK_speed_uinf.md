# 実装依頼：貨物の速さを u∞(t) で規格化したモデル検証（2026-11-24）

背景と理論：`/mnt/NAS-A/sasaki/MTC/supplement_v2.md` の S8 節を参照。

## 目的
S8 節のモデル $\mathbf v=A\hat{\mathbf e}+\boldsymbol\zeta$（$s=A/\sigma$ はフレームごとに揺らぐ）を、**向きではなく速さ**で検証する。

S8.5 節で、$\ln|\mathbf v|$ の幅は予測より 0.3–0.5 大きかった。仮説は「雑音の大きさ $\sigma$ が活性 $u_\infty(t)$ に比例して揺らいでいる」である。これが正しければ、$u_\infty(t)$ で割ると以下の三つが起こるはず：
- 幅が予測まで下がる。
- $v_\perp$ がガウス分布に近づく。
- $s$ の分布の形（対数正規分布かガンマ分布か）が、向きのフィットとは独立に決まる。

## 入力
- `scratch/contrast_inst_cache.pkl`：列 `bead_name, exp_dir, particle, frame, cargo_dx, cargo_dy, flow_u, flow_v, rc_over_xi`。
  - `cargo_dx, cargo_dy` は貨物の変位。
  - `flow_u, flow_v` は足印の平均流。**列の定義は `verify_cos_decomposition.py` で必ず確認すること。**
- `scratch/u_inf_per_frame_cache.pkl`：列 `exp_dir, frame, u_inf_t`。
- `scratch/harmonics_frames.pkl`：解析に使うフレームの集合。キーは `(exp, pid, frame)` で、`pid = f"{exp_dir}_{particle}"`。これと同じフレームに絞る。
- 単位：`cargo_dx` と `u_inf_t` の単位（px/フレームか µm/s か）をコードで確認し、比が無次元になるようにそろえる。**わからなければ推測せず REPORT に書く。**

## 定義
- 方向 $\hat{\mathbf e}=(\texttt{flow\_u},\texttt{flow\_v})/|\cdot|$。
- 成分：$v_\parallel=\mathbf v\cdot\hat{\mathbf e}$、$v_\perp=\mathbf v\times\hat{\mathbf e}$（符号付き）。
- 規格化：$w_\parallel=v_\parallel/u_\infty(t)$、$w_\perp=v_\perp/u_\infty(t)$、$w=|\mathbf v|/u_\infty(t)$。
- 除外：$u_\infty(t)$ が動画ごとの下位 5% のフレーム、$|\mathbf v|=0$ のフレーム。除外した割合を報告する。

## 解析（サイズごと。CI は動画→貨物の階層ブートストラップ 300 回）

### A. 雑音が $u_\infty$ に比例するか
1. $v_\perp$ と $w_\perp$ の超過尖度（excess kurtosis）を比べる。$\sigma\propto u_\infty$ なら、$w_\perp$ の方が 0 に近づく。
   - 比較のため、$|\langle\mathbf u\rangle_\mathrm{fp}|=\sqrt{\texttt{flow\_u}^2+\texttt{flow\_v}^2}$ で割った版も同じ表に入れる。
2. $u_\infty(t)$ で 8 分位にビン分けし、各ビンの $\mathrm{Var}(v_\perp)$ を $u_\infty^2$ に対して直線フィットする：$\mathrm{Var}(v_\perp)=\sigma_\mathrm{track}^2+k^2u_\infty^2$。
   - 切片は活性によらない雑音（追跡の誤差）、傾きは能動的な雑音にあたる。
   - 図：`var_vperp_vs_uinf2.png`（サイズごとのパネル）。

### B. $\ln$ の幅の予測との比較
3. $\mathrm{SD}[\ln w]$ を全体と同じ貨物の中（貨物ごとに平均を引く）の両方で求め、下の予測値と並べる。
   - 予測：$s\sim P(s)$、$\mathbf n\sim\mathcal N(0,\mathbf 1_2)$ として $\mathrm{SD}[\ln|s\hat{\mathbf e}+\mathbf n|]$ をモンテカルロで計算する（$2\times10^5$ 標本）。
   - 比較のため、規格化しない $\mathrm{SD}[\ln|\mathbf v|]$ も同じ表に入れる。

### C. $P(s)$ を速さから直接決める（向きのフィットとは独立）
4. $\beta\equiv\mathrm{SD}(w_\perp)$ とおき、$\tilde s\equiv w_\parallel/\beta=s+n_\parallel$（$n_\parallel\sim\mathcal N(0,1)$）と考える。ガウス雑音のキュムラントは 2 次以外 0 なので、次の量が向きのフィットとは独立に求まる：
   - 平均：$\langle s\rangle=\langle\tilde s\rangle$
   - 分散：$\mathrm{Var}(s)=\mathrm{Var}(\tilde s)-1$
   - 3 次のキュムラント：$\kappa_3(s)=\kappa_3(\tilde s)$
   これから CV と歪度を出し、下の 2 つのモデルの予測値と並べる。どちらが近いかを表にする。
5. $w$ の分布全体の比較：
   - モデルは $w=\beta\,|s\hat{\mathbf e}+\mathbf n|$ で、$P(s)$ は下の値に固定し、自由なのは $\beta$ だけとする。
   - $\ln w$ のヒストグラムを、対数正規版とガンマ版の予測と重ねる。
   - $\beta$ は $\mathrm{SD}(w_\perp)$ から決めた値を使う（フィットしない）。
   - KS 距離を表にする。
   - 図：`lnw_hist_vs_model.png`。

### 予測に使う $P(s)$（S8.3 節の値、固定）
| サイズ | 対数正規：中央値 | 対数正規：σ_ln | ガンマ：平均 | ガンマ：CV |
|---|---|---|---|---|
| beads06um | 0.32 | 1.89 | 1.51 | 1.45 |
| beads1um | 0.26 | 1.84 | 1.26 | 1.45 |
| beads3um | 0.35 | 1.00 | 0.57 | 1.08 |
| beads5um | 0.09 | 1.54 | 0.42 | 1.45 |
| beads7um | 0.21 | 0.95 | 0.34 | 1.10 |
| beads20um | 0.02 | 2.52 | 0.40 | 2.43 |

## 出力
- スクリプト：`scratch/speed_uinf_test.py`（新規）。**既存のスクリプトとキャッシュは変更しない。**
- 図と CSV：`figure/speed_uinf_test/`。
- 結果：`figure/speed_uinf_test/REPORT.md` に、A–C の数値の表と、仮定・単位・除外の割合を書く。解釈は短くてよい。
- `result3.md`、`supplement*.md`、`OVERNIGHT_INSTRUCTION.md` は書き換えない。

## 判定の目安（REPORT の最後に書く）
- $w_\perp$ の超過尖度が $v_\perp$ より明らかに 0 に近く、$\mathrm{SD}[\ln w]$ が予測の ±0.15 以内 → 「雑音 ∝ 活性」を支持。
- C で CV と歪度がどちらのモデルに近いか → 分布の形の判定。
- どれも成り立たない → そのまま報告する（モデルを合わせにいかない）。
