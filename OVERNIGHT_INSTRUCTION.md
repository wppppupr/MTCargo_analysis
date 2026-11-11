# 📝 今夜の解析・研究リクエスト (Overnight Instruction)

研究室を出る前に、今夜 LLM（Claude & Gemini）に進めてほしい要望をここに自由に書いてください。
空欄のまま起動した場合は、自動的に「直近の作業差分から改善点を見つけて進める」モードになります。

---

## 🎯 優先タスク（チェックボックス形式）

- [ ] 各トラジェクトリの各フレームにおいて、貨物瞬時速度 $\mathbf{v}(t)$ を局所微小管流向 $\hat{\mathbf{u}}$ に対する縦成分 $v_\parallel = \mathbf{v}\cdot\hat{\mathbf{u}}$ と直交成分 $v_\perp = -v_x \sin\theta_u + v_y \cos\theta_u$ に分解し、解析用データフレームに格納する（列名: v_parallel, v_perp, v_mag）。
- [ ] 軌跡のスプライン平滑化（または差分間隔 $\Delta t$）のパラメータを変え、局所トラッキングノイズが $v_\parallel$ および $v_\perp$ の分布幅（分散）に与える影響を検証・確定する。
- [ ] 貨物サイズごとに $\langle v_\parallel \rangle$, $\langle v_\parallel^2 \rangle$, $\langle v_\perp^2 \rangle$, $\langle v^2 \rangle$ をクラスターブートストラップ法（トラジェクトリ単位リサンプリング）により計算し、95% 信頼区間を算出する。
- [ ] 横軸 $R_c/\xi$ に対し、正味の直進駆動力 $\langle v_\parallel \rangle$ と直交揺動強度 $\sqrt{\langle v_\perp^2 \rangle}$ を重ねてプロットし、$R_c \approx \xi$ におけるパリティ回復と直交モードへのエネルギー集中を示す図を作成する（figure/v_parallel_perp_vs_rc_over_xi.pdf）。
- [ ] 直交エネルギー分配比 $\langle v_\perp^2 \rangle / \langle v^2 \rangle$ のサイズ依存性をプロットし、大粒子において運動エネルギーが消失（凍結）せず直交揺らぎへ 100% 転換されていることを確認する。
- [ ] 各サイズにおいて、機械的仕事率の相関指標 $\eta_T \equiv \frac{\langle \mathbf{v} \cdot \mathbf{u} \rangle}{\sqrt{\langle \vert{}\mathbf{v}\vert{}^2 \rangle}\sqrt{\langle \vert{}\mathbf{u}\vert{}^2 \rangle}}$ を算出し、ブートストラップ誤差棒を算出する。
- [ ] 横軸 $R_c/\xi$ に対する $\eta_T$ のスケーリング曲線をプロットし、$R_c/\xi \approx 1$ で微小管流からの推進力取得が急激に遮断（Quenching）される転移挙動を可視化する


## 進め方の注意

1 タスク = 1 イテレーション。400 s で打ち切られるので、重い計算はキャッシュを作ってから回す。
図の見た目（凡例の位置など）の調整はしない。
結果は数値（表）で MORNING_REPORT.md に書く。新しい図と CSV は figure/<タスク名>/ に置く。
result3.md 本体は書き換えない。
