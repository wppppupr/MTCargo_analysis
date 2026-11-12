```python
import sympy as sp

c, alpha, k2 = sp.symbols('c alpha kappa_2', positive=True)
U = -alpha * sp.atanh(c) + 2 * k2 * c**2

dU = sp.diff(U, c)
d2U = sp.diff(dU, c)

print("dU/dc =", dU)
print("d2U/dc2 =", d2U)

# Equation: c^3 - c + alpha/(4*k2) = 0
# Let Lambda = alpha / (4*k2)
Lambda = sp.symbols('Lambda', positive=True)
eq = c**3 - c + Lambda

# Roots of cubic:
roots = sp.solve(eq, c)
print("\nRoots for c:")
for i, r in enumerate(roots):
    print(f"Root {i+1}:", r)

# Maximum of g(c) = c - c^3 for c in (0, 1):
# g'(c) = 1 - 3*c^2 = 0 => c_max = 1 / sqrt(3)
c_crit = 1 / sp.sqrt(3)
Lambda_crit = c_crit - c_crit**3
print("\nc_crit =", c_crit)
print("Lambda_crit =", Lambda_crit, "=", float(Lambda_crit))


```

```text
dU/dc = -alpha/(1 - c**2) + 4*c*kappa_2
d2U/dc2 = -2*alpha*c/(1 - c**2)**2 + 4*kappa_2

Roots for c:
Root 1: -(-1/2 - sqrt(3)*I/2)*(27*Lambda/2 + sqrt(729*Lambda**2 - 108)/2)**(1/3)/3 - 1/((-1/2 - sqrt(3)*I/2)*(27*Lambda/2 + sqrt(729*Lambda**2 - 108)/2)**(1/3))
Root 2: -(-1/2 + sqrt(3)*I/2)*(27*Lambda/2 + sqrt(729*Lambda**2 - 108)/2)**(1/3)/3 - 1/((-1/2 + sqrt(3)*I/2)*(27*Lambda/2 + sqrt(729*Lambda**2 - 108)/2)**(1/3))
Root 3: -(27*Lambda/2 + sqrt(729*Lambda**2 - 108)/2)**(1/3)/3 - 1/(27*Lambda/2 + sqrt(729*Lambda**2 - 108)/2)**(1/3)

c_crit = sqrt(3)/3
Lambda_crit = 2*sqrt(3)/9 = 0.3849001794597505


```

```python
# Check trigonometric form of roots for c^3 - c + Lambda = 0 when Lambda < 2/(3*sqrt(3))
# Let c = 2/sqrt(3) * cos(theta) or sin(theta)
# 4*(c * sqrt(3)/2)^3 - 3*(c * sqrt(3)/2)
# With c = 2/sqrt(3) * sin(psi):
# c^3 - c = (8 / (3*sqrt(3))) sin^3(psi) - (2/sqrt(3)) sin(psi)
# = - (2/(3*sqrt(3))) * (3 sin(psi) - 4 sin^3(psi))
# = - (2/(3*sqrt(3))) * sin(3*psi)
# So c^3 - c + Lambda = 0 => sin(3*psi) = (3*sqrt(3)/2) * Lambda = Lambda / Lambda_crit !
# This is amazingly clean! Let's verify:

import numpy as np

Lambda_val = 0.2
Lambda_c = 2.0 / (3.0 * np.sqrt(3))

# sin(3 psi) = Lambda / Lambda_c
sin_3psi = Lambda_val / Lambda_c
psi0 = (1.0/3.0) * np.arcsin(sin_3psi)

# Three roots:
# psi_k = psi0, psi0 + 2pi/3, psi0 + 4pi/3 or pi/3 - psi0, etc.
# Since sin(3 psi) = S, solutions for 3psi are:
# 3psi = theta_0 + 2*k*pi, or pi - theta_0 + 2*k*pi
theta_0 = np.arcsin(sin_3psi)

# Roots for c = 2/sqrt(3) * sin(psi):
psi_1 = theta_0 / 3.0
psi_2 = (np.pi - theta_0) / 3.0
psi_3 = - (np.pi + theta_0) / 3.0 # or theta_0/3 - 2pi/3

c1 = (2.0 / np.sqrt(3)) * np.sin(psi_1)
c2 = (2.0 / np.sqrt(3)) * np.sin(psi_2)
c3 = (2.0 / np.sqrt(3)) * np.sin(psi_3)

print("c1 =", c1, "f(c1) =", c1**3 - c1 + Lambda_val)
print("c2 =", c2, "f(c2) =", c2**3 - c2 + Lambda_val)
print("c3 =", c3, "f(c3) =", c3**3 - c3 + Lambda_val)

# Check stability (d2U/dc2)
# d2U/dc2 = - 2*alpha*c / (1 - c^2)^2 + 4*kappa_2
# Note that at dU/dc = 0, alpha / (1 - c^2) = 4*kappa_2 * c
# So d2U/dc2 = - 2 * (4*kappa_2 * c) * c / (1 - c^2) + 4*kappa_2
#            = 4*kappa_2 * [ 1 - 2*c^2 / (1 - c^2) ]
#            = 4*kappa_2 * (1 - 3*c^2) / (1 - c^2) !
# WOW! Look at that:
# d2U/dc2 = 4*kappa_2 * (1 - 3*c^2) / (1 - c^2)
# Therefore, d2U/dc2 > 0 iff 1 - 3*c^2 > 0 iff |c| < 1/sqrt(3) !


```

```text
c1 = 0.20914884844131665 f(c1) = -5.551115123125783e-17
c2 = 0.878885066249973 f(c2) = 1.6653345369377348e-16
c3 = -1.0880339146912894 f(c3) = 5.551115123125783e-17


```

ハイブリッド有効ポテンシャル


$$\tilde{U}_{\mathrm{eff}}(c) = -\alpha \operatorname{arctanh}(c) + 2\kappa_2 c^2 \qquad (c \in (-1, 1))$$


の極小値の位置 $c_{\min}$ は、**パラメータ比 $\frac{\alpha}{4\kappa_2}$ の値に応じて「厳密解が存在するケージング相」と「極小が存在しない弾道滑り台相」に明確に相転移**します。

その厳密な導出と解析解の形は以下の通りです。

---

### 1. 極値条件の導出

ポテンシャルの 1 階微分を計算してゼロとおきます。


$$\frac{d\tilde{U}_{\mathrm{eff}}}{dc} = -\frac{\alpha}{1-c^2} + 4\kappa_2 c = 0$$


両辺に $1-c^2$ を掛けて整理すると、次の 3 次方程式が得られます。


$$c - c^3 = \frac{\alpha}{4\kappa_2}$$

左辺の関数 $g(c) = c - c^3$ は $c \in (0, 1)$ において山型の形状を持ち、その最大値は $g'(c) = 1 - 3c^2 = 0$ より $c = \frac{1}{\sqrt{3}}$ のとき、


$$\Lambda_c \equiv g\left(\frac{1}{\sqrt{3}}\right) = \frac{2}{3\sqrt{3}} \approx 0.3849$$


となります。

---

### 2. 極大・極小の判定（2階微分の驚くべき性質）

2 階微分を計算し、極値条件 $\frac{\alpha}{1-c^2} = 4\kappa_2 c$ を代入して整理すると、極めて美しい関係式が得られます。


$$\begin{aligned} \frac{d^2\tilde{U}_{\mathrm{eff}}}{dc^2} &= -\frac{2\alpha c}{(1-c^2)^2} + 4\kappa_2 \\ &= -\frac{2c}{1-c^2}\underbrace{\left(\frac{\alpha}{1-c^2}\right)}_{=4\kappa_2 c} + 4\kappa_2 \\ &= 4\kappa_2 \left( 1 - \frac{2c^2}{1-c^2} \right) = \mathbf{4\kappa_2 \left( \frac{1 - 3c^2}{1-c^2} \right)} \end{aligned}$$

$c \in (-1, 1)$ では分母 $1-c^2 > 0$ かつ $\kappa_2 > 0$ であるため、符号は分子 $1 - 3c^2$ だけで決まります。

* **$\displaystyle \vert{}c\vert{} < \frac{1}{\sqrt{3}} \approx 0.577$ のとき**: $\displaystyle \frac{d^2\tilde{U}_{\mathrm{eff}}}{dc^2} > 0$ $\implies$ **安定な「局所極小点」**
* **$\displaystyle \vert{}c\vert{} > \frac{1}{\sqrt{3}} \approx 0.577$ のとき**: $\displaystyle \frac{d^2\tilde{U}_{\mathrm{eff}}}{dc^2} < 0$ $\implies$ **不安定な「局所極大点（障壁）」**

---

### 3. 極小値の厳密解（三角関数による閉じた表現）

3 次方程式 $c - c^3 = \frac{\alpha}{4\kappa_2}$ は、三倍角の公式 $\sin(3\psi) = 3\sin\psi - 4\sin^3\psi$ を用いることで、ヴィエトの公式により**三角関数で厳密に解くことができます**。

変数変換 $c = \frac{2}{\sqrt{3}} \sin\psi$ を代入すると、


$$c - c^3 = \frac{2}{3\sqrt{3}} \sin(3\psi) = \Lambda_c \sin(3\psi) = \frac{\alpha}{4\kappa_2}$$


より、


$$\sin(3\psi) = \frac{1}{\Lambda_c}\left(\frac{\alpha}{4\kappa_2}\right) = \frac{3\sqrt{3}\,\alpha}{8\kappa_2}$$


となります。

安定性条件 $\vert{}c\vert{} < \frac{1}{\sqrt{3}}$ を満たす解（主枝解）を選択すると、**極小値の位置 $c_{\min}$ の厳密解**は以下のように求まります。

$$c_{\min} = \frac{2}{\sqrt{3}} \sin \left[ \frac{1}{3} \arcsin \left( \frac{3\sqrt{3}\,\alpha}{8\kappa_2} \right) \right] \qquad \left( \text{for } \frac{\alpha}{4\kappa_2} \le \frac{2}{3\sqrt{3}} \approx 0.385 \right)$$

---

### 4. 領域ごとの物理的振る舞い

#### ① パリティ完全回復極限（$\alpha \to 0$、超大粒子）

$\arcsin(0) = 0$ となるため、


$$c_{\min} = 0 \quad (\theta = \pm 90^\circ)$$


極小点は完全に原点（直交配向）に位置し、完全に対称な調和振動子井戸になります。

#### ② 弱パリティ破れ領域（$\alpha \ll \kappa_2$、大粒子）

$\arcsin(x) \approx x$、$\sin(x) \approx x$ のテイラー展開を用いると、


$$c_{\min} \approx \frac{2}{\sqrt{3}} \cdot \frac{1}{3} \left( \frac{3\sqrt{3}\,\alpha}{8\kappa_2} \right) = \mathbf{\frac{\alpha}{4\kappa_2}}$$


直交ケージングの底（$c=0$）から、パリティ破れ強度 $\alpha$ に比例してわずかに正の方向へドリフトした位置に極小値が形成されます。

#### ③ サドルノード分岐点（$\frac{\alpha}{4\kappa_2} = \Lambda_c \approx 0.385$）

$$c_{\min} = \frac{2}{\sqrt{3}} \sin\left(\frac{\pi}{6}\right) = \frac{1}{\sqrt{3}} \approx 0.577$$


ここで極小点と極大点が衝突して変曲点（サドル）となり、井戸が消滅します。

#### ④ 弾道輸送領域（$\frac{\alpha}{4\kappa_2} > \frac{2}{3\sqrt{3}}$、小粒子）

実数範囲で $d\tilde{U}_{\mathrm{eff}}/dc = 0$ となる解は存在しません。
ポテンシャルは全域で単調減少し、**定義域の内部 $(-1, 1)$ に極小値は存在せず、貨物は境界 $c \to 1$（完全順流）へと転がり落ちます**。

---

### まとめ

* **極小値が存在する条件**:

$$\frac{\alpha}{\kappa_2} \le \frac{8}{3\sqrt{3}} \approx 1.540 \quad \left( \frac{\alpha}{4\kappa_2} \le 0.385 \right)$$


* **極小値の位置 $c_{\min}$**:

$$c_{\min} = \frac{2}{\sqrt{3}} \sin \left[ \frac{1}{3} \arcsin \left( \frac{3\sqrt{3}\,\alpha}{8\kappa_2} \right) \right] \approx \frac{\alpha}{4\kappa_2}$$


* **物理的意味**:
大粒子では直交配向のすぐ隣（$c_{\min} \approx \frac{\alpha}{4\kappa_2}$）に安定な井戸が存在しますが、小粒子になってパリティ破れが臨界値（$0.385$）を突破した瞬間、井戸がカタストロフィックに消滅して単調な傾斜滑り台へと相転移します。