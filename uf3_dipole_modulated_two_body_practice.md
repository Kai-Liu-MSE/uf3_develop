# UF3 中 dipole-modulated two-body 项的实践方案

## 1. 目标

本文档总结一种在 UF3 框架中加入 **dipole-modulated two-body** 项的实践方案。

该项的目标不是替代原始 two-body 或 density-modulated two-body，而是作为一个低成本的 **局域不对称性 correction**，用于改善：

- surface energy；
- vacancy / surface vacancy；
- step surface；
- grain boundary；
- undercoordinated asymmetric environments；
- slab relaxation force；
- nanoindentation 中局域不对称缺陷环境。

核心思想是：

> 普通 pair 项只看 \(r_{ij}\)，density-modulated pair 项看 \(r_{ij}\) 与局域配位密度，而 dipole-modulated pair 项进一步让 pair interaction 感知局域邻居分布是否具有方向性不对称。

---

## 2. 为什么需要 dipole-modulated two-body？

标准 UF3 中的 two-body 项为：

\[
E_2 = \sum_{i<j} V_2(r_{ij})
\]

它是 density-independent radial baseline。

density-modulated two-body 可以写成：

\[
E_{2\rho}
=
\sum_{i<j}
\frac{
\psi(\rho_i)+\psi(\rho_j)
}{2}
V_{2\rho}(r_{ij})
\]

其中：

\[
\rho_i = \sum_j f_\rho(r_{ij})
\]

该项描述局域配位密度对 pair bond 的调节。

但 scalar density 只知道“邻居有多少、距离多近”，不知道“邻居是否集中在某个方向”。因此它无法直接区分：

- bulk-like symmetric environment；
- surface-like one-sided environment；
- vacancy-induced asymmetric environment；
- step / edge environment；
- some grain-boundary asymmetric local structures。

dipole-like descriptor 用来补充这一点。

---

## 3. 局域 dipole-like descriptor

定义中心原子 \(i\) 的局域 dipole-like vector：

\[
\mathbf{p}_i
=
\sum_j
 g(r_{ij})\hat{\mathbf{r}}_{ij}
\]

其中：

\[
\hat{\mathbf{r}}_{ij}
=
\frac{\mathbf{r}_{ij}}{r_{ij}}
\]

这里 \(g(r)\) 是平滑 radial weight function。

### 3.1 物理含义

如果 \(i\) 处在完美 bulk-like 环境中，邻居分布大致各向对称，则：

\[
\mathbf{p}_i \approx 0
\]

如果 \(i\) 处在表面、空位、台阶、晶界等局域不对称环境中，则：

\[
\mathbf{p}_i \neq 0
\]

因此 \(\mathbf{p}_i\) 表征的是局域环境的一阶方向不对称性。

---

## 4. 不建议直接使用 \(|\mathbf{p}_i|\)

不建议直接把：

\[
D_i = |\mathbf{p}_i|
\]

作为能量特征。

原因：

1. 在 \(\mathbf{p}_i \to 0\) 附近，\(|\mathbf{p}_i|\) 的导数方向可能不稳定；
2. bulk 环境中 \(\mathbf{p}_i\) 通常接近 0，这可能影响 MD 稳定性；
3. \(|\mathbf{p}_i|\) 与 density / coordination 仍可能强相关；
4. force implementation 中容易出现数值不光滑问题。

更推荐使用平滑 rotational invariant：

\[
P_i = \mathbf{p}_i\cdot\mathbf{p}_i
\]

或归一化形式：

\[
a_i^2 =
\frac{
\mathbf{p}_i\cdot\mathbf{p}_i
}{
(\rho_i+\epsilon)^2
}
\]

其中：

\[
\rho_i = \sum_j f_\rho(r_{ij})
\]

\(\epsilon\) 是小正数，用于避免低密度环境下除零。

---

## 5. 推荐 descriptor：normalized asymmetry \(a_i^2\)

推荐第一版使用：

\[
a_i^2 =
\frac{
\mathbf{p}_i\cdot\mathbf{p}_i
}{
(\rho_i+\epsilon)^2
}
\]

它的优点是：

- rotationally invariant；
- 平滑；
- bulk 中接近 0；
- 表面/空位/台阶中明显变大；
- 比 raw \(P_i\) 更少受 coordination 数值大小影响；
- 更像“形状不对称性”，而不是“低配位程度”。

推荐命名：

```text
p_i      : dipole vector
p2_i     : dot(p_i, p_i)
rho_i    : scalar density
a2_i     : p2_i / (rho_i + eps)^2
```

---

## 6. 推荐的 dipole-modulated two-body 能量形式

第一版建议实现：

\[
E_{2D}
=
\sum_{i<j}
\frac{
a_i^2 + a_j^2
}{2}
V_{2D}(r_{ij})
\]

其中 \(V_{2D}(r)\) 是一维 radial spline，与普通 two-body 类似。

这个形式的含义是：

> 同一根 pair bond 在对称 bulk 环境和非对称 surface / defect 环境下可以有不同能量修正。

---

## 7. 为什么推荐 pair-symmetric 形式？

不推荐只写成 atom-centered form：

\[
E_{2D,i} = a_i^2 \sum_j V_{2D}(r_{ij})
\]

因为 pair contribution 在 \(i,j\) 之间不显式对称，虽然最终能量仍可整理，但实现 force / virial 时更容易混乱。

推荐 pair-symmetric form：

\[
E_{2D}
=
\sum_{i<j}
M_{ij}^{D}
V_{2D}(r_{ij})
\]

其中：

\[
M_{ij}^{D}
=
\frac{a_i^2+a_j^2}{2}
\]

优点：

- pair \(i,j\) 对称；
- energy bookkeeping 清晰；
- 与 LAMMPS pair loop 结构更一致；
- 更容易测试能量-力一致性。

---

## 8. 不推荐一开始加入过多 powers

不要一开始使用：

\[
a_i,
\quad (a_i^2)^{1/2},
\quad a_i^2,
\quad (a_i^2)^2,
\quad (a_i^2)^3
\]

第一版只用：

\[
a_i^2
\]

如果确实需要更强表达能力，可以以后加入：

\[
\chi_1(a_i^2)=a_i^2
\]

\[
\chi_2(a_i^2)=(a_i^2)^2
\]

但这应作为第二阶段，而不是 MVP。

---

## 9. 与 density-modulated two-body 的关系

建议最终模型结构为：

\[
E
=
E_2^{(0)}
+
E_3^{(0)}
+
E_{2\rho}
+
E_{2D}
+
E_{3\rho}
\]

其中：

- \(E_2^{(0)}\)：普通 density-independent radial baseline；
- \(E_3^{(0)}\)：普通 UF3 three-body；
- \(E_{2\rho}\)：density-dependent radial correction；
- \(E_{2D}\)：asymmetry-dependent radial correction；
- \(E_{3\rho}\)：density-dependent angular / stacking correction。

dipole-modulated two-body 不应替代 density-modulated two-body。二者作用不同：

| 项 | 主要信息 | 主要作用 |
|---|---|---|
| \(E_{2\rho}\) | 局域 scalar density / coordination | 修正欠配位下的 bond strength |
| \(E_{2D}\) | 局域方向不对称性 | 修正 surface / vacancy / step 等 asymmetric environment |
| \(E_{3\rho}\) | density-dependent angular environment | 修正 stacking / surface angular response |

---

## 10. 预期有效范围

### 10.1 可能有帮助

dipole-modulated two-body 预计对以下性质有帮助：

- surface energy；
- open surface ranking；
- vacancy formation energy；
- surface vacancy；
- step surface；
- grain boundary excess energy；
- slab relaxation force；
- nanoindentation 中缺陷/表面附近局域环境；
- nanoparticle edge / corner energy。

### 10.2 可能帮助有限

对以下性质帮助可能有限：

- HCP vs FCC bulk energy difference；
- FCC(111) vs HCP(0001) close-packed surface ranking；
- intrinsic stacking fault energy；
- HCP/FCC coherent interface energy。

原因是这些性质更依赖 angular / stacking sequence，而 dipole 只是 \(l=1\) 的低阶不对称性指标。

如果目标是修正 FCC(111) vs HCP(0001)，更可能需要：

```text
density-modulated three-body
higher angular descriptor
ACE-like local density correlation
```

---

## 11. Force implementation 思路

dipole-modulated term 的 force 比 density term 更容易写错，因为：

\[
\mathbf{p}_i
=
\sum_j g(r_{ij})\hat{\mathbf{r}}_{ij}
\]

同时包含 radial dependence 和 direction dependence。

### 11.1 需要计算的中间量

建议在 force calculation 中保存：

```text
rho_i
drho_i / dr_ij
p_i
dp_i / dr_ij
p2_i = dot(p_i, p_i)
a2_i = p2_i / (rho_i + eps)^2
dE / da2_i
```

### 11.2 pair energy part

对于：

\[
E_{2D}
=
\sum_{i<j}
\frac{a_i^2+a_j^2}{2}
V_{2D}(r_{ij})
\]

direct pair radial force 包含：

\[
-\frac{a_i^2+a_j^2}{2}
\frac{dV_{2D}}{dr_{ij}}
\hat{\mathbf{r}}_{ij}
\]

除此以外还必须考虑 \(a_i^2\) 和 \(a_j^2\) 对原子位置的依赖。

### 11.3 chain-rule accumulation

建议在 pair loop 中先累计每个原子的：

\[
G_i^D =
\frac{\partial E_{2D}}{\partial a_i^2}
\]

由能量形式可得：

\[
G_i^D
=
\frac{1}{2}
\sum_j
V_{2D}(r_{ij})
\]

更一般地，如果使用 spline channel 或多个 \(V_{2D}^{(q)}\)，则对每个 channel 分别累计。

然后将：

\[
G_i^D
\frac{\partial a_i^2}{\partial \mathbf{r}}
\]

通过第二个 neighbor loop 加回力。

### 11.4 \(a_i^2\) 的导数

\[
a_i^2 =
\frac{p_i^2}{(\rho_i+\epsilon)^2}
\]

其中：

\[
p_i^2 = \mathbf{p}_i\cdot\mathbf{p}_i
\]

所以：

\[
d a_i^2
=
\frac{2\mathbf{p}_i\cdot d\mathbf{p}_i}{(\rho_i+\epsilon)^2}
-
\frac{2p_i^2}{(\rho_i+\epsilon)^3}
d\rho_i
\]

这里：

\[
d\rho_i
=
\sum_j f_\rho'(r_{ij})dr_{ij}
\]

而：

\[
d\mathbf{p}_i
=
d
\left[
\sum_j g(r_{ij})\hat{\mathbf{r}}_{ij}
\right]
\]

对某个 pair \(ij\)，其导数包含：

1. radial part：

\[
g'(r_{ij})\hat{\mathbf{r}}_{ij}\otimes\hat{\mathbf{r}}_{ij}
\]

2. angular projection part：

\[
\frac{g(r_{ij})}{r_{ij}}
\left[
I-
\hat{\mathbf{r}}_{ij}\otimes\hat{\mathbf{r}}_{ij}
\right]
\]

这一部分是最容易实现错误的地方。

---

## 12. 建议的计算流程

推荐 LAMMPS / ASE 计算流程如下。

### Step 1: neighbor loop 计算 scalar density 和 dipole vector

```text
for atoms i:
    rho_i = 0
    p_i = [0, 0, 0]

for pairs i-j:
    r_vec = r_j - r_i
    r = norm(r_vec)
    e = r_vec / r

    rho_i += f_rho(r)
    rho_j += f_rho(r)

    p_i += g(r) * e
    p_j -= g(r) * e
```

注意：如果 \(\hat{\mathbf{r}}_{ij}\) 定义为从 \(i\) 指向 \(j\)，则对 \(j\) 的贡献方向相反。

### Step 2: 计算 \(a_i^2\)

```text
for atoms i:
    p2_i = dot(p_i, p_i)
    a2_i = p2_i / (rho_i + eps)**2
```

### Step 3: pair loop 计算 energy、direct radial force、累计 \(G_i^D\)

```text
for pairs i-j:
    M_ij = 0.5 * (a2_i + a2_j)
    V, dV_dr = spline_V2D(r)

    E += M_ij * V

    direct_force += -M_ij * dV_dr * e

    G_i_D += 0.5 * V
    G_j_D += 0.5 * V
```

### Step 4: neighbor loop 加 chain-rule force

对每个 pair \(ij\)，计算该 pair 对 \(a_i^2\) 和 \(a_j^2\) 的影响，并用：

```text
G_i_D * da2_i/dr_ij
G_j_D * da2_j/dr_ij
```

加回力。

这一步需要非常仔细做 finite-difference force check。

---

## 13. Cutoff 和 radial functions

### 13.1 \(f_\rho(r)\) 和 \(g(r)\)

第一版建议令：

```text
g(r) = f_rho(r)
```

也就是 density 和 dipole 使用相同 radial weight，减少超参数。

后续如果需要，可允许：

```text
g(r) != f_rho(r)
```

但这会增加调参复杂度。

### 13.2 cutoff 平滑性

必须保证：

\[
g(r_c)=0
\]

\[
g'(r_c)=0
\]

最好二阶导数也平滑，尤其用于 MD。

同理：

\[
f_\rho(r_c)=0
\]

\[
f_\rho'(r_c)=0
\]

如果 cutoff 不平滑，dipole force 中的方向导数和 radial 导数都会放大数值问题。

---

## 14. 正则化策略

dipole-modulated term 应强正则化。建议：

\[
\lambda_{2D} > \lambda_2
\]

初始扫描：

```text
lambda_2D = 30  * lambda_2
lambda_2D = 100 * lambda_2
lambda_2D = 300 * lambda_2
```

如果 feature 已做标准化，也可以单独扫：

```text
lambda_2D = 1e-3
lambda_2D = 1e-2
lambda_2D = 1e-1
lambda_2D = 1
lambda_2D = 10
lambda_2D = 100
lambda_2D = 1000
```

不要一开始让 dipole 项过度自由。

原因：

- dipole 主要在少数不对称环境中激活；
- surface 数据通常相对少；
- 过弱正则化会导致 surface overfitting；
- high-T distorted / GB / liquid-like 环境可能被错误惩罚；
- 可能导致 MD unstable。

---

## 15. 训练 protocol

### 15.1 不建议直接 joint fit

不要一开始训练：

\[
E =
E_2 + E_3 + E_{2\rho} + E_{2D}
\]

并同时释放所有参数。

这会导致：

- \(E_{2D}\) 与 \(E_{2\rho}\) 竞争；
- \(E_{2D}\) 与普通 \(E_2\) 竞争；
- surface improvement 的来源难以判断；
- 训练结果不可解释。

### 15.2 推荐 residual fit

先训练或选择 baseline：

\[
E_{\rm base}=E_2+E_3
\]

或：

\[
E_{\rm base}=E_2+E_3+E_{2\rho}
\]

冻结 baseline，计算残差：

\[
y_{\rm res}=y_{\rm DFT}-y_{\rm base}
\]

再拟合：

\[
y_{\rm res}\approx X_{2D}\beta_{2D}
\]

这样可以判断：

> dipole-modulated two-body 是否能解释 baseline / density model 仍未解释的系统性残差。

### 15.3 如果 residual fit 有效，再做 prior joint fit

目标函数：

\[
\min_\beta
\|W(X\beta-y)\|^2
+
\lambda_{\rm base}
\|\beta_{\rm base}-\beta_{\rm base}^{(0)}\|^2
+
\lambda_{2D}
\|\beta_{2D}\|^2
\]

其中：

- \(\beta_{\rm base}^{(0)}\) 为 baseline 系数；
- \(\beta_{2D}\) 为 dipole-modulated two-body 系数；
- \(\lambda_{2D}\) 应较大。

---

## 16. 训练集注意事项

### 16.1 不能只用 surface 来训练 dipole 项

如果 large \(a_i^2\) 只出现在 surface 中，模型容易学到：

```text
large dipole = surface energy penalty
```

但真实 MD 中 large dipole 也可能来自：

- vacancy；
- rattled bulk；
- high-temperature thermal disorder；
- liquid-like local structure；
- grain boundary；
- dislocation core；
- compressed distorted environment；
- collision / indentation event；
- nanoparticle edge / corner。

因此训练集必须包括 **non-surface asymmetric environments**。

### 16.2 推荐加入的数据类型

建议至少包括：

```text
bulk rattled structures
strained bulk
vacancy
surface vacancy
step surface
grain boundary
high-T short MD snapshots
distorted slabs
small clusters
possibly dislocation core
```

这样 dipole 项不会把所有不对称性都错误解释成 surface penalty。

---

## 17. Validation panel

不要只看 total energy / force RMSE。应建立固定 benchmark panel。

### 17.1 Surface

检查：

```text
HCP(0001)
HCP(10-10)
HCP(11-20)
FCC(111)
FCC(100)
FCC(110)
step surface
surface vacancy
```

### 17.2 Defects

检查：

```text
bulk vacancy
surface vacancy
grain boundary
dislocation core
```

### 17.3 Non-surface asymmetric environments

检查：

```text
rattled bulk
high-T bulk
compressed distorted structures
liquid-like snapshots
```

### 17.4 Stability

检查：

```text
300 K bulk MD
1000 K bulk MD
slab relaxation
short indentation test
compressed pair / close contact test
```

---

## 18. Pass / fail criteria

### 18.1 通过条件

dipole-modulated two-body 可以认为有价值，如果满足若干条件：

- surface energy MAE 下降；
- open surface ranking 改善；
- vacancy formation energy 改善；
- surface vacancy energy 改善；
- slab force RMSE 改善；
- GB excess energy 不变坏或改善；
- non-surface asymmetric structures 不出现异常高能；
- bulk EOS / elastic constants 不明显变坏；
- short MD stability 保持。

### 18.2 失败条件

如果出现以下情况，应降低或暂时放弃该项：

- 只改善 surface，但 vacancy / GB / high-T distorted structures 明显变坏；
- 只有极窄 regularization 范围有效；
- force finite-difference check 不通过；
- MD 中出现异常吸引或异常排斥；
- bulk thermal vibration 中产生异常大 dipole penalty；
- 与 density-modulated two-body 高度共线，没有独立贡献；
- 对目标 surface ranking 没有稳定改善。

---

## 19. Feature sensitivity analysis

在训练前，可检查 \(X_{2D}\) 是否能区分目标性质。

对于两个结构 A 和 B：

\[
\Delta X_{2D}=X_{2D,A}-X_{2D,B}
\]

如果：

\[
\|\Delta X_{2D}\|\approx 0
\]

则 dipole-modulated two-body 很难修正 A 和 B 的能量差。

重点检查：

```text
FCC(100) vs FCC(110)
FCC(111) vs HCP(0001)
HCP(11-20) vs HCP(0001)
surface vacancy vs clean surface
vacancy vs bulk
GB vs bulk
rattled bulk vs surface
```

如果对 FCC(111) vs HCP(0001) 的 \(\Delta X_{2D}\) 很小，不应期待 dipole 项解决 close-packed stacking surface 排序。

---

## 20. Oracle test

建立小型 target-focused training set：

```text
low-index surfaces
vacancy
surface vacancy
step surface
GB
rattled bulk
high-T distorted structures
```

比较：

```text
baseline UF3
UF3 + E2rho
UF3 + E2D
UF3 + E2rho + E2D
```

如果 \(E_{2D}\) 在这个小数据集上都不能改善目标性质，则说明表达能力可能不足。

如果小数据集上有效，但完整训练集上无效，说明可能是：

- loss weight 不对；
- regularization 太强；
- force rows 压制 surface target；
- \(E_{2D}\) 与 \(E_{2\rho}\) 竞争；
- training data 缺少 non-surface asymmetric examples；
- explicit property targets 不足。

---

## 21. 与 ADP 的关系和预期

dipole-modulated two-body 与 ADP 的思想有相似性：都试图通过低阶局域方向矩描述环境不对称性。

因此需要谨慎预期：

> 它可能对 surface / vacancy / step 有帮助，但不应期待它单独解决 Co 的 HCP/FCC stacking 和 close-packed surface 排序问题。

如果 \(E_{2D}\) 改善 open surface，但 FCC(111) vs HCP(0001) 仍然错误，下一步更应优先尝试：

```text
density-modulated three-body
higher angular descriptors
ACE-like local density correlations
```

而不是继续堆更多 dipole powers。

---

## 22. 推荐实现 MVP

第一版只实现：

\[
E_{2D}
=
\sum_{i<j}
\frac{
a_i^2+a_j^2
}{2}
V_{2D}(r_{ij})
\]

其中：

\[
a_i^2 =
\frac{
\left|
\sum_j g(r_{ij})\hat{\mathbf{r}}_{ij}
\right|^2
}{
(\rho_i+\epsilon)^2
}
\]

推荐设置：

```text
g(r) = f_rho(r)
epsilon = small positive value, e.g. 1e-6 * rho_bulk or similar scale
one radial spline V2D(r)
one descriptor channel a2
strong regularization
residual fitting first
finite-difference force check required
```

---

## 23. Codex implementation checklist

### 23.1 Descriptor calculation

- [ ] compute `rho_i`
- [ ] compute vector `p_i`
- [ ] compute `p2_i = dot(p_i, p_i)`
- [ ] compute `a2_i = p2_i / (rho_i + eps)**2`
- [ ] support same or separate cutoff for `rho` and `p`
- [ ] ensure cutoff smoothing for `f_rho` and `g`

### 23.2 Feature construction

- [ ] build pair feature:

```text
0.5 * (a2_i + a2_j) * B_m(r_ij)
```

- [ ] assign feature group name:

```text
dipole_modulated_pair
```

- [ ] support enable / disable flag
- [ ] support group-wise regularization

### 23.3 Force implementation

- [ ] direct pair force from \(V_{2D}'(r)\)
- [ ] chain-rule force through \(a_i^2\)
- [ ] derivative through \(p_i\)
- [ ] derivative through \(\rho_i\) if normalized \(a_i^2\) is used
- [ ] virial / stress contribution
- [ ] finite-difference force test
- [ ] cutoff continuity test

### 23.4 Training modes

- [ ] residual fitting mode
- [ ] prior joint fitting mode
- [ ] group-wise regularization
- [ ] weighted feature standardization
- [ ] property panel report

### 23.5 Validation

- [ ] surface energy panel
- [ ] vacancy panel
- [ ] GB / asymmetric defect panel
- [ ] high-T distorted structures
- [ ] MD stability tests
- [ ] compare with density-modulated two-body baseline

---

## 24. 最核心建议

实践上，不要把 dipole-modulated two-body 当作第一核心多体修正。

推荐优先级是：

```text
1. density-modulated two-body
2. density-modulated three-body
3. dipole-modulated two-body
```

但 dipole-modulated two-body 很值得实现为一个低成本 correction，尤其用于测试：

> UF3 是否缺少对 surface / vacancy / step 等局域不对称环境的能量响应。

最推荐的第一版是：

\[
E_{2D}
=
\sum_{i<j}
\frac{
a_i^2+a_j^2
}{2}
V_{2D}(r_{ij})
\]

其中 \(a_i^2\) 使用 normalized dipole magnitude squared。

训练时必须：

- 强正则化；
- residual fit 优先；
- 加入 non-surface asymmetric structures；
- 做 feature sensitivity analysis；
- 做严格 finite-difference force check；
- 不要期待它单独解决 FCC(111) vs HCP(0001) 排序问题。
