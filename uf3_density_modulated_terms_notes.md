# UF3 density-modulated two-body / three-body extension notes

本文档整理当前关于在 UF3 中加入 `density-modulated two-body` 与 `density-modulated three-body` 的设计讨论，目标是给代码修改提供一个清晰的实现参考。

核心背景：

- 标准 UF3 主要是二体项 + 三体项：
  \[
  E = E_2 + E_3
  \]
- 当前观察到：对于 Co 的 HCP/FCC 表面能排序，qSNAP/ACE 可以较好复现 DFT，而 UF3 会出现与 ADP/EAM 类似的系统性错误。
- 一个可能原因是：标准 UF3 中同一个 pair 或 triangle 的能量贡献基本不依赖局域环境，导致 bulk、surface、vacancy、GB 等不同配位环境下的键强度缺少环境依赖。
- 因此考虑加入 low-cost many-body correction，使 pair 或 triangle 的有效贡献依赖于局域 density。

---

## 1. 基本 density 定义

对每个原子 \(i\)，定义一个 scalar local density：

\[
\rho_i = \sum_{j \ne i} f_\rho(r_{ij})
\]

其中 \(f_\rho(r)\) 是一个平滑截断的径向函数。

建议要求：

1. \(f_\rho(r)\) 与 UF3 pair cutoff 一致或略短。
2. 在 cutoff 处函数值和一阶导数连续。
3. 对单元素 Co，可以先只做一个 density channel。
4. 后续多元素扩展时，density 可以按元素类型分 channel：
   \[
   \rho_i^{(s)} = \sum_{j \in s} f_{\rho,s}(r_{ij})
   \]

### 推荐归一化

定义 bulk reference density：

\[
\rho_{\mathrm{bulk}}
\]

然后：

\[
x_i = \frac{\rho_i}{\rho_{\mathrm{bulk}}}
\]

这里 \(\rho_{\mathrm{bulk}}\) 可以取 0 K equilibrium HCP Co bulk 中的平均 \(\rho_i\)，或者从训练集中 bulk-like structures 的平均值确定。

---

## 2. Density modulation functions

不要直接使用 \(\rho_i^1\)。因为：

\[
\sum_i \rho_i = \sum_i \sum_j f_\rho(r_{ij}) = 2\sum_{i<j} f_\rho(r_{ij})
\]

所以 \(\rho^1\) 项严格等价于一个 pair-like correction，容易和 UF3 原始二体项抢权重。

建议使用去除常数项和线性项的非线性 density functions：

\[
\psi_\alpha(x_i) = x_i^\alpha - 1 - \alpha(x_i - 1)
\]

这样：

\[
\psi_\alpha(1) = 0
\]

\[
\psi_\alpha'(1) = 0
\]

好处：

- 在 bulk reference 附近不轻易重写原始 pair/EOS。
- 主要在 surface、vacancy、GB、liquid-like 或 strongly distorted environments 中起作用。
- 有利于把新项作为 correction，而不是让它替代原始 UF3 二体/三体项。

建议第一版只使用：

\[
\alpha = \frac{1}{2}
\]

如果有效，再加入：

\[
\alpha = \frac{3}{2}
\]

最多再考虑：

\[
\alpha = 2
\]

---

## 3. Density-modulated two-body

### 3.1 物理意义

density-modulated two-body 的目标是表达：

> 同一根 pair bond 的有效强度依赖于两端原子的局域 density / coordination。

这比单纯 \(F(\rho_i)\) embedding 更强，因为它同时知道：

- bond length \(r_{ij}\)
- atom \(i\) 的 density
- atom \(j\) 的 density

但它仍然是 angle-independent 的 many-body correction。

它尤其可能改善：

- surface energy
- vacancy formation energy
- open surfaces such as FCC(100), FCC(110), HCP prism surfaces
- undercoordinated environments
- grain-boundary environments

但它可能不足以完全解决：

- FCC(111) vs HCP(0001) 这种 close-packed stacking 差异
- stacking fault energetics
- angular/stacking-dependent effects

---

### 3.2 推荐 symmetric pair form

推荐先实现 pair-symmetric 形式：

\[
E^{(2\rho)}
=
\sum_{i<j}
\left[
V_2^{(0)}(r_{ij})
+
\sum_{\alpha}
\frac{
\psi_\alpha(x_i)+\psi_\alpha(x_j)
}{2}
V_{2\rho}^{(\alpha)}(r_{ij})
\right]
\]

其中：

- \(V_2^{(0)}(r)\) 是原始 UF3 二体项，或者保留现有 UF3 二体实现。
- \(V_{2\rho}^{(\alpha)}(r)\) 是 density-modulated two-body correction 的径向 spline。
- \(\psi_\alpha(x_i)\) 是 density modulation function。

如果只加一个 channel：

\[
E^{(2\rho)}
=
\sum_{i<j}
\frac{
\psi_{1/2}(x_i)+\psi_{1/2}(x_j)
}{2}
V_{2\rho}^{(1/2)}(r_{ij})
\]

总能量：

\[
E = E_2 + E_3 + E^{(2\rho)}
\]

---

### 3.3 复杂度

density-modulated two-body 是 pair-like 成本：

1. 计算 \(\rho_i\)：\(O(Nz)\)
2. 计算 modulated pair energy/force：\(O(Nz)\)
3. 计算 density chain-rule force：\(O(Nz)\)

因此总复杂度仍然是：

\[
O(Nz)
\]

相对于 UF3 原始三体项 \(O(Nz^2)\)，这个项应当比较便宜。

粗略预期：

| density channels | 额外成本 |
|---:|---:|
| 1 | 约 5--15% |
| 2 | 约 10--30% |
| 3+ | 约 20--50% |

具体取决于 LAMMPS 实现、邻居表复用、缓存和 spline 插值方式。

---

### 3.4 Force implementation sketch

对 pair-symmetric form：

\[
E_{ij}^{(2\rho)} =
M_{ij} V(r_{ij})
\]

其中：

\[
M_{ij} =
\frac{\psi(x_i)+\psi(x_j)}{2}
\]

力有两类贡献：

#### A. Direct pair-distance contribution

来自 \(V(r_{ij})\) 对距离的导数：

\[
\frac{\partial E_{ij}}{\partial r_{ij}}
=
M_{ij} V'(r_{ij})
\]

这部分和普通 pair force 类似，只是乘了 modulation factor \(M_{ij}\)。

#### B. Density chain-rule contribution

因为 \(M_{ij}\) 依赖 \(\rho_i,\rho_j\)，而 \(\rho_i\) 又依赖周围 pair distances：

\[
\frac{\partial E}{\partial \rho_i}
\]

需要先累计每个原子的 density adjoint：

\[
G_i = \frac{\partial E}{\partial \rho_i}
\]

对于一个 pair \(ij\) 的 modulated pair correction：

\[
G_i \mathrel{+}= \frac{1}{2} \psi'(x_i)\frac{1}{\rho_{\mathrm{bulk}}} V(r_{ij})
\]

\[
G_j \mathrel{+}= \frac{1}{2} \psi'(x_j)\frac{1}{\rho_{\mathrm{bulk}}} V(r_{ij})
\]

然后通过 density definition：

\[
\rho_i = \sum_j f_\rho(r_{ij})
\]

把 density force 加回 pair：

\[
\mathbf{F}_{ij}^{\rho}
\sim
-
\left[
G_i + G_j
\right]
f_\rho'(r_{ij}) \hat{\mathbf{r}}_{ij}
\]

注意实际符号需要与代码中的 force convention 保持一致。

---

### 3.5 Recommended MVP for two-body modulation

第一版最小模型：

\[
E
=
E_{\mathrm{UF3}}
+
\sum_{i<j}
\frac{
\psi_{1/2}(x_i)+\psi_{1/2}(x_j)
}{2}
V_{2\rho}^{(1/2)}(r_{ij})
\]

其中：

\[
\psi_{1/2}(x) = x^{1/2} - 1 - \frac{1}{2}(x-1)
\]

不要一开始加太多 density channels。

建议开发顺序：

1. 实现 \(\rho_i\) 计算。
2. 实现 \(\psi_{1/2}(x_i)\) 与 \(\psi'_{1/2}(x_i)\)。
3. 实现 modulated two-body energy。
4. 实现 direct pair force。
5. 实现 density chain-rule force。
6. 用 finite difference 检查 force。
7. 加入训练矩阵 feature 生成。
8. 用 residual fit 测试是否改善 surface energy ranking。

---

## 4. Density-modulated three-body

### 4.1 物理意义

density-modulated three-body 的目标是表达：

> 同一个三体角度/triangle contribution 在 bulk 和 surface 等不同 density 环境下具有不同有效强度。

标准 UF3 中：

\[
E_i^{(3)}
=
\sum_{j<k}
V_3(r_{ij},r_{ik},r_{jk})
\]

其中同一个 triangle 的贡献基本不依赖中心原子整体环境。

Density-modulated three-body 允许：

\[
V_3^{\mathrm{eff}}
=
V_3^{(0)}
+
\psi(\rho_i) V_3^{(1)}
+
\cdots
\]

这比 density-modulated two-body 更有可能修正：

- angular-dependent surface energetics
- HCP/FCC local stacking difference
- stacking fault energetics
- coherent HCP/FCC interface
- GB angular environments

但计算成本也更高。

---

### 4.2 Cheap scalar modulation form

最便宜的版本是：

\[
E_i^{(3\rho)}
=
h(\rho_i)
\sum_{j<k}
V_3(r_{ij},r_{ik},r_{jk})
\]

即：

\[
E_i^{(3\rho)}
=
h(\rho_i) E_{3,i}
\]

这个版本额外成本较低：

- 需要先算 \(\rho_i\)
- 三体循环中多乘一个 scalar \(h(\rho_i)\)
- force 中多一个 density chain-rule correction

但表达能力有限。它只能调制整个三体能量的整体强弱，不能让不同三体 basis 在不同 density 下独立变化。

适合作为 sanity check：

> density modulation of total three-body energy 是否能改变 surface ranking？

---

### 4.3 Recommended channel form

更推荐的真正 density-conditioned three-body 形式：

\[
E_i^{(3)}
=
E_{3,i}^{(0)}
+
\sum_\alpha
\psi_\alpha(x_i)
E_{3,i}^{(\alpha)}
\]

其中：

\[
E_{3,i}^{(\alpha)}
=
\sum_{j<k}
V_3^{(\alpha)}(r_{ij},r_{ik},r_{jk})
\]

展开后：

\[
E_i^{(3)}
=
\sum_{j<k}
\left[
V_3^{(0)}(r_{ij},r_{ik},r_{jk})
+
\sum_\alpha
\psi_\alpha(x_i)
V_3^{(\alpha)}(r_{ij},r_{ik},r_{jk})
\right]
\]

第一版建议：

\[
E_i^{(3)}
=
E_{3,i}^{(0)}
+
\psi_{1/2}(x_i)E_{3,i}^{(1/2)}
+
\psi_{3/2}(x_i)E_{3,i}^{(3/2)}
\]

总能量：

\[
E = E_2 + E_3^{(0)} + E_{3\rho}
\]

---

### 4.4 Complexity

标准三体 UF3 的复杂度：

\[
O(Nz^2)
\]

density-modulated three-body 仍然是：

\[
O(Nz^2)
\]

不会改变阶数，但会增加三体循环内的常数因子。

粗略预期：

| Implementation | Expressiveness | Speed impact |
|---|---:|---:|
| \(h(\rho_i) E_{3,i}\) | medium | +10--40% |
| 1 density channel with separate \(V_3^{(\alpha)}\) | high | ~1.3--2x |
| 2 density channels | higher | ~1.5--3x |
| 3+ density channels | high but risky | ~2--4x |
| only modulate selected 3-body basis | medium-high | +30--100% |

注意：

- 如果三体 spline basis values 可以共享，只是不同 channel 使用不同 coefficients，则成本可以降低。
- 如果每个 channel 都独立做完整 spline 查表/插值，则成本接近线性增加。
- 可以让 \(V_3^{(\alpha)}\) 使用更少 knots 或只对筛选 basis 开启 modulation。

---

### 4.5 Force implementation sketch

对 atom-centered form：

\[
E_i^{(3\rho)}
=
\sum_{j<k}
\psi(x_i)
V_3(r_{ij},r_{ik},r_{jk})
\]

力也分两类。

#### A. Direct three-body force

普通 UF3 三体力乘以 \(\psi(x_i)\)：

\[
\psi(x_i)
\frac{\partial V_3}{\partial \mathbf{r}}
\]

#### B. Density chain-rule force

三体循环中累计：

\[
G_i = \frac{\partial E}{\partial \rho_i}
\]

对于中心原子 \(i\)：

\[
G_i \mathrel{+}=
\psi'(x_i)
\frac{1}{\rho_{\mathrm{bulk}}}
V_3(r_{ij},r_{ik},r_{jk})
\]

如果采用多个 channels：

\[
G_i \mathrel{+}=
\sum_\alpha
\psi'_\alpha(x_i)
\frac{1}{\rho_{\mathrm{bulk}}}
V_3^{(\alpha)}(r_{ij},r_{ik},r_{jk})
\]

最后通过 density pair loop 加回 density-chain-rule force：

\[
\mathbf{F}_{ij}^{\rho}
\sim
-
\left[
G_i + G_j
\right]
f_\rho'(r_{ij})\hat{\mathbf{r}}_{ij}
\]

如果只用 atom-centered central density，某些实现可能只需要中心原子 \(G_i\)，但由于每个 atom 都会作为不同 center 出现，最后 pair loop 中仍然自然得到两端贡献。

---

### 4.6 Recommended MVP for three-body modulation

建议不要直接做 full version。先做两个层级：

#### MVP 3B-1: scalar modulation of existing three-body energy

\[
E_i
=
E_{2,i}
+
E_{3,i}^{(0)}
+
c_1 \psi_{1/2}(x_i) E_{3,i}^{(0)}
\]

这个版本只测试：density modulation of total three-body energy 是否有用。

优点：

- 成本低
- 实现简单
- 可以快速验证方向

缺点：

- 表达能力有限
- 不是严格独立的新三体 channel

#### MVP 3B-2: one independent density-modulated three-body channel

\[
E_i
=
E_{2,i}
+
E_{3,i}^{(0)}
+
\psi_{1/2}(x_i) E_{3,i}^{(1/2)}
\]

其中 \(E_{3,i}^{(1/2)}\) 是独立拟合的三体 spline channel。

如果这版有效，再加入 \(\psi_{3/2}\)。

---

## 5. Training matrix / linear regression

这些扩展都可以保持线性回归结构。

标准 UF3：

\[
E = X_2 \beta_2 + X_3 \beta_3
\]

加入 density-modulated two-body：

\[
E =
X_2 \beta_2
+
X_3 \beta_3
+
X_{2\rho} \beta_{2\rho}
\]

加入 density-modulated three-body：

\[
E =
X_2 \beta_2
+
X_3 \beta_3
+
X_{2\rho} \beta_{2\rho}
+
X_{3\rho} \beta_{3\rho}
\]

其中新 features 是普通 UF3 basis 乘上 density modulation factor。

### 5.1 Feature examples

Two-body modulated feature：

\[
X_{2\rho,m}^{(\alpha)}
=
\sum_{i<j}
\frac{
\psi_\alpha(x_i)+\psi_\alpha(x_j)
}{2}
B_m^{(2)}(r_{ij})
\]

Three-body modulated feature：

\[
X_{3\rho,m}^{(\alpha)}
=
\sum_i
\sum_{j<k}
\psi_\alpha(x_i)
B_m^{(3)}(r_{ij},r_{ik},r_{jk})
\]

---

## 6. Regularization strategy

不要把所有 feature 用同一个 ridge penalty 直接混合训练。建议 group-wise regularization：

\[
\min_\beta
\|W(X\beta-y)\|^2
+
\lambda_2\|L_2\beta_2\|^2
+
\lambda_3\|L_3\beta_3\|^2
+
\lambda_{2\rho}\|\beta_{2\rho}\|^2
+
\lambda_{3\rho}\|\beta_{3\rho}\|^2
\]

建议：

\[
\lambda_{2\rho} > \lambda_2
\]

\[
\lambda_{3\rho} > \lambda_3
\]

初始可以尝试：

\[
\lambda_{2\rho} = 10\lambda_2,\ 30\lambda_2,\ 100\lambda_2
\]

\[
\lambda_{3\rho} = 10\lambda_3,\ 30\lambda_3,\ 100\lambda_3
\]

### 6.1 Why stronger regularization?

新项应该是 correction，而不是重写原来的 pair/three-body terms。

尤其是 density-modulated two-body，如果训练集中大多数结构都是 bulk-like，则：

\[
\psi(x_i) B_m(r_{ij})
\]

可能和普通 pair basis 强相关。强正则化和 \(\psi(1)=\psi'(1)=0\) 可以降低这种抢权重问题。

---

## 7. Residual fitting workflow

推荐先做 residual fit，而不是一开始 joint training。

### Step 1: Fit baseline UF3

\[
E_{\mathrm{base}} = E_2 + E_3
\]

保存 baseline coefficients：

\[
\beta_2^{(0)}, \beta_3^{(0)}
\]

### Step 2: Compute residual

\[
y_{\mathrm{res}} = y_{\mathrm{DFT}} - y_{\mathrm{base}}
\]

### Step 3: Fit new terms only

\[
y_{\mathrm{res}}
\approx
X_{2\rho}\beta_{2\rho}
\]

or:

\[
y_{\mathrm{res}}
\approx
X_{2\rho}\beta_{2\rho}
+
X_{3\rho}\beta_{3\rho}
\]

This tests whether the new terms explain systematic residuals.

### Step 4: Joint refit with prior to baseline

Optional final step:

\[
\min_\beta
\|W(X\beta-y)\|^2
+
\lambda_{base}\|\beta_{2,3}-\beta_{2,3}^{(0)}\|^2
+
\lambda_{2\rho}\|\beta_{2\rho}\|^2
+
\lambda_{3\rho}\|\beta_{3\rho}\|^2
\]

This keeps the original UF3 model as the baseline and lets new terms correct only what is needed.

---

## 8. Explicit property constraints

Since surface energy is a small difference of large energies, ordinary energy/force fitting may not emphasize it enough.

Surface energy:

\[
\gamma =
\frac{
E_{\mathrm{slab}} - N E_{\mathrm{bulk}}
}{2A}
\]

Since \(E = X\beta\), surface energy is also a linear target:

\[
\gamma =
\frac{
X_{\mathrm{slab}}\beta - N X_{\mathrm{bulk}}\beta
}{2A}
\]

Define:

\[
X_\gamma =
\frac{
X_{\mathrm{slab}} - N X_{\mathrm{bulk}}
}{2A}
\]

Then add rows to the regression:

\[
X_\gamma\beta \approx \gamma_{\mathrm{DFT}}
\]

Similarly, one can add linear constraints/targets for:

\[
E_{\mathrm{FCC}} - E_{\mathrm{HCP}}
\]

\[
\gamma_{\mathrm{FCC(111)}} - \gamma_{\mathrm{HCP(0001)}}
\]

\[
\gamma_{\mathrm{FCC(100)}} - \gamma_{\mathrm{FCC(110)}}
\]

\[
\gamma_{\mathrm{FCC(100)}} - \gamma_{\mathrm{HCP(11\bar{2}0)}}
\]

This is especially useful for linear UF3-type models.

---

## 9. Validation targets

Do not select hyperparameters only by random train/test RMSE. Use property-driven validation.

### 9.1 Bulk

- HCP vs FCC energy difference
- equation of state
- elastic constants
- strained HCP/FCC
- phonon-like rattled bulk structures

### 9.2 Surface

- HCP(0001)
- HCP(10\(\bar{1}\)0)
- HCP(11\(\bar{2}\)0)
- FCC(111)
- FCC(100)
- FCC(110)

Check both:

- absolute surface energy
- relative ranking

### 9.3 Defect / stacking

- intrinsic stacking fault
- twin fault
- HCP/FCC coherent interface
- vacancy formation energy

### 9.4 MD stability

- 300 K bulk
- high-temperature bulk
- relaxed slabs
- compressed short-distance configurations
- small nanoindentation stability test

---

## 10. Recommended implementation order

### Stage 1: density infrastructure

- [ ] Implement density calculation \(\rho_i\).
- [ ] Store \(\rho_i\), \(x_i\), \(\psi_\alpha(x_i)\), \(\psi'_\alpha(x_i)\).
- [ ] Implement finite-difference force test for density-only embedding term \(F(\rho_i)\) if useful.

### Stage 2: density-modulated two-body

- [ ] Implement one-channel \(2\rho\) correction:
  \[
  \psi_{1/2}(x) V_{2\rho}^{(1/2)}(r)
  \]
- [ ] Add direct pair force.
- [ ] Add density chain-rule force through \(G_i = \partial E/\partial \rho_i\).
- [ ] Add feature generation for fitting.
- [ ] Fit residual of baseline UF3.
- [ ] Validate surface energy ranking.

### Stage 3: two-channel density-modulated two-body

- [ ] Add \(\psi_{3/2}(x)\) channel if Stage 2 helps.
- [ ] Use stronger regularization for second channel.

### Stage 4: cheap density-modulated three-body sanity check

- [ ] Implement:
  \[
  E_i^{3\rho} = c_1 \psi_{1/2}(x_i) E_{3,i}^{(0)}
  \]
- [ ] Use it as a quick test of whether density modulation of angular energy matters.

### Stage 5: independent density-modulated three-body channel

- [ ] Implement:
  \[
  E_i^{3\rho} = \psi_{1/2}(x_i) E_{3,i}^{(1/2)}
  \]
- [ ] Reuse existing three-body basis values if possible.
- [ ] Apply stronger regularization than standard \(E_3\).
- [ ] Consider fewer knots or selected basis subset for \(E_{3,i}^{(1/2)}\).

---

## 11. Expected outcome for Co surface problem

Expected likely effects:

| Extension | Expected benefit | Limitation |
|---|---|---|
| \(F(\rho_i)\) embedding | helps undercoordination | no bond-length resolution |
| density-modulated two-body | likely helps FCC(100), FCC(110), prism surfaces | angle/stacking blind |
| density-modulated three-body | more likely helps HCP/FCC close-packed surface differences | higher cost |
| selected-basis \(3\rho\) | good compromise | needs feature sensitivity analysis |

For current Co problem:

- density-modulated two-body is probably the best first low-cost attempt.
- If FCC(100) surface energy is too low, \(2\rho\) may help significantly.
- If FCC(111) vs HCP(0001) remains wrong, this likely indicates a need for density-modulated three-body or higher angular descriptors.
- The minimal recommended sequence is:

\[
\mathrm{UF3}
\rightarrow
\mathrm{UF3} + 2\rho
\rightarrow
\mathrm{UF3} + 2\rho + 3\rho
\]

---

## 12. Notes for code review

When modifying code, please check:

1. Energy-force consistency by finite difference on small structures.
2. Invariance:
   - translation
   - rotation
   - permutation of identical atoms
3. Pair symmetry for two-body modulation.
4. Cutoff continuity for:
   - \(f_\rho(r)\)
   - \(V_{2\rho}(r)\)
   - \(V_{3\rho}\)
5. No accidental double counting of pair interactions.
6. Correct MPI ghost atom communication for \(\rho_i\), \(\psi_i\), and \(G_i\).
7. Stress/virial contribution includes both direct force and density-chain-rule force.
8. New feature columns are standardized or regularized group-wise in fitting.
9. \(\rho_{\mathrm{bulk}}\) is stored consistently in the potential file.
10. The potential file format remains backward compatible where possible.

---

## 13. Minimal equations summary

### Density

\[
\rho_i = \sum_j f_\rho(r_{ij})
\]

\[
x_i = \frac{\rho_i}{\rho_{\mathrm{bulk}}}
\]

\[
\psi_\alpha(x_i) = x_i^\alpha - 1 - \alpha(x_i - 1)
\]

### Density-modulated two-body

\[
E^{(2\rho)}
=
\sum_{i<j}
\sum_\alpha
\frac{
\psi_\alpha(x_i)+\psi_\alpha(x_j)
}{2}
V_{2\rho}^{(\alpha)}(r_{ij})
\]

### Density-modulated three-body

\[
E^{(3\rho)}
=
\sum_i
\sum_{j<k}
\sum_\alpha
\psi_\alpha(x_i)
V_{3\rho}^{(\alpha)}(r_{ij}, r_{ik}, r_{jk})
\]

### Total model

\[
E
=
E_2
+
E_3
+
E^{(2\rho)}
+
E^{(3\rho)}
\]

---

## 14. Practical first experiment

First experiment should be intentionally small:

\[
E
=
E_{\mathrm{UF3}}
+
\sum_{i<j}
\frac{
\psi_{1/2}(x_i)+\psi_{1/2}(x_j)
}{2}
V_{2\rho}^{(1/2)}(r_{ij})
\]

Training:

- Fit baseline UF3.
- Freeze or strongly regularize baseline.
- Fit \(2\rho\) residual.
- Add explicit surface-energy linear target rows.
- Sweep:
  \[
  \lambda_{2\rho} = 10\lambda_2,\ 30\lambda_2,\ 100\lambda_2
  \]
- Validate Co surface energy ranking.

Success criterion:

- FCC(100) no longer anomalously lower than FCC(110) and HCP(11\(\bar{2}\)0).
- FCC(111) vs HCP(0001) improves, but full correction may require \(3\rho\).
- Bulk HCP/FCC energy difference and elastic constants are not degraded.
- Short MD remains stable.
