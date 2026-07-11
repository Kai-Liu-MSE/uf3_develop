# 将 density-modulated two-body 作为真正 MLIP 项的训练思路

## 1. 背景

当前已经尝试了 density-modulated two-body，简称 `2rho`，其形式大致为：

\[
E_{2\rho}
=
\sum_{i<j}
\left[
\frac{
\phi(\rho_i)+\phi(\rho_j)
}{2}
\right]
V_{2\rho}(r_{ij})
\]

其中：

\[
\rho_i = \sum_j f_\rho(r_{ij})
\]

该项的目标是让 pair interaction 具有局域环境依赖性：

> 同一根 Co--Co bond 在 bulk、surface、vacancy、GB、high-temperature distorted environment 中可以具有不同有效强度。

---

## 2. 当前观察结果

### 2.1 Static surface oracle 表现很好

此前用 surface energy target 做 residual fit 时，`2rho` 可以几乎完美修正 baseline 的表面能误差。

例如某组模型中：

| model | abs MP MAE | rel DFT MAE |
|---|---:|---:|
| M0 baseline | 0.08259 J/m² | 0.03896 J/m² |
| 2rho residual fit | 0.00288 J/m² | 0.00105 J/m² |

这说明：

> `2rho` descriptor 本身具有足够的静态表面能表达能力。

也就是说，之前普通 residual training 没有明显改善，并不是因为 `2rho` 看不到 surface error，而是训练目标没有把它导向这件事。

---

### 2.2 Relaxation 后可能严重失败

但是 surface-only 或 force-weight 较弱的 residual fit 会出现严重问题。

例如某组 force-aware surface residual fit 中：

```text
slab force weight = 0.1
static absolute surface MAE = 0.0045 J/m²
static relative-delta MAE   = 0.0007 J/m²
slab force RMSE             = 0.0439 eV/Å
```

但 500-step FIRE relaxation 后，部分 open surfaces 出现巨大负表面能：

```text
HCP(11-20)  -110.2540 J/m²  converged=False
FCC(001)    -17.3235 J/m²   converged=False
FCC(011)    -27.1349 J/m²   converged=False
```

其中 HCP(11-20) slab 的 energy per atom 跑到：

```text
epa ≈ -13.13 eV/atom
```

远低于 bulk：

```text
bulk epa ≈ -7.05 eV/atom
```

这说明：

> 该模型不是正常 surface relaxation error，而是跑进了非物理低能坑。

---

## 3. 关键结论

当前结果说明两件事同时成立：

1. `2rho` 具有很强的静态能量表达能力；
2. 如果只用 surface energy / surface residual 训练，它很容易成为危险的表面能插值器，而不是真正稳定的 MLIP 项。

因此，如果目标是把 `2rho` 做成真正 MLIP 项，就不应继续把它当作 surface residual correction，而应该从一开始就把它放进完整 UF3 训练框架中。

---

## 4. 思路切换：从 residual correction 到 production MLIP architecture

不要再问：

> `2rho` 能不能拟合 surface residual？

这个问题已经被 static oracle 证明了。

现在应该问：

> `2rho` 能不能在完整 energy/force/stress 数据约束下，作为势函数本身的一部分，稳定改善 surface / defect / undercoordinated environment，同时不制造非物理 relaxation path？

因此训练方式应从：

```text
frozen baseline + 2rho residual fit
```

切换为：

```text
full joint training: E2 + E3 + E2rho
```

---

## 5. 推荐模型形式

### 5.1 保留原始 two-body

不建议用 `2rho` 完全替代普通 two-body。

推荐形式：

\[
E =
E_2^{(0)}
+
E_3^{(0)}
+
E_{2\rho}
\]

其中：

\[
E_2^{(0)}
=
\sum_{i<j}V_2^{(0)}(r_{ij})
\]

\[
E_3^{(0)}
=
\sum_i\sum_{j<k}V_3^{(0)}(r_{ij},r_{ik},r_{jk})
\]

\[
E_{2\rho}
=
\sum_{i<j}
\left[
\frac{
\phi_\alpha(\rho_i)+\phi_\alpha(\rho_j)
}{2}
\right]
V_{2\rho}^{(\alpha)}(r_{ij})
\]

普通 \(E_2^{(0)}\) 负责：

- short-range repulsion；
- density-independent radial baseline；
- low-density limit；
- compressed pair behavior；
- dimer-like behavior；
- basic radial bonding.

`2rho` 负责：

- environment-dependent radial correction；
- undercoordinated surface bonding；
- vacancy / GB / high-temperature distorted local environments；
- surface-vs-bulk effective bond strength change.

---

## 6. 不推荐让 2rho 成为 surface-only correction

此前 static oracle 的成功说明 `2rho` 可以很容易把 surface energy 点贴准。

但这会导致：

```text
E(R0) correct
gradient near R0 wrong
curvature near R0 wrong
relaxation path unstable
```

因此：

> surface energy target 可以作为辅助目标，但不能成为 `2rho` 的主要训练来源。

---

## 7. 训练目标设计

作为真正 MLIP 项，`2rho` 应该在完整训练目标中与 \(E_2\)、\(E_3\) 一起训练。

目标函数可写为：

\[
\min_\beta
\|W_E(X_E\beta-y_E)\|^2
+
\|W_F(X_F\beta-y_F)\|^2
+
\|W_\sigma(X_\sigma\beta-y_\sigma)\|^2
+
w_\gamma\|X_\gamma\beta-y_\gamma\|^2
+
R(\beta)
\]

其中：

\[
\beta =
\{\beta_2,\beta_3,\beta_{2\rho}\}
\]

\(X_\gamma\) 可以包括：

- absolute surface energy；
- surface energy relative to FCC(111)；
- surface energy differences；
- HCP/FCC energy difference；
- stacking fault energy；
- vacancy formation energy.

但主目标仍应是：

```text
DFT energy
DFT force
DFT stress
```

surface rows 只是引导 property，不应主导整个势能面。

---

## 8. 数据覆盖要求

如果 `2rho` 作为真正 MLIP 项，训练集必须覆盖它会激活的环境。

### 8.1 Bulk-like structures

至少包括：

```text
HCP equilibrium
FCC equilibrium
HCP/FCC EOS
compressed cells
expanded cells
strained HCP/FCC
rattled HCP/FCC
high-temperature bulk snapshots
```

目的：

- 约束普通 bulk energetics；
- 防止 `2rho` 破坏 EOS；
- 防止在热振动中产生异常力；
- 约束 density around bulk reference.

---

### 8.2 Surface structures

不能只加入 relaxed slabs。

至少包括：

```text
HCP(0001)
HCP(10-10)
HCP(11-20)
FCC(111)
FCC(001)
FCC(011)
```

每个 surface 建议包括：

```text
DFT/MP-relaxed slab
top-layer rattled slab
d12 compressed slab
d12 expanded slab
top-two-layer displacement
surface atoms random rattle
possibly reconstructed or slightly distorted slab
```

目的：

- 约束 surface energy；
- 约束 surface force；
- 约束 surface local curvature；
- 防止 relaxation 进入非物理低能坑。

---

### 8.3 Failed relaxation structures

当前已经观察到 `2rho` 会让这些 surface 崩溃：

```text
HCP(11-20)
FCC(001)
FCC(011)
```

这些 failed structures 很有价值。

建议把以下结构加入 training 或至少加入 validation：

```text
initial slab R0
failed FIRE final structure R1
trajectory midpoint structures
early-stage collapse structures
```

最好的做法是对这些结构做 DFT single-point energy + force，然后加入训练。

如果 DFT 资源有限，至少把它们作为 hard-fail validation set：

```text
if E_model_per_atom << E_bulk_per_atom: fail
if gamma < 0: fail
if min_pair_distance abnormal: fail
```

这不是 delicate hack，而是标准 active learning：

> 模型在哪里找到错误低能坑，就把那里加入训练覆盖。

---

### 8.4 Defect and asymmetric environments

建议包括：

```text
bulk vacancy
surface vacancy
step surface
grain boundary
dislocation core if available
small clusters
high-T surface snapshots
liquid-like or heavily distorted local structures
short-distance repulsive structures
```

原因：

`2rho` 会在 low-coordination / distorted environments 中激活。  
如果训练集中只有 ideal surfaces 具有 low density，那么模型会错误学习：

```text
low rho = surface correction
```

而不是一般性的 environment-dependent bonding。

---

## 9. 2rho regularization

### 9.1 不要使用 near-zero ridge 作为 production 设置

此前：

```text
ridge_2rho = 1e-8
```

可以作为 expression-capacity oracle，但不适合作为 production MLIP。

如果 `2rho` 是真正 MLIP 项，建议从更保守范围开始：

```text
lambda_2rho = 1  * lambda_2
lambda_2rho = 3  * lambda_2
lambda_2rho = 10 * lambda_2
lambda_2rho = 30 * lambda_2
```

如果 feature 已标准化，也可以单独扫：

```text
lambda_2rho = 1e-4
lambda_2rho = 1e-3
lambda_2rho = 1e-2
lambda_2rho = 1e-1
lambda_2rho = 1
lambda_2rho = 10
```

---

### 9.2 Smoothness regularization 应作为标准项

普通 ridge 只限制 coefficient magnitude，不直接限制 spline slope / curvature。

对 `2rho`，更重要的是限制：

\[
V_{2\rho}(r)
\]

的斜率和曲率。

建议正则化写成：

\[
R_{2\rho}
=
\lambda_0\|\beta_{2\rho}\|^2
+
\lambda_1\|D_1\beta_{2\rho}\|^2
+
\lambda_2\|D_2\beta_{2\rho}\|^2
\]

其中：

- \(D_1\)：一阶差分，限制 slope；
- \(D_2\)：二阶差分，限制 curvature。

这不是特殊补丁，而是 spline potential 中很自然的稳定化手段。

尤其当前已观察到 open surface relaxation 会跑进巨大负能坑，因此 derivative / curvature regularization 是必要的。

---

## 10. Density modulation function

设：

\[
x_i=\frac{\rho_i}{\rho_{\rm bulk}}
\]

可选 modulation basis：

### 10.1 Conservative correction basis

\[
\psi_\alpha(x)
=
x^\alpha - 1 - \alpha(x-1)
\]

特点：

\[
\psi_\alpha(1)=0
\]

\[
\psi_\alpha'(1)=0
\]

优点：

- 对 bulk reference 干扰较小；
- 更像 correction；
- 不容易重写普通 pair baseline。

缺点：

- 可能偏保守；
- surface 激活较弱；
- 需要 surface target 才容易发挥作用。

---

### 10.2 Production channel basis

\[
\phi_\alpha(x)=x^\alpha-1
\]

优点：

- 更像真正 density-dependent channel；
- 能在 bulk 附近提供一阶环境响应；
- 不完全依赖 surface-specific target。

缺点：

- 更容易和普通 two-body 竞争；
- 需要更强 group-wise regularization；
- 需要 force-rich data 约束。

如果目标是把 `2rho` 做成真正 MLIP 项，可以优先比较：

```text
basis = x^alpha - 1
basis = x^alpha - 1 - alpha * (x - 1)
```

其中：

```text
alpha = [0.5, 1.5]
```

不建议一开始加入太多 alpha。

---

### 10.3 不建议加入 rho^1 独立通道

因为：

\[
\sum_i \rho_i
=
\sum_i\sum_j f(r_{ij})
=
2\sum_{i<j}f(r_{ij})
\]

在 unary system 中，\(\rho^1\) 基本等价于 pair term，会和 \(E_2^{(0)}\) 强烈共线。

---

## 11. 模型选择不能只看 static surface energy

如果 `2rho` 是 MLIP 项，必须基于 relaxation 和 stability 选模型。

每个候选模型都应该跑 mini relaxation benchmark：

```text
six low-index surfaces
500-step FIRE
record converged / not converged
surface energy
energy per atom
min pair distance
max force
d12 / d23
```

Hard fail criteria：

```text
any gamma < 0: fail
any epa < bulk epa - 0.5 eV/atom: fail
any open surface non-converged with huge energy drop: fail
any min pair distance abnormal: fail
any surface reconstruction obviously unphysical: fail
```

不能选择 static surface MAE 最低的模型。

应选择满足：

```text
static surface energy acceptable
relaxed surface energy acceptable
all surfaces stable
bulk force/RMSE not degraded too much
short MD stable
```

的模型。

---

## 12. 推荐最小 production experiment

### 12.1 Baseline

```text
2body cutoff = 6.0
2body knots  = 12

3body cutoff = 4.25
3body knots  = [4, 4, 8]
```

### 12.2 Add 2rho

```text
2rho cutoff = 6.0
2rho alpha  = [0.5, 1.5]
```

测试两个 basis：

```text
basis A = x^alpha - 1
basis B = x^alpha - 1 - alpha * (x - 1)
```

### 12.3 Training target

包括：

```text
DFT energies
DFT forces
DFT stresses if available
MP absolute surface energy rows
DFT relative-to-FCC(111) surface difference rows
HCP/FCC energy difference
vacancy formation energy if available
surface perturbation structures
failed-relaxation structures if available
```

### 12.4 Regularization sweep

```text
lambda_2rho = 1  * lambda_2
lambda_2rho = 3  * lambda_2
lambda_2rho = 10 * lambda_2
lambda_2rho = 30 * lambda_2
```

Smoothness：

```text
D2_smooth_2rho = 0.01
D2_smooth_2rho = 0.1
D2_smooth_2rho = 1.0
```

Surface force / perturbation weights should be tested, but not used as the sole stabilizer.

---

## 13. Staged joint training

如果直接 joint training 不稳定，可以用 staged training，但最终必须 joint。

### Stage 1: baseline training

Train:

\[
E_2+E_3
\]

得到：

\[
\beta_2^{(0)}, \beta_3^{(0)}
\]

### Stage 2: joint fit with weak prior

Train:

\[
E_2+E_3+E_{2\rho}
\]

with weak prior:

\[
\|\beta_2-\beta_2^{(0)}\|^2
+
\|\beta_3-\beta_3^{(0)}\|^2
\]

This prevents an immediate unstable redistribution of contributions.

### Stage 3: relax prior

Reduce prior strength and allow \(E_2\), \(E_3\), and \(E_{2\rho}\) to repartition.

This ensures `2rho` is not merely a residual patch, but a real part of the final model.

---

## 14. What if full production training still collapses?

If `2rho` still causes open surfaces to collapse even under full energy/force training, strong regularization, and broader data coverage, then the likely conclusion is:

> density-dependent radial correction is too prone to fixing surface energy by creating excessive low-coordination pair attraction.

This would imply that the missing physics is not just:

```text
density-dependent bond strength
```

but rather:

```text
density-dependent angular / stacking response
```

In that case, the next model extension should be:

\[
E_{3\rho}
=
\sum_i\sum_{j<k}
\phi(\rho_i)
V_{3\rho}(r_{ij},r_{ik},r_{jk})
\]

That is, density-modulated three-body.

This gives the model a more structure-selective correction direction and may avoid the overly radial collapse channel.

---

## 15. Recommended immediate next steps

### Step 1: Do not use static oracle as production candidate

Treat previous `ridge_2rho=1e-8` and `sf0p1` models as diagnostic only.

### Step 2: Add 2rho to full joint training

Use:

\[
E=E_2+E_3+E_{2\rho}
\]

with energy + force + surface property rows.

### Step 3: Add surface perturbation and failed-relaxation structures

At minimum, include failed structures in validation.  
Preferably label them with DFT single-point energy and force.

### Step 4: Use group-wise and smoothness regularization

Do not rely on coefficient ridge alone.

### Step 5: Select models by relaxation stability

Every candidate should pass six-surface FIRE relaxation benchmark.

### Step 6: If 2rho remains unstable, move to 3rho

Do not continue to over-engineer 2rho if the radial correction channel remains intrinsically unstable.

---

## 16. Core conclusion

If the aim is to make `2rho` a true MLIP term, the correct approach is:

> Put it into the full energy/force/stress training from the beginning, constrain it with broad low-coordination and distorted environments, regularize its slope/curvature, and select models by relaxed-surface stability rather than static surface accuracy.

In one sentence:

> `2rho` should be trained as an environment-dependent radial channel in a full force-rich MLIP dataset, not as a surface-energy residual interpolator.

