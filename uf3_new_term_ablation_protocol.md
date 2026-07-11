# UF3 新多体项开发中的可判定训练逻辑

## 1. 背景问题

在 UF3 势函数开发中，当前存在一个非常棘手的问题：

> UF3 本身已有大量超参数，包括 cutoff、knot 数量、regularization、energy/force/stress 权重、训练集权重等。  
> 当加入新的多体项后，如果预测效果相差不大，很难判断到底是：
>
> 1. 新项本身没有用；
> 2. 新项有用，但正则化没有调好；
> 3. 新项有用，但被原始 UF3 的 pair / three-body 项吸收或掩盖；
> 4. 新项有用，但训练目标没有充分暴露 surface / stacking 的误差；
> 5. 新项改善了目标性质，但被整体 RMSE 指标淹没；
> 6. joint fitting 时各类项重新分工，导致结果不可解释。

因此，不能一开始就把问题设定为：

> 加入新项后，最终模型是否比最优 UF3 更好？

这个问题太耦合，几乎不可判定。

更合理的问题是：

> 在固定 baseline 和固定训练设置下，新项是否能稳定解释 UF3 的特定系统性残差？

这份文档的目的，是给 UF3 新项开发建立一个更清晰的训练与模型选择 protocol。

---

## 2. 核心原则

### 2.1 将“判断新项是否有用”和“寻找最终最优模型”分开

建议区分两种模式：

| 模式 | 目的 | 是否调 UF3 原始超参数 |
|---|---|---|
| diagnostic mode | 判断新项有没有独立解释能力 | 不调 |
| production mode | 在确认新项有用后寻找最终模型 | 可以调 |

### 2.2 先做可判定实验，再做全局超参数优化

新项开发应该遵循：

```text
fixed baseline UF3
        ↓
residual fitting
        ↓
property-specific validation
        ↓
joint refit with prior
        ↓
limited hyperparameter tuning
        ↓
production-level tuning
```

而不是：

```text
add new term
        ↓
tune everything
        ↓
compare global RMSE
        ↓
unclear conclusion
```

---

## 3. 当前候选新项

当前主要考虑三个方向：

### 3.1 density-modulated two-body

形式示意：

$$
E_{2\rho}
=
\sum_{i<j}
\frac{
\psi(\rho_i)+\psi(\rho_j)
}{2}
V_{2\rho}(r_{ij})
$$

其中：

$$
\rho_i = \sum_j f_\rho(r_{ij})
$$

该项表达：

> pair bond 的有效强度依赖于局域 density / coordination。

预期主要帮助：

- open surface；
- vacancy；
- undercoordinated environment；
- surface vs bulk energy difference。

对 FCC(111) vs HCP(0001) 这类 close-packed stacking 差异，可能帮助有限。

---

### 3.2 density-modulated three-body

形式示意：

$$
E_{3\rho}
=
\sum_i
\sum_{j<k}
\psi(\rho_i)
V_{3\rho}(r_{ij}, r_{ik}, r_{jk})
$$

或更完整地：

$$
E_i^{(3)}
=
E_{3,i}^{(0)}
+
\psi_{1/2}(\rho_i)E_{3,i}^{(1)}
+
\psi_{3/2}(\rho_i)E_{3,i}^{(2)}
$$

该项表达：

> 三体角度相互作用可以随局域 density / coordination 改变。

预期主要帮助：

- surface angular environment；
- HCP/FCC stacking-sensitive environments；
- FCC(111) vs HCP(0001)；
- stacking fault；
- coherent HCP/FCC interface。

成本明显高于 density-modulated two-body，但仍保持三体复杂度，不引入四体循环。

---

### 3.3 dipole-modulated two-body

定义局域 dipole-like asymmetry：

$$
\mathbf{p}_i = \sum_j g(r_{ij})\hat{\mathbf{r}}_{ij}
$$

推荐使用归一化不对称性：

$$
a_i^2=
\frac{
\mathbf{p}_i\cdot \mathbf{p}_i
}{
(\rho_i+\epsilon)^2
}
$$

对应 pair correction：

$$
E_{2D}
=
\sum_{i<j}
\frac{
a_i^2+a_j^2
}{2}
V_{2D}(r_{ij})
$$

该项表达：

> pair bond 的有效强度依赖于局域环境是否对称。

预期主要帮助：

- surface；
- vacancy；
- step；
- grain boundary；
- asymmetric local environments。

注意：该项不应作为 density 项的替代，而是 asymmetry correction。它对 FCC(111) vs HCP(0001) 可能帮助有限。

---

## 4. 推荐的 density modulation function

不要直接使用：

$$
\rho^{1/2},\quad \rho^1,\quad \rho^{3/2}
$$

尤其不建议直接加入 \(\rho^1\)，因为：

$$
\sum_i \rho_i
=
\sum_i \sum_j f(r_{ij})
=
2\sum_{i<j}f(r_{ij})
$$

在 unary system 中，它基本等价于 pair term，会和原始 UF3 two-body 项强烈竞争。

推荐先归一化：

$$
x_i = \frac{\rho_i}{\rho_{\rm bulk}}
$$

再使用去掉常数项和线性项的形式：

$$
\psi_\alpha(x_i)
=
x_i^\alpha - 1 - \alpha(x_i-1)
$$

这样有：

$$
\psi_\alpha(1)=0
$$

$$
\psi_\alpha'(1)=0
$$

其好处是：

> 在 bulk reference 附近，新项不会轻易重写原来的 pair potential 或 EOS，主要在 surface / vacancy / GB 等 density 明显偏离 bulk 的环境中起作用。

推荐第一批 exponent：

```text
alpha = 1/2
alpha = 3/2
```

可选：

```text
alpha = 2
```

但不建议一开始开太多 channel。

---

## 5. Diagnostic Mode: 判断新项是否有用

### 5.1 固定 baseline UF3

先选择一个或少数几个 baseline UF3 势函数：

| Baseline | 选择理由 |
|---|---|
| UF3-A | 当前综合表现最好 |
| UF3-B | surface 表现相对最好 |
| UF3-C | MD 稳定性最好 |

不要在 diagnostic mode 中重新搜索所有 UF3 超参数。

固定：

- two-body cutoff；
- three-body cutoff；
- knot grid；
- data split；
- energy / force / stress weights；
- baseline regularization；
- training set；
- validation set。

---

### 5.2 计算 baseline residual

对于训练目标：

$$
y_{\rm DFT}
$$

baseline 预测为：

$$
y_{\rm base} = X_{\rm base}\beta_{\rm base}
$$

残差为：

$$
y_{\rm res} = y_{\rm DFT} - y_{\rm base}
$$

然后只用新项拟合残差：

$$
y_{\rm res}
\approx
X_{\rm new}\beta_{\rm new}
$$

其中：

```text
X_new = X_2rho
X_new = X_3rho
X_new = X_2D
```

分别测试。

---

### 5.3 residual fitting 的目的

Residual fitting 的目的不是得到最终可用势函数，而是回答：

> 新项是否能解释 baseline UF3 无法解释的系统性误差？

如果 residual fit 都不能改善目标性质，则该项可能没有足够信息量，或者该项无法区分目标结构。

如果 residual fit 能明显改善目标性质，则说明该项有真实价值，值得进入 joint fitting。

---

### 5.4 正则化扫描

对每个新项单独扫自己的正则化强度：

$$
\lambda_{\rm new}
$$

例如：

```text
lambda_new = 1e-4
lambda_new = 1e-3
lambda_new = 1e-2
lambda_new = 1e-1
lambda_new = 1
lambda_new = 10
lambda_new = 100
```

前提：所有 feature columns 应先做标准化，否则不同 lambda 不可比较。

关注点不是某一个最低 RMSE 点，而是是否存在稳定窗口：

> 在一段 lambda 范围内，目标性质都得到改善。

如果只有一个极窄 lambda 点改善，可能是偶然过拟合。

---

## 6. Feature Sensitivity Analysis

在训练前，可以先判断新项是否“看得见”目标差异。

对于两个结构或性质差：

$$
\Delta y = y_A - y_B
$$

对应 feature difference：

$$
\Delta X = X_A - X_B
$$

检查：

$$
\|\Delta X_{\rm new}\|
$$

如果对于某个关键差异：

```text
Delta X_new ≈ 0
```

说明该新项几乎无法区分这两个结构。

---

### 6.1 例子：FCC(111) vs HCP(0001)

需要检查：

```text
Delta X_2rho for FCC(111) - HCP(0001)
Delta X_2D   for FCC(111) - HCP(0001)
Delta X_3rho for FCC(111) - HCP(0001)
```

可能出现的情况：

| 新项 | 结果解释 |
|---|---|
| Delta X_2rho 很小 | density-modulated two-body 难以修正 close-packed stacking surface |
| Delta X_2D 很小 | dipole 对 FCC(111) vs HCP(0001) 不敏感 |
| Delta X_3rho 明显非零 | density-modulated three-body 更有希望 |

这一步不依赖训练结果，可以直接判断 descriptor 的分辨能力。

---

## 7. Oracle Test: 判断表达能力上限

建立一个很小的 target-focused training set，只包含：

- HCP/FCC bulk；
- HCP/FCC stacking-related structures；
- low-index surfaces；
- vacancy；
- distorted surface；
- stacking fault；
- HCP/FCC coherent interface。

然后分别训练：

```text
baseline UF3
UF3 + E_2rho
UF3 + E_2D
UF3 + E_3rho
```

### 7.1 Oracle test 的解释

如果在这个小训练集上，新项仍不能把 surface ranking 或 stacking energy fit 对，则说明该项表达能力可能不足。

如果在小训练集上可以 fit 对，但放回完整训练集后又错，则说明问题不是项本身无用，而可能是：

- loss weight 不对；
- regularization 太强；
- bulk/force 数据压制 surface 目标；
- 新旧项之间竞争；
- training set 中存在冲突；
- surface difference target 没有被显式加入。

Oracle test 不是为了得到可用势函数，而是为了测试模型形式的上限。

---

## 8. 建立固定 benchmark panel

不要只看整体 train/test RMSE。应建立一个固定的 `Co_surface_stacking_panel`。

每个模型都必须输出同一组性质。

---

### 8.1 Panel A: bulk phase

建议包括：

- HCP equilibrium energy；
- FCC equilibrium energy；
- BCC / SC / high-energy reference；
- HCP/FCC EOS；
- HCP/FCC elastic strain structures；
- HCP/FCC energy difference：

$$
E_{\rm FCC}-E_{\rm HCP}
$$

---

### 8.2 Panel B: stacking

建议包括：

- intrinsic stacking fault；
- twin fault；
- HCP/FCC coherent interface；
- HCP to FCC stacking path；
- mixed ABC / ABAB stacking supercells。

---

### 8.3 Panel C: surface

建议包括：

- HCP(0001)；
- HCP\((10\bar{1}0)\)；
- HCP\((11\bar{2}0)\)；
- FCC(111)；
- FCC(100)；
- FCC(110)。

对每个 surface，建议至少保留：

```text
DFT-relaxed slab geometry
ML-relaxed slab geometry
DFT-relaxed geometry single-point by ML
surface-layer distorted / rattled variants
```

---

### 8.4 Panel D: undercoordinated / asymmetric environments

建议包括：

- vacancy；
- surface vacancy；
- step surface；
- small cluster；
- grain boundary；
- dislocation core；
- high-temperature distorted surface；
- liquid-like or highly distorted short-MD snapshots。

这对 dipole-modulated 项尤其重要。否则模型可能错误学习：

```text
large dipole = surface penalty
```

但实际 MD 中 large dipole 也可能来自 non-surface asymmetric environments。

---

## 9. 使用 property-specific validation metrics

每个模型应输出 property table。

示例：

```text
model_id
E_fcc_minus_hcp
SFE
gamma_hcp_0001
gamma_hcp_10m10
gamma_hcp_11m20
gamma_fcc_111
gamma_fcc_100
gamma_fcc_110
gamma_fcc100_minus_fcc110
gamma_fcc111_minus_hcp0001
vacancy_formation_energy
slab_force_RMSE
bulk_force_RMSE
highT_MD_stable
```

重点不是普通 RMSE，而是：

- surface energy MAE；
- surface ranking；
- HCP/FCC energy difference；
- stacking fault energy；
- vacancy formation energy；
- slab relaxation force；
- MD stability。

---

## 10. 将 surface energy / difference target 显式加入训练

表面能是差分量：

$$
\gamma_s
=
\frac{
E_{{\rm slab},s} - N_sE_{\rm bulk}
}{
2A_s
}
$$

对于线性势函数：

$$
E = X\beta
$$

因此：

$$
\gamma_s = X_{\gamma_s}\beta
$$

其中：

$$
X_{\gamma_s}
=
\frac{
X_{{\rm slab},s} - N_s X_{\rm bulk}
}{
2A_s
}
$$

可以直接把：

$$
X_{\gamma_s}\beta \approx \gamma_s^{\rm DFT}
$$

作为训练行加入。

---

### 10.1 加入 surface difference target

可以进一步加入：

$$
\gamma_{\rm FCC(100)}-\gamma_{\rm FCC(110)}
$$

$$
\gamma_{\rm FCC(111)}-\gamma_{\rm HCP(0001)}
$$

$$
\gamma_{\rm HCP(11\bar{2}0)}-\gamma_{\rm HCP(0001)}
$$

这些也是线性目标。

对应：

$$
X_{\Delta \gamma}
=
X_{\gamma_A}-X_{\gamma_B}
$$

训练目标：

$$
X_{\Delta \gamma}\beta
\approx
\Delta \gamma^{\rm DFT}
$$

这比仅靠 slab 总能量间接学习 surface ranking 更稳定。

---

## 11. Joint Refit: 只有 residual test 通过后才做

如果 residual fitting 显示新项有用，再进入 joint refit。

不要直接完全释放所有系数。建议对 baseline 系数加 prior：

$$
\min_\beta
\|W(X\beta-y)\|^2
+
\lambda_{\rm base}
\|\beta_{\rm base}-\beta_{\rm base}^{(0)}\|^2
+
\lambda_{\rm new}
\|\beta_{\rm new}\|^2
$$

其中：

- \(\beta_{\rm base}^{(0)}\) 是 baseline UF3 的系数；
- \(\beta_{\rm new}\) 是新项系数；
- \(\lambda_{\rm base}\) 控制 baseline 允许偏离多少；
- \(\lambda_{\rm new}\) 控制新项强度。

这样可以避免新旧项完全重新分工，导致结果不可解释。

---

## 12. Group-wise regularization

不要用一个统一 regularization 控制所有项。

建议：

$$
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
+
\lambda_{2D}\|\beta_{2D}\|^2
$$

不同 group 应分开控制。

建议初始关系：

```text
lambda_2rho  > lambda_2
lambda_3rho  > lambda_3
lambda_2D    > lambda_2
```

例如：

```text
lambda_2rho = 10, 30, 100 times lambda_2
lambda_3rho = 10, 30, 100 times lambda_3
lambda_2D   = 30, 100, 300 times lambda_2
```

dipole-modulated 项尤其需要强正则化，因为它容易过拟合 surface / defect asymmetry。

---

## 13. Feature standardization

在比较不同 regularization 之前，必须标准化 feature columns。

建议使用 weighted standardization：

$$
\tilde{X}_k
=
\frac{X_k-\mu_k}{s_k}
$$

其中 \(\mu_k\) 和 \(s_k\) 应根据训练权重计算。

原因：

- energy rows 和 force rows 数量差别很大；
- 不同 feature 的自然尺度不同；
- 未标准化时，lambda 对不同 feature 没有可比性；
- 新项可能因为尺度问题被过度惩罚或过度释放。

---

## 14. 通过 / 失败标准

每个新项都应有明确判定标准，避免凭感觉判断。

---

### 14.1 density-modulated two-body 通过标准

该项主要应改善 coordination-dependent radial bonding。

重点检查：

- FCC(100) 表面能异常偏低是否改善；
- FCC(100) vs FCC(110) 排序是否恢复；
- open surface energy MAE 是否下降；
- vacancy formation energy 是否改善；
- slab force RMSE 是否改善；
- bulk EOS / elastic constants 不明显变坏；
- short MD stability 不明显变坏。

如果它只改善一个窄性质但破坏 bulk/MD，则不通过。

---

### 14.2 density-modulated three-body 通过标准

该项主要应改善 density-dependent angular / stacking response。

重点检查：

- FCC(111) vs HCP(0001) 是否改善；
- stacking fault energy 是否改善；
- HCP/FCC coherent interface 是否改善；
- HCP/FCC energy difference 是否不变坏；
- open surface 不明显变坏；
- MD stability 保持。

---

### 14.3 dipole-modulated two-body 通过标准

该项主要应改善 asymmetry-dependent environments。

重点检查：

- surface vs bulk residual；
- vacancy；
- surface vacancy；
- step surface；
- grain boundary；
- non-surface asymmetric environments 不被错误惩罚；
- high-T distorted structures 不出现异常高能或不稳定力。

---

## 15. 分层超参数搜索

不要进行全组合爆炸式搜索。

---

### Level 0: Diagnostic

固定 baseline UF3，只改：

- 是否加入新项；
- 新项 channel 数；
- 新项正则化；
- 是否加入 surface difference target。

不改：

- cutoff；
- knot grid；
- base UF3 regularization；
- training set split。

---

### Level 1: Limited joint tuning

在新项通过 Level 0 后，允许调：

- \(\lambda_2\)；
- \(\lambda_3\)；
- \(\lambda_{\rm new}\)；
- surface target weight；
- energy / force balance 的小范围变化。

仍不建议大幅改变 cutoff 和 knot grid。

---

### Level 2: Structural tuning

只有当 Level 1 证明方向有效后，再调：

- density cutoff；
- density radial function；
- density channel 数；
- modulated pair knot grid；
- modulated three-body knot grid；
- base UF3 knot grid；
- base UF3 cutoff。

---

## 16. 推荐的模型开发矩阵

建议每个模型只回答一个问题。

| Model ID | Base UF3 | New term | Fit mode | Joint refit | Surface target | 目的 |
|---|---|---|---|---|---|---|
| M0 | A | none | full | no | no | baseline |
| M1 | A | E2rho | residual | no | no | 判断 E2rho 是否有信息量 |
| M2 | A | E2rho | residual | no | yes | 判断 surface target 是否必要 |
| M3 | A | E2rho | prior joint | yes | yes | 可用候选 |
| M4 | A | E2D | residual | no | yes | 判断 dipole correction |
| M5 | A | E3rho | residual | no | yes | 判断 stacking-sensitive correction |
| M6 | A | E2rho + E2D | prior joint | yes | yes | 判断 density + dipole 是否互补 |
| M7 | A | E2rho + E3rho | prior joint | yes | yes | 判断 two-body + three-body density modulation |

---

## 17. 推荐开发顺序

### Step 1: 固定 baseline UF3

选择 1--3 个 baseline，不要一开始全局调参。

### Step 2: 建立 benchmark panel

尤其是 Co surface / stacking / vacancy / asymmetric structures。

### Step 3: 加入 explicit property rows

包括：

- surface energy；
- surface energy difference；
- HCP/FCC energy difference；
- stacking fault energy；
- vacancy formation energy。

### Step 4: residual fit 新项

分别测试：

```text
E2rho
E2D
E3rho
```

### Step 5: 做 feature sensitivity analysis

尤其检查：

```text
FCC(100) vs FCC(110)
FCC(111) vs HCP(0001)
HCP(11-20) vs HCP(0001)
vacancy vs bulk
surface vacancy vs clean surface
```

### Step 6: 做 oracle test

判断新项表达能力上限。

### Step 7: 只有通过 residual / oracle test 的项进入 joint refit

并使用 baseline prior。

### Step 8: 最后才做 production tuning

包括 cutoff、knot grid、full regularization sweep。

---

## 18. Codex 实现建议

Codex 在修改训练代码时，可以优先实现以下模块。

---

### 18.1 Feature group registry

每一类 feature 都应有明确 group name：

```python
feature_groups = {
    "pair": ...,
    "three_body": ...,
    "density_modulated_pair": ...,
    "density_modulated_three_body": ...,
    "dipole_modulated_pair": ...,
}
```

每个 group 需要记录：

```text
column indices
regularization strength
standardization scale
optional smoothing operator
whether included in current model
```

---

### 18.2 Residual fitting mode

增加模式：

```python
fit_mode = "residual"
```

功能：

1. 读取 frozen baseline coefficients；
2. 计算 baseline prediction；
3. 构造 residual target；
4. 只拟合指定新项 group；
5. 输出 property panel。

---

### 18.3 Prior joint fitting mode

增加模式：

```python
fit_mode = "prior_joint"
```

目标函数：

$$
\|W(X\beta-y)\|^2
+
\lambda_{\rm base}
\|\beta_{\rm base}-\beta_{\rm base}^{(0)}\|^2
+
\lambda_{\rm new}
\|\beta_{\rm new}\|^2
$$

---

### 18.4 Property row builder

实现函数：

```python
build_energy_difference_row(structure_A, structure_B)
build_surface_energy_row(slab, bulk, n_atoms, area)
build_surface_difference_row(surface_A, surface_B)
build_stacking_fault_row(...)
build_vacancy_formation_row(...)
```

所有这些 row 都应返回：

```python
X_property
y_property
weight_property
property_name
```

---

### 18.5 Benchmark report generator

每次训练后自动输出：

```text
model_id
training_config
feature_groups_enabled
regularization_values
energy_RMSE
force_RMSE
stress_RMSE
property_panel
surface_ranking
pass_fail_flags
```

建议输出为：

```text
results/model_id/property_report.csv
results/model_id/property_report.md
results/model_id/training_config.yaml
```

---

## 19. 最核心结论

当前不要把目标设定为：

> 加入新项后直接得到最优 Co UF3 势函数。

而应该先设定为：

> 新项是否在固定 baseline 下稳定解释 UF3 的特定系统性残差？

推荐优先级：

```text
1. density-modulated two-body
2. density-modulated three-body
3. dipole-modulated two-body
```

判断新项是否有用时，应优先使用：

```text
residual fitting
feature sensitivity analysis
oracle test
property-specific benchmark panel
explicit surface / stacking difference targets
```

只有当新项在这些 controlled tests 中表现出稳定收益，才进入 full joint training 和 production hyperparameter search。

这样可以避免陷入：

```text
训练没搞好？
还是新项没用？
还是指标不敏感？
还是正则化不对？
```

的不可判定状态。
