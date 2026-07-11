# 快速测试 UF3+2rho 中 feature standardization 与正则化参数影响的实验方案

## 1. 目标

本文件用于指导 Codex 实现一个快速、受控的消融实验，用来回答：

1. 在含有 `2rho` 的 UF3 模型中，**feature standardization / column scaling** 是否会显著改变训练结果？
2. 修改 `2rho` 的 ridge regularization 和 smoothness regularization 是否能改善 relaxed surface behavior？
3. 如果这些训练侧操作影响很小，是否说明 `2rho` 已经接近能力边界，应转向 `3rho`？

本实验应满足：

```text
不新增训练数据
不改变构型权重
不改变 energy / force / stress 总体权重
不改变势函数运行时形式
不增加 MD 计算成本
```

重点是测试训练矩阵处理和正则化方式，而不是做 surface-specific tuning。

---

## 2. 当前背景

已有模型：

```text
M0 = baseline UF3 = E2 + E3
M1 = UF3 + conservative 2rho = E2 + E3 + E2rho
```

其中 `2rho` 形式为：

```math
E_{2rho} = sum_{i<j} [(phi_alpha(x_i) + phi_alpha(x_j))/2] V_{2rho}^{alpha}(r_ij)
```

```math
x_i = rho_i / rho_bulk
```

当前较好的 `2rho` basis 为 conservative basis：

```math
phi_alpha(x) = x^alpha - 1 - alpha*(x - 1)
```

当前使用：

```text
alpha = [0.5, 1.5]
basis = conservative
```

已有现象：

1. `2rho` 在 static surface 上明显改善 UF3；
2. joint training 后所有 surface relaxation 收敛；
3. 但 relaxed surface energy 整体偏低；
4. relaxed surface ranking 仍未完全纠正；
5. static result 比 relaxed result 好，说明主要问题在 surface relaxation PES / force / curvature，而不是 static energy 表达能力。

因此本实验重点不是继续拟合 static surface energy，而是测试：

```text
standardization / regularization 是否能改善 relaxed surface behavior
```

---

## 3. 总体实验逻辑

实验分三阶段：

```text
Phase 1: feature standardization sensitivity
Phase 2: relax top candidates
Phase 3: smoothness regularization sweep
```

不要一开始做大规模全搜索。第一轮只做少量 regression，先用 static + force diagnostics 筛选，再对 top candidates 做 surface relaxation。

---

## 4. Feature standardization 的定义

第一版只做 **column RMS scaling**，不要做 mean-centering。

### 4.1 不要做 mean-centering

不建议第一版使用：

```math
X_tilde_j = (X_j - mu_j) / s_j
```

原因：

- UF3 训练矩阵同时包含 energy rows 和 force rows；
- force rows 是 feature 对坐标的导数；
- mean-centering 需要额外处理 energy offset；
- 容易引入 bookkeeping 错误；
- 目前只需要测试 column scale 对 regularization 的影响。

### 4.2 推荐使用 weighted column RMS scaling

对训练矩阵每一列 `j`，定义：

```math
s_j = sqrt( sum_i w_i^2 X_ij^2 / sum_i w_i^2 )
```

其中：

- `X_ij` 是训练矩阵元素；
- `w_i` 是该 row 的训练权重；
- sum over all training rows, including energy, force, stress/property rows if present.

然后：

```math
X_tilde_ij = X_ij / s_j
```

训练得到 `beta_tilde_j` 后，转换回原始系数：

```math
beta_j = beta_tilde_j / s_j
```

这样最终导出的 potential 仍然使用原始 feature 和原始 spline coefficients。

> Column scaling 只改变训练数值条件和 regularization 的实际含义，不增加 MD 计算成本。

### 4.3 避免除以零

如果某列几乎全零，设置：

```python
if s_j < eps:
    s_j = 1.0
```

建议：

```text
eps = 1e-12 or 1e-14
```

并记录这些 near-zero columns。

---

## 5. Scaling 应该对哪些 columns 做？

建议先对所有 active feature columns 做 scaling：

```text
E2 columns
E3 columns
E2rho columns
```

如果训练矩阵里还有其他 group，也按 group 记录，但 scaling 仍然是 per-column。

需要保存：

```text
column_scale[j] = s_j
feature_group[j] = E2 / E3 / E2rho / ...
```

方便后续分析不同 feature group 的 coefficient norm 和 contribution。

---

## 6. 正则化矩阵处理

原始 ridge 问题可写为：

```math
min_beta || W (X beta - y) ||^2 + R(beta)
```

其中：

```math
X = [X2, X3, X2rho]
```

推荐使用 group-wise regularization：

```math
R(beta) = lambda_2 R_2(beta_2) + lambda_3 R_3(beta_3) + lambda_2rho R_2rho(beta_2rho)
```

### 6.1 Ridge 部分

最简单：

```math
R_g^ridge = || beta_g ||^2
```

### 6.2 Smoothness 部分

对 spline coefficients 增加二阶差分惩罚：

```math
R_g^smooth = || D2 beta_g ||^2
```

对 `2rho`：

```math
R_2rho = r_2rho || beta_2rho ||^2 + c_2rho || D2 beta_2rho ||^2
```

这里：

- `r_2rho` 是 ridge；
- `c_2rho` 是 curvature / D2 smoothness penalty。

如果现有代码已经有 `r` 和 `c` 参数，则保持命名一致。

### 6.3 Scaling 和 regularization 的关系

如果在 scaled feature space 中训练：

```math
X_tilde = X S^{-1}
```

其中：

```math
S = diag(s_j)
```

训练变量为 `beta_tilde`，预测为：

```math
X_tilde beta_tilde = X beta
```

```math
beta = S^{-1} beta_tilde
```

最简单实现方式：

1. 先把 `X` 的 columns 除以 `s_j`；
2. 在 scaled coefficient `beta_tilde` 上直接施加 group-wise ridge/smoothness；
3. 解出 `beta_tilde`；
4. 导出前转换为原始 coefficient：

```math
beta_j = beta_tilde_j / s_j
```

注意：

- smoothness penalty 如果直接对 `beta_tilde` 施加，其含义是“scaled feature space 的 coefficient smoothness”；
- 这正是本实验要测试的 regularization 方式；
- 第一版不要过度复杂化为原始 coefficient space 的等价 penalty。

---

## 7. Phase 1: feature standardization sensitivity

### 7.1 固定条件

保持以下完全不变：

```text
same dataset
same train/validation split
same energy weight
same force weight
same stress weight if available
same property rows if already used
same surface property target if already used
same knots/cutoffs
same basis
```

模型形式固定为：

```text
E = E2 + E3 + E2rho
```

2rho 设置固定为：

```text
basis = conservative
alpha = [0.5, 1.5]
c_2rho = 0.001
```

### 7.2 扫描参数

测试：

```text
feature_scaling = off / on
r_2rho = [0.001, 0.003, 0.01, 0.03, 0.1]
c_2rho = 0.001
```

总共：

```text
2 × 5 = 10 regressions
```

### 7.3 每个模型先不要全做 relaxation

Phase 1 每个模型只输出 static + force diagnostics：

```text
energy RMSE
force RMSE
stress RMSE if available
static surface abs MAE vs MP
static surface relative-ratio MAE vs DFT
static surface rel-delta MAE J/m2
initial slab force RMSE
group contribution statistics
```

---

## 8. Phase 1 输出表格

建议输出一个 CSV / markdown table：

```text
model_id
feature_scaling
basis
r_2rho
c_2rho
energy_RMSE
force_RMSE
stress_RMSE
static_abs_MAE_MP
static_relratio_MAE_DFT
static_reldelta_MAE_Jm2
initial_slab_force_RMSE
E2rho_surface_RMS
E2rho_train_RMS
coef_norm_E2
coef_norm_E3
coef_norm_E2rho
scaled_coef_norm_E2rho
num_near_zero_columns
```

如果某些指标暂时无法实现，至少保留：

```text
model_id
feature_scaling
r_2rho
c_2rho
energy_RMSE
force_RMSE
static_abs_MAE_MP
static_reldelta_MAE_Jm2
initial_slab_force_RMSE
E2rho_surface_RMS
```

---

## 9. Group contribution diagnostics

为了判断 feature scaling 是否改变了不同项的分工，建议输出 group contribution。

### 9.1 Training set group RMS

对每个 structure 计算：

```text
E2 contribution per atom
E3 contribution per atom
E2rho contribution per atom
```

然后计算 RMS 或 std：

```math
RMS_g = sqrt( mean( (E_g/N)^2 ) )
```

建议输出：

```text
E2_train_RMS
E3_train_RMS
E2rho_train_RMS
```

### 9.2 Surface excess group contribution

对 surface energy，分解：

```math
gamma_g = (E_g,slab - N_slab E_g,bulk) / (2A)
```

输出每个 surface 的：

```text
gamma_E2
gamma_E3
gamma_E2rho
```

并计算：

```text
E2rho_surface_RMS = RMS over six surfaces of gamma_E2rho
```

这对判断 2rho 是否被压死或过度激活非常重要。

---

## 10. Phase 1 判断标准

### 10.1 Standardization 有明显影响

如果 scaling on/off 之间出现：

```text
static rel-delta MAE 改变 > 0.02 J/m2
或 static abs MAE 改变 > 0.03 J/m2
或 force RMSE 改变 > 5–10%
或 E2rho_surface_RMS 改变超过 2 倍
```

则说明 feature scaling 对结果有显著影响。

后续训练应默认：

```text
feature_scaling = on
```

### 10.2 Standardization 影响小

如果 scaling on/off 之间：

```text
static surface MAE 差 < 0.01 J/m2
force RMSE 基本不变
E2rho contribution 基本不变
```

则说明 standardization 不是当前主要变量。可以优先测试 smoothness 或转向 3rho。

---

## 11. Phase 2: relax top candidates

Phase 1 不要所有模型都做 relaxation。从 10 个模型中选择 3–4 个 top candidates。

### 11.1 候选选择标准

选择以下几类：

```text
1. static surface rel-delta MAE 最好
2. force RMSE 最好且 surface 还不错
3. E2rho_surface_RMS 适中
4. 当前 reference model，例如 conservative r=0.01 c=0.001
```

避免选择：

```text
force RMSE 明显变差
E2rho contribution 异常大
static surface 已明显差
```

### 11.2 对候选模型做 six-surface FIRE relaxation

对每个候选模型跑：

```text
HCP(0001)
HCP(10-10)
HCP(11-20)
FCC(001)
FCC(011)
FCC(111)
```

每个：

```text
500-step FIRE
record converged / not converged
record relaxed surface energy
record final energy per atom
record max force
record min pair distance
record d12 / d23 if available
```

---

## 12. Phase 2 输出表格

输出 summary：

```text
model_id
feature_scaling
r_2rho
c_2rho
all_converged
static_abs_MAE_MP
relaxed_abs_MAE_MP
static_reldelta_MAE_Jm2
relaxed_reldelta_MAE_Jm2
mean_relax_drop
max_relax_drop
FCC111_minus_HCP0001_static
FCC111_minus_HCP0001_relaxed
FCC001_minus_FCC011_static
FCC001_minus_FCC011_relaxed
min_pair_distance_min
max_force_final_max
```

其中：

```math
Delta gamma_relax = gamma_relaxed - gamma_static
```

```text
mean_relax_drop = average over surfaces of Delta gamma_relax
max_relax_drop = most negative Delta gamma_relax
```

---

## 13. 排序相关指标

重点输出两个差值。

### 13.1 FCC(111) vs HCP(0001)

```math
Delta gamma_111_0001 = gamma_FCC111 - gamma_HCP0001
```

目标 MP / DFT-like 排序要求：

```text
Delta(FCC111 - HCP0001) < 0
```

因为 FCC(111) 应低于 HCP(0001)。

### 13.2 FCC(001) vs FCC(011)

```math
Delta gamma_001_011 = gamma_FCC001 - gamma_FCC011
```

目标排序要求：

```text
Delta(FCC001 - FCC011) > 0
```

因为 FCC(001) 应高于 FCC(011)。

这些差值需要分别输出：

```text
static
relaxed
```

因为当前已经观察到：

```text
static 排序可能正确
relaxed 后排序可能又翻转
```

---

## 14. Phase 3: smoothness regularization sweep

如果 Phase 1/2 显示 scaling 有影响，先固定：

```text
feature_scaling = on
```

选择 Phase 2 中最好的一个或两个 `r_2rho`，然后扫 `c_2rho`。

### 14.1 推荐 sweep

```text
c_2rho = [0, 0.0003, 0.001, 0.003, 0.01]
```

保持：

```text
basis = conservative
alpha = [0.5, 1.5]
same dataset
same weights
same r_2rho
```

总共 5 或 10 个 regression。

### 14.2 观察重点

Smoothness 主要应该影响：

```text
surface relaxation drop
initial slab force
relaxed surface energy
surface ranking after relaxation
```

不一定明显改善 static surface energy。

重点看：

```text
mean_relax_drop
max_relax_drop
FCC111_minus_HCP0001_relaxed
FCC001_minus_FCC011_relaxed
relaxed_rel_delta_MAE
force_RMSE
```

### 14.3 预期判断

如果提高 `c_2rho` 后：

```text
static surface MAE 只稍微变差
relaxed surface drop 明显变小
relaxed ranking 更好
force RMSE 不明显变差
```

说明 2rho 的主要问题是 spline curvature / local force 太软，smoothness 很关键。

如果提高 `c_2rho` 后：

```text
static 和 relaxed 都变差
surface ranking 没改善
force RMSE 也没改善
```

说明 smoothness 不是主要瓶颈，2rho 可能接近能力边界。

---

## 15. 推荐最小执行顺序

### Step 1

运行 Phase 1 的 10 个 regression：

```text
feature_scaling = off/on
r_2rho = [0.001, 0.003, 0.01, 0.03, 0.1]
c_2rho = 0.001
```

只输出 static + force diagnostics。

### Step 2

从 10 个模型中选 3–4 个做 surface relaxation。

### Step 3

如果 scaling 有影响，则后续默认 scaling on。

### Step 4

固定 scaling on 和较好的 `r_2rho`，做 Phase 3 smoothness sweep：

```text
c_2rho = [0, 0.0003, 0.001, 0.003, 0.01]
```

### Step 5

只对 smoothness sweep 中的 top candidates 做 relaxation。

---

## 16. 快速判定逻辑

### Case A: feature scaling 明显改善

如果：

```text
scaling on
relaxed rel-delta MAE 明显下降
surface ranking 更接近目标
force RMSE 不变坏
```

结论：

```text
之前的 2rho 正则化强度受 feature scale 干扰较大。
后续训练应默认使用 column RMS scaling。
```

### Case B: feature scaling 影响小，但 smoothness 有效

如果：

```text
scaling on/off 差不多
但提高 c_2rho 后 relaxation drop 变小
```

结论：

```text
2rho 的主要问题是 PES curvature / force softness。
后续应保留 D2 smoothness regularization。
```

### Case C: scaling 和 smoothness 都影响小

如果：

```text
static 可改善
relaxed 排序仍不行
surface drop 仍大
```

结论：

```text
2rho 可能已接近能力边界。
下一步应转向 density-modulated three-body, i.e. 3rho。
```

### Case D: stronger regularization 让 static 变差但 relaxed 变好

这是可能且合理的。不要只选 static 最好模型。

优先选择：

```text
relaxed surface stable
relaxed ranking 更接近目标
force RMSE 可接受
static surface 不严重退化
```

---

## 17. Hard fail criteria

任何模型如果出现以下情况，应直接标记为 fail：

```text
any surface gamma < 0
any surface relaxation non-converged with huge energy drop
any slab epa < bulk epa - 0.5 eV/atom
any min pair distance abnormal
force RMSE significantly worse than baseline
short MD unstable if tested
```

这些不是 surface-specific tuning，而是基本势函数稳定性要求。

---

## 18. 不要在第一轮做的事

为了保持实验干净，第一轮不要做：

```text
新增训练构型
改变 surface 构型权重
改变 force/energy 总权重
加入 3rho
加入 dipole
增加 2rho alpha channel
改变 cutoff
改变 knot number
使用 mean-centering
使用 runtime clipping / tanh safeguard
```

第一轮只测试：

```text
column scaling
r_2rho
c_2rho
```

---

## 19. 最终推荐输出文件

建议 Codex 生成以下文件：

```text
phase1_standardization_summary.csv
phase1_standardization_summary.md
phase2_relaxation_summary.csv
phase2_relaxation_summary.md
phase3_smoothness_summary.csv
phase3_smoothness_summary.md
surface_details_by_model.csv
group_contribution_by_model.csv
```

其中 `surface_details_by_model.csv` 建议包含：

```text
model_id
surface
static_gamma
relaxed_gamma
target_MP
static_error
relaxed_error
relax_drop
converged
epa_final
min_pair_distance
max_force_final
gamma_E2_static
gamma_E3_static
gamma_E2rho_static
gamma_E2_relaxed
gamma_E3_relaxed
gamma_E2rho_relaxed
```

---

## 20. 核心结论

本测试的核心不是找到最终最佳势函数，而是快速判断：

> 对 UF3+2rho 来说，feature standardization 和 group/smoothness regularization 是否足以显著改变 relaxed surface behavior？

如果答案是 yes：

```text
继续优化 2rho 的训练侧设置。
```

如果答案是 no：

```text
不要继续在 2rho 上花太多时间，应转向 3rho 或更高阶 angular environment descriptors。
```

最小实验量：

```text
Phase 1: 10 regressions, no relaxation
Phase 2: 3–4 candidate relaxations
Phase 3: 5 regressions, top candidates relaxed
```

这应该足够快速判断 standardization/regularization 是否值得继续深入。
