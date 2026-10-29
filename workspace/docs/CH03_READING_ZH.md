# 第三章阅读路线：从输入函数到解

书稿3.3–3.5；完整训练与评测后生成[本轮结果解释](../results/ch03/final_test_v1/summary.md)。正式问题为单位正方形上的

\[
-\varepsilon\Delta y+c(x)y=f(x),\qquad y|_{\partial\Omega}=0,\qquad\varepsilon=0.01.
\]

`c` 是反应系数，`epsilon` 是固定扩散系数。四算子主对照学习固定源项下的 `c → y`；VarMiON及HINTS涉及变化源项/残差，输入信息和任务边界须分别说明。

## 先读哪些文件

| 步骤 | 代码 | 阅读重点 |
|---|---|---|
| 问题与离散 | [reaction.py](../src/neuralpde_opt/chapter3/reaction.py) | 反应项进入质量型矩阵，扩散项进入刚度矩阵 |
| 四种表示 | [reaction_operators.py](../src/neuralpde_opt/chapter3/reaction_operators.py)、[upstream_operators.py](../src/neuralpde_opt/chapter3/upstream_operators.py) | 模型构造、归一化和硬边界；作者移植模块由构造器追踪 |
| 训练与选模 | [reaction_training.py](../src/neuralpde_opt/chapter3/reaction_training.py) | train-only统计量，验证集选模，best/last与断点身份 |
| POD | [pod.py](../experiments/ch03_pod_v1/pod.py) | 质量矩阵加权降阶、输入编码与系数重建 |
| 物理FNO | [physics.py](../experiments/ch03_physics_v1/physics.py) | 能量、弱残差和监督项的区别 |
| VarMiON | [model.py](../experiments/ch03_varmion_v1/model.py)、[model_v2.py](../experiments/ch03_varmion_v2/model_v2.py) | 载荷线性性、矩阵分支、SPD参数化 |
| HINTS | [hybrid.py](../chapters/ch03/hybrid.py) | Jacobi步、神经残差校正、保护机制及停止条件 |
| 完整流程 | [registry.py](../chapters/ch03/registry.py)、[training.py](../chapters/ch03/training.py)、[predictors.py](../chapters/ch03/predictors.py) | 57条正式配置、统一训练与无标签预测接口 |

## 符号和数组

| 数学对象 | 代码名 | 解释 |
|---|---|---|
| 解预测 | `y_prediction` / `y` | 不再用易与伴随混淆的局部 `p` |
| POD空间 | `Phi` / `basis` | 列为基向量；保留离散质量内积 |
| 约化输入载荷 \(z_f=\Phi^T M f\) | `load_coordinates` | 书稿(3.33)–(3.35)，输入线性分支 |
| 约化解系数 \(\beta\) | `beta_coefficients` | 旧数组/文件键中可能仍叫 `alpha`；不是控制正则化系数 |
| 约化映射 \(K_\theta(c)\) | 矩阵分支输出 | 作用于载荷的逆算子型映射，不是刚度矩阵 |
| 线性方程残差 | `r = rhs - A @ y` | HINTS校正方向，书稿(3.43)及算法2 |

四算子公共物理接口使用 `(batch,65,65)` 的反应系数和状态。GNO内部使用17×17节点及P1重建；不能把内部节点数误当输出分辨率。POD和VarMiON v2使用289个传感器；约化系数形状为 `(batch,rank)`，基矩阵为 `(4225,rank)`。原生FNO/Transformer内部轴顺序在各适配器中转换，不以一次全局转置替代。

HINTS接收的是已装配的载荷残差，约化编码为 `Phi_I.T @ r`，不能再乘一次质量矩阵。物理训练中的方程残差可能写成 `A@y-rhs`；平方范数相同，但校正方向不同，应连同更新式一起读。

Fourier增强DeepONet的坐标编码与FNO的频域卷积是不同机制；Galerkin Transformer的积分式注意力也不等同于标准softmax Transformer。方法ID与旧权重键保留，模型含义由构造器及原正式报告限定。

## 训练资产与结果

`neuralpde-book prepare` 在新工作区生成以下训练资产并核验清单；路径均相对于该工作区：

| 路径 | 用途 |
|---|---|
| `data/ch03/reaction_v1`、`data/ch03/varmion_v1`、`data/ch03/hints_v1` | 固定源项、变化源项、残差训练的train/val及几何和清单 |
| `results/ch03/pod_v1/basis.npz`、`normalization.json` | POD基与训练统计量 |
| `results/ch03/varmion_v1` 中 `prepared_manifest.json` 及其声明文件 | 基、归一化、train/val载荷坐标；HINTS还需对应seed的VarMiON初始权重 |
| `results/ch03/varmion_v2` 的预处理产物 | SPD/联合训练的预处理数据与算子池 |

通过统一入口启动正式训练。POD基和归一化只拟合train；HINTS按方法定义使用同一新工作区中新训练的VarMiON模型。每个训练配置写入声明目录下的 `run/`，包含训练历史、选定权重和完整性清单。

选模冻结后生成新的固定测试；历史test不用于本轮选模。不同方法的物理信息、输出空间与计算成本需分别解释，不构成架构家族排名。
