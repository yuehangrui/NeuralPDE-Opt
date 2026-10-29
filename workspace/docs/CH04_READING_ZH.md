# 第四章阅读路线：光滑最优控制

书稿4.3–4.6。先读[统一交付入口](../chapters/ch04_delivery/README.md)，正式完成后结果位于[基础报告](../results/ch04/unified_delivery_v1/summary.md)及[深化报告](../results/ch04/nonlinear_depth_v1/summary.md)。本章以线性二次椭圆控制为基础：

\[
\min_u J=\tfrac12\|y-y_d\|^2+\tfrac\alpha2\|u\|^2,
\quad -\Delta y=u+f,\quad y|_{\partial\Omega}=0.
\]

采用物理伴随约定 `-Δp=y-yd`，因此约化梯度为 `alpha*u+p`，对应书稿(4.2)、(4.5)。问题对象写作 `problem`，文件位置写作 `path`，预测写作 `y_prediction`；不占用物理伴随的 `p`。

## 方法与阅读路径

| 方法 | 从问题到实现 | 特别说明 |
|---|---|---|
| 经典与PINN | [problem.py](../chapters/ch04/problem.py) → [models.py](../chapters/ch04/models.py) → [training.py](../chapters/ch04/training.py)；[fem.py](../chapters/ch04/fem.py)作基准 | direct、KKT、ALM目标不同；空间微分与参数微分分开 |
| 频率增强实例方法 | [instance.py](../chapters/ch04_adaptation/instance.py) | Gauss积分、完整频率增强；神经阶段与头修正阶段单列 |
| FNO/DeepONet/FEDONet | [models.py](../chapters/ch04_operator_adaptation/models.py) → [training.py](../chapters/ch04_operator_adaptation/training.py) | `state`学状态映射；`kkt`学最优状态/控制 |
| MIMOONet与RB-DINO | [operator_models.py](../chapters/ch04_adaptation/operator_models.py) → [operators.py](../chapters/ch04_delivery/operators.py) | 四个乘法分支与共享trunk；RB-DINO另有导数监督 |
| 双层、Deep Uzawa、ISMO、FEM–NN基线 | [ch04_completion](../chapters/ch04_completion/README.md) → [hybrids.py](../chapters/ch04_delivery/hybrids.py) | 有限正弦特征基线和非线性代理分开解释 |
| Deep Uzawa/FEM–NN深化 | [common.py](../chapters/ch04_depth/common.py) → [training.py](../chapters/ch04_depth/training.py) → [assessment.py](../chapters/ch04_depth/assessment.py) | 隐藏层可训练坐标网络；稀疏/噪声源项识别 |
| 统一交付 | [cli.py](../chapters/ch04_delivery/cli.py) → [protocol.py](../chapters/ch04_delivery/protocol.py) → [assessment.py](../chapters/ch04_delivery/assessment.py) | 训练、冻结、独立评测、审计，阶段身份各自保留 |

## 乘子、缩放和残差方向

| 对象 | 定义及约定 |
|---|---|
| 物理伴随 `p` | 解 `-Δp=y-yd`，用于真实约化梯度 |
| 缩放伴随 `q` / `q_scaled` | 仅在明确规定时表示 `p/alpha`；不是所有旧局部 `q` 的含义 |
| ALM乘子 | 作用于 `r=-Δy-u-f` 的 `+<lambda,r>`，可行最优处对应 `lambda=-p` |
| ALM罚参数 | 代码是 `mu*||r||²`，书稿(4.25)–(4.26)的 `beta/2*||r||²` 对应 `beta=2mu`；更新为 `lambda+=2mu*r` |
| Deep Uzawa乘子 `z` | 修正目标含 `alpha/4*||u||² + alpha/4*||Ay-f||² + <z,Ay-u-f>`；控制块 `u=2z/alpha`，可行最优处 `z=-p/2` |
| 旧谱实现的局部 `q` | 若定义为 `K@y-f`，它是由状态恢复的控制量，不应改名为缩放伴随 |

神经网络的原始状态、由控制重求的PDE状态、由代理求出的梯度和真实约化梯度均保留单独字段。低目标值不能替代可行性与驻点检查。

## 张量、离散与实现深度

点坐标为 `(N,2)`，标量场通常为 `(N,1)`。问题族采用17个正弦模态，模态向量为 `(batch,17)`；MIMO输入/输出为 `(batch,2,17)`。适配后的网格算子输入为 `(batch,2,33,33)`，KKT输出两通道为状态和控制，物理伴随可由 `p=-alpha*u` 恢复。基础FNO与其他章节的channels-last实现不可直接混接。

`head_polished` 是训练数据上的输出头修正，不能标成原生网络精度。模态化MIMO、RB-DINO和17模态输出空间也需保留假设，不能因入口统一就省去表示边界。

`ch04_depth` 中“非线性”指可训练的非线性网络表示；基础PDE仍为线性Poisson。FEM–NN学习源项，状态由有限元求解；积分使用一致质量矩阵。新Deep Uzawa训练所有隐藏层，但控制块仍为解析更新。有限特征双层基线不因此成为一般非凸深层双层优化实验。

基础正式交付48组配置、72个阶段检查点；深化另有12组。保留原训练预算、train-only归一化与最早最小验证分数的选模规则。精度以本工作区完整训练与独立评测后生成的结果为准；半线性、抛物型扩展继续后置。
