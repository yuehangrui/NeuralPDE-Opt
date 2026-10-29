# 第五章阅读路线：控制约束与状态约束

书稿5.2.1和已实现的5.2.2 CNP部分；[正式完成后报告](../results/ch05/unified_delivery_v1/summary.md)。控制盒约束 `a≤u≤b` 与状态上界约束 `y≤psi` 是两个独立问题，保留不同参考求解器和指标。

## 七条流程

| 方法 | 核心文件 | 对照含义 |
|---|---|---|
| OSNN | [models.py](../chapters/ch05/models.py) | 投影最优性 `u=Π_[a,b](-p/alpha)`，学习状态及缩放伴随 |
| ADMM-PINN | 同上 `SplitState`；[delivery/training.py](../chapters/ch05_delivery/training.py) | 控制分裂 `u=z`，独立内层更新与外层投影/乘子更新 |
| 投影KKT-FNO | [operators.py](../chapters/ch05/operators.py) | 历史方法ID `mimo`，输出状态与缩放伴随后投影；不是第四章MIMOONet |
| PDOL | 同上 `action`；[reference.py](../chapters/ch05/reference.py) | 两个训练代理分别近似状态与伴随算子，嵌入原始—对偶迭代 |
| iUzawa-F / iUzawa-S | [operator_blocks.py](../chapters/ch05/operator_blocks.py) | 两种展开架构保留各自层与共享方式，不能按名字假定所有层同权 |
| CNP | [models.py](../chapters/ch05/models.py) 的 `method=cnp` | 光滑状态可行表示，再由 `u=-Δy` 恢复控制（本例 `f=0`） |

先读[problem.py](../chapters/ch05/problem.py)定义物理量和约束，再读上述表示及[损失与训练](../chapters/ch05_delivery/training.py)，最后读[protocol.py](../chapters/ch05_delivery/protocol.py)和[assessment.py](../chapters/ch05_delivery/assessment.py)。[common.py](../chapters/ch05_delivery/common.py)定义正式预算，[data.py](../chapters/ch05_delivery/data.py)定义数据身份。

## 四类容易混淆的量

| 代码/书稿符号 | 数学含义 |
|---|---|
| `p` | PDE物理伴随，满足 `-Δp=y-yd` |
| `q_scaled` | `p/alpha`，OSNN和投影KKT使用；检查点中 `q_net` 键保留 |
| `lambda_admm` | 控制分裂 `u-z` 的未缩放乘子；旧存储键 `lam` 保留 |
| `p_dual` | PDOL/iUzawa观测空间对偶量；书稿(5.11)–(5.13)也写 `p`，代码显式区分 |

对 `p_dual`，精确固定点满足 `p_dual=y-yd`，而物理伴随是 `p=S* p_dual`；两者不能直接互换。代理算子和有限展开未必已达到精确固定点，这一关系应作为定义和极限关系解释。

ADMM采用 `+<lambda,u-z>`，内层控制与投影更新为

\[
u=\frac{\beta z-\lambda-\alpha q}{\alpha+\beta},\quad
z^+=\Pi_{[a,b]}(u+\lambda/\beta),\quad
\lambda^+=\lambda+\beta(u-z^+).
\]

保留 `u` 与可行分裂控制 `z` 两个输出，分别报告原始、对偶残差。`SplitState` 保存每轮内层之后、外层之前的网络快照，以便在任意新坐标恢复乘子历史。不能仅靠最后一个 `q_net` 代替全部ADMM状态。

iUzawa内部原来的 `p` 参数改为 `p_dual`，包括 `_u_step`、`_p_step`、`forward`；位置参数调用兼容。外部若使用这些低层函数的 `p=` 关键字，需要改成 `p_dual=`。模型层名称与状态字典键没有变化。

## 形状与约束检查

实例方法坐标 `(N,2)`，场和积分权重 `(N,1)`；算子方法采用内部31×31网格，输入 `(batch,31,31,1)`，即channels-last。它与第四章33×33的含边界网格不同，不强行统一轴顺序或分辨率。`q_scaled` 输出恢复为物理伴随时必须乘 `alpha`。

CNP使用 `psi-(sqrt(psi)+D*N_theta)^2`，其中 `D` 是边界消失因子，满足状态上界和本例齐次边界。保留其光滑可行表示及高阶空间微分；不将straight-through裁剪混入梯度。CNP与控制盒约束的最优性条件不同，也不是5.4障碍问题。

正式交付为七条流程×三个种子，共21个选定模型。历史test不用于本轮选模；新test按预定随机流在选模冻结后生成。5.2.2状态分裂ADMM、5.3界面控制、5.4障碍控制继续后置；本轮没有补造这些实验。
