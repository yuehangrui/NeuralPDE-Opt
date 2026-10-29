# 第二章阅读路线：神经函数近似

书稿2.2–2.6，共同问题为1D/2D Poisson。状态统一写作 `y`；网络为 `y_theta=G+D*N_theta`，FEM状态为 `y_h`。`u`留作后续章节的控制量。

先读 [problems.py](../src/neuralpde_ch02/problems.py) 和 [models.py](../src/neuralpde_ch02/models.py)，再读 [objectives.py](../src/neuralpde_ch02/objectives.py) 中的强残差PINN、直接能量最小化Deep Ritz及Gram范数感知VPINN。弱式实现不是WAN。

[training.py](../src/neuralpde_ch02/training.py) 管理Adam/L-BFGS及检查点恢复；[metrics.py](../src/neuralpde_ch02/metrics.py) 进行独立积分、L2/H1梯度半范、边界和能量诊断；[fem.py](../src/neuralpde_ch02/fem.py) 给出经典基准。

正式 [full.json](../chapters/ch02/configs/full.json) 包含2维度×3方法×3种子，共18组训练及8组FEM。保存最终迭代模型，验证数据仅诊断，不用于选择最优迭代。L-BFGS预算为名义预算，实际迭代与目标调用分别记录。

使用[统一入口](../README.md)在新工作区重训，结果写入 `results/ch02/full`。旧第二章冻结包保持独立，不会作为新训练输出。
