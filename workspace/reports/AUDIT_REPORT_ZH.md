# 全书独立版：核查报告

生成时间：2026-10-29T01:47:54.176068+00:00。

本轮独立目录重新生成训练资产并完成156组正式训练。旧检查点未用于正式初始化；HINTS按既定方法使用同一工作区中新训练的VarMiON。第二章保存最终迭代，第三至五章按验证集选模。

数值回归：334项通过；优化模式检查：True。第三章各训练变体共21项小预算验证通过。

第三至五章最终测试使用独立版预先声明的新随机流，全部训练结束且选模冻结后才生成。第二章以独立积分、物理一致性和FEM收敛核查。

| 部分 | 核查证据 |
|---|---|
| ch02 | [DELIVERY_AUDIT.json](../results/ch02/full/DELIVERY_AUDIT.json)、[结果报告](../results/ch02/full/summary.md) |
| ch03 | [verification.json](../results/ch03/final_test_v1/verification.json)、[结果报告](../results/ch03/final_test_v1/summary.md) |
| ch04 | [DELIVERY_AUDIT.json](../results/ch04/unified_delivery_v1/DELIVERY_AUDIT.json)、[结果报告](../results/ch04/unified_delivery_v1/summary.md) |
| ch04-depth | [DELIVERY_AUDIT.json](../results/ch04/nonlinear_depth_v1/DELIVERY_AUDIT.json)、[结果报告](../results/ch04/nonlinear_depth_v1/summary.md) |
| ch05 | [DELIVERY_AUDIT.json](../results/ch05/unified_delivery_v1/DELIVERY_AUDIT.json)、[结果报告](../results/ch05/unified_delivery_v1/summary.md) |

精度按各章结果报告逐项解释。本报告不把流程完成等同于算法收敛；固定实例空间验证不是新实例泛化，固定网格延拓也不是分辨率迁移。不同预算、数据与物理信息不构成架构家族排名。

本版可从wheel展开到任意新工作区，运行不需要原项目或作者仓库。科学计算仍依赖PyTorch、NumPy、SciPy和Matplotlib。DeepXDE及Galerkin移植组件和来源许可证随包保留；ocpde_dl来源记录继续保留，不统一改写许可证，也不声称全部原创。

本机安装使用独立虚拟环境，共享服务器已有科学计算依赖；未声称完成全新操作系统或离线全依赖安装。安装记录见[INSTALLATION.json](../verification/INSTALLATION.json)。

版本身份：[PAYLOAD_MANIFEST.json](../PAYLOAD_MANIFEST.json)；来源：[SOURCE_ORIGINS.json](../SOURCE_ORIGINS.json)；封装改动：[SOURCE_CHANGES.json](../SOURCE_CHANGES.json)。训练、选模和最终产物身份由各章清单分别核查。
