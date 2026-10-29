# NeuralPDE-Opt：书稿配套独立版

覆盖书稿第二至第五章已核定的配套方法。v1.0.0完成156组正式神经训练、8组FEM、新测试与独立复核；不含原训练/验证数据，不自动启动训练。

## 阅读与下载

- [10页全书清单与核查报告](workspace/reports/publication/main.pdf)、[LaTeX源码](workspace/reports/publication/main.tex)
- [书稿—代码映射](workspace/reports/BOOK_CHECKLIST_ZH.md)、[核查报告](workspace/reports/AUDIT_REPORT_ZH.md)、[人工复核附记](workspace/reports/REVIEW_NOTES_ZH.md)
- [v1.0.0发布附件](https://github.com/yuehangrui/NeuralPDE-Opt/releases/tag/v1.0.0)：约998 MiB的完整上传实验包、安装wheel、报告及SHA256清单。实验包含180个正式选定/阶段模型、最终测试与预测、模型基、训练标量历史和审计证据。

Git仓库提供源码、文档及文本/图表证据；模型与测试数组在Release附件内。未公开发布的Release链接在上传完成后生效。

## 安装与运行

```bash
git clone git@github.com:yuehangrui/NeuralPDE-Opt.git
cd NeuralPDE-Opt
python -m pip install .
neuralpde-book init /absolute/path/new-experiments
neuralpde-book status --workspace /absolute/path/new-experiments
```

`init`只展开新工作区，已有目录不会被覆盖。`src/neuralpde_book/_payload/`是可安装的完整数值源码模板；`workspace/`是对应正式实验的源码与证据快照，317项数值源码与模板保持相同身份。不要直接在冻结快照目录继续训练。

需要从头复现时，在全新目录显式执行：

```bash
neuralpde-book check --workspace /absolute/path/new-experiments
neuralpde-book prepare --workspace /absolute/path/new-experiments
neuralpde-book smoke --workspace /absolute/path/new-experiments
neuralpde-book pipeline --workspace /absolute/path/new-experiments
```

这些命令分别执行回归检查、生成数据、小预算验证和完整训练评测。完整套件需要CUDA，实测环境见[依赖记录](requirements-tested.txt)；未声称验证全新操作系统或离线全依赖安装。安装本包不需要原项目目录或原作者仓库，但仍依赖Python、PyTorch、NumPy、SciPy和Matplotlib。

## 已完成实验

| 部分 | 正式训练 | 模型/结果 |
|---|---:|---|
| 第二章 | 18组神经训练，另8组FEM | PINN、Deep Ritz、弱式；18个最终模型 |
| 第三章 | 57组 | 算子学习、物理/变分扩展、VarMiON与HINTS；57个模型 |
| 第四章基础 | 48组 | 最优控制基线和算子学习等；72个阶段模型 |
| 第四章深化 | 12组 | 非线性Deep Uzawa、稀疏观测FEM-NN；12个模型 |
| 第五章 | 21组 | 已核定的约束控制配套；21个模型 |

334项数值回归和独立复核通过。训练数与模型数不同，是因为第四章保留多个交付阶段。全部预算完成不表示同精度、严格最优性或架构家族排名；本轮test已查看，后续开发应使用train/val。第五章5.2–5.3及其他后置分支不计为已完成；详细边界见全书清单。

## 核验与来源

Release实验包解压后可运行`python verify_package.py`核对全部文件。仓库与精简包都排除了训练数据和部分恢复文件，因此原始FINAL_AUDIT引用的全部训练产物只能在完整工作区核验；不要在此仓库直接运行历史全量审计或声称精确续训。

第四章原摘要包含历史模板文字“复用9组seed0”，请以[更正摘要](workspace/results/ch04/unified_delivery_v1/summary.reviewed.md)和人工复核附记为准。原证据保持原字节。

移植代码的原许可证与出处保留：[第三方声明](src/neuralpde_book/_payload/third_party/ch03_upstream/NOTICE.md)、[源码来源](workspace/SOURCE_ORIGINS.json)。DeepXDE组件保留LGPL-2.1，Galerkin Transformer组件保留MIT，ocpde_dl来源记录保留。没有将全库统一改为MIT，也不宣称全部原创。
