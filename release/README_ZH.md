# NeuralPDE-Opt 独立版：上传精简包

日期：2026-10-29；独立版1.0.0，对应156组正式神经训练、8组FEM、新测试和审计。

## 保留内容

- workspace/：317项冻结数值源码、配置、生成器、运行说明、许可证。
- 180个正式选定模型：第二章18个最终模型、第三章57个、第四章基础72个阶段模型、第四章深化12个、第五章21个。第四章的一次训练可包含多个交付阶段，因此模型数不同于训练组数。模型索引及哈希见MODEL_INDEX.json。
- 训练曲线、标量指标、最终测试数据/预测、经典参考解、HINTS求解结果、推理所需POD/VarMiON基及归一化统计。
- workspace/reports/publication/main.pdf与main.tex：10页全书对应清单与核查报告；其余日志及审计证据保留。
- wheel/可安装包；package_source/封装源码与测试。

## 精简范围

不含原训练/验证样本、采样缓存、训练/验证预测数组、初始/末次/恢复检查点、未被正式选模清单选中的中间阶段模型、运行锁和开发试跑。保留的选定检查点沿用原始字节，可能仍包含优化器状态；不通过重写模型来缩小体积。

EXCLUDED_FILES.json列出原完整下载包与本次额外排除项；FULL_PACKAGE_MANIFEST.json保存原完整下载包清单身份。PACKAGE_MANIFEST.json对应本上传包实际内容。原始FINAL_AUDIT和选择冻结清单没有改写，其中引用的训练样本和恢复检查点可能已被有意排除，不能在此包上直接重跑全部历史审计。

## 查看与核验

请解压到新的独立目录，先阅读workspace/reports/publication/main.pdf及workspace/reports/REVIEW_NOTES_ZH.md。第四章原summary.md的历史模板文字已在summary.reviewed.md更正；原始审计文件保留。

在解压根目录运行：

```bash
python verify_package.py
```

该命令只用Python标准库逐项核对文件，不训练。ZIP的外部SHA256见同名.sha256文件。

需要安装时：

```bash
python -m pip install wheel/neuralpde_book-1.0.0-py3-none-any.whl
neuralpde-book status --workspace /absolute/path/to/workspace
```

仍需Python/PyTorch/NumPy/SciPy/Matplotlib等科学计算依赖，见package_source/requirements-tested.txt。完整第三章训练使用CUDA；不包含Python环境。复训应在新的工作区重建数据，不覆盖本包的冻结结果。本包不支持精确恢复所有训练，也不是完整训练数据备份。

## 发布使用

本ZIP适合作为版本发布附件保存。代码仓库可使用package_source/中的可安装源码，并保留许可证与来源说明；模型、预测和大型结果可作为单独的发布附件。制作此包没有创建远程仓库或上传内容。
