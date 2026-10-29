# NeuralPDE-Opt 全书独立版

覆盖既有第二至第五章方法。依赖Python、PyTorch、NumPy、SciPy和Matplotlib，不需要原作者仓库、原NeuralPDE-Opt目录或服务器固定路径。第三方移植源码、许可证及来源记录随包保留；本版不宣称全部代码为原创，也不把不同来源统一改为MIT许可。

采用可安装的控制包与随包提供的数值工作区模板。`init`把完整源码、配置和数据生成器展开到用户指定的新目录。每次训练在该目录使用独立子进程和明确的导入路径，保持已验证数值内核的模块边界；安装位置与输出位置可以不同。全书模型、数据和正式结果均不依赖安装包所在目录之外的旧项目。

```bash
python -m pip install neuralpde_book-1.0.0-py3-none-any.whl
neuralpde-book init /absolute/path/my-experiments
neuralpde-book check --workspace /absolute/path/my-experiments
neuralpde-book prepare --workspace /absolute/path/my-experiments
neuralpde-book smoke --workspace /absolute/path/my-experiments
neuralpde-book pipeline --workspace /absolute/path/my-experiments
neuralpde-book status --workspace /absolute/path/my-experiments
```

默认无训练动作。完整第三章预算沿用CUDA配置；第二章和部分模块支持CPU，当前完整套件需要CUDA。第一次运行先执行数值回归及小预算验证。旧检查点不会作为正式重训练初值；HINTS只使用同一新工作区中先训练得到的VarMiON初值。POD、归一化及所有训练数据均可在工作区生成。

所有源码在训练前固定，新模型携带本版本源码身份。第三至五章使用新预定随机流生成最终测试，只有选模冻结后才会生成。第二章仍保留最终迭代模型，以独立积分和FEM进行验证。本版不启动第五章状态分裂ADMM、界面、障碍控制或其他既定后置方法。

`PAYLOAD_MANIFEST.json`记录本版本源码；旧`PORT_MANIFEST.json`仅保留历史来源，新运行校验使用`RUNTIME_MANIFEST.json`。`SOURCE_CHANGES.json`记录封装与协议改动。不得把来源清单改写为新源码的历史身份。

训练中断会保留产物；第三章和第二章有检查点恢复能力，其他流程只跳过经过审计的已完成任务，不宣称优化器精确续训。失败任务不会自动覆盖或以较低预算重跑。最终清单和核查报告只在全部训练、独立评测及审计完成后形成。
