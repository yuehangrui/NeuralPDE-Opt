# ch04_delivery：独立版模块

符号、公式与阅读路线见[第4章导读](../../docs/CH04_READING_ZH.md)。本目录保留经过验证的数值内核，统一入口见[独立版说明](../../README.md)。

从已安装的 `neuralpde-book` 创建新工作区，按 `check → prepare → smoke → pipeline` 执行。所有训练资产均在新工作区内生成；正式模型从头训练，HINTS只使用本轮VarMiON初值。

本目录的底层CLI供统一驱动调用；独立版使用工作区根的 `book_driver.py` 组织全流程。正式结果写入本工作区 `results/`，不会读取原项目目录。原始数值方法、预算及表示边界见导读和本目录配置；本版源码身份见根目录 `PAYLOAD_MANIFEST.json`。
