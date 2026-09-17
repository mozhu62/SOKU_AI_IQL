# 从最新 BC TCN256 初始化 IQL

支持源版本 soku_bc_tcn256_joint144_v1。加载时比较完整结构清单（输入字段、动作、时序、当前编码器和融合），再严格加载全部权重；仅在内存中将版本名适配为 IQL 内部同构版本。原 BC 文件不改写，记录原版本与源文件 SHA256。旧 IQL 续训版本不变。

Actor 完整继承 BC；Q1/Q2/V 的编码部分从 BC 复制但参数独立，输出头随机初始化；优化器、步数重新开始。BC 的损失函数/PALR不会变成IQL训练损失。本次不修改IQL奖励、折扣、N-step和Actor加权。

在 configs/iql_suika.yaml 中确认 data.directory 和 split_file 指向本次 IQL 要使用的数据集与固定划分；它可以包含新的训练或验证数据，不要求与 BC 原划分哈希一致。归一化数值仍继承 BC，不重新拟合，当前 IQL 划分哈希会单独写入 checkpoint。TCN256需 training.burn_in=255。将 output.directory 指向新的目录，例如 outputs/iql_from_bc97k，避免覆盖已有IQL训练。路径相对IQL项目解析。

```powershell
cd C:\FXTZ_AI\soku_iql
python scripts/train.py --config configs/iql_suika.yaml --init-bc "C:\Users\Administrator\Desktop\step_000097000_1789265318471816453.pt"
```

默认打开工作台，点击开始训练；--headless 无网页直接训练，--start 网页准备完毕自动训练。后续恢复使用 --resume 新输出目录/last.pt，不再传 --init-bc。

本次仅源码适配和静态检查，已补充测试源码但未执行、未启动训练。不承诺未验证的其他BC架构兼容；旧Joint432或不同输入仍拒绝。
