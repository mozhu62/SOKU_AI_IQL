# BC → IQL 源码交付核对

本清单保留最初静态核对，并记录用户授权后的离线运行验收。短步验收不等于实战变强。

## 2026-09-12 离线验收结果：通过

用户提供 `C:\Users\Administrator\Desktop\step_000100000_1789113724875280399.pt` 并明确允许离线验收。

- 环境：Python 3.11.15，PyTorch 2.13.0+cu132，RTX 3050 Laptop GPU；实际使用 CUDA/AMP。
- 自动用例：`6 passed in 9.78s`。首次调用有两项因指定临时目录缺少父目录而未执行；创建父目录后全部重跑通过，不是算法用例失败。
- 来源 BC：step=100000，SHA256=`9bf49038e59d9d10d25f914c3a4e49e4688af8df9ef43595a423255233e313ee`；验收结束再次计算相同。
- 原划分哈希：`f4084ee8dd801b03480b2add5822080c2ff6b6e59d12f4a23db15cbb5bafed4b`。正式入口两次完成全部 6000 个分片的内容校验和读取：train=4800、validation=1200；继承原 normalization。
- 正式入口先运行 1 步预热，保存后通过 `--resume` 再运行 2 步。预热 Actor 逐张量不变；最终 Actor 实际更新 2 次，Actor/Q/V 优化器步数分别为 2/3/3，无 AMP 跳步。
- 导出的 `actor_bc.pt` 由 **原 `soku_bc` 项目** 的 checkpoint loader 和 BCNetwork 严格加载；同一真实数据批次，导出前后 FP32 logits 最大绝对差为 **0**。
- 额外以默认 batch_size=16、sequence_length=32 运行预热和联合更新，各 512 个有效转移；Actor/Q/V 更新成功，无 OOM、NaN/Inf。PyTorch 峰值 allocated 278.6 MiB、reserved 294.0 MiB（不是整机显存占用）。
- 默认形状联合更新单次耗时约 0.393 秒；仅为初始化后的单批检查，不能当成长期吞吐基准。
- 完整 IQL 包 136,589,833 字节，Actor 包 29,659,403 字节；Actor 包包括其优化器状态，不包含 Q/V、Target 或数据集。

复现入口：`scripts/verify_offline.py --init-bc <实际BC模型路径>`，须在明确允许离线训练检查后运行。它自动建立独立验收目录、记录配置快照和结果，不覆盖正式输出。

证据位于 `outputs/offline_acceptance_20260912_174827_102357/`：`result.json`、`default_batch.json`、`acceptance.log`、`run/train.jsonl` 和 `run/validation.jsonl`。

范围限制：短步链路验收每次验证仅 8 个有效转移，本次这批没有动作切换帧，故 `change_top1=null`；未做全量模型质量评估、长时间训练或游戏对局。不能据此承诺策略提升。

验收期间正式 YAML 的数据路径被外部修改为 `../replay_shards_resources_v4_mirror`，该目录在本机不存在；保留该修改。验收使用记录的 `C:\FXTZ_AI\soku_cql\data\replay_shards_resources_v4_mirror`。本机正式启动前需核对路径，服务器可按实际布局保留。

## 最初静态核对记录（下表运行状态为授权验收之前的历史记录）

| 要求 | 当前证据 | 运行状态 |
|---|---|---|
| 保持 BC 策略结构 | `bc_core/models.py`、`temporal.py`、`nn_modules.py` 等 14 文件逐个与源 BC 规范化文本比对一致；`BC_CORE_MANIFEST.json` 记录哈希 | 已静态比对；未前向实测 |
| BC 权重完整初始化 Actor | `models.IQLNetworks` 使用 `strict=True`，`runtime.run` 首次训练还逐张量检查等值 | 检查代码已接入，尚未加载真实 BC 执行 |
| 复用 BC 输入/数据划分/归一化 | 私有 BC reader/observation 原样保留；split 必须已存在，hash 与 BC 包一致，ReplayStore 传入已有 normalization | 未扫描实际完整数据集 |
| 实现 IQL | `learner.py`：target 双 Q 最小值→expectile V；Q 回归 reward+gamma V(next)；detach 优势加权 CE；EMA target | 公式静态核对；未运行反向传播 |
| 不改变策略结构 | Actor 是原 BCNetwork；仅独立 Q/V 有评分头，导出只取 Actor | 已核对参数所有权；无实战结果 |
| 连续历史与 next_state | `dataset.py`：31 前导+L+1 真实序列，合法转移 mask、终局禁止 bootstrap | 提供序列边界测试，未执行 |
| BC→IQL→原 BC 推理 | `checkpoint.export_actor` 输出原 BC loader 要求字段，provenance 标记 IQL；原 BC 文件和实战代码未修改 | 提供严格加载/输出一致性测试，未执行真实实战加载 |
| 离线训练/续训 | `scripts/train.py`、`runtime.py`、完整网络/优化器/scaler/RNG、原子保存、预取种子分离 | 未启动训练 |
| 不覆盖旧模型或数据 | IQL 独立目录；首次拒绝已有目标模型；BC 源和 NPZ 只读 | 本次未写旧项目、数据和 checkpoint |

下一验收：选择实际 BC checkpoint，核对同一 split 后进行最小训练与导出加载对照。由于用户要求“不需要编译与测试”，本次不擅自运行以上用例、训练或游戏。目标的运行闭环尚不能标记为已经验证完成。

2026-09-12 继续核对：已确认本地默认镜像数据目录中的 `train_val_split.json` 存在；BC `outputs`/`save` 未发现 `.pt`。尚需用户提供实际当前 Joint144 BC 模型路径。新增真实 NPZ 读取器的 BC/IQL 输入、mask、标签和 HP 奖励对照用例（未执行）；补充来源文件稳定性检查、固定验证采样参数的续训约束，以及初始化后释放 CPU 原始包。

再次静态核对原 BC `runtime.py` 后，将导出包的 `best` 修正为其原生的 `None`（原 BC 只接受 None 或指标字典，而非浮点无穷）。重新检索 BC/IQL 目录仍未发现 `.pt` 或运行验收结果。当前剩余的闭环证明需要实际 BC checkpoint 和用户对离线验收的许可；不得以未执行的用例宣称通过。
