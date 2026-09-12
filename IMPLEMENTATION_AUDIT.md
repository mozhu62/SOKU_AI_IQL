# BC → IQL 源码交付核对

本清单记录源码证据与尚未完成的运行验收，不以“文件存在”冒充已训练成功。

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
