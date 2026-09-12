# SOKU IQL：BC → 离线 IQL → 原 BC 实战端

本目录是独立 Python 项目，不修改 `soku_bc`、`soku_cql` 或 `soku_ai`。保留当前 BC 的 228D 扩展状态输入、Current 256、TCN32 256、每侧最近三个对象、Fusion 1024、Joint144 分类策略。不是 DQfD，也不把 BC logits 直接作为 Q 值。

## 首次训练

使用已能运行 BC、且 PyTorch/CUDA 与驱动匹配的 Python 环境。在本项目根目录：

```powershell
pip install -e .
python scripts/train.py --config configs/iql_suika.yaml --init-bc ../soku_bc/outputs/bc_suika_tcn32_joint144/last.pt
```

模型路径是示例，换成实际训练好的当前 Joint144 BC 文件。旧 Joint432/GRU checkpoint 明确拒绝。`configs/iql_suika.yaml` 中的数据目录与 split_file 必须指向该 BC 原来的数据集和固定划分；服务器上按实际位置调整。所有配置路径相对 IQL 项目根目录，命令行模型路径按当前工作目录解析。目录实际为 `C:\FXTZ_AI\soku_iql`。

初始化严格复制全部 Actor 权重；沿用 BC normalization、方向规则、action_shift 和 split_hash，不重新拟合归一化、不重新随机划分。Q1、Q2、V 的编码器从 BC 复制但参数独立，评分头新初始化，target Q 从 Q 复制。首次训练先记录 BC 基线，默认前 1000 步只学习 Q/V，此时 Actor 不更新，之后采用 IQL 优势加权 CE 更新策略。

加载时核对文件签名并立即计算来源 SHA256；若训练中的 BC `last.pt` 同时被覆盖，会明确拒绝。迁移建议先把所选 BC 文件复制为固定版本，避免数据预读期间来源变化。

## 续训和实战

```powershell
python scripts/train.py --config configs/iql_suika.yaml --resume outputs/iql_suika/last.pt
```

Ctrl+C 请求在完整更新结束后保存；不启动游戏。默认训练日志间隔 20 步、保存/验证间隔 1000 步。没有新增浏览器服务或端口，也不依赖 BC 的网页服务。

续训锁定 batch_size、sequence_length、burn_in、replays_per_batch、validation_batches，以及 seed/模型/IQL/奖励定义，保证固定验证样本与已有 best 的比较口径不变。设备、缓存、总步数及记录间隔可以调整；跨设备数值不承诺逐 bit 重现。

| 文件 | 用途 |
|---|---|
| `last.pt` | 完整 IQL 续训：Actor、Q1/Q2、V、target Q、优化器、随机状态 |
| `actor_bc.pt` | 最新策略，直接在原 BC 实战界面“选择模型”加载 |
| `bc_initial_actor.pt` | IQL 更新前的 BC 策略基线 |
| `best_actor_bc.pt` | 验证 NLL 优于初始 BC 基线后才产生；不是实战最强的保证 |
| `train.jsonl` / `validation.jsonl` | 损失、优势权重、速度、TD 误差、EV、动作一致率 |
| `bc_baseline.json` | BC 初始化策略的固定验证基线；其中 Q/V 是初始化评分，不是 BC 训练结果 |

`actor_bc.pt` 为原 BC 加载器兼容格式，`algorithm=bc` 表示文件协议兼容，不表示仍用普通 BC 训练；`provenance.trained_algorithm=iql` 明确记录训练来源。文件只带 Actor 和 Actor 优化器，不带 Critic/Target 或数据集，不应像 PER 树那样随数据量增大。

也可单独导出，不需要数据集：

```powershell
python scripts/export_actor.py --checkpoint outputs/iql_suika/last.pt --output outputs/iql_actor_for_live.pt
```

输出文件必须不存在。导出的 BC 兼容文件不用于 IQL `--resume`；IQL 续训只用完整 `last.pt`。BC 实战端的 DLL、连续 32 帧窗口及按键映射无需因 IQL 改动。

## 算法和默认奖励

按 [IQL 官方实现](https://github.com/ikostrikov/implicit_q_learning) 实现离散动作版本：

1. 数据动作的 target `min(Q1,Q2)` 与 V 做 expectile 回归，默认 expectile=0.7。
2. 双 Q 回归 `r + 0.99 × (未终局) × V(next)`，使用平方误差。
3. Actor 最小化 `mean(min(exp(3×(Q−V)),100) × CE)`；优势完全 detach，不通过 Actor 更新 Critic。不是把权重归一化后平均，也不是 CQL 的全动作惩罚。
4. target Q 用 tau=0.005 EMA 更新；没有 argmax 后继动作、PPO/GAE、PER 或专家 margin。

BC 不学习奖励，因此本版显式给 IQL 定义可配置的 HP 奖励：每造成 1 点对方扣血 +0.001，每受到 1 点扣血 −0.001；只在有效同段转移计奖，不额外加胜负/接近/出招奖励。不读取 NPZ 旧 rewards，避免混入旧 CQL shaping。此默认是实现选择，可在首次训练前修改 YAML；续训不允许静默改变奖励/IQL 参数。

IQL 替代了 BC 的训练目标，本版不叠加关键帧加权或 PALR；Actor 结构和输入保持不变。Q/V 是训练所需额外网络，不参与实战推理，不能声称训练内存或速度与 BC 一致。

## 数据与时序

复用 BC 分片校验、Joint144 投影、START=144、技能类型、对象 mask 和 normalization。每条样本包含 31 帧真实前导上下文、L 个监督位置和额外 1 帧后继。当前/后继从同一个因果 TCN 序列计算，历史不会因为计算 next_state 被清空；不足上下文按原 BC 对齐与 mask 处理。终局不 bootstrap；非终局片段末尾只在确有合法同段下一帧时 bootstrap。连续性以既有 NPZ 的 episode_id/transition_valid/terminated 契约为准，不凭空恢复缺失帧。

采用分片缓存和单后台线程预取，训练采样 RNG 由 seed、step 派生，验证使用另一组固定种子；预取丢弃不改变续训抽样序列。验证不进入训练，不更新参数。模型结构源码保存在 `soku_iql/bc_core`，14 个文件与当前 BC 的规范化文本完全一致，哈希见 `BC_CORE_MANIFEST.json`；不依赖服务器上另装一份 BC 仓库。

## 验收状态

2026-09-12：用户提供真实 BC checkpoint 并授权后，已完成 **6 项自动用例、全量 6000 分片校验/读取、3 步 CUDA 短训及续训、原 BC 加载器导出兼容性验收**。原模型哈希未变，同输入导出前后 logits 最大差为 0；默认 16×32 批次也完成了 GPU 更新。详情与边界见 `IMPLEMENTATION_AUDIT.md`。这不代表完成长期训练或实战效果验证。

可复现离线验收（会运行几步训练，输出到独立目录，不启动游戏）：

```powershell
python scripts/verify_offline.py --init-bc "实际的BC模型.pt"
```

以下为初次源码交付时的验收说明；其中“未执行”已由上述授权验收结果补充：

已完成源码实现及静态协议核对，未执行编译、测试、训练、推理或对局。`tests/` 提供损失公式、连续窗口、迁移/导出和参数隔离用例。实际 BC→IQL 训练与原 BC 实战加载还需要运行验收，不能把源码兼容当作已经打过对局。

建议先核对：初始化 Actor 输出与 BC 一致；预热期间 Actor 不变；非预热阶段仅优势加权 Actor 更新；终局目标没有 V(next)；导出后原 BC 加载器严格加载成功；实战比较 `bc_initial_actor.pt` 与 `actor_bc.pt`。离线 NLL/EV 不是伤害差或胜率，仍需固定对手实测。
