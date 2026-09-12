# IQL TCN64

此文为上一版64帧记录；当前默认配置已改为256帧，请使用 [TCN256说明与迁移命令](tcn256.md)。旧64迁移脚本要求63前导配置，不能直接配合当前255前导配置使用。

当前正式配置将 `training.burn_in` 改为 63，加载配置后明确选用 `model.temporal_mode=tcn64`，网络版本为 `soku_iql_tcn64_joint144_v1`。旧 `tcn` 仍代表 TCN32；缺省历史仍保持旧配置兼容，不能仅靠文件名判断模型窗口。

结构增加 dilation=16 的双卷积残差块，膨胀率为 `[1,2,4,8,16]`。感受野 `1+1+2*(1+2+4+8+16)=64`，包括当前及前 63 帧，约 1.05 秒历史跨度。通道宽度、228D 状态、对象分支、Joint144 输出保持不变。历史弹幕对象仍不进入 TCN。

监督段仍为 32，N-step 保留用户当前配置 30，所以完整训练序列为 `63+32+30=125` 帧（改成 N=3 则为 98 帧）。因果卷积不会读取当前时刻之后的输入；短片段使用有效历史 mask，不跨回合补历史。训练与推理计算量会增加，本次未测耗时。

## 从旧完整 IQL 模型迁移

在项目根目录手动执行（本次没有执行）：

```powershell
python scripts/migrate_tcn64.py --source outputs/iql_suika/last.pt --output outputs/iql_suika_tcn64/initial.pt --config configs/iql_suika.yaml
python scripts/train.py --config configs/iql_suika.yaml --resume outputs/iql_suika_tcn64/initial.pt
```

迁移复用 Actor、Q1、Q2、V 和两个目标 Q 的全部旧层，只新增各网络的第五个 TCN 块。新增块随机初始化，新目标块与对应 online 块同步；优化器、训练计数与 best 排名重置，归一化和固定划分沿用来源。会重新执行配置的 Actor 预热。输出不保证等同原策略，必须重新训练评估。迁移文件存在时拒绝覆盖。使用配置的 seed 初始化新增块。

默认训练输出改为 `outputs/iql_suika_tcn64`，旧目录保留。从原 BC 初始化仍可用 `--init-bc`，会明确记录 TCN32→64 扩展，复用旧 Actor 层后初始化 Q/V；新增结构不是精确 BC 策略复制。

## 实战与兼容

CPU 迁移包保存的空 GradScaler 状态可直接续训：CUDA AMP 会新建缩放历史并打印警告，不改变已迁移权重；无需重新迁移。

实战配置默认模型更新为 `outputs/iql_suika_tcn64/actor_bc.pt`。窗口根据实际加载模型的 spec 选择 32 或 64，满真实连续窗口才发键；断帧、换局重新积累。网页与评估报告显示实际窗口长度。无需改 DLL 的帧数据结构。

64 帧完整包有独立版本，不能直接 `--resume` 旧32帧包跳过迁移。导出策略仍使用 Actor 包形式，但有独立网络版本，未更新的原 BC 推理端会拒绝加载；请使用本项目更新后的实战入口。原 BC 项目源码未改。

已补结构、迁移、因果窗口和实战缓存测试源码；遵照要求，未编译、运行测试、执行迁移或启动游戏。
