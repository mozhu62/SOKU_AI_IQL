# Actor Neutral 监督筛选

当前默认配置已改为损失加权，下面的筛选方案仅作为旧模式保留，默认关闭。二者禁止同时启用。

```yaml
actor_sampling:
  enabled: false
actor_weighting:
  enabled: true
  neutral_weight: 0.25
```

加权模式保留所有有效监督帧，Actor loss 为 `sum(category_weight * detached_advantage_weight * CE) / sum(category_weight)`。Neutral 类别权重为 0.25，其他动作均为 1；不是将 Neutral 样本比例限制到 25%。全 Neutral 批次仍更新 Actor，类别权重在分子分母抵消。Q/V、验证、网络和实战输入不变。新日志记录 `actor_weighting_enabled` 和实际生效的 `actor_neutral_weight`；网页展示加权状态。旧 checkpoint 可加载，重启时按 YAML 应用；目标权重改变会打印警告。未提供在线热修改。

以下是旧筛选模式说明（仅手动重新启用且关闭加权后生效）：

```yaml
actor_sampling:
  enabled: true
  neutral_max_fraction: 0.25
```

Neutral 为 Joint144 ID 64：方向 5，ABCD 全部松开。蹲下、后退、攻击不属于 Neutral。

这是监督位置欠采样，不另建 Actor 数据加载器、不补采样到原 batch 大小。每批对全部有效监督位置统一统计（不是逐条 sequence），保留全部非 Neutral；若非 Neutral 有 M 帧，上限 p<1 时最多保留 floor(p*M/(1-p)) 个 Neutral，从原有 Neutral 中均匀随机选取。因整数取整，实际比例可小于上限。

例如 9 个 Neutral、3 个其他动作，上限 25% 时保留 1 个 Neutral 和 3 个其他动作。全 Neutral 批次在上限小于 100% 时没有 Actor 监督，跳过该批 Actor 更新，Q/V 正常更新。上限 0 表示不监督 Neutral，上限 1 或 enabled=false 保持原始有效 mask。缺少整个配置段的旧 YAML 默认关闭，保持原行为。

完整 observation、上一帧动作、TCN 历史、burn-in、Q/V mask、奖励、折扣和验证分布不变。Actor 仍计算 `mean(detached_advantage_weight * CE)`，只是平均范围变成筛选后的有效监督位置；不除以优势权重和，不添加惩罚。

## 日志与界面

训练日志新增 `actor_candidate_samples`、`actor_samples`、`actor_neutral_before/after`、`actor_filtered_samples`、`actor_neutral_fraction_before/after`、`actor_sampling_enabled`、`actor_neutral_max_fraction`、`actor_skip_reason`、`actor_optimized_samples`。筛选后无样本时比例为 null，不能当成 0%。预热期间筛选统计只表示候选监督，实际优化帧数为零。

学习诊断页新增 Neutral 筛选前后比例曲线、有效帧数和跳过原因；原 `samples` 与训练吞吐仍对应 Q/V 完整批次。配置可在 YAML 中修改，网页运行配置中可查看并导出，不提供在线热修改。

## 续训与限制

权重形状、checkpoint 版本和实战推理不变，旧 IQL checkpoint 可以续训。采样设置保存在完整 checkpoint/config.json/运行配置导出中。续训按指定 YAML 生效；与旧 checkpoint 不同时打印明确警告，这属于改变训练分布，不是精确复现旧训练。随机筛选使用 PyTorch RNG，由现有 checkpoint 保存恢复。

该比例不是推理约束，也不保证模型输出 Neutral 不超过 25%，不会解决所有蹲下问题。Actor 样本数可能显著减少，需同时观察验证与实战表现。

已补充边界与集成测试源码；遵照要求，本次未运行测试、编译或启动训练。
