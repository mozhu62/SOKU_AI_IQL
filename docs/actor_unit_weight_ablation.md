# Actor 权重恒1消融

当前配置：actor_advantage_weighting=false，actor_warmup_steps=0，actor_sampling.enabled=false，actor_weighting.enabled=false。Actor使用全部有效位置的普通等权CE，Q/V和配置指定的N-step TD照常训练。保留neutral_weight编辑值但关闭其生效开关。

日志：actor_objective=plain_bc，actor_advantage_weighting=false，weight_mean=1，weight_max=1。检查actor_updated确认实际更新；AMP溢出仍可跳步。

这只是损失层面的普通BC，不等于复现BC项目的采样、优化器、学习率、关键帧权重或PALR。退化不能直接证明Actor trainer有bug；稳定而优势加权版本退化，则支持进一步排查价值估计和加权机制。

用同一BC源、同一固定划分和新输出目录，以--init-bc开始对照。旧配置缺开关默认true，保持原IQL；续训改变开关会拒绝，避免静默切换目标。未运行测试或训练。
