# 单步 IQL（历史配置说明）

当前训练已恢复可配置 N-step，本页记录的固定单步方案不再是现行实现。将 `iql.n_step` 显式设为 1 时，训练与验证使用 y=r+gamma*(1-terminal)*V(next_state)，与原单步行为等价。

正式 Dataset 和损失现在会调用 N-step 路径；N=1 时只读取一个额外后继。奖励和 gamma 不因 N 的变化而自动调整。

旧 IQL checkpoint 可以恢复权重和优化器。若配置中的 N 与 checkpoint 不同，运行时会提示 TD 跨度变化；旧、新跨度的 TD 统计不能直接比较。从 BC 重新初始化仍用 `--init-bc` 与新输出目录。

设置 N=1 只会恢复标准单步 TD。项目原有 Neutral 类别降权、Actor 预热等额外设置保持原样，不能据此称所有训练设置都已恢复论文原版。

未运行测试、编译或训练。
