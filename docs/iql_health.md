# IQL Actor 偏攻击诊断

本次增加只读统计，不调整 reward、网络、beta、expectile、target_tau、Keyframe 定义、
Actor LR、max_weight 或损失公式。现有 checkpoint 可以继续 --resume；新增诊断配置默认启用。
完整配置示例见 configs/diagnostics.example.yaml，将 diagnostics 段合并到实际使用的 YAML。
示例文件不是独立训练配置。训练每100步采集一次实际更新张量；每次validation产生健康报告；
固定probe每1000步及validation时评估，同一步不重复评估。新初始化记录step 0；续训从恢复步开始，
无法还原丢失的历史日志。

## 输出

全部输出在实际 output.directory：

| 文件 | 内容 |
|---|---|
| health_train.jsonl | 诊断步的真实训练batch、实际Actor筛选及更新状态 |
| health_validation.jsonl | 所有验证batch合并后的精确统计与告警 |
| iql_health_report.txt | 每次validation可读报告，与终端一致 |
| health_probe.jsonl | 固定probe的统计、probe_id、与上次同一probe的差值 |
| probe_manifest_*.json | 独立采样配方、分片和帧身份、覆盖及缺失标签、输入内容哈希 |
| probe_samples.jsonl | 每个固定状态逐步的144维logits/probabilities、专家动作Q1/Q2/target Q、V、A及权重 |

每条训练/验证报告带step和source_sha256，工作台状态接口也包含最新validation的health对象。
既有 train.jsonl / validation.jsonl 和 best选择逻辑保持原口径。

## 数值与权重口径

Q1/Q2为专家动作对应的在线Q，target_q1/target_q2单独记录。训练时统计
在线Q更新前、target Q的EMA更新前、V更新后、Actor更新前logits；A与实际Actor损失一致。
validation在冻结权重上计算A，不能拿在线Qmin-V代替target Qmin-V。
std为总体标准差；所有分位数在拼接有效样本后计算，绝不平均各batch的分位数。
padding均排除；NaN/Inf计数单列，JSON用null表示不能解释的数值。

final_weight = keyframe_weight * iql_weight。
effective_weight = final_weight * neutral_category_weight * actor_selected_mask。
loss_coefficient = effective_weight / sum(keyframe_weight * neutral_category_weight * actor_selected_mask)。
这些是交叉熵的系数，不是实测梯度范数；CE导数和网络Jacobian仍影响真实梯度。
applied_loss_coefficient在warmup、无样本或AMP跳步时为0。
validation/probe不执行随机Actor筛选，权重标记为候选权重，不能冒充实际更新。
验证合并时loss_coefficient保留每个batch自己的分母。

weight_clip_ratio按 beta*A >= log(max_weight) 计算，避免浮点舍入误判。
关闭actor_advantage_weighting时实际weight=1、clip=0，同时记录假设开启时的截顶率。
positive/negative比例按A正负统计，near_zero按abs(A)<=阈值；三者不是互斥分组。
reward保留原计算，并额外记录扣血差、错防和KO奖励分量；td_target采用运行配置指定的N-step公式。
类别的mean_td_target是N-step TD估计，不能称为完整实测Return；不同N的结果不可直接横向比较。

## 动作类别

Joint144=(绝对方向-1)*16+ABCD bitmask，bit顺序A、D、B、C。
默认互斥分类：A与B/C同时按为MixedAttack；仅含A攻击为Melee；含B/C攻击为Projectile；
无攻击而含D为Dash；方向5全松为Neutral；其他无按键方向为Movement。
Dash是D输入，不保证游戏成功执行冲刺。Movement是方向输入，包含蹲/跳/后退，
不等于实际走路。攻击组直接按A/B/C是否按下定义，与自定义类别名字无关。
同一帧多键有明确归属；不把组合动作重复计入多个类别。
Flight/Defense/Skill/Spell/Walk仅凭该动作空间无法可靠识别，默认比率为null，不能解释为0。
可用diagnostics.action_classes覆盖ID归属；未列ID沿用默认归属，不允许重复ID。
Keyframe定义仍调用原build_changepoint_mask，keyframe_fraction的分母为该类别样本数。

## 切换率与固定状态

expert_switch_rate和predicted_switch_rate使用同一组连续相邻有效帧，排除终局、
片段边界及序列第一帧；比较predicted[t]与predicted[t-1]。
expert_keyframe_rate另按真实previous_expert_action统计，切片首帧有真实历史时可计入。
因此keyframe_sample_ratio与expert_switch_rate不必相等。
所有预测都基于专家轨迹历史，这是离线诊断，不等价于自由对局的动作切换率。

probe从validation以独立局部numpy RNG采样，训练随机流不受影响。
先固定候选batch，再按距离、HP差、空地、guard/hurt、压制事件和动作类别轮流取样。
近中远阈值使用原始relative_x绝对值，HP优劣势只表示当前HP差。
覆盖清单明确列出缺失条件；压制标签严格使用“对手防御且灵力下降”的事件定义，不把距离或攻击按键单独解释为压制。
probe身份哈希包含实际张量、采样配置、类别映射和选中帧。配置改变产生新的probe_id，
不同ID禁止做跨步趋势比较。同一ID在续训后可以继续比较；逐样本文件持续追加，需关注磁盘。
固定probe的类别比例经过覆盖抽样，不能代表自然数据频率；整体漂移结合完整validation报告判断。

## 读报告

每份报告的assessment对应需求的8项问题。先查看攻击/非攻击Advantage差、beta*A尺度，
再对比sample_fraction、final_weight_sum_fraction、effective_weight_sum_fraction；
结合截顶率、在线与target双Q分歧、近零比例及同一probe多步漂移判断。
警告阈值只是诊断阈值，可通过YAML配置，并不会自动改变训练参数。
Q一致不证明估值正确；攻击类Q高也可能来自真实数据回报，不能单凭相关性断言Critic bias。
当前尚无本次诊断产生的真实训练日志，因此不能预先判定偏攻击的根因。

本次按要求仅静态检查，未编译、运行测试、启动训练或修改已有模型。
