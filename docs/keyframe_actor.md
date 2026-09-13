# IQL Actor关键帧加权

复用BC的build_changepoint_mask：比较当前专家动作和Dataset保存的真实上一帧专家动作，不使用模型预测，不在batch内shift。START=144、断帧、终局及episode边界没有合法上一帧时，不算切换；切片首项有真实前驱时正常比较。padding排除。

当前配置enabled=true、changepoint_weight=32，与当前BC配置一致；保持帧权重1。Actor优势恒1实验及无预热保持现状，Neutral加权/筛选仍关闭。因此当前loss=sum(k*CE)/sum(k)，是关键帧BC，而非普通等权BC。Q/V依旧全部有效转移、单步TD；不引入SMDP、不改变采样/历史/网络。

若以后打开优势加权，loss=sum(k*c*w*CE)/sum(k*c)，w不进入分母。32倍是相同优势和类别条件下的相对权重，不保证每个关键帧最终权重都大于所有保持帧，也不保证总梯度贡献占多数。

日志新增keyframe_weighting_enabled、changepoint_weight、actor_changepoint_samples、actor_history_eligible_samples；当前actor_objective=keyframe_bc。验证切换帧计数复用同一mask，总体NLL/Top1保持等权且best选择不变。

旧配置缺项默认关闭。更换关键帧目标建议--init-bc新目录，旧IQL续训目标不一致时拒绝。已新增边界、1.8加权示例、weight=1等价测试源码，按要求未运行测试/训练。
