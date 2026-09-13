# IQL 实战 TCN 加速

本次复用 BC 已离线验证并完成用户实战验证的增量 TCN、固定缓存及批量 Graph 实现，复制为 IQL 自有模块，不依赖旁边的 BC 工程。IQL 的 TCN 为逐帧通道 LayerNorm，kernel=2，结构符合缓存算法要求。保留32/64/256帧兼容。

configs/live_eval.yaml 默认 hybrid、amp=true、streaming_tcn=true、tcn_cuda_graph=true。CPU承担状态、对象编码和输出；CUDA:0执行TCN。GPU不可用时hybrid明确报错。可改cpu/cuda，或关闭Graph、增量模式做对照。FP16只作用CUDA，不改变保存的FP32权重或embedding索引。

首次真实窗口初始化走window_seed_eager；1帧single_frame_graph；2/3/4/8/18帧batch_graph；其他长度batch_eager。所有实际新增帧按顺序进入历史，不重复补假帧。换局/缺帧重置历史，下一次满窗重新初始化缓存；图捕获仅发生在模型加载时。捕获失败拒绝加载，不暗中回退。

网页参数页显示实际路径、新增帧数、精度，配置修改需暂停并重新加载。训练算法、reward、N-step、Actor权重和checkpoint加载逻辑不变；没有迁移BC模型参数。

本次仅源码接入和静态检查，未运行IQL模型测试、前端编译或游戏。BC结果不能当作IQL性能验收；需使用自己的IQL策略比较输出和端到端P50/P95。无需重训，重启服务；前端改动需手动 npm --prefix web run build。
