# IQL 训练与实战工作台

## 安装与构建（用户手动执行）

在 `soku_iql` 项目根目录，使用已有可用的 PyTorch/CUDA 环境、Python 3.11+、Node 20.19+ 或 22.12+：

```powershell
pip install -e .
npm --prefix web ci
npm --prefix web run build
```

前端复用 BC 的 React、TypeScript、Recharts 和已有 shadcn/Radix 组件，包版本及 lockfile 保留。构建同时生成 `web/dist/index.html` 和 `play.html`；不从 CDN 加载脚本或字体，不需要另开 Node 服务。缺少构建产物时明确报错，不以空白页冒充工作台。

## 训练入口

```powershell
python scripts/train.py --config configs/iql_suika.yaml --init-bc "实际BC模型.pt"
```

默认地址 `http://127.0.0.1:8806/`，启动时立即提供网页，然后后台校验数据和初始化模型；准备完毕保持暂停，点击“开始 / 继续”。希望准备后自动训练可加 `--start`。旧命令行自动训练方式加 `--headless`，不会启动网页。

续训：

```powershell
python scripts/train.py --config configs/iql_suika.yaml --resume outputs/iql_suika/last.pt
```

端口占用会实际绑定并依次尝试后续 9 个端口，以终端打印的地址为准；也可用 `--port` 指定起始端口。

### 服务器与 SSH

服务器保持默认回环监听，在本地电脑执行：

```text
ssh -N -L 8806:127.0.0.1:8806 用户名@服务器地址
```

然后访问 `http://localhost:8806/`，不带 token、不加路径。如果服务器最终使用的端口不是 8806，修改第二个端口；本地映射端口也可不同。

可信局域网需要直接访问时加 `--host 0.0.0.0`。此模式没有身份认证，访问者能控制训练和下载登记模型；仅用于可信私网，不能暴露公网。保留 Host/Origin 同源限制，禁止跨站控制和任意路径下载。

### 三个训练页面

- 总览：更新次数、Actor 实际更新次数、预热/联合训练状态、有效帧/秒、验证 NLL/Top-1、复制基线、切换准确率、分阶段耗时。
- 学习诊断：Actor 加权 CE、双 Q TD MSE 和、V expectile、梯度范数、优势权重、验证 TD MSE/MAE/EV，以及全部 144 个训练数据动作频率。
- 参数与模型：运行值/编辑值、只读算法定义、参数修改记录、模型列表、下载运行配置和模型。

TD EV 比较 min(Q1,Q2) 与当前一步 TD 目标，不是实战回报 EV。目标方差为零或无动作切换帧时显示不适用/未记录。NLL 最佳仍不等于实战最强。

### 控制与配置

开始、暂停、保存、验证均进入训练线程命令队列，在完整更新边界执行。重复请求编号只执行一次。停止准备阶段会取消分片扫描；模型就绪后停止会等待完整更新/验证结束再保存。关闭浏览器不停止训练，重连不自动恢复暂停。

“保存版本”同时更新 last.pt/actor_bc.pt，并写入不会被覆盖的 snapshots 版本。下载 last.pt 等可变文件前暂停训练；固定 snapshots 可以直接下载。

可在线修改：total_steps、log_interval、save_interval、validation_interval、prefetch_batches、cpu_threads、cache_gb。应用前必须暂停；排空并丢弃旧预取，以原 step 的种子重建，不改变训练数据抽样顺序。应用后保存模型、config.json、runtime_config.json 和 configuration.jsonl，**不写原 YAML**。

奖励、gamma、IQL 损失参数、学习率、batch、序列长度、验证批次数和网络结构只读；本轮没有实现自定义冻结实验或改变算法的在线调参。不要把只读展示误认为已应用修改。

要恢复前端修改后的运行参数：

```powershell
python scripts/train.py --config outputs/iql_suika/runtime_config.json --resume outputs/iql_suika/last.pt
```

历史图默认最近 300 条，服务缓存最近 2000 条，完整日志保留磁盘。准备时历史只加载一次，不在每次网页刷新扫描全部日志。实时状态 2Hz、历史图 1Hz；暂停仅累计等待命令的时间，不包含手动验证或保存。

## Windows 实战入口

```powershell
python scripts/play.py --config configs/live_eval.yaml
```

默认 `http://localhost:8826/`，端口占用时自动递增。此服务只能在 Windows 游戏电脑运行，只监听本机。Linux 训练得到模型后，下载或复制到 Windows。

顶部“选择本机模型”打开 Windows 文件选择器，选择后点击“加载模型”，然后“继续”并切回游戏。也可从已登记输出目录的模型列表选择，不需要修改命令行才能换模型。取消文件选择、加载失败不会自动恢复持键。

支持模型：

- `actor_bc.pt`、`best_actor_bc.pt`：推荐的 IQL 实战策略包。
- 完整 IQL `last.pt` / snapshots：只提取 Actor，不构造 Q/V、不恢复训练优化器；加载时需读取较大的文件。
- 当前 Joint144 BC checkpoint：作为初始基线对照。

不支持旧 Joint432、CQL Q 头、DQfD 或 PPO 模型。网络、归一化、144 类单帧按键语义仍沿用 BC。

实战页面展示完整动作 logits/概率排序（可选全部 144 类）、九宫格方向、ABCD 模型要求/当前发送/游戏回读、实际游戏 actionId、32 帧窗口状态、推理 P50/P95、逐局伤害差与胜负、历史报告对照。softmax 只用于显示，实际选动作仍是 argmax。

原 BC 的 DLL 协议、LiveFrames.v1 连续帧队列、资源读取与按键控制已迁入 IQL 私有模块；无需安装 BC Python 项目。游戏仍需要此前适配 BC 的采集 DLL（含 LiveFrames.v1），本次不编译、替换或自动安装 DLL。不能取得真实连续 32 帧时暂停等待，不以稀疏观测或伪造帧补齐。

F10、失焦、推理/观测超时、停止均沿用原安全松键逻辑；控制器互斥锁与原 PPO/BC 名称一致，不允许同时控制同一个游戏。实战报告写入独立 evaluations 目录，模型和离线训练数据不变。

## 本轮交付与待验收

本轮只完成源码开发和静态核对，未安装前端依赖、未构建、未运行新增测试、未启动网页服务或游戏。之前的 IQL 离线验收记录不覆盖这次新增控制与界面。

后续人工验收：构建双入口；训练暂停/继续/保存/停止；重复请求只执行一次；准备过程中取消；断网重连；在线改参并用快照续训；SSH 不同本地端口；实战加载基线/Actor/完整IQL包；F10/失焦松键；全量144输出与ABCD回读；1024×768和1366×768下局部滚动。
