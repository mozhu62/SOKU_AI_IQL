"""固定地址单帧 CUDA Graph；真实多帧按顺序 replay，不丢弃历史。"""
import torch
from torch.nn import functional as F
from .streaming_tcn import StreamingTCN


class FixedCacheTCN(StreamingTCN):
    def __init__(self, model):
        super().__init__(model)
        self.work = {}

    def _conv(self, key, conv, values):
        delay = conv.dilation[0]
        if key not in self.cache:
            self.cache[key] = values.new_zeros(1, delay, values.shape[-1])
            self.work[key] = values.new_empty(1, delay + 1, values.shape[-1])
        cache, work = self.cache[key], self.work[key]
        # 工作区与缓存分离，避免重叠 copy；地址固定，不逐帧 cat/clone。
        work[:, :delay].copy_(cache)
        work[:, delay:].copy_(values)
        result = F.conv1d(work.transpose(1, 2), conv.weight, conv.bias,
                          dilation=delay).transpose(1, 2)
        cache.copy_(work[:, 1:])
        return result


class GraphTCN:
    @torch.inference_mode()
    def __init__(self, model, device, amp):
        self.model = model
        self.seeded = False
        self.core = FixedCacheTCN(model)
        self.batch = BatchCacheTCN(model)
        self.last_path = 'not_run'
        self.input = torch.zeros(1, 1, model.input_dim, device=device)
        self.graph = torch.cuda.CUDAGraph()
        # 只在加载模型时预热/捕获，不能在游戏逐帧控制中首次编译或捕获。
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.float16,
                                                           enabled=amp, cache_enabled=False):
                for _ in range(3):
                    self.core.forward(self.input)
            stream.synchronize()
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp, cache_enabled=False):
                with torch.cuda.graph(self.graph, stream=stream):
                    self.output = self.core.forward(self.input)
            torch.cuda.current_stream(device).wait_stream(stream)

    def reset(self):
        self.seeded = False

    @torch.inference_mode()
    def forward(self, features):
        if not self.seeded:
            self.last_path = 'window_seed_eager'
            # 完整窗口用批量参考前向初始化所有层缓存，捕获预热的假历史不进入实战。
            reference = StreamingTCN(self.model)
            result = reference.forward(features)
            for key, value in reference.cache.items():
                self.core.cache[key].copy_(value)
            self.seeded = True
            return result[:, -1:]
        if features.shape[1] > 1:
            self.last_path = 'batch_eager'
            # 共用固定缓存张量，批量路径直接原地更新，不再 clone 后二次回写。
            self.batch.cache = self.core.cache
            result = self.batch.forward(features)
            return result[:, -1:]
        self.last_path = 'single_frame_graph'
        self.input.copy_(features)
        self.graph.replay()
        return self.output


class BatchCacheTCN(StreamingTCN):
    def __init__(self, model):
        super().__init__(model)
        self.work = {}

    def _conv(self, key, conv, values):
        delay, length = conv.dilation[0], values.shape[1]
        cache = self.cache[key]
        shape = (values.shape[0], values.shape[2], delay + length)
        work = self.work.get(key)
        # 每层只留最近形状的工作区，避免按所有历史长度永久缓存；NCT连续输入。
        if work is None or tuple(work.shape) != shape:
            work = values.new_empty(shape)
            self.work[key] = work
        work[:, :, :delay].copy_(cache.transpose(1, 2))
        work[:, :, delay:].copy_(values.transpose(1, 2))
        output = F.conv1d(work, conv.weight, conv.bias, dilation=delay).transpose(1, 2)
        cache.copy_(work[:, :, -delay:].transpose(1, 2))
        return output
