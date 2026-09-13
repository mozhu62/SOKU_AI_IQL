"""已离线验证的实战批量 Graph；各长度共享缓存，工作区各自固定。"""
import torch
from .graph_tcn import GraphTCN, BatchCacheTCN


class BatchGraphCandidate(GraphTCN):
    @torch.inference_mode()
    def __init__(self, model, device, amp, lengths=(2, 3, 4, 8, 18)):
        super().__init__(model, device, amp)
        self.graphs = {}
        with torch.cuda.device(device):
            for length in lengths:
                core = BatchCacheTCN(model)
                core.cache = self.core.cache
                inputs = torch.zeros(1, length, model.input_dim, device=device)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.float16,
                                                               enabled=amp, cache_enabled=False):
                    for _ in range(3):
                        core.forward(inputs)
                stream.synchronize()
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp, cache_enabled=False):
                    with torch.cuda.graph(graph, stream=stream):
                        output = core.forward(inputs)
                torch.cuda.current_stream(device).wait_stream(stream)
                self.graphs[length] = (core, inputs, graph, output)

    @torch.inference_mode()
    def forward(self, features):
        if not self.seeded or features.shape[1] not in self.graphs:
            return super().forward(features)
        _, inputs, graph, output = self.graphs[features.shape[1]]
        inputs.copy_(features)
        graph.replay()
        self.last_path = 'batch_graph'
        return output[:, -1:]
