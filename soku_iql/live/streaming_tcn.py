"""仅用于实战的增量 TCN；缓存不进入模型 state_dict，不改变训练前向。"""
import torch
from torch.nn import functional as F


class StreamingTCN:
    def __init__(self, model):
        self.model = model
        self.cache = {}
        # 当前结构每个 LayerNorm 只归一化最后的通道维；拒绝未来不兼容的结构。
        norms = [model.projection[1], model.stem_norm]
        norms += [norm for block in model.blocks for norm in (block.norm1, block.norm2)]
        if any(not isinstance(norm, torch.nn.LayerNorm) or tuple(norm.normalized_shape) != (model.output_dim,) for norm in norms):
            raise ValueError('Streaming TCN 仅支持逐帧通道 LayerNorm')
        for conv in [model.stem] + [conv for block in model.blocks for conv in (block.conv1, block.conv2)]:
            if conv.kernel_size != (2,) or conv.stride != (1,) or conv.padding != (0,) or conv.groups != 1:
                raise ValueError('Streaming TCN 结构不匹配，禁止静默使用近似计算')

    @torch.inference_mode()
    def _conv(self, key, conv, values):
        # 缓存该卷积的输入而不是输出；kernel=2 时需保留 dilation 个输入位置。
        delay = conv.dilation[0]
        previous = self.cache.get(key)
        if previous is None:
            previous = values.new_zeros(values.shape[0], delay, values.shape[2])
        if previous.device != values.device or previous.dtype != values.dtype or previous.shape[0] != values.shape[0]:
            raise ValueError('Streaming 缓存设备/精度/批次变化，必须重新建立会话')
        joined = torch.cat((previous, values), dim=1)
        self.cache[key] = joined[:, -delay:].detach().clone()
        return F.conv1d(joined.transpose(1, 2), conv.weight, conv.bias,
                        dilation=delay).transpose(1, 2)

    @torch.inference_mode()
    def forward(self, features):
        if self.model.training or features.ndim != 3 or features.shape[1] < 1:
            raise ValueError('Streaming TCN 只接受 eval 模型和非空 [batch,time,features]')
        # 与训练前向的有效帧 mask 保持相同 dtype 提升规则，尤其是 autocast 下。
        values = self.model.projection(features) * features.new_ones(features.shape[:2] + (1,))
        values = F.silu(values + self.model.stem_norm(self._conv('stem', self.model.stem, values)))
        for index, block in enumerate(self.model.blocks):
            temporal = F.silu(block.norm1(self._conv((index, 1), block.conv1, values)))
            temporal = block.norm2(self._conv((index, 2), block.conv2, temporal))
            values = F.silu(values + temporal)
        return values
