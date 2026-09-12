import pytest
import torch
from torch import nn

from soku_iql.bc_core.config import MODEL_DEFAULTS, context_frames_for
from soku_iql.bc_core.models import network_spec
from soku_iql.bc_core.temporal import TemporalConvEncoder
from soku_iql.models import expand_tcn_weights
from soku_iql.live.tcn_window import TCNObservationWindow


def tiny_network(context):
    model = nn.Module()
    model.tcn = TemporalConvEncoder(4, 8, 8, context_frames=context)
    model.head = nn.Linear(8, 2)
    return model


def test_spec_and_receptive_field():
    spec = network_spec({**MODEL_DEFAULTS, 'temporal_mode': 'tcn256'})
    assert spec['temporal']['context_frames'] == 256
    assert spec['temporal']['dilations'] == [1, 2, 4, 8, 16, 32, 64]
    assert 1 + 1 + 2*sum(spec['temporal']['dilations']) == 256
    assert context_frames_for(MODEL_DEFAULTS) == 32
    assert context_frames_for({**MODEL_DEFAULTS, 'temporal_mode': 'tcn64'}) == 64


@pytest.mark.parametrize('source_context', [32, 64])
def test_reuse_every_old_layer_only_append_blocks(source_context):
    source, target = tiny_network(source_context), tiny_network(256)
    expand_tcn_weights(target, source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    broken = dict(source.state_dict())
    del broken['tcn.blocks.0.conv1.weight']
    with pytest.raises(ValueError):
        expand_tcn_weights(target, broken)
    with pytest.raises(ValueError):
        expand_tcn_weights(source, target.state_dict())


def test_causality_and_window_equivalence():
    torch.manual_seed(4)
    model = TemporalConvEncoder(4, 8, 8, context_frames=256).eval()
    data = torch.randn(1, 317, 4)
    with torch.no_grad():
        full = model(data)
        window = model(data[:, -256:])
        torch.testing.assert_close(full[:, -1], window[:, -1], atol=1e-5, rtol=1e-5)
        changed = data.clone()
        changed[:, 270:] += 10
        torch.testing.assert_close(model(changed)[:, :270], full[:, :270])


def test_live_buffer_and_short_history_rejection():
    window = TCNObservationWindow(256)
    assert window.rows.maxlen == 256
    with pytest.raises(ValueError, match='256'):
        window.logits(None, 'cpu')
