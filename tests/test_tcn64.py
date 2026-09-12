import copy

import pytest
import torch

from soku_iql.bc_core.config import MODEL_DEFAULTS
from soku_iql.bc_core.models import BCNetwork
from soku_iql.bc_core.temporal import TemporalConvEncoder
from soku_iql.models import expand_tcn32_weights
from soku_iql.live.tcn_window import TCNObservationWindow


def test_structure_and_explicit_migration():
    old = BCNetwork(copy.deepcopy(MODEL_DEFAULTS))
    cfg = {**MODEL_DEFAULTS, 'temporal_mode': 'tcn64'}
    new = BCNetwork(cfg)
    assert new.spec['temporal']['context_frames'] == 64
    assert new.spec['temporal']['dilations'] == [1, 2, 4, 8, 16]
    assert 1 + 1 + sum(2 * block.dilation for block in new.tcn.blocks) == 64
    expand_tcn32_weights(new, old.state_dict())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    with pytest.raises(ValueError):
        expand_tcn32_weights(old, new.state_dict())


def test_causal_window_not_full_sequence():
    torch.manual_seed(3)
    model = TemporalConvEncoder(4, 8, 8, context_frames=64).eval()
    values = torch.randn(1, 80, 4)
    with torch.no_grad():
        full = model(values)
        window = model(values[:, -64:])
        torch.testing.assert_close(full[:, -1], window[:, -1], atol=1e-5, rtol=1e-5)
        changed = values.clone()
        changed[:, 70:] += 10
        torch.testing.assert_close(model(changed)[:, :70], full[:, :70])


def test_live_window_keeps_64_real_frames():
    window = TCNObservationWindow(64)
    assert window.rows.maxlen == 64
    assert TCNObservationWindow(32).rows.maxlen == 32
