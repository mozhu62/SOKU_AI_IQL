import pytest

from soku_iql.amp_state import restore_grad_scaler


class ScalerStub:
    def __init__(self, enabled):
        self.enabled = enabled
        self.loaded = None

    def is_enabled(self):
        return self.enabled

    def load_state_dict(self, state):
        if not state or 'scale' not in state:
            raise RuntimeError('invalid scaler state')
        self.loaded = state.copy()


def test_cpu_migration_to_cuda_keeps_new_scaler(caplog):
    scaler = ScalerStub(True)
    assert restore_grad_scaler(scaler, {}, 'actor') == 'initialized'
    assert scaler.loaded is None
    assert 'GradScaler 状态为空' in caplog.text


def test_cuda_resume_restores_saved_scale():
    scaler = ScalerStub(True)
    state = dict(scale=32768., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=7)
    assert restore_grad_scaler(scaler, state, 'critic') == 'restored'
    assert scaler.loaded == state


@pytest.mark.parametrize('state', [{}, {'scale': 32768.}])
def test_disabled_scaler_does_not_restore(state):
    scaler = ScalerStub(False)
    assert restore_grad_scaler(scaler, state, 'value') == 'disabled'
    assert scaler.loaded is None


def test_bad_nonempty_state_is_not_silently_reset():
    with pytest.raises(RuntimeError):
        restore_grad_scaler(ScalerStub(True), {'broken': True}, 'actor')
    with pytest.raises(ValueError):
        restore_grad_scaler(ScalerStub(True), None, 'actor')
