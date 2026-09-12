"""工作台控制契约用例；新增源码，需用户另行授权后执行。"""
import copy
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from soku_iql.bc_core.config import DEFAULTS
from soku_iql.config import TRAINING, IQL, REWARD
from soku_iql.workbench import Workbench
from soku_iql.web_service import _private_host, _same_origin


def ready(tmp_path):
    w = Workbench("configs/iql_suika.yaml")
    cfg = dict(data=copy.deepcopy(DEFAULTS["data"]), training=copy.deepcopy(TRAINING),
               iql=copy.deepcopy(IQL), reward=copy.deepcopy(REWARD), output=dict(directory=str(tmp_path)))
    w.output = tmp_path
    w.publish(state="paused", ready=True, config=cfg)
    return w, cfg


def test_duplicate_commands_execute_once_at_boundary(tmp_path):
    w, cfg = ready(tmp_path)
    saved = []
    one = w.submit("save_request_1", "save", {})
    assert w.submit("save_request_1", "save", {}) == one
    assert w.queue.qsize() == 1
    w.submit("resume_request_1", "resume", {})
    assert w.boundary(lambda **kw: saved.append(kw), lambda: None, cfg, None, 0)
    assert saved == [{"snapshot": True}]
    assert w.request("save_request_1")["status"] == "done"


def test_runtime_parameters_do_not_change_algorithm(tmp_path):
    w, cfg = ready(tmp_path)
    store = SimpleNamespace(lock=threading.RLock(), cache=OrderedDict(), bytes=0, budget=0)
    original = copy.deepcopy(cfg["iql"])
    w.apply({"changes": {"training.prefetch_batches": 4}}, cfg, store, 10)
    assert cfg["training"]["prefetch_batches"] == 4
    assert cfg["iql"] == original
    assert (tmp_path / "runtime_config.json").is_file()
    with pytest.raises(ValueError):
        w.apply({"changes": {"iql.gamma": .5}}, cfg, store, 10)


def test_snapshots_are_detached_and_json_safe(tmp_path):
    w, _ = ready(tmp_path)
    w.publish(latest_train={"actor_gradient_norm": float("inf")})
    state = w.snapshot()
    assert state["latest_train"]["actor_gradient_norm"] is None
    state["latest_train"]["actor_gradient_norm"] = 12
    assert w.snapshot()["latest_train"]["actor_gradient_norm"] is None


def test_stop_during_initialization_is_allowed():
    w = Workbench("configs/iql_suika.yaml")
    with pytest.raises(ValueError):
        w.submit("resume_early", "resume", {})
    result = w.submit("stop_early", "stop", {})
    assert w.stop_event.is_set() and result["status"] == "queued"


def test_ssh_forwarding_and_origin_guards():
    assert _private_host("localhost:9000", 8806, client_host="127.0.0.1")
    assert not _private_host("localhost:9000", 8806, client_host="192.168.1.2")
    assert not _private_host("evil.example:8806", 8806, client_host="127.0.0.1")
    assert _same_origin("http://localhost:9000", "localhost:9000")
    assert not _same_origin("http://evil.example", "localhost:9000")
