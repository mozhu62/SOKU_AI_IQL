from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path

from ..config import ROOT, resolve


class Repository:
    """沿用 PPO 的文件登记方式：网页只能选择已登记模型/评估报告，不能提交任意路径。"""

    def __init__(self, config):
        self.model_root = (ROOT / "outputs").resolve()
        self.explicit_model = resolve(config["checkpoint"]).resolve()
        self.report_root = resolve(config["output"]).resolve()
        self.lock = threading.RLock()
        self.registry = {}
        self.catalog = {}
        self.catalog_at = 0.0
        self.cache = {}
        self.selected_models = {}

    def register_model(self, path: Path):
        """仅由原生文件选择结果调用；授权单个文件，不扩大到它所在的整个目录。"""
        actual = path.resolve(strict=True)
        if not actual.is_file() or actual.suffix.lower() != ".pt":
            raise ValueError("请选择已经存在的 IQL/BC .pt 模型文件")
        identifier = hashlib.sha256(str(actual).encode()).hexdigest()[:24]
        with self.lock:
            self.selected_models[identifier] = actual
            self.catalog_at = 0.0
            catalog = self.files()
        for row in catalog["models"]:
            if row["id"] == identifier:
                return row
        raise ValueError("选中的文件已移动、删除或无法访问，请重新选择")

    def files(self):
        with self.lock:
            if time.monotonic() - self.catalog_at < 2:
                return copy.deepcopy(self.catalog)
            registry, rows = {}, {"models": [], "evaluations": []}

            def register(kind, path, root):
                try:
                    actual = path.resolve()
                    if not actual.is_file() or not actual.is_relative_to(root):
                        return
                    stat = actual.stat()
                except OSError:
                    # 外置盘断开或模型正在移动时，略过这一个登记文件，不阻塞整个模型列表。
                    return
                identifier = hashlib.sha256(str(actual).encode()).hexdigest()[:24]
                if identifier in registry:
                    return
                registry[identifier] = kind, actual, root
                rows[kind].append({"id": identifier, "name": str(path.relative_to(root)),
                                   "path": str(actual), "filename": actual.name,
                                   "modified": stat.st_mtime, "bytes": stat.st_size})

            if self.explicit_model.is_file():
                register("models", self.explicit_model, self.explicit_model.parent)
            for path in self.selected_models.values():
                register("models", path, path.parent)
            for pattern in ("*.pt", "*/*.pt", "*/snapshots/*.pt"):
                for path in self.model_root.glob(pattern):
                    register("models", path, self.model_root)
            for path in self.report_root.glob("*/summary.json"):
                register("evaluations", path, self.report_root)
            for values in rows.values():
                values.sort(key=lambda row: row["modified"], reverse=True)
            self.registry, self.catalog, self.catalog_at = registry, rows, time.monotonic()
            return copy.deepcopy(rows)

    def path(self, identifier, kind):
        self.files()
        with self.lock:
            item = self.registry.get(identifier)
        if not item or item[0] != kind:
            raise ValueError("模型/报告未登记，请刷新列表")
        _, path, root = item
        if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("登记文件路径已变化，请重新启动服务")
        return path

    def _read(self, path: Path, *, lines=False):
        if path.resolve() != path or not path.is_relative_to(self.report_root):
            raise ValueError("拒绝访问评估目录以外的记录")
        stat = path.stat()
        if stat.st_size > 64 * 1024 * 1024:
            raise ValueError("报告超过网页读取上限，请离线查看")
        key = str(path), stat.st_mtime_ns, stat.st_size
        with self.lock:
            if key in self.cache:
                return copy.deepcopy(self.cache[key])
        content = path.read_text(encoding="utf-8-sig")
        result = [json.loads(line) for line in content.splitlines() if line.strip()] if lines else json.loads(content)
        with self.lock:
            if len(self.cache) >= 9:
                self.cache.clear()
            self.cache[key] = result
        return copy.deepcopy(result)

    def report(self, identifier, offset=0, limit=100):
        path = self.path(identifier, "evaluations")
        summary = self._read(path)
        session = self._read(path.parent / "session.json")
        rounds_path = path.parent / "rounds.jsonl"
        results = self._read(rounds_path, lines=True) if rounds_path.is_file() else []
        config = session.get("config", {})
        env = config.get("environment", {})
        conditions = {"environment": {k: v for k, v in env.items() if k not in ("battle_mode", "battle_submode")},
                      "observed_modes": summary.get("observed_modes"),
                      "keyboard": config.get("keyboard"), "device": session.get("device"),
                      "action_selection": session.get("action_selection"),
                      "action_schema": session.get("action_schema"),
                      "metric": session.get("damage_metric")}
        return {"id": identifier, "name": path.parent.name, "summary": summary, "session": session,
                "conditions": conditions, "results": results[offset:offset + limit], "total": len(results)}
