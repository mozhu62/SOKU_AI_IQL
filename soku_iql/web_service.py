from __future__ import annotations

import asyncio
import errno
import ipaddress
import socket
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT
from .workbench import EDITABLE
from .bc_core.action_space import action_catalog

WEB_DIST = ROOT / "web" / "dist"


class Command(BaseModel):
    id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    name: str
    value: dict = Field(default_factory=dict)


def bind_port(host, first, attempts):
    """检测即绑定并保留 socket，避免先检测后启动之间端口被其他进程抢占。"""
    failures = []
    for port in range(first, min(65536, first + attempts)):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind((host, port))
            listener.listen(128)
            listener.setblocking(False)
            return listener, port, failures
        except OSError as error:
            listener.close()
            if error.errno not in (errno.EADDRINUSE, errno.EACCES, 10048, 10013):
                raise
            failures.append(port)
    raise OSError(f"本机端口 {first} 起的 {attempts} 个候选均无法绑定，请用 --port 指定其他起始端口")


def _loopback_client(client_host):
    try:
        return ipaddress.ip_address(client_host or "").is_loopback
    except ValueError:
        return False


def _private_host(value, port, *, client_host=None):
    try:
        parsed = urlsplit("//" + value)
        if (parsed.username is not None or parsed.password is not None or parsed.path or parsed.query
                or parsed.fragment or parsed.port is None or not 1 <= parsed.port <= 65535):
            return False
        name = parsed.hostname or ""
        if name == "localhost":
            local_name = True
        else:
            address = ipaddress.ip_address(name)
            if not (address.is_private or address.is_loopback or address.is_link_local):
                return False
            local_name = address.is_loopback
        # SSH -L 保留浏览器的 Host，客户端端口可以不同；仅允许回环连接使用回环名称转发。
        forwarded_local = local_name and _loopback_client(client_host)
        return parsed.port == port or forwarded_local
    except ValueError:
        return False


def _same_origin(origin, host):
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        return (parsed.scheme == "http" and parsed.netloc.lower() == host.lower()
                and not parsed.path and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def _lan_addresses():
    addresses = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = ipaddress.ip_address(item[4][0])
            if address.is_private and not address.is_loopback:
                addresses.add(str(address))
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.0.2.1", 9))
        address = ipaddress.ip_address(probe.getsockname()[0])
        probe.close()
        if address.is_private and not address.is_loopback:
            addresses.add(str(address))
    except OSError:
        pass
    return sorted(addresses)


def model_catalog(runtime):
    state = runtime.snapshot()
    if not state.get("output"):
        return []
    root = Path(state["output"]).resolve()
    rows = []
    for pattern in ("*.pt", "snapshots/*.pt"):
        for path in root.glob(pattern):
            actual = path.resolve()
            if actual.is_file() and actual.is_relative_to(root):
                stat = actual.stat()
                rows.append(dict(id=hashlib.sha256(str(actual).encode()).hexdigest()[:24],
                                 name=actual.relative_to(root).as_posix(), bytes=stat.st_size, modified=stat.st_mtime,
                                 kind="策略/实战" if "actor" in actual.name else "完整IQL/续训"))
    return sorted(rows, key=lambda row: row["modified"], reverse=True)[:200]


def create_app(runtime, port):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        peer = request.client.host if request.client else None
        if (not _private_host(host, port, client_host=peer) or not _same_origin(request.headers.get("origin"), host)
                or request.headers.get("sec-fetch-site") == "cross-site"):
            return JSONResponse({"detail": "仅允许本机/私有局域网同源访问，可通过 SSH 转发"}, status_code=403)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "Content-Security-Policy":
                                 "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                                 "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'"})
        return response

    @app.get("/api/status")
    def status():
        return {**runtime.snapshot(), "ui_mode": "iql_training"}

    @app.get("/api/history")
    def history(kind="train", offset: int = 0, limit: int = 300):
        try:
            return runtime.history_page(kind, max(0, offset), max(1, min(1000, limit)))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/parameters")
    def parameters():
        state = runtime.snapshot()
        return dict(config=state["config"], source=state["config_source"],
                    fields=[dict(key=k, min=v[0], max=v[1], type=v[2], label=v[3]) for k, v in EDITABLE.items()])

    @app.get("/api/config")
    def export_config():
        cfg = runtime.snapshot()["config"]
        return JSONResponse({k: cfg[k] for k in ("data", "training", "iql", "reward", "output", "actor_sampling", "actor_weighting") if k in cfg},
                            headers={"Content-Disposition": 'attachment; filename="iql_runtime_config.json"'})

    @app.get("/api/actions")
    def actions():
        return dict(catalog=action_catalog(), counts=runtime.snapshot().get("data", {}).get("train_action_counts", []))

    @app.get("/api/models")
    def models():
        return dict(rows=model_catalog(runtime))

    @app.get("/api/models/{identifier}")
    def download_model(identifier: str):
        row = next((item for item in model_catalog(runtime) if item["id"] == identifier), None)
        if row is None:
            raise HTTPException(404, "模型不存在或未登记")
        root = Path(runtime.snapshot()["output"]).resolve()
        if not row["name"].startswith("snapshots/") and runtime.snapshot()["state"] not in ("paused", "completed", "stopped", "error"):
            raise HTTPException(409, "请暂停训练后下载会被覆盖的模型，或下载固定 snapshots 版本")
        path = (root / row["name"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(403, "拒绝越界文件访问")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.post("/api/commands")
    def command(value: Command):
        try:
            return runtime.submit(value.id, value.name, value.value)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/commands/{identifier}")
    def request_status(identifier: str):
        value = runtime.request(identifier)
        if value is None:
            raise HTTPException(404, "请求不存在")
        return value

    @app.websocket("/api/stream")
    async def stream(websocket: WebSocket):
        host = websocket.headers.get("host", "")
        peer = websocket.client.host if websocket.client else None
        if (not _private_host(host, port, client_host=peer) or not _same_origin(websocket.headers.get("origin"), host)
                or websocket.headers.get("sec-fetch-site") == "cross-site"
                or websocket.headers.get("sec-websocket-protocol") != "soku-iql"):
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol="soku-iql")
        try:
            while True:
                await websocket.send_json(status())
                await asyncio.sleep(.5)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass

    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="iql-training")
    return app


def run_workbench(runtime, host="127.0.0.1", first_port=8806, attempts=10):
    import uvicorn
    if not (WEB_DIST / "index.html").is_file():
        raise RuntimeError("IQL 前端尚未构建：npm --prefix web ci，再执行 npm --prefix web run build；无界面训练可用 --headless")
    listener, port, occupied = bind_port(host, first_port, attempts)
    try:
        if occupied:
            print(f"端口 {occupied} 不可用，已使用 {port}", flush=True)
        server = uvicorn.Server(uvicorn.Config(create_app(runtime, port), host=host, port=port,
                                               access_log=False, proxy_headers=False))
        print(f"IQL 训练工作台：http://127.0.0.1:{port}/", flush=True)
        print(f"SSH：ssh -N -L 8806:127.0.0.1:{port} 用户名@服务器地址；本地浏览器 http://localhost:8806/", flush=True)
        if host == "0.0.0.0":
            print(f"已开放可信局域网，无身份认证：http://服务器局域网IP:{port}/；不要暴露公网", flush=True)
        runtime.start()
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            pass
        finally:
            runtime.close()
    finally:
        listener.close()
