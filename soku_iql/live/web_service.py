from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ..web_service import WEB_DIST, Command, bind_port, _same_origin
from .workbench import Workbench


def create_app(workbench, port):
    # 复用原战斗端的命令队列，网页线程不直接接触模型或键盘。
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "").lower()
        if (host not in allowed_hosts or not _same_origin(request.headers.get("origin"), host)
                or request.headers.get("sec-fetch-site") == "cross-site"):
            return JSONResponse({"detail": "实战键盘控制仅允许本机同源页面；不需要 token"}, status_code=403)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "Content-Security-Policy":
                                 f"default-src 'self'; connect-src 'self' ws://{host}; img-src 'self' data:; "
                                 "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'"})
        return response

    @app.get("/")
    @app.get("/index.html")
    def index():
        # 根地址直接提供战斗页，不要求 token，也不需要记忆额外路径。
        return FileResponse(WEB_DIST / "play.html")

    @app.get("/api/status")
    def status():
        return workbench.snapshot()

    @app.get("/api/parameters")
    def parameters():
        return {"config": workbench.snapshot()["runtime_config"], "source": "本机实战配置；加载模型后生效"}

    @app.get("/api/files")
    def files():
        return workbench.repository.files()

    @app.get("/api/records/{identifier}")
    def record(identifier: str, offset: int = 0, limit: int = 100):
        try:
            return workbench.repository.report(identifier, max(0, offset), max(1, min(500, limit)))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/history")
    def history(kind: str = "rounds", limit: int = 100):
        if kind != "rounds":
            raise HTTPException(400, "实战服务只有小局历史，没有训练损失")
        rows = workbench.snapshot().get("rounds", [])
        return {"rows": rows[-max(1, min(100, limit)):], "total": len(rows)}

    @app.post("/api/commands")
    def command(value: Command):
        try:
            return workbench.submit(value.id, value.name, value.value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/commands/{identifier}")
    def result(identifier: str):
        result = workbench.request(identifier)
        if result is None:
            raise HTTPException(404, "请求不存在")
        return result

    @app.websocket("/api/stream")
    async def stream(websocket: WebSocket):
        host = websocket.headers.get("host", "").lower()
        protocols = [part.strip() for part in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        if (host not in allowed_hosts or not _same_origin(websocket.headers.get("origin"), host)
                or websocket.headers.get("sec-fetch-site") == "cross-site"
                or protocols != ["soku-iql"]):
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol="soku-iql")
        try:
            while True:
                await websocket.send_json(workbench.snapshot())
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass

    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="bc-workbench")
    return app


def run_workbench(config, auto_start=False):
    import uvicorn
    if not (WEB_DIST / "play.html").is_file():
        raise RuntimeError("浏览器实战页面尚未构建：在 soku_iql 目录手动执行 npm --prefix web run build；不再使用 Tk 界面")
    web = config["web"]
    listener, port, occupied = bind_port(web["host"], web["port"], web["port_attempts"])
    workbench = Workbench(config, auto_start)
    try:
        if occupied:
            print(f"端口不可用 {occupied}，已切换到 {port}", flush=True)
        print(f"IQL 实战工作台：http://localhost:{port}/", flush=True)
        print("顶部选择本机模型 → 加载模型 → 继续；换模型不需要重启或改命令行。F10 松键，Ctrl+C 关闭服务。", flush=True)
        app = create_app(workbench, port)
        server = uvicorn.Server(uvicorn.Config(app, host=web["host"], port=port, access_log=False, proxy_headers=False))
        workbench.start()
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            # 正常退出仍经过 finally 松键和保存报告，不输出无关的取消堆栈。
            pass
        finally:
            workbench.close()
    finally:
        listener.close()
