"""FastAPI entry point.

- wipes the job cache directory on startup (intermediate files never persist)
- serves the REST/SSE API under /api
- serves the MCP server under /mcp (off by default)
- serves the built frontend (frontend/dist) as the web UI

Run it as `python -m app.main`, not `uvicorn app.main:app`: the bind address
and port come from the saved settings, and only main() reads them.
"""

from __future__ import annotations

import os

# must be set before anything imports huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core import config, server
from app.core.auth import MCP_PREFIX, AccessGuard
from app.core.cache import clear_cache
from app.services import mcp_server

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    clear_cache()
    async with contextlib.AsyncExitStack() as stack:
        mcp = mcp_server.build()
        if mcp is not None:
            mcp_server.mount.inner = mcp.streamable_http_app()
            # required by the SDK: without it the first /mcp request fails.
            # Entered unconditionally — whether the endpoint answers at all
            # is the access guard's call, so the switch in settings takes
            # effect without a restart.
            await stack.enter_async_context(mcp.session_manager.run())
        try:
            yield
        finally:
            mcp_server.mount.inner = None


app = FastAPI(title="MovieTranslator", lifespan=lifespan)
app.add_middleware(AccessGuard)
app.include_router(router)

# Must be mounted before the SPA fallback below: routes match in
# registration order, and the fallback would otherwise swallow the GET the
# MCP transport uses to open its stream.
app.mount(MCP_PREFIX, mcp_server.mount)

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        # A mistyped or removed endpoint must not come back as the page
        # shell: api.js checks resp.ok first, so a 200 full of HTML surfaces
        # as a JSON syntax error instead of "no such endpoint".
        if full_path.startswith("api/") or full_path.split("/")[0] == "mcp":
            raise HTTPException(status_code=404, detail="接口不存在")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


def _startup_banner(settings) -> str:
    port = settings.server.port
    lines = [f"[MovieTranslator] 本机访问 http://127.0.0.1:{port}"]
    if not settings.server.lan_access:
        return "\n".join(lines)

    token = settings.server.access_token if settings.server.require_token else ""
    suffix = f"/?token={token}" if token else "/"
    for ip in server.lan_ips():
        lines.append(f"[MovieTranslator] 局域网访问 http://{ip}:{port}{suffix}")
    if not token:
        lines.append("[MovieTranslator] 警告：局域网访问已开启且未要求令牌，"
                     "同网段任何人都能浏览本机文件并发起任务")
    if settings.mcp.enabled:
        for ip in server.lan_ips():
            lines.append(f"[MovieTranslator] MCP 服务 http://{ip}:{port}{MCP_PREFIX}")
    return "\n".join(lines)


def main() -> None:
    import uvicorn

    settings = config.load_settings()
    if server.ensure_token(settings):
        config.save_settings(settings)
        settings = config.load_settings()

    host = server.bind_host(settings)
    port = settings.server.port
    server.RUNNING.update({"host": host, "port": port})
    print(_startup_banner(settings), flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
