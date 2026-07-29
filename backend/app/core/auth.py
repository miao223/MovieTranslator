"""Who is allowed to reach this server once it stops being loopback-only.

A plain ASGI middleware rather than BaseHTTPMiddleware on purpose: the job
progress endpoint is an open-ended SSE stream and MCP can stream too, and
BaseHTTPMiddleware wraps responses in a way that has never been kind to
either.

The rule set is deliberately small:

- loopback is always allowed, so the person at the machine can never be
  locked out and can always read the token off the settings page;
- with LAN access off, anything non-loopback is refused outright — that is
  the state the app ships in, and the guard says so rather than relying on
  the bind address alone;
- otherwise a token is required unless the user turned that off.

Browsers get in via a one-time `?token=` link that is converted into a
cookie. That matters more than it looks: EventSource and plain download
links cannot carry a custom header, so a header-only scheme would have
meant rewriting how the frontend fetches progress and results.
"""

from __future__ import annotations

import ipaddress
import json
import secrets
from urllib.parse import parse_qs, urlencode

from app.core import config

COOKIE_NAME = "mt_token"
HEADER_NAME = b"x-mt-token"
COOKIE_MAX_AGE = 31536000  # a year; the token itself is the expiry control

# MCP is mounted here; while the switch is off the path must not exist
MCP_PREFIX = "/mcp"

_DENY_HTML = """<!doctype html><meta charset="utf-8">
<title>MovieTranslator</title>
<style>body{{font-family:system-ui,sans-serif;margin:12vh auto;max-width:32em;
line-height:1.8;color:#333}}code{{background:#eee;padding:2px 6px;border-radius:4px}}</style>
<h2>{title}</h2><p>{body}</p>
"""


def _is_loopback(client) -> bool:
    if not client:
        return False  # no peer address: treat as remote, never as trusted
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return ""


def _presented_token(scope) -> tuple[str, str]:
    """The token the caller offered, and where it came from."""
    cookies = ""
    for key, value in scope.get("headers", []):
        if key == b"authorization":
            raw = value.decode("latin-1").strip()
            if raw.lower().startswith("bearer "):
                return raw[7:].strip(), "header"
        elif key == HEADER_NAME:
            return value.decode("latin-1").strip(), "header"
        elif key == b"cookie":
            cookies = value.decode("latin-1")
    for part in cookies.split(";"):
        name, _, value = part.strip().partition("=")
        if name == COOKIE_NAME and value:
            return value, "cookie"
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if query.get("token"):
        return query["token"][0], "query"
    return "", ""


def _redirect_stripping_token(scope) -> tuple[int, list, bytes]:
    """Swap the ?token= link for a cookie and a clean URL.

    Done as a redirect so the token stops being in the address bar, in the
    history, and in any link the user copies from it afterwards.
    """
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    token = query.pop("token")[0]
    rest = urlencode(query, doseq=True)
    target = scope.get("path", "/") + (f"?{rest}" if rest else "")
    cookie = (
        f"{COOKIE_NAME}={token}; Path=/; Max-Age={COOKIE_MAX_AGE}; "
        f"HttpOnly; SameSite=Lax"
    )
    return 302, [
        (b"location", target.encode("latin-1")),
        (b"set-cookie", cookie.encode("latin-1")),
    ], b""


class AccessGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # lifespan and websocket traffic must pass through untouched
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if self._is_mcp(path):
            # These two run before the loopback exemption below. Off has to
            # mean off for a client on this machine as well, and loopback is
            # exactly the address a rebinding attack aims at — exempting it
            # first would hand back both holes.
            if not config.load_settings().mcp.enabled:
                return await self._deny(
                    scope, send, 404, "未找到", "MCP 服务未开启。",
                )
            # An MCP client is not a browser and sends no Origin. One that
            # does is a web page reaching into this machine.
            if _header(scope, b"origin"):
                return await self._deny(
                    scope, send, 403, "拒绝访问",
                    "MCP 服务不接受来自网页的请求。",
                )
            # Starlette's Mount only matches "/mcp/…", but the URL handed to
            # clients is "/mcp" — without this the bare form falls through
            # to the SPA route and comes back 405.
            if path == MCP_PREFIX:
                scope = {**scope, "path": MCP_PREFIX + "/"}

        if _is_loopback(scope.get("client")):
            return await self.app(scope, receive, send)

        settings = config.load_settings()

        if not settings.server.lan_access:
            return await self._deny(
                scope, send, 403,
                "未开启局域网访问",
                "本程序默认只允许本机访问。请在这台机器上打开设置页，"
                "开启「局域网访问」并重启程序。",
            )
        if not settings.server.require_token:
            return await self.app(scope, receive, send)

        expected = settings.server.access_token
        presented, origin = _presented_token(scope)
        if expected and presented and secrets.compare_digest(presented, expected):
            if origin == "query" and scope.get("method") == "GET":
                status, headers, body = _redirect_stripping_token(scope)
                return await self._send(send, status, headers, body)
            return await self.app(scope, receive, send)

        return await self._deny(
            scope, send, 401,
            "需要访问令牌",
            "请在运行本程序的机器上打开设置页，复制「局域网访问」里带令牌的链接，"
            "用它打开一次即可（之后本设备无需再带令牌）。",
        )

    @staticmethod
    def _is_mcp(path: str) -> bool:
        return path == MCP_PREFIX or path.startswith(MCP_PREFIX + "/")

    async def _deny(self, scope, send, status: int, title: str, body: str):
        path = scope.get("path", "")
        if path.startswith("/api") or self._is_mcp(path):
            payload = json.dumps(
                {"detail": f"{title}：{body}"}, ensure_ascii=False
            ).encode()
            content_type = b"application/json; charset=utf-8"
        else:
            payload = _DENY_HTML.format(title=title, body=body).encode()
            content_type = b"text/html; charset=utf-8"
        await self._send(send, status, [(b"content-type", content_type)], payload)

    @staticmethod
    async def _send(send, status: int, headers: list, body: bytes):
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": headers + [(b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
