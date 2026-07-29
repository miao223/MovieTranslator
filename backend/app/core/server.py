"""Where the app listens, and the token that guards it from outside.

Binding is decided once, at process start, from the saved settings — so a
change to `lan_access` or `port` only takes effect on restart. RUNNING
records what was actually bound so the settings page can say "saved, takes
effect after restart" instead of guessing.
"""

from __future__ import annotations

import secrets
import socket
from typing import Optional

# filled in by main(); stays empty when started via `uvicorn app.main:app`,
# which means "launched by an external command, actual binding unknown"
RUNNING: dict = {}

LOOPBACK_HOST = "127.0.0.1"
ANY_HOST = "0.0.0.0"


def bind_host(settings) -> str:
    return ANY_HOST if settings.server.lan_access else LOOPBACK_HOST


def new_token() -> str:
    return secrets.token_urlsafe(24)


def ensure_token(settings) -> bool:
    """Fill in a token if LAN access needs one and none is set.

    Returns whether anything changed, so the caller can decide to persist.
    Called both when settings are saved and at startup, which makes
    "LAN access on, token required, token empty" an unreachable state.
    """
    srv = settings.server
    if srv.lan_access and srv.require_token and not srv.access_token.strip():
        srv.access_token = new_token()
        return True
    return False


def _primary_ip() -> Optional[str]:
    """The address this machine would use to reach the outside world.

    A UDP socket has no handshake, so connect() sends nothing — it just asks
    the routing table which local address applies. That answer is the one
    the user's other devices need, and unlike gethostbyname it does not
    depend on /etc/hosts being sensible.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def lan_ips() -> list[str]:
    """Addresses of this machine worth showing the user, best guess first."""
    found: list[str] = []
    primary = _primary_ip()
    if primary and not primary.startswith("127."):
        found.append(primary)
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        addrs = []
    for addr in addrs:
        if not addr.startswith("127.") and addr not in found:
            found.append(addr)
    return found
