"""Who gets in once the server stops being loopback-only.

The property every test here defends: nothing reachable from another
machine without the user having deliberately opened it. The endpoints
behind this guard hand out the LLM API key, list any directory on the
disk, and start jobs on arbitrary paths, so "closed unless configured
otherwise" has to hold by default and not by luck.
"""

from tests.conftest import local_client, remote_client

PATH = "/api/version"


def app():
    from app.main import app as fastapi_app

    return fastapi_app


# ----------------------------------------------------------- shipped state


def test_by_default_another_machine_is_refused(settings_file):
    settings_file()  # untouched defaults
    r = remote_client(app()).get(PATH)
    assert r.status_code == 403
    assert "局域网访问" in r.json()["detail"]


def test_by_default_this_machine_still_works(settings_file):
    settings_file()
    assert local_client(app()).get(PATH).status_code == 200


def test_loopback_is_exempt_even_from_the_token(settings_file):
    """The person at the machine can never be locked out of their own
    server — that is also how they read the token off the settings page."""
    settings_file(server__lan_access=True, server__access_token="s3cret")
    assert local_client(app()).get(PATH).status_code == 200


# ------------------------------------------------------------ token checks


def test_lan_access_without_a_token_is_refused(settings_file):
    settings_file(server__lan_access=True, server__access_token="s3cret")
    r = remote_client(app()).get(PATH)
    assert r.status_code == 401
    assert "令牌" in r.json()["detail"]


def test_a_wrong_token_is_refused(settings_file):
    settings_file(server__lan_access=True, server__access_token="s3cret")
    r = remote_client(app()).get(PATH, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_every_way_of_presenting_the_token_is_accepted(settings_file):
    """Four carriers because the callers differ: MCP clients send a bearer
    header, the browser has only cookies for SSE and download links."""
    settings_file(server__lan_access=True, server__access_token="s3cret")
    c = remote_client(app())
    assert c.get(PATH, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get(PATH, headers={"X-MT-Token": "s3cret"}).status_code == 200
    assert c.get(PATH, cookies={"mt_token": "s3cret"}).status_code == 200


def test_a_token_link_becomes_a_cookie_and_leaves_the_url(settings_file):
    """The one-time link is how a phone or laptop gets in. Redirecting
    rather than serving in place keeps the token out of the address bar,
    the history, and any URL copied out of it afterwards."""
    settings_file(server__lan_access=True, server__access_token="s3cret")
    c = remote_client(app())
    r = c.get(f"{PATH}?token=s3cret&keep=1", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"{PATH}?keep=1"
    cookie = r.headers["set-cookie"]
    assert "mt_token=s3cret" in cookie and "HttpOnly" in cookie


def test_the_token_can_be_switched_off(settings_file):
    settings_file(server__lan_access=True, server__require_token=False)
    assert remote_client(app()).get(PATH).status_code == 200


# -------------------------------------------------------------------- mcp


def test_mcp_is_absent_while_the_switch_is_off(settings_file):
    settings_file(server__lan_access=True, server__require_token=False)
    assert remote_client(app()).post("/mcp").status_code == 404


def test_mcp_still_needs_the_token(settings_file):
    settings_file(
        server__lan_access=True, server__access_token="s3cret", mcp__enabled=True
    )
    assert remote_client(app()).post("/mcp").status_code == 401


# ------------------------------------------------------------ api key mask


def test_the_api_key_is_masked_for_a_remote_reader(settings_file):
    from app.api.routes import MASKED

    settings_file(
        server__lan_access=True, server__require_token=False, llm__api_key="sk-real"
    )
    body = remote_client(app()).get("/api/settings").json()
    assert body["llm"]["api_key"] == MASKED
    assert local_client(app()).get("/api/settings").json()["llm"]["api_key"] == "sk-real"


def test_saving_from_a_remote_page_does_not_wipe_the_key(settings_file):
    """Without this the mask would be a footgun: one save from a LAN
    browser and the stored key is the literal asterisks."""
    from app.api.routes import MASKED
    from app.core import config

    settings_file(
        server__lan_access=True, server__require_token=False, llm__api_key="sk-real"
    )
    c = remote_client(app())
    body = c.get("/api/settings").json()
    assert body["llm"]["api_key"] == MASKED
    body["llm"]["model"] = "changed-model"
    assert c.put("/api/settings", json=body).status_code == 200

    saved = config.load_settings()
    assert saved.llm.api_key == "sk-real"
    assert saved.llm.model == "changed-model"


def test_regenerating_the_token_is_loopback_only(settings_file):
    from app.core import config

    settings_file(
        server__lan_access=True, server__require_token=False, server__access_token="old"
    )
    assert remote_client(app()).post("/api/server/token/regenerate").status_code == 403

    r = local_client(app()).post("/api/server/token/regenerate")
    assert r.status_code == 200
    assert r.json()["token"] not in ("", "old")
    assert config.load_settings().server.access_token == r.json()["token"]


def test_enabling_lan_access_always_leaves_a_usable_token(settings_file):
    """"Open to the LAN, token required, token empty" must be unreachable."""
    from app.core import config

    settings_file()
    c = local_client(app())
    body = c.get("/api/settings").json()
    body["server"]["lan_access"] = True
    c.put("/api/settings", json=body)
    assert config.load_settings().server.access_token != ""


def test_server_info_hides_the_token_from_remote_callers(settings_file):
    settings_file(
        server__lan_access=True, server__require_token=False, server__access_token="s3cret"
    )
    assert remote_client(app()).get("/api/server/info").json()["token"] == ""
    assert local_client(app()).get("/api/server/info").json()["token"] == "s3cret"
