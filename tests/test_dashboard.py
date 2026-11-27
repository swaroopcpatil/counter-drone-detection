import json

import pytest

from dashboard.app import Store, create_app, read_alerts
from dashboard.detector_service import DetectorService

PASSWORD = "correct-horse-battery"


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "users.db"
    alerts = tmp_path / "events.jsonl"
    alerts.write_text("\n".join(json.dumps({
        "t": float(i), "track_id": 1, "level": "HIGH", "message": f"alert {i}",
    }) for i in range(5)) + "\nnot json\n")
    store = Store(db)
    store.add_user("admin1", PASSWORD, "admin")
    store.add_user("viewer1", PASSWORD, "viewer")
    app = create_app(db_path=db, alerts_path=alerts, testing=True, detector=DetectorService(sources={}))
    return app, app.test_client()


def token(client):
    """Put a known CSRF token in the session and return it."""
    with client.session_transaction() as s:
        s.setdefault("csrf", "test-csrf-token")
        return s["csrf"]


def login(client, username, password=PASSWORD):
    return client.post("/login", data={"username": username, "password": password,
                                       "csrf_token": token(client)})


def test_anonymous_is_redirected_to_login(env):
    _, client = env
    r = client.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_api_requires_login(env):
    _, client = env
    assert client.get("/api/alerts").status_code == 401
    assert client.get("/api/detector/status").status_code == 401
    assert client.get("/video_feed").status_code == 401


def test_login_without_csrf_token_is_rejected(env):
    _, client = env
    r = client.post("/login", data={"username": "viewer1", "password": PASSWORD})
    assert r.status_code == 400
    assert client.get("/api/alerts").status_code == 401


def test_bad_password_is_rejected(env):
    _, client = env
    r = login(client, "viewer1", "wrong-password-value")
    assert r.status_code == 200 and b"Invalid username or password" in r.data
    assert client.get("/api/alerts").status_code == 401


def test_viewer_reads_feed_newest_first_and_skips_bad_lines(env):
    _, client = env
    assert login(client, "viewer1").status_code == 302
    alerts = client.get("/api/alerts?limit=3").get_json()
    assert [a["message"] for a in alerts] == ["alert 4", "alert 3", "alert 2"]


def test_viewer_cannot_manage_users(env):
    _, client = env
    login(client, "viewer1")
    assert client.get("/admin/users").status_code == 403
    assert client.post("/admin/users", data={"username": "x1y", "password": "x" * 12, "role": "admin",
                                             "csrf_token": token(client)}).status_code == 403


def test_admin_creates_user_who_can_sign_in(env):
    app, client = env
    login(client, "admin1")
    r = client.post("/admin/users", data={"username": "newbie", "password": PASSWORD, "role": "viewer",
                                          "csrf_token": token(client)}, follow_redirects=True)
    assert b"User created." in r.data
    other = app.test_client()
    assert login(other, "newbie").status_code == 302


def test_post_without_csrf_token_is_rejected(env):
    _, client = env
    login(client, "admin1")
    r = client.post("/admin/users", data={"username": "newbie", "password": PASSWORD, "role": "viewer"})
    assert r.status_code == 400


def test_weak_password_and_duplicate_user_are_refused(env):
    _, client = env
    login(client, "admin1")
    r = client.post("/admin/users", data={"username": "short", "password": "tiny", "role": "viewer",
                                          "csrf_token": token(client)}, follow_redirects=True)
    assert b"at least 12 characters" in r.data
    r = client.post("/admin/users", data={"username": "viewer1", "password": PASSWORD, "role": "viewer",
                                          "csrf_token": token(client)}, follow_redirects=True)
    assert b"already exists" in r.data


def test_logout_ends_session(env):
    _, client = env
    login(client, "viewer1")
    client.post("/logout", data={"csrf_token": token(client)})
    assert client.get("/api/alerts").status_code == 401


def test_passwords_are_not_stored_in_plain_text(tmp_path):
    db = tmp_path / "u.db"
    Store(db).add_user("someone", PASSWORD, "viewer")
    assert PASSWORD.encode() not in db.read_bytes()


def test_missing_alert_file_gives_empty_feed(tmp_path):
    assert read_alerts(tmp_path / "absent.jsonl") == []


def test_open_access_login_takes_any_name_without_password(tmp_path):
    app = create_app(db_path=tmp_path / "u.db", alerts_path=tmp_path / "a.jsonl", testing=True,
                     detector=DetectorService(sources={}), open_access=True)
    client = app.test_client()
    page = client.get("/login").data
    assert b'name="password"' not in page and b"Demo mode" in page
    r = client.post("/login", data={"username": "Anyone <b>", "csrf_token": token(client)})
    assert r.status_code == 302
    assert client.get("/api/alerts").status_code == 200
    # a name with no usable characters falls back to "guest"
    other = app.test_client()
    assert other.post("/login", data={"username": "!!!", "csrf_token": token(other)}).status_code == 302
    assert b"guest" in other.get("/").data


def test_normal_mode_still_needs_a_password(env):
    _, client = env
    r = client.post("/login", data={"username": "viewer1", "csrf_token": token(client)})
    assert r.status_code == 200 and b"Invalid username or password" in r.data


def test_serve_refuses_open_access_on_a_public_host():
    from dashboard.app import _cli
    with pytest.raises(SystemExit):
        _cli(["serve", "--host", "0.0.0.0", "--open-access"])


def test_repeated_failed_logins_are_blocked_then_allowed_after_the_window(tmp_path):
    from dashboard.app import LoginLimiter
    now = [0.0]
    lim = LoginLimiter(max_failures=3, window=60, clock=lambda: now[0])
    for _ in range(3):
        assert not lim.blocked("u")
        lim.fail("u")
    assert lim.blocked("u") and not lim.blocked("someone-else")
    now[0] = 61
    assert not lim.blocked("u")                    # failures expire
    lim.fail("u"); lim.clear("u")
    assert not lim.blocked("u")


def test_login_is_rate_limited_even_for_the_correct_password(env):
    _, client = env
    for _ in range(5):
        assert login(client, "viewer1", "wrong-password-value").status_code == 200
    r = login(client, "viewer1", PASSWORD)
    assert r.status_code == 429 and b"Too many failed attempts" in r.data
    assert client.get("/api/alerts").status_code == 401
    other = login(client, "admin1")                # a different account from the same address is also held back
    assert other.status_code in (302, 429)


def test_security_headers_and_secure_cookie_in_public_mode(tmp_path):
    store = Store(tmp_path / "u.db")
    store.add_user("boss", PASSWORD, "admin")
    pub = create_app(db_path=tmp_path / "u.db", testing=True, detector=DetectorService(sources={}), public=True)
    r = pub.test_client().get("/login")
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    assert "max-age" in r.headers["Strict-Transport-Security"]
    assert "Secure" in r.headers.get("Set-Cookie", "")
    local = create_app(db_path=tmp_path / "u2.db", testing=True, detector=DetectorService(sources={}))
    assert "Strict-Transport-Security" not in local.test_client().get("/login").headers


def test_public_mode_cannot_be_combined_with_open_access(tmp_path):
    with pytest.raises(ValueError):
        create_app(db_path=tmp_path / "u.db", testing=True, detector=DetectorService(sources={}),
                   public=True, open_access=True)


def test_open_access_demo_refuses_requests_that_came_through_a_proxy(tmp_path):
    app = create_app(db_path=tmp_path / "u.db", testing=True, detector=DetectorService(sources={}), open_access=True)
    client = app.test_client()
    assert client.get("/login").status_code == 200
    for header in ("X-Forwarded-For", "CF-Connecting-IP", "X-Real-IP", "Forwarded"):
        assert client.get("/login", headers={header: "203.0.113.9"}).status_code == 403, header


def test_serve_public_needs_an_admin_account(tmp_path):
    from dashboard.app import _cli
    with pytest.raises(SystemExit) as e:
        _cli(["serve", "--public", "--db", str(tmp_path / "none.db")])
    assert "no admin account" in str(e.value)
    with pytest.raises(SystemExit):
        _cli(["serve", "--public", "--open-access"])
