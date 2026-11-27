import time

import cv2
import pytest

from counter_drone.config import Config
from counter_drone.synthetic import SyntheticScene
from dashboard.app import Store, create_app
from dashboard.detector_service import DetectorService

PASSWORD = "correct-horse-battery"


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 3 s synthetic RGB clip written as an mp4."""
    path = tmp_path_factory.mktemp("clip") / "scene.mp4"
    scene = SyntheticScene(seed=0, duration=3.0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), scene.fps, scene.rgb_size)
    for _t, rgb, _thermal, _gts in scene:
        vw.write(rgb)
    vw.release()
    return str(path)


def wait_for(cond, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def service(clip):
    svc = DetectorService(sources={"clip": clip}, make_config=Config, loop=False)
    yield svc
    svc.stop()


@pytest.fixture
def env(tmp_path, service):
    store = Store(tmp_path / "users.db")
    store.add_user("admin1", PASSWORD, "admin")
    store.add_user("viewer1", PASSWORD, "viewer")
    app = create_app(db_path=tmp_path / "users.db", alerts_path=tmp_path / "none.jsonl",
                     testing=True, detector=service)
    return app, service


def token(client):
    with client.session_transaction() as s:
        s.setdefault("csrf", "test-csrf-token")
        return s["csrf"]


def login(client, username):
    client.post("/login", data={"username": username, "password": PASSWORD, "csrf_token": token(client)})


def post_json(client, url, body=None, csrf=True):
    headers = {"X-CSRF-Token": token(client)} if csrf else {}
    return client.post(url, json=body or {}, headers=headers)


def test_unknown_source_and_double_start(service):
    with pytest.raises(KeyError):
        service.start("../../etc/passwd")
    service.start("clip")
    with pytest.raises(RuntimeError):
        service.start("clip")


def test_missing_source_file_reports_error(tmp_path):
    svc = DetectorService(sources={"bad": str(tmp_path / "nope.mp4")}, make_config=Config)
    svc.start("bad")
    assert wait_for(lambda: not svc.status()["running"], 20)
    assert "cannot open source" in svc.status()["error"]


def test_only_admin_with_csrf_can_start(env):
    app, _ = env
    viewer = app.test_client()
    login(viewer, "viewer1")
    assert post_json(viewer, "/api/detector/start", {"source": "clip"}).status_code == 403
    admin = app.test_client()
    login(admin, "admin1")
    assert post_json(admin, "/api/detector/start", {"source": "clip"}, csrf=False).status_code == 400
    r = post_json(admin, "/api/detector/start", {"source": "/etc/passwd"})
    assert r.status_code == 400 and r.get_json()["error"] == "unknown source"


def test_run_streams_video_serves_alerts_and_stops(env):
    app, service = env
    admin = app.test_client()
    login(admin, "admin1")
    assert post_json(admin, "/api/detector/start", {"source": "clip"}).status_code == 200
    assert post_json(admin, "/api/detector/start", {"source": "clip"}).status_code == 409
    assert wait_for(lambda: admin.get("/api/detector/status").get_json()["frames"] >= 3)

    status = admin.get("/api/detector/status").get_json()
    assert status["running"] and status["source"] == "clip" and status["error"] is None

    resp = admin.get("/video_feed", buffered=False)
    assert resp.status_code == 200
    assert resp.mimetype == "multipart/x-mixed-replace"
    chunk = next(iter(resp.response))
    assert chunk.startswith(b"--frame") and b"\xff\xd8" in chunk      # JPEG start-of-image marker
    resp.close()

    assert isinstance(admin.get("/api/alerts").get_json(), list)
    assert all(e["t"] >= 0 for e in admin.get("/api/alerts?limit=500").get_json())

    stopped = post_json(admin, "/api/detector/stop").get_json()
    assert stopped["running"] is False


@pytest.fixture
def upload_env(tmp_path, service):
    app = create_app(db_path=tmp_path / "u.db", alerts_path=tmp_path / "a.jsonl", testing=True,
                     detector=service, open_access=True, upload_dir=tmp_path / "uploads")
    client = app.test_client()
    client.post("/login", data={"username": "tester", "csrf_token": token(client)})
    return client, service, tmp_path / "uploads"


def upload(client, name, data, csrf=True):
    import io
    headers = {"X-CSRF-Token": token(client)} if csrf else {}
    return client.post("/api/detector/upload", data={"video": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", headers=headers)


def test_upload_registers_a_source_and_it_can_run(upload_env, clip):
    client, service, up_dir = upload_env
    data = open(clip, "rb").read()
    r = upload(client, "My Drone Flight.mp4", data)
    assert r.status_code == 200
    name = r.get_json()["source"]
    assert name == "Upload: My_Drone_Flight"
    saved = list(up_dir.iterdir())
    assert len(saved) == 1 and "drone" not in saved[0].name.lower()      # stored under a random name
    assert post_json(client, "/api/detector/start", {"source": name}).status_code == 200
    assert wait_for(lambda: client.get("/api/detector/status").get_json()["frames"] >= 3)
    # cannot upload while running
    assert upload(client, "again.mp4", data).status_code == 409


def test_upload_rejects_bad_files(upload_env, clip):
    client, service, up_dir = upload_env
    assert upload(client, "notes.txt", b"hello").status_code == 400
    r = upload(client, "fake.mp4", b"this is not a video")
    assert r.status_code == 400 and "could not read" in r.get_json()["error"]
    assert not list(up_dir.iterdir())                                      # the bad file was removed
    assert upload(client, "x.mp4", open(clip, "rb").read(), csrf=False).status_code == 400   # CSRF


def test_upload_requires_admin(env, clip):
    app, _ = env
    viewer = app.test_client()
    login(viewer, "viewer1")
    assert upload(viewer, "a.mp4", open(clip, "rb").read()).status_code == 403


def test_local_folder_videos_are_listed_as_sources(tmp_path, clip, service):
    import shutil
    vids = tmp_path / "videos"
    vids.mkdir()
    app = create_app(db_path=tmp_path / "u.db", testing=True, detector=service, open_access=True,
                     upload_dir=tmp_path / "up", video_dir=vids)
    client = app.test_client()
    client.post("/login", data={"username": "t", "csrf_token": token(client)})
    assert "Local: flight" not in client.get("/api/detector/status").get_json()["sources"]
    shutil.copy(clip, vids / "flight.mp4")
    assert "Local: flight" in client.get("/api/detector/status").get_json()["sources"]
    assert post_json(client, "/api/detector/start", {"source": "Local: flight"}).status_code == 200
    assert wait_for(lambda: client.get("/api/detector/status").get_json()["frames"] >= 3)
    post_json(client, "/api/detector/stop")
    (vids / "flight.mp4").unlink()
    assert "Local: flight" not in client.get("/api/detector/status").get_json()["sources"]


def test_sensitivity_is_validated_and_reported(upload_env):
    client, service, _ = upload_env
    assert post_json(client, "/api/detector/start", {"source": "clip", "sensitivity": "extreme"}).status_code == 400
    assert post_json(client, "/api/detector/start", {"source": "clip", "sensitivity": "high"}).status_code == 200
    assert wait_for(lambda: client.get("/api/detector/status").get_json()["frames"] >= 3)
    st = client.get("/api/detector/status").get_json()
    assert st["sensitivity"] == "high" and "detections" in st


def test_no_demo_data_without_a_run_or_alerts_file(upload_env):
    client, _, _ = upload_env
    assert client.get("/api/alerts").get_json() == []


def test_dashboard_js_parses():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = Path(__file__).resolve().parent.parent / "dashboard" / "static" / "dashboard.js"
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_short_clip_loops_without_error_and_time_keeps_increasing(clip):
    svc = DetectorService(sources={"clip": clip}, make_config=Config, loop=True)
    try:
        svc.start("clip")
        n_clip = 3 * 15                                   # 3 s at 15 fps
        assert wait_for(lambda: svc.status()["frames"] > n_clip + 10 or svc.status()["error"], 60)
        st = svc.status()
        assert st["error"] is None and st["running"]
        times = [a["t"] for a in svc.alerts(500)][::-1]   # oldest first
        assert times == sorted(times)                     # the looped clip does not rewind the clock
    finally:
        svc.stop()
