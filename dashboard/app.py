"""Research monitoring dashboard: login, roles, a live detector view and an alert feed.

Research use only. An admin starts the counter_drone detector on a named, server-side source;
every signed-in user then sees the annotated video, the sky dome and the alerts update live.
When the detector has not been started the feed falls back to the JSON alert lines in an
events.jsonl file (the demo run). Two roles:

  admin   - everything a viewer can do, plus creating accounts and starting/stopping the detector
  viewer  - watch the video and read the alert feed

Run from the repository root, with a single server process (one detector per process):

    python -m dashboard.app create-user --role admin
    python -m dashboard.app serve --host 127.0.0.1 --port 5000
"""
import argparse
import getpass
import hmac
import json
import os
import re
import secrets
import sqlite3
import sys
import uuid
from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:          # lets `python dashboard/app.py` import counter_drone too
    sys.path.insert(0, str(HERE.parent))

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,  # noqa: E402
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from dashboard.detector_service import DetectorService  # noqa: E402

DEFAULT_DB = HERE / "dashboard.db"
DEFAULT_VIDEOS = HERE / "videos"
ROLES = ("admin", "viewer")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 12
MAX_ALERTS = 200
UPLOAD_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
KEEP_UPLOADS = 5
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _secret_key(path: Path) -> str:
    """Use DASHBOARD_SECRET if set, otherwise a random key saved next to the database."""
    env = os.environ.get("DASHBOARD_SECRET")
    if env:
        return env
    if path.exists():
        return path.read_text().strip()
    key = secrets.token_hex(32)
    path.write_text(key)
    return key


class Store:
    """Users in a small SQLite file. Passwords are stored only as salted hashes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'viewer')))""")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_user(self, username: str, password: str, role: str) -> None:
        if not USERNAME_RE.match(username):
            raise ValueError("username must be 3-32 letters, digits, '.', '_' or '-'")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
        with self._conn() as c:
            try:
                c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                          (username, generate_password_hash(password), role))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"user {username!r} already exists") from exc

    def check(self, username: str, password: str):
        with self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            return {"username": row["username"], "role": row["role"]}
        return None

    def list_users(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT username, role FROM users ORDER BY username").fetchall()]


def read_alerts(path: Path, limit: int = MAX_ALERTS):
    """Return the newest alerts first. Missing file or bad lines are skipped, not fatal."""
    if path is None or not Path(path).exists():
        return []
    tail = deque(maxlen=limit)  # keeps only the newest `limit` valid alerts
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                tail.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(tail))


def create_app(db_path=None, alerts_path=None, testing=False, detector=None, open_access=False, upload_dir=None,
               video_dir=None):
    app = Flask(__name__)
    db_path = Path(db_path or os.environ.get("DASHBOARD_DB", DEFAULT_DB))
    alerts_env = alerts_path or os.environ.get("DASHBOARD_ALERTS")   # optional fallback feed; none by default
    alerts_path = Path(alerts_env) if alerts_env else None
    app.config.update(
        SECRET_KEY=_secret_key(db_path.with_suffix(".key")) if not testing else "test-key",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        TESTING=testing,
    )
    store = Store(db_path)
    detector = detector or DetectorService()
    app.extensions["store"] = store
    app.extensions["detector"] = detector
    app.config["OPEN_ACCESS"] = open_access
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    upload_dir = Path(upload_dir or os.environ.get("DASHBOARD_UPLOADS", HERE / "uploads"))
    video_dir = Path(video_dir or os.environ.get("DASHBOARD_VIDEOS", DEFAULT_VIDEOS))
    video_dir.mkdir(parents=True, exist_ok=True)       # drop video files here to list them as sources
    detector.scan_local(video_dir)

    @app.context_processor
    def inject_mode():
        return {"open_access": open_access, "video_dir": str(video_dir)}
    app.config["ALERTS_PATH"] = alerts_path

    def csrf_token():
        tok = session.get("csrf")
        if not tok:
            tok = session["csrf"] = secrets.token_urlsafe(32)
        return tok

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def csrf_protect():
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
            expected = session.get("csrf", "")
            if not expected or not hmac.compare_digest(sent, expected):
                abort(400)

    def login_required(role=None):
        def deco(view):
            @wraps(view)
            def wrapped(*args, **kwargs):
                user = session.get("user")
                if not user:
                    if request.path.startswith("/api/") or request.path == "/video_feed":
                        abort(401)
                    return redirect(url_for("login", next=request.path))
                if role and user["role"] != role:
                    abort(403)
                return view(*args, **kwargs)
            return wrapped
        return deco

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if open_access:           # demo mode: any name, no password (loopback only, see serve)
                name = re.sub(r"[^A-Za-z0-9_.-]", "", username)[:32] or "guest"
                user = {"username": name, "role": "admin"}
            else:
                user = store.check(username, password)
            if user:
                session.clear()
                session["user"] = user
                return redirect(request.args.get("next") or url_for("index"))
            error = "Invalid username or password."
        return render_template("login.html", error=error)

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required()
    def index():
        detector.scan_local(video_dir)
        return render_template("index.html", user=session["user"], sources=sorted(detector.sources))

    @app.get("/api/alerts")
    @login_required()
    def api_alerts():
        limit = min(max(request.args.get("limit", 50, type=int), 1), MAX_ALERTS)
        if detector.has_alerts():                     # live run (or its last results)
            return jsonify(detector.alerts(limit))
        return jsonify(read_alerts(alerts_path, limit))

    @app.get("/video_feed")
    @login_required()
    def video_feed():
        if detector.latest_jpeg() is None:
            abort(404)
        return Response(detector.stream(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/detector/status")
    @login_required()
    def detector_status():
        detector.scan_local(video_dir)
        return jsonify(detector.status())

    @app.post("/api/detector/start")
    @login_required(role="admin")
    def detector_start():
        body = request.get_json(silent=True) or {}
        name = str(body.get("source", ""))
        try:
            detector.scan_local(video_dir)
            detector.start(name, str(body.get("sensitivity", "standard")))
        except KeyError:
            return jsonify(error="unknown source"), 400
        except ValueError:
            return jsonify(error="unknown sensitivity"), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(detector.status())

    @app.post("/api/detector/stop")
    @login_required(role="admin")
    def detector_stop():
        detector.stop()
        return jsonify(detector.status())

    @app.post("/api/detector/upload")
    @login_required(role="admin")
    def detector_upload():
        """Accept a video, check that OpenCV can read it, and register it as a named source."""
        f = request.files.get("video")
        if f is None or not f.filename:
            return jsonify(error="choose a video file"), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in UPLOAD_EXTS:
            return jsonify(error="unsupported file type (use " + ", ".join(sorted(UPLOAD_EXTS)) + ")"), 400
        if detector.status()["running"]:
            return jsonify(error="stop the detector before uploading"), 409
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / (uuid.uuid4().hex + ext)        # never use the client's file name on disk
        f.save(dest)
        import cv2
        cap = cv2.VideoCapture(str(dest))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            dest.unlink(missing_ok=True)
            return jsonify(error="could not read frames from that file"), 400
        base = "Upload: " + (secure_filename(Path(f.filename).stem)[:40] or "video")
        name, n = base, 2
        while name in detector.sources:
            name, n = f"{base} ({n})", n + 1
        detector.add_source(name, str(dest))
        for old in sorted((p for p in upload_dir.iterdir() if p.is_file()), key=lambda q: q.stat().st_mtime)[:-KEEP_UPLOADS]:
            detector.remove_source_path(str(old))
            old.unlink(missing_ok=True)
        return jsonify(source=name, status=detector.status())

    @app.route("/admin/users", methods=["GET", "POST"])
    @login_required(role="admin")
    def admin_users():
        if request.method == "POST":
            try:
                store.add_user(request.form.get("username", "").strip(),
                               request.form.get("password", ""),
                               request.form.get("role", "viewer"))
                flash("User created.", "ok")
            except ValueError as exc:
                flash(str(exc), "error")
            return redirect(url_for("admin_users"))
        return render_template("admin_users.html", user=session["user"],
                               users=store.list_users(), roles=ROLES)

    return app


def _cli(argv=None):
    p = argparse.ArgumentParser(description="Research monitoring dashboard")
    sub = p.add_subparsers(dest="cmd", required=True)
    cu = sub.add_parser("create-user", help="create an account")
    cu.add_argument("--role", choices=ROLES, default="viewer")
    cu.add_argument("--db", default=None)
    sv = sub.add_parser("serve", help="run the web server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)
    sv.add_argument("--db", default=None)
    sv.add_argument("--alerts", default=None)
    sv.add_argument("--open-access", action="store_true",
                    help="demo mode: any username, no password (only allowed on a loopback host)")
    args = p.parse_args(argv)

    if args.cmd == "create-user":
        store = Store(Path(args.db or os.environ.get("DASHBOARD_DB", DEFAULT_DB)))
        username = input("username: ").strip()
        password = getpass.getpass("password: ")
        if getpass.getpass("repeat password: ") != password:
            raise SystemExit("passwords do not match")
        try:
            store.add_user(username, password, args.role)
        except ValueError as exc:
            raise SystemExit(str(exc))
        print(f"created {args.role} account {username!r}")
    else:
        if args.open_access and args.host not in LOOPBACK:
            raise SystemExit("--open-access disables sign-in; it is only allowed with --host 127.0.0.1")
        app = create_app(db_path=args.db, alerts_path=args.alerts, open_access=args.open_access)
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    _cli()
