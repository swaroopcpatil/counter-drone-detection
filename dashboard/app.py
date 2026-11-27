"""Research monitoring dashboard: login, roles and a live alert feed.

Research use only. It reads the JSON alert lines that `python -m counter_drone run`
writes (events.jsonl) and shows them to signed-in users. Two roles:

  admin   - everything a viewer can do, plus creating user accounts
  viewer  - read the alert feed

Create the first account before starting the server:

    python dashboard/app.py create-user --role admin
    python dashboard/app.py serve --host 127.0.0.1 --port 5000
"""
import argparse
import getpass
import json
import os
import re
import secrets
import sqlite3
from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "dashboard.db"
DEFAULT_ALERTS = HERE.parent / "runs" / "demo" / "events.jsonl"
ROLES = ("admin", "viewer")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 12
MAX_ALERTS = 200


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
    if not Path(path).exists():
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


def create_app(db_path=None, alerts_path=None, testing=False):
    app = Flask(__name__)
    db_path = Path(db_path or os.environ.get("DASHBOARD_DB", DEFAULT_DB))
    alerts_path = Path(alerts_path or os.environ.get("DASHBOARD_ALERTS", DEFAULT_ALERTS))
    app.config.update(
        SECRET_KEY=_secret_key(db_path.with_suffix(".key")) if not testing else "test-key",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        TESTING=testing,
    )
    store = Store(db_path)
    app.extensions["store"] = store
    app.config["ALERTS_PATH"] = alerts_path

    def login_required(role=None):
        def deco(view):
            @wraps(view)
            def wrapped(*args, **kwargs):
                user = session.get("user")
                if not user:
                    if request.path.startswith("/api/"):
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
        return render_template("index.html", user=session["user"])

    @app.get("/api/alerts")
    @login_required()
    def api_alerts():
        limit = min(max(request.args.get("limit", 50, type=int), 1), MAX_ALERTS)
        return jsonify(read_alerts(alerts_path, limit))

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
        app = create_app(db_path=args.db, alerts_path=args.alerts)
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    _cli()
