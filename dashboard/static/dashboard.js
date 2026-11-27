(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var CX = 300, CY = 260, R = 220;
  var POLL_MS = 2000, FEED_ROWS = 14;

  var $ = function (id) { return document.getElementById(id); };
  var el = {
    feed: $("feed"), tracks: $("tracks"), shown: $("shown"), live: $("live"), liveText: $("live-text"),
    sActive: $("s-active"), sActiveN: $("s-active-note"), sHigh: $("s-high"), sHighN: $("s-high-note"),
    sConf: $("s-conf"), sLast: $("s-last"), sLastN: $("s-last-note"), empty: $("sky-empty")
  };

  function fmtTime(t) {
    var m = Math.floor(t / 60), s = t - m * 60;
    return String(m).padStart(2, "0") + ":" + (s < 10 ? "0" : "") + s.toFixed(2);
  }
  function node(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  var EL_MAX = 30;                            // outer ring = 0 deg, centre = 30 deg (the camera sees low elevations)
  function polar(az, elv) {
    var r = R * (1 - Math.min(Math.max(elv, 0), EL_MAX) / EL_MAX), a = az * Math.PI / 180;
    return { x: CX + r * Math.sin(a), y: CY - r * Math.cos(a) };
  }
  function highlight(id, on) {
    document.querySelectorAll('[data-track="' + id + '"]').forEach(function (n) { n.classList.toggle("hl", on); });
  }

  function latestPerTrack(alerts) {          // alerts arrive newest first
    var seen = {}, out = [];
    alerts.forEach(function (a) {
      if (!(a.track_id in seen)) { seen[a.track_id] = true; out.push(a); }
    });
    return out;
  }

  function renderTracks(latest) {
    el.tracks.textContent = "";
    var active = latest.filter(function (a) { return a.level === "LOW" || a.level === "MEDIUM" || a.level === "HIGH"; });
    active.forEach(function (a) {
      var p = polar(a.azimuth_deg || 0, a.elevation_deg || 0);
      var g = node("g", { "class": "track L-" + a.level, "data-track": a.track_id, transform: "translate(" + p.x.toFixed(1) + " " + p.y.toFixed(1) + ")" });
      g.appendChild(node("title", {}, "Track " + a.track_id + ": " + a.level + ", az " + a.azimuth_deg + ", el " + a.elevation_deg));
      g.appendChild(node("circle", { "class": "halo", r: 15 }));
      g.appendChild(node("circle", { "class": "core", r: 6 }));
      g.appendChild(node("text", { x: 20, y: 4 }, "T" + a.track_id + "  " + a.level));
      g.addEventListener("mouseenter", function () { highlight(a.track_id, true); });
      g.addEventListener("mouseleave", function () { highlight(a.track_id, false); });
      el.tracks.appendChild(g);
    });
    el.empty.hidden = active.length > 0;
    return active;
  }

  function renderFeed(alerts) {
    el.feed.textContent = "";
    alerts.slice(0, FEED_ROWS).forEach(function (a) {
      var li = document.createElement("li");
      li.className = "row L-" + a.level;
      li.setAttribute("data-track", a.track_id);
      var bar = document.createElement("span"); bar.className = "bar";
      var chip = document.createElement("span"); chip.className = "chip";
      chip.textContent = a.level === "MEDIUM" ? "MED" : a.level;
      var meta = document.createElement("div"); meta.className = "meta";
      var id = document.createElement("b"); id.textContent = "T" + a.track_id;
      var t = document.createElement("small"); t.textContent = fmtTime(Number(a.t) || 0);
      meta.appendChild(id); meta.appendChild(t);
      var p = document.createElement("p"); p.textContent = a.message || "";
      [bar, chip, meta, p].forEach(function (c) { li.appendChild(c); });
      li.addEventListener("mouseenter", function () { highlight(a.track_id, true); });
      li.addEventListener("mouseleave", function () { highlight(a.track_id, false); });
      el.feed.appendChild(li);
    });
    el.shown.textContent = Math.min(alerts.length, FEED_ROWS) + " shown";
    if (!alerts.length) {
      var li = document.createElement("li"); li.className = "empty"; li.textContent = "No alerts yet."; el.feed.appendChild(li);
    }
  }

  function renderStats(alerts, latest, active) {
    el.sActive.textContent = String(active.length);
    el.sActiveN.textContent = latest.length + " tracks seen";
    var highs = alerts.filter(function (a) { return a.level === "HIGH"; });
    el.sHigh.textContent = String(highs.length);
    el.sHighN.textContent = highs.length ? "latest at " + fmtTime(Number(highs[0].t) || 0) : "none in window";
    if (active.length) {
      var mean = active.reduce(function (s, a) { return s + (Number(a.drone_prob) || 0); }, 0) / active.length;
      el.sConf.textContent = mean.toFixed(2);
    } else { el.sConf.textContent = "—"; }
    if (alerts.length) {
      el.sLast.textContent = fmtTime(Number(alerts[0].t) || 0);
      el.sLastN.textContent = alerts[0].level + " on T" + alerts[0].track_id;
    } else { el.sLast.textContent = "—"; el.sLastN.textContent = "waiting for data"; }
  }

  function setLive(ok, text) {
    el.live.className = "pill " + (ok ? "live" : "stale");
    el.liveText.textContent = text;
  }

  function poll() {
    fetch("/api/alerts?limit=200", { headers: { "Accept": "application/json" }, credentials: "same-origin" })
      .then(function (r) {
        if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (alerts) {
        var latest = latestPerTrack(alerts);
        var active = renderTracks(latest);
        renderFeed(alerts);
        renderStats(alerts, latest, active);
        setLive(true, "LIVE  " + new Date().toLocaleTimeString([], { hour12: false }));
      })
      .catch(function (e) { if (e.message !== "signed out") setLive(false, "OFFLINE  " + e.message); });
  }

  poll();
  setInterval(poll, POLL_MS);
})();
