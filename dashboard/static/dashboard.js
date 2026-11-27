(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var CX = 300, CY = 260, R = 220;
  var POLL_MS = 2000, FEED_ROWS = 14;

  var $ = function (id) { return document.getElementById(id); };
  var el = {
    feed: $("feed"), tracks: $("tracks"), shown: $("shown"), live: $("live"), liveText: $("live-text"),
    sActive: $("s-active"), sActiveN: $("s-active-note"), sHigh: $("s-high"), sHighN: $("s-high-note"),
    sConf: $("s-conf"), sLast: $("s-last"), sLastN: $("s-last-note"), empty: $("sky-empty"),
    video: $("video"), vEmpty: $("video-empty"), vErr: $("video-error"), vSub: $("video-sub"),
    source: $("source"), sens: $("sens"), toggle: $("toggle"), upload: $("upload"), uploadLabel: $("upload-label")
  };
  var csrf = (document.querySelector('meta[name="csrf-token"]') || {}).content || "";
  var detector = { running: false };

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
      })
      .catch(function (e) { if (e.message !== "signed out") setLive(false, "OFFLINE  " + e.message); });
  }

  function showVideo(on) {
    el.video.hidden = !on;
    el.vEmpty.hidden = on;
  }

  function applyStatus(st) {
    var wasRunning = detector.running;
    detector.running = !!st.running;
    if (st.error) {
      el.vErr.hidden = false; el.vErr.textContent = "Detector error: " + st.error;
      setLive(false, "ERROR");
    } else {
      el.vErr.hidden = true;
      if (st.running) setLive(true, "LIVE  " + st.fps + " fps  " + st.ms_per_frame + " ms");
      else setLive(false, "DETECTOR STOPPED");
    }
    if (st.running) {
      var seen = st.frames ? Math.round(100 * st.detections / st.frames) : 0;
      el.vSub.textContent = st.source + " / " + st.frames + " frames / drone seen in " + seen + "%";
    } else {
      el.vSub.textContent = "annotated detector output";
    }
    if (st.running) {
      // (re)attach the stream once the first frame exists; the route answers 404 until then
      if (el.video.hidden || !el.video.getAttribute("src") || !wasRunning) {
        el.video.src = "/video_feed?t=" + Date.now();
      }
      if (st.frames > 0) showVideo(true);       // do not rely on the <img> load event for a multipart stream
    } else {
      el.video.removeAttribute("src");
      showVideo(false);
    }
    if (el.toggle) {
      el.toggle.textContent = st.running ? "Stop" : "Start";
      el.toggle.disabled = false;
      if (el.source && st.source) el.source.value = st.source;
      syncSources(st.sources || [], st.running);
      el.source.disabled = !!st.running;
      if (el.sens) { el.sens.disabled = !!st.running; if (st.running) el.sens.value = st.sensitivity; }
      if (!st.running && !(st.sources || []).length) { el.toggle.disabled = true; }
      if (el.upload) { el.upload.disabled = !!st.running; el.uploadLabel.classList.toggle("disabled", !!st.running); }
    }
  }

  function syncSources(names, running) {
    if (!el.source) return;
    var have = Array.prototype.map.call(el.source.options, function (o) { return o.value; });
    if (have.join("|") === names.join("|")) return;
    var keep = el.source.value;
    el.source.textContent = "";
    names.forEach(function (n) {
      var o = document.createElement("option"); o.value = n; o.textContent = n; el.source.appendChild(o);
    });
    if (names.indexOf(keep) >= 0) el.source.value = keep;
  }

  function showError(msg) { el.vErr.hidden = false; el.vErr.textContent = msg; }

  function startDetector(name) {
    return fetch("/api/detector/start", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: JSON.stringify({ source: name, sensitivity: el.sens ? el.sens.value : "standard" })
    }).then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); });
  }

  function uploadFile(file) {
    if (!file || detector.running) return;
    var fd = new FormData(); fd.append("video", file);
    el.vSub.textContent = "uploading " + file.name + " ...";
    el.toggle.disabled = true;
    fetch("/api/detector/upload", { method: "POST", credentials: "same-origin", headers: { "X-CSRF-Token": csrf }, body: fd })
      .then(function (r) {
        if (r.status === 413) throw new Error("file is too large (limit 300 MB)");
        return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || "upload failed"); return j; });
      })
      .then(function (j) {
        el.vErr.hidden = true;
        syncSources(j.status.sources, false);
        el.source.value = j.source;
        return startDetector(j.source);
      })
      .then(function (res) { if (res && !res.ok) showError(res.body.error || "could not start"); })
      .catch(function (e) { showError(e.message); el.vSub.textContent = "annotated detector output"; })
      .then(function () { if (el.upload) el.upload.value = ""; el.toggle.disabled = false; pollStatus(); poll(); });
  }

  if (el.upload) {
    el.upload.addEventListener("change", function () { uploadFile(el.upload.files[0]); });
    var box = document.querySelector(".video");
    ["dragenter", "dragover"].forEach(function (ev) {
      box.addEventListener(ev, function (e) { e.preventDefault(); box.classList.add("drop"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      box.addEventListener(ev, function (e) { e.preventDefault(); box.classList.remove("drop"); });
    });
    box.addEventListener("drop", function (e) { uploadFile(e.dataTransfer.files[0]); });
  }

  function pollStatus() {
    fetch("/api/detector/status", { headers: { "Accept": "application/json" }, credentials: "same-origin" })
      .then(function (r) {
        if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
        return r.json();
      })
      .then(applyStatus)
      .catch(function (e) { if (e.message !== "signed out") setLive(false, "OFFLINE"); });
  }

  if (el.video) {
    el.video.addEventListener("load", function () { showVideo(true); });
    el.video.addEventListener("error", function () { if (detector.running) showVideo(false); });
  }

  if (el.toggle) {
    el.toggle.addEventListener("click", function () {
      var start = !detector.running;
      el.toggle.disabled = true;
      fetch(start ? "/api/detector/start" : "/api/detector/stop", {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({ source: el.source.value, sensitivity: el.sens ? el.sens.value : "standard" })
      })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
        .then(function (res) {
          if (!res.ok) { el.vErr.hidden = false; el.vErr.textContent = res.body.error || "request failed"; el.toggle.disabled = false; return; }
          el.vErr.hidden = true;
          pollStatus(); poll();
        })
        .catch(function (e) { el.vErr.hidden = false; el.vErr.textContent = e.message; el.toggle.disabled = false; });
    });
  }

  poll();
  pollStatus();
  setInterval(poll, POLL_MS);
  setInterval(pollStatus, POLL_MS);
})();
