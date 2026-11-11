// The Lab page: picks the data source, fills the sidebar, loads clips and panes, and wires the
// timeline, view toggles and camera buttons. Lab.app is the small state/event API the knob panel,
// explanations and save/load (knobs.js, explain.js, yaml_edit.js) plug into:
//
//   Lab.app.state = {clipId, entryId, profile, toggles,
//                    panes: {A: {perEntry: {entryId: {values, mode, edge, reroll}}}, B: ...}}
//   Lab.app.on(event, fn) / off(event, fn) / emit(event, data); events:
//     'ready' {source, status, schema, clips}, 'clip' {clipId, base}, 'entry' {entryId, entry},
//     'pane' {which, pane, request}, 'pane-error' {which, error, request}, 'busy' {which, busy},
//     'toggles' toggles, 'frame' frameMap, 'saved' list, 'camera' camera
//   Lab.app.entry(id?) -> schema entry (or a view-only whole-profile entry: {id, title, whole: true})
//   Lab.app.paneSettings(which, entryId?) -> the pane's {values, mode, edge, reroll} (created on use)
//   Lab.app.setPaneSettings(which, patch, {refresh = true}) -> merges and recomputes that pane
//   Lab.app.request(which) -> the PaneRequest for the current clip and entry
//   Lab.app.refreshPane(which) / refreshPanes(): newest request wins; an error keeps the last frames
//   Lab.app.selectClip(id) / selectEntry(id) / reloadClips(selectId?) / reloadSaved()
//   Lab.app.setStatus(text, kind) and Lab.app.mounts (the containers in index.html)
// Settings live per pane and per entry, so switching clips or entries keeps them (spec §5.4, §9).
// Classic script: adds Lab.app to window.Lab; it boots on DOMContentLoaded, so scripts loaded after
// it can still subscribe first.
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var GROUPS = [['body', 'Body'], ['fingers', 'Fingers'], ['face', 'Face'], ['scene', 'Scene'], ['temporal', 'Timing']];
  var PROFILES = [['default', 'Default profile'], ['ambitious', 'Ambitious profile'], ['literature', 'Literature profile']];
  var TOGGLE_LABELS = [
    ['video', 'source video', 'the video frame the original shows'],
    ['original', 'original', 'the unaugmented clip, in blue, in both panes'],
    ['augmented', 'augmented', 'each pane\'s augmented version (A orange, B purple)'],
    ['mesh', 'body mesh', 'the fitted SMPL-X body surface'],
    ['points', 'landmark points', 'dots at the arm, hand and face landmarks'],
    ['skeleton', 'skeleton lines', 'lines joining the landmarks (arms, fingers, face contours)'],
    ['face', 'face', 'face points and contour lines'],
    ['hands', 'hands', 'hand points and finger lines'],
  ];
  var DEFAULT_TOGGLES = { video: true, original: true, augmented: true, mesh: true, points: true, skeleton: true, face: true, hands: true };

  var TOGGLES_KEY = 'labToggles.v2';
  var safeGet = function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } };
  var safeSet = function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } };
  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  var clone = function (x) { return x === undefined ? undefined : JSON.parse(JSON.stringify(x)); };

  function loadToggles() {
    var t = Object.assign({}, DEFAULT_TOGGLES);
    try { Object.assign(t, JSON.parse(safeGet(TOGGLES_KEY) || '{}')); } catch (e) { /* ignore */ }
    return t;
  }

  var listeners = {};
  var app = {
    source: null,
    status: null,
    schema: null,
    clips: [],
    saved: [],
    base: null,
    panes: null,
    timeline: null,
    mounts: {},
    state: {
      clipId: null,
      entryId: null,
      profile: 'default',
      toggles: loadToggles(),
      panes: { A: { perEntry: {} }, B: { perEntry: {} } },
    },
    last: { A: null, B: null }, // the last good pane of each side
    _seq: { A: 0, B: 0 },
    _images: [],
  };

  app.on = function (event, fn) { (listeners[event] = listeners[event] || []).push(fn); return app; };
  app.off = function (event, fn) { listeners[event] = (listeners[event] || []).filter(function (f) { return f !== fn; }); return app; };
  app.emit = function (event, data) {
    (listeners[event] || []).slice().forEach(function (fn) {
      try { fn(data); } catch (err) { console.error('Lab listener for ' + event + ' failed', err); }
    });
  };

  app.setStatus = function (text, kind) {
    var el = app.mounts.status;
    if (!el) return;
    el.textContent = text || '';
    el.className = 'status' + (kind ? ' ' + kind : '');
    el.hidden = !text;
  };

  // ---- entries and pane settings ----------------------------------------------------------
  function wholeEntry(id) {
    var name, title;
    if (id.indexOf('combined.') === 0) {
      name = id.slice('combined.'.length);
      var p = PROFILES.filter(function (x) { return x[0] === name; })[0];
      title = (p ? p[1] : name + ' profile') + ' (whole chain)';
    } else {
      name = id.slice('saved:'.length);
      title = 'Saved setting "' + name + '" (whole chain)';
    }
    return { id: id, title: title, whole: true, extreme: false, knobs: [], group: 'whole', name: name };
  }

  app.entry = function (id) {
    id = id || app.state.entryId;
    if (!id) return null;
    if (id.indexOf('combined.') === 0 || id.indexOf('saved:') === 0) return wholeEntry(id);
    var list = (app.schema && app.schema.entries) || [];
    return list.filter(function (e) { return e.id === id; })[0] || null;
  };

  app.paneSettings = function (which, entryId) {
    entryId = entryId || app.state.entryId;
    var per = app.state.panes[which].perEntry;
    if (!per[entryId]) per[entryId] = { values: {}, mode: 'random', edge: 'high', reroll: 0 };
    return per[entryId];
  };

  app.setPaneSettings = function (which, patch, opts) {
    var s = app.paneSettings(which);
    Object.keys(patch || {}).forEach(function (k) { s[k] = k === 'values' ? clone(patch[k]) : patch[k]; });
    if (!opts || opts.refresh !== false) return app.refreshPane(which);
    return Promise.resolve();
  };

  app.request = function (which) {
    var s = app.paneSettings(which);
    return {
      clip_id: app.state.clipId,
      entry_id: app.state.entryId,
      values: clone(s.values) || {},
      mode: s.mode,
      reroll: s.reroll,
      edge: s.edge,
      profile: app.state.profile,
    };
  };

  function paneTitle(which) {
    var s = app.paneSettings(which);
    var mode = s.mode === 'extreme' ? 'extreme ' + s.edge : 'random · draw ' + s.reroll;
    return mode;
  }

  // ---- panes ------------------------------------------------------------------------------
  app.refreshPane = function (which) {
    if (!app.state.clipId || !app.state.entryId || !app.base) return Promise.resolve(null);
    var seq = ++app._seq[which];
    var req = app.request(which);
    app.panes.setTitle(which, paneTitle(which));
    app.panes.setBusy(which, true);
    app.emit('busy', { which: which, busy: true });
    return app.source.pane(req).then(function (pane) {
      if (seq !== app._seq[which]) return null; // a newer request for this pane superseded it
      app.last[which] = pane;
      app.panes.setPane(which, pane);
      app.panes.setNote(which, null);
      app.timeline.setPane(which, pane.meta.n, pane.arrays.times ? pane.arrays.times.data : null);
      renderReadout(which, pane);
      app.emit('pane', { which: which, pane: pane, request: req });
      return pane;
    }, function (err) {
      if (seq !== app._seq[which]) return null;
      var msg = err && err.message ? err.message : String(err);
      app.panes.setNote(which, (app.last[which] ? 'kept the last frames: ' : '') + msg);
      app.emit('pane-error', { which: which, error: err, request: req });
      return null;
    }).then(function (out) {
      if (seq === app._seq[which]) {
        app.panes.setBusy(which, false);
        app.emit('busy', { which: which, busy: false });
      }
      return out;
    });
  };

  app.refreshPanes = function () { return Promise.all([app.refreshPane('A'), app.refreshPane('B')]); };

  function renderReadout(which, pane) {
    var el = app.mounts['readout' + which];
    if (!el) return;
    if (Lab.readoutHtml) { el.innerHTML = Lab.readoutHtml(pane.meta, which); return; }
    var m = pane.meta;
    var applied = (m.applied || []).join(', ') || 'nothing applied';
    var fb = (m.fallbacks || []).length;
    el.innerHTML = '<b class="tag-' + which.toLowerCase() + '">' + which + '</b> ' + esc(m.n) + ' frames · ' +
      esc(applied) + (m.mirrored ? ' · mirrored' : '') + (fb ? ' · ' + fb + ' fallback(s)' : '');
  }

  function renderExplain() {
    var el = app.mounts.explain;
    var entry = app.entry();
    if (!el) return;
    if (!entry) { el.innerHTML = ''; return; }
    if (Lab.explainHtml) { el.innerHTML = Lab.explainHtml(entry); return; }
    if (entry.whole) {
      el.innerHTML = '<h2>' + esc(entry.title) + '</h2><p>Every augmentation the profile turns on, as one ' +
        'sample per pane. View only: knobs are edited one augmentation at a time.</p>';
      return;
    }
    el.innerHTML = '<h2>' + esc(entry.title) + ' <span class="tech">' + esc(entry.id) + '</span></h2><p>' +
      esc(entry.does) + '</p>';
  }

  // ---- source video frame -----------------------------------------------------------------
  function loadFrames(clipId, n) {
    app._images = [];
    for (var i = 0; i < n; i++) {
      var im = new Image();
      im.decoding = 'async';
      im.onload = (function (idx) {
        return function () { if (app.state.clipId === clipId && app._frame === idx) drawFrame(idx); };
      })(i);
      im.onerror = function () { /* a missing frame just leaves the last one */ };
      im.src = app.source.frameUrl(clipId, i);
      app._images.push(im);
    }
  }

  function drawFrame(i) {
    app._frame = i;
    var canvas = app.mounts.sourceFrame, im = app._images[i];
    if (app.mounts.sourceLabel) app.mounts.sourceLabel.textContent = app.base ? (i + 1) + ' / ' + app.base.meta.n : '';
    if (!canvas || !im || !im.complete || !im.naturalWidth) return;
    if (canvas.width !== im.naturalWidth || canvas.height !== im.naturalHeight) {
      canvas.width = im.naturalWidth;
      canvas.height = im.naturalHeight;
    }
    canvas.getContext('2d').drawImage(im, 0, 0);
  }

  function onFrame(fm) {
    var A = fm.A || { k: 0, ended: false }, B = fm.B || { k: 0, ended: false };
    app.panes.showFrame(A.k, B.k, fm.base, { A: A.ended, B: B.ended });
    drawFrame(Math.max(0, Math.min(fm.base, app._images.length - 1)));
    app.emit('frame', fm);
  }

  // ---- clips and entries ------------------------------------------------------------------
  app.selectClip = function (id) {
    if (!id) return Promise.resolve();
    // state.clipId changes only once the clip's base has loaded, so pane requests made meanwhile
    // (and after a failure) stay on the clip whose base is shown.
    var previous = app.state.clipId;
    markSelected(app.mounts.clipList, id);
    app.timeline.pause();
    app.setStatus('loading clip ' + id + '… (the first time a clip opens, the server fits the body: about 10 s)', 'busy');
    var token = ++app._clipSeq;
    return app.source.base(id).then(function (base) {
      if (token !== app._clipSeq) return;
      app.state.clipId = id;
      writeHash();
      app.base = base;
      app.last = { A: null, B: null };
      loadFrames(id, base.meta.n); // before the timeline shows frame 0, so its image is awaited
      app.panes.setBase(base);
      app.timeline.setClip(base.meta.n, base.arrays.times ? base.arrays.times.data : null, base.meta.fps);
      app.setStatus('');
      app.emit('clip', { clipId: id, base: base });
      return app.refreshPanes();
    }, function (err) {
      if (token !== app._clipSeq) return;
      app.state.clipId = previous;
      markSelected(app.mounts.clipList, previous);
      app.setStatus('could not load clip ' + id + ': ' + err.message, 'error');
    });
  };
  app._clipSeq = 0;

  app.selectEntry = function (id) {
    if (!id || !app.entry(id)) return Promise.resolve();
    app.state.entryId = id;
    markSelected(app.mounts.entryList, id);
    markSelected(app.mounts.profileList, id);
    markSelected(app.mounts.savedList, id);
    writeHash();
    renderExplain();
    app.emit('entry', { entryId: id, entry: app.entry(id) });
    return app.refreshPanes();
  };

  function markSelected(list, id) {
    if (!list) return;
    Array.prototype.forEach.call(list.querySelectorAll('[data-id]'), function (b) {
      b.classList.toggle('on', b.dataset.id === id);
      b.setAttribute('aria-pressed', b.dataset.id === id ? 'true' : 'false');
    });
  }

  function button(id, html, onclick, title) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'item';
    b.dataset.id = id;
    b.innerHTML = html;
    if (title) b.title = title;
    b.onclick = onclick;
    return b;
  }

  function renderClips() {
    var list = app.mounts.clipList;
    list.innerHTML = '';
    if (!app.clips.length) list.innerHTML = '<p class="empty">no clips yet</p>';
    app.clips.forEach(function (c) {
      var frames = c.n || c.frames;
      list.appendChild(button(c.id, esc(c.id) + (frames ? ' <span class="tech">' + frames + ' frames</span>' : ''),
        function () { app.selectClip(c.id); }));
    });
    markSelected(list, app.state.clipId);
  }

  function renderEntries() {
    var list = app.mounts.entryList;
    list.innerHTML = '';
    var entries = (app.schema && app.schema.entries) || [];
    GROUPS.forEach(function (g) {
      var mine = entries.filter(function (e) { return e.group === g[0]; });
      if (!mine.length) return;
      var h = document.createElement('h3');
      h.textContent = g[1];
      list.appendChild(h);
      mine.forEach(function (e) {
        list.appendChild(button(e.id, esc(e.title) + ' <span class="tech">' + esc(e.id) + '</span>',
          function () { app.selectEntry(e.id); }, e.does));
      });
    });
    markSelected(list, app.state.entryId);
  }

  function renderProfiles() {
    var list = app.mounts.profileList;
    list.innerHTML = '';
    PROFILES.forEach(function (p) {
      list.appendChild(button('combined.' + p[0], esc(p[1]) + ' <span class="tech">slr_week4_' + p[0] + '.yaml</span>',
        function () { app.selectEntry('combined.' + p[0]); }));
    });
    markSelected(list, app.state.entryId);
  }

  function renderSaved() {
    var list = app.mounts.savedList;
    list.innerHTML = '';
    if (!app.saved.length) {
      list.innerHTML = '<p class="empty">' + (app.source.isShare ? 'saved settings live in full mode' : 'none saved yet') + '</p>';
    }
    app.saved.forEach(function (s) {
      list.appendChild(button('saved:' + s.name, esc(s.name), function () { app.selectEntry('saved:' + s.name); }));
    });
    markSelected(list, app.state.entryId);
  }

  app.reloadSaved = function () {
    return app.source.saved().then(function (list) {
      app.saved = list || [];
      renderSaved();
      app.emit('saved', app.saved);
      return app.saved;
    }, function () { app.saved = []; renderSaved(); return []; });
  };

  app.reloadClips = function (selectId) {
    return app.source.clips().then(function (clips) {
      app.clips = clips || [];
      renderClips();
      if (selectId) return app.selectClip(selectId);
    });
  };

  // ---- toggles and camera -----------------------------------------------------------------
  app.setToggles = function (patch) {
    Object.assign(app.state.toggles, patch || {});
    safeSet(TOGGLES_KEY, JSON.stringify(app.state.toggles));
    applyToggles();
  };

  function applyToggles() {
    var t = app.state.toggles;
    app.panes.setToggles(t);
    if (app.mounts.stage) app.mounts.stage.classList.toggle('no-video', !t.video);
    Array.prototype.forEach.call(app.mounts.toggles.querySelectorAll('input[data-toggle]'), function (box) {
      box.checked = !!t[box.dataset.toggle];
    });
    app.panes.requestRender();
    app.emit('toggles', t);
  }

  function buildToggles() {
    var row = app.mounts.toggles;
    var cams = document.createElement('div');
    cams.className = 'cams';
    cams.innerHTML = '<span class="label">camera</span>';
    [['front', 'front'], ['side', 'side'], ['hands', 'hands'], ['face', 'face']].forEach(function (c) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = c[1];
      b.dataset.cam = c[0];
      b.onclick = function () { app.panes.cameraPreset(c[0]); };
      cams.appendChild(b);
    });
    var boxes = document.createElement('div');
    boxes.className = 'view-toggles';
    boxes.innerHTML = '<span class="label">show</span>' + TOGGLE_LABELS.map(function (t) {
      return '<label title="' + esc(t[2]) + '"><input type="checkbox" data-toggle="' + t[0] + '"> ' + esc(t[1]) + '</label>';
    }).join('');
    var legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = '<i style="--c: var(--orig)"></i>original <i style="--c: var(--pane-a)"></i>A ' +
      '<i style="--c: var(--pane-b)"></i>B';
    row.innerHTML = '';
    row.appendChild(cams);
    row.appendChild(boxes);
    row.appendChild(legend);
    Array.prototype.forEach.call(boxes.querySelectorAll('input'), function (box) {
      box.onchange = function () { var p = {}; p[box.dataset.toggle] = box.checked; app.setToggles(p); };
    });
  }

  // ---- URL hash ---------------------------------------------------------------------------
  function readHash() {
    var h = {};
    location.hash.slice(1).split('&').forEach(function (kv) {
      var i = kv.indexOf('=');
      if (i > 0) h[decodeURIComponent(kv.slice(0, i))] = decodeURIComponent(kv.slice(i + 1));
    });
    return h;
  }
  function writeHash() {
    var parts = [];
    if (app.state.clipId) parts.push('clip=' + encodeURIComponent(app.state.clipId));
    if (app.state.entryId) parts.push('entry=' + encodeURIComponent(app.state.entryId));
    if (app.timeline) parts.push('clock=' + app.timeline.clock);
    try { history.replaceState(null, '', '#' + parts.join('&')); } catch (e) { /* file:// in some browsers */ }
  }

  // ---- start ------------------------------------------------------------------------------
  function findMounts() {
    app.mounts = {
      status: $('status'),
      clipList: $('clip-list'),
      addVideo: $('add-video'),
      entryList: $('entry-list'),
      profileList: $('profile-list'),
      savedList: $('saved-list'),
      savedActions: $('saved-actions'),
      shareFooter: $('share-footer'),
      modeBadge: $('mode-badge'),
      explain: $('explain'),
      stage: $('stage'),
      sourceFrame: $('source-frame'),
      sourceLabel: $('source-label'),
      panes: $('panes'),
      timeline: $('timeline'),
      toggles: $('toggles'),
      readoutA: $('readout-A'),
      readoutB: $('readout-B'),
      knobs: $('knobs'),
    };
  }

  function setupMode(status) {
    var missing = (status.missing || []).map(function (m) { return '<li><code>' + esc(m) + '</code></li>'; }).join('');
    app.mounts.explain.innerHTML = '<h2>The model files are missing</h2><p>Full mode needs the body, hand and ' +
      'face model files. Missing from <code>' + esc(status.model_dir || '(no folder set)') + '</code>:</p><ul>' +
      missing + '</ul><p><a href="setup.html">Open the setup page</a> to point the Lab at your model folder.</p>';
    app.setStatus('setup needed: model files missing', 'error');
  }

  app.start = function () {
    findMounts();
    if (typeof THREE === 'undefined' || !THREE.OrbitControls) {
      app.setStatus('three.js did not load (vendor/three.min.js)', 'error');
      return Promise.resolve();
    }
    var hash = readHash();
    try {
      app.panes = new Lab.Panes(app.mounts.panes, { onCameraChange: function () { app.emit('camera', app.panes.camera); } });
    } catch (err) {
      console.error(err);
      app.setStatus('This browser can\'t draw 3-D (WebGL is off): ' + (err && err.message ? err.message : err), 'error');
      return Promise.resolve();
    }
    app.timeline = new Lab.Timeline(app.mounts.timeline, {
      clock: hash.clock, onFrame: onFrame,
    });
    app.mounts.timeline.querySelector('select').addEventListener('change', writeHash);
    buildToggles();
    applyToggles();
    app.setStatus('connecting…', 'busy');
    return Lab.chooseSource().then(function (chosen) {
      app.source = chosen.source;
      app.status = chosen.status;
      document.body.classList.toggle('share', app.source.isShare);
      document.body.classList.toggle('live', !app.source.isShare);
      app.mounts.modeBadge.textContent = app.source.isShare ? 'shared version' : 'full mode';
      app.mounts.shareFooter.hidden = !app.source.isShare;
      if (!app.source.isShare && chosen.status.mode === 'setup') { setupMode(chosen.status); return null; }
      return Promise.all([app.source.schema(app.state.profile), app.source.clips()]).then(function (got) {
        app.schema = got[0];
        app.clips = got[1] || [];
        renderClips();
        renderEntries();
        renderProfiles();
        app.setStatus('');
        var clipIds = app.clips.map(function (c) { return c.id; });
        var entryOk = hash.entry && app.entry(hash.entry);
        app.state.entryId = entryOk ? hash.entry : (app.schema.entries[0] || {}).id || null;
        markSelected(app.mounts.entryList, app.state.entryId);
        markSelected(app.mounts.profileList, app.state.entryId);
        renderExplain();
        app.emit('ready', { source: app.source, status: app.status, schema: app.schema, clips: app.clips });
        app.emit('entry', { entryId: app.state.entryId, entry: app.entry() });
        var savedDone = app.reloadSaved().then(function () { markSelected(app.mounts.savedList, app.state.entryId); });
        var clip = clipIds.indexOf(hash.clip) >= 0 ? hash.clip : clipIds[0];
        if (!clip) { app.setStatus('no clips yet' + (app.source.isShare ? '' : ': add a video'), 'error'); return savedDone; }
        return Promise.all([savedDone, app.selectClip(clip)]);
      });
    }).catch(function (err) {
      console.error(err);
      app.setStatus('could not start the Lab: ' + (err && err.message ? err.message : err), 'error');
    });
  };

  Lab.app = app;
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', app.start);
    else setTimeout(app.start, 0);
  }
})();
