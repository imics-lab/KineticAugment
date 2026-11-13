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
//   Lab.app.knobPanel (the Lab.KnobPanel of the selected entry), buildSaved(name, which) ->
//     Promise<{name, text, entries}>, loadSettings(text, which, label) -> message; the Save / Load
//     forms under the saved list and the full-mode "＋ add video" flow are wired on 'ready'.
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
      if (err && err.status === 422) {  // the server refused the value itself
        app.panes.setNote(which, 'This value isn\'t allowed: ' + (err.detail || msg) + (app.last[which] ? ' (kept the last frames)' : ''));
      } else {
        app.panes.setNote(which, (app.last[which] ? 'kept the last frames: ' : '') + msg);
      }
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
      // A shared version is built for one profile; its pane keys carry that profile's name.
      var sharedProfile = app.source.isShare && app.source.profile();
      if (sharedProfile) app.state.profile = sharedProfile;
      app.mounts.modeBadge.textContent = app.source.isShare
        ? 'shared version' + (sharedProfile ? ' \u00b7 ' + sharedProfile + ' profile' : '') : 'full mode';
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

  // ---- knob panel (knobs.js) --------------------------------------------------------------
  function mountKnobs() {
    var entry = app.entry();
    if (app.knobPanel) { app.knobPanel.destroy(); app.knobPanel = null; }
    if (!entry || !Lab.KnobPanel || !app.mounts.knobs) return;
    app.knobPanel = new Lab.KnobPanel(app.mounts.knobs, entry, app.state, {
      isShare: app.source.isShare,
      stops: function (entryId, key) { return app.source.stops(entryId, key); },
      rerolls: app.source.isShare ? ((app.source.data || {}).rerolls || 0) : 0,
      onChange: function (which) { app.refreshPane(which); },
    });
  }
  app.on('entry', function () { if (app.source) mountKnobs(); });
  app.on('pane', function (e) {
    if (app.knobPanel) app.knobPanel.paneResult(e.which, e.pane.meta, app.base ? app.base.meta.n : null);
  });

  // ---- save and load settings (yaml_edit.js; spec §5.7) -----------------------------------
  // Python's saving.slugify: lowercase [a-z0-9_], other runs one '-', at most 60 characters.
  function slugify(name) {
    return String(name).toLowerCase().replace(/[^a-z0-9_]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60).replace(/-+$/, '');
  }
  app.slugify = slugify;

  function profileText() {
    if (!app._profileText) {
      app._profileText = app.source.profileText(app.state.profile).catch(function (err) {
        app._profileText = null;
        throw err;
      });
    }
    return app._profileText;
  }

  function entryTitles(ids) {
    return ids.map(function (id) { var e = app.entry(id); return e ? e.title : id; }).join(', ');
  }

  // The saved YAML for one pane: {name, text, entries} (throws when nothing was changed).
  app.buildSaved = function (name, which) {
    var slug = slugify(name);
    if (!slug) return Promise.reject(new Error('the name needs at least one letter or digit'));
    var perEntry = app.state.panes[which].perEntry;
    if (!Lab.savedEdits(app.schema, perEntry).entries.length) {
      return Promise.reject(new Error('pane ' + which + ' has no changed knobs yet: move a knob first'));
    }
    return profileText().then(function (base) {
      var got = Lab.savedEdits(app.schema, perEntry, Lab.parseYaml(base));
      return {
        name: slug,
        text: Lab.buildSavedYaml(base, got.edits, got.entries, { name: slug, weights: got.weights }),
        entries: got.entries,
      };
    });
  };

  function download(name, text) {
    var url = URL.createObjectURL(new Blob([text], { type: 'application/x-yaml' }));
    var a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // Sets one pane's values from a profile's text: every lab entry the file turns on. The shared
  // version snaps each value to its stops and keeps one knob off default per entry.
  app.loadSettings = function (text, which, label) {
    var data = Lab.parseYaml(text);
    var got = Lab.valuesFromProfile(app.schema, data);
    var ids = Object.keys(got);
    var dropped = [];
    ids.forEach(function (id) {
      var values = got[id];
      if (app.source.isShare) {
        var entry = app.entry(id), K = Lab.knobMath, defaults = {}, kept = {};
        entry.knobs.forEach(function (k) { defaults[k.key] = k.default; });
        entry.knobs.forEach(function (k) {
          if (!(k.key in values)) return;
          // a single-choice enum has every option worked out ahead; a scalar needs stops
          var choice = k.kind === 'enum' && !k.multi && k.options.indexOf(values[k.key]) >= 0;
          var stops = choice ? [] : (k.share && k.kind !== 'enum' ? app.source.stops(id, k.key) : null);
          if (!stops || Object.keys(kept).length) { dropped.push(k.technical); return; }
          var v = choice ? values[k.key] : K.snap(values[k.key], stops);
          if (k.kind === 'range' && typeof v === 'number') v = K.rangeFromMarker(k, v);
          kept = K.shareSet(kept, k.key, v, defaults).values;
        });
        values = kept;
      }
      var s = app.paneSettings(which, id);
      s.values = values;
      if (app.source.isShare) s.reroll = 0;
    });
    mountKnobs();
    app.refreshPane(which);
    var msg = ids.length
      ? 'Loaded ' + label + ' into pane ' + which + ': ' + entryTitles(ids) + '.'
      : label + ' turns on none of the Lab\'s augmentations; nothing changed.';
    if (dropped.length) msg += ' Not in the shared version (live only, or one knob at a time): ' + dropped.join(', ') + '.';
    return msg;
  };

  function paneRadios(name) {
    return '<span class="pane-pick" role="radiogroup" aria-label="which pane">pane ' +
      '<label class="tag-a"><input type="radio" name="' + name + '" value="A" checked> A</label> ' +
      '<label class="tag-b"><input type="radio" name="' + name + '" value="B"> B</label></span>';
  }

  function buildSavedActions() {
    var box = app.mounts.savedActions;
    if (!box || !Lab.buildSavedYaml) return;
    var share = app.source.isShare;
    box.innerHTML =
      '<div class="sa-buttons"><button type="button" data-sa="save">Save setting…</button>' +
      '<button type="button" data-sa="load">Load setting…</button></div>' +
      '<form class="sa-form" data-form="save" hidden>' +
      '<label>Name <input name="name" required maxlength="60" placeholder="elbow-test" autocomplete="off"></label>' +
      paneRadios('save-pane') +
      '<p class="small sa-what"></p>' +
      '<div class="sa-row"><button type="submit">' + (share ? 'Download .yaml' : 'Save') + '</button>' +
      '<button type="button" data-sa="cancel">Cancel</button></div></form>' +
      '<form class="sa-form" data-form="load" hidden>' +
      (share ? '<label>File <input type="file" name="file" accept=".yaml,.yml,text/yaml"></label>'
        : '<label>Setting <select name="name"></select></label>') +
      paneRadios('load-pane') +
      '<div class="sa-row"><button type="submit">Load</button><button type="button" data-sa="cancel">Cancel</button></div></form>' +
      '<p class="small sa-msg" role="status" hidden></p>';
    var saveForm = box.querySelector('[data-form=save]'), loadForm = box.querySelector('[data-form=load]');
    var msg = box.querySelector('.sa-msg');
    var say = function (html, kind) { msg.innerHTML = html; msg.hidden = !html; msg.className = 'small sa-msg' + (kind ? ' ' + kind : ''); };
    var pane = function (form) { return form.querySelector('input[type=radio]:checked').value; };
    var describe = function () {
      var which = pane(saveForm);
      var got = Lab.savedEdits(app.schema, app.state.panes[which].perEntry);
      var render = function (turned) {
        var note = turned.length ? ' Turned on ' + entryTitles(turned) + ' in the saved profile so your change is used in training.' : '';
        saveForm.querySelector('.sa-what').textContent = describe.text + note;
      };
      var text = got.ids.length
        ? 'Saves the base profile with pane ' + which + '\'s changed knobs, each augmentation turned on: ' +
          entryTitles(got.ids) + '.'
        : 'Pane ' + which + ' has no changed knobs yet.';
      if (got.ids.some(function (id) { return Lab.demoOverrides && Lab.demoOverrides(app.entry(id)).length; })) {
        text += ' ' + Lab.DEMO_NOT_SAVED;
      }
      describe.text = text;
      render([]);
      if (got.ids.length) {
        profileText().then(function (base) {
          if (pane(saveForm) === which) render(Lab.savedEdits(app.schema, app.state.panes[which].perEntry, Lab.parseYaml(base)).turnedOn);
        }).catch(function () { /* the base text is read again when saving */ });
      }
    };
    Array.prototype.forEach.call(saveForm.querySelectorAll('input[type=radio]'), function (r) { r.onchange = describe; });
    box.querySelector('[data-sa=save]').onclick = function () {
      loadForm.hidden = true;
      saveForm.hidden = !saveForm.hidden;
      say('');
      if (!saveForm.hidden) { describe(); saveForm.querySelector('input[name=name]').focus(); }
    };
    box.querySelector('[data-sa=load]').onclick = function () {
      saveForm.hidden = true;
      say('');
      if (!share) {
        var sel = loadForm.querySelector('select');
        sel.innerHTML = app.saved.map(function (s) { return '<option>' + esc(s.name) + '</option>'; }).join('');
        if (!app.saved.length) { say('No saved settings yet.'); loadForm.hidden = true; return; }
      }
      loadForm.hidden = !loadForm.hidden;
    };
    Array.prototype.forEach.call(box.querySelectorAll('[data-sa=cancel]'), function (b) {
      b.onclick = function () { saveForm.hidden = true; loadForm.hidden = true; say(''); };
    });

    var doSave = function (name, which, overwrite) {
      return app.buildSaved(name, which).then(function (out) {
        if (share) {
          download(out.name + '.yaml', out.text);
          say('Downloaded <code>' + esc(out.name) + '.yaml</code> (pane ' + which + ').');
          saveForm.hidden = true;
          return null;
        }
        return app.source.save(out.name, out.text, overwrite).then(function (res) {
          saveForm.hidden = true;
          say('Saved pane ' + which + ' to <code>' + esc(res.path || 'configs/viewer_saved/' + res.name + '.yaml') + '</code>.');
          return app.reloadSaved();
        }, function (err) {
          if (err.status !== 409) throw err;
          say('A saved setting called "' + esc(out.name) + '" already exists. Replace it? ' +
            '<button type="button" data-sa="replace">Replace</button> <button type="button" data-sa="keep">Keep it</button>', 'warn');
          msg.querySelector('[data-sa=replace]').onclick = function () { doSave(name, which, true); };
          msg.querySelector('[data-sa=keep]').onclick = function () { say(''); };
          return null;
        });
      }).catch(function (err) { say('Could not save: ' + esc(err.detail || err.message), 'error'); });
    };
    saveForm.onsubmit = function (ev) {
      ev.preventDefault();
      doSave(saveForm.querySelector('input[name=name]').value, pane(saveForm), false);
    };
    loadForm.onsubmit = function (ev) {
      ev.preventDefault();
      var which = pane(loadForm);
      var read;
      if (share) {
        var file = loadForm.querySelector('input[type=file]').files[0];
        if (!file) { say('Pick a .yaml file first.', 'error'); return; }
        read = file.text().then(function (t) { return { text: t, label: file.name }; });
      } else {
        var name = loadForm.querySelector('select').value;
        read = app.source.savedText(name).then(function (t) { return { text: t, label: '"' + name + '"' }; });
      }
      read.then(function (got) {
        say(esc(app.loadSettings(got.text, which, got.label)));
        loadForm.hidden = true;
      }).catch(function (err) { say('Could not load: ' + esc(err.detail || err.message), 'error'); });
    };
  }

  // ---- add a video (full mode only) -------------------------------------------------------
  var STEP_WORDS = { '': 'starting', 'extracting landmarks': 'extracting landmarks', fitting: 'fitting the body',
    'saving frames': 'saving frames', done: 'done' };

  function buildAddVideo() {
    var btn = app.mounts.addVideo, status = document.getElementById('add-video-status');
    if (!btn || app.source.isShare || !app.source.addVideo) return;
    btn.hidden = false;
    var input = document.createElement('input');
    input.type = 'file';
    input.accept = '.mp4,.mov,.m4v,.avi,.webm'; // the server's VIDEO_SUFFIXES
    input.hidden = true;
    btn.parentNode.insertBefore(input, btn.nextSibling);
    var show = function (text, kind) {
      if (!status) return;
      status.textContent = text;
      status.className = 'small' + (kind ? ' ' + kind : '');
    };
    btn.onclick = function () { input.value = ''; input.click(); };
    input.onchange = function () {
      var file = input.files[0];
      if (!file) return;
      btn.disabled = true;
      show('uploading ' + file.name + '…');
      app.source.addVideo(file).then(function (job) {
        return Lab.watchJob(app.source, job.job_id, {
          onUpdate: function (s) {
            var step = s.state === 'waiting' ? 'waiting' : STEP_WORDS[s.step] || s.step || s.state;
            show(job.clip_id + ': ' + step + (s.state === 'running' ? ' (' + Math.round(100 * (s.progress || 0)) + '%)' : ''));
          },
        }).then(function () {
          show(job.clip_id + ': done');
          return app.reloadClips(job.clip_id);
        });
      }).catch(function (err) {
        show('could not add the video: ' + (err.detail || err.message), 'error');
      }).then(function () { btn.disabled = false; });
    };
  }

  app.on('ready', function () {
    buildSavedActions();
    buildAddVideo();
  });

  Lab.app = app;
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', app.start);
    else setTimeout(app.start, 0);
  }
})();
