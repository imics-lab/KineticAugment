// Plain-English text for the Lab (spec §5.3):
//   Lab.explainHtml(entry) -> the explanation box: what it does, what it changes, what it keeps,
//     proven vs used range, note (a whole profile: what it runs, and that it is view only).
//   Lab.readoutHtml(meta, which) -> the readout of one pane: its draws, the report.measure numbers
//     in words with units, and any fallback.
//   Lab.unchangedReason(meta, baseN) -> a short sentence when a pane's draw changed nothing (no
//     entry applied, or a frame drop that kept every frame), else ''.
// Classic script: adds to window.Lab and module.exports.
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  function num(v, digits) {
    if (typeof v !== 'number') return String(v);
    if (!isFinite(v)) return v > 0 ? '∞' : '−∞';
    if (Math.abs(v) < 1e-9) return '0';
    return String(Number(v.toPrecision(digits || 3)));
  }
  function list(items) {
    return '<ul>' + (items || []).map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul>';
  }

  var PROFILE_TEXT = {
    default: 'The default week-4 profile as one sample: mirror, camera angle, camera distance, off-centre ' +
      'signer, whole-clip speed, arm and finger joint noise and random dropped frames, each at its used range.',
    ambitious: 'The ambitious week-4 profile as one sample: every augmentation on, at its used range.',
    literature: 'The literature week-4 profile as one sample: the default augmentations at the ranges the ' +
      'literature has shown to be safe.',
  };

  Lab.explainHtml = function explainHtml(entry) {
    if (!entry) return '';
    if (entry.whole) {
      var saved = entry.id.indexOf('saved:') === 0;
      var text = saved
        ? 'Your saved setting "' + entry.name + '": the base profile with your changes, every augmentation ' +
          'it turns on, one sample per pane.'
        : PROFILE_TEXT[entry.name] || 'Every augmentation the profile turns on, one sample per pane.';
      return '<h2>' + esc(entry.title) + ' <span class="tech">' + esc(saved ? 'configs/viewer_saved/' + entry.name + '.yaml'
        : 'slr_week4_' + entry.name + '.yaml') + '</span></h2><p>' + esc(text) + '</p>' +
        '<p class="ex-note">View only: knobs are set one augmentation at a time. Use "new draw" for ' +
        'another sample.</p>';
    }
    var html = '<h2>' + esc(entry.title) + ' <span class="tech">' + esc(entry.id) + '</span></h2>';
    html += '<p class="ex-does">' + esc(entry.does) + '</p>';
    html += '<div class="ex-cols">' +
      '<div><h3>Changes</h3>' + list(entry.changes) + '</div>' +
      '<div><h3>Keeps</h3>' + list(entry.keeps) + '</div>' +
      '<div><h3>Proven vs used</h3><p><b class="ex-proven">Proven:</b> ' + esc(entry.proven) + '</p>' +
      '<p><b class="ex-used">Used:</b> ' + esc(entry.used) + '</p></div></div>';
    var demo = Object.keys(entry.overrides || {}).filter(function (k) { return k !== 'mode_weights'; });
    if (demo.length) {
      html += '<p class="ex-note">In this lab the panes start from ' + demo.map(function (k) {
        var v = entry.overrides[k];
        return esc(k) + ' = ' + esc(Array.isArray(v) ? '[' + v.join(', ') + ']' : v);
      }).join(', ') + ' rather than the profile\'s own value, so that the effect shows. The knob starts there; ' +
        'drag it to change it.</p>';
    }
    if (entry.note) html += '<p class="ex-note">' + esc(entry.note) + '</p>';
    return html;
  };

  // ---- readout --------------------------------------------------------------------------
  var METRICS = [
    ['wrist_moved_mm_max', function (v) { return 'The wrists moved at most ' + num(v) + ' mm.'; }],
    ['bone_length_change_mm_max', function (v) { return 'Bone lengths changed by at most ' + num(v) + ' mm.'; }],
    ['jitter_ratio', function (v) { return 'Wrist shake is ' + num(v) + '× the original\'s (1 = unchanged).'; }],
    ['face_rigidity_mm', function (v) {
      return 'The face points wobble ' + num(v) + ' mm more against the head than in the original (should be about 0).';
    }],
    ['hand_face_cm_min', function (v) { return 'The closest a hand comes to the face: ' + num(v) + ' cm.'; }],
    ['in_frame_pct', function (v) { return num(v, 4) + '% of the points stay inside the video frame.'; }],
  ];
  var FALLBACKS = {
    no_camera: 'skipped: the clip has no fitted camera',
    no_valid_redraw: 'every redraw would have changed the handshape, so the hands were left unchanged',
    too_few_face_frames: 'too few frames with a face, so the face was left unchanged',
  };

  function unitOf(key) {
    if (/_deg$/.test(key)) return '°';
    if (/_mm$/.test(key)) return ' mm';
    if (/_ms$/.test(key)) return ' ms';
    if (/fps$/.test(key)) return ' fps';
    return '';
  }

  function numericLeaves(v, out) {
    if (typeof v === 'number') out.push(v);
    else if (Array.isArray(v)) v.forEach(function (x) { numericLeaves(x, out); });
    else if (v && typeof v === 'object') Object.keys(v).forEach(function (k) { numericLeaves(v[k], out); });
    return out;
  }

  // One draw value in a few words.
  function drawValue(key, v, n) {
    if (v === null || v === undefined) return null;
    if (typeof v === 'number') return num(v) + unitOf(key);
    if (typeof v !== 'object') return String(v);
    if (key === 'keep' && Array.isArray(v)) return v.length + ' of ' + (n || '?') + ' frames';
    if (Array.isArray(v) && v.length && v[0] && typeof v[0] === 'object' && 'angle_deg' in v[0]) {
      return v.map(function (j) { return [j.side, j.joint, num(j.angle_deg) + '°'].filter(Boolean).join(' '); }).join(', ');
    }
    if (Array.isArray(v) && v.length <= 6 && v.every(function (x) { return typeof x !== 'object'; })) {
      return '[' + v.map(function (x) { return typeof x === 'number' ? num(x) : String(x); }).join(', ') + ']' + unitOf(key);
    }
    if (!Array.isArray(v) && Object.keys(v).length <= 6 && numericLeaves(v, []).length <= 6) {
      var parts = Object.keys(v).map(function (k) {
        var s = drawValue(k, v[k], n);
        return s === null ? null : k + ' ' + s;
      }).filter(Boolean);
      return parts.join(', ');
    }
    var leaves = numericLeaves(v, []);
    if (!leaves.length) return JSON.stringify(v).slice(0, 80);
    var big = leaves.reduce(function (m, x) { return Math.max(m, Math.abs(x)); }, 0);
    return leaves.length + ' values, largest ' + num(big) + unitOf(key);
  }

  function entryTitle(name) {
    var schema = Lab.app && Lab.app.schema;
    var hit = schema && (schema.entries || []).filter(function (e) { return e.entry === name; })[0];
    return hit ? hit.title.replace(/:.*$/, '') : name;
  }

  Lab.unchangedReason = function unchangedReason(meta, baseN) {
    if (!meta || !Array.isArray(meta.applied)) return '';
    if (!meta.applied.length) return 'This draw left the clip unchanged.';
    var draws = meta.draws || {};
    var noop = meta.applied.every(function (name) {
      var d = draws[name];
      return /^temporal\.frame_drop/.test(name) && d && Array.isArray(d.keep) && baseN && d.keep.length === baseN;
    });
    if (!noop) return '';
    return meta.mode === 'extreme'
      ? 'The ' + meta.edge + ' edge of this setting drops no frame (its drop chance is 0), so this pane equals the original.'
      : 'This draw dropped no frame, so this pane equals the original.';
  };

  Lab.readoutHtml = function readoutHtml(meta, which) {
    if (!meta) return '';
    which = which || '';
    var head = meta.mode === 'extreme' ? 'extreme, ' + meta.edge + ' edge' : 'random draw ' + (meta.reroll || 0);
    var html = '<div class="ro-head"><b class="tag-' + which.toLowerCase() + '">' + esc(which) + '</b> ' + esc(head) +
      ' · ' + esc(meta.n) + ' frames' + (meta.mirrored ? ' · mirrored' : '') + '</div>';
    var applied = meta.applied || [];
    if (!applied.length) html += '<p>Nothing was applied: this draw left the clip unchanged.</p>';
    var draws = meta.draws || {};
    var rows = Object.keys(draws).map(function (name) {
      var d = draws[name] || {};
      var parts = Object.keys(d).filter(function (k) { return k !== 'wrist_moved_mm_max'; }).map(function (k) {
        var s = drawValue(k, d[k], meta.n);
        return s === null || s === '' ? null : '<span class="ro-k">' + esc(k) + '</span> ' + esc(s);
      }).filter(Boolean);
      return '<li><b>' + esc(entryTitle(name)) + '</b> <span class="tech">' + esc(name) + '</span>' +
        (applied.indexOf(name) < 0 ? ' (not applied)' : '') + (parts.length ? ': ' + parts.join('; ') : '') + '</li>';
    });
    if (rows.length) html += '<h4>Drawn</h4><ul class="ro-draws">' + rows.join('') + '</ul>';
    var m = meta.metrics || {};
    var lines = METRICS.filter(function (x) { return typeof m[x[0]] === 'number'; }).map(function (x) {
      return '<li>' + esc(x[1](m[x[0]])) + ' <span class="tech">' + esc(x[0]) + '</span></li>';
    });
    if (lines.length) html += '<h4>Measured</h4><ul class="ro-metrics">' + lines.join('') + '</ul>';
    var fb = meta.fallbacks || {};
    var fbl = (Array.isArray(fb) ? fb.map(function (r, i) { return [String(i), r]; })
      : Object.keys(fb).map(function (k) { return [k, fb[k]]; })).map(function (p) {
      return '<li><b>' + esc(entryTitle(p[0])) + '</b>: ' + esc(FALLBACKS[p[1]] || p[1]) + '</li>';
    });
    if (fbl.length) html += '<h4>Fallbacks</h4><ul class="ro-fallbacks">' + fbl.join('') + '</ul>';
    return html;
  };
})();

if (typeof module !== 'undefined') module.exports = Lab;
