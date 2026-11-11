// The knob panel (spec §5.2, §6.1): one bar per knob with an orange A and a purple B marker,
// coloured green (proven), amber (past proven up to the profile's used value) and grey (beyond, up
// to the slider limit). Enum knobs get a dropdown per pane (a checkbox per option and pane when
// several can be on), on/off knobs a checkbox per pane. Above the bars, per pane: random / extreme,
// low / high in extreme, a new-draw (dice) button and the draw number; and "copy A → B".
//
//   new Lab.KnobPanel(el, schemaEntry, state, {onChange(which), isShare, stops(entryId, key),
//                     rerolls, setTimeout, clearTimeout})
//     .render()                 redraw from state (after a load)
//     .paneResult(which, meta, baseN)  a pane arrived: notes a draw that changed nothing
//     .destroy()
//   Lab.knobMath: the pure maths, Node-tested (bands, snap, range markers, the share rule, the
//   request scheduler).
//
// State is app.state: state.panes.A|B.perEntry[entryId] = {values, mode, edge, reroll}; values
// holds only knobs that differ from their default. Live mode: onChange fires 150 ms after the last
// move while dragging and at once on release. Share mode: markers snap to the precomputed stops,
// knobs without stops are "live only", and only one knob per pane is off its default.
// Classic script: adds to window.Lab and module.exports.
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  // ---- pure maths -----------------------------------------------------------------------
  function decimals(step) {
    var s = String(step);
    if (/e-/.test(s)) return parseInt(s.split('e-')[1], 10);
    return s.indexOf('.') < 0 ? 0 : s.length - s.indexOf('.') - 1;
  }
  function roundTo(v, step) {
    if (!step) return v;
    return Number((Math.round(v / step) * step).toFixed(decimals(step)));
  }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  function same(a, b) {
    if (Array.isArray(a) || Array.isArray(b)) {
      return Array.isArray(a) && Array.isArray(b) && a.length === b.length &&
        a.every(function (x, i) { return same(x, b[i]); });
    }
    if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
    return a === b;
  }

  // Bar segments as fractions of the slider limits: {segments: [{kind, from, to}], noProven}.
  function bands(knob) {
    if ((knob.kind !== 'scalar' && knob.kind !== 'range') || !knob.limits) return { segments: [], noProven: false };
    var L0 = knob.limits[0], L1 = knob.limits[1], span = L1 - L0 || 1;
    var p = knob.proven, d = knob.default, used;
    if (knob.kind === 'scalar') used = p ? [Math.min(p[0], d), Math.max(p[1], d)] : [L0, d];
    else used = p ? [Math.min(p[0], d[0]), Math.max(p[1], d[1])] : [d[0], d[1]];
    var cuts = [L0, L1].concat(p || [], used).map(function (x) { return clamp(x, L0, L1); });
    cuts = cuts.sort(function (a, b) { return a - b; }).filter(function (x, i, a) { return i === 0 || x > a[i - 1]; });
    var segs = [];
    for (var i = 0; i + 1 < cuts.length; i++) {
      var mid = (cuts[i] + cuts[i + 1]) / 2;
      var kind = p && mid >= p[0] && mid <= p[1] ? 'proven' : mid >= used[0] && mid <= used[1] ? 'used' : 'beyond';
      var last = segs[segs.length - 1];
      if (last && last.kind === kind) last.to = (cuts[i + 1] - L0) / span;
      else segs.push({ kind: kind, from: (cuts[i] - L0) / span, to: (cuts[i + 1] - L0) / span });
    }
    return { segments: segs, noProven: !p };
  }

  // Nearest stop; ranges ([lo, hi]) by the sum of end distances, a range against number stops by
  // its high end.
  function snap(value, stops) {
    if (!stops || !stops.length) return value;
    var dist = function (s) {
      if (Array.isArray(s) && Array.isArray(value)) return Math.abs(s[0] - value[0]) + Math.abs(s[1] - value[1]);
      var v = Array.isArray(value) ? value[1] : value;
      return Math.abs((Array.isArray(s) ? s[1] : s) - v);
    };
    var best = stops[0];
    stops.forEach(function (s) { if (dist(s) < dist(best)) best = s; });
    return Array.isArray(best) ? best.slice() : best;
  }

  // A range knob is one marker: its high end. The low end follows the profile's range: fixed when
  // it sits at the slider limit (p: [0, 0.2]), mirrored about the geometric centre for factors
  // around 1 (duration [0.6, 1.65]: lo = c² / hi), about the middle otherwise (fps [12, 25]), and
  // equal to the high end for a single value ([0.05, 0.05]). The low end stops at the limit.
  function rangeModel(knob) {
    var d = knob.default, L0 = knob.limits[0];
    if (same(d[0], d[1])) return { type: 'single' };
    if (d[0] <= L0) return { type: 'fixed', lo: d[0] };
    if (d[0] > 0 && d[0] < 1 && d[1] > 1) return { type: 'log', c: Math.sqrt(d[0] * d[1]), c2: d[0] * d[1] };
    return { type: 'linear', c: (d[0] + d[1]) / 2 };
  }
  function mirror(model, x) { return model.type === 'log' ? model.c2 / x : 2 * model.c - x; }

  function rangeFromMarker(knob, v) {
    var L0 = knob.limits[0], L1 = knob.limits[1], step = knob.step;
    v = clamp(roundTo(v, step), L0, L1);
    var m = rangeModel(knob);
    if (m.type === 'single') return [v, v];
    if (m.type === 'fixed') return [m.lo, Math.max(v, m.lo)];
    if (v >= m.c) return [Math.max(L0, roundTo(mirror(m, v), step)), v];
    return [v, Math.min(L1, roundTo(mirror(m, v), step))];
  }

  function markerOf(knob, value) {
    if (knob.kind === 'range') return Array.isArray(value) ? value[1] : value;
    return value;
  }

  // The value at a fraction along the bar.
  function valueAt(knob, f) {
    var L0 = knob.limits[0], L1 = knob.limits[1];
    var v = L0 + clamp(f, 0, 1) * (L1 - L0);
    if (knob.kind === 'range') return rangeFromMarker(knob, v);
    v = clamp(roundTo(v, knob.step), L0, L1);
    return knob.integer ? Math.round(v) : v;
  }

  function frac(knob, v) {
    var L0 = knob.limits[0], L1 = knob.limits[1];
    return clamp((v - L0) / ((L1 - L0) || 1), 0, 1);
  }

  // Live mode: every knob kept; a value equal to its default is dropped.
  function setValue(values, key, value, defaults) {
    var out = Object.assign({}, values);
    if (key in defaults && same(value, defaults[key])) delete out[key];
    else out[key] = value;
    return out;
  }

  // Share mode: one knob off its default per pane; moving another resets the first.
  function shareSet(values, key, value, defaults) {
    var reset = Object.keys(values || {}).filter(function (k) { return k !== key && !same(values[k], defaults[k]); });
    var out = {};
    if (!same(value, defaults[key])) out[key] = value;
    return { values: out, reset: reset };
  }

  // Per-pane request timing: drag() fires `delay` ms after the last call, release() at once; a
  // pane has at most one pending timer.
  function scheduler(fire, delay, clock) {
    clock = clock || {
      set: function (fn, ms) { return setTimeout(fn, ms); },
      clear: function (id) { clearTimeout(id); },
    };
    var pending = {};
    var cancel = function (w) { if (pending[w] != null) clock.clear(pending[w]); pending[w] = null; };
    return {
      drag: function (w) {
        cancel(w);
        pending[w] = clock.set(function () { pending[w] = null; fire(w); }, delay);
      },
      release: function (w) { cancel(w); fire(w); },
      cancel: cancel,
    };
  }

  Lab.knobMath = {
    roundTo: roundTo, clamp: clamp, same: same, bands: bands, snap: snap, rangeModel: rangeModel,
    rangeFromMarker: rangeFromMarker, markerOf: markerOf, valueAt: valueAt, frac: frac,
    setValue: setValue, shareSet: shareSet, scheduler: scheduler,
  };

  // ---- the panel ------------------------------------------------------------------------
  var PANES = ['A', 'B'];
  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  function fmtNum(v) {
    if (typeof v !== 'number') return String(v);
    return String(Number(v.toFixed(4)));
  }
  function unitText(knob) {
    var u = knob.unit || '';
    return u && !/^[°×]/.test(u) ? ' ' + u : u;
  }
  function fmtValue(knob, v) {
    if (v === undefined || v === null) return '—';
    if (knob.kind === 'bool') return v ? 'on' : 'off';
    if (Array.isArray(v) && knob.kind === 'range') return fmtNum(v[0]) + '–' + fmtNum(v[1]) + unitText(knob);
    if (Array.isArray(v)) return v.length ? v.join(', ') : 'none';
    return fmtNum(v) + (knob.kind === 'scalar' ? unitText(knob) : '');
  }

  function KnobPanel(el, entry, state, opts) {
    this.el = el;
    this.entry = entry;
    this.state = state;
    this.opts = opts || {};
    this.isShare = !!this.opts.isShare;
    this.notes = { A: '', B: '' };
    this.message = '';
    var self = this;
    this.sched = scheduler(function (w) { if (self.opts.onChange) self.opts.onChange(w); }, 150,
      this.opts.clock);
    this.defaults = {};
    (entry.knobs || []).forEach(function (k) { self.defaults[k.key] = k.default; });
    this.render();
  }

  KnobPanel.prototype.settings = function (which) {
    var per = this.state.panes[which].perEntry;
    if (!per[this.entry.id]) per[this.entry.id] = { values: {}, mode: 'random', edge: 'high', reroll: 0 };
    return per[this.entry.id];
  };

  KnobPanel.prototype.value = function (which, knob) {
    var vals = this.settings(which).values || {};
    return knob.key in vals ? vals[knob.key] : knob.default;
  };

  KnobPanel.prototype.stops = function (knob) {
    if (!this.isShare || !this.opts.stops) return null;
    return this.opts.stops(this.entry.id, knob.key);
  };

  // Why a knob can't be moved here, or null.
  KnobPanel.prototype.lockReason = function (knob) {
    if (knob.active === false) return 'not used by this variant';
    if (this.isShare && (!knob.share || !this.stops(knob))) return 'live only';
    return null;
  };

  KnobPanel.prototype.setKnob = function (which, knob, value) {
    var s = this.settings(which);
    if (this.isShare) {
      var stop = snap(value, this.stops(knob));
      if (knob.kind === 'range' && typeof stop === 'number') stop = rangeFromMarker(knob, stop);
      value = stop;
      var r = shareSet(s.values, knob.key, value, this.defaults);
      if (r.reset.length) {
        var names = r.reset.map(function (k) {
          var kn = this.entry.knobs.filter(function (x) { return x.key === k; })[0];
          return kn ? kn.technical : k;
        }, this);
        this.message = 'One knob at a time in the shared version: pane ' + which + '\'s ' +
          names.join(', ') + ' went back to the default.';
      }
      var keys = Object.keys(r.values).concat(Object.keys(s.values || {}));
      var changed = s.reroll !== 0 || keys.some(function (k) { return !same(r.values[k], (s.values || {})[k]); });
      if (changed) s.reroll = 0; // the shared version has new draws only at the default settings
      s.values = r.values;
      return changed;
    }
    var next = setValue(s.values, knob.key, value, this.defaults);
    var moved = !same(this.value(which, knob), value);
    s.values = next;
    return moved;
  };

  KnobPanel.prototype.render = function () {
    var self = this, entry = this.entry;
    var html = '<div class="kp">';
    html += '<header class="kp-head"><h2>Knobs</h2><span class="tech">' + esc(entry.id) + '</span></header>';
    html += this.modesHtml();
    html += '<p class="kp-message" role="status"' + (this.message ? '' : ' hidden') + '>' + esc(this.message) + '</p>';
    if (entry.whole) {
      html += '<p class="kp-viewonly">A whole profile is view only: it runs every augmentation the profile ' +
        'turns on, one sample per pane. Knobs are set one augmentation at a time (pick one on the left).</p>';
    } else if (!(entry.knobs || []).length) {
      html += '<p class="kp-viewonly">This augmentation has no knobs.</p>';
    } else {
      html += '<p class="kp-legend"><i class="sw proven"></i>proven in the literature ' +
        '<i class="sw used"></i>past proven, up to what the profile uses <i class="sw beyond"></i>beyond, up to the slider limit</p>';
      html += '<div class="kp-knobs">' + entry.knobs.map(function (k, i) { return self.knobHtml(k, i); }).join('') + '</div>';
    }
    html += '</div>';
    this.el.innerHTML = html;
    this.bind();
  };

  KnobPanel.prototype.modesHtml = function () {
    var self = this, entry = this.entry;
    var rows = PANES.map(function (w) {
      var s = self.settings(w), lw = w.toLowerCase();
      var ext = s.mode === 'extreme';
      var draw = self.isShare
        ? '<span class="kp-draw mono" title="which precomputed random draw (0 = the profile seed)">draw ' + s.reroll + '</span>'
        : '<label class="kp-draw" title="which random draw (0 = the profile seed); the seed is the profile seed + this number">draw ' +
          '<input type="number" min="0" step="1" value="' + s.reroll + '" data-act="reroll" data-pane="' + w + '"' +
          (ext ? ' disabled' : '') + '></label>';
      return '<div class="kp-pane kp-' + lw + '">' +
        '<span class="badge">' + w + '</span>' +
        '<span class="seg" role="group" aria-label="draw mode for pane ' + w + '">' +
        '<button type="button" data-act="mode" data-mode="random" data-pane="' + w + '" class="' + (ext ? '' : 'on') + '" ' +
        'title="one seeded random draw at these settings">random</button>' +
        '<button type="button" data-act="mode" data-mode="extreme" data-pane="' + w + '" class="' + (ext ? 'on' : '') + '"' +
        (entry.extreme ? '' : ' disabled') + ' title="every sampled value pinned to the edge of its range">extreme</button></span>' +
        '<span class="seg"' + (ext ? '' : ' hidden') + ' role="group" aria-label="which edge for pane ' + w + '">' +
        '<button type="button" data-act="edge" data-edge="low" data-pane="' + w + '" class="' + (s.edge === 'low' ? 'on' : '') + '">low</button>' +
        '<button type="button" data-act="edge" data-edge="high" data-pane="' + w + '" class="' + (s.edge === 'high' ? 'on' : '') + '">high</button></span>' +
        '<button type="button" class="dice" data-act="dice" data-pane="' + w + '"' + (ext ? ' disabled' : '') +
        ' title="a new random draw for pane ' + w + '">new draw</button>' + draw +
        '<span class="kp-pane-note small"' + (self.notes[w] ? '' : ' hidden') + '>' + esc(self.notes[w]) + '</span>' +
        '</div>';
    }).join('');
    var foot = '<div class="kp-modes-foot">' +
      (entry.extreme ? '' : '<span class="small">no extreme for this one</span>') +
      '<button type="button" data-act="copy" title="give pane B the same settings and draw as pane A">copy A → B</button></div>';
    return '<div class="kp-modes">' + rows + foot + '</div>';
  };

  KnobPanel.prototype.knobHtml = function (knob, i) {
    var self = this;
    var lock = this.lockReason(knob);
    var cls = 'knob' + (lock ? ' locked' : '') + (knob.active === false ? ' inactive' : '');
    var dis = lock ? ' disabled' : '';
    var label = '<div class="knob-label"><span class="plain">' + esc(knob.plain) + '</span> ' +
      '<span class="tech">' + esc(knob.technical) + (knob.unit ? ', ' + esc(knob.unit) : '') + '</span>' +
      (lock ? ' <span class="knob-lock">' + esc(lock) + '</span>' : '') + '</div>';
    var body = '';
    if (knob.kind === 'scalar' || knob.kind === 'range') {
      var b = bands(knob);
      var segs = b.segments.map(function (s) {
        return '<i class="band ' + s.kind + '" style="left:' + (100 * s.from) + '%;width:' + (100 * (s.to - s.from)) + '%"></i>';
      }).join('');
      var marks = PANES.map(function (w) {
        var v = self.value(w, knob), lw = w.toLowerCase();
        var hi = frac(knob, markerOf(knob, v));
        var out = '';
        if (knob.kind === 'range') {
          var lo = frac(knob, v[0]);
          out += '<i class="span span-' + lw + '" style="left:' + (100 * lo) + '%;width:' + (100 * (hi - lo)) + '%"></i>' +
            '<i class="lo lo-' + lw + '" style="left:' + (100 * lo) + '%"></i>';
        }
        out += '<span class="mk mk-' + lw + '" style="left:' + (100 * hi) + '%" role="slider" tabindex="' + (lock ? -1 : 0) + '" ' +
          'aria-label="pane ' + w + ': ' + esc(knob.technical) + '" aria-valuetext="' + esc(fmtValue(knob, v)) + '" ' +
          'data-pane="' + w + '">' + w + '</span>';
        return out;
      }).join('');
      var def = '<i class="def" style="left:' + (100 * frac(knob, markerOf(knob, knob.default))) + '%" title="profile value"></i>';
      body += '<div class="kbar" data-knob="' + i + '"><div class="track">' + segs + def + '</div>' + marks + '</div>';
      var used = 'profile uses ' + fmtValue(knob, knob.default);
      var ranges = b.noProven ? '<b>no proven range</b> · ' + esc(used)
        : esc('proven ' + fmtNum(knob.proven[0]) + '–' + fmtNum(knob.proven[1]) + unitText(knob) + ' · ' + used);
      body += '<div class="kvals"><span class="tag-a">A ' + esc(fmtValue(knob, self.value('A', knob))) + '</span>' +
        '<span class="tag-b">B ' + esc(fmtValue(knob, self.value('B', knob))) + '</span>' +
        '<span class="small">' + ranges + '</span></div>';
    } else if (knob.kind === 'enum' && !knob.multi) {
      body += '<div class="kchoice">' + PANES.map(function (w) {
        var v = self.value(w, knob);
        return '<label class="tag-' + w.toLowerCase() + '">' + w + ' <select data-knob="' + i + '" data-pane="' + w + '"' + dis + '>' +
          knob.options.map(function (o) {
            return '<option value="' + esc(o) + '"' + (o === v ? ' selected' : '') + '>' + esc(o) + (o === knob.default ? ' (profile)' : '') + '</option>';
          }).join('') + '</select></label>';
      }).join('') + '</div>';
    } else if (knob.kind === 'enum') {
      body += '<table class="kmulti"><tr><th></th><th class="tag-a">A</th><th class="tag-b">B</th></tr>' +
        knob.options.map(function (o) {
          return '<tr><td>' + esc(o) + '</td>' + PANES.map(function (w) {
            var on = (self.value(w, knob) || []).indexOf(o) >= 0;
            return '<td><input type="checkbox" data-knob="' + i + '" data-pane="' + w + '" data-opt="' + esc(o) + '"' +
              (on ? ' checked' : '') + dis + ' aria-label="pane ' + w + ': ' + esc(o) + '"></td>';
          }).join('') + '</tr>';
        }).join('') + '</table><div class="small">profile: ' + esc(fmtValue(knob, knob.default)) + '</div>';
    } else if (knob.kind === 'bool') {
      body += '<div class="kchoice">' + PANES.map(function (w) {
        return '<label class="tag-' + w.toLowerCase() + '"><input type="checkbox" data-knob="' + i + '" data-pane="' + w + '"' +
          (self.value(w, knob) ? ' checked' : '') + dis + '> ' + w + '</label>';
      }).join('') + '<span class="small">profile: ' + (knob.default ? 'on' : 'off') + '</span></div>';
    }
    return '<div class="' + cls + '">' + label + body + '</div>';
  };

  // Redraw after a change; keeps keyboard focus on the marker that had it.
  KnobPanel.prototype.refresh = function () {
    var focus = typeof document !== 'undefined' && document.activeElement;
    var key = null;
    if (focus && this.el.contains(focus) && focus.classList.contains('mk')) {
      key = focus.closest('.kbar').dataset.knob + '|' + focus.dataset.pane;
    }
    this.render();
    if (key) {
      var p = key.split('|');
      var mk = this.el.querySelector('.kbar[data-knob="' + p[0] + '"] .mk[data-pane="' + p[1] + '"]');
      if (mk) mk.focus();
    }
  };

  KnobPanel.prototype.bind = function () {
    var self = this, el = this.el, knobs = this.entry.knobs || [];
    Array.prototype.forEach.call(el.querySelectorAll('[data-act]'), function (b) {
      var w = b.dataset.pane;
      var ev = b.dataset.act === 'reroll' ? 'change' : 'click';
      b.addEventListener(ev, function () { self.action(b.dataset.act, w, b); });
    });
    Array.prototype.forEach.call(el.querySelectorAll('.kbar'), function (bar) {
      self.bindBar(bar, knobs[+bar.dataset.knob]);
    });
    Array.prototype.forEach.call(el.querySelectorAll('select[data-knob], input[type=checkbox][data-knob]'), function (box) {
      box.addEventListener('change', function () {
        var knob = knobs[+box.dataset.knob], w = box.dataset.pane, v;
        if (knob.kind === 'bool') v = box.checked;
        else if (knob.multi) {
          var cur = (self.value(w, knob) || []).slice();
          var o = box.dataset.opt;
          v = knob.options.filter(function (x) { return x === o ? box.checked : cur.indexOf(x) >= 0; });
        } else v = box.value;
        if (self.setKnob(w, knob, v)) self.sched.release(w);
        self.refresh();
      });
    });
  };

  KnobPanel.prototype.action = function (act, w, b) {
    var s = w ? this.settings(w) : null;
    this.message = '';
    if (act === 'mode') {
      if (s.mode === b.dataset.mode) return;
      s.mode = b.dataset.mode;
      if (this.isShare && s.mode === 'extreme') s.reroll = 0; // extremes are precomputed at draw 0
    } else if (act === 'edge') {
      if (s.edge === b.dataset.edge) return;
      s.edge = b.dataset.edge;
    } else if (act === 'dice') {
      if (this.isShare) {
        var R = Math.max(0, this.opts.rerolls || 0);
        if (Object.keys(s.values).length) {
          s.values = {};
          this.message = 'New draws exist only at the profile\'s settings in the shared version: pane ' + w +
            '\'s knobs went back to the default.';
        }
        s.reroll = R ? (s.reroll + 1) % (R + 1) : 0;
      } else {
        s.reroll += 1;
      }
    } else if (act === 'reroll') {
      var k = Math.max(0, Math.floor(Number(b.value) || 0));
      if (k === s.reroll) return;
      s.reroll = k;
    } else if (act === 'copy') {
      var a = this.settings('A');
      this.state.panes.B.perEntry[this.entry.id] = JSON.parse(JSON.stringify(a));
      w = 'B';
    }
    this.notes[w] = '';
    this.sched.release(w);
    this.refresh();
  };

  KnobPanel.prototype.bindBar = function (bar, knob) {
    var self = this;
    if (this.lockReason(knob)) return;
    var drag = null;
    var fracOf = function (ev) {
      var r = bar.querySelector('.track').getBoundingClientRect();
      return (ev.clientX - r.left) / (r.width || 1);
    };
    // the pane whose marker (or range end) is nearest the pointer
    var nearest = function (f) {
      var best = null, bestD = Infinity;
      PANES.forEach(function (w) {
        var v = self.value(w, knob);
        var pts = knob.kind === 'range' ? [frac(knob, v[0]), frac(knob, v[1])] : [frac(knob, v)];
        pts.forEach(function (x) { var d = Math.abs(x - f); if (d < bestD - 1e-9) { bestD = d; best = w; } });
      });
      return best;
    };
    var move = function (ev) {
      if (!drag) return;
      if (self.setKnob(drag.w, knob, valueAt(knob, fracOf(ev)))) { drag.changed = true; self.sched.drag(drag.w); }
      self.updateBar(bar, knob);
    };
    bar.addEventListener('pointerdown', function (ev) {
      if (ev.button !== 0) return;
      var mk = ev.target.closest('.mk');
      drag = { w: mk ? mk.dataset.pane : nearest(fracOf(ev)), changed: false };
      try { bar.setPointerCapture(ev.pointerId); } catch (e) { /* synthetic events */ }
      self.message = '';
      move(ev);
      ev.preventDefault();
    });
    bar.addEventListener('pointermove', move);
    var end = function () {
      if (!drag) return;
      var d = drag;
      drag = null;
      if (d.changed) { self.notes[d.w] = ''; self.sched.release(d.w); }
      self.refresh();
    };
    bar.addEventListener('pointerup', end);
    bar.addEventListener('pointercancel', end);
    bar.addEventListener('lostpointercapture', end);
    Array.prototype.forEach.call(bar.querySelectorAll('.mk'), function (mk) {
      mk.addEventListener('keydown', function (ev) {
        var dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[ev.key];
        if (!dir) return;
        ev.preventDefault();
        var w = mk.dataset.pane, v = markerOf(knob, self.value(w, knob)), next;
        var stops = self.stops(knob);
        if (stops) {
          var marks = stops.map(function (s) { return markerOf(knob, s); });
          var cands = marks.map(function (m, i) { return { m: m, s: stops[i] }; })
            .filter(function (c) { return dir > 0 ? c.m > v + 1e-9 : c.m < v - 1e-9; })
            .sort(function (a, b) { return dir > 0 ? a.m - b.m : b.m - a.m; });
          if (!cands.length) return;
          next = typeof cands[0].s === 'number' && knob.kind === 'range' ? rangeFromMarker(knob, cands[0].s) : cands[0].s;
        } else {
          next = knob.kind === 'range' ? rangeFromMarker(knob, v + dir * knob.step) : valueAt(knob, frac(knob, v + dir * knob.step));
        }
        if (self.setKnob(w, knob, next)) self.sched.drag(w);
        self.updateBar(bar, knob);
      });
      mk.addEventListener('keyup', function (ev) {
        if (!/^Arrow/.test(ev.key)) return;
        self.sched.release(mk.dataset.pane);
        self.refresh();
      });
    });
  };

  // Moves the markers of one bar in place while dragging (a full redraw would drop the capture).
  KnobPanel.prototype.updateBar = function (bar, knob) {
    var self = this;
    PANES.forEach(function (w) {
      var v = self.value(w, knob), lw = w.toLowerCase();
      var hi = frac(knob, markerOf(knob, v));
      var mk = bar.querySelector('.mk-' + lw);
      if (mk) { mk.style.left = (100 * hi) + '%'; mk.setAttribute('aria-valuetext', fmtValue(knob, v)); }
      if (knob.kind === 'range') {
        var lo = frac(knob, v[0]);
        var span = bar.querySelector('.span-' + lw), tick = bar.querySelector('.lo-' + lw);
        if (span) { span.style.left = (100 * lo) + '%'; span.style.width = (100 * (hi - lo)) + '%'; }
        if (tick) tick.style.left = (100 * lo) + '%';
      }
      var tag = bar.parentNode.querySelector('.kvals .tag-' + lw);
      if (tag) tag.textContent = w + ' ' + fmtValue(knob, v);
    });
    var msg = this.el.querySelector('.kp-message');
    if (msg) { msg.textContent = this.message; msg.hidden = !this.message; }
  };

  // A pane arrived: say so when its draw changed nothing (e.g. random frame drop's high edge pins
  // p to 0). baseN is the original clip's frame count.
  KnobPanel.prototype.paneResult = function (which, meta, baseN) {
    var note = '';
    if (!this.entry.whole && meta && Lab.unchangedReason) note = Lab.unchangedReason(meta, baseN);
    else if (!this.entry.whole && meta && Array.isArray(meta.applied) && !meta.applied.length) {
      note = 'This draw left the clip unchanged.';
    }
    this.notes[which] = note;
    var el = this.el.querySelector('.kp-' + which.toLowerCase() + ' .kp-pane-note');
    if (el) { el.textContent = note; el.hidden = !note; }
  };

  KnobPanel.prototype.destroy = function () {
    this.sched.cancel('A');
    this.sched.cancel('B');
    this.el.innerHTML = '';
  };

  Lab.KnobPanel = KnobPanel;
})();

if (typeof module !== 'undefined') module.exports = Lab;
