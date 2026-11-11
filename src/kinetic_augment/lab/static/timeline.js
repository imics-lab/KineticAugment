// One timeline for both panes: play/pause, step, scrub, speed, frame label and the clock.
// Clock "playback": frame t of each pane plays at the clip's frame rate, so a time warp runs ahead
// of or behind the original and the shorter pane holds its last frame, marked "ended".
// Clock "source": each pane shows its frame nearest the current source frame (its `times`).
// Classic script: adds Lab.Timeline, Lab.frameMap, Lab.nearestFrames, Lab.playFrame (and module.exports).
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  var CLOCKS = ['playback', 'source'];

  // For each base frame, the pane frame whose source time is nearest to it.
  Lab.nearestFrames = function nearestFrames(paneTimes, baseTimes) {
    var out = new Int32Array(baseTimes.length);
    for (var f = 0; f < baseTimes.length; f++) {
      var best = 0, bestD = Infinity;
      for (var k = 0; k < paneTimes.length; k++) {
        var d = Math.abs(paneTimes[k] - baseTimes[f]);
        if (d < bestD) { bestD = d; best = k; }
      }
      out[f] = best;
    }
    return out;
  };

  // Playback by wall-clock time: the frame `elapsed` seconds after `start`, wrapping at `length`.
  Lab.playFrame = function playFrame(start, elapsed, fps, speed, length) {
    if (!length) return 0;
    return (start + Math.floor(elapsed * fps * speed + 1e-9)) % length;
  };

  // Frames on the clock: the base length in source time, the longest stream in playback.
  Lab.clockLength = function clockLength(clock, clip) {
    if (!clip || !clip.nBase) return 1;
    if (clock === 'source') return clip.nBase;
    var n = clip.nBase;
    ['A', 'B'].forEach(function (w) { if (clip[w]) n = Math.max(n, clip[w].n); });
    return n;
  };

  // Where each stream is at clock position t: {base, A: {k, ended} | null, B: ...}.
  // clip = {nBase, A: {n, nearest} | null, B: {n, nearest} | null}.
  Lab.frameMap = function frameMap(clock, t, clip) {
    var out = { t: t, base: Math.min(t, clip.nBase - 1), A: null, B: null };
    ['A', 'B'].forEach(function (w) {
      var p = clip[w];
      if (!p) return;
      if (clock === 'source') out[w] = { k: p.nearest ? p.nearest[out.base] : Math.min(t, p.n - 1), ended: false };
      else out[w] = { k: Math.min(t, p.n - 1), ended: t >= p.n };
    });
    return out;
  };

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // Lab.Timeline(el, {fps, onFrame, clock}); onFrame(frameMap result) on every change.
  function Timeline(root, opts) {
    opts = opts || {};
    this.root = root;
    this.fps = opts.fps || 30;
    this.onFrame = opts.onFrame || function () {};
    this.clock = CLOCKS.indexOf(opts.clock) >= 0 ? opts.clock : 'playback';
    this.speed = 1;
    this.t = 0;
    this.playing = false;
    this.clip = { nBase: 0, baseTimes: null, A: null, B: null };
    this._build();
  }

  Timeline.prototype._build = function () {
    var self = this;
    this.root.classList.add('timeline');
    this.root.innerHTML = '';
    var prev = el('button', { type: 'button', title: 'previous frame (←)', 'aria-label': 'previous frame' }, '|◀');
    var play = el('button', { type: 'button', class: 'play', title: 'play / pause (space)', 'aria-label': 'play' }, '▶');
    var next = el('button', { type: 'button', title: 'next frame (→)', 'aria-label': 'next frame' }, '▶|');
    var scrub = el('input', { type: 'range', min: '0', max: '0', step: '1', value: '0', 'aria-label': 'frame' });
    var label = el('span', { class: 'mono frame-label' }, '');
    var clock = el('select', { title: 'playback: each pane plays its own frames; source time: each pane shows its frame nearest the source frame' });
    clock.add(new Option('clock: playback', 'playback'));
    clock.add(new Option('clock: source time', 'source'));
    clock.value = this.clock;
    var speed = el('select', { title: 'playback speed' });
    [['0.25', '¼×'], ['0.5', '½×'], ['1', '1×']].forEach(function (s) { speed.add(new Option(s[1], s[0])); });
    speed.value = '1';
    [prev, play, next, scrub, label, clock, speed].forEach(function (e) { self.root.appendChild(e); });
    this.els = { prev: prev, play: play, next: next, scrub: scrub, label: label, clock: clock, speed: speed };

    play.onclick = function () { self.toggle(); };
    prev.onclick = function () { self.pause(); self.show(self.t - 1); };
    next.onclick = function () { self.pause(); self.show(self.t + 1); };
    scrub.oninput = function () { self.show(Number(scrub.value)); self._anchor(); };
    clock.onchange = function () { self.setClock(clock.value); };
    speed.onchange = function () { self.speed = Number(speed.value); self._anchor(); };
    window.addEventListener('keydown', function (e) {
      var tag = e.target && e.target.tagName;
      if (tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range')) return;
      if (e.target && e.target.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return; // leave browser shortcuts (Cmd+←) alone
      if (e.key === ' ' && tag !== 'BUTTON') { e.preventDefault(); self.toggle(); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); self.pause(); self.show(self.t - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); self.pause(); self.show(self.t + 1); }
    });
  };

  Timeline.prototype.length = function () { return Lab.clockLength(this.clock, this.clip); };

  // A new clip: its frame count, source times and frame rate. Both panes are cleared.
  Timeline.prototype.setClip = function (n, baseTimes, fps) {
    this.clip = { nBase: n, baseTimes: baseTimes, A: null, B: null };
    if (fps) this.fps = fps;
    this._relength();
  };

  // A pane's frames (null clears it): its count and source times.
  Timeline.prototype.setPane = function (which, n, times) {
    if (n == null) this.clip[which] = null;
    else {
      this.clip[which] = {
        n: n,
        nearest: times && this.clip.baseTimes ? Lab.nearestFrames(times, this.clip.baseTimes) : null,
      };
    }
    this._relength();
  };

  Timeline.prototype.setClock = function (clock) {
    if (CLOCKS.indexOf(clock) < 0) return;
    this.clock = clock;
    this.els.clock.value = clock;
    this._relength();
  };

  Timeline.prototype._relength = function () {
    this.els.scrub.max = String(Math.max(0, this.length() - 1));
    this.show(this.t);
  };

  Timeline.prototype.frame = function () { return Lab.frameMap(this.clock, this.t, this.clip); };

  Timeline.prototype.show = function (t) {
    var len = this.length();
    this.t = Math.min(Math.max(Math.round(t) || 0, 0), len - 1);
    this.els.scrub.value = String(this.t);
    var n = this.clip.nBase;
    this.els.label.textContent = !n ? '' : this.clock === 'source'
      ? 'source frame ' + (this.t + 1) + ' / ' + n
      : 'frame ' + (this.t + 1) + ' / ' + len + (len !== n ? ' (original ' + n + ')' : '');
    if (n) this.onFrame(this.frame());
  };

  Timeline.prototype.play = function () {
    if (this.playing || !this.clip.nBase) return;
    var self = this;
    this.playing = true;
    this.els.play.textContent = '❚❚';
    this.els.play.setAttribute('aria-label', 'pause');
    if (this.t >= this.length() - 1) this.show(0);
    this._anchor();
    // A token per loop: a pause then a quick play must not leave two loops running.
    var token = this._loop = {};
    var loop = function (now) {
      if (!self.playing || self._loop !== token) return;
      var elapsed = Math.max(0, now - self._start.time) / 1000;
      var t = Lab.playFrame(self._start.t, elapsed, self.fps, self.speed, self.length());
      if (t !== self.t) self.show(t);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  };

  // Restart the wall clock from the current frame (on play, scrub and speed change).
  Timeline.prototype._anchor = function () {
    this._start = { t: this.t, time: performance.now() };
  };

  Timeline.prototype.pause = function () {
    this.playing = false;
    this._loop = null;
    this.els.play.textContent = '▶';
    this.els.play.setAttribute('aria-label', 'play');
  };

  Timeline.prototype.toggle = function () { if (this.playing) this.pause(); else this.play(); };

  Lab.Timeline = Timeline;
})();

if (typeof module !== 'undefined') module.exports = Lab;
