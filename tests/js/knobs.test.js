// The knob panel's pure maths (Lab.knobMath): bar bands, snapping to share stops, the range-knob
// marker, the share-mode "one knob at a time" rule and the pane request scheduler.
const test = require('node:test');
const assert = require('node:assert/strict');

const Lab = require('../../src/kinetic_augment/lab/static/knobs.js');
const K = Lab.knobMath;

const near = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, (msg || '') + ` ${a} != ${b}`);
const nearBands = (got, want) => {
  assert.equal(got.length, want.length, JSON.stringify(got));
  want.forEach((w, i) => {
    assert.equal(got[i].kind, w[0], JSON.stringify(got));
    near(got[i].from, w[1], 'from');
    near(got[i].to, w[2], 'to');
  });
};

const ELBOW_TRUNC = { key: 'elbow_truncate_deg', kind: 'scalar', proven: [0, 7.5], limits: [0, 15], step: 0.25, default: 9.5 };
const SIGMA = { key: 'sigma_deg', kind: 'scalar', proven: null, limits: [0, 10], step: 0.25, default: 2.5 };
const DURATION = { key: 'duration', kind: 'range', proven: [0.67, 1.5], limits: [0.4, 2.5], step: 0.01, default: [0.6, 1.65] };
const P = { key: 'p', kind: 'range', proven: null, limits: [0, 0.5], step: 0.01, default: [0, 0.2] };
const FPS = { key: 'target_fps', kind: 'range', proven: [10, 30], limits: [8, 50], step: 1, default: [12, 25] };
const START = { key: 'start_rate', kind: 'range', proven: null, limits: [0, 0.1], step: 0.005, default: [0.05, 0.05] };
const YAW = { key: 'yaw_deg', kind: 'scalar', proven: [0, 20], limits: [0, 50], step: 0.5, default: 25 };

test('bands: green proven, amber past proven up to the used value, grey beyond (fractions)', () => {
  nearBands(K.bands(ELBOW_TRUNC).segments, [['proven', 0, 0.5], ['used', 0.5, 9.5 / 15], ['beyond', 9.5 / 15, 1]]);
  assert.equal(K.bands(ELBOW_TRUNC).noProven, false);
  nearBands(K.bands(YAW).segments, [['proven', 0, 0.4], ['used', 0.4, 0.5], ['beyond', 0.5, 1]]);
  // used inside proven: no amber
  nearBands(K.bands(Object.assign({}, YAW, { default: 10 })).segments, [['proven', 0, 0.4], ['beyond', 0.4, 1]]);
});

test('bands: no proven range shows the used range in amber and says so', () => {
  const b = K.bands(SIGMA);
  assert.equal(b.noProven, true);
  nearBands(b.segments, [['used', 0, 0.25], ['beyond', 0.25, 1]]);
  nearBands(K.bands(P).segments, [['used', 0, 0.4], ['beyond', 0.4, 1]]);
});

test('bands: a range knob colours both sides of its proven and used ranges', () => {
  const w = 2.1;
  nearBands(K.bands(DURATION).segments, [
    ['beyond', 0, 0.2 / w], ['used', 0.2 / w, 0.27 / w], ['proven', 0.27 / w, 1.1 / w],
    ['used', 1.1 / w, 1.25 / w], ['beyond', 1.25 / w, 1],
  ]);
  assert.deepEqual(K.bands({ kind: 'bool' }).segments, []);
});

test('snap: a marker goes to the nearest precomputed stop', () => {
  assert.equal(K.snap(6.1, [0, 2.5, 4.75, 7.5, 10, 15]), 4.75);
  assert.equal(K.snap(6.2, [0, 2.5, 4.75, 7.5, 10, 15]), 7.5);
  assert.equal(K.snap(-3, [0, 2.5]), 0);
  assert.equal(K.snap(99, [0, 2.5]), 2.5);
  // range stops ([lo, hi]) are compared end by end
  assert.deepEqual(K.snap([0.52, 1.9], [[0.6, 1.65], [0.5, 2.0], [0.67, 1.5]]), [0.5, 2.0]);
  assert.equal(K.snap(3, null), 3, 'no stops: unchanged');
});

test('range marker: log-symmetric for factors around 1, fixed low end at the limit, linear otherwise', () => {
  const c2 = 0.6 * 1.65;
  assert.deepEqual(K.rangeFromMarker(DURATION, 1.8), [K.roundTo(c2 / 1.8, 0.01), 1.8]);
  assert.deepEqual(K.rangeFromMarker(DURATION, 1.65), [0.6, 1.65], 'the default range comes back');
  assert.deepEqual(K.rangeFromMarker(DURATION, 2.5), [0.4, 2.5], 'the low end stops at the limit');
  assert.deepEqual(K.rangeFromMarker(P, 0.3), [0, 0.3]);
  assert.deepEqual(K.rangeFromMarker(FPS, 20), [17, 20]);
  assert.deepEqual(K.rangeFromMarker(FPS, 40), [8, 40]);
  assert.deepEqual(K.rangeFromMarker(START, 0.08), [0.08, 0.08], 'a single value stays single');
  assert.equal(K.markerOf(DURATION, [0.6, 1.65]), 1.65);
  // dragging below the centre moves the mirrored (low) end
  assert.deepEqual(K.valueAt(DURATION, (0.8 - 0.4) / 2.1), [0.8, K.roundTo(c2 / 0.8, 0.01)]);
  near(K.valueAt(SIGMA, 0.475), 4.75);
  assert.equal(K.valueAt({ kind: 'scalar', limits: [0, 10], step: 1, integer: true }, 0.33), 3);
});

test('share mode: moving a second knob resets the first to its default', () => {
  const defaults = { sigma_deg: 2.5, elbow_sigma_deg: 4.75 };
  let r = K.shareSet({}, 'elbow_sigma_deg', 10, defaults);
  assert.deepEqual(r, { values: { elbow_sigma_deg: 10 }, reset: [] });
  r = K.shareSet(r.values, 'elbow_sigma_deg', 7.5, defaults);
  assert.deepEqual(r, { values: { elbow_sigma_deg: 7.5 }, reset: [] }, 'the same knob again: no reset');
  r = K.shareSet(r.values, 'sigma_deg', 4, defaults);
  assert.deepEqual(r, { values: { sigma_deg: 4 }, reset: ['elbow_sigma_deg'] });
  r = K.shareSet(r.values, 'sigma_deg', 2.5, defaults);
  assert.deepEqual(r, { values: {}, reset: [] }, 'back at the default: nothing off default');
  // live mode keeps every knob (setValue), dropping only values equal to the default
  assert.deepEqual(K.setValue({ sigma_deg: 4 }, 'elbow_sigma_deg', 10, defaults), { sigma_deg: 4, elbow_sigma_deg: 10 });
  assert.deepEqual(K.setValue({ sigma_deg: 4 }, 'sigma_deg', 2.5, defaults), {});
  assert.ok(K.same([0.6, 1.65], [0.6, 1.65]) && K.same(1, 1.0) && !K.same(1, 2) && !K.same([1, 2], 1));
});

test('scheduler: debounced while dragging, immediate on release, one request per pane', () => {
  const timers = [];
  const clock = {
    set: (fn, ms) => { timers.push({ fn, ms, live: true }); return timers.length - 1; },
    clear: (id) => { if (timers[id]) timers[id].live = false; },
  };
  const fired = [];
  const s = K.scheduler((which) => fired.push(which), 150, clock);
  s.drag('A'); s.drag('A'); s.drag('B');
  assert.deepEqual(fired, []);
  assert.equal(timers.filter((t) => t.live).length, 2, 'one pending timer per pane');
  assert.equal(timers.filter((t) => t.live)[0].ms, 150);
  s.release('A');
  assert.deepEqual(fired, ['A'], 'release fires at once');
  timers.filter((t) => t.live).forEach((t) => { t.live = false; t.fn(); });
  assert.deepEqual(fired, ['A', 'B']);
  timers.forEach((t) => { if (t.live) t.fn(); });
  assert.deepEqual(fired, ['A', 'B'], 'the cancelled A timer never fires');
});

// ---- the panel, rendered into a stub element (no DOM library in Node) ----------------------
const stubEl = () => ({ innerHTML: '', querySelectorAll: () => [], querySelector: () => null, contains: () => false });
const ENTRY = {
  id: 'body.joint_coupled_noise', extreme: true,
  knobs: [
    Object.assign({ plain: 'How far the shoulders turn, typically.', technical: 'shoulder/wrist σ (sigma_deg)', unit: '°', share: true, active: true }, SIGMA),
    Object.assign({ key: 'elbow_sigma_deg', plain: 'How far the elbows bend, typically.', technical: 'elbow σ (elbow_sigma_deg)', unit: '°', share: true, active: true },
      { kind: 'scalar', proven: null, limits: [0, 15], step: 0.25, default: 4.75 }),
    Object.assign({ plain: 'The most an elbow can turn.', technical: 'elbow truncation (elbow_truncate_deg)', unit: '°', share: false, active: true }, ELBOW_TRUNC),
    { key: 'side_mode', kind: 'enum', options: ['mirror', 'independent', 'synchronous'], multi: false, default: 'mirror',
      plain: 'How the two arms relate.', technical: 'side mode (side_mode)', unit: '', share: false, active: true },
    { key: 'bones', kind: 'enum', options: ['upper_arm', 'forearm', 'hand'], multi: true, default: ['upper_arm', 'forearm'],
      plain: 'Which bones.', technical: 'bones (bones)', unit: '', share: false, active: true },
    { key: 'retarget', kind: 'bool', default: true, plain: 'Re-bend the arm.', technical: 'retarget (retarget)', unit: '', share: false, active: true },
    Object.assign({ plain: 'Which knots.', technical: 'knot rate (knot_rate)', unit: '×', share: false, active: false }, DURATION, { key: 'knot_rate' }),
  ],
};
const newState = () => ({ panes: { A: { perEntry: {} }, B: { perEntry: {} } } });

test('panel: plain words first, technical name second; locked knobs say why', () => {
  const el = stubEl();
  const p = new Lab.KnobPanel(el, ENTRY, newState(), { isShare: false });
  assert.match(el.innerHTML, /<span class="plain">How far the elbows bend, typically\.<\/span> <span class="tech">elbow σ \(elbow_sigma_deg\), °<\/span>/);
  assert.match(el.innerHTML, /knot rate \(knot_rate\), ×<\/span> <span class="knob-lock">not used by this variant<\/span>/);
  assert.match(el.innerHTML, /<b>no proven range<\/b> · profile uses 4\.75°/);
  assert.match(el.innerHTML, /proven 0–7\.5° · profile uses 9\.5°/);
  assert.match(el.innerHTML, /class="mk mk-a" style="left:31\.666/, 'A marker at the elbow default 4.75 of 15');
  assert.match(el.innerHTML, /data-mode="extreme" data-pane="A" class=""/, 'extreme is enabled');
  assert.doesNotMatch(el.innerHTML, /no extreme for this one/);
  assert.equal(p.lockReason(ENTRY.knobs[2]), null, 'every active knob is live in full mode');
  const off = stubEl();
  new Lab.KnobPanel(off, Object.assign({}, ENTRY, { extreme: false }), newState(), {});
  assert.match(off.innerHTML, /no extreme for this one/);
  assert.match(off.innerHTML, /data-mode="extreme" data-pane="A" class="" disabled/);
  const whole = stubEl();
  new Lab.KnobPanel(whole, { id: 'combined.default', whole: true, extreme: false, knobs: [] }, newState(), {});
  assert.match(whole.innerHTML, /view only/);
});

test('panel, live: two markers set two panes; dice and copy A → B ask for that pane', () => {
  const state = newState();
  const asked = [];
  const p = new Lab.KnobPanel(stubEl(), ENTRY, state, { onChange: (w) => asked.push(w) });
  assert.equal(p.setKnob('A', ENTRY.knobs[1], 4.75), false, 'the default again is no change');
  assert.equal(p.setKnob('A', ENTRY.knobs[1], 10), true);
  p.setKnob('A', ENTRY.knobs[0], 3);
  p.setKnob('B', ENTRY.knobs[1], 7.5);
  assert.deepEqual(state.panes.A.perEntry[ENTRY.id].values, { elbow_sigma_deg: 10, sigma_deg: 3 });
  assert.deepEqual(state.panes.B.perEntry[ENTRY.id].values, { elbow_sigma_deg: 7.5 });
  p.action('dice', 'A', { dataset: {} });
  p.action('dice', 'A', { dataset: {} });
  assert.equal(state.panes.A.perEntry[ENTRY.id].reroll, 2);
  p.action('copy', null, { dataset: {} });
  assert.deepEqual(state.panes.B.perEntry[ENTRY.id], state.panes.A.perEntry[ENTRY.id]);
  assert.notEqual(state.panes.B.perEntry[ENTRY.id], state.panes.A.perEntry[ENTRY.id], 'a copy, not the same object');
  p.action('mode', 'B', { dataset: { mode: 'extreme' } });
  p.action('edge', 'B', { dataset: { edge: 'low' } });
  assert.deepEqual([state.panes.B.perEntry[ENTRY.id].mode, state.panes.B.perEntry[ENTRY.id].edge], ['extreme', 'low']);
  assert.deepEqual(asked, ['A', 'A', 'B', 'B', 'B']);
});

test('panel, share: snaps to stops, one knob at a time, dice cycles the precomputed draws', () => {
  const state = newState();
  const stops = { sigma_deg: [0, 2.5, 4, 6], elbow_sigma_deg: [0, 4.75, 7.5, 10, 15] };
  const el = stubEl();
  const p = new Lab.KnobPanel(el, ENTRY, state, { isShare: true, rerolls: 2, stops: (id, k) => stops[k] || null });
  assert.match(el.innerHTML, /elbow truncation \(elbow_truncate_deg\), °<\/span> <span class="knob-lock">live only<\/span>/);
  assert.equal(p.lockReason(ENTRY.knobs[1]), null);
  const s = state.panes.A.perEntry[ENTRY.id];
  assert.equal(p.setKnob('A', ENTRY.knobs[1], 9.1), true);
  assert.deepEqual(s.values, { elbow_sigma_deg: 10 }, 'snapped to the nearest stop');
  assert.equal(p.setKnob('A', ENTRY.knobs[1], 10.4), false, 'the same stop again: no new request');
  p.setKnob('A', ENTRY.knobs[0], 3.6);
  assert.deepEqual(s.values, { sigma_deg: 4 }, 'the elbow went back to its default');
  assert.match(p.message, /One knob at a time in the shared version: pane A's elbow σ \(elbow_sigma_deg\) went back to the default\./);
  p.action('dice', 'A', { dataset: {} });
  assert.deepEqual(s.values, {}, 'new draws exist only at the defaults');
  assert.equal(s.reroll, 1);
  p.action('dice', 'A', { dataset: {} });
  p.action('dice', 'A', { dataset: {} });
  assert.equal(s.reroll, 0, 'draws cycle 0, 1, 2, 0');
  p.action('dice', 'A', { dataset: {} });
  p.action('mode', 'A', { dataset: { mode: 'extreme' } });
  assert.equal(s.reroll, 0, 'extremes are precomputed at draw 0');
  p.setKnob('B', ENTRY.knobs[1], 12);
  assert.equal(state.panes.B.perEntry[ENTRY.id].values.elbow_sigma_deg, 10);
});

test('panel: a pane whose draw changed nothing gets a note', () => {
  require('../../src/kinetic_augment/lab/static/explain.js');
  const p = new Lab.KnobPanel(stubEl(), ENTRY, newState(), {});
  p.paneResult('A', { applied: [], mode: 'random' }, 53);
  assert.match(p.notes.A, /unchanged/);
  p.paneResult('A', { applied: ['body.joint_coupled_noise'], draws: {}, mode: 'random' }, 53);
  assert.equal(p.notes.A, '');
});
