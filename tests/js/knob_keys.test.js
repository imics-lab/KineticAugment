// Knob marker keys and the Python/JavaScript range-knob parity: knobs.js rangeFromMarker must equal
// share.range_from_marker (src/kinetic_augment/lab/share.py) so a snapped marker is a precomputed pane;
// arrow keys on a focused marker step to the neighbouring stop; the timeline leaves such keys alone.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const Lab = require('../../src/kinetic_augment/lab/static/knobs.js');
require('../../src/kinetic_augment/lab/static/timeline.js');
const K = Lab.knobMath;

const DIR = process.env.LAB_FIXTURE_DIR || path.join(__dirname, 'fixtures');

test('rangeFromMarker equals share.range_from_marker, and every stop is reachable', () => {
  const data = JSON.parse(fs.readFileSync(path.join(DIR, 'range_parity.json'), 'utf8'));
  assert.ok(data.length >= 9);
  const r9 = (r) => r.map((x) => Number(x.toFixed(9)));
  for (const { id, knob, stops, markers, expected } of data) {
    markers.forEach((m, i) => {
      assert.deepEqual(r9(K.rangeFromMarker(knob, m)), r9(expected[i]), `${id} ${knob.key} marker ${m}`);
    });
    for (const s of stops) {
      // a stop is what the page's rule makes of its own high or low end (or the profile default)
      const ok = [K.rangeFromMarker(knob, s[1]), K.rangeFromMarker(knob, s[0]), knob.default]
        .some((r) => Math.abs(r[0] - s[0]) < 1e-9 && Math.abs(r[1] - s[1]) < 1e-9);
      assert.ok(ok, `${id} ${knob.key} stop ${JSON.stringify(s)}`);
      assert.deepEqual(K.snap(s, stops), s);
    }
  }
});

test('markerKey: arrows step to the neighbouring stop (share) or by step (live); other keys are left alone', () => {
  const scalar = { kind: 'scalar', limits: [0, 10], step: 0.25, default: 2.5 };
  const stops = [0, 2.5, 5, 10];
  assert.equal(K.markerKey(scalar, 'ArrowRight', 2.5, stops), 5);
  assert.equal(K.markerKey(scalar, 'ArrowUp', 2.5, stops), 5);
  assert.equal(K.markerKey(scalar, 'ArrowLeft', 2.5, stops), 0);
  assert.equal(K.markerKey(scalar, 'ArrowRight', 10, stops), null, 'at the last stop');
  assert.equal(K.markerKey(scalar, 'ArrowLeft', 0, stops), null);
  assert.equal(K.markerKey(scalar, 'a', 2.5, stops), undefined);
  assert.equal(K.markerKey(scalar, ' ', 2.5, stops), undefined);
  assert.equal(K.markerKey(scalar, 'ArrowRight', 2.5, null), 2.75, 'live mode: by the step');
  assert.equal(K.markerKey(scalar, 'ArrowLeft', 0, null), 0, 'live mode stops at the limit');
  const range = { kind: 'range', limits: [0.4, 2.5], step: 0.01, default: [0.6, 1.65] };
  const rstops = [[0.6, 1.65], [0.5, 2], [0.4, 2.5]];
  assert.deepEqual(K.markerKey(range, 'ArrowRight', [0.6, 1.65], rstops), [0.5, 2]);
  assert.deepEqual(K.markerKey(range, 'ArrowLeft', [0.5, 2], rstops), [0.6, 1.65]);
  assert.equal(K.markerKey(range, 'ArrowRight', [0.4, 2.5], rstops), null);
  assert.equal(K.markerKey(range, 'ArrowRight', [0.6, 1.65], null)[1], 1.66);
});

test('keys typed into a slider, field or menu are not the timeline\'s', () => {
  const el = (tagName, role, extra) => Object.assign({ tagName, getAttribute: (n) => (n === 'role' ? role : null) }, extra);
  assert.equal(Lab.keyIsForControl(el('SPAN', 'slider')), true, 'a knob marker');
  assert.equal(Lab.keyIsForControl(el('INPUT', null, { type: 'range' })), true);
  assert.equal(Lab.keyIsForControl(el('INPUT', null, { type: 'checkbox' })), true);
  assert.equal(Lab.keyIsForControl(el('SELECT', null)), true);
  assert.equal(Lab.keyIsForControl(el('TEXTAREA', null)), true);
  assert.equal(Lab.keyIsForControl(el('DIV', null, { isContentEditable: true })), true);
  assert.equal(Lab.keyIsForControl(el('BODY', null)), false);
  assert.equal(Lab.keyIsForControl(el('BUTTON', null)), false);
  assert.equal(Lab.keyIsForControl(null), false);
});
