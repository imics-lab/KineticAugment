// The timeline's clock maths (Lab.frameMap, Lab.clockLength, Lab.nearestFrames), as all.js does it.
const test = require('node:test');
const assert = require('node:assert/strict');

const Lab = require('../../src/kinetic_augment/lab/static/timeline.js');

const base = Float32Array.from([10, 11, 12, 13, 14, 15]); // source frames of the original
const slow = Float32Array.from([10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15]); // 11 frames
const short = Float32Array.from([10, 12, 14]); // every other frame kept

function clip() {
  return {
    nBase: 6,
    A: { n: 11, nearest: Lab.nearestFrames(slow, base) },
    B: { n: 3, nearest: Lab.nearestFrames(short, base) },
  };
}

test('nearestFrames picks the pane frame nearest each source frame', () => {
  assert.deepEqual([...Lab.nearestFrames(short, base)], [0, 0, 1, 1, 2, 2]);
  assert.deepEqual([...Lab.nearestFrames(slow, base)], [0, 2, 4, 6, 8, 10]);
});

test('playback clock: the longest stream sets the length; the shorter pane holds and ends', () => {
  assert.equal(Lab.clockLength('playback', clip()), 11);
  const f = Lab.frameMap('playback', 7, clip());
  assert.equal(f.base, 5);
  assert.deepEqual(f.A, { k: 7, ended: false });
  assert.deepEqual(f.B, { k: 2, ended: true });
});

test('source clock: the original sets the length; panes show their nearest frame', () => {
  assert.equal(Lab.clockLength('source', clip()), 6);
  const f = Lab.frameMap('source', 3, clip());
  assert.equal(f.base, 3);
  assert.deepEqual(f.A, { k: 6, ended: false });
  assert.deepEqual(f.B, { k: 1, ended: false });
});

test('a pane not loaded yet is null', () => {
  const f = Lab.frameMap('playback', 2, { nBase: 6, A: null, B: null });
  assert.equal(f.A, null);
  assert.equal(Lab.clockLength('playback', { nBase: 6, A: null, B: null }), 6);
});

test('playback follows the wall clock, not the screen refresh rate', () => {
  // 30 fps: after 1 s the clip is 30 frames on, however many animation frames ran
  assert.equal(Lab.playFrame(0, 1.0, 30, 1, 100), 30);
  assert.equal(Lab.playFrame(10, 0.5, 30, 0.5, 100), 17); // half speed: 7.5 frames -> 7
  assert.equal(Lab.playFrame(90, 1.0, 30, 1, 100), 20, 'wraps at the clip end');
  assert.equal(Lab.playFrame(5, 0.0, 30, 1, 100), 5);
  assert.equal(Lab.playFrame(0, 0.1, 30, 1, 100), 3, 'no float drift below a whole frame');
});
