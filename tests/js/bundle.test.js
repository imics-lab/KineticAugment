// Lab.decodePane / Lab.gunzip against a blob written by bundle.encode_pane (make_fixture.py).
// LAB_FIXTURE_DIR points at freshly generated fixtures (tests/test_lab_js.py sets it).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const Lab = require('../../src/kinetic_augment/lab/static/bundle.js');

const DIR = process.env.LAB_FIXTURE_DIR || path.join(__dirname, 'fixtures');
const bytes = (name) => new Uint8Array(fs.readFileSync(path.join(DIR, name)));
const expected = JSON.parse(fs.readFileSync(path.join(DIR, 'pane.expected.json'), 'utf8'));

function sameValues(arr, want, name) {
  assert.equal(arr.data.length, want.values.length, `${name} length`);
  for (let i = 0; i < want.values.length; i++) {
    const w = want.values[i];
    if (w === null) assert.ok(Number.isNaN(arr.data[i]), `${name}[${i}] should be NaN`);
    else assert.equal(arr.data[i], arr.data instanceof Float32Array ? Math.fround(w) : w, `${name}[${i}]`);
  }
}

test('decodePane matches bundle.decode_pane for every array', () => {
  const { meta, arrays } = Lab.decodePane(bytes('pane.bin'));
  assert.deepEqual(Object.keys(arrays).sort(), Object.keys(expected.arrays).sort());
  for (const [name, want] of Object.entries(expected.arrays)) {
    const a = arrays[name];
    assert.deepEqual(a.shape, want.shape, `${name} shape`);
    assert.equal(a.stride, want.values.length / want.shape[0], `${name} stride`);
    sameValues(a, want, name);
  }
  assert.deepEqual(meta, expected.meta);
  assert.equal(meta.arrays, undefined, 'the manifest is removed from meta');
});

test('typed arrays use the manifest dtypes', () => {
  const { arrays } = Lab.decodePane(bytes('pane.bin'));
  assert.ok(arrays.times.data instanceof Float32Array);
  assert.ok(arrays.mesh_verts.data instanceof Int16Array);
  assert.ok(arrays.mesh_faces.data instanceof Uint16Array);
  assert.equal(arrays.mesh_faces.data[6], 65000, 'uint16 above 32767 stays unsigned');
});

test('delta mesh: the int16 wrap is undone by the running sum', () => {
  const { arrays } = Lab.decodePane(bytes('pane.bin'));
  const v = arrays.mesh_verts; // (4, 5, 3)
  const z0 = (t) => v.data[t * v.stride + 2];
  assert.deepEqual([z0(0), z0(1), z0(2), z0(3)], [-30000, 30000, -32767, 32767]);
  assert.equal(v.data[3 * v.stride + 4 * 3], 4 * 100 + 3 * 7, 'x of vertex 4 in frame 3');
});

test('NaN points survive', () => {
  const { arrays } = Lab.decodePane(bytes('pane.bin'));
  const arm = arrays.aug_arm;
  assert.ok(Number.isNaN(arm.data[1 * arm.stride + 2 * 3]), 'elbow missing in frame 1');
  assert.ok(Number.isFinite(arm.data[1 * arm.stride + 1 * 3]));
  const hands = arrays.aug_hands;
  for (let i = 21 * 3; i < 42 * 3; i++) assert.ok(Number.isNaN(hands.data[2 * hands.stride + i]));
  assert.ok(Number.isFinite(hands.data[2 * hands.stride]));
});

test('decodePane works on an unaligned view of the bytes', () => {
  const src = bytes('pane.bin');
  const shifted = new Uint8Array(src.length + 1);
  shifted.set(src, 1);
  const { arrays } = Lab.decodePane(shifted.subarray(1));
  sameValues(arrays.aug_hands, expected.arrays.aug_hands, 'aug_hands');
  sameValues(arrays.mesh_verts, expected.arrays.mesh_verts, 'mesh_verts');
});

test('decodePane refuses bytes that are not a pane blob', () => {
  assert.throws(() => Lab.decodePane(new Uint8Array([1, 2, 3, 4, 0, 0, 0, 0])), /not a Lab pane/);
});

test('gunzip inflates a gzipped blob', async () => {
  const raw = await Lab.gunzip(bytes('pane.bin.gz'));
  assert.deepEqual(Buffer.from(raw), Buffer.from(bytes('pane.bin')));
});

test('gunzip says the browser is too old without DecompressionStream', async () => {
  const saved = globalThis.DecompressionStream;
  try {
    globalThis.DecompressionStream = undefined;
    await assert.rejects(() => Lab.gunzip(bytes('pane.bin.gz')), /browser too old/);
  } finally {
    globalThis.DecompressionStream = saved;
  }
});
