// Lab.ShareSource / Lab.requestKey against a share manifest keyed by PaneRequest.key() in Python.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');

const Lab = require('../../src/kinetic_augment/lab/static/sources.js');

const DIR = process.env.LAB_FIXTURE_DIR || path.join(__dirname, 'fixtures');
const blob = fs.readFileSync(path.join(DIR, 'pane.bin'));
const share = JSON.parse(fs.readFileSync(path.join(DIR, 'share.json'), 'utf8'));

const SCHEMA = {
  profile: 'default',
  seed: 7,
  entries: [
    { id: 'face.expression_jitter', knobs: [{ key: 'brows.sigma', default: 1.0 }, { key: 'mouth.sigma', default: 0.5 }] },
    { id: 'body.joint_coupled_noise', knobs: [{ key: 'sigma_deg', default: 2.5 }] },
  ],
};

function makeData() {
  Lab.blobs = {};
  Lab.addBlob(share.sha1, zlib.gzipSync(blob).toString('base64'));
  Lab.addBlob('base1', blob.toString('base64')); // stored without gzip: also accepted
  return {
    schema: SCHEMA,
    profileText: 'profile_name: default\n',
    clips: [{ id: 'fx', n: 4, fps: 29.97 }],
    base: { fx: 'base1' },
    panes: share.panes,
    stops: share.stops,
    rerolls: share.rerolls,
  };
}

test('requestKey matches PaneRequest.key() after a JSON round trip', () => {
  const req = { clip_id: 'fx', entry_id: 'face.expression_jitter', values: { 'mouth.sigma': 0.5, 'brows.sigma': 1.0 } };
  // Python writes 1.0 where JavaScript writes 1, so both sides are compared after a JSON round trip.
  assert.equal(Lab.requestKey(req), Lab.canonicalKey(share.keys[0]));
  assert.equal(
    Lab.requestKey({ clip_id: 'fx', entry_id: 'combined.default', reroll: 2 }),
    Lab.canonicalKey(share.keys[2]),
  );
});

test('ShareSource.pane finds the blob by request key and decodes it', async () => {
  const src = new Lab.ShareSource(makeData());
  assert.equal(src.isShare, true);
  const pane = await src.pane({
    clip_id: 'fx', entry_id: 'face.expression_jitter', values: { 'brows.sigma': 1, 'mouth.sigma': 0.5 },
    mode: 'random', reroll: 0, edge: 'high', profile: 'default',
  });
  assert.equal(pane.meta.clip, 'fx');
  assert.deepEqual(pane.arrays.mesh_verts.shape, [4, 5, 3]);
  const extreme = await src.pane({ clip_id: 'fx', entry_id: 'body.joint_coupled_noise', mode: 'extreme', edge: 'low' });
  assert.equal(extreme.meta.n, 4);
});

test('ShareSource.pane treats values at their default like no value', async () => {
  const src = new Lab.ShareSource(makeData());
  // the manifest key has brows.sigma 1.0 and mouth.sigma 0.5, both the schema defaults
  const pane = await src.pane({ clip_id: 'fx', entry_id: 'face.expression_jitter', values: {} });
  assert.equal(pane.meta.clip, 'fx');
  // and random mode ignores the extreme edge
  await src.pane({ clip_id: 'fx', entry_id: 'combined.default', reroll: 2, edge: 'low' });
});

test('ShareSource.pane rejects a setting that was not precomputed', async () => {
  const src = new Lab.ShareSource(makeData());
  await assert.rejects(
    () => src.pane({ clip_id: 'fx', entry_id: 'body.joint_coupled_noise', values: { sigma_deg: 3.1 } }),
    /not in the shared version/,
  );
});

test('ShareSource.stops returns the manifest stops, null where there are none', () => {
  const src = new Lab.ShareSource(makeData());
  assert.deepEqual(src.stops('body.joint_coupled_noise', 'sigma_deg'), [0, 1.25, 2.5, 4, 5, 7.5, 10]);
  assert.equal(src.stops('body.joint_coupled_noise', 'truncate_deg'), null);
  assert.equal(src.stops('nope', 'sigma_deg'), null);
});

test('ShareSource serves the base blob, clips, schema, profile text and frame paths', async () => {
  const src = new Lab.ShareSource(makeData());
  const base = await src.base('fx');
  assert.deepEqual(base.arrays.mesh_faces.shape, [3, 3]);
  assert.deepEqual(await src.clips(), [{ id: 'fx', n: 4, fps: 29.97 }]);
  assert.equal((await src.schema('default')).seed, 7);
  assert.equal(await src.profileText('default'), 'profile_name: default\n');
  assert.equal(src.frameUrl('fx', 7), 'data/fx/frames/0007.jpg');
  assert.equal((await src.status()).mode, 'share');
  assert.deepEqual(await src.saved(), []);
  await assert.rejects(() => src.addVideo(null), /full mode/);
});

test('LiveSource is not share mode and has no stops', () => {
  const src = new Lab.LiveSource('');
  assert.equal(src.isShare, false);
  assert.equal(src.stops('body.joint_coupled_noise', 'sigma_deg'), null);
  assert.equal(src.frameUrl('57949', 3), '/api/clips/57949/frames/3.jpg');
});

test('LiveSource.pane posts the request and decodes the blob; a 422 carries its detail', async () => {
  const calls = [];
  const saved = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    if (JSON.parse(init.body).reroll === 9) {
      return { ok: false, status: 422, json: async () => ({ detail: 'sigma_deg: too big' }) };
    }
    return { ok: true, status: 200, arrayBuffer: async () => blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.length) };
  };
  try {
    const src = new Lab.LiveSource('');
    const pane = await src.pane({ clip_id: 'fx', entry_id: 'body.joint_coupled_noise' });
    assert.equal(calls[0].url, '/api/pane');
    assert.equal(calls[0].init.method, 'POST');
    assert.deepEqual(JSON.parse(calls[0].init.body), {
      clip_id: 'fx', entry_id: 'body.joint_coupled_noise', values: {}, mode: 'random', reroll: 0, edge: 'high', profile: 'default',
    });
    assert.equal(pane.meta.n, 4);
    await assert.rejects(
      () => src.pane({ clip_id: 'fx', entry_id: 'body.joint_coupled_noise', reroll: 9 }),
      (err) => err.status === 422 && /too big/.test(err.message),
    );
  } finally {
    globalThis.fetch = saved;
  }
});
