// Lab.ShareSource / Lab.requestKey against a share manifest keyed by PaneRequest.share_key() in
// Python (tests/js/make_fixture.py): each case must be found from requests as the UI sends them.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');

const Lab = require('../../src/kinetic_augment/lab/static/sources.js');

const DIR = process.env.LAB_FIXTURE_DIR || path.join(__dirname, 'fixtures');
const blob = fs.readFileSync(path.join(DIR, 'pane.bin'));
const share = JSON.parse(fs.readFileSync(path.join(DIR, 'share.json'), 'utf8'));

function makeData() {
  Lab.blobs = {};
  const gz = zlib.gzipSync(blob).toString('base64');
  for (const c of share.cases) Lab.addBlob(c.id, gz);
  Lab.addBlob('base1', blob.toString('base64')); // stored without gzip: also accepted
  return {
    schema: share.schema,
    profileText: 'profile_name: default\n',
    clips: [{ id: 'fx', n: 4, fps: 29.97 }],
    base: { fx: 'base1' },
    panes: share.panes,
    stops: share.stops,
    rerolls: share.rerolls,
  };
}

const ui = (lookup) => Object.assign({ clip_id: 'fx' }, lookup);

test('requestKey matches PaneRequest.key() after a JSON round trip', () => {
  // Python writes 1.0 where JavaScript writes 1, so keys are compared after a JSON round trip.
  const req = { clip_id: 'fx', entry_id: 'face.expression_jitter', values: { 'mouth.sigma': 0.8, 'brows.sigma': 1.0 } };
  assert.equal(Lab.requestKey(req), Lab.canonicalKey(share.keys[0]));
  assert.equal(Lab.requestKey({ clip_id: 'fx', entry_id: 'combined.default', reroll: 1 }), Lab.canonicalKey(share.keys[4]));
});

test('every share_key case is found from its UI requests, normalised or not', () => {
  const src = new Lab.ShareSource(makeData());
  assert.ok(share.cases.length >= 5);
  for (const c of share.cases) {
    for (const lookup of c.lookups) {
      assert.equal(src.paneId(ui(lookup)), c.id, `${JSON.stringify(lookup)} -> ${c.key}`);
    }
    // the manifest key itself, read back as a request, finds it too
    assert.equal(src.paneId(JSON.parse(c.key)), c.id);
  }
});

test('requests that were not precomputed find nothing', async () => {
  const src = new Lab.ShareSource(makeData());
  for (const miss of share.misses) assert.equal(src.paneId(ui(miss)), null, JSON.stringify(miss));
  await assert.rejects(() => src.pane(ui(share.misses[1])), /not in the shared version/);
});

test('ShareSource.pane decodes the blob it finds', async () => {
  const src = new Lab.ShareSource(makeData());
  assert.equal(src.isShare, true);
  const pane = await src.pane(ui(share.cases[0].lookups[0]));
  assert.equal(pane.meta.clip, 'fx');
  assert.deepEqual(pane.arrays.mesh_verts.shape, [4, 5, 3]);
  const extreme = await src.pane(ui(share.cases[2].lookups[0]));
  assert.equal(extreme.meta.n, 4);
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

test('watchJob polls every interval until done or failed, reporting each status', async () => {
  const states = [
    { state: 'waiting', step: '', progress: 0, error: null },
    { state: 'running', step: 'extracting landmarks', progress: 0.1, error: null },
    { state: 'running', step: 'fitting', progress: 0.4, error: null },
    { state: 'done', step: 'done', progress: 1, error: null },
  ];
  const waits = [];
  const seen = [];
  const src = { job: async (id) => { assert.equal(id, 'j1'); return states.shift(); } };
  const done = await Lab.watchJob(src, 'j1', {
    interval: 2000, onUpdate: (s) => seen.push(s.step || s.state), wait: async (ms) => { waits.push(ms); },
  });
  assert.equal(done.state, 'done');
  assert.deepEqual(seen, ['waiting', 'extracting landmarks', 'fitting', 'done']);
  assert.deepEqual(waits, [2000, 2000, 2000]);

  const failing = { job: async () => ({ state: 'failed', step: 'fitting', progress: 0.4, error: 'no hands found' }) };
  await assert.rejects(() => Lab.watchJob(failing, 'j2', { wait: async () => {} }), /no hands found/);
  const gone = { job: async () => { const e = new Error('no such job'); e.status = 404; throw e; } };
  await assert.rejects(() => Lab.watchJob(gone, 'j3', { wait: async () => {} }), /no such job/);
});
