// The share zip loads blob scripts on first use (Lab.loadScript), not at page open.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');

const Lab = require('../../src/kinetic_augment/lab/static/sources.js');

const DIR = process.env.LAB_FIXTURE_DIR || path.join(__dirname, 'fixtures');
const blob = fs.readFileSync(path.join(DIR, 'pane.bin'));
const share = JSON.parse(fs.readFileSync(path.join(DIR, 'share.json'), 'utf8'));
const gz = zlib.gzipSync(blob).toString('base64');
const ui = (lookup) => Object.assign({ clip_id: 'fx' }, lookup);

function makeData() {
  return {
    schema: share.schema,
    clips: [{ id: 'fx', n: 4, fps: 29.97 }],
    base: { fx: 'base1' },
    panes: share.panes,
    stops: share.stops,
    rerolls: share.rerolls,
  };
}

test('blobs load on first use through Lab.loadScript, once, and are dropped after decoding', async () => {
  Lab.blobs = {};
  const loaded = [];
  const saved = Lab.loadScript;
  Lab.loadScript = (src) => {
    loaded.push(src);
    const sha = src.split('/').pop().replace('.js', '');
    return new Promise((resolve) => setTimeout(() => { Lab.addBlob(sha, gz); resolve(); }, 5));
  };
  try {
    const src = new Lab.ShareSource(makeData());
    assert.deepEqual(Object.keys(Lab.blobs), [], 'nothing loaded at open');
    const lookup = ui(share.cases[0].lookups[0]);
    const [a, b] = await Promise.all([src.pane(lookup), src.pane(lookup)]);
    assert.equal(a, b, 'concurrent requests share one load');
    assert.deepEqual(loaded, [`data/fx/${share.cases[0].id}.js`]);
    assert.equal(Lab.blobs[share.cases[0].id], undefined, 'the base64 text is freed after decoding');
    await src.pane(lookup);
    assert.equal(loaded.length, 1, 'a decoded pane is cached');
    await src.pane(ui(share.cases[2].lookups[0]));
    assert.equal(loaded[1], `data/fx/${share.cases[2].id}.js`);
    await src.base('fx'); // the clip folder of the base blob comes from the manifest's base map
    assert.equal(loaded[2], 'data/fx/base1.js');
  } finally {
    Lab.loadScript = saved;
  }
});

test('a data script that is missing rejects, and a later request tries again', async () => {
  Lab.blobs = {};
  const saved = Lab.loadScript;
  let calls = 0;
  Lab.loadScript = async (src) => { calls++; throw new Error('missing data file ' + src); };
  try {
    const src = new Lab.ShareSource(makeData());
    await assert.rejects(() => src.pane(ui(share.cases[0].lookups[0])), /missing data file/);
    await assert.rejects(() => src.pane(ui(share.cases[0].lookups[0])), /missing data file/);
    assert.equal(calls, 2);
  } finally {
    Lab.loadScript = saved;
  }
});

test('Lab.loadScript adds a script tag, resolves on load and rejects on error', async () => {
  const added = [];
  const head = {
    appendChild: (t) => { added.push(t); setTimeout(() => (t.src.includes('bad') ? t.onerror() : t.onload()), 1); },
  };
  globalThis.document = { head, createElement: () => ({ parentNode: null }) };
  try {
    const p1 = Lab.loadScript('data/fx/a.js');
    assert.equal(Lab.loadScript('data/fx/a.js'), p1, 'one tag per script');
    await p1;
    await assert.rejects(() => Lab.loadScript('data/fx/bad.js'), /missing data file data\/fx\/bad\.js/);
    assert.equal(added.length, 2);
    await Lab.loadScript('data/fx/a.js'); // a finished script may be loaded again (an evicted blob)
    assert.equal(added.length, 3);
  } finally {
    delete globalThis.document;
  }
});
