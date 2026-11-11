// The explanation box and readout text (Lab.explainHtml, Lab.readoutHtml, Lab.unchangedReason).
const test = require('node:test');
const assert = require('node:assert/strict');

const Lab = require('../../src/kinetic_augment/lab/static/explain.js');

const ENTRY = {
  id: 'scene.mirror', entry: 'scene.mirror', title: 'Mirror left and right', does: 'Flips the body <left> to right.',
  changes: ['which hand does what'], keeps: ['the timing'], proven: 'Half the clips flipped', used: 'Half (p = 0.5)',
  note: '', overrides: { p: 1.0 }, knobs: [],
};

test('explainHtml: does, changes, keeps, proven vs used, escaped; demo overrides said in words', () => {
  const html = Lab.explainHtml(ENTRY);
  assert.match(html, /<h2>Mirror left and right <span class="tech">scene\.mirror<\/span><\/h2>/);
  assert.match(html, /Flips the body &lt;left&gt; to right\./);
  assert.match(html, /<h3>Changes<\/h3><ul><li>which hand does what<\/li><\/ul>/);
  assert.match(html, /<h3>Keeps<\/h3><ul><li>the timing<\/li><\/ul>/);
  assert.match(html, /Proven:<\/b> Half the clips flipped/);
  assert.match(html, /Used:<\/b> Half \(p = 0\.5\)/);
  assert.match(html, /panes start from p = 1 rather than the profile/);
  const warp = Lab.explainHtml(Object.assign({}, ENTRY, { overrides: { mode_weights: { none: 1 } }, note: 'Opt-in.' }));
  assert.doesNotMatch(warp, /panes start from/);
  assert.match(warp, /<p class="ex-note">Opt-in\.<\/p>/);
});

test('explainHtml: a whole profile is explained and marked view only', () => {
  const html = Lab.explainHtml({ id: 'combined.default', title: 'Default profile (whole chain)', whole: true, name: 'default' });
  assert.match(html, /slr_week4_default\.yaml/);
  assert.match(html, /View only/);
  assert.match(Lab.explainHtml({ id: 'saved:elbow-test', title: 's', whole: true, name: 'elbow-test' }), /viewer_saved\/elbow-test\.yaml/);
});

test('readoutHtml: draws, metrics in words with units, fallbacks', () => {
  const meta = {
    n: 53, mode: 'random', edge: 'high', reroll: 2, mirrored: false,
    applied: ['body.joint_coupled_noise'],
    draws: { 'body.joint_coupled_noise': {
      joints: [{ side: 'left', joint: 'elbow', angle_deg: 2.6747, axis: [0, 0, 1] }], drift: null, wrist_moved_mm_max: 27.1,
    } },
    metrics: { wrist_moved_mm_max: 27.1109, bone_length_change_mm_max: 1.9e-13, jitter_ratio: 1.0069, in_frame_pct: 99.9514 },
    fallbacks: { 'fingers.handshape_jitter': 'no_valid_redraw' },
  };
  const html = Lab.readoutHtml(meta, 'A');
  assert.match(html, /random draw 2 · 53 frames/);
  assert.match(html, /joints<\/span> left elbow 2\.67°/);
  assert.doesNotMatch(html, /drift/);
  assert.match(html, /The wrists moved at most 27\.1 mm\./);
  assert.match(html, /Bone lengths changed by at most 0 mm\./);
  assert.match(html, /Wrist shake is 1\.01× the original/);
  assert.match(html, /99\.95% of the points stay inside/);
  assert.match(html, /every redraw would have changed the handshape/);
  const keep = Lab.readoutHtml({ n: 40, mode: 'extreme', edge: 'low', applied: ['temporal.frame_drop'],
    draws: { 'temporal.frame_drop': { keep: new Array(40).fill(0), p: 0.2, fill: 'none' } }, metrics: {}, fallbacks: {} }, 'B');
  assert.match(keep, /extreme, low edge/);
  assert.match(keep, /keep<\/span> 40 of 40 frames/);
});

test('unchangedReason: nothing applied, or a frame drop that kept every frame', () => {
  assert.match(Lab.unchangedReason({ applied: [], mode: 'random' }, 53), /unchanged/);
  const drop = { applied: ['temporal.frame_drop'], mode: 'extreme', edge: 'high',
    draws: { 'temporal.frame_drop': { keep: Array.from({ length: 53 }, (_, i) => i), p: 0 } } };
  assert.match(Lab.unchangedReason(drop, 53), /high edge .* drops no frame/);
  drop.draws['temporal.frame_drop'].keep = [0, 2, 4];
  assert.equal(Lab.unchangedReason(drop, 53), '');
  assert.equal(Lab.unchangedReason({ applied: ['scene.scale'], draws: { 'scene.scale': { factor: 0.75 } } }, 53), '');
});
