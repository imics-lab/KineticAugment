// Saved settings: the line-based profile edit (Lab.buildSavedYaml), the pane-to-edits mapping
// (Lab.savedEdits), the small YAML reader (Lab.parseYaml) and loading a file back into pane values
// (Lab.valuesFromProfile).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const Lab = require('../../src/kinetic_augment/lab/static/yaml_edit.js');

const BASE = fs.readFileSync(path.join(__dirname, '..', '..', 'configs', 'slr_week4_default.yaml'), 'utf8');

const line = (text, re) => text.split('\n').find((l) => re.test(l));
// The lines of one entry block (`  <region>:` then `    <key>:` up to the next sibling).
function block(text, region, key) {
  const lines = text.split('\n');
  let i = lines.indexOf('  ' + region + ':');
  i = lines.indexOf('    ' + key + ':', i);
  const out = [];
  for (let j = i + 1; j < lines.length && /^ {6,}|^\s*$/.test(lines[j]); j++) out.push(lines[j]);
  return out.join('\n');
}

test('a scalar edit keeps the comment, notes the old value and renames the profile', () => {
  const out = Lab.buildSavedYaml(BASE, { 'body.joint_coupled_noise': { elbow_sigma_deg: 10 } },
    ['body.joint_coupled_noise'], { name: 'elbow-test' });
  assert.equal(line(out, /elbow_sigma_deg:/),
    '        elbow_sigma_deg: 10  # proven: ±7.5° elbow, no harm in pose-based translation [T]; ' +
    'default/ambitious use 4.75  # viewer: was 4.75');
  assert.equal(line(out, /^profile_name:/), 'profile_name: elbow-test');
  assert.equal(line(out, /^\s+sigma_deg: 2.5/), '        sigma_deg: 2.5  # proven: ±4° shoulder/wrist at p=0.3 (SPOTER [M1], [T]); default/ambitious use 2.5');
  assert.equal(out.split('\n').length, BASE.split('\n').length, 'no line added or removed');
});

test('an edited entry is enabled; its probability and the other entries stay as they were', () => {
  const out = Lab.buildSavedYaml(BASE, { 'body.limb_length_scaling': { factor: [0.9, 1.1] } },
    ['body.limb_length_scaling'], { name: 'limbs' });
  const limb = block(out, 'body', 'limb_length_scaling');
  assert.match(limb, /^ {6}enabled: true {2}# setting: opt-in experiment arm$/m);
  assert.match(limb, /^ {6}probability: 1.0 /m);
  assert.match(limb, /^ {8}factor: \[0\.9, 1\.1\] {2}# proven: none — no literature;.* {2}# viewer: was \[0\.93, 1\.07\]$/m);
  assert.match(block(out, 'body', 'tier1_arm_joints'), /^ {6}enabled: false/m);
  assert.match(block(out, 'body', 'movement_size'), /^ {6}enabled: false/m);
});

test('a dotted face key edits only its group; a range is written in flow style', () => {
  const out = Lab.buildSavedYaml(BASE, {
    'face.expression_jitter': { 'brows.sigma': 0.1, 'mouth_jaw.enabled': true },
    'scene.scale': { factor: [0.7, 1.3] },
  }, ['face.expression_jitter', 'scene.scale'], { name: 'face' });
  const face = block(out, 'face', 'expression_jitter');
  assert.match(face, /^ {6}enabled: true/m);
  assert.match(face, /brows:\n {10}enabled: true {2}# setting: blendshape group on\n {10}sigma: 0\.1 {2}# proven: none — no published range; lexical mouth patterns excluded \[L7\.3, L1\.5\] {2}# viewer: was 0\.0625\n/);
  assert.match(face, /eyes:\n {10}enabled: true .*\n {10}sigma: 0\.0625 {2}# proven: none — no published range \[L7\.3, L1\.5\]\n/);
  assert.match(face, /mouth_jaw:\n {10}enabled: true {2}# setting: off, lexical .* {2}# viewer: was false\n/);
  assert.match(line(out, /^\s+factor: \[0\.7/), /^ {8}factor: \[0\.7, 1\.3\] {2}# proven: 0\.8-1\.2 \(GISLR \[M10\]\); default\/ambitious use 0\.75-1\.25 {2}# viewer: was \[0\.75, 1\.25\]$/);
});

test('a value equal to the old one is left alone; an unknown key is an error', () => {
  const out = Lab.buildSavedYaml(BASE, { 'scene.view': { yaw_deg: 25 } }, ['scene.view'], { name: 'same' });
  assert.doesNotMatch(out, /viewer: was/);
  assert.throws(() => Lab.buildSavedYaml(BASE, { 'scene.view': { roll_deg: 3 } }, ['scene.view'], { name: 'x' }),
    /roll_deg/);
  assert.throws(() => Lab.buildSavedYaml(BASE, {}, ['scene.nothing'], { name: 'x' }), /scene\.nothing/);
});

// A cut-down schema with the shapes savedEdits / valuesFromProfile read.
const SCHEMA = {
  entries: [
    { id: 'temporal.time_warp.uniform', entry: 'temporal.time_warp', overrides: { mode_weights: { none: 1, varied: 0, segment: 0 } },
      knobs: [{ key: 'duration', kind: 'range', default: [0.6, 1.65], active: true },
        { key: 'segment_factor', kind: 'range', default: [0.6, 1.65], active: false }] },
    { id: 'temporal.time_warp.segment', entry: 'temporal.time_warp', overrides: { mode_weights: { none: 0, varied: 0, segment: 1 } },
      knobs: [{ key: 'duration', kind: 'range', default: [0.6, 1.65], active: true },
        { key: 'segment_factor', kind: 'range', default: [0.6, 1.65], active: true }] },
    { id: 'temporal.hand_dropout', entry: 'temporal.hand_dropout', overrides: { start_rate: [0.05, 0.05] },
      knobs: [{ key: 'start_rate', kind: 'range', default: [0.05, 0.05], active: true },
        { key: 'max_run_frames', kind: 'scalar', default: 10, active: true }] },
    { id: 'scene.mirror', entry: 'scene.mirror', overrides: { p: 1.0 },
      knobs: [{ key: 'p', kind: 'scalar', default: 1.0, active: true }] },
    { id: 'body.joint_coupled_noise', entry: 'body.joint_coupled_noise', overrides: {},
      knobs: [{ key: 'elbow_sigma_deg', kind: 'scalar', default: 4.75, active: true },
        { key: 'side_mode', kind: 'enum', default: 'mirror', active: true }] },
    { id: 'face.expression_jitter', entry: 'face.expression_jitter', overrides: {},
      knobs: [{ key: 'brows.sigma', kind: 'scalar', default: 0.0625, active: true },
        { key: 'mouth_jaw.enabled', kind: 'bool', default: false, active: true }] },
  ],
};

test('savedEdits: only the knobs the pane moved; demo overrides are never written (R18)', () => {
  const got = Lab.savedEdits(SCHEMA, {
    'temporal.time_warp.segment': { values: { segment_factor: [0.5, 2.0] } },
    'temporal.hand_dropout': { values: { max_run_frames: 6 } },
    'body.joint_coupled_noise': { values: {} }, // visited, nothing moved: not saved
  });
  assert.deepEqual(got.entries.sort(), ['temporal.hand_dropout', 'temporal.time_warp']);
  assert.deepEqual(got.ids.sort(), ['temporal.hand_dropout', 'temporal.time_warp.segment']);
  assert.deepEqual(got.edits['temporal.time_warp'], { segment_factor: [0.5, 2.0] }, 'no mode_weights');
  assert.deepEqual(got.edits['temporal.hand_dropout'], { max_run_frames: 6 }, 'no start_rate');
  const out = Lab.buildSavedYaml(BASE, got.edits, got.entries, { name: 'warp' });
  assert.doesNotMatch(out, /^ {10}(none|varied|segment|rate|random): .*viewer: was/m, 'mode weights as the base has them');
  assert.match(out, /^ {8}start_rate: \[0\.0, 0\.05\] {2}# proven: none/m, 'start_rate as the base has it');
  assert.match(out, /^ {8}max_run_frames: 6 {2}# proven: .* {2}# viewer: was 10$/m);
});

test('savedEdits: mirror is saved with the base p unless the user moved p', () => {
  let got = Lab.savedEdits(SCHEMA, { 'scene.mirror': { values: {} }, 'body.joint_coupled_noise': { values: { elbow_sigma_deg: 10 } } });
  assert.deepEqual(got.edits, { 'body.joint_coupled_noise': { elbow_sigma_deg: 10 } });
  assert.equal(Object.keys(got.edits).some((e) => 'p' in got.edits[e] || 'start_rate' in got.edits[e]), false);
  got = Lab.savedEdits(SCHEMA, { 'scene.mirror': { values: { p: 0.8 } } });
  assert.deepEqual(got.edits, { 'scene.mirror': { p: 0.8 } });
  const out = Lab.buildSavedYaml(BASE, got.edits, got.entries, { name: 'm' });
  assert.match(out, /^ {8}p: 0\.8 {2}# proven: p=0\.5 .* {2}# viewer: was 0\.5$/m);
});

test('parseYaml reads the profile subset: nested maps, flow lists, scalars, comments', () => {
  const data = Lab.parseYaml(BASE);
  assert.equal(data.profile_name, 'slr_week4_default');
  assert.equal(data.seed, 20261107);
  const r = data.regions;
  assert.equal(r.body.joint_coupled_noise.ranges.side_mode, 'mirror');
  assert.deepEqual(r.body.limb_length_scaling.ranges.bones, ['upper_arm', 'forearm']);
  assert.equal(r.body.limb_length_scaling.ranges.retarget, true);
  assert.equal(r.body.limb_length_scaling.enabled, false);
  assert.equal(r.face.expression_jitter.ranges.brows.sigma, 0.0625);
  assert.deepEqual(r.temporal.time_warp.ranges.mode_weights, { none: 1, varied: 0, segment: 0 });
  assert.equal(r.temporal.frame_drop.ranges.fill, 'none');
  assert.deepEqual(Lab.parseYaml('a:\n  - 1\n  - x\nb: "q # not a comment"\nc: ~\n'), { a: [1, 'x'], b: 'q # not a comment', c: null });
});

test('valuesFromProfile: an enabled entry loads into each variant that uses the key', () => {
  const edits = Lab.savedEdits(SCHEMA, {
    'temporal.time_warp.segment': { values: { segment_factor: [0.5, 2.0] } },
    'face.expression_jitter': { values: { 'brows.sigma': 0.1, 'mouth_jaw.enabled': true } },
    'body.joint_coupled_noise': { values: { elbow_sigma_deg: 10 } },
  });
  const text = Lab.buildSavedYaml(BASE, edits.edits, edits.entries, { name: 'back' });
  let got = Lab.valuesFromProfile(SCHEMA, Lab.parseYaml(text));
  assert.deepEqual(got['temporal.time_warp.segment'], { segment_factor: [0.5, 2.0] });
  assert.deepEqual(got['temporal.time_warp.uniform'], {}, 'segment_factor is not a uniform knob');
  assert.deepEqual(got['face.expression_jitter'], { 'brows.sigma': 0.1, 'mouth_jaw.enabled': true });
  assert.deepEqual(got['body.joint_coupled_noise'], { elbow_sigma_deg: 10 }, 'defaults are not copied');
  assert.equal(got['temporal.hand_dropout'], undefined, 'hand dropout is off in the file: not touched');
  assert.deepEqual(got['scene.mirror'], { p: 0.5 }, 'the file\'s p, not the demo p = 1');
  // a uniform-only edit (duration) loads into uniform, and into segment, which also uses it
  const warp = Lab.savedEdits(SCHEMA, { 'temporal.time_warp.uniform': { values: { duration: [0.7, 1.4] } } });
  got = Lab.valuesFromProfile(SCHEMA, Lab.parseYaml(Lab.buildSavedYaml(BASE, warp.edits, warp.entries, { name: 'u' })));
  assert.deepEqual(got['temporal.time_warp.uniform'], { duration: [0.7, 1.4] });
  assert.deepEqual(got['temporal.time_warp.segment'], { duration: [0.7, 1.4] });
});
