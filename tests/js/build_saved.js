// Helper for tests/test_lab_yaml_edit.py (not a test itself):
//   node build_saved.js <schema.json> <base.yaml> <out.json>
// For every lab entry, moves every knob one step off its default, builds the saved YAML the Save
// button would write (Lab.savedEdits + Lab.buildSavedYaml) and writes
// {entryId: {name, text, values, chain}} for Python to validate.
const fs = require('node:fs');
const Lab = require('../../src/kinetic_augment/lab/static/yaml_edit.js');
require('../../src/kinetic_augment/lab/static/knobs.js');

const [schemaPath, basePath, outPath] = process.argv.slice(2);
const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const base = fs.readFileSync(basePath, 'utf8');
const K = Lab.knobMath;

function moved(knob) {
  const d = knob.default;
  if (knob.kind === 'bool') return !d;
  if (knob.kind === 'enum') {
    if (knob.multi) return d.length > 1 ? d.slice(0, -1) : knob.options.slice(0, 2);
    return knob.options[(knob.options.indexOf(d) + 1) % knob.options.length];
  }
  const [lo, hi] = knob.limits;
  if (knob.kind === 'scalar') {
    const up = K.roundTo(d + knob.step, knob.step);
    return up <= hi ? up : K.roundTo(d - knob.step, knob.step);
  }
  const m = K.markerOf(knob, d);
  const next = m + 2 * knob.step <= hi ? m + 2 * knob.step : m - 2 * knob.step;
  const r = K.rangeFromMarker(knob, next);
  return r[0] < lo ? [lo, r[1]] : r;
}

const out = {};
schema.entries.forEach((entry) => {
  const values = {};
  entry.knobs.forEach((k) => { values[k.key] = moved(k); });
  const perEntry = {};
  perEntry[entry.id] = { values };
  const got = Lab.savedEdits(schema, perEntry);
  const name = 'lab-test-' + entry.id.replace(/\./g, '-');
  out[entry.id] = {
    name,
    chain: entry.entry,
    values,
    edits: got.edits[entry.entry],
    text: Lab.buildSavedYaml(base, got.edits, got.entries, { name }),
  };
});
fs.writeFileSync(outPath, JSON.stringify(out));
