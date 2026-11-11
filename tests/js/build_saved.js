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

function build(entry, values, name) {
  const perEntry = {};
  perEntry[entry.id] = { values };
  const got = Lab.savedEdits(schema, perEntry, Lab.parseYaml(base));
  return {
    weights: got.weights,
    name,
    chain: entry.entry,
    values,
    edits: got.edits[entry.entry],
    text: Lab.buildSavedYaml(base, got.edits, got.entries, { name, weights: got.weights }),
  };
}

// Every knob moved, and (`single`) one knob moved that is not one of the run's demo overrides, so
// Python can check the overrides stay as the base has them (Ruling R18).
const out = {};
schema.entries.forEach((entry) => {
  const name = 'lab-test-' + entry.id.replace(/\./g, '-');
  const values = {};
  entry.knobs.forEach((k) => { values[k.key] = moved(k); });
  out[entry.id] = build(entry, values, name);
  const other = entry.knobs.find((k) => !(k.key in (entry.overrides || {})));
  if (other) out[entry.id].single = build(entry, { [other.key]: moved(other) }, name + '-single');
});
fs.writeFileSync(outPath, JSON.stringify(out));
