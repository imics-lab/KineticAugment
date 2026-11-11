// Saved settings as profile YAML, without a YAML library (spec §5.7):
//   Lab.buildSavedYaml(baseText, edits, entries, {name}) -> text: a line-based edit of the base
//     profile. Each entry in `entries` (and in `edits`) gets `enabled: true` (its probability is
//     left as the base has it); each edited range leaf gets the new value (ranges in flow style,
//     `[0.9, 1.1]`), keeps its comment and adds `  # viewer: was <old>`; `profile_name` becomes
//     `name`. Entry keys are YAML paths `region.key`; range keys are dotted for nested ones
//     (`brows.sigma`, `mode_weights.segment`). An edit with no line in the base is an error.
//   Lab.savedEdits(schema, perEntry) -> {edits, entries}: a pane's per-entry settings as edits.
//     An entry counts as edited when it has values; its run's overrides (`mode_weights` of a
//     time-warp or frame-drop variant, the demo-only `start_rate` / `p`) are written too, so the
//     file runs what the pane ran (`run_profile` merges overrides into `ranges`). Two variants of
//     one chain entry add up their mode weights.
//   Lab.parseYaml(text) -> object: the YAML subset profiles use (block maps, block and flow
//     lists of scalars, plain and quoted scalars, comments).
//   Lab.valuesFromProfile(schema, data) -> {entryId: values}: for every lab entry the profile
//     enables (a variant only when its mode weight is on), the knob values that differ from the
//     knob defaults.
// Classic script: adds to window.Lab and module.exports.
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  // ---- values ---------------------------------------------------------------------------
  var RESERVED = /^(true|false|yes|no|on|off|null|~|y|n)$/i;

  function fmtNumber(n) {
    if (!isFinite(n)) throw new Error('not a finite number: ' + n);
    if (Number.isInteger(n)) return String(n);
    var s = String(Number(n.toPrecision(12)));
    if (/e/i.test(s)) s = n.toFixed(12).replace(/0+$/, '').replace(/\.$/, '.0');
    return s;
  }

  function fmtValue(v) {
    if (Array.isArray(v)) return '[' + v.map(fmtValue).join(', ') + ']';
    if (typeof v === 'number') return fmtNumber(v);
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (v === null || v === undefined) return 'null';
    var s = String(v);
    if (/^[A-Za-z_][A-Za-z0-9_.\-]*$/.test(s) && !RESERVED.test(s)) return s;
    return JSON.stringify(s);
  }

  function same(a, b) {
    if (Array.isArray(a) || Array.isArray(b)) {
      return Array.isArray(a) && Array.isArray(b) && a.length === b.length &&
        a.every(function (x, i) { return same(x, b[i]); });
    }
    if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
    return a === b;
  }

  // ---- scalars and comments -------------------------------------------------------------
  function parseScalar(s) {
    s = s.trim();
    if (s === '' || s === '~' || s === 'null' || s === 'Null' || s === 'NULL') return null;
    if (/^(true|True|TRUE)$/.test(s)) return true;
    if (/^(false|False|FALSE)$/.test(s)) return false;
    if (s[0] === '"') return JSON.parse(s);
    if (s[0] === "'") return s.slice(1, -1).replace(/''/g, "'");
    if (/^[-+]?(\d[\d_]*)$/.test(s)) return parseInt(s.replace(/_/g, ''), 10);
    if (/^[-+]?(\d*\.\d*([eE][-+]?\d+)?|\d+[eE][-+]?\d+)$/.test(s) && /\d/.test(s)) return parseFloat(s);
    if (/^[-+]?\.(inf|Inf|INF)$/.test(s)) return s[0] === '-' ? -Infinity : Infinity;
    if (s[0] === '[') return splitFlow(s.slice(1, s.lastIndexOf(']'))).map(parseScalar);
    if (s[0] === '{') {
      var obj = {};
      splitFlow(s.slice(1, s.lastIndexOf('}'))).forEach(function (kv) {
        var i = kv.indexOf(':');
        obj[kv.slice(0, i).trim()] = parseScalar(kv.slice(i + 1));
      });
      return obj;
    }
    return s;
  }

  function splitFlow(body) {
    var parts = [], depth = 0, quote = null, cur = '';
    for (var i = 0; i < body.length; i++) {
      var c = body[i];
      if (quote) { cur += c; if (c === quote) quote = null; continue; }
      if (c === '"' || c === "'") { quote = c; cur += c; continue; }
      if (c === '[' || c === '{') depth++;
      if (c === ']' || c === '}') depth--;
      if (c === ',' && depth === 0) { parts.push(cur); cur = ''; continue; }
      cur += c;
    }
    if (cur.trim() !== '') parts.push(cur);
    return parts.map(function (p) { return p.trim(); });
  }

  // Splits "value  # comment" at the first '#' that follows whitespace outside quotes.
  function splitComment(rest) {
    var quote = null;
    for (var i = 0; i < rest.length; i++) {
      var c = rest[i];
      if (quote) { if (c === quote) quote = null; continue; }
      if (c === '"' || c === "'") { quote = c; continue; }
      if (c === '#' && (i === 0 || /\s/.test(rest[i - 1]))) {
        return { value: rest.slice(0, i), comment: rest.slice(i) };
      }
    }
    return { value: rest, comment: '' };
  }

  var KEY_LINE = /^(\s*)([A-Za-z0-9_][A-Za-z0-9_.\-]*)\s*:(\s.*|)$/;

  // ---- reading --------------------------------------------------------------------------
  Lab.parseYaml = function parseYaml(text) {
    var root = {};
    var stack = [{ indent: -1, node: root, key: null, parent: null }];
    text.split(/\r?\n/).forEach(function (raw, n) {
      var stripped = splitComment(raw).value;
      if (!stripped.trim() || /^\s*(---|\.\.\.)\s*$/.test(stripped)) return;
      var indent = stripped.match(/^\s*/)[0].length;
      while (stack.length > 1 && stack[stack.length - 1].indent >= indent) stack.pop();
      var top = stack[stack.length - 1];
      var item = stripped.match(/^(\s*)-\s+(.*)$/) || stripped.match(/^(\s*)-()$/);
      if (item) {
        // a block list item under the last key: turn that key's empty map into a list
        if (!Array.isArray(top.node)) {
          if (top.parent && Object.keys(top.node).length === 0) {
            top.node = top.parent[top.key] = [];
          } else {
            throw new Error('line ' + (n + 1) + ': a list item where a key was expected');
          }
        }
        top.node.push(parseScalar(item[2]));
        return;
      }
      var m = stripped.match(KEY_LINE);
      if (!m) throw new Error('line ' + (n + 1) + ': cannot read "' + raw.trim() + '"');
      if (Array.isArray(top.node)) throw new Error('line ' + (n + 1) + ': a key inside a list');
      var key = m[2], rest = m[3].trim();
      if (rest === '') {
        top.node[key] = {};
        stack.push({ indent: indent, node: top.node[key], key: key, parent: top.node });
      } else {
        top.node[key] = parseScalar(rest);
      }
    });
    return root;
  };

  // ---- writing --------------------------------------------------------------------------
  function flatten(obj, prefix, out) {
    Object.keys(obj || {}).forEach(function (k) {
      var v = obj[k], key = prefix ? prefix + '.' + k : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, key, out);
      else out[key] = v;
    });
    return out;
  }

  Lab.buildSavedYaml = function buildSavedYaml(baseText, edits, entries, opts) {
    edits = edits || {};
    var name = opts && opts.name;
    var enable = {};
    (entries || []).concat(Object.keys(edits)).forEach(function (e) { enable[e] = false; });
    var pending = {}; // "region.key|range.key" -> true until written
    Object.keys(edits).forEach(function (e) {
      Object.keys(edits[e]).forEach(function (k) { pending[e + '|' + k] = true; });
    });
    var nameDone = !name;
    var stack = [];
    var lines = baseText.split('\n').map(function (raw) {
      var m = raw.match(KEY_LINE);
      if (!m || /^\s*#/.test(raw)) return raw;
      var indent = m[1].length;
      while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
      var parts = splitComment(m[3]);
      var value = parts.value.trim();
      var path = stack.map(function (s) { return s.key; }).concat([m[2]]);
      if (value === '') { stack.push({ indent: indent, key: m[2] }); return raw; }
      if (path.length === 1 && path[0] === 'profile_name' && name) {
        nameDone = true;
        return m[1] + 'profile_name: ' + fmtValue(name) + (parts.comment ? '  ' + parts.comment : '');
      }
      if (path[0] !== 'regions' || path.length < 4) return raw;
      var entryKey = path[1] + '.' + path[2];
      if (path.length === 4 && path[3] === 'enabled' && entryKey in enable) {
        enable[entryKey] = true;
        if (parseScalar(value) === true) return raw;
        return m[1] + 'enabled: true' + (parts.comment ? '  ' + parts.comment : '');
      }
      if (path[3] !== 'ranges' || path.length < 5 || !edits[entryKey]) return raw;
      var rangeKey = path.slice(4).join('.');
      if (!(rangeKey in edits[entryKey])) return raw;
      delete pending[entryKey + '|' + rangeKey];
      var next = edits[entryKey][rangeKey], old = parseScalar(value);
      if (same(old, next)) return raw;
      var comment = parts.comment ? '  ' + parts.comment.replace(/\s+$/, '') : '';
      return m[1] + m[2] + ': ' + fmtValue(next) + comment + '  # viewer: was ' + value;
    });
    var missing = Object.keys(enable).filter(function (e) { return !enable[e]; })
      .map(function (e) { return e + ' (no enabled line)'; })
      .concat(Object.keys(pending).map(function (p) { return p.replace('|', ' → ranges.'); }));
    if (!nameDone) missing.push('profile_name');
    if (missing.length) throw new Error('the base profile has no line for ' + missing.join(', '));
    return lines.join('\n');
  };

  Lab.savedEdits = function savedEdits(schema, perEntry) {
    var edits = {}, entries = [];
    ((schema && schema.entries) || []).forEach(function (entry) {
      var s = perEntry && perEntry[entry.id];
      if (!s || !s.values || !Object.keys(s.values).length) return;
      var target = edits[entry.entry] = edits[entry.entry] || {};
      if (entries.indexOf(entry.entry) < 0) entries.push(entry.entry);
      var over = flatten(entry.overrides || {}, '', {});
      Object.keys(over).forEach(function (k) {
        if (/^mode_weights\./.test(k) && typeof target[k] === 'number') target[k] += over[k];
        else target[k] = over[k];
      });
      Object.keys(s.values).forEach(function (k) { target[k] = s.values[k]; });
    });
    return { edits: edits, entries: entries };
  };

  function dig(obj, dotted) {
    return dotted.split('.').reduce(function (node, part) {
      return node && typeof node === 'object' && !Array.isArray(node) ? node[part] : undefined;
    }, obj);
  }

  Lab.valuesFromProfile = function valuesFromProfile(schema, data) {
    var out = {};
    var regions = (data && data.regions) || {};
    ((schema && schema.entries) || []).forEach(function (entry) {
      var dot = entry.entry.indexOf('.');
      var body = (regions[entry.entry.slice(0, dot)] || {})[entry.entry.slice(dot + 1)];
      if (!body || body.enabled !== true) return;
      var ranges = body.ranges || {};
      var weights = (entry.overrides || {}).mode_weights;
      if (weights) {
        var file = ranges.mode_weights || {};
        var on = Object.keys(weights).filter(function (k) { return weights[k] > 0; });
        if (!on.every(function (k) { return Number(file[k]) > 0; })) return;
      }
      var values = {};
      entry.knobs.forEach(function (k) {
        var v = dig(ranges, k.key);
        if (v !== undefined && !same(v, k.default)) values[k.key] = v;
      });
      out[entry.id] = values;
    });
    return out;
  };

  Lab.yamlValue = fmtValue;
  Lab.sameValue = same;
})();

if (typeof module !== 'undefined') module.exports = Lab;
