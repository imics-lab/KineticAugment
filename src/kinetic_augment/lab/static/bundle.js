// Lab pane blobs (bundle.encode_pane): "KLAB", uint32 LE header length, UTF-8 JSON header padded
// with spaces so the array bytes start 4-byte aligned, then the arrays (little-endian). Manifest
// offsets count from the first array byte. `delta: true` (mesh_verts) stores frame 0 followed by
// int16 frame differences that may wrap; a running sum wraps them back.
// Classic script: adds Lab.decodePane, Lab.gunzip, Lab.at to window.Lab (and module.exports).
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  var DTYPES = { float32: Float32Array, int16: Int16Array, uint16: Uint16Array };
  var MAGIC = [75, 76, 65, 66]; // "KLAB"

  function product(shape) {
    var n = 1;
    for (var i = 0; i < shape.length; i++) n *= shape[i];
    return n;
  }

  // Undo the frame differences in place: an Int32 running sum per element, stored as int16
  // (the store wraps exactly as numpy's int32 cumsum cast back to int16).
  function undoDelta(data, frames, stride) {
    var acc = new Int32Array(stride);
    for (var t = 0; t < frames; t++) {
      var row = t * stride;
      for (var i = 0; i < stride; i++) {
        acc[i] = t === 0 ? data[row + i] : (acc[i] + data[row + i]) | 0;
        data[row + i] = acc[i];
      }
    }
  }

  // decodePane(bytes: Uint8Array) -> {meta, arrays: {name: {data, shape, stride}}}
  Lab.decodePane = function decodePane(bytes) {
    if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
    for (var m = 0; m < 4; m++) {
      if (bytes[m] !== MAGIC[m]) throw new Error('not a Lab pane blob');
    }
    var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    var size = view.getUint32(4, true);
    var meta = JSON.parse(new TextDecoder('utf-8').decode(bytes.subarray(8, 8 + size)));
    var manifest = meta.arrays || {};
    delete meta.arrays;
    var base = 8 + size;
    var arrays = {};
    Object.keys(manifest).forEach(function (name) {
      var info = manifest[name];
      var Ctor = DTYPES[info.dtype];
      if (!Ctor) throw new Error('unknown dtype ' + info.dtype + ' for ' + name);
      var count = product(info.shape);
      var start = base + info.offset;
      if (start + count * Ctor.BYTES_PER_ELEMENT > bytes.byteLength) {
        throw new Error('pane blob is truncated at ' + name);
      }
      var abs = bytes.byteOffset + start;
      var data;
      if (abs % Ctor.BYTES_PER_ELEMENT === 0 && !info.delta) {
        data = new Ctor(bytes.buffer, abs, count); // a view, no copy
      } else {
        data = new Ctor(bytes.slice(start, start + count * Ctor.BYTES_PER_ELEMENT).buffer);
      }
      var frames = info.shape.length ? info.shape[0] : 1;
      var stride = frames ? count / frames : 0;
      if (info.delta) undoDelta(data, frames, stride);
      arrays[name] = { data: data, shape: info.shape.slice(), stride: stride };
    });
    return { meta: meta, arrays: arrays };
  };

  // gunzip(bytes) -> Promise<Uint8Array>, through the browser's DecompressionStream.
  Lab.gunzip = function gunzip(bytes) {
    var DS = typeof DecompressionStream !== 'undefined' ? DecompressionStream : undefined;
    if (!DS) return Promise.reject(new Error('browser too old'));
    return Promise.resolve().then(function () {
      var stream = new Blob([bytes]).stream().pipeThrough(new DS('gzip'));
      return new Response(stream).arrayBuffer();
    }).then(function (buf) { return new Uint8Array(buf); });
  };

  // The frame `f` slice of a decoded array (a subarray view).
  Lab.at = function at(arr, f) {
    return arr.data.subarray(f * arr.stride, (f + 1) * arr.stride);
  };
})();

if (typeof module !== 'undefined') module.exports = Lab;
