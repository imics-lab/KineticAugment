// Data sources for the Lab page. Both implement one interface:
//   status(), schema(profile), profileText(profile), clips(), base(clipId), pane(request),
//   frameUrl(clipId, i), saved(), savedText(name), save(name, text, overwrite),
//   stops(entryId, knobKey) -> number[] | null, addVideo(file), job(jobId), isShare.
// Lab.watchJob(source, jobId, {onUpdate, interval}) polls an add-video job until it ends.
// LiveSource talks to the local server (kinetic-lab); ShareSource reads window.LAB_DATA and the
// blobs that the share zip's data scripts add with Lab.addBlob(sha1, base64).
// Classic script (no modules: they fail from file://); adds to window.Lab and module.exports.
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;
if (!Lab.decodePane && typeof require !== 'undefined') require('./bundle.js');

(function () {
  'use strict';

  // ---- request keys -----------------------------------------------------------------------
  // PaneRequest fields and their defaults (lab/pane.py).
  var REQUEST_DEFAULTS = { values: {}, mode: 'random', reroll: 0, edge: 'high', profile: 'default' };

  // JSON with every object's keys sorted and no spaces, as Python's
  // json.dumps(sort_keys=True, separators=(",", ":")) writes it (numbers in JavaScript's form).
  function canonical(value) {
    if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
    if (value && typeof value === 'object') {
      return '{' + Object.keys(value).sort().map(function (k) {
        return JSON.stringify(k) + ':' + canonical(value[k]);
      }).join(',') + '}';
    }
    return JSON.stringify(value === undefined ? null : value);
  }

  function fullRequest(req) {
    var out = { clip_id: req.clip_id, entry_id: req.entry_id };
    Object.keys(REQUEST_DEFAULTS).forEach(function (k) {
      out[k] = req[k] === undefined || req[k] === null ? REQUEST_DEFAULTS[k] : req[k];
    });
    return out;
  }

  // The canonical key of a request, every PaneRequest field present.
  Lab.requestKey = function requestKey(req) { return canonical(fullRequest(req)); };
  // A key written by Python's PaneRequest.key(), re-serialised the JavaScript way (1.0 -> 1).
  Lab.canonicalKey = function canonicalKey(pythonKey) { return canonical(JSON.parse(pythonKey)); };
  Lab.canonicalJson = canonical;

  // ---- live mode --------------------------------------------------------------------------
  function httpError(res) {
    var make = function (detail) {
      var err = new Error(detail || 'request failed (HTTP ' + res.status + ')');
      err.status = res.status;
      err.detail = detail || null;
      return err;
    };
    return Promise.resolve()
      .then(function () { return res.json(); })
      .then(function (body) {
        var d = body && body.detail;
        return make(typeof d === 'string' ? d : d ? JSON.stringify(d) : null);
      }, function () { return make(null); });
  }

  function LiveSource(baseUrl) {
    this.baseUrl = baseUrl || '';
    this.isShare = false;
  }
  LiveSource.prototype._fetch = function (path, init, kind) {
    return fetch(this.baseUrl + path, init).then(function (res) {
      if (!res.ok) return httpError(res).then(function (err) { throw err; });
      if (kind === 'json') return res.json();
      if (kind === 'text') return res.text();
      return res.arrayBuffer();
    });
  };
  LiveSource.prototype._post = function (path, body, kind) {
    return this._fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }, kind);
  };
  LiveSource.prototype.status = function () { return this._fetch('/api/status', undefined, 'json'); };
  LiveSource.prototype.schema = function (profile) {
    return this._fetch('/api/schema?profile=' + encodeURIComponent(profile || 'default'), undefined, 'json');
  };
  LiveSource.prototype.profileText = function (profile) {
    return this._fetch('/api/profile-text?profile=' + encodeURIComponent(profile || 'default'), undefined, 'text');
  };
  LiveSource.prototype.clips = function () { return this._fetch('/api/clips', undefined, 'json'); };
  LiveSource.prototype.base = function (clipId) {
    return this._fetch('/api/clips/' + encodeURIComponent(clipId) + '/base', undefined, 'bytes')
      .then(function (buf) { return Lab.decodePane(new Uint8Array(buf)); });
  };
  LiveSource.prototype.pane = function (request) {
    return this._post('/api/pane', fullRequest(request), 'bytes')
      .then(function (buf) { return Lab.decodePane(new Uint8Array(buf)); });
  };
  LiveSource.prototype.frameUrl = function (clipId, i) {
    return this.baseUrl + '/api/clips/' + encodeURIComponent(clipId) + '/frames/' + i + '.jpg';
  };
  LiveSource.prototype.saved = function () { return this._fetch('/api/saved', undefined, 'json'); };
  LiveSource.prototype.savedText = function (name) {
    return this._fetch('/api/saved/' + encodeURIComponent(name), undefined, 'text');
  };
  LiveSource.prototype.save = function (name, text, overwrite) {
    return this._post('/api/saved', { name: name, text: text, overwrite: !!overwrite }, 'json');
  };
  LiveSource.prototype.stops = function () { return null; };
  LiveSource.prototype.addVideo = function (file) {
    var form = new FormData();
    form.append('file', file, file.name);
    return this._fetch('/api/videos', { method: 'POST', body: form }, 'json');
  };
  LiveSource.prototype.job = function (jobId) {
    return this._fetch('/api/jobs/' + encodeURIComponent(jobId), undefined, 'json');
  };

  // ---- share mode -------------------------------------------------------------------------
  Lab.blobs = Lab.blobs || {};
  // Called by the share zip's data scripts: Lab.addBlob("<sha1>", "<base64 of the gzipped blob>").
  Lab.addBlob = function addBlob(sha1, base64) { Lab.blobs[sha1] = base64; };

  function fromBase64(text) {
    var bin = atob(text);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  var CACHE_SIZE = 12;

  function ShareSource(data) {
    this.data = data || (typeof window !== 'undefined' ? window.LAB_DATA : undefined) || {};
    this.isShare = true;
    this._cache = new Map(); // sha1 -> Promise<{meta, arrays}>, most recent last
    this._index = {};
    var self = this;
    var panes = this.data.panes || {};
    Object.keys(panes).forEach(function (key) {
      var req = JSON.parse(key);
      self._index[canonical(req)] = panes[key];
      var loose = self._looseKey(req);
      if (!(loose in self._index)) self._index[loose] = panes[key];
    });
  }

  // The share key: what cannot change the pane is ignored (values equal to the schema default;
  // the edge in random mode). The one definition of this rule is PaneRequest.share_key in
  // src/kinetic_augment/lab/pane.py, which writes the manifest keys; keep the two in step.
  ShareSource.prototype._looseKey = function (req) {
    var r = fullRequest(req);
    var entry = ((this.data.schema || {}).entries || []).filter(function (e) { return e.id === r.entry_id; })[0];
    var values = {};
    Object.keys(r.values).forEach(function (k) {
      var knob = entry && entry.knobs.filter(function (kn) { return kn.key === k; })[0];
      if (!knob || canonical(knob.default) !== canonical(r.values[k])) values[k] = r.values[k];
    });
    r.values = values;
    if (r.mode === 'random') r.edge = 'high';
    return canonical(r);
  };

  ShareSource.prototype._blob = function (sha1) {
    var cache = this._cache;
    if (cache.has(sha1)) {
      var hit = cache.get(sha1);
      cache.delete(sha1);
      cache.set(sha1, hit);
      return hit;
    }
    var text = Lab.blobs[sha1];
    if (text === undefined) return Promise.reject(new Error('missing data file for ' + sha1));
    var bytes = fromBase64(text);
    var gz = bytes[0] === 0x1f && bytes[1] === 0x8b;
    var done = (gz ? Lab.gunzip(bytes) : Promise.resolve(bytes)).then(Lab.decodePane);
    done.catch(function () { cache.delete(sha1); });
    cache.set(sha1, done);
    while (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value);
    return done;
  };

  function shareOnly(what) {
    return Promise.reject(new Error(what + ' needs full mode (the local server)'));
  }

  ShareSource.prototype.status = function () { return Promise.resolve({ mode: 'share' }); };
  // The shared version is built from one profile; its schema answers for any profile name.
  ShareSource.prototype.schema = function () {
    var s = this.data.schema;
    if (!s) return Promise.reject(new Error('no schema in the shared data'));
    return Promise.resolve(s);
  };
  ShareSource.prototype.profileText = function () { return Promise.resolve(this.data.profileText || ''); };
  ShareSource.prototype.clips = function () { return Promise.resolve((this.data.clips || []).slice()); };
  ShareSource.prototype.base = function (clipId) {
    var sha1 = (this.data.base || {})[clipId];
    if (!sha1) return Promise.reject(new Error('clip ' + clipId + ' is not in the shared version'));
    return this._blob(sha1);
  };
  // The blob id (sha1) of a request's pane, or null when it was not precomputed.
  ShareSource.prototype.paneId = function (request) {
    return this._index[Lab.requestKey(request)] || this._index[this._looseKey(request)] || null;
  };
  ShareSource.prototype.pane = function (request) {
    var sha1 = this.paneId(request);
    if (!sha1) return Promise.reject(new Error('this setting is not in the shared version'));
    return this._blob(sha1);
  };
  ShareSource.prototype.frameUrl = function (clipId, i) {
    return 'data/' + clipId + '/frames/' + String(i).padStart(4, '0') + '.jpg';
  };
  ShareSource.prototype.saved = function () { return Promise.resolve([]); };
  ShareSource.prototype.savedText = function () { return shareOnly('loading a saved setting by name'); };
  ShareSource.prototype.save = function () { return shareOnly('saving to the settings folder'); };
  ShareSource.prototype.stops = function (entryId, knobKey) {
    var s = (this.data.stops || {})[entryId];
    return s && Array.isArray(s[knobKey]) ? s[knobKey].slice() : null;
  };
  ShareSource.prototype.addVideo = function () { return shareOnly('adding a video'); };
  ShareSource.prototype.job = function () { return shareOnly('a video job'); };

  Lab.LiveSource = LiveSource;
  Lab.ShareSource = ShareSource;

  // Polls source.job(jobId) every `interval` ms (2 s) until the job is done (resolves with its
  // status) or failed (rejects with its error); onUpdate sees every status.
  Lab.watchJob = function watchJob(source, jobId, opts) {
    opts = opts || {};
    var interval = opts.interval || 2000;
    var wait = opts.wait || function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
    function poll() {
      return source.job(jobId).then(function (s) {
        if (opts.onUpdate) opts.onUpdate(s);
        if (s.state === 'done') return s;
        if (s.state === 'failed') throw new Error(s.error || 'the video could not be processed');
        return wait(interval).then(poll);
      });
    }
    return poll();
  };

  // The source for this page: the server when the page came over http and /api/status answers,
  // else the shared data. Resolves to {source, status}.
  Lab.chooseSource = function chooseSource() {
    var share = function () {
      var src = new ShareSource();
      return src.status().then(function (status) { return { source: src, status: status }; });
    };
    var http = typeof location !== 'undefined' && /^http/.test(location.protocol);
    if (!http || typeof fetch === 'undefined') return share();
    var live = new LiveSource('');
    return live.status().then(function (status) {
      return { source: live, status: status };
    }, function () {
      if (typeof window !== 'undefined' && window.LAB_DATA) return share();
      throw new Error('the Lab server is not answering at ' + location.origin);
    });
  };
})();

if (typeof module !== 'undefined') module.exports = Lab;
