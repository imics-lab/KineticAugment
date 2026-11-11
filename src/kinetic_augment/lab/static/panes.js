// The two 3-D panes. Each overlays the original (blue) with its own augmented version (A orange,
// B purple). One WebGL renderer draws both panes into their rectangles of one canvas; both share
// one PerspectiveCamera and one orbit target (drag either pane to orbit both).
// Needs vendor/three.min.js and vendor/OrbitControls.js (THREE, THREE.OrbitControls).
// Classic script: adds Lab.Panes to window.Lab.
var Lab = (typeof window !== 'undefined' ? window : globalThis).Lab || {};
(typeof window !== 'undefined' ? window : globalThis).Lab = Lab;

(function () {
  'use strict';

  // aug_arm rows are MediaPipe pose points 11..16: shoulders, elbows, wrists (left, right).
  var ARM_BONES = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5]];
  // MediaPipe HAND_CONNECTIONS.
  var HAND_BONES = [[0, 1], [0, 5], [0, 17], [1, 2], [2, 3], [3, 4], [5, 6], [5, 9], [6, 7], [7, 8],
    [9, 10], [9, 13], [10, 11], [11, 12], [13, 14], [13, 17], [14, 15], [15, 16], [17, 18], [18, 19],
    [19, 20]];
  // Contours of the 105 embedded face points (rows of aug_face): eyes, brows, lips, nose. They depend
  // only on the face correspondence and the rest mesh, so they are the same for every clip
  // (scripts/export_round_trip_demo.py face_contours, as written into each clip's face105.contours).
  var FACE105_CONTOURS = [
    [36, 49], [49, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 38], [38, 50], [50, 44], [44, 45],
    [45, 46], [46, 47], [47, 48], [48, 51], [51, 37], [37, 36], [34, 22], [22, 27], [27, 26], [26, 25],
    [25, 24], [24, 23], [23, 33], [33, 20], [20, 21], [21, 35], [35, 32], [32, 31], [31, 30], [30, 29],
    [29, 28], [28, 34],
    [17, 12], [12, 10], [10, 13], [13, 15], [15, 19], [19, 16], [16, 18], [18, 11], [11, 14], [14, 17],
    [5, 3], [3, 0], [0, 2], [2, 7], [7, 4], [4, 1], [1, 8], [8, 6], [6, 9], [9, 5],
    [72, 85], [85, 71], [71, 70], [70, 69], [69, 65], [65, 87], [87, 88], [88, 89], [89, 103], [103, 90],
    [90, 100], [100, 98], [98, 102], [102, 95], [95, 68], [68, 77], [77, 84], [84, 80], [80, 82], [82, 72],
    [73, 86], [86, 74], [74, 75], [75, 76], [76, 66], [66, 94], [94, 93], [93, 92], [92, 104], [104, 91],
    [91, 99], [99, 97], [97, 101], [101, 96], [96, 67], [67, 78], [78, 83], [83, 79], [79, 81], [81, 73],
    [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [58, 59], [59, 60], [60, 61], [61, 62], [62, 63],
    [63, 64],
  ];
  // Dots are drawn in screen pixels, a little larger in the close-up (hands, face) cameras.
  var DOT_PX = { wide: 3.5, close: 6 };
  var SKIN = '#c9a58f';
  var TOGGLE_KEYS = ['video', 'original', 'augmented', 'mesh', 'points', 'skeleton', 'face', 'hands'];

  var css = function (name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  };
  var finite3 = function (p, i) {
    return Number.isFinite(p[3 * i]) && Number.isFinite(p[3 * i + 1]) && Number.isFinite(p[3 * i + 2]);
  };
  // MediaPipe / camera axes (y down, z away) to three.js (y up, z towards the viewer).
  var put = function (out, o, p, i) { out[o] = p[3 * i]; out[o + 1] = -p[3 * i + 1]; out[o + 2] = -p[3 * i + 2]; };
  var at = function (arr, f) { return arr.data.subarray(f * arr.stride, (f + 1) * arr.stride); };

  // ---- geometry buffers ---------------------------------------------------------------------
  function LineBuf(pairs) {
    this.pairs = pairs;
    this.pos = new Float32Array(Math.max(1, pairs.length) * 6);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geometry.setDrawRange(0, 0);
  }
  // Draw every bone whose two ends exist; skip the rest.
  LineBuf.prototype.fill = function (p) {
    var n = 0;
    if (p) {
      for (var s = 0; s < this.pairs.length; s++) {
        var a = this.pairs[s][0], b = this.pairs[s][1];
        if (!finite3(p, a) || !finite3(p, b)) continue;
        put(this.pos, n * 6, p, a);
        put(this.pos, n * 6 + 3, p, b);
        n++;
      }
    }
    this.geometry.setDrawRange(0, n * 2);
    this.geometry.attributes.position.needsUpdate = true;
  };

  function DotBuf(count) {
    this.count = count;
    this.pos = new Float32Array(Math.max(1, count) * 3);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geometry.setDrawRange(0, 0);
  }
  DotBuf.prototype.fill = function (p, first, count) {
    var n = 0;
    if (p) {
      for (var i = first; i < first + count; i++) {
        if (!finite3(p, i)) continue;
        put(this.pos, n * 3, p, i);
        n++;
      }
    }
    this.geometry.setDrawRange(0, n);
    this.geometry.attributes.position.needsUpdate = true;
  };

  // Hand pairs for both hands in one (2 x 21) array.
  var BOTH_HANDS = HAND_BONES.concat(HAND_BONES.map(function (b) { return [b[0] + 21, b[1] + 21]; }));

  // One stream's buffers (the original, or a pane's augmented version): mesh, lines and dots.
  function Figure(source, faces, nVerts) {
    this.source = source; // {meta, arrays}
    var verts = source.arrays.mesh_verts;
    this.hasMesh = !!(verts && faces && verts.shape[1] === nVerts);
    this.mesh = new THREE.BufferGeometry();
    if (this.hasMesh) {
      this.mesh.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nVerts * 3), 3));
      var index = new Uint16Array(faces); // copy: a mirror reverses the winding to keep fronts outward
      if (source.meta.mirrored) {
        for (var i = 0; i < index.length; i += 3) { var x = index[i + 1]; index[i + 1] = index[i + 2]; index[i + 2] = x; }
      }
      this.mesh.setIndex(new THREE.BufferAttribute(index, 1));
    }
    var face = source.arrays.aug_face;
    this.nFace = face ? face.stride / 3 : 0;
    this.lines = {
      arm: new LineBuf(ARM_BONES),
      hands: new LineBuf(BOTH_HANDS),
      face: new LineBuf(this.nFace === 105 ? FACE105_CONTOURS : []),
    };
    this.dots = { arm: new DotBuf(6), hands: new DotBuf(42), face: new DotBuf(this.nFace) };
    this.ok = false;
    this.frame = -1;
  }
  Figure.prototype.update = function (f) {
    var src = this.source, n = src.meta.n;
    f = Math.min(Math.max(f, 0), n - 1);
    this.frame = f;
    var okList = src.meta.body_ok;
    this.ok = okList ? !!okList[f] : true;
    if (this.hasMesh) {
      var v = at(src.arrays.mesh_verts, f), pos = this.mesh.attributes.position.array;
      var s = 1 / (src.meta.scale_i16 || 10000);
      for (var i = 0; i < v.length; i += 3) { pos[i] = v[i] * s; pos[i + 1] = -v[i + 1] * s; pos[i + 2] = -v[i + 2] * s; }
      this.mesh.attributes.position.needsUpdate = true;
    }
    var arm = src.arrays.aug_arm ? at(src.arrays.aug_arm, f) : null;
    var hands = src.arrays.aug_hands ? at(src.arrays.aug_hands, f) : null;
    var face = src.arrays.aug_face ? at(src.arrays.aug_face, f) : null;
    // Only the mesh needs the body fit (`ok`); the points come from the pose, hand and face
    // outputs, so whatever is finite is drawn.
    this.lines.arm.fill(arm);
    this.lines.hands.fill(hands);
    this.lines.face.fill(face);
    this.dots.arm.fill(arm, 0, 6);
    this.dots.hands.fill(hands, 0, 42);
    this.dots.face.fill(face, 0, this.nFace);
  };
  Figure.prototype.dispose = function () {
    this.mesh.dispose();
    var self = this;
    ['arm', 'hands', 'face'].forEach(function (k) { self.lines[k].geometry.dispose(); self.dots[k].geometry.dispose(); });
  };

  // The pane colour mixed into the skin tone: the augmented body reads as solid and tinted.
  function tinted(color) { return new THREE.Color(SKIN).lerp(new THREE.Color(color), 0.4); }

  // The scene objects drawing one Figure. `style` 'ghost' (the original): a faint see-through
  // mesh pushed slightly behind, so it shows only where it sticks out of the augmented body;
  // 'solid' (a pane's augmented version): an opaque skin-tone mesh tinted toward the pane colour.
  // Lines and dots are drawn on top of every mesh, in the stream's own colour.
  function Drawn(scene, fig, color, style, order) {
    this.scene = scene;
    this.fig = fig;
    this.style = style;
    this.meshMat = style === 'solid'
      ? new THREE.MeshStandardMaterial({ color: tinted(color), roughness: 0.85, flatShading: true })
      : new THREE.MeshStandardMaterial({
        color: color, transparent: true, opacity: 0.2, depthWrite: false, roughness: 0.9, flatShading: true,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      });
    var overlay = { color: color, transparent: true, depthTest: false, depthWrite: false };
    this.lineMat = new THREE.LineBasicMaterial(overlay);
    this.dotMat = new THREE.PointsMaterial(Object.assign({ size: DOT_PX.wide, sizeAttenuation: false }, overlay));
    this.mesh = new THREE.Mesh(fig.mesh, this.meshMat);
    this.lines = {}; this.dots = {};
    var self = this, objs = [this.mesh];
    ['arm', 'hands', 'face'].forEach(function (k) {
      self.lines[k] = new THREE.LineSegments(fig.lines[k].geometry, self.lineMat);
      self.dots[k] = new THREE.Points(fig.dots[k].geometry, self.dotMat);
      objs.push(self.lines[k], self.dots[k]);
    });
    // a fixed order: meshes (the solid one first, then the ghost), then lines and dots on top
    objs.forEach(function (o, i) { o.frustumCulled = false; o.renderOrder = order + (i === 0 ? 0 : 10); scene.add(o); });
    this.objs = objs;
  }
  // `alone`: the other stream's mesh is hidden, so a ghost is drawn stronger to stay readable.
  Drawn.prototype.visible = function (on, t, alone) {
    var ok = this.fig.ok;
    this.mesh.visible = on && t.mesh && ok && this.fig.hasMesh;
    if (this.style === 'ghost') this.meshMat.opacity = alone ? 0.5 : 0.2;
    var region = { arm: true, hands: t.hands, face: t.face };
    var self = this;
    ['arm', 'hands', 'face'].forEach(function (k) {
      self.lines[k].visible = on && t.skeleton && region[k];
      self.dots[k].visible = on && t.points && region[k];
    });
  };
  Drawn.prototype.dotSize = function (px) { this.dotMat.size = px; };
  Drawn.prototype.remove = function () {
    var self = this;
    this.objs.forEach(function (o) { self.scene.remove(o); });
    this.meshMat.dispose(); this.lineMat.dispose(); this.dotMat.dispose();
  };

  function makeScene() {
    var scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.5));
    var sun = new THREE.DirectionalLight(0xffffff, 1.4);
    scene.add(sun, sun.target);
    scene.userData.sun = sun;
    return scene;
  }

  // ---- the panes ----------------------------------------------------------------------------
  // Lab.Panes(container, {onCameraChange}).
  function Panes(container, opts) {
    opts = opts || {};
    var self = this;
    this.container = container;
    this.onCameraChange = opts.onCameraChange || function () {};
    this.colors = { orig: css('--orig', '#4a8fe7'), A: css('--pane-a', '#e8913a'), B: css('--pane-b', '#b56cf0') };
    this.toggles = { video: true, original: true, augmented: true, mesh: true, points: true, skeleton: true, face: true, hands: true };
    this.dotPx = DOT_PX.wide;
    this.base = null; this.baseFig = null;
    this.frames = { A: 0, B: 0, base: 0 };
    this.ended = { A: false, B: false };
    container.classList.add('panes');
    container.innerHTML = '';
    this.cells = {};
    ['A', 'B'].forEach(function (w) {
      var cell = document.createElement('figure');
      cell.className = 'pane pane-' + w.toLowerCase();
      cell.dataset.pane = w;
      cell.innerHTML = '<figcaption><span class="badge">' + w + '</span><span class="pane-title"></span>' +
        '<span class="pane-frame mono"></span></figcaption><div class="vp"><div class="pane-empty">no pane yet</div>' +
        '<div class="pane-busy" hidden>computing…</div></div><div class="pane-note" hidden></div>';
      container.appendChild(cell);
      self.cells[w] = {
        el: cell, vp: cell.querySelector('.vp'), title: cell.querySelector('.pane-title'),
        frame: cell.querySelector('.pane-frame'), note: cell.querySelector('.pane-note'),
        busy: cell.querySelector('.pane-busy'), empty: cell.querySelector('.pane-empty'),
      };
    });
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'panes-gl';
    container.appendChild(this.canvas);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setClearColor(0x000000, 0);
    this.camera = new THREE.PerspectiveCamera(28, 3 / 4, 0.01, 20);
    this.camera.position.set(0, 0, 2.4);
    this.target = new THREE.Vector3();
    this.scenes = { A: makeScene(), B: makeScene() };
    this.drawn = { A: { orig: null, aug: null }, B: { orig: null, aug: null } };
    this.figs = { A: null, B: null };
    this.controls = ['A', 'B'].map(function (w) {
      var c = new THREE.OrbitControls(self.camera, self.cells[w].vp);
      c.target = self.target;
      c.addEventListener('change', function () { self.requestRender(); self.onCameraChange(self.camera); });
      return c;
    });
    this._pending = false;
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(function () { self.requestRender(); }).observe(container);
    window.addEventListener('resize', function () { self.requestRender(); });
  }

  Panes.prototype.requestRender = function () {
    if (this._pending) return;
    var self = this;
    this._pending = true;
    requestAnimationFrame(function () { self._pending = false; self.render(); });
  };

  Panes.prototype.render = function () {
    var r = this.renderer, box = this.container.getBoundingClientRect();
    var w = Math.max(1, Math.round(box.width)), h = Math.max(1, Math.round(box.height));
    var size = r.getSize(new THREE.Vector2());
    if (size.x !== w || size.y !== h) r.setSize(w, h, false);
    r.setScissorTest(false);
    r.clear();
    r.setScissorTest(true);
    var self = this;
    ['A', 'B'].forEach(function (which) {
      var vr = self.cells[which].vp.getBoundingClientRect();
      if (vr.width < 2 || vr.height < 2) return;
      var left = vr.left - box.left, bottom = box.bottom - vr.bottom;
      self.camera.aspect = vr.width / vr.height;
      self.camera.updateProjectionMatrix();
      r.setViewport(left, bottom, vr.width, vr.height);
      r.setScissor(left, bottom, vr.width, vr.height);
      var sun = self.scenes[which].userData.sun;
      sun.position.copy(self.camera.position).add(new THREE.Vector3(0.4, 0.8, 0));
      sun.target.position.copy(self.target);
      r.render(self.scenes[which], self.camera);
    });
  };

  // The original (base blob): its mesh faces serve both panes. Clears both panes.
  Panes.prototype.setBase = function (base) {
    var self = this;
    var sameClip = this.base && base && this.base.meta.clip === base.meta.clip;
    ['A', 'B'].forEach(function (w) {
      self.setPane(w, null);
      if (self.drawn[w].orig) { self.drawn[w].orig.remove(); self.drawn[w].orig = null; }
    });
    if (this.baseFig) this.baseFig.dispose();
    this.base = base;
    this.baseFig = null;
    if (!base) { this.requestRender(); return; }
    var verts = base.arrays.mesh_verts, faces = base.arrays.mesh_faces;
    this.faces = faces ? faces.data : null;
    this.nVerts = verts ? verts.shape[1] : 0;
    this.baseFig = new Figure(base, this.faces, this.nVerts);
    ['A', 'B'].forEach(function (w) {
      self.drawn[w].orig = new Drawn(self.scenes[w], self.baseFig, self.colors.orig, 'ghost', 1);
      self.drawn[w].orig.dotSize(self.dotPx);
    });
    this.frames.base = 0;
    this.baseFig.update(0);
    if (!sameClip) this.cameraPreset('front');
    this._apply();
  };

  // A pane's augmented version (a decoded pane blob), or null to clear it.
  Panes.prototype.setPane = function (which, pane) {
    var d = this.drawn[which];
    if (d.aug) { d.aug.remove(); d.aug = null; }
    if (this.figs[which]) { this.figs[which].dispose(); this.figs[which] = null; }
    this.cells[which].empty.hidden = !!pane;
    if (pane && this.base) {
      var fig = new Figure(pane, this.faces, this.nVerts);
      this.figs[which] = fig;
      d.aug = new Drawn(this.scenes[which], fig, this.colors[which], 'solid', 0);
      d.aug.dotSize(this.dotPx);
      fig.update(Math.min(this.frames[which], pane.meta.n - 1));
    }
    this._apply();
  };

  // Show pane A's frame fA, pane B's frame fB and the original's frame fBase.
  // `ended` ({A, B}) marks a pane that has run out of frames on the playback clock.
  Panes.prototype.showFrame = function (fA, fB, fBase, ended) {
    this.frames = { A: fA, B: fB, base: fBase };
    this.ended = { A: !!(ended && ended.A), B: !!(ended && ended.B) };
    if (this.baseFig) this.baseFig.update(fBase);
    if (this.figs.A) this.figs.A.update(fA);
    if (this.figs.B) this.figs.B.update(fB);
    this._apply();
  };

  Panes.prototype._apply = function () {
    var t = this.toggles, self = this;
    var baseTimes = this.base && this.base.arrays.times ? this.base.arrays.times.data : null;
    ['A', 'B'].forEach(function (w) {
      var d = self.drawn[w], fig = self.figs[w], cell = self.cells[w];
      var augMesh = !!(d.aug && t.augmented && t.mesh && fig && fig.ok && fig.hasMesh);
      if (d.orig) d.orig.visible(t.original, t, !augMesh);
      if (d.aug) d.aug.visible(t.augmented, t);
      cell.el.classList.toggle('ended', !!(fig && self.ended[w]));
      if (fig) {
        var k = fig.frame, n = fig.source.meta.n, times = fig.source.arrays.times;
        var src = times && baseTimes ? ' ← source ' + (times.data[k] - baseTimes[0] + 1).toFixed(1) : '';
        cell.frame.textContent = (k + 1) + ' / ' + n + src + (self.ended[w] ? ' · ended' : '') +
          (fig.ok ? '' : ' · no body this frame');
      } else cell.frame.textContent = '';
    });
    this.requestRender();
  };

  Panes.prototype.setToggles = function (toggles) {
    var self = this;
    TOGGLE_KEYS.forEach(function (k) { if (toggles && k in toggles) self.toggles[k] = !!toggles[k]; });
    this._apply();
  };

  // Camera presets, as the round-trip pages have them: front / side frame the upper body over the
  // whole clip; hands / face close in on the original's current frame.
  Panes.prototype.cameraPreset = function (name) {
    if (!this.base) return;
    var a = this.base.arrays, n = this.base.meta.n, ok = this.base.meta.body_ok || [];
    var sum = new THREE.Vector3(), count = 0, d = 2.4;
    var add = function (p, i) {
      if (!finite3(p, i)) return;
      sum.x += p[3 * i]; sum.y -= p[3 * i + 1]; sum.z -= p[3 * i + 2]; count++;
    };
    var fb = Math.min(this.frames.base, n - 1);
    if (name === 'hands' && a.aug_hands) {
      var hands = at(a.aug_hands, fb);
      for (var i = 0; i < 42; i++) add(hands, i);
      d = 0.8;
    } else if (name === 'face' && a.aug_face) {
      var face = at(a.aug_face, fb);
      for (var j = 0; j < face.length / 3; j++) add(face, j);
      d = 0.5;
    }
    if (!count) { // front / side, or nothing to aim at: shoulders and face centre over the clip
      d = 2.4;
      for (var f = 0; f < n; f++) {
        if (ok.length && !ok[f]) continue;
        if (a.aug_arm) { var arm = at(a.aug_arm, f); add(arm, 0); add(arm, 1); }
        if (a.aug_face) {
          var fc = at(a.aug_face, f), c = new Float32Array(3), m = 0;
          for (var q = 0; q < fc.length / 3; q++) {
            if (!finite3(fc, q)) continue;
            c[0] += fc[3 * q]; c[1] += fc[3 * q + 1]; c[2] += fc[3 * q + 2]; m++;
          }
          if (m) { c[0] /= m; c[1] /= m; c[2] /= m; add(c, 0); }
        }
      }
    }
    if (count) this.target.copy(sum.divideScalar(count));
    this.setDotSize(d === 2.4 ? DOT_PX.wide : DOT_PX.close);
    if (d === 2.4) this.target.y -= 0.15;
    var offset = name === 'side' ? new THREE.Vector3(-d, 0, 0) : new THREE.Vector3(0, 0, d);
    this.camera.position.copy(this.target).add(offset);
    this.camera.lookAt(this.target);
    this.controls.forEach(function (c) { c.update(); });
    this.requestRender();
  };

  // Dot size in screen pixels for every stream (camera presets set it).
  Panes.prototype.setDotSize = function (px) {
    var self = this;
    this.dotPx = px;
    ['A', 'B'].forEach(function (w) {
      if (self.drawn[w].orig) self.drawn[w].orig.dotSize(px);
      if (self.drawn[w].aug) self.drawn[w].aug.dotSize(px);
    });
    this.requestRender();
  };

  Panes.prototype.setTitle = function (which, text) { this.cells[which].title.textContent = text || ''; };
  // A message under the pane (an error such as a 422 detail), or null to hide it.
  Panes.prototype.setNote = function (which, text) {
    var note = this.cells[which].note;
    note.textContent = text || '';
    note.hidden = !text;
  };
  Panes.prototype.setBusy = function (which, busy) { this.cells[which].busy.hidden = !busy; };

  Lab.Panes = Panes;
  Lab.PANE_TOGGLES = TOGGLE_KEYS;
  Lab.Panes.Figure = Figure; // exposed for the Node tests
  Lab.Panes.Drawn = Drawn;
})();

if (typeof module !== 'undefined') module.exports = Lab;
