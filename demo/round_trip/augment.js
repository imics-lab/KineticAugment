// Augmentation viewer: reads clips/<id>/{frames/, v/off/, aug/<augmentation>/} (each a
// clip.json + data.bin) written by scripts/export_round_trip_demo.py. Three views share one
// camera: the unaugmented round trip, it and the augmentation one over the other, and the
// augmentation alone. An augmented frame shows the source frame in its `times` (a duration
// change re-times the clip).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { orig: css('--orig'), aug: css('--aug') };
const ARM_BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const SKIN = '#c9a58f';

let index = null, clip = null, A = {}, aug = null, G = {}, augName = null, images = [], t = 0, playing = false;

// ---------- data ----------
function view(buf, m) {
  const Ctor = { float32: Float32Array, int16: Int16Array, uint16: Uint16Array }[m.dtype];
  const size = m.shape.reduce((a, b) => a * b, 1);
  return { data: new Ctor(buf, m.offset, size), shape: m.shape, stride: size / m.shape[0] };
}
const at = (src, name, f) => { const a = src[name]; return a.data.subarray(f * a.stride, (f + 1) * a.stride); };
const ok3 = (p, i) => Number.isFinite(p[3 * i]) && Number.isFinite(p[3 * i + 1]) && Number.isFinite(p[3 * i + 2]);
const fin = (v) => v !== null && v !== undefined && Number.isFinite(v);
const fmt = (v, d = 1) => (fin(v) ? v.toFixed(d) : '–');
const loadImage = (src) => new Promise((res, rej) => {
  const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src;
});
async function fetchBundle(dir) {
  const [meta, buf] = await Promise.all([
    fetch(`${dir}/clip.json`).then((r) => r.json()),
    fetch(`${dir}/data.bin`).then((r) => r.arrayBuffer()),
  ]);
  const arrays = {};
  for (const [name, m] of Object.entries(meta.arrays)) arrays[name] = view(buf, m);
  return { meta, arrays };
}
const srcFrame = (k) => Math.min(Math.max(Math.round(G.times.data[k]), 0), clip.n - 1);
function augFrameFor(f) { // the augmented frame nearest source frame f
  let best = 0;
  for (let k = 1; k < aug.n; k++) if (Math.abs(G.times.data[k] - f) < Math.abs(G.times.data[best] - f)) best = k;
  return best;
}

async function loadIndex() {
  index = await fetch('clips.json').then((r) => r.json());
  const ids = Object.keys(index.clips).filter((id) => index.clips[id].augmentations);
  const ordered = [...index.order.filter((id) => ids.includes(id)), ...ids.filter((id) => !index.order.includes(id))];
  const select = $('clip');
  select.innerHTML = '';
  for (const id of ordered) select.add(new Option(id, id));
  if (!ordered.length) {
    $('summary').innerHTML = '<span class="chip">no clip has exported augmentations: run export_round_trip_demo.py</span>';
    return;
  }
  const h = new URLSearchParams(location.hash.slice(1));
  select.value = ordered.includes(h.get('clip')) ? h.get('clip') : ordered[0];
  augName = h.get('a');
  select.onchange = () => loadClip(select.value, 0);
  $('aug').onchange = () => { augName = $('aug').value; loadClip(clip.clip, srcFrame(t)); };
  await loadClip(select.value, Number(h.get('f') || 1) - 1);
}

async function loadClip(id, startFrame = 0) {
  playing = false; $('play').textContent = '▶';
  const exported = index.clips[id].augmentations;
  const names = Object.keys(index.augmentation).filter((a) => exported[a]);
  if (!names.includes(augName)) augName = names.includes('combined') ? 'combined' : names[0];
  const sel = $('aug');
  sel.innerHTML = '';
  for (const a of names) {
    const o = new Option(index.augmentation[a].label, a);
    o.title = `${exported[a].title} · ${index.augmentation[a].description}`;
    sel.add(o);
  }
  sel.value = augName;
  const first = !clip || clip.clip !== id;
  const [b, g] = await Promise.all([
    first ? fetchBundle(`clips/${id}/v/off`) : { meta: clip, arrays: A },
    fetchBundle(`clips/${id}/aug/${augName}`),
  ]);
  clip = b.meta; A = b.arrays; aug = g.meta; G = g.arrays;
  if (first) {
    images = await Promise.all(Array.from({ length: clip.n },
      (_, i) => loadImage(`clips/${id}/frames/${String(i).padStart(4, '0')}.jpg`)));
    const cv = $('raw');
    cv.width = clip.width; cv.height = clip.height; cv.style.aspectRatio = `${clip.width} / ${clip.height}`;
  }
  build3d();
  if (first) resetCamera('front');
  $('scrub').max = aug.n - 1;
  summaryChips();
  show(augFrameFor(Math.min(Math.max(startFrame, 0), clip.n - 1)));
}

function summaryChips() {
  const d = aug.draw;
  $('summary').innerHTML = [
    `<span class="chip" title="${aug.description}"><b>${aug.title}</b></span>`,
    `<span class="chip">frames <b>${clip.n}</b>${aug.n !== clip.n ? ` → <b>${aug.n}</b>` : ''}</span>`,
    `<span class="chip" title="the clip's Tier 1 draw; 0 is the clip's own">draw <b>${aug.sample || 0}</b>${d.mirror ? ' · mirrors' : ''}</span>`,
  ].join('');
}

// ---------- 3-D ----------
// MediaPipe world (x right, y down, z away) -> three.js (y up, camera on +z)
const W3 = (p, i, out = new THREE.Vector3()) => out.set(p[3 * i], -p[3 * i + 1], -p[3 * i + 2]);
const camera = new THREE.PerspectiveCamera(28, 3 / 4, 0.01, 20);
const views = {};

function makeView(el) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);
  const note = document.createElement('div'); note.className = 'note'; el.appendChild(note);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#0b0d11');
  scene.add(new THREE.HemisphereLight('#ffffff', '#30343c', 1.6));
  const sun = new THREE.DirectionalLight('#ffffff', 1.6); scene.add(sun); // follows the camera
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener('change', renderAll);
  return { el, renderer, scene, controls, note, sun, objects: {} };
}

class Dots {
  constructor(n, radius, color) {
    this.mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(radius, 10, 8),
      new THREE.MeshBasicMaterial({ color }), n);
    this.m = new THREE.Matrix4();
  }
  set(i, v) {
    if (v) this.m.makeTranslation(v.x, v.y, v.z); else this.m.makeScale(0, 0, 0);
    this.mesh.setMatrixAt(i, this.m);
  }
  done() { this.mesh.instanceMatrix.needsUpdate = true; }
}

class Segs {
  constructor(n, color) {
    this.pos = new Float32Array(n * 6);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.mesh = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
    this.mesh.frustumCulled = false; this.i = 0; this.n = n;
  }
  reset() { this.i = 0; this.pos.fill(0); }
  add(a, b) {
    if (!a || !b || this.i >= this.n) return;
    this.pos.set([a.x, a.y, a.z, b.x, b.y, b.z], this.i * 6); this.i++;
  }
  done() { this.mesh.geometry.attributes.position.needsUpdate = true; }
}

// One figure: a mesh and its read-back skeleton (arms, hands, face contours) and points.
function figure(geomRef, color, material) {
  const n105 = clip.face105.mp_indices.length;
  const nContour = Object.values(clip.face105.contours).reduce((s, l) => s + l.length, 0);
  return {
    mesh: new THREE.Mesh(geomRef, material),
    bones: new Segs(5 + 42 + nContour, color),
    dots: new Dots(6 + 42 + n105, 0.0055, color),
  };
}

function meshGeometry(faces, mirrored) {
  const nv = A.mesh_verts.shape[1];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
  const index = new Uint16Array(faces);
  // a mirror reverses the triangles' winding; reversing the index keeps the normals outward
  if (mirrored) for (let i = 0; i < index.length; i += 3) { const x = index[i + 1]; index[i + 1] = index[i + 2]; index[i + 2] = x; }
  g.setIndex(new THREE.BufferAttribute(index, 1));
  return g;
}

let geoms = {}, figs = {};
function build3d() {
  if (!views.orig) {
    views.orig = makeView($('view-orig')); views.both = makeView($('view-both')); views.aug = makeView($('view-aug'));
    views.both.controls.target = views.aug.controls.target = views.orig.controls.target;
  }
  for (const v of Object.values(views)) for (const o of [...v.scene.children]) if (o.userData.clip) v.scene.remove(o);
  geoms = { orig: meshGeometry(A.mesh_faces.data, false), aug: meshGeometry(A.mesh_faces.data, aug.mirrored) };
  const solid = () => new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.85 });
  const ghost = (color, opacity) => new THREE.MeshStandardMaterial({
    color, transparent: true, opacity, depthWrite: false, roughness: 0.9 });
  figs = {
    orig: figure(geoms.orig, C.orig, solid()),
    bothOrig: figure(geoms.orig, C.orig, ghost(C.orig, 0.28)),
    bothAug: figure(geoms.aug, C.aug, ghost(C.aug, 0.32)),
    aug: figure(geoms.aug, C.aug, solid()),
  };
  const put = (v, fig) => { for (const o of [fig.mesh, fig.bones.mesh, fig.dots.mesh]) { o.userData.clip = true; v.scene.add(o); } };
  put(views.orig, figs.orig); put(views.both, figs.bothOrig); put(views.both, figs.bothAug); put(views.aug, figs.aug);
}

function setMesh(g, verts, scale) {
  const pos = g.attributes.position.array;
  for (let i = 0; i < verts.length; i += 3) {
    pos[i] = verts[i] * scale; pos[i + 1] = -verts[i + 1] * scale; pos[i + 2] = -verts[i + 2] * scale;
  }
  g.attributes.position.needsUpdate = true;
  g.computeVertexNormals();
  g.computeBoundingSphere();
}

// Fill a figure's skeleton and points from (arm 6, hands 2x21, face 105) arrays.
function pose(fig, arm, hands, face, ok) {
  const p3 = (arr, i) => (ok && ok3(arr, i) ? W3(arr, i) : null);
  const armIdx = (m) => clip.arm_mp.indexOf(m);
  fig.bones.reset();
  for (const [a, b] of ARM_BONES) fig.bones.add(p3(arm, armIdx(a)), p3(arm, armIdx(b)));
  let n = 0;
  for (let i = 0; i < 6; i++) fig.dots.set(n++, p3(arm, i));
  for (let s = 0; s < 2; s++) {
    for (const [a, b] of clip.connections.hand) fig.bones.add(p3(hands, s * 21 + a), p3(hands, s * 21 + b));
    for (let j = 0; j < 21; j++) fig.dots.set(n++, p3(hands, s * 21 + j));
  }
  for (const pairs of Object.values(clip.face105.contours)) for (const [a, b] of pairs) fig.bones.add(p3(face, a), p3(face, b));
  for (let r = 0; r < clip.face105.mp_indices.length; r++) fig.dots.set(n++, p3(face, r));
  fig.bones.done(); fig.dots.done();
}

function update3d(f, k) {
  const okO = !!clip.flags.body_ok[f], okA = !!aug.body_ok[k];
  setMesh(geoms.orig, at(A, 'mesh_verts', f), 1 / clip.scale_i16);
  setMesh(geoms.aug, at(G, 'mesh_verts', k), 1 / aug.scale_i16);
  const orig = [at(A, 'rt_arm', f), at(A, 'rt_hands', f), at(A, 'rt_face', f), okO];
  const augm = [at(G, 'aug_arm', k), at(G, 'aug_hands', k), at(G, 'aug_face', k), okA];
  pose(figs.orig, ...orig); pose(figs.bothOrig, ...orig);
  pose(figs.aug, ...augm); pose(figs.bothAug, ...augm);
  const vis = (fig, mesh, pts, ok) => {
    fig.mesh.visible = mesh && ok; fig.bones.mesh.visible = pts; fig.dots.mesh.visible = pts;
  };
  vis(figs.orig, $('t-mesh1').checked, $('t-pts1').checked, okO);
  vis(figs.bothOrig, $('t-mesh2').checked, $('t-pts2').checked, okO);
  vis(figs.bothAug, $('t-mesh2').checked, $('t-pts2').checked, okA);
  vis(figs.aug, $('t-mesh3').checked, $('t-pts3').checked, okA);
  const note = (v, msg) => { v.note.textContent = msg; v.note.style.display = msg ? '' : 'none'; };
  note(views.orig, okO ? '' : 'no SMPL-X fit this frame');
  note(views.both, okO && okA ? '' : 'no SMPL-X fit this frame');
  note(views.aug, okA ? '' : 'no augmented body this frame');
}

function resetCamera(which) {
  // frame the upper body: shoulders to head, from the clip's median pose
  const pts = [];
  for (let f = 0; f < clip.n; f++) {
    const w = at(A, 'w_pose', f), v = at(A, 'pose_vis', f);
    for (const i of [0, 11, 12]) if (v[i] >= 0.5 && ok3(w, i)) pts.push(W3(w, i));
  }
  const c = new THREE.Vector3();
  pts.forEach((p) => c.add(p)); if (pts.length) c.divideScalar(pts.length);
  c.y -= 0.15;
  const d = 2.4;
  views.orig.controls.target.copy(c);
  camera.position.copy(c).add(which === 'side' ? new THREE.Vector3(-d, 0, 0) : new THREE.Vector3(0, 0, d));
  camera.lookAt(c);
  for (const v of Object.values(views)) v.controls.update();
  renderAll();
}

function resize() {
  for (const v of Object.values(views)) {
    const { clientWidth: w, clientHeight: h } = v.el;
    v.renderer.setSize(w, h, false);
    v.renderer.domElement.style.width = '100%'; v.renderer.domElement.style.height = '100%';
  }
  const v = views.orig; if (v) { camera.aspect = v.el.clientWidth / v.el.clientHeight; camera.updateProjectionMatrix(); }
  renderAll();
}
function renderAll() {
  for (const v of Object.values(views)) {
    v.sun.position.copy(camera.position).add(new THREE.Vector3(0.4, 0.8, 0)); v.sun.target.position.copy(v.controls.target);
    v.sun.target.updateMatrixWorld();
    v.renderer.render(v.scene, camera);
  }
}

// ---------- readout ----------
function readout(f, k) {
  const m = aug.metrics, d = aug.draw;
  const row = (label, v) => `<tr><td>${label}</td><td>${v}</td></tr>`;
  const joints = d.arm_joints.length
    ? d.arm_joints.map((j) => `${j.side[0].toUpperCase()} ${j.joint} ${j.angle_deg.toFixed(1)}°`).join(', ') : 'none';
  const moved = fin(m.wrist_moved_mm_median) ? `${fmt(m.wrist_moved_mm_median, 0)} / ${fmt(m.wrist_moved_mm_max, 0)}` : '–';
  $('readout').innerHTML = `<table>
    <tr><th colspan="2">${aug.title}</th></tr>
    <tr><td colspan="2" style="color:#8b93a7;white-space:normal">${aug.description}</td></tr>
    ${row('frame', `${k + 1} / ${aug.n} ← source ${fmt(G.times.data[k] + 1, 2)} / ${clip.n}`)}
    ${row('wrists moved, median / max (mm)', moved)}
    ${row('hand points in frame', `${fmt(m.hand_points_in_frame_pct)}% (original ${fmt(m.round_trip_hand_points_in_frame_pct)}%)`)}
    ${row('zero strength reproduces the original', `max diff ${m.identity_max_abs_diff_m.toExponential(1)} m`)}
    <tr><th colspan="2">the clip's draw${aug.sample ? ` (draw ${aug.sample}; the clip's own is 0)` : ''}</th></tr>
    ${row('view', `yaw ${d.yaw_deg.toFixed(1)}°, pitch ${d.pitch_deg.toFixed(1)}°`)}
    ${row('mirror', d.mirror ? 'yes' : 'no (the mirror view always mirrors)')}
    ${row('duration', `×${d.duration.toFixed(3)}`)}
    ${row('arm joints', joints)}
    ${row('scale', `×${d.scale.toFixed(3)}`)}
    ${row('uncentre shift (frame)', `(${aug.uncentre_shift.map((v) => v.toFixed(3)).join(', ')})`)}
    ${row('camera fit residual', `${fmt(aug.camera.residual_frame_pct)}% of the frame`)}
  </table>`;
}

// ---------- frame / playback ----------
function show(k) {
  t = k;
  const f = srcFrame(k);
  $('scrub').value = k;
  $('frame-label').textContent = `frame ${k + 1} / ${aug.n} (source ${f + 1})`;
  $('raw').getContext('2d').drawImage(images[f], 0, 0);
  update3d(f, k); renderAll(); readout(f, k);
  history.replaceState(null, '', `#clip=${clip.clip}&a=${augName}&f=${f + 1}`);
}

let last = 0, acc = 0;
function tick(now) {
  if (!playing) return;
  acc += (now - last) / 1000; last = now;
  const step = 1 / (clip.fps * Number($('speed').value));
  if (acc >= step) { acc %= step; show((t + 1) % aug.n); }
  requestAnimationFrame(tick);
}
function toggle() {
  playing = !playing; $('play').textContent = playing ? '❚❚' : '▶';
  if (playing) { last = performance.now(); acc = 0; requestAnimationFrame(tick); }
}

$('play').onclick = toggle;
$('prev').onclick = () => show(Math.max(t - 1, 0));
$('next').onclick = () => show(Math.min(t + 1, aug.n - 1));
$('scrub').oninput = (e) => show(Number(e.target.value));
$('reset-cam').onclick = () => resetCamera('front');
$('side-cam').onclick = () => resetCamera('side');
for (const id of ['t-mesh1', 't-pts1', 't-mesh2', 't-pts2', 't-mesh3', 't-pts3']) $(id).onchange = () => show(t);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range') return;
  if (e.key === ' ') { e.preventDefault(); toggle(); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); show(Math.max(t - 1, 0)); }
  if (e.key === 'ArrowRight') { e.preventDefault(); show(Math.min(t + 1, aug.n - 1)); }
});
window.addEventListener('resize', resize);

loadIndex().then(() => { if (clip) { resize(); show(t); } });
