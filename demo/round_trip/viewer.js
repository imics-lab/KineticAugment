// Round-trip viewer: reads clips/<id>/{frames/, v/<smoothing>/{clip.json, data.bin}} written by
// scripts/export_round_trip_demo.py and shows the four panels plus the error strips. The
// unsmoothed ('off') export is always loaded too, as the reference a smoothing setting is compared to.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {};
for (const k of ['pose', 'handL', 'handR', 'face', 'gap', 'in', 'rt']) C[k] = css(`--${k}`);
const HAND_COL = [C.handL, C.handR];
const SIDES = ['left', 'right'];
// Where the round-trip colour reaches full red, per stream.
const LIMIT = { arm: 15, hand: 25, face: 10 }; // % shoulder width, mm, mm
const ARM_BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

let clip = null, A = {}, images = [], t = 0, playing = false;
let base = null, B = {}, index = null, smoothing = 'off', loadedId = null; // B: 'off' arrays
const smoothed = () => smoothing !== 'off' && base !== clip;
const GHOST = '#6b7385';

// ---------- data ----------
function view(buf, m) {
  const Ctor = { float32: Float32Array, int16: Int16Array, uint16: Uint16Array }[m.dtype];
  const size = m.shape.reduce((a, b) => a * b, 1);
  return { data: new Ctor(buf, m.offset, size), shape: m.shape, stride: size / m.shape[0] };
}
const at = (name, f, src = A) => { const a = src[name]; return a.data.subarray(f * a.stride, (f + 1) * a.stride); };
const ok3 = (p, i) => Number.isFinite(p[3 * i]) && Number.isFinite(p[3 * i + 1]) && Number.isFinite(p[3 * i + 2]);
const fin = (v) => v !== null && v !== undefined && Number.isFinite(v);
const fmt = (v, d = 1) => (fin(v) ? v.toFixed(d) : '–');
const loadImage = (src) => new Promise((res, rej) => {
  const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src;
});

async function loadIndex() {
  index = await fetch('clips.json').then((r) => r.json());
  const select = $('clip');
  const order = index.order.filter((id) => index.clips[id]);
  const extra = Object.keys(index.clips).filter((id) => !index.order.includes(id));
  select.innerHTML = '';
  index.order.forEach((id, i) => {
    const o = new Option(`${i + 1} · ${id}${index.clips[id] ? '' : ' (not exported)'}`, id);
    o.disabled = !index.clips[id];
    select.add(o);
  });
  extra.forEach((id) => select.add(new Option(`– · ${id}`, id)));
  const h = new URLSearchParams(location.hash.slice(1));
  select.value = h.get('clip') && index.clips[h.get('clip')] ? h.get('clip') : (order[0] || extra[0]);
  smoothing = h.get('s') || 'off';
  select.onchange = () => loadClip(select.value, 0);
  $('smooth').onchange = () => { smoothing = $('smooth').value; loadClip(clip.clip, t); };
  await loadClip(select.value, Number(h.get('f') || 1) - 1);
}

async function fetchVariant(id, v) {
  const dir = `clips/${id}/v/${v}`;
  const [meta, buf] = await Promise.all([
    fetch(`${dir}/clip.json`).then((r) => r.json()),
    fetch(`${dir}/data.bin`).then((r) => r.arrayBuffer()),
  ]);
  const arrays = {};
  for (const [name, m] of Object.entries(meta.arrays)) arrays[name] = view(buf, m);
  // Exports from before the finger-curl measurement lack these; show them as all missing.
  for (const k of ['curl_deg', 'mano_curl_deg']) meta.series[k] ??= Array.from({ length: meta.n }, () => [null, null]);
  for (const k of ['curl_deg_median', 'mano_curl_deg_median']) meta.summary[k] ??= [null, null];
  return { meta, arrays };
}

async function loadClip(id, startFrame = 0) {
  playing = false; $('play').textContent = '▶';
  const exported = index.clips[id].variants;
  const variants = Object.keys(index.smoothing).filter((v) => exported[v]);
  if (!variants.includes(smoothing)) smoothing = 'off';
  const sel = $('smooth');
  sel.innerHTML = '';
  for (const v of variants) {
    const o = new Option(index.smoothing[v]?.label || v, v);
    o.title = index.smoothing[v]?.description || '';
    sel.add(o);
  }
  sel.value = smoothing;
  sel.title = index.smoothing[smoothing]?.description || '';
  const ref = variants.includes('off') ? 'off' : smoothing;
  const [b, c] = await Promise.all([
    fetchVariant(id, ref), smoothing === ref ? null : fetchVariant(id, smoothing)]);
  base = b.meta; B = b.arrays;
  clip = c ? c.meta : base; A = c ? c.arrays : B;
  document.body.classList.toggle('smoothed', smoothed());
  if (loadedId !== id) {
    images = await Promise.all(Array.from({ length: clip.n },
      (_, i) => loadImage(`clips/${id}/frames/${String(i).padStart(4, '0')}.jpg`)));
    for (const cv of [$('raw'), $('mp2d')]) {
      cv.width = clip.width; cv.height = clip.height;
      cv.style.aspectRatio = `${clip.width} / ${clip.height}`;
    }
    $('scrub').max = clip.n - 1;
    build3d();
    resetCamera('front');
    loadedId = id;
  } else {
    build3d();
  }
  summaryChips();
  show(Math.min(Math.max(startFrame, 0), clip.n - 1));
}

function summaryChips() {
  const s = clip.summary, r = base.summary;
  const chip = (label, get) => {
    const was = smoothed() ? ` <span class="was">off ${get(r)}</span>` : '';
    return `<span class="chip">${label} <b>${get(s)}</b>${was}</span>`;
  };
  const pair = (a, d = 1) => `${fmt(a[0], d)} / ${fmt(a[1], d)}`;
  $('summary').innerHTML = [
    `<span class="chip">frames <b>${clip.n}</b></span>`,
    chip('body', (m) => `${fmt(m.body_pct_median)}%`),
    chip('hands L/R', (m) => `${pair(m.hand_mm_median)} mm`),
    chip('MANO L/R', (m) => `${pair(m.mano_mm_median)} mm`),
    chip('curl L/R', (m) => `${pair(m.curl_deg_median)} °`),
    chip('face rigid/non', (m) => `${fmt(m.face_rigid_mm_median)} / ${fmt(m.face_nonrigid_mm_median)} mm`),
    chip('jitter arm/hand/face', (m) => m.jitter_rt_median.map((v) => fmt(v)).join(' / ')),
    chip('wrist gap L/R', (m) => `${pair(m.wrist_gap_px_median)} px`),
    chip('face gap', (m) => `${fmt(m.face_gap_px_median)} px`),
    smoothed() ? chip('moved pose/hand/face', (m) => `${m.shift_px_median.map((v) => fmt(v)).join(' / ')} px`) : '',
  ].join('') + '<span class="chip" title="medians over the clip; jitter in mm per frame² on the round trip">medians</span>';
}

// ---------- 2-D panels ----------
function draw2d(f) {
  const raw = $('raw').getContext('2d');
  raw.drawImage(images[f], 0, 0);

  const g = $('mp2d').getContext('2d');
  const W = clip.width, H = clip.height, lw = Math.max(1, H / 300);
  g.drawImage(images[f], 0, 0);
  g.fillStyle = 'rgba(0,0,0,0.45)'; g.fillRect(0, 0, W, H);
  g.lineCap = 'round';
  const pose = at('img_pose', f), vis = at('pose_vis', f);
  const pp = (i) => [pose[3 * i] * W, pose[3 * i + 1] * H];
  const pvis = (i) => vis[i] >= 0.5 && Number.isFinite(pose[3 * i]);
  const line = (a, b, col, w) => {
    g.strokeStyle = col; g.lineWidth = w; g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
  };
  const dot = (p, col, r) => { g.fillStyle = col; g.beginPath(); g.arc(p[0], p[1], r, 0, 7); g.fill(); };

  const drawLandmarks = (src, col, ghost) => {
    const P = at('img_pose', f, src), V = at('pose_vis', f, src);
    const q = (i) => [P[3 * i] * W, P[3 * i + 1] * H];
    const ok = (i) => V[i] >= 0.5 && Number.isFinite(P[3 * i]);
    const k = ghost ? 0.6 : 1;
    for (const [a, b] of clip.connections.pose) if (ok(a) && ok(b)) line(q(a), q(b), col.pose, 2 * lw * k);
    if (!ghost) for (let i = 0; i < 33; i++) if (ok(i)) dot(q(i), col.pose, 2 * lw);
    const Hd = at('img_hands', f, src);
    const h = (s, j) => [Hd[(s * 21 + j) * 3] * W, Hd[(s * 21 + j) * 3 + 1] * H];
    for (let s = 0; s < 2; s++) {
      if (!Number.isFinite(Hd[s * 63])) continue;
      for (const [a, b] of clip.connections.hand) line(h(s, a), h(s, b), col.hand[s], 1.5 * lw * k);
      if (!ghost) for (let j = 0; j < 21; j++) dot(h(s, j), col.hand[s], 1.6 * lw);
    }
    const Fc = at('img_face', f, src);
    const fq = (i) => [Fc[2 * i] * W, Fc[2 * i + 1] * H];
    if (Number.isFinite(Fc[0])) {
      if (!ghost && $('t-mesh478').checked) for (let i = 0; i < 478; i++) dot(fq(i), 'rgba(255,255,255,0.7)', 0.8 * lw);
      for (const [a, b] of clip.connections.face_contours) line(fq(a), fq(b), col.face, lw * k);
      for (const [a, b] of clip.connections.face_irises) line(fq(a), fq(b), col.face, lw * k);
    }
  };
  if (smoothed() && $('t-raw2d').checked) {
    drawLandmarks(B, { pose: GHOST, hand: [GHOST, GHOST], face: GHOST }, true);
  }
  drawLandmarks(A, { pose: C.pose, hand: HAND_COL, face: C.face }, false);
  const hands = at('img_hands', f), face = at('img_face', f);
  const hp = (s, j) => [hands[(s * 21 + j) * 3] * W, hands[(s * 21 + j) * 3 + 1] * H];
  const fp = (k) => [face[2 * k] * W, face[2 * k + 1] * H];
  // stream agreement: pose wrist -> hand wrist, pose face point -> face-mesh point
  if ($('t-agree').checked) {
    for (let s = 0; s < 2; s++) {
      const w = 15 + s;
      if (pvis(w) && Number.isFinite(hands[s * 63])) {
        line(pp(w), hp(s, 0), C.gap, 2 * lw);
        g.strokeStyle = C.gap; g.lineWidth = lw; g.beginPath(); g.arc(...pp(w), 5 * lw, 0, 7); g.stroke();
      }
    }
    if (Number.isFinite(face[0])) {
      for (const [p, k] of clip.pose_face_pairs) {
        if (!pvis(p)) continue;
        line(pp(p), fp(k), C.gap, 1.5 * lw);
        g.strokeStyle = C.gap; g.lineWidth = lw; g.beginPath(); g.arc(...pp(p), 3 * lw, 0, 7); g.stroke();
      }
    }
  }
}

// ---------- 3-D panels ----------
const heat = (v, limit) => {
  const c = new THREE.Color();
  if (!Number.isFinite(v)) return c.set('#888888');
  const x = Math.min(Math.max(v / limit, 0), 1);
  return x < 0.5 ? c.set('#3ddc84').lerp(new THREE.Color('#ffd23d'), x * 2)
    : c.set('#ffd23d').lerp(new THREE.Color('#ff3b3b'), (x - 0.5) * 2);
};
// MediaPipe world (x right, y down, z away) -> three.js (y up, camera on +z)
const W3 = (p, i, out = new THREE.Vector3()) => out.set(p[3 * i], -p[3 * i + 1], -p[3 * i + 2]);

const camera = new THREE.PerspectiveCamera(28, 4 / 3, 0.01, 20);
const views = {};
let geom = null;

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
  return { el, renderer, scene, controls, note, sun };
}

class Dots {
  constructor(n, radius, color) {
    this.mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(radius, 10, 8),
      new THREE.MeshBasicMaterial({ color: '#ffffff' }), n);
    this.n = n; this.m = new THREE.Matrix4(); this.base = new THREE.Color(color);
    for (let i = 0; i < n; i++) this.mesh.setColorAt(i, this.base);
  }
  set(i, v, color) {
    if (v) this.m.makeTranslation(v.x, v.y, v.z); else this.m.makeScale(0, 0, 0);
    this.mesh.setMatrixAt(i, this.m);
    if (color) this.mesh.setColorAt(i, color);
  }
  done() { this.mesh.instanceMatrix.needsUpdate = true; if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true; }
}

class Segs {
  constructor(n, color, opacity = 1) {
    this.pos = new Float32Array(n * 6); this.col = new Float32Array(n * 6).fill(1);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.base = new THREE.Color(color);
    this.mesh = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: opacity < 1, opacity }));
    this.mesh.frustumCulled = false; this.i = 0; this.n = n;
  }
  reset() { this.i = 0; this.pos.fill(0); }
  add(a, b, color = this.base) {
    if (!a || !b || this.i >= this.n) return;
    this.pos.set([a.x, a.y, a.z, b.x, b.y, b.z], this.i * 6);
    this.col.set([color.r, color.g, color.b, color.r, color.g, color.b], this.i * 6);
    this.i++;
  }
  done() { const g = this.mesh.geometry; g.attributes.position.needsUpdate = true; g.attributes.color.needsUpdate = true; }
}

function build3d() {
  if (!views.a) { views.a = makeView($('view3d')); views.b = makeView($('viewbody')); views.b.controls.target = views.a.controls.target; }
  for (const v of Object.values(views)) {
    for (const o of [...v.scene.children]) if (o.userData.clip) v.scene.remove(o);
  }
  const nv = A.mesh_verts.shape[1];
  geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
  geom.setIndex(new THREE.BufferAttribute(new Uint16Array(A.mesh_faces.data), 1));
  geom3 = {}; H3 = {};
  const add = (v, name, obj) => {
    const o = obj.mesh || obj; o.userData.clip = true; v.scene.add(o); geom3[name] = o; H3[name] = obj;
  };
  const n105 = clip.face105.mp_indices.length;
  const nContour = Object.values(clip.face105.contours).reduce((s, l) => s + l.length, 0);
  const skin = '#c9a58f';
  add(views.a, 'ghost', new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: skin, transparent: true, opacity: 0.25, depthWrite: false, roughness: 0.9 })));
  add(views.b, 'body', new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: skin, roughness: 0.85 })));
  add(views.b, 'wire', new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color: '#000000', wireframe: true, transparent: true, opacity: 0.15 })));
  // panel 3: extraction (blue), round trip (heat), error lines
  add(views.a, 'inBones', new Segs(clip.connections.pose.length + 42 + nContour, C.in));
  add(views.a, 'inDots', new Dots(33 + 42, 0.006, C.in));
  add(views.a, 'inFace', new Dots(478, 0.0015, C.in));
  add(views.a, 'rtBones', new Segs(5 + 42 + nContour, C.rt));
  add(views.a, 'rtDots', new Dots(6 + 42 + n105, 0.005, C.rt));
  add(views.a, 'errLines', new Segs(6 + 42 + n105, C.gap));
  add(views.a, 'rawBones', new Segs(clip.connections.pose.length + 42 + nContour, GHOST, 0.7));
  // panel 4: the mesh with the round-trip points on it
  add(views.b, 'bodyRt', new Dots(6 + 42 + n105, 0.006, C.rt));
  add(views.b, 'bodyIn', new Dots(6 + 42 + n105, 0.005, C.in));
}
let geom3 = {}, H3 = {};
const helper = (name) => H3[name];

function update3d(f) {
  const ok = clip.flags.body_ok[f];
  const scale = 1 / clip.scale_i16;
  const verts = at('mesh_verts', f), pos = geom.attributes.position.array;
  for (let i = 0; i < verts.length; i += 3) {
    pos[i] = verts[i] * scale; pos[i + 1] = -verts[i + 1] * scale; pos[i + 2] = -verts[i + 2] * scale;
  }
  geom.attributes.position.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();

  const wPose = at('w_pose', f), vis = at('pose_vis', f), hin = at('w_hands_in', f);
  const fin3 = at('w_face_in', f), rtArm = at('rt_arm', f), rtH = at('rt_hands', f), rtF = at('rt_face', f);
  const eArm = at('err_arm_pct', f), eHand = at('err_hand_mm', f), eFace = at('err_face_mm', f);
  const mp = clip.face105.mp_indices;
  const pv = (i) => (vis[i] >= 0.5 && ok3(wPose, i) ? W3(wPose, i) : null);
  const p3 = (arr, i) => (ok3(arr, i) ? W3(arr, i) : null);

  const inBones = helper('inBones'), inDots = helper('inDots'), inFace = helper('inFace');
  const rtBones = helper('rtBones'), rtDots = helper('rtDots'), errLines = helper('errLines');
  const bodyRt = helper('bodyRt'), bodyIn = helper('bodyIn');
  [inBones, rtBones, errLines].forEach((s) => s.reset());
  const rawBones = helper('rawBones');
  rawBones.reset();
  if (smoothed()) { // the unsmoothed extraction, as fitted in the 'off' export
    const P = at('w_pose', f, B), V = at('pose_vis', f, B), Hn = at('w_hands_in', f, B), Fc = at('w_face_in', f, B);
    const q = (arr, i) => (ok3(arr, i) ? W3(arr, i) : null);
    for (const [a, b] of clip.connections.pose) if (V[a] >= 0.5 && V[b] >= 0.5) rawBones.add(q(P, a), q(P, b));
    for (let s = 0; s < 2; s++) for (const [a, b] of clip.connections.hand) rawBones.add(q(Hn, s * 21 + a), q(Hn, s * 21 + b));
    for (const pairs of Object.values(clip.face105.contours))
      for (const [a, b] of pairs) rawBones.add(q(Fc, mp[a]), q(Fc, mp[b]));
  }
  rawBones.done();

  // extraction
  for (const [a, b] of clip.connections.pose) inBones.add(pv(a), pv(b));
  for (let i = 0; i < 33; i++) inDots.set(i, pv(i));
  for (let s = 0; s < 2; s++) {
    for (const [a, b] of clip.connections.hand) inBones.add(p3(hin, s * 21 + a), p3(hin, s * 21 + b));
    for (let j = 0; j < 21; j++) inDots.set(33 + s * 21 + j, p3(hin, s * 21 + j));
  }
  for (let k = 0; k < 478; k++) inFace.set(k, p3(fin3, k));
  for (const pairs of Object.values(clip.face105.contours))
    for (const [a, b] of pairs) inBones.add(p3(fin3, mp[a]), p3(fin3, mp[b]));

  // round trip, coloured by error; error lines from extraction to round trip
  let k = 0;
  const armIdx = (m) => clip.arm_mp.indexOf(m);
  for (const [a, b] of ARM_BONES) rtBones.add(p3(rtArm, armIdx(a)), p3(rtArm, armIdx(b)));
  for (let i = 0; i < 6; i++, k++) {
    const r = ok ? p3(rtArm, i) : null, col = heat(eArm[i], LIMIT.arm);
    rtDots.set(k, r, col); bodyRt.set(k, r, col);
    const src = pv(clip.arm_mp[i]); bodyIn.set(k, src);
    if (r && src) errLines.add(src, r, col);
  }
  for (let s = 0; s < 2; s++) {
    for (const [a, b] of clip.connections.hand) rtBones.add(p3(rtH, s * 21 + a), p3(rtH, s * 21 + b));
    for (let j = 0; j < 21; j++, k++) {
      const r = p3(rtH, s * 21 + j), col = heat(eHand[s * 21 + j], LIMIT.hand);
      rtDots.set(k, r, col); bodyRt.set(k, r, col);
      const src = p3(hin, s * 21 + j); bodyIn.set(k, src);
      if (r && src) errLines.add(src, r, col);
    }
  }
  for (const pairs of Object.values(clip.face105.contours))
    for (const [a, b] of pairs) rtBones.add(p3(rtF, a), p3(rtF, b));
  for (let r = 0; r < mp.length; r++, k++) {
    const q = ok ? p3(rtF, r) : null, col = heat(eFace[r], LIMIT.face);
    rtDots.set(k, q, col); bodyRt.set(k, q, col);
    const src = p3(fin3, mp[r]); bodyIn.set(k, src);
    if (q && src) errLines.add(src, q, col);
  }
  [inBones, inDots, inFace, rtBones, rtDots, errLines, bodyRt, bodyIn].forEach((h) => h.done());

  const show = (name, on) => { geom3[name].visible = on; };
  show('inBones', $('t-in').checked); show('inDots', $('t-in').checked); show('inFace', $('t-in').checked);
  show('rtBones', $('t-rt').checked); show('rtDots', $('t-rt').checked);
  show('rawBones', smoothed() && $('t-raw3d').checked);
  show('errLines', $('t-err').checked); show('ghost', $('t-ghost').checked && ok);
  show('body', !!ok); show('wire', $('t-wire').checked && ok);
  show('bodyRt', $('t-onbody').checked); show('bodyIn', $('t-onbody-in').checked);
  const msg = ok ? '' : 'no SMPL-X fit this frame (shoulder not visible)';
  views.a.note.textContent = msg; views.b.note.textContent = msg;
  views.a.note.style.display = views.b.note.style.display = msg ? '' : 'none';
}

function resetCamera(which) {
  // frame the upper body: shoulders to head, from the clip's median pose
  const pts = [];
  for (let f = 0; f < clip.n; f++) {
    const w = at('w_pose', f), v = at('pose_vis', f);
    for (const i of [0, 11, 12]) if (v[i] >= 0.5 && ok3(w, i)) pts.push(W3(w, i));
  }
  const c = new THREE.Vector3();
  pts.forEach((p) => c.add(p)); if (pts.length) c.divideScalar(pts.length);
  c.y -= 0.12;
  const d = 1.9;
  views.a.controls.target.copy(c);
  camera.position.copy(c).add(which === 'side' ? new THREE.Vector3(-d, 0, 0) : new THREE.Vector3(0, 0, d));
  camera.lookAt(c);
  views.a.controls.update(); views.b.controls.update();
  renderAll();
}

function resize() {
  for (const v of Object.values(views)) {
    const { clientWidth: w, clientHeight: h } = v.el;
    v.renderer.setSize(w, h, false);
    v.renderer.domElement.style.width = '100%'; v.renderer.domElement.style.height = '100%';
  }
  const v = views.a; if (v) { camera.aspect = v.el.clientWidth / v.el.clientHeight; camera.updateProjectionMatrix(); }
  drawStrips(); renderAll();
}
function renderAll() {
  for (const v of Object.values(views)) {
    v.sun.position.copy(camera.position).add(new THREE.Vector3(0.4, 0.8, 0)); v.sun.target.position.copy(v.controls.target);
    v.sun.target.updateMatrixWorld();
    v.renderer.render(v.scene, camera);
  }
}

// ---------- strips ----------
const col = (i) => (r) => r[i];
const STRIPS = [
  { title: 'body · arm landmarks 11–16 (% shoulder width)', ref: 15, series: [
    { label: 'body', col: '#d0d4de', get: (m) => m.series.body_pct }] },
  { title: 'hands · wrist-relative pts 1–20 (mm)', ref: 25, series: [
    { label: 'L on SMPL-X', col: C.handL, get: (m) => m.series.hand_mm.map(col(0)) },
    { label: 'R on SMPL-X', col: C.handR, get: (m) => m.series.hand_mm.map(col(1)) },
    { label: 'L MANO', col: C.handL, dash: [3, 3], get: (m) => m.series.mano_mm.map(col(0)) },
    { label: 'R MANO', col: C.handR, dash: [3, 3], get: (m) => m.series.mano_mm.map(col(1)) }] },
  { title: 'hands · finger curl error (° per joint, mean |Δ angle| over 15 joints)', ref: null, series: [
    { label: 'L on SMPL-X', col: C.handL, get: (m) => m.series.curl_deg.map(col(0)) },
    { label: 'R on SMPL-X', col: C.handR, get: (m) => m.series.curl_deg.map(col(1)) },
    { label: 'L MANO', col: C.handL, dash: [3, 3], get: (m) => m.series.mano_curl_deg.map(col(0)) },
    { label: 'R MANO', col: C.handR, dash: [3, 3], get: (m) => m.series.mano_curl_deg.map(col(1)) }] },
  { title: 'face · 105 embedded pts, RMS (mm)', ref: 10, series: [
    { label: 'rigid', col: C.face, get: (m) => m.series.face_rigid_mm },
    { label: 'non-rigid', col: C.face, dash: [3, 3], get: (m) => m.series.face_nonrigid_mm }] },
  { title: 'round-trip jitter (mm / frame²)', ref: null, series: [
    { label: 'arms', col: '#d0d4de', get: (m) => m.series.jitter_rt.map(col(0)) },
    { label: 'hands', col: '#ffd23d', get: (m) => m.series.jitter_rt.map(col(1)) },
    { label: 'face', col: C.face, get: (m) => m.series.jitter_rt.map(col(2)) }] },
  { title: 'stream gaps in the image (px)', ref: null, series: [
    { label: 'L wrist: pose↔hand', col: C.handL, get: (m) => m.series.wrist_gap_px.map(col(0)) },
    { label: 'R wrist: pose↔hand', col: C.handR, get: (m) => m.series.wrist_gap_px.map(col(1)) },
    { label: 'face: pose↔mesh RMS', col: C.face, get: (m) => m.series.face_gap_rms_px }] },
  { title: 'moved by smoothing (px)', ref: null, smoothedOnly: true, series: [
    { label: 'pose', col: C.pose, get: (m) => m.series.shift_px.map(col(0)) },
    { label: 'hands', col: '#ffd23d', get: (m) => m.series.shift_px.map(col(1)) },
    { label: 'face', col: C.face, get: (m) => m.series.shift_px.map(col(2)) }] },
];
const FLAGS = [
  ['SMPL-X fit', (f) => clip.flags.body_ok[f]],
  ['face found', (f) => clip.flags.face_valid[f]],
  ['head reliable', (f) => clip.flags.head_reliable[f]],
  ['left hand', (f) => clip.flags.hand_present[f][0]],
  ['right hand', (f) => clip.flags.hand_present[f][1]],
  ['hand near face', (f) => clip.flags.hand_near_face[f], true],
];
const STRIP_H = 52, FLAG_H = 9, PAD_L = 8, GAP = 20;

function drawStrips() {
  if (!clip) return;
  const cv = $('strips'), dpr = window.devicePixelRatio || 1;
  const strips = STRIPS.filter((st) => !st.smoothedOnly || smoothed());
  const Wc = cv.clientWidth, Hc = strips.length * (STRIP_H + GAP) + FLAGS.length * FLAG_H + 18;
  cv.style.height = `${Hc}px`; cv.width = Wc * dpr; cv.height = Hc * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  g.font = '11px -apple-system, sans-serif';
  const x = (f) => PAD_L + (f / Math.max(clip.n - 1, 1)) * (Wc - PAD_L - 4);
  let y = 0;
  for (const strip of strips) {
    const series = strip.series.map((sr) => ({ ...sr, v: sr.get(clip), ref: smoothed() && !strip.smoothedOnly ? sr.get(base) : null }));
    y += GAP;
    const all = series.flatMap((s) => [...s.v, ...(s.ref || [])].filter(fin));
    // scale to the 98th percentile so one spike does not flatten the strip; spikes are marked
    const sorted = [...all].sort((a, b) => a - b);
    const p98 = sorted.length ? sorted[Math.floor(0.98 * (sorted.length - 1))] : 0;
    const top = Math.max(Math.min(sorted[sorted.length - 1] ?? 0, p98 * 1.6), strip.ref || 0, 1) * 1.08;
    const Y = (v) => y + STRIP_H - (v / top) * STRIP_H;
    g.fillStyle = '#8b93a7'; g.fillText(strip.title, PAD_L, y - 6);
    g.strokeStyle = '#2a2f3a'; g.lineWidth = 1; g.strokeRect(PAD_L, y, Wc - PAD_L - 4, STRIP_H);
    if (strip.ref) {
      g.setLineDash([2, 4]); g.strokeStyle = '#ff3b3b88'; g.beginPath(); g.moveTo(PAD_L, Y(strip.ref)); g.lineTo(Wc - 4, Y(strip.ref)); g.stroke(); g.setLineDash([]);
    }
    g.fillStyle = '#5c6478'; g.fillText(top.toFixed(0), Wc - 28, y + 10);
    let lx = Wc - 4;
    if (smoothed() && !strip.smoothedOnly) {
      const label = 'off (grey)'; lx -= g.measureText(label).width + 10;
      g.fillStyle = GHOST; g.fillText(label, lx, y - 6);
    }
    for (const s of [...series].reverse()) {
      const w = g.measureText(s.label).width; lx -= w + 22;
      g.strokeStyle = s.col; g.setLineDash(s.dash || []); g.beginPath(); g.moveTo(lx, y - 9); g.lineTo(lx + 12, y - 9); g.stroke();
      g.fillStyle = '#c4c9d6'; g.fillText(s.label, lx + 15, y - 6);
    }
    const trace = (vals, colour, dash, width) => {
      g.strokeStyle = colour; g.setLineDash(dash || []); g.lineWidth = width; g.beginPath();
      let pen = false;
      vals.forEach((v, f) => { if (!fin(v)) { pen = false; return; } const yy = Y(Math.min(v, top)); pen ? g.lineTo(x(f), yy) : g.moveTo(x(f), yy); pen = true; });
      g.stroke();
      g.fillStyle = '#ff3b3b';
      vals.forEach((v, f) => { if (fin(v) && v > top) { g.beginPath(); g.moveTo(x(f) - 3, y + 5); g.lineTo(x(f) + 3, y + 5); g.lineTo(x(f), y); g.fill(); } });
    };
    for (const s of series) if (s.ref) trace(s.ref, GHOST, s.dash, 1.1); // 'off', underneath
    for (const s of series) trace(s.v, s.col, s.dash, 1.4);
    g.setLineDash([]);
    y += STRIP_H;
  }
  y += 12;
  const cell = (Wc - PAD_L - 4) / clip.n;
  for (const [label, fn, alarm] of FLAGS) {
    for (let f = 0; f < clip.n; f++) {
      const on = fn(f);
      g.fillStyle = alarm ? (on ? '#ff3b3b' : '#20242d') : (on ? '#3a6b52' : '#7a2b2b');
      g.fillRect(PAD_L + f * cell, y, Math.max(cell - 0.5, 0.5), FLAG_H - 2);
    }
    g.fillStyle = '#c4c9d6'; g.font = '9px -apple-system, sans-serif';
    g.fillText(label, Wc - g.measureText(label).width - 6, y + 7);
    g.font = '11px -apple-system, sans-serif';
    y += FLAG_H;
  }
  // playhead
  g.strokeStyle = '#ffffff'; g.lineWidth = 1; g.beginPath(); g.moveTo(x(t), 0); g.lineTo(x(t), Hc); g.stroke();
}

function seekFromStrip(ev) {
  const r = $('strips').getBoundingClientRect();
  const f = Math.round(((ev.clientX - r.left - PAD_L) / (r.width - PAD_L - 4)) * (clip.n - 1));
  show(Math.min(Math.max(f, 0), clip.n - 1));
}

// ---------- readout ----------
function readout(f) {
  const s = clip.series, m = clip.summary, fl = clip.flags, r = base.summary, cmp = smoothed();
  const row = (label, now, med, off = '', note = '') => `<tr><td>${label}</td><td>${now}</td><td>${med}</td>${cmp ? `<td class="was">${off}</td>` : ''}<td class="flag">${note}</td></tr>`;
  const sw = (c) => `<span class="sw" style="background:${c}"></span>`;
  const shPx = s.shoulder_px[f];
  const pct = (px) => (fin(px) && fin(shPx) ? ` (${(100 * px / shPx).toFixed(0)}%)` : '');
  const pairs = clip.pose_face_pairs.map(([, , name], k) => `${name} ${fmt(s.face_gap_px[f][k], 0)}`).join(' · ');
  const span = cmp ? 5 : 4;
  const triple = (a) => a.map((v) => fmt(v)).join(' / ');
  $('readout').innerHTML = `<table>
    <tr><th>frame ${f + 1}</th><th>now</th><th>median</th>${cmp ? '<th>off med.</th>' : ''}<th></th></tr>
    ${row('body % sh.w.', fmt(s.body_pct[f]), fmt(m.body_pct_median), fmt(r.body_pct_median), fl.body_ok[f] ? '' : 'no fit')}
    ${row(`${sw(C.handL)}L hand mm`, fmt(s.hand_mm[f][0]), fmt(m.hand_mm_median[0]), fmt(r.hand_mm_median[0]), fl.hand_present[f][0] ? '' : 'missing')}
    ${row(`${sw(C.handR)}R hand mm`, fmt(s.hand_mm[f][1]), fmt(m.hand_mm_median[1]), fmt(r.hand_mm_median[1]), fl.hand_present[f][1] ? '' : 'missing')}
    ${row('&nbsp;&nbsp;MANO L / R', `${fmt(s.mano_mm[f][0])} / ${fmt(s.mano_mm[f][1])}`, `${fmt(m.mano_mm_median[0])} / ${fmt(m.mano_mm_median[1])}`, `${fmt(r.mano_mm_median[0])} / ${fmt(r.mano_mm_median[1])}`)}
    ${row(`${sw(C.handL)}L curl °`, fmt(s.curl_deg[f][0]), fmt(m.curl_deg_median[0]), fmt(r.curl_deg_median[0]), fl.hand_present[f][0] ? '' : 'missing')}
    ${row(`${sw(C.handR)}R curl °`, fmt(s.curl_deg[f][1]), fmt(m.curl_deg_median[1]), fmt(r.curl_deg_median[1]), fl.hand_present[f][1] ? '' : 'missing')}
    ${row('&nbsp;&nbsp;MANO curl L / R', `${fmt(s.mano_curl_deg[f][0])} / ${fmt(s.mano_curl_deg[f][1])}`, `${fmt(m.mano_curl_deg_median[0])} / ${fmt(m.mano_curl_deg_median[1])}`, `${fmt(r.mano_curl_deg_median[0])} / ${fmt(r.mano_curl_deg_median[1])}`)}
    ${row(`${sw(C.face)}face rigid mm`, fmt(s.face_rigid_mm[f]), fmt(m.face_rigid_mm_median), fmt(r.face_rigid_mm_median), fl.face_valid[f] ? (fl.head_reliable[f] ? '' : 'unreliable') : 'no face')}
    ${row('&nbsp;&nbsp;non-rigid mm', fmt(s.face_nonrigid_mm[f]), fmt(m.face_nonrigid_mm_median), fmt(r.face_nonrigid_mm_median))}
    <tr><th colspan="${span}">jitter, arms / hands / face (mm per frame²)</th></tr>
    ${row('fitted input', triple(s.jitter_in[f]), triple(m.jitter_in_median), triple(r.jitter_in_median))}
    ${row('round trip', triple(s.jitter_rt[f]), triple(m.jitter_rt_median), triple(r.jitter_rt_median))}
    ${cmp ? row('moved px p/h/f', triple(s.shift_px[f]), triple(m.shift_px_median), '0') : ''}
    <tr><th colspan="${span}">streams in the image (px, % of shoulder width)</th></tr>
    ${row('L wrist pose↔hand', fmt(s.wrist_gap_px[f][0], 0) + pct(s.wrist_gap_px[f][0]), fmt(m.wrist_gap_px_median[0], 0), fmt(r.wrist_gap_px_median[0], 0))}
    ${row('R wrist pose↔hand', fmt(s.wrist_gap_px[f][1], 0) + pct(s.wrist_gap_px[f][1]), fmt(m.wrist_gap_px_median[1], 0), fmt(r.wrist_gap_px_median[1], 0))}
    ${row('face pose↔mesh RMS', fmt(s.face_gap_rms_px[f], 0) + pct(s.face_gap_rms_px[f]), fmt(m.face_gap_px_median, 0), fmt(r.face_gap_px_median, 0))}
    <tr><td colspan="${span}" style="color:#8b93a7;white-space:normal">${pairs}</td></tr>
    ${fl.hand_near_face[f] ? `<tr><td colspan="${span}" class="flag">hand near face (2-D)</td></tr>` : ''}
  </table>`;
}

// ---------- frame / playback ----------
function show(f) {
  t = f;
  $('scrub').value = f;
  $('frame-label').textContent = `frame ${f + 1} / ${clip.n}`;
  draw2d(f); update3d(f); renderAll(); drawStrips(); readout(f);
  history.replaceState(null, '', `#clip=${clip.clip}&s=${smoothing}&f=${f + 1}`);
}

let last = 0, acc = 0;
function tick(now) {
  if (!playing) return;
  acc += (now - last) / 1000; last = now;
  const step = 1 / (clip.fps * Number($('speed').value));
  if (acc >= step) { acc %= step; show((t + 1) % clip.n); }
  requestAnimationFrame(tick);
}
function toggle() {
  playing = !playing; $('play').textContent = playing ? '❚❚' : '▶';
  if (playing) { last = performance.now(); acc = 0; requestAnimationFrame(tick); }
}

$('play').onclick = toggle;
$('prev').onclick = () => show(Math.max(t - 1, 0));
$('next').onclick = () => show(Math.min(t + 1, clip.n - 1));
$('scrub').oninput = (e) => show(Number(e.target.value));
$('reset-cam').onclick = () => resetCamera('front');
$('side-cam').onclick = () => resetCamera('side');
for (const id of ['t-agree', 't-mesh478', 't-in', 't-rt', 't-err', 't-ghost', 't-onbody', 't-onbody-in', 't-wire', 't-raw2d', 't-raw3d'])
  $(id).onchange = () => show(t);
let dragging = false;
$('strips').addEventListener('pointerdown', (e) => { dragging = true; seekFromStrip(e); });
window.addEventListener('pointermove', (e) => dragging && seekFromStrip(e));
window.addEventListener('pointerup', () => { dragging = false; });
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range') return;
  if (e.key === ' ') { e.preventDefault(); toggle(); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); show(Math.max(t - 1, 0)); }
  if (e.key === 'ArrowRight') { e.preventDefault(); show(Math.min(t + 1, clip.n - 1)); }
});
window.addEventListener('resize', resize);
window.addEventListener('hashchange', () => {
  const h = new URLSearchParams(location.hash.slice(1));
  if (!clip || (h.get('clip') && !index.clips[h.get('clip')])) return; // not loaded yet / not exported
  const f = Number(h.get('f') || 1) - 1;
  const sm = h.get('s') || 'off';
  if ((h.get('clip') && h.get('clip') !== clip.clip) || sm !== smoothing) {
    smoothing = sm; $('clip').value = h.get('clip') || clip.clip; loadClip($('clip').value, f);
  }
  else if (f !== t) show(Math.min(Math.max(f, 0), clip.n - 1));
});

loadIndex().then(() => { resize(); show(t); });
