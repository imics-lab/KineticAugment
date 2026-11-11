// All-augmentations grid: every aug/<name>/ bundle of one clip (Tier 1 and week 4) in its own
// panel, drawn by one WebGL renderer into each panel's rectangle (the page would run out of
// WebGL contexts with one per panel). All panels share one camera and one clock.
//
// Clock "playback": frame k of every panel plays at the clip's frame rate, so a time warp runs
// ahead of or behind the original (blue) and a shorter clip holds its last frame. Clock "source
// time": each panel shows its frame nearest the current source frame (its `times`).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GROUPS, W4_EXPLAIN } from './week4.js';

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { orig: css('--orig'), aug: css('--aug') };
const ARM_BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const SKIN = '#c9a58f';

let index = null, clip = null, A = null, B = null, panels = [], images = [], t = 0, playing = false;
let length = 1; // frames on the clock
const hidden = new Set(JSON.parse(safeGet('allHidden') || '[]'));

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } }

// ---------- data ----------
function view(buf, m) {
  const Ctor = { float32: Float32Array, int16: Int16Array, uint16: Uint16Array }[m.dtype];
  const size = m.shape.reduce((a, b) => a * b, 1);
  return { data: new Ctor(buf, m.offset, size), shape: m.shape, stride: size / m.shape[0] };
}
const at = (src, name, f) => { const a = src[name]; return a.data.subarray(f * a.stride, (f + 1) * a.stride); };
const ok3 = (p, i) => Number.isFinite(p[3 * i]) && Number.isFinite(p[3 * i + 1]) && Number.isFinite(p[3 * i + 2]);
const loadImage = (src) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
async function fetchBundle(dir) {
  const [meta, buf] = await Promise.all([
    fetch(`${dir}/clip.json`).then((r) => r.json()),
    fetch(`${dir}/data.bin`).then((r) => r.arrayBuffer()),
  ]);
  const arrays = {};
  for (const [name, m] of Object.entries(meta.arrays)) arrays[name] = view(buf, m);
  return { meta, arrays };
}

// The original's frame f: the Tier 1 round trip (v/off) or the week-4 original (w4.base).
function original(week4, f) {
  if (week4 && B) {
    return { verts: at(B.arrays, 'mesh_verts', f), scale: B.meta.scale_i16, arm: at(B.arrays, 'aug_arm', f),
      hands: at(B.arrays, 'aug_hands', f), face: at(B.arrays, 'aug_face', f), ok: !!B.meta.body_ok[f] };
  }
  return { verts: at(A.arrays, 'mesh_verts', f), scale: A.meta.scale_i16, arm: at(A.arrays, 'rt_arm', f),
    hands: at(A.arrays, 'rt_hands', f), face: at(A.arrays, 'rt_face', f), ok: !!A.meta.flags.body_ok[f] };
}

// ---------- three.js ----------
const W3 = (p, i, out = new THREE.Vector3()) => out.set(p[3 * i], -p[3 * i + 1], -p[3 * i + 2]);
const renderer = new THREE.WebGLRenderer({ canvas: $('gl'), antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0);
const camera = new THREE.PerspectiveCamera(28, 3 / 4, 0.01, 20);
const target = new THREE.Vector3();
let pending = false;
const requestRender = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; renderAll(); }); } };

class Segs {
  constructor(n) {
    this.pos = new Float32Array(n * 6);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.i = 0; this.n = n;
  }
  reset() { this.i = 0; this.pos.fill(0); }
  add(a, b) { if (!a || !b || this.i >= this.n) return; this.pos.set([a.x, a.y, a.z, b.x, b.y, b.z], this.i * 6); this.i++; }
  done() { this.geometry.attributes.position.needsUpdate = true; }
  line(color) { const l = new THREE.LineSegments(this.geometry, new THREE.LineBasicMaterial({ color })); l.frustumCulled = false; return l; }
}

function meshGeometry(mirrored = false) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(A.arrays.mesh_verts.shape[1] * 3), 3));
  const index = new Uint16Array(A.arrays.mesh_faces.data);
  // a mirror reverses the triangles' winding; reversing the index keeps the front faces outward
  if (mirrored) for (let i = 0; i < index.length; i += 3) { const x = index[i + 1]; index[i + 1] = index[i + 2]; index[i + 2] = x; }
  g.setIndex(new THREE.BufferAttribute(index, 1));
  return g;
}
const nSegs = () => 5 + 42 + Object.values(A.meta.face105.contours).reduce((s, l) => s + l.length, 0);

function setMesh(g, verts, scale) {
  const pos = g.attributes.position.array;
  for (let i = 0; i < verts.length; i += 3) {
    pos[i] = verts[i] * scale; pos[i + 1] = -verts[i + 1] * scale; pos[i + 2] = -verts[i + 2] * scale;
  }
  g.attributes.position.needsUpdate = true;
}

// Skeleton (arms, hands, face contours) of one frame into a Segs.
function skeleton(segs, arm, hands, face, ok) {
  const p3 = (arr, i) => (ok && ok3(arr, i) ? W3(arr, i) : null);
  segs.reset();
  for (const [a, b] of ARM_BONES) segs.add(p3(arm, A.meta.arm_mp.indexOf(a)), p3(arm, A.meta.arm_mp.indexOf(b)));
  for (let s = 0; s < 2; s++) for (const [a, b] of A.meta.connections.hand) segs.add(p3(hands, s * 21 + a), p3(hands, s * 21 + b));
  for (const pairs of Object.values(A.meta.face105.contours)) for (const [a, b] of pairs) segs.add(p3(face, a), p3(face, b));
  segs.done();
}

// flatShading takes the normals from screen-space derivatives, so no per-frame normal pass. The
// ghosts are single-sided (a double-sided ghost blends twice) and drawn in a fixed order: the
// original first, the augmentation over it, the skeleton lines last.
const ghost = (color, opacity) => new THREE.MeshStandardMaterial({
  color, transparent: true, opacity, depthWrite: false, roughness: 0.9, flatShading: true });
const solid = () => new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.85, flatShading: true });

let shared = null; // the two originals' geometry and skeleton, updated once per frame
function makeScene() {
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight('#ffffff', '#30343c', 1.6));
  const sun = new THREE.DirectionalLight('#ffffff', 1.6);
  scene.add(sun, sun.target);
  scene.userData.sun = sun;
  return scene;
}

function buildPanels() {
  const grid = $('grid');
  grid.innerHTML = '';
  panels = [];
  shared = {
    tier1: { geom: meshGeometry(), segs: new Segs(nSegs()) },
    week4: { geom: meshGeometry(), segs: new Segs(nSegs()) },
  };
  // the first cell: the source video frame
  const src = document.createElement('figure');
  src.className = 'cell';
  src.innerHTML = '<figcaption><span>source video</span><span class="mono frame" id="src-frame"></span></figcaption><canvas id="raw"></canvas><div class="cap">the frame the original shows</div>';
  grid.appendChild(src);
  const raw = $('raw'); raw.width = A.meta.width; raw.height = A.meta.height;

  const entries = [{ name: 'original', label: 'original', group: 'original', meta: null }];
  const order = Object.keys(index.augmentation);
  for (const name of order) {
    const bundle = clip.bundles[name];
    if (bundle) entries.push({ name, label: index.augmentation[name].label, group: index.augmentation[name].group || 'tier1', ...bundle });
  }
  for (const e of entries) {
    const cell = document.createElement('figure');
    cell.className = 'cell'; cell.dataset.group = e.group;
    const title = e.meta ? `<a href="augment.html#clip=${clip.id}&a=${e.name}" title="${e.meta.description}">${e.label}</a>` : '<span>original (round trip, no augmentation)</span>';
    const w4 = e.meta && e.meta.kind === 'week4';
    const explain = w4 && W4_EXPLAIN[e.name] ? ` title="proven: ${W4_EXPLAIN[e.name].proven} · used: ${W4_EXPLAIN[e.name].used}"` : '';
    const group = e.group === 'original' ? '' : `<span class="tag">${(GROUPS[e.group] || e.group).replace('week 4 · ', 'w4 ')}</span>`;
    cell.innerHTML = `<figcaption>${title}${group}<span class="mono frame"></span></figcaption><div class="vp"></div><div class="cap"${explain}>${e.meta ? (e.meta.detail || e.meta.title) : 'the round trip with the face read back off the mesh'}${e.meta && e.meta.note ? ` <span class="was">(${e.meta.note.replace(/^This demo /, 'demo ')})</span>` : ''}</div>`;
    grid.appendChild(cell);
    const vp = cell.querySelector('.vp');
    const scene = makeScene();
    const own = e.meta ? shared[w4 ? 'week4' : 'tier1'] : shared.tier1;
    const p = { ...e, w4, cell, vp, scene, frameEl: cell.querySelector('.frame') };
    p.origMesh = new THREE.Mesh(own.geom, e.meta ? ghost(C.orig, 0.28) : solid());
    p.origLines = own.segs.line(C.orig);
    scene.add(p.origMesh, p.origLines);
    if (e.meta) {
      p.geom = meshGeometry(e.meta.mirrored);
      p.augMesh = new THREE.Mesh(p.geom, ghost(C.aug, 0.34));
      p.segs = new Segs(nSegs());
      p.augLines = p.segs.line(C.aug);
      scene.add(p.augMesh, p.augLines);
      // source time -> the panel's nearest frame
      const times = e.arrays.times.data;
      p.nearest = Int32Array.from({ length: A.meta.n }, (_, f) => {
        let best = 0;
        for (let k = 1; k < e.meta.n; k++) if (Math.abs(times[k] - f) < Math.abs(times[best] - f)) best = k;
        return best;
      });
    }
    [p.origMesh, p.augMesh, p.origLines, p.augLines].forEach((o, i) => { if (o) { o.frustumCulled = false; o.renderOrder = i; } });
    p.controls = new OrbitControls(camera, vp);
    p.controls.target = target;
    p.controls.enableZoom = false;
    p.controls.addEventListener('change', requestRender);
    panels.push(p);
  }
  applyGroups();
}

function groupToggles() {
  const groups = ['tier1', ...Object.keys(GROUPS).filter((g) => g !== 'tier1')]
    .filter((g) => panels.some((p) => p.group === g));
  $('groups').innerHTML = groups.map((g) => `<label><input type="checkbox" data-group="${g}" ${hidden.has(g) ? '' : 'checked'}> ${GROUPS[g]}</label>`).join('');
  for (const box of $('groups').querySelectorAll('input')) {
    box.onchange = () => {
      if (box.checked) hidden.delete(box.dataset.group); else hidden.add(box.dataset.group);
      safeSet('allHidden', JSON.stringify([...hidden]));
      applyGroups();
    };
  }
}
function applyGroups() {
  for (const p of panels) p.cell.style.display = hidden.has(p.group) ? 'none' : '';
  requestRender();
}

// ---------- frames ----------
function frameOf(p) { // the panel's frame and the original's, for the clock position t
  if ($('clock').value === 'source') return { k: p.nearest ? p.nearest[t] : t, f: t };
  if (!p.meta) return { k: Math.min(t, A.meta.n - 1), f: Math.min(t, A.meta.n - 1) };
  return { k: Math.min(t, p.meta.n - 1), f: Math.min(t, A.meta.n - 1), ended: t >= p.meta.n };
}

function update() {
  const f0 = Math.min(t, A.meta.n - 1);
  const src = $('clock').value === 'source' ? t : f0;
  for (const [key, w4] of [['tier1', false], ['week4', true]]) {
    const o = original(w4, src);
    setMesh(shared[key].geom, o.verts, 1 / o.scale);
    skeleton(shared[key].segs, o.arm, o.hands, o.face, o.ok);
    shared[key].ok = o.ok;
  }
  const showOrig = $('t-orig').checked, mesh = $('t-mesh').checked, pts = $('t-pts').checked, solidAug = $('t-solid').checked;
  for (const p of panels) {
    if (p.cell.style.display === 'none') continue;
    const okO = shared[p.w4 ? 'week4' : 'tier1'].ok;
    if (!p.meta) {
      p.origMesh.visible = mesh && okO; p.origLines.visible = pts;
      p.frameEl.textContent = `${src + 1}/${A.meta.n}`;
      continue;
    }
    const { k, ended } = frameOf(p);
    const okA = !!p.meta.body_ok[k];
    setMesh(p.geom, at(p.arrays, 'mesh_verts', k), 1 / p.meta.scale_i16);
    skeleton(p.segs, at(p.arrays, 'aug_arm', k), at(p.arrays, 'aug_hands', k), at(p.arrays, 'aug_face', k), okA);
    const wantSolid = solidAug ? 'solid' : 'ghost';
    if (p.augMesh.userData.kind !== wantSolid) { p.augMesh.material = solidAug ? solid() : ghost(C.aug, 0.34); p.augMesh.userData.kind = wantSolid; }
    p.augMesh.visible = mesh && okA; p.augLines.visible = pts;
    p.origMesh.visible = showOrig && mesh && okO; p.origLines.visible = showOrig && pts;
    p.frameEl.textContent = `${k + 1}/${p.meta.n} ← ${(p.arrays.times.data[k] + 1).toFixed(1)}${ended ? ' · ended' : ''}`;
    p.cell.classList.toggle('ended', !!ended);
  }
  $('raw').getContext('2d').drawImage(images[src], 0, 0);
  $('src-frame').textContent = `${src + 1}/${A.meta.n}`;
}

// ---------- rendering ----------
function renderAll() {
  const canvas = renderer.domElement;
  const w = window.innerWidth, h = window.innerHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) renderer.setSize(w, h, false);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setScissorTest(true);
  const top = $('bar').getBoundingClientRect().bottom; // panels scrolled under the sticky bar are cut
  for (const p of panels) {
    if (p.cell.style.display === 'none') continue;
    const r = p.vp.getBoundingClientRect();
    if (r.bottom < top || r.top > h || r.right < 0 || r.left > w || r.width < 2) continue;
    const y0 = Math.max(r.top, top);
    camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
    renderer.setViewport(r.left, h - r.bottom, r.width, r.height);
    renderer.setScissor(r.left, h - r.bottom, r.width, r.bottom - y0);
    const sun = p.scene.userData.sun;
    sun.position.copy(camera.position).add(new THREE.Vector3(0.4, 0.8, 0)); sun.target.position.copy(target);
    renderer.render(p.scene, camera);
  }
}

function aim(which) {
  const pts = [];
  let d = 2.4;
  const f = Math.min(t, A.meta.n - 1);
  const o = original(false, f);
  if (which === 'hands') { for (let i = 0; i < 42; i++) if (ok3(o.hands, i)) pts.push(W3(o.hands, i)); d = 0.8; }
  if (which === 'face') { for (let i = 0; i < A.meta.face105.mp_indices.length; i++) if (ok3(o.face, i)) pts.push(W3(o.face, i)); d = 0.5; }
  if (!pts.length) {
    for (let g = 0; g < A.meta.n; g++) {
      const w = at(A.arrays, 'w_pose', g), v = at(A.arrays, 'pose_vis', g);
      for (const i of [0, 11, 12]) if (v[i] >= 0.5 && ok3(w, i)) pts.push(W3(w, i));
    }
    d = 2.4;
  }
  target.set(0, 0, 0);
  pts.forEach((p) => target.add(p)); if (pts.length) target.divideScalar(pts.length);
  if (d === 2.4) target.y -= 0.15;
  camera.position.copy(target).add(which === 'side' ? new THREE.Vector3(-d, 0, 0) : new THREE.Vector3(0, 0, d));
  camera.lookAt(target);
  for (const p of panels) p.controls.update();
  requestRender();
}
function zoom(factor) {
  camera.position.sub(target).multiplyScalar(factor).add(target);
  requestRender();
}

// ---------- loading ----------
async function loadIndex() {
  index = await fetch('clips.json').then((r) => r.json());
  const ids = Object.keys(index.clips).filter((id) => index.clips[id].augmentations);
  const ordered = [...index.order.filter((id) => ids.includes(id)), ...ids.filter((id) => !index.order.includes(id))];
  const select = $('clip');
  for (const id of ordered) select.add(new Option(id, id));
  if (!ordered.length) {
    $('summary').innerHTML = '<span class="chip">no clip has exported augmentations: run export_round_trip_demo.py</span>';
    return;
  }
  const h = new URLSearchParams(location.hash.slice(1));
  select.value = ordered.includes(h.get('clip')) ? h.get('clip') : ordered[0];
  if (h.get('clock')) $('clock').value = h.get('clock');
  select.onchange = () => loadClip(select.value, 0);
  await loadClip(select.value, Number(h.get('f') || 1) - 1);
}

async function loadClip(id, start = 0) {
  playing = false; $('play').textContent = '▶';
  const names = Object.keys(index.clips[id].augmentations).filter((n) => index.augmentation[n]);
  let done = 0;
  const status = () => { $('summary').innerHTML = `<span class="chip">loading ${done} / ${names.length + 2}</span>`; };
  status();
  const tick = (x) => { done++; status(); return x; };
  const hasBase = Object.keys(index.clips[id].augmentations).includes('w4.base');
  const [a, b, ...bundles] = await Promise.all([
    fetchBundle(`clips/${id}/v/off`).then(tick),
    hasBase ? fetchBundle(`clips/${id}/aug/w4.base`).then(tick) : Promise.resolve(tick(null)),
    ...names.map((n) => fetchBundle(`clips/${id}/aug/${n}`).then(tick)),
  ]);
  A = a; B = b;
  clip = { id, bundles: Object.fromEntries(names.map((n, i) => [n, bundles[i]])) };
  images = await Promise.all(Array.from({ length: A.meta.n }, (_, i) => loadImage(`clips/${id}/frames/${String(i).padStart(4, '0')}.jpg`)));
  buildPanels();
  groupToggles();
  const counts = {};
  for (const p of panels) if (p.meta) counts[p.group] = (counts[p.group] || 0) + 1;
  $('summary').innerHTML = [
    `<span class="chip"><b>${panels.length - 1}</b> augmentations</span>`,
    `<span class="chip">${A.meta.n} frames @ ${A.meta.fps.toFixed(0)} fps</span>`,
    ...Object.entries(counts).map(([g, n]) => `<span class="chip">${GROUPS[g] || g} <b>${n}</b></span>`),
  ].join('');
  setClock();
  aim('front');
  show(Math.min(Math.max(start, 0), length - 1));
}

function setClock() {
  length = $('clock').value === 'source' ? A.meta.n : Math.max(A.meta.n, ...panels.filter((p) => p.meta).map((p) => p.meta.n));
  $('scrub').max = length - 1;
}

// ---------- playback ----------
function show(k) {
  t = Math.min(Math.max(k, 0), length - 1);
  $('scrub').value = t;
  $('frame-label').textContent = $('clock').value === 'source'
    ? `source frame ${t + 1} / ${A.meta.n}` : `frame ${t + 1} / ${length} (original ${A.meta.n})`;
  update();
  requestRender();
  history.replaceState(null, '', `#clip=${clip.id}&clock=${$('clock').value}&f=${t + 1}`);
}
let last = 0, acc = 0;
function loop(now) {
  if (!playing) return;
  acc += (now - last) / 1000; last = now;
  const step = 1 / (A.meta.fps * Number($('speed').value));
  if (acc >= step) { acc %= step; show((t + 1) % length); }
  requestAnimationFrame(loop);
}
function toggle() {
  playing = !playing; $('play').textContent = playing ? '❚❚' : '▶';
  if (playing) { last = performance.now(); acc = 0; requestAnimationFrame(loop); }
}

$('play').onclick = toggle;
$('prev').onclick = () => show(t - 1);
$('next').onclick = () => show(t + 1);
$('scrub').oninput = (e) => show(Number(e.target.value));
$('clock').onchange = () => { setClock(); show(Math.min(t, length - 1)); };
$('cols').onchange = () => { document.documentElement.style.setProperty('--cols', $('cols').value); safeSet('allCols', $('cols').value); requestRender(); };
for (const b of document.querySelectorAll('[data-cam]')) b.onclick = () => aim(b.dataset.cam);
$('zoom-in').onclick = () => zoom(0.8);
$('zoom-out').onclick = () => zoom(1.25);
for (const id of ['t-orig', 't-mesh', 't-pts', 't-solid']) $(id).onchange = () => show(t);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range')) return;
  if (e.key === ' ') { e.preventDefault(); toggle(); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); show(t - 1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); show(t + 1); }
});
window.addEventListener('resize', requestRender);
window.addEventListener('scroll', requestRender, { passive: true });

const cols = safeGet('allCols');
if (cols) $('cols').value = cols;
document.documentElement.style.setProperty('--cols', $('cols').value);
loadIndex();
