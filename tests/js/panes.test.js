// The panes' per-frame buffers (Lab.Panes.Figure) with the vendored three.js, no WebGL needed.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const STATIC = path.join(__dirname, '../../src/kinetic_augment/lab/static');
globalThis.THREE = require(path.join(STATIC, 'vendor/three.min.js'));
const Lab = require(path.join(STATIC, 'panes.js'));
require(path.join(STATIC, 'bundle.js'));

const DIR = process.env.LAB_FIXTURE_DIR || path.join(__dirname, 'fixtures');
const pane = () => Lab.decodePane(new Uint8Array(fs.readFileSync(path.join(DIR, 'pane.bin'))));

test('a figure fills its mesh in metres with y and z flipped', () => {
  const p = pane();
  const fig = new Lab.Panes.Figure(p, p.arrays.mesh_faces.data, 5);
  assert.ok(fig.hasMesh);
  fig.update(1);
  const pos = fig.mesh.attributes.position.array;
  const raw = p.arrays.mesh_verts.data.subarray(15, 30);
  assert.ok(Math.abs(pos[0] - raw[0] / 10000) < 1e-7);
  assert.ok(Math.abs(pos[1] + raw[1] / 10000) < 1e-7);
  assert.ok(Math.abs(pos[2] - -3) < 1e-6, 'z of vertex 0 in frame 1 (30000) becomes -3 m');
});

test('missing points are left out of the lines and dots', () => {
  const p = pane();
  const fig = new Lab.Panes.Figure(p, p.arrays.mesh_faces.data, 5);
  fig.update(1); // elbow (arm row 2) missing: bones 0-2 and 2-4 dropped
  assert.equal(fig.lines.arm.geometry.drawRange.count, (5 - 2) * 2);
  assert.equal(fig.dots.arm.geometry.drawRange.count, 5);
  fig.update(3);
  assert.equal(fig.lines.hands.geometry.drawRange.count, 42 * 2);
  assert.equal(fig.dots.hands.geometry.drawRange.count, 42);
});

test('a frame without a body fit still draws the points that exist', () => {
  const p = pane();
  const fig = new Lab.Panes.Figure(p, p.arrays.mesh_faces.data, 5);
  fig.update(2); // body_ok[2] = 0: only the mesh is left out; the right hand is missing
  assert.equal(fig.ok, false, 'the mesh is hidden on this frame');
  assert.equal(fig.dots.arm.geometry.drawRange.count, 6);
  assert.equal(fig.dots.hands.geometry.drawRange.count, 21, 'the left hand draws');
  assert.equal(fig.lines.hands.geometry.drawRange.count, 21 * 2);
  assert.equal(fig.dots.face.geometry.drawRange.count, 3);
});

test('a mirrored pane reverses the triangle winding; a wrong vertex count drops the mesh', () => {
  const p = pane();
  p.meta.mirrored = true;
  const fig = new Lab.Panes.Figure(p, p.arrays.mesh_faces.data, 5);
  assert.deepEqual([...fig.mesh.index.array.slice(0, 3)], [0, 2, 1]);
  assert.deepEqual([...p.arrays.mesh_faces.data.slice(0, 3)], [0, 1, 2], 'the base faces are untouched');
  assert.equal(new Lab.Panes.Figure(p, p.arrays.mesh_faces.data, 6).hasMesh, false);
});

test('only the mesh follows the body fit; the original is a ghost, the augmented solid', () => {
  const p = pane();
  const fig = new Lab.Panes.Figure(p, p.arrays.mesh_faces.data, 5);
  const scene = new THREE.Scene();
  const t = { mesh: true, points: true, skeleton: true, face: true, hands: true };
  const ghost = new Lab.Panes.Drawn(scene, fig, '#4a8fe7', 'ghost', 1);
  const solid = new Lab.Panes.Drawn(scene, fig, '#e8913a', 'solid', 0);
  fig.update(2); // no body fit
  solid.visible(true, t);
  assert.equal(solid.mesh.visible, false);
  assert.equal(solid.dots.hands.visible, true);
  assert.equal(solid.lines.arm.visible, true);
  fig.update(1);
  solid.visible(true, t);
  ghost.visible(true, t, false);
  assert.equal(solid.mesh.visible, true);
  assert.equal(solid.meshMat.transparent, false, 'augmented mesh is opaque');
  assert.ok(ghost.meshMat.transparent && ghost.meshMat.opacity < 0.3, 'original mesh is a faint ghost');
  ghost.visible(true, t, true);
  assert.ok(ghost.meshMat.opacity >= 0.5, 'a ghost on its own is drawn stronger');
  ghost.dotSize(6);
  assert.equal(ghost.dotMat.size, 6);
  assert.equal(ghost.dotMat.depthTest, false, 'points draw over the meshes');
});
