// Plain-words explanations of the week-4 runs written by scripts/demo_week4.py, shared by the
// augmentations page (augment.html) and the all-augmentations grid (all.html). Keyed by the
// run's name (aug/<name>/); the proven and used ranges are those of docs/Augmentation_Ranges.md.

export const GROUPS = {
  tier1: 'Tier 1 (proven ranges)', body: 'week 4 · body', fingers: 'week 4 · fingers', face: 'week 4 · face',
  temporal: 'week 4 · temporal', scene: 'week 4 · scene', combined: 'week 4 · whole profiles',
};

const TIMING_KEEPS = ['every pose the clip passes through, in the same order', 'the start and the end: nothing is cropped',
  'the body, handshapes and face of each frame'];
const SCENE_KEEPS = ['the pose: every joint angle and bone length', 'the handshapes and face', 'the timing'];

export const W4_EXPLAIN = {
  'body.joint_coupled_noise': {
    does: 'Turns the shoulder, elbow and wrist of both arms by one draw for the whole clip. The draws are coupled: shoulder and wrist share part of a common draw, and the two arms are correlated (mirrored), so the arms change together the way one signer’s would.',
    changes: ['the arm posture, so each hand moves a few centimetres and turns with the wrist'],
    keeps: ['the finger poses (handshape)', 'the torso, head and face', 'the bone lengths', 'the timing'],
    proven: '±4° shoulder/wrist at p = 0.3 (SPOTER [M1]); ±7.5° elbow [T]',
    used: 'shoulder/wrist σ 2.5°, ≤ 5°; elbow σ 4.75°, ≤ 9.5°',
  },
  'body.limb_length_scaling': {
    does: 'Lengthens or shortens the upper arm and forearm, as for a signer with other proportions, then re-solves each arm with two-bone IK so the wrist stays where it was.',
    changes: ['the arm bone lengths and where the elbows are'],
    keeps: ['the handshapes and face', 'the timing'],
    proven: 'no literature range; the between-signer spread in this project’s data is 5.7% (Grounding §6.1)',
    used: '0.93–1.07',
    note: 'An opt-in experiment arm: off in the default profile.',
  },
  'body.movement_size': {
    does: 'Makes every movement bigger or smaller: each wrist’s path is scaled about its own centre, and the arm is re-solved to follow it.',
    changes: ['how far the hands travel'],
    keeps: ['where the movement happens (its centre)', 'the handshapes and face', 'the timing'],
    proven: 'no measured range; 0.85–1.15 is Grounding’s linguistic example [L2.10, L2.11]',
    used: '0.8–1.2, log-uniform',
  },
  'body.location_jitter': {
    does: 'Moves where the signing happens by one 3-D offset for the whole clip, in the torso frame, and re-solves the arms so the hands follow.',
    changes: ['where the hands are relative to the body, by up to 25 mm'],
    keeps: ['the shape and size of every movement', 'the handshapes and face', 'the timing'],
    proven: 'σ = 10 mm, ≤ 20 mm (production spread [T])',
    used: 'σ = 12.5 mm, ≤ 25 mm',
  },
  'fingers.joint_coupled_noise': {
    does: 'Bends or straightens each finger a little, the same for the whole clip. The knuckle (MCP) and middle joint (PIP) get one draw each, neighbouring fingers are correlated, and the end joint (DIP) follows the PIP.',
    changes: ['the finger bends, by a few degrees'],
    keeps: ['the arms, wrists and face', 'the timing'],
    proven: 'no published range',
    used: 'σ 3.75°, ≤ 7.5° (an estimate of 3° / 6°, widened)',
    note: 'The change is a few degrees per joint: use the hand views to see it.',
  },
  'fingers.handshape_jitter': {
    does: 'Jitters every finger joint while keeping the handshape class: which fingers are selected, open or closed, their spread and the thumb. A draw that would change the class is drawn again; after 5 tries the clip is left unchanged.',
    changes: ['the joints of the non-selected fingers and the thumb most (σ 7.5°), the selected fingers less'],
    keeps: ['the handshape class', 'the arms, wrists and face', 'the timing'],
    proven: 'no published range; GISLR’s “finger-tree rotate” gives no angle [T]',
    used: 'non-selected and thumb σ 7.5°, selected σ 3.75°, selected flexion σ 2°, all ≤ 2σ',
  },
  'face.expression_jitter': {
    does: 'Shifts the facial expression a little for the whole clip: the brow, eye and cheek/nose blendshapes get one offset each, and the face points move by a linear map from blendshapes to points fitted on the clip. Mouth and jaw stay as they are, since mouth patterns can change the sign.',
    changes: ['the brows, eyelids and cheeks, by a few millimetres', 'the blendshape scores'],
    keeps: ['the mouth and jaw', 'the head pose and the rigid face points', 'the body, hands and timing'],
    proven: 'no published range; lexical mouth patterns excluded [L7.3, L1.5]',
    used: 'brows and eyes σ 0.0625, ≤ 0.125; cheeks/nose σ 0.0375, ≤ 0.075; mouth/jaw off',
    note: 'The change is a few millimetres: use the face view to see it.',
  },
  'temporal.time_warp.uniform': {
    does: 'Plays the whole clip faster or slower by one factor, interpolating the new frames (joint rotations by slerp).',
    changes: ['the clip’s length and the speed of every movement, uniformly'],
    keeps: TIMING_KEEPS,
    proven: '0.67–1.5×; 0.5× hurt once [L6.2, M6, M3, T]',
    used: '0.6–1.65×, log-uniform',
  },
  'temporal.time_warp.varied': {
    does: 'Changes the speed smoothly over the clip: four speed knots of 0.7–1.43× each, on top of the uniform factor. The last 20% of the clip keeps the uniform speed.',
    changes: ['the clip’s length, and which parts are fast or slow'],
    keeps: TIMING_KEEPS,
    proven: 'none: non-uniform warps are excluded in the literature [L6.3, L6.4]',
    used: 'knot rates 0.7–1.43×, log-uniform',
    note: 'An opt-in experiment arm: off in the default profile.',
  },
  'temporal.time_warp.segment': {
    does: 'Slows down or speeds up one window of the clip (20–60% of it) by one factor, on top of the uniform factor.',
    changes: ['the clip’s length, and the speed inside the window'],
    keeps: TIMING_KEEPS,
    proven: 'none: non-uniform warps are excluded in the literature [L6.3, L6.4]',
    used: 'window 20–60% of the clip, factor 0.6–1.65×',
    note: 'An opt-in experiment arm: off in the default profile.',
  },
  'temporal.frame_drop.rate': {
    does: 'Drops frames down to a lower frame rate: keeps the frames nearest a 12–25 fps grid, never below 8 fps, and never opens a gap over 125 ms.',
    changes: ['the number of frames and the gaps between them'],
    keeps: ['every kept frame exactly as it was', 'the first and last frames'],
    proven: '≥ 10 fps, where intelligibility stops improving [T]; masked runs ≤ 8 frames [M4, M10]',
    used: '12–25 fps, never below 8 fps, gaps ≤ 125 ms',
    note: 'The kept frames keep their timestamps; played back to back at the clip’s frame rate, the clip looks faster.',
  },
  'temporal.frame_drop.random': {
    does: 'Drops frames at random, each with p 0–0.2, never opening a gap over 125 ms.',
    changes: ['the number of frames and the gaps between them'],
    keeps: ['every kept frame exactly as it was', 'the first and last frames'],
    proven: 'masked runs ≤ 8 frames [M4, M10]; no range for the drop probability',
    used: 'p 0–0.2, gaps ≤ 125 ms',
    note: 'The kept frames keep their timestamps; played back to back at the clip’s frame rate, the clip looks faster.',
  },
  'temporal.hand_dropout': {
    does: 'Removes a hand for short runs of frames, as when MediaPipe loses it mid-sign. Runs start with a set chance per frame, last 1.9 frames on average and at most 10.',
    changes: ['the hand in the dropped frames: missing (NaN) and marked invalid'],
    keeps: ['the body, face and the other hand', 'every other frame', 'the timing'],
    proven: 'runs ≤ 8 frames [M4, M10]; the start rate has no literature value, measured 0.039 here',
    used: 'runs of 1–10 frames, start rate 0–0.05 per frame',
  },
  'scene.mirror': {
    does: 'Reflects the body left to right about the vertical plane through the shoulders and swaps the sides: hands, pose points, face points and the Left/Right blendshapes.',
    changes: ['which hand does what', 'each handshape and the face become their mirror images'],
    keeps: ['the shape and size of every movement (mirrored)', 'the timing'],
    proven: 'p = 0.5 [M6]; cost 1.3 pp on WLASL once [T]',
    used: 'p = 0.5',
  },
  'scene.view': {
    does: 'Turns the whole scene about the vertical axis through the shoulders (yaw) and tilts it (pitch), as if the camera stood elsewhere.',
    changes: ['the body’s orientation to the camera'],
    keeps: SCENE_KEEPS,
    proven: 'yaw ±20° (Spoken2Sign [M7]); pitch ±5° [T]',
    used: 'yaw ±25°, pitch ±6°',
  },
  'scene.scale': {
    does: 'Scales the whole scene about the shoulder centre, as if the camera were nearer or farther.',
    changes: ['the overall size'],
    keeps: ['the proportions: every position relative to every other', 'the pose, handshapes and timing'],
    proven: '0.8–1.2 (GISLR [M10])',
    used: '0.75–1.25',
  },
  'scene.uncentre': {
    does: 'Moves the whole scene sideways and up or down, as if the signer stood elsewhere in the frame. The shift is limited so no hand point inside the frame leaves it.',
    changes: ['where the signer is in the frame'],
    keeps: ['the pose, size and handshapes', 'every position relative to every other', 'the timing'],
    proven: '±0.1 of the frame (GISLR [M10])',
    used: '±0.125 of the frame, hands kept in it',
  },
  'combined.default': {
    does: 'The default profile as one augmented sample: each entry draws from its own seeded stream and runs in the chain’s fixed order (fingers, arms, face, scene, timing).',
    changes: [], keeps: [], proven: 'see each entry', used: 'slr_week4_default.yaml',
  },
  'combined.ambitious': {
    does: 'The ambitious profile as one augmented sample: every entry on at the used ranges (about 25% past the proven ones), with all three time-warp modes and both frame-drop modes possible.',
    changes: [], keeps: [], proven: 'see each entry', used: 'slr_week4_ambitious.yaml',
  },
  'combined.literature': {
    does: 'The literature profile as one augmented sample: the default profile’s entries at the proven ranges; entries with no proven range are off.',
    changes: [], keeps: [], proven: 'the ranges themselves', used: 'slr_week4_literature.yaml',
  },
};

const fin = (v) => v !== null && v !== undefined && Number.isFinite(v);
const f = (v, d = 1) => (fin(v) ? v.toFixed(d) : '–');
const STILL_MM = 0.01; // below this a measured movement is rounding, not a change

// What a run measurably changed, from its metrics (kinetic_augment.augmentations.report).
export function measuredChanges(aug, n0) {
  const m = aug.metrics || {};
  const out = [];
  if (aug.n !== n0) out.push(`frames: ${n0} → ${aug.n}`);
  if (fin(m.wrist_moved_mm_max) && m.wrist_moved_mm_max >= STILL_MM) out.push(`the wrists moved at most ${f(m.wrist_moved_mm_max, 1)} mm`);
  if (fin(m.bone_length_change_mm_max) && m.bone_length_change_mm_max >= STILL_MM) out.push(`arm bone lengths changed at most ${f(m.bone_length_change_mm_max, 1)} mm`);
  if (fin(m.face_rigidity_mm) && m.face_rigidity_mm > 0.05) out.push(`the face points spread ${f(m.face_rigidity_mm, 2)} mm more (expression)`);
  if (fin(m.jitter_ratio) && Math.abs(m.jitter_ratio - 1) > 0.01) out.push(`wrist jitter ×${f(m.jitter_ratio, 2)}`);
  return out;
}

// What a run measurably left alone: the aligned measures under STILL_MM.
export function measuredKeeps(aug) {
  const m = aug.metrics || {};
  const out = [];
  if (fin(m.wrist_moved_mm_max) && m.wrist_moved_mm_max < STILL_MM) out.push('where the wrists are (moved under 0.01 mm)');
  if (fin(m.bone_length_change_mm_max) && m.bone_length_change_mm_max < STILL_MM) out.push('the arm bone lengths (changed under 0.01 mm)');
  return out;
}

export function appliedText(aug) {
  return (aug.applied || []).map((a) => a.replace(/^(\w+)\./, '$1 ').replace(/_/g, ' ')).join(', ') || 'nothing';
}

// A short readable form of a draw dict: numbers rounded, long lists summarised.
export function drawText(draw) {
  if (!draw) return '';
  if (draw.gated) return 'not drawn (the probability gate)';
  const parts = [];
  for (const [k, v] of Object.entries(draw)) {
    if (v === null || v === undefined || k === 'noise') continue;
    let s;
    if (typeof v === 'number') s = Number.isInteger(v) ? String(v) : v.toPrecision(3);
    else if (typeof v === 'boolean' || typeof v === 'string') s = String(v);
    else if (Array.isArray(v)) {
      const flat = v.flat(3).filter((x) => typeof x === 'number');
      if (k === 'joints') s = v.map((j) => `${j.side[0].toUpperCase()} ${j.joint} ${j.angle_deg.toFixed(1)}°`).join(', ') || 'none';
      else if (k === 'keep') s = `${v.length} frames`;
      else if (k === 'runs') s = v.map(([h, t0, len]) => `${h ? 'R' : 'L'} @${t0}+${len}`).join(', ') || 'none';
      else if (flat.length > 6) s = `${flat.length} values, |max| ${Math.max(...flat.map(Math.abs)).toFixed(3)}`;
      else s = `[${flat.map((x) => x.toPrecision(3)).join(', ')}]`;
    } else if (typeof v === 'object') {
      const inner = Object.entries(v).map(([a, b]) => {
        const flat = [b].flat(3).filter((x) => typeof x === 'number');
        return `${a} ${flat.length > 3 ? `|max| ${Math.max(...flat.map(Math.abs)).toFixed(2)}` : flat.map((x) => x.toPrecision(3)).join('/')}`;
      });
      s = inner.join(', ');
    }
    parts.push(`${k.replace(/_/g, ' ')}: ${s}`);
  }
  return parts.join(' · ');
}
