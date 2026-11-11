# 07: Object-Aware Constraints

**Status: planned (Phase 7 of the [roadmap](./06_Development_Roadmap.md)).** Nothing in this document is implemented yet.

## 1. Why

The sixth core principle, [Environmental Interaction](./01_Core_Principles.md), asks that a body keep a plausible relationship with the floor and with the objects it handles. Today the constraint system covers the body alone: joint limits, velocity limits, and self-collision between capsule and sphere primitives. It knows nothing about a mug, a table, or a floor.

Two uses need more:

- **Augmenting motion that involves objects.** Rotating a wrist or warping time must not push a hand through the object it holds or pull it off the handle.
- **Validating and repairing motion from other sources.** A generator of human motion (see [KineticSynth](https://github.com/imics-lab/KineticSynth)) needs a check that says where a hand penetrates an object and where an intended contact is missing, and a repair that fixes it.

## 2. What exists to build on

| Piece | Where | What it gives |
|---|---|---|
| Signed distance from hand points to a surface, with a contact threshold of 2 cm | `body_model/hand_face_geometry.py` | The geometry is already written for the head and neck surface. Replacing that surface with an arbitrary object mesh is the starting point. |
| Collision checking in PyBullet | `constraints/collision/` | The engine in which object bodies will be checked alongside body parts. |
| The constraint engine and its enforcement order | `constraints/engine.py` | Where the new constraints are registered and ordered. |

## 3. Design

### 3.1 The object module

A new subpackage, `constraints/objects/`, with three responsibilities:

1. **Load an object.** A mesh, the unit it is in, and its canonical frame. Objects are referred to by an identifier that resolves through a registry, so that a motion sequence can name the objects it involves.
2. **Bake a signed distance grid, once per object.** The planned tool is NVIDIA Warp's winding-number query, because it gives a correct inside/outside sign for scanned meshes that are not watertight, which many object scans are not.
3. **Answer queries for any set of points and any object pose.** The grid is sampled by trilinear interpolation, on the GPU. For each point it returns the signed distance (negative inside) and its gradient. From these follow:
   - *penetration depth*: how far inside the object a point is, zero outside;
   - *contact*: the point is within a threshold of the surface.

Because sampling a grid is differentiable, the same queries serve the hard checks of this document and the losses of [08](./08_Differentiable_Losses_and_Violation_Report.md).

### 3.2 Contact targets

Avoiding penetration is half of the constraint. The other half is that a hand that should be holding something is actually touching it. A *contact target* says which part of which hand should touch which object during which frames. The attraction term that pulls a hand point onto the surface uses the unsigned point-to-face distance (PyTorch3D provides one).

### 3.3 The physics path

For the checks that run in PyBullet, an object mesh is split into convex pieces (CoACD) and becomes a set of collision bodies, so that object penetration is checked in the same engine as self-collision.

### 3.4 Floor and static scene

A floor plane, and furniture as static objects handled by the same module.

### 3.5 Order of enforcement

Joint limits, velocity limits, self-collision, object penetration, contact targets. Each is available as a hard projection for repair. All but the PyBullet self-collision check are also available as a differentiable loss.

### 3.6 Independence from the body model

The constraint code should need only three things from a body model: a skinned mesh, a kinematic tree with joint-limit tables, and a differentiable forward function. SMPL-X and MANO remain the default, because the datasets with recorded contact are registered to them, but any model that offers the same three things should be usable. When writing new code, take these as inputs instead of reaching into `SMPLXWrapper`.

## 4. Tasks

**O1. Object assets and registry.** Read a mesh with its unit and canonical frame by identifier. Agree on the registry format with KineticSynth, which keeps the object files (KineticSynth `docs/05`, task D4).
*Done when* a test loads an object by identifier and reports its bounding box in meters.

**O2. Signed distance grid.** Bake, cache, and sample.
*Done when* tests pass on shapes with known answers (a sphere and a box: distance error below a stated tolerance at a stated resolution) and on a mesh with a hole (the sign is correct on both sides).

**O3. Penetration and contact queries for the body.** Given body-model parameters, an object, and its pose, return per-vertex signed distance, then penetration depth and contact per body part.
*Done when* a recorded grasp from GRAB shows contact at the fingers and near-zero penetration, and a deliberately shifted hand shows penetration that grows with the shift.

**O4. Contact targets.** A data structure for targets and the attraction term.
*Open:* where targets come from. For augmentation, the recorded contact before the transformation is the natural target. For generated motion there is no recording. Propose a definition for each case before implementing.

**O5. Object collision bodies.** Convex decomposition and registration in PyBullet next to the body parts.
*Done when* the PyBullet check and the signed-distance check agree on which frames of a test sequence penetrate.

**O6. Engine integration and repair.** Objects become an optional input of `ConstraintEngine`; validation reports object penetration and missing contact; enforcement repairs them in the order of section 3.5.
*Open:* the repair itself. Pushing a penetrating hand out along the distance gradient can break the grasp it was part of, and correcting a hand moves the arm. Write down the proposed procedure and what it guarantees before coding it.

**O7. Floor and static scene.**
*Done when* a walking sequence lowered by a few centimeters is reported as penetrating the floor and one raised is reported as floating.

**O8. Validation on recorded grasps.** Specified in [09](./09_Plausibility_Metrics_and_Validation.md).

## 5. Open decisions

| ID | Question |
|---|---|
| O-1 | The contact threshold for hands and objects. The hand-to-head code uses 2 cm. |
| O-2 | Grid resolution and padding, and how the error grows near thin parts such as handles. |
| O-3 | Which body points are tested: every vertex, a fixed subset, or a subset per body part. This sets both accuracy and cost. |
| O-4 | How articulated objects are handled: one grid per rigid part, posed by the articulation value. |
| O-5 | Packaging. The new tools are optional dependencies. PyTorch3D in particular is hard to install on some systems; decide whether to depend on it or to reuse the point-to-triangle code already in `hand_face_geometry.py`. |
| O-6 | Licenses. Record the license of every new dependency and of every object mesh before adopting it. |

---

### Next Up

➡️ **Next: [08 - Differentiable Losses and the Violation Report](./08_Differentiable_Losses_and_Violation_Report.md)**
