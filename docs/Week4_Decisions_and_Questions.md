# Week 4: decisions and questions for advisor (2026-11-11)

From the week-4 whiteboard and the 2026-11-10 brainstorms (augmentation ideas; validation framework).

## Decisions

- **Two tiers.** Tier 1 = literature ranges (yaw ±20°, pitch ±5°, mirror, duration 0.8–1.25×,
  scale 0.85–1.15×, uncentre ±0.1 frame, arm joints p=0.3 ±4°). Tier 2 = new augmentations and
  ranges ~25% beyond the proven ones (joint noise, movement size, location jitter, handshape/finger
  noise, expression jitter, timing variants, frame/hand drop).
- **Classifier sees raw MediaPipe** (normalised + unnormalised, no shoulder centring). So scale,
  uncentring and framing are *not* cancelled and should matter more than in the papers.
- **Limb length / body shape: dropped.** Signer proportion spread in MediaPipe is ~2–5%, at the
  noise floor.
- **Baselines before anything new:** train on original clips, unedited round-trip clips, and the
  literature profile; signer-held-out split, 3–5 seeds.
- **Adding MediaPipe noise back = experiment, not a fix.** The residual also holds real motion the
  fit missed; split into per-clip bias vs time-varying part and test each.
- **Render SMPL-X → re-run MediaPipe:** side experiment only (avatar is out of MediaPipe's domain).
- **Older/weak classifier:** alarm that flags clips for review, never an automatic reject.
- **Per-sign metadata:** later stage, via ASL-LEX 2.0 (handshape, location, movement, contact).
  No facial/non-manual data there; would need annotation or cross-signer spread.
- **Scaling z alone:** low priority. z is noisy but carries palm orientation and hand–face contact.
- **Camera roll:** test as an ablation arm, not a default. **Onset/offset trim:** small, idle frames only.
- **Aspect ratio/resolution:** worth testing, but both coordinate forms must come from one camera transform.
- **Validation:** pytest + NumPy, per-augmentation rules (rotation keeps bone lengths, scale
  doesn't), per-body-part coverage, and FAIL / REVIEW / NOT EVALUATED; usefulness stays
  "not measured" until classifier runs.
- **Augmentation lab** (visualise and play with augmentations): designed, plan written.

## Questions for professor

1. Can I get the classifier training code, a signer-held-out split and GPU time to run the baselines?
2. Is "25% beyond proven" OK for first experiments, or start from literature ranges only?
3. Scale and uncentre: should we push until body parts leave the frame (as in real video), or keep
   hands always in view? Does the target data ever crop hands?
4. Does the classifier actually use z? How are missing landmarks encoded (zeros, NaN, mask)?
5. What camera formats (aspect ratio, resolution) does the target data have?
6. Is there an older/weaker classifier I can use as the review alarm?
7. Can we get a Deaf signer or ASL expert to check that augmented signs keep their meaning?
8. Is per-sign metadata worth the effort now, or after baselines?
9. What counts as success: which dataset and metric (e.g. WLASL100 top-1, per-class change)?
10. Is matching MediaPipe's noise a priority, or is clean geometry acceptable for now?
