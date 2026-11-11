# KineticAugment résumé handoff

Use this brief to write a résumé entry for **Bemnet Beshah** about KineticAugment, an existing collaborative research project. The relevant work is documented in October–November 2026 commits. Confirm the formal role, affiliation, and résumé date range with Bemnet before adding them. The evidence below reflects the local `feat/week4-augmentations` branch at `aff4e84` and the experiment reports it contains.

## Project and research question

KineticAugment is a Python framework for augmenting skeletal motion data. Bemnet’s work focused on **word-level American Sign Language recognition** using WLASL video clips. The research question was whether a clip’s body, hand, and face landmarks could be mapped into a shared 3D representation, edited with anatomical and sign-specific constraints, and converted back into the landmark format used by recognition models. Handshape, motion, and location relative to the body or face can distinguish signs, so small edits need to be inspected in context.

The working pipeline extracts pose, hands, and face from video with MediaPipe, fits the observed upper body with SMPL-X and the hands with MANO, and attaches observed face landmarks in a head-local coordinate frame. It edits the fitted motion and projects the result back into MediaPipe-format landmarks. An unedited round trip provides the baseline for separating reconstruction discrepancies from changes introduced by augmentation. [Project overview](../README.md) · [Round-trip viewer](../demo/round_trip/README.md) · [Landmark reconstruction code](../src/kinetic_augment/augmentations/outputs.py)

## Contributions supported by Git history

| Contribution | What Bemnet implemented or evaluated | Evidence |
| --- | --- | --- |
| Upper-body reconstruction | Built a MediaPipe-to-SMPL-X converter using metric pose landmarks and analytic inverse kinematics. Fitted one plausible body shape per clip and measured the residual against a constant-skeleton reference. | `f1c3481`, `498a83b` · [V1 findings](V1_Findings.md) |
| Hand fitting and error analysis | Fitted MANO hands from 21 landmarks per detected hand. Evaluated a shared shape per track and a per-frame scale around the wrist to account for apparent hand-size drift in MediaPipe. Exported both variants for comparison. | `0b96b11`, `21bbcf4`, `05d0fb8` · [Hand fitting findings](V1_2_Findings.md) · [Saved variants](Hand_Pipeline_Variants.md) |
| Facial features | Implemented reversible head-local face attachment, validity masks, and synchronized feature exports. Added a masked multimodal classifier and checked batching and optimization with real feature archives and synthetic labels. | `a6d1a00`, `cab5de5`, `9b94e6f`, `9732ae8` · [Face integration results](V1_3_Face_Integration_Results.md) |
| Reproducible augmentation | Built a seeded, YAML-configurable chain with three profiles and independent draws per clip and operation. Added wrist-retargeted limb edits, coupled arm and finger perturbations, scene transforms, temporal resampling, and dropout. Logged sampled settings, skips, and fallbacks. Documented literature-backed ranges separately from exploratory settings and mapped local clips to WLASL glosses for a mirror-exclusion audit. | `fbe7140`, `78c6985`, `ea9a3fc`, `93d3dec` · [Chain implementation](../src/kinetic_augment/augmentations/chain.py) · [Augmentation ranges](Augmentation_Ranges.md) · [Mirror audit](Mirror_Unsafe_Signs.md) |
| Visual evaluation | Built an interactive viewer that compares source video, extracted landmarks, reconstructed landmarks, and the fitted 3D body. Added frame-level diagnostics and augmentation comparisons for inspecting where the fit or edit changes motion. | `adca5de`, `3dea300` · [Viewer documentation](../demo/round_trip/README.md) · [Demo recording](media/round_trip_demo.mp4) |

## Numbers that can support a bullet

| Reported result | What the number measures |
| --- | --- |
| **5.92% → 3.52%** on 5 pilot clips | Median arm-landmark reconstruction error as a share of shoulder width, before and after per-clip shape fitting under the same direct metric. Error was **3.99%** when shape was fit on one half of each clip and scored on the other. [V1 findings](V1_Findings.md#2-how-the-number-moved) |
| **3.13%** median arm error on 20 additional WLASL clips | Direct error relative to shoulder width. Holding out a time segment from the per-clip shape fit gave **3.37%**. These clips were new to the pipeline evaluation and came from the same WLASL pool, with some signer overlap among clips. [Week 2 findings](Week_2_Findings.md#2-does-the-pipeline-generalise-yes-within-wlasl) |
| **5.10 → 3.84 mm** on the 20-clip evaluation set | Pooled mean wrist-relative MANO hand-landmark error against MediaPipe, across 26 detected hand tracks. The lower figure uses an in-sample per-frame scale fitted to and stored from the same observations. It measures reproduction of detector output, including some of its apparent size drift. [Hand follow-up](V1_2_Findings.md#104-the-random-20-clips) |
| **1,549 of 1,550** detected face frames attached successfully | Coverage of the reversible face attachment. The validation processed **1,646 total frames across 25 clips**, including frames with no detected face. The counts measure pipeline coverage. [Face integration results](V1_3_Face_Integration_Results.md#real-cache-coverage) |
| **334 clips across 39 glosses** screened for mirror exclusions | A separate inventory of local WLASL clips matched to annotation metadata. Zero clips matched the documented exclusion list. This count describes the gloss audit. [Mirror audit](Mirror_Unsafe_Signs.md#3-our-clips) · [Clip-to-gloss map](../data/wlasl/clip_glosses.csv) |

## Drafting guidance

For a computer vision or research role, lead with the 3D fitting pipeline and one reconstruction result. For an ML engineering role, emphasize the seeded augmentation pipeline and the synchronized feature outputs, masks, or diagnostics. Keep the first mention of WLASL tied to ASL video clips. Use Git evidence for personal contribution and credit KineticAugment as the team project.

Two starting bullets to tailor to the role:

- Built a MediaPipe-to-SMPL-X/MANO reconstruction pipeline for ASL video and evaluated it on 20 additional WLASL clips, measuring median arm-landmark error of 3.13% of shoulder width against MediaPipe.
- Implemented three seeded YAML profiles for 3D sign-motion augmentation, with wrist-retargeted limb edits, synchronized temporal resampling, and per-operation diagnostics for visual review.

Keep claims at the level the experiments tested. Reconstruction metrics use MediaPipe as the reference. The classifier check used synthetic labels, and the repository records signer-held-out augmentation ablations as future work. Some augmentation ranges are exploratory, and the saved week-4 visual trial covers one 53-frame clip. Face attachment coverage and coordinate reversibility leave absolute depth and hand-to-face contact accuracy unresolved. Confirm any publication, release, deployment, or leadership claim with Bemnet before including it. [Classifier validation](V1_3_Face_Integration_Results.md#mixed-archive-classifier-integration) · [Open benchmark work](../TODO.md) · [Week-4 trial data](../data/augment_trials/augment_week4_trials.json)
