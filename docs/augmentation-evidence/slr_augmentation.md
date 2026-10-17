# Pose / skeleton augmentation for sign language recognition: verified literature notes

Compiled 2026-10-11. Purpose: tell KineticAugment (MediaPipe → SMPL-X body + MANO hands) which augmentations the literature says keep a sign's meaning, what parameter ranges people use, and what ablations show.

**Marking convention**
- **VERIFIED** means I read the claim myself in the paper PDF, the abstract page, or the official code. Section and table numbers are given.
- **UNVERIFIED** means it came from a search snippet or from memory and I did not open the source. Venue metadata I did not confirm is flagged inline.
- Gains are absolute percentage points (pp) of top-1 accuracy unless stated otherwise.
- I computed every "Δ" myself as a difference between two numbers printed in the paper's own table.

---

## 1. SPOTER: Boháček & Hrúz, "Sign Pose-based Transformer for Word-level Sign Language Recognition", WACV Workshops (HADCV) 2022

- Paper: https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf (read in full, VERIFIED)
- Code: https://github.com/matyasbohacek/spoter (files read: `augmentations/__init__.py`, `datasets/czech_slr_dataset.py`, `train.py`, `spoter/gaussian_noise.py`; VERIFIED)

**Input.** 2D landmarks in frame-relative [0,1] coordinates (Apple Vision pose). Missing landmarks are filled with zeros.

**Augmentations (paper §4.2).** One random parameter set is drawn per sign instance and kept constant across all frames.

| Augmentation | Paper range | Code default (`czech_slr_dataset.py`) |
|---|---|---|
| In-plane rotation about frame centre (0.5, 0.5) | θ "up to 13 degrees" | `augment_rotate(..., (-13, 13))`, uniform |
| Squeeze (cut from left and right sides independently, x only) | each side "up to 15%" of width | `augment_shear(..., "squeeze", (0, 0.15))` |
| Perspective transform (simulated camera tilt; left or right side chosen at random) | the paper says the proportion is drawn from [0, 1) | `augment_shear(..., "perspective", (0, 0.1))`. **The code uses 0–10%, which disagrees with the paper text.** |
| Sequential arm-joint rotation (walk down the arm chain and rotate each distal joint about its parent) | each joint rotated with probability 3/10, angle uniform ±4° | `augment_arm_joint_rotate(..., 0.3, (-4, 4))` |
| Gaussian noise (added after normalisation; the paper treats it as separate from the "augmentations") | mean 0, std 10⁻³ (footnote 4) | `GaussianNoise(0, 0.001)` in `train.py` |

- The code augments a sample with probability 0.5. When it does, it picks exactly one of the four augmentations uniformly at random (`randrange(4)`). They are never stacked.
- The paper motivates arm-joint rotation as simulating "slight, negligible variances" in execution "which do not change its semantic meaning".
- The paper's normalisation (§4.3) is separate from augmentation: body landmarks are projected into the signing space, and each hand is normalised in its own local bounding box.

**Ablation (Table 6, WLASL100, top-1 macro).** Number of runs and variance are not stated.

| Model | Accuracy | Δ vs B |
|---|---|---|
| A: no normalisation, no augmentation | 44.96 | |
| B: normalisation only | 58.97 | (+14.01 vs A) |
| C: B + rotation | 61.24 | +2.27 |
| D: B + squeeze | 60.85 | +1.88 |
| E: B + perspective | 60.47 | +1.50 |
| F: B + arm-joint rotation | 61.24 | +2.27 |
| G: B + all four | 62.79 | +3.82 |
| H: G + Gaussian noise | 63.18 | +0.39 vs G |

- In the ablation, each single augmentation was applied "with the chance of 0.5".
- The authors found that generic augmentations (rotation, squeeze) helped about as much as the sign-specific ones (perspective, arm rotation): roughly 1–2 pp each.
- Normalisation (+14 pp) mattered far more than any augmentation.
- They report 100% on LSA64 but give no augmentation ablation for that dataset.

---

## 2. OpenHands: Selvaraj, NC, Kumar, Khapra, ACL 2022 (2022.acl-long.150)

- ACL version: https://aclanthology.org/2022.acl-long.150.pdf (augmentation passages read, VERIFIED)
- arXiv v1 (submitted to AAAI-22): https://arxiv.org/abs/2110.05877 (VERIFIED)
- Code: https://github.com/AI4Bharat/OpenHands, files `openhands/datasets/pose_transforms.py` and `examples/configs/wlasl/decoupled_gcn.yaml` (VERIFIED)

**Input.** 27 MediaPipe Holistic keypoints, 2D. `CenterAndScaleNormalize` uses the shoulders as the reference.

**Augmentations in the library:**
- ShearTransform: shear factor s_x ~ N(0, `shear_std`). Default std 0.2; the WLASL config uses 0.1.
- RotatationTransform: θ ~ N(0, `rotation_std`) in radians. Default 0.2; the WLASL config uses 0.1 rad, which is about 5.7° std.
  - **Discrepancy.** arXiv v1 §5.2 says the angle is chosen "from −π/3 to π/3". The released code samples a Gaussian with std 0.1–0.2 rad. The ACL camera-ready shows `rotation_std: 0.1` in its example config.
- ScaleTransform: std 0.2. It is commented out in the WLASL config, and the paper says scaling is "generally not necessary" once CenterAndScaleNormalize is applied.
- RandomMove: shift drawn from (−2.5, 2.5) in steps of 0.5. It is also commented out in the WLASL config.

**Temporal tools:**
- PoseRandomShift: shifts the sequence by a time offset.
- Uniform or random temporal subsample, and FrameSkipping.
- Presented as robustness to inaccurate segmentation. No ablation.

**Horizontal flip.** OpenHands has no flip transform.

**Ablation.**
- None for augmentations.
- The paper's contrastive pretraining used shear, scale and rotation as positive-pair views. It converged, but fine-tuning on INCLUDE showed no gain over training from scratch, and the embeddings did not cluster by class (arXiv §"Contrastive-learning based", VERIFIED).
- Implication: geometric-invariance views alone did not teach sign semantics.

---

## 3. ASL Citizen: Desai, Berger, Minakov, Milan, Singh, Pumphrey, Ladner, Daumé III, Lu, Caselli, Bragg

- Venue (NeurIPS 2023 Datasets & Benchmarks): from the brief, not confirmed on the arXiv page.
- Source: https://arxiv.org/abs/2304.05934 (PDF §4 read, VERIFIED)

**I3D (RGB) baseline:**
- Random horizontal flips "to simulate left and right handed signers".
- The authors **deliberately avoided random temporal crops**. They argue cropping "might alter sign semantics" because compound signs could be cut down to one root sign.
- Instead they standardise length by frame skipping: every 2nd frame if the video is longer than 95 frames, every 3rd if longer than 159, then pad or trim to 64.

**ST-GCN (pose) baseline:**
- 27 MediaPipe keypoints following OpenHands, centre-scaled by shoulder distance, capped at 128 frames with even downsampling.
- "As with OpenHands, we apply random shearing and rotation".
- The paper does not state the parameters; presumably OpenHands defaults, but that is UNVERIFIED.

**Ablation.**
- None for augmentations.
- Results: I3D 63.10% Rec@1 and ST-GCN 59.52% Rec@1 on 2,731 glosses with unseen signers.

---

## 4. SignBERT (Hu, Zhao, Zhou, Wang, Li, 2021) and SignBERT+ (Hu et al., IEEE TPAMI 2023)

- SignBERT: https://arxiv.org/abs/2110.05382 (PDF read, VERIFIED). Venue: ICCV 2021, UNVERIFIED (not stated in the text I read).
- SignBERT+: https://arxiv.org/abs/2305.04868 (PDF read, VERIFIED). The PDF header reads "IEEE Transactions on Pattern Analysis and Machine Intelligence, April 2023".

**What it is.** Self-supervised pretraining on 2D hand-pose sequences. The decoder is "hand-model-aware": it regresses MANO pose θ and shape β plus a weak-perspective camera, and reconstructs the 2D joints. These are corruption strategies for pretraining, not label-preserving augmentations for the classifier, but they are the closest published use of MANO in sign language recognition.

**SignBERT masking:**
- 50% of tokens are chosen. Each chosen token gets, with equal probability, masked-joint modelling, masked-frame modelling, or identity.
- Masked joint: m joints, m ∈ [1, M]. Each chosen joint is either zero-masked or given a "random spatial disturbance" (magnitude not stated in what I read).
- Masked frame: the whole hand token is zeroed.
- Downstream training samples 32 frames: random sampling at train time, centre sampling at test.

**SignBERT ablation (Table 4, MSASL, per-instance top-1):**

| Setting | MSASL100 | MSASL200 | MSASL1000 |
|---|---|---|---|
| No pretraining | 63.01 | 57.69 | 41.85 |
| Joint masking only | 72.66 | 68.51 | 48.87 |
| Frame masking only | 74.77 | 68.65 | 49.02 |
| Both | 76.09 | 70.64 | 49.54 |

- A MANO-aware decoder beat a fully connected decoder (Table 5; numbers not transcribed here).

**SignBERT+ additions:**
- Masked clip modelling: k consecutive tokens zeroed, k ∈ [2, K].
- Table 4: K = 8 was best (MSASL1000 61.94 vs 60.43 at K = 4 and 60.93 at K = 12).
- Table 7, masking ratio R: R = 40% was best (MSASL1000 P-I 62.42; 20% gave 59.52, 60% gave 59.80).
- Table 8: no pretraining → 53.31 (MSASL1000 P-I); joint+frame+clip → 62.42.
- Table 3: MANO pose θ dimension of 25 PCA components was best (MSASL1000 62.42 vs 61.65 at 15 and 61.60 at 35).
- Keypoints come from MMPose: 133 points, of which 42 are hand joints.

---

## 5a. WLASL / Pose-TGCN: Li, Rodriguez Opazo, Yu, Li, WACV 2020

- Source: https://arxiv.org/abs/1910.11006 (PDF §4.3.1 read, VERIFIED)
- RGB models (VGG-GRU, I3D): random 224² crop plus horizontal flip with p = 0.5, applied to the whole video.
- Pose-GRU and Pose-TGCN: the only augmentation is a randomly selected window of **50 consecutive frames** (temporal crop).
- Flip is described only for the RGB models.
- No augmentation ablation.

## 5b. SAM-SLR / SL-GCN: Jiang, Sun, Wang, Bai, Li, Fu, CVPR Workshops 2021 (ChaLearn LAP, AUTSL)

- Paper: https://arxiv.org/abs/2103.08833 (PDF read, VERIFIED)
- Code: https://github.com/jackyjsy/CVPR21Chal-SLR, files `SL-GCN/feeders/feeder.py`, `SL-GCN/feeders/tools.py`, `SL-GCN/config/sign/train/train_joint.yaml` (VERIFIED)

**Paper (§4.x).** "Randomly sampling, mirroring, rotating, scaling, jittering and shifting". Uses 27 nodes from HRNet whole-body keypoints, sample length 150, coordinates normalised to [−1, 1].

**What the released `train_joint.yaml` actually enables:**
- `random_choose`: window of 100 frames.
- `random_mirror` with p = 0.5. The code reflects x (`512 − x`) **and swaps left/right joints through `flip_index`**: shoulders, elbows and wrists swap, and the two 10-point hand sets swap.
- `random_shift`: uniform ±10 px translation in x and y.
- `random_move` (from ST-GCN `tools.py`: angle ∈ {−10, −5, 0, 5, 10}°, scale ∈ {0.9, 1.0, 1.1}, translate ∈ {−0.2 … 0.2}) is **not enabled** in this config. So the "rotating/scaling" in the paper text does not appear in this released config.

**Ablation (Table 3, AUTSL validation, joint stream):**
- Full SL-GCN: 95.02.
- "w/o Keypoints Augmentation": 90.16 (−4.86 pp).
- The augmentations are not ablated individually.

## 5c. NLA-SLR: Zuo, Wei, Mak, CVPR 2023

- Source: https://arxiv.org/abs/2303.12080 (PDF read, VERIFIED)

**Input and augmentations:**
- Inputs are RGB plus HRNet keypoint *heatmaps* (σ = 4), not coordinate vectors.
- Spatial crop with scale range [0.7–1.0] and temporal crop, applied identically to video and heatmaps.
- Intra-modality Mixup with λ ~ Beta(0.8, 0.8).
- Inter-modality mixup: blends visual features with gloss text embeddings.
- I found no horizontal flip described.

**Ablation (WLASL, "due to its large vocabulary size", per-instance top-1; Tables 5–6):**
- Sign mixup overall (VKNet + language-aware label smoothing): 58.41 → 61.05 (+2.64).
- Intra-modality Mixup only: 59.56 (+1.15).
- Inter-modality only: 59.66 (+1.25).
- Language-aware label smoothing alone: 57.19 → 58.41 (+1.22).

---

## 6. Horizontal flip / hand-dominance mirroring

No paper I found ablates L/R-swapped mirroring for *pose-based isolated* SLR on a dataset with many left-handed signers. Here is the evidence that does exist.

| Source | Setting | Finding | Status |
|---|---|---|---|
| Laines, Gonzalez-Mendoza, Ochoa-Ruiz, CVPR-W (LatinX) 2023. https://arxiv.org/abs/2304.05403 (CVF version listed in search) | WLASL100, skeleton → tree-structure skeleton image (TSSI) + DenseNet; 5-fold CV | Removing **Flip** (horizontal, p = 0.5): 81.47 → 80.78 (**−0.69**). Removing **Scale** (factor 0.5–1.0): 81.25 (−0.22). Removing **Speed**: 65.82 (**−15.65**). All augmentation without pretraining: 39.15 → 58.45. | VERIFIED (Table 5–6). The paper does not say whether L/R joint labels were swapped. |
| Moryossef et al., "Evaluating the Immediate Applicability of Pose Estimation for SLR", CVPR-W (ChaLearn) 2021. CVF PDF | AUTSL, pose + BiLSTM | The only augmentation was horizontal frame flip "to account for left-handed signers". No ablation. | VERIFIED |
| Min, Hao, Chai, Chen, "Visual Alignment Constraint for CSLR" (arXiv 2104.02330; ICCV 2021 per citations, venue UNVERIFIED) | PHOENIX14, RGB, continuous | Appendix Table 9: flip still helps although every signer is right-dominant. Crop alone 23.8/24.6 WER → crop+flip 23.2/23.8 (about 0.6 WER better). All three augmentations: 22.1/23.0. | VERIFIED |
| Bowen Shi, PhD thesis, TTIC 2023. https://arxiv.org/abs/2308.12419 | ChicagoFSWild fingerspelling, RGB; 10.6% of signers left-handed | Flipping all left-handed videos to right and training a right-only model gave 76.0 ± 0.2% letter accuracy vs 76.1 ± 0.3% for the original model. Handedness normalisation changed nothing. | VERIFIED |
| ASL Citizen (above) | I3D | Flip used "to simulate left and right handed signers". No ablation. | VERIFIED |
| SAM-SLR code (above) | AUTSL SL-GCN | Mirror with p = 0.5 *with* L/R joint swap; bundled into the −4.86 pp "w/o keypoint aug" result. | VERIFIED |
| Kaggle GISLR 1st place, Hoyeol Sohn. https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution (notebook read) | ASL Signs (MediaPipe 543 landmarks, 250 signs) | `flip_lr` reflects x = 1 − x and swaps L/R hand, lips, pose, eyes and nose landmarks; applied with p = 0.5. No ablation in the repo. Not peer-reviewed. | VERIFIED (code) |
| AimCLR (generic action recognition) | NTU | Spatial flip is justified by "throwing with the left hand and throwing with the right hand" being the same action. That is the same assumption mirroring makes for sign (L-dominant ≡ R-dominant). | VERIFIED |
| Fingerspelling PoseNet (arXiv 2311.12128): picks the dominant hand by motion energy and flips left-handed signers' x | fingerspelling | Search snippet only | UNVERIFIED |

**Synthesis.**
- Mirroring with L/R label swap is universally *used* and treated as meaning-preserving for lexical signs, because left-dominant signers mirror right-dominant ones.
- Measured gains are small: about 0.7 pp on WLASL100 in Laines et al., and about 0.6 WER on PHOENIX.
- Normalising handedness in fingerspelling showed no effect.
- No source reports flip *hurting*. None tests the edge cases either: directional or indexing signs, or two-handed asymmetric signs where the non-dominant hand is a base.

---

## 7. Parametric body/hand models (SMPL-X / MANO) for sign data

| Work | What it does | Relevance and evidence | Status |
|---|---|---|---|
| **SGNify**: Forte, Kulits, Huang, Choutas, Tzionas, Kuchenbecker, Black, "Reconstructing Signing Avatars From Video Using Linguistic Priors" (CVPR 2023 per brief). https://arxiv.org/abs/2304.10482 | SMPLify-X style fitting with two linguistic hand priors drawn from Battison's sign classes (8 classes = one- vs two-handed × whether handshape changes): (1) **symmetry**: when both hands move, the finger articulation must be identical; (2) **hand-pose invariance**: the handshape is static or transitions smoothly, penalised against a "reference pose sequence". A classifier predicts the sign class. | Directly usable as **constraints for MANO-space augmentation**: keep the two hands' poses tied for symmetric classes, and only perturb within the handshape invariance. The paper evaluates reconstruction quality, not recognition gains from augmentation. | VERIFIED (intro §1) |
| **Spoken2Sign**: Zuo, Wei, Chen, Mak, Yang, Tong, "A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars" (ECCV 2024 per ECVA listing in search). https://arxiv.org/abs/2401.04730 | Fits SMPL-X ("SMPLSign-X") to sign videos. **"3D keypoint augmentation"**: rotate the global orientation by δ ~ U[−20°, 20°], then project to 2D. | **Only verified ISLR gain from SMPL-X-space augmentation that I found.** Table 7, NLA-SLR-Keypoint-64, dev sets: WLASL 46.20 → 47.66 top-1 (**+1.46**); MSASL 51.81 → 53.10 (**+1.29**). The rotation axis is not specified beyond "global orientation". | VERIFIED |
| **SignAvatars**: Yu, Huang, Cheng, Birdal (ECCV 2024 per ECVA listing). https://arxiv.org/abs/2310.20436 | 3D holistic SMPL-X + MANO dataset (includes ~21K word-level clips sourced from WLASL and others). Fitting uses **biomechanical hand constraints**: bone length, palm curvature, joint-angle limits, after Spurr et al. | A source of plausibility bounds for MANO perturbation. It is a dataset and benchmark, not an augmentation study. | VERIFIED (§3 grep) |
| **NGT200**: Ranum, Wessels, Otterspeer, Bekkers, Roelofsen, Andersen. https://arxiv.org/abs/2409.15284 | Multi-view NGT isolated sign recognition. A synthetic avatar (3D-LEX mocap → Ready Player Me, rendered in BabylonJS) adds views and a signer. | Table 6, SL-GCN, unseen signer 3: adding one frontal synthetic view raised front-view top-1 from 0.14 to 0.27, close to adding a real SignBank signer (0.28). More synthetic views helped more. The authors caution synthetic data "could introduce false motion patterns". | VERIFIED |
| **HandCraft**: Rios et al. https://arxiv.org/abs/2508.14345 | Lightweight conditional pose generator (CMLPe). Synthetic pretraining, then fine-tuning with rotation ±5° and scale ±5%. | Reports consistent gains on INCLUDE, LSFB and DiSPLaY, and additive gains with classical augmentation (exact per-dataset deltas not transcribed). Pose generator, not SMPL-X/MANO. | VERIFIED (abstract/highlights, §4 lines) |
| **DexAvatar**: Kundu et al., WACV 2026 per arXiv comment. https://arxiv.org/abs/2512.21054 | Learned 3D hand and body priors for biomechanically accurate sign reconstruction. | +35.11% pose estimation on the SGNify mocap set. No recognition results. | VERIFIED (abstract) |
| **SignNet-1M**: He et al., arXiv 2606.24361 (June 2026) | Augmentation via 3D Gaussian Splatting novel views, diffusion identity/background edits, and capture artifacts. | The abstract claims improved robustness but gives no numbers. The abstract does not mention SMPL-X. | VERIFIED (abstract only) |
| **HandAugment** (arXiv 2001.00702) | MANO pose/shape/camera Gaussian perturbation for *depth hand-pose estimation* | Not sign language | UNVERIFIED (snippet) |
| **PoseAug**: Gong, Zhang, Feng, CVPR 2021. https://arxiv.org/abs/2105.02465 | Differentiable augmentation of "posture, body size, view point and position" for 3D pose estimation | Precedent for body-size and bone-length perturbation. +9.1% PCK over the previous best augmentation on 3DHP. Not SLR. | VERIFIED (abstract) |

**Gap.** I found **no** published work that perturbs MANO hand pose under phonological or linguistic constraints to augment SLR training data. SGNify's symmetry and invariance priors and SignAvatars' biomechanical limits are the closest ready-made constraint sets.

---

## 8. Temporal augmentations for SLR

| Source | Temporal op | Range | Effect |
|---|---|---|---|
| Laines et al. 2023 (VERIFIED) | Speed: resample to N frames | N between the 25th and 75th percentile of training lengths (48–74 frames) | Largest single effect seen anywhere: removing it costs **−15.65 pp** on WLASL100 |
| GISLR 1st place (code VERIFIED) | `resample` by rate U(0.5, 1.5), p = 0.8; `temporal_crop`; `temporal_mask` of 20–40% of frames (NaN), p = 0.5 | as stated | No ablation available |
| WLASL Pose-TGCN (VERIFIED) | Random 50-consecutive-frame window | 50 frames | No ablation |
| SAM-SLR (VERIFIED) | Random 100-frame window (`random_choose`) | | Bundled into −4.86 pp |
| SignBERT (VERIFIED) | Random 32-frame sampling at train, centre at test | | No ablation |
| OpenHands (VERIFIED) | PoseRandomShift, random or uniform temporal subsample | | No ablation |
| NLA-SLR (VERIFIED) | Temporal crop (64 frames, plus a 32-frame counterpart) | | No ablation |
| VAC, PHOENIX14 RGB (VERIFIED) | Random temporal scaling | not transcribed | Crop+flip 23.2 → +temporal scaling 22.1 dev WER |
| ASL Citizen (VERIFIED) | **Rejects** random temporal crop because it can reduce compound signs to one root sign; uses deterministic frame skipping instead | | Rationale only |
| AS-CAL (NTU, VERIFIED) | Temporal reverse with p = 0.5 | | Generic action recognition. **Not meaning-preserving for sign:** reversal changes movement direction and can turn one sign into another (my inference, not a published finding). |

---

## 9. Linguistic / phonological knowledge in training

| Work | Mechanism | Result | Status |
|---|---|---|---|
| Kezar, Thomason, Sehyr, "Improving Sign Recognition with Phonology", EACL 2023 (2023.eacl-main.200, pp. 2732–2737). https://arxiv.org/abs/2302.05759; https://aclanthology.org/2023.eacl-main.200 | Auxiliary heads predicting ASL-LEX 2.0 phonemes (16 types; **handshape + minor location** were the best subset). WLASL is aligned to ASL-LEX, and 48% of videos have phonological labels. | Table 2, WLASL test-all top-1: SL-GCN 29.4 → 38.1 (**+8.7**); Transformer 20.5 → 23.4 (+2.8). Gains also hold for signs with no phonological labels (+8.3 for SL-GCN). | VERIFIED |
| Kezar et al., "The Sem-Lex Benchmark", ASSETS 2023. https://arxiv.org/abs/2310.00196 | Phoneme features as an auxiliary target | Phonemes recognisable at 85% accuracy; +6% few-shot ISR, +2% overall ISR | VERIFIED (abstract) |
| Yin, Carter, Lu, Kocab, "Phonological Perception of Sign Language Models", CogSci 2026. https://arxiv.org/abs/2606.28667 | Minimal-pair probing | "pose-based models are sensitive to handshape contrasts"; pixel models capture location better | VERIFIED (abstract) |
| SGNify (above) | Battison sign classes used as fitting priors | Reconstruction only | VERIFIED |
| NLA-SLR (above) | Gloss text embeddings drive label smoothing and inter-modality mixup | +1.22 and +1.25 pp | VERIFIED |

No paper found uses phonology to *constrain augmentation* directly. The use so far is as auxiliary supervision or as fitting priors.

---

## 10. Skeleton action-recognition augmentation that SLR borrows (3D, NTU)

| Source | Augmentation and range | Evidence | Status |
|---|---|---|---|
| **AimCLR**: Guo et al. (arXiv 2112.03590; AAAI 2022 per common citation, venue UNVERIFIED). Appendix A | Shear: all 6 off-diagonal factors ~ U[−0.5, 0.5]. Crop: pad T/6 then crop. Spatial flip (L/R swap) p = 0.5. Temporal flip p = 0.5. Rotate: main axis U[0, π/6] (30°), other two axes U[0, π/180]. Axis mask p = 0.5. Gaussian noise N(0, 0.01). Gaussian blur σ ∈ [0.1, 2], kernel 15, p = 0.5. | Table 1, NTU-60 linear eval: normal augmentations (shear + crop) 75.0/79.8; *extreme* augmentations used directly 71.3/77.8 (worse). Extreme augmentations only help when routed through their distributional-divergence loss (77.4 → 78.9). | VERIFIED |
| **AS-CAL**: Rao, Xu, Hu, Cheng, Hu (arXiv 2008.00188; Information Sciences 2021, venue UNVERIFIED) | Rotation: main axis [0, π/6], others [0, π/180]. Shear factors ∈ [−1, 1]. Reverse p = 0.5. Gaussian noise N(0, 0.05). Gaussian blur. Joint mask 5–15 joints over 50–100 frames. Channel (axis) mask. | Fig. 9, single augmentation, NTU-60 C-Sub linear eval: none 38.1; rotation 45.7; shear 48.3; reverse 48.0; noise 41.5; blur 45.1; joint mask 44.4; **channel mask 3.9** (collapse). "double Shear" degrades, attributed to "drastic changes of body shape". | VERIFIED |
| **STARS**: Mehraban, Rajabi, Iaboni, Taati (arXiv 2407.10935, venue UNVERIFIED) | Spatial flip, rotation, axis mask, shear | Table 7, NTU-60 KNN: no augmentation 84.5/89.6; flip 85.0/90.6; rotation 84.8/90.1; shear 83.2/90.4; **axis mask 81.2/89.0**. The authors note shearing and axis masking can degrade performance, and saw no linear-eval gain. | VERIFIED |
| ST-GCN `random_move` (inherited in SAM-SLR `tools.py`) | angle ∈ {−10, −5, 0, 5, 10}°, scale ∈ {0.9, 1.0, 1.1}, translate ∈ {−0.2 … 0.2} | No ablation here | VERIFIED (code) |
| PoseAug (above) | Body size, posture, view, position | 3D pose estimation | VERIFIED (abstract) |

**Bone length / scale.**
- I found no SLR ablation of bone-length augmentation.
- Laines et al.'s whole-skeleton Scale 0.5–1.0 removal cost only −0.22 pp.
- OpenHands says scale augmentation is unnecessary after centre/scale normalisation.
- SPOTER's normalisation explicitly removes body proportions.

---

## Synthesis table: augmentation → evidence → typical range

| Augmentation | Evidence it helps / hurts (SLR unless noted) | Typical range in the literature | Meaning-preservation notes for KineticAugment |
|---|---|---|---|
| In-plane 2D rotation | Helps: SPOTER +2.27 pp on WLASL100 alone (VERIFIED). Generic action recognition: small gain in STARS. | ±13° uniform (SPOTER); N(0, 0.1 rad ≈ 5.7°) (OpenHands config); ±30° (GISLR); ±5° (HandCraft); π/3 claimed in OpenHands arXiv text (code disagrees) | Safe at small angles. It approximates camera roll. |
| 3D global yaw/pitch (view) rotation of the body model | Helps: Spoken2Sign SMPL-X global-orient ±20° → +1.46 WLASL, +1.29 MSASL (VERIFIED). NGT200 synthetic extra views help. M3GCLR (snippet, UNVERIFIED) says large angles hurt in self-supervised learning. | ±20° (Spoken2Sign). Action recognition: main axis up to 30°, others ≤1°. | Best-supported 3D augmentation for an SMPL-X pipeline. Keep it moderate, because location relative to the body is phonemic and should be preserved, which global rotation does. |
| Squeeze / anisotropic x-scale | SPOTER +1.88 pp | 0–15% per side | Changes signing-space proportions. Fine at small values. |
| Perspective / tilt | SPOTER +1.50 pp | 0–10% (code) or [0, 1) (paper text) | Use the code value. |
| Shear | No SLR ablation. Action recognition: helps in AS-CAL single-augmentation tests, slightly hurts in STARS KNN, "double shear" degrades. | std 0.1 (OpenHands WLASL), 0.2 default; ±0.15 (GISLR); ±0.5 (AimCLR), ±1 (AS-CAL) for 3D action recognition | Shear distorts body geometry, which an SMPL-X refit would make anatomically inconsistent. Prefer real 3D view rotation. |
| Per-joint arm rotation (kinematic chain) | SPOTER +2.27 pp (equal best) | p = 0.3 per joint, ±4° | Maps naturally onto small SMPL-X arm-joint rotations. Must not move the hand to a different location phoneme. |
| Global scale / body size / bone length | Small or none: Laines removing scale −0.22 pp. OpenHands calls it unnecessary after normalisation. | 0.5–1.0 (Laines); 0.8–1.2 (GISLR); 0.9–1.1 (ST-GCN); ±5% (HandCraft) | SMPL-X β (shape) variation is the principled version. No SLR evidence yet, so it is a gap. |
| Translation / shift | Bundled only (SAM-SLR) | ±10 px of 512 (SAM-SLR); ±0.1 normalised (GISLR) | Neutralised by root-centred normalisation. |
| Horizontal mirror with L/R swap | Small help: Laines −0.69 pp when removed. VAC +0.6 WER (RGB). Handedness normalisation neutral in fingerspelling (Shi). Never reported to hurt. | p = 0.5 | Standard and meaning-preserving for lexical signs (left-dominant signers mirror right-dominant ones). It *must* swap L/R joints and the MANO hands, and mirror SMPL-X joint rotations correctly. Unverified edge cases: directional and indexing signs. |
| Gaussian coordinate noise | SPOTER +0.39 pp on top of everything else. Action recognition: weakest single augmentation in AS-CAL. | std 1e-3 (SPOTER, normalised); 0.01–0.05 (action recognition) | Tiny values only. |
| Joint / frame / clip masking (dropout) | SignBERT(+) pretraining: big gains as a pretext (+13 pp MSASL100), with ratio 40% and clip span ≤8 frames best. GISLR uses spatial and temporal masks. **Axis/channel masking hurts** in action recognition (AS-CAL collapse to 3.9; STARS −3.3). | Mask ratio 40%; clip 2–8 frames; temporal mask 20–40% | Simulates detector failures. Do not zero a whole coordinate axis. |
| Speed change / temporal resampling | **Strongest single effect:** Laines −15.65 pp when removed (WLASL100) | Rate 0.5–1.5 (GISLR); 25th–75th percentile length (Laines) | Meaning-preserving within normal signing-rate variation. |
| Random temporal crop / window | Universal (50, 100, 32, 64 frames) | | ASL Citizen warns it can truncate compounds. Prefer resampling over aggressive cropping for isolated signs. |
| Temporal reversal | Helps only in action-recognition self-supervised learning (AS-CAL) | p = 0.5 | **Not meaning-preserving for sign** (my inference). Exclude. |
| Mixup (input or feature) | NLA-SLR: intra +1.15, inter (with text) +1.25, both +2.64 (WLASL) | λ ~ Beta(0.8, 0.8) | Label-mixing, not geometry. Orthogonal to KineticAugment. |
| Synthetic avatar or generated signs | NGT200: a synthetic signer helps almost as much as a real one. HandCraft synthetic pretraining helps. | | Supports the SMPL-X/MANO-driven augmentation direction. |
| Phonology-aware training | Kezar: +8.7 pp (SL-GCN, WLASL) from handshape + minor-location auxiliary heads | | Strong case for using ASL-LEX handshape and location to *constrain* augmentations (an open gap). |

### Unverified or excluded items (not used as evidence above)
- M3GCLR (arXiv 2603.09367) large-angle rotation result: search snippet only.
- The topology-masked Scientific Reports 2026 numbers: snippet only.
- KArSL Arabic transformer rotation/scale gains: snippet only.
- Fingerspelling PoseNet dominant-hand flipping: snippet only.
- HandAugment MANO noise: snippet only.
- Siformer reuse of SPOTER augmentations: snippet only.
