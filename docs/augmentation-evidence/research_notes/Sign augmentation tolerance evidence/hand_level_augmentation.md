# Hand-level (finger / handshape / wrist) pose augmentation in SLR and hand-gesture recognition

Scope note: this pass used about 20 search/fetch calls. Labels: VERIFIED = primary text or code read in this pass; VERIFIED-SECONDARY = read through a summary of the primary page (a search-engine digest or an LLM page digest that did not show the full table); EXISTS = the source exists, but its content was not checked; UNVERIFIED = recalled or inferred and not confirmed. Items in the brief's "already covered" list are only mentioned where this pass adds a hand-level detail.

## Q1. Which SLR, fingerspelling and hand-gesture papers (2019–2026) augment hand keypoints or joint angles, and with what ranges and measured gains?

### Takeaway
Explicit hand-local augmentation is rare in the SLR literature, and ablations that isolate it are rarer still. Most pipelines apply a global affine to the whole skeleton. The few hand-specific transforms we found have no published per-transform ablation: finger-tree rotation, per-part shift-scale-rotate, finger dropout and hand dropout. The only paper with finger-level anatomical keypoint augmentation (BISINDO, 2026) reports a large cross-signer gain, but it does not give ranges in the accessible text and it does not ablate each transform.

### Cited Findings
- **BISINDO alphabet, "Anatomy-Aware Realistic 3D Keypoint Augmentation" (Alfarisi, Cahyani, Setumin; Scientific Journal of Informatics 13(2):267–280, 2026).** The paper uses four 3-D hand-keypoint augmentations on MediaPipe Hands keypoints with an MLP classifier: Gaussian Jitter, Anisotropic Scale, Bone Length Scale, and Depth & Tilt Jitter. Cross-domain F1 (unseen subjects and different capture devices) rose from **67.20% to 89.55%**. The authors say in-domain accuracy did not degrade and that variance across folds fell. The abstract page gives no per-transform ablation and no numeric ranges (degrees, scale factors, noise sigma). The training data is a single-subject dataset. — VERIFIED (abstract page only) — [SJI article](https://journal.unnes.ac.id/journals/sji/article/view/42221)
  - The same paper argues that augmentations built for full-body motion do not transfer to hand-sign recognition, because small changes in finger configuration can change meaning. — VERIFIED-SECONDARY (search digest) — [SJI article](https://journal.unnes.ac.id/journals/sji/article/view/42221)
- **AugmentGest (arXiv 2506.07216, 2025).** The framework applies random cropping, rotation and zoom (plus brightness and contrast) to hand-skeleton or point-cloud gesture data and makes three augmented copies of each sample. It was evaluated on DHG-14/28, SHREC'17 (14g and 28g) and JHMDB, with e2eET, FPPR-PCD and DD-Net models. Reported gains: e2eET on DHG-28g went from **91.67% to 92.98% (+1.31)**, and FPPR-PCD on SHREC'17-14g from **95.90% to 96.40%**. In the e2eET ablation on SHREC'17, cropping alone and rotation alone each reached **97.50% (+0.83)**, while zoom had no effect. The rotation is a whole-hand (global) transform, not per-finger. — VERIFIED-SECONDARY (search digest; check the exact values against Table 3) — [arXiv 2506.07216](https://arxiv.org/html/2506.07216v1); code at github.com/NadaAbodeshish/Random-Cropping-augmentation-HGR
- **Imaginative GAN (Shen, Dudley, Kristensson; FG 2021).** A learned (GAN) augmentation for skeleton-based hand gesture data, evaluated on SHREC'17 and MSR Action3D. It is not a per-finger hand-crafted perturbation. — EXISTS — [FG 2021 PDF](https://www.pokristensson.com/pubs/ShenEtAlFG2021.pdf)
- **Training Strategies for ISLR (arXiv 2412.11553, 2024; RGB MViTv2).** The paper deliberately avoids hand-geometry augmentations. It uses no hand rotation, finger rotation, hand scale or hand dropout. Its flip ablation is still relevant to handedness: horizontal flip **lowers WLASL accuracy by 1.3%** and **raises AUTSL accuracy by 0.11%**. For Slovo, flip is applied only to mirror-invariant signs. On SlovoExt, removing all image augmentations costs −4.06 top-1, and removing CutMix/MixUp costs −1.89. — VERIFIED — [arXiv 2412.11553](https://arxiv.org/html/2412.11553)
- **SPOTER augmentation code (hand-level detail).** The augmentations include a whole-frame rotation centred at (0.5, 0.5) and a separate rotation of the arm joints. The search digest shows no finger-level rotation. — VERIFIED-SECONDARY — [SPOTER augmentations on Hugging Face](https://huggingface.co/spaces/CVPR/SPOTER_Sign_Language_Recognition/blob/main/spoter_mod/augmentations/__init__.py)
- **SignBART (arXiv 2506.21592, 2025).** This is a modality ablation, not an augmentation: right-hand keypoints alone reach **70.20%** and the left hand alone **23.02%**. The authors attribute the gap to right-hand dominance. — VERIFIED-SECONDARY — [arXiv 2506.21592](https://arxiv.org/pdf/2506.21592)
- **SKIM: Skeleton-Based ISLR with Part Mixing.** It mixes body parts (e.g., hands) between samples as an augmentation. Its exact gains were not retrieved; one attempted fetch hit the wrong arXiv ID. — EXISTS — [SKIM listing](https://bohrium.dp.tech/paper/arxiv/5afa2dae5e767d74dd60e95fb74149fa241e1b9e4589eeb4209d03b0cf2a9b6e)
- **"Skeleton-Based Data Augmentation for SLR Using Adversarial Learning" (IEEE Access, Jan 2025).** — EXISTS — [DOAJ record](https://doaj.org/article/a427a24051744b9daa703a93d63c292c)
- **"Local-Global Data Augmentation" (Basso Madjoukeng et al., UNamur, 2025; Springer chapter, DOI 10.1007/978-3-031-91398-3_5).** A hand-focused local/global crop augmentation for contrastive learning. It works on *static RGB images*, not skeletons. The PDF could not be text-extracted in this pass. — EXISTS — [UNamur portal](https://researchportal.unamur.be/en/publications/local-global-data-augmentation-for-contrastive-learning-in-static/)
- **A 2026 arXiv paper (2603.06661)** argues that common time-series augmentations ignore skeletal geometry and kinematic dependencies. — EXISTS (search snippet only) — [arXiv 2603.06661](https://arxiv.org/html/2603.06661)
- **A 2026 arXiv paper (2603.09213)** uses 20 inter-joint angles from MediaPipe hand keypoints. These are invariant to SO(3) rotation, translation and isotropic scale, which removes the need for hand-rotation augmentation. — VERIFIED-SECONDARY — [arXiv 2603.09213](https://arxiv.org/html/2603.09213)

### Inferences
- We found no paper that perturbs individual finger joint angles within an articulatory class and ablates that perturbation on an SLR benchmark. This supports, and slightly extends, the existing review's claim that no published work perturbs MANO hand pose under phonological constraints. The negative finding is for keypoint-space work generally, not only MANO.
- Measured hand-gesture gains from global rotation or cropping are small, about +0.5 to +1.3 points on DHG/SHREC. The one large gain (BISINDO, +22 F1) comes from a single-subject, cross-device setting where any anatomical variation helps. It should not be read as a general effect size.
- The flip result (−1.3% on WLASL) is the clearest published evidence that a hand-level transform which changes handedness or chirality semantics can hurt. A within-category jitter should avoid transforms that move a sign across a phonological boundary in the same way.

### Gaps
- No numeric ranges were obtainable for the BISINDO augmentations without the full PDF. The DOI listed on the page (10.15294/sji.v11i2.42221) does not match the stated volume 13(2), so check it.
- The AugmentGest rotation range (degrees) and its exact Table 3 values were not read from the primary text.
- SKIM's part-mixing gains on WLASL/AUTSL were not retrieved.
- No DHG/SHREC/FPHA/IPN study was found that ablates *per-finger* rotation or finger-joint noise. IPN and HaGRID are mostly RGB benchmarks, and no "HaGRID-skeleton" augmentation study was found.

## Q2. Which hand-specific augmentations did the top Kaggle GISLR and ASL Fingerspelling (2023) solutions use?

### Takeaway
The GISLR 2nd-place solution (ffs333) used a **finger-tree rotation** and a per-part (hand, lips, pose) shift-scale-rotate. The rotation angle is unspecified, augmentation probabilities were tuned with Optuna over roughly 0.1–0.5, and no per-augmentation ablation was published. The Fingerspelling 1st-place solution (Henkel) applies a strong **global** affine to all landmarks: scale 0.8–1.2, shear ±0.15, shift ±0.1, rotation ±30°, p=0.75. Its library also includes FingersDrop and HandDrop transforms, but they do not appear in its cfg_1.

### Cited Findings
- **GISLR 2nd place (ffs333).**
  - **Finger tree rotate.** Each 5-vertex finger chain is split into 4 root→children groups (thumb: 0→[1,2,3,4], 1→[2,3,4], 2→[3,4], 3→[4]). A random subset of these groups has its children rotated about the root by a small random angle. The README does not give the angle range. — VERIFIED (README) — [ffs333/2nd_place_GISLR](https://github.com/ffs333/2nd_place_GISLR)
  - **Per-part affine.** In the CNN branch, a separate shift-scale-rotate is applied to the hand, lip and body-pose parts after the global random affine, which follows the version shared by user hengck23. No numeric ranges are given. — VERIFIED (README) — [ffs333/2nd_place_GISLR](https://github.com/ffs333/2nd_place_GISLR)
  - Other details: x-flip; time interpolation; mixup (CNN only; reported not to work for transformers); "replace" augmentation, which swaps random parts with same-class samples; and torchaudio FrequencyMasking(80) / TimeMasking(18) (CNN). The author called the last "exceptionally" effective. — VERIFIED (README) — [ffs333/2nd_place_GISLR](https://github.com/ffs333/2nd_place_GISLR)
  - **Probabilities and gains.** All augmentation probabilities were tuned with Optuna in about 0.1–0.5+. The README gives no CV/LB gain for any individual augmentation. Single EfficientNet-B0 fold: CV 0.898, LB about 0.80. Adding transformers raised the LB to 0.81. — VERIFIED (README) — [ffs333/2nd_place_GISLR](https://github.com/ffs333/2nd_place_GISLR)
  - Preprocessing (not augmentation): the solution keeps the hand with fewer NaNs and mirrors a right hand to the left. Its features include all 210 pairwise distances among the 21 hand points. — VERIFIED (README) — [ffs333/2nd_place_GISLR](https://github.com/ffs333/2nd_place_GISLR)
- **ASL Fingerspelling 1st place (Christof Henkel et al.).**
  - `configs/cfg_1.py` train_aug: `Resample(sample_rate=(0.5,1.5), p=0.8)`; `SpatialAffine(scale=(0.8,1.2), shear=(-0.15,0.15), shift=(-0.1,0.1), degree=(-30,30), p=0.75)`; `TemporalMask(size=(0.2,0.4), p=0.5)`; `SpatialMask(size=(0.05,0.1), mode='relative', p=0.5)`. The config uses 130 landmarks. — VERIFIED (code) — [cfg_1.py](https://raw.githubusercontent.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/main/configs/cfg_1.py)
  - The augmentation library (`configs/augmentations.py`) defines these landmark-level transforms:
    - **FingersDrop**: drops ≥1 finger landmark group for the whole sequence.
    - **HandDrop2**: drops both hands.
    - PoseDrop / PoseDrop2 / FaceDrop.
    - **OnLandmarkIds**: applies any transform only to chosen landmark indices, i.e., a hand-only affine is possible.
    - **SpatialAffineCone**: interpolates between two random affines across 6–7 time windows.
    - SpatialNoise: uniform ±0.05.

    There is no finger-tree rotation transform. — VERIFIED (code, through an LLM page digest) — [augmentations.py](https://raw.githubusercontent.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/main/configs/augmentations.py)
  - Henkel called data augmentation "essential" for generalising across signers. No ablation figure is given. — VERIFIED — [NVIDIA blog](https://blogs.nvidia.com/blog/2023/10/23/nvidia-life-kaggle-ai-contest-sign-language-recognition/)
- **Another ASLFR solution (qklent/aslfr; placement not stated).** It uses horizontal flip, random affine and random masking, with no hand-specific transforms listed. — VERIFIED-SECONDARY — [qklent/aslfr](https://github.com/qklent/aslfr)
- **GISLR 6th place (TheoViel/kaggle_islr).** The repo exists; its augmentations were not read. — EXISTS — [relatedrepos listing](https://relatedrepos.com/gh/TheoViel/kaggle_islr)

### Inferences
- The finger-tree rotation in GISLR 2nd place is the closest public precedent for our within-category handshape jitter. It rotates distal sub-chains about proximal joints, which perturbs flexion along each finger. It is applied in 2-D/3-D keypoint space, not in joint-angle space, and it does not enforce class constraints. The angle is described only as "small", so we cannot cite a tolerance range from it.
- Henkel's ±30° rotation is a whole-skeleton global rotation applied to normalised landmarks. It is a viewpoint and global-pose tolerance, not a handshape tolerance, and should not be cited as evidence that fingers tolerate ±30° articulation.
- Neither top solution published leave-one-out gains for its hand-specific transforms. The Kaggle evidence shows only that these transforms were used, not that they helped.

### Gaps
- The finger-tree rotation angle range has to be read from the ffs333 source code, which was not opened in this pass.
- Whether a later Henkel config (other than cfg_1) enabled FingersDrop or HandDrop2, and with what probability, was not checked.
- Write-ups for GISLR 3rd–10th place and ASLFR 2nd–5th place were not found by web search. The Kaggle discussion pages need direct access, which failed or was not attempted.

## Q3. Is MANO-space augmentation or synthetic hand data used to train recognisers, and does it transfer?

### Takeaway
We found no SLR paper that samples or interpolates MANO pose (e.g., PCA space) to augment a recogniser and reports the transfer gain. The closest measured evidence is 2-D generative handshape augmentation. Rios et al. (2025) got **+4.7 points** on RWTH-PHOENIX handshapes (80.62% to 85.34%) with GAN pretraining, and very large gains in few-shot regimes. Synthetic-pretraining results also exist for fingerspelling and for the LSFB dataset.

### Cited Findings
- **Rios et al., "Bringing Balance to Hand Shape Classification" (arXiv 2507.17008, 2025; listed as Applied Soft Computing).**
  - Data: RWTH-PHOENIX-Weather handshapes (2,098 images, 39 classes, heavily imbalanced) and HaGRID (about 553K images, 18 classes plus "no gesture").
  - Classifier and generators: an EfficientNetV2 classifier, with handshape-conditioned ReACGAN and pose-conditioned SPADE generators. The pose input is 2-D OpenPose keypoints drawn as bones; MANO is not used.
  - RWTH full-data test accuracy:
    - Baseline: **80.62%**
    - ReACGAN pretrain, then real fine-tune: **85.34%**
    - SPADE pose-conditioned: 80.91%
    - SPADE multi-source (generator trained on HaGRID): **85.15%**
    - Mixup and regularisation variants: about 75–80%
  - Few-shot (5/10/20 per class): baseline 1.89 / 4.11 / 53.16%; ReACGAN pretrain **37.74 / 54.16 / 75.58%**.
  - On HaGRID, generated-data pretraining did not help: real-only training was best at 91.08%.

  — VERIFIED — [arXiv 2507.17008](https://arxiv.org/html/2507.17008v1)
- **UCD PhD thesis (2024).** It describes a fingerspelling recogniser trained entirely on computer-generated data and says data augmentation mitigates the synthetic-to-real gap. Quantitative results were not read. — EXISTS — [UCD repository](https://researchrepository.ucd.ie/entities/publication/d174e3e5-ebe7-40f2-a8ce-37e88f4c55f2)
- **Synthetic-data pretraining for SLR on LSFB and DiSPLaY.** A related group reports that it sometimes beats traditional augmentation. — VERIFIED-SECONDARY (search digest) — source cluster [catalyzex author pages](https://www.catalyzex.com/author/Facundo%20Quiroga)
- **DART (MANO extension with accessories and textures).** A Unity tool that renders synthetic hand images with MANO pose labels. It targets hand pose estimation, not sign recognition. — EXISTS (search digest) — [arXiv 2210.07650](https://arxiv.org/pdf/2210.07650v1.pdf)
- **DeepHPS / SynHand5M.** Uses a non-MANO procedural hand model. The authors note that MANO was built from 31 subjects and deliberately generate more shape variation than real cohorts show. — VERIFIED-SECONDARY — [arXiv 1808.09208](https://arxiv.org/pdf/1808.09208)

### Inferences
- Synthetic or generative handshape data helps most in low-data, imbalanced handshape classification. On a large, balanced gesture set (HaGRID) it gave no gain. For our trial, within-category handshape jitter is most likely to pay off on rare handshape and sign classes, and it may be neutral on head classes.
- Rios et al.'s pose-conditioned generator did not help by itself (80.91% against 80.62%). It helped only when the generator was trained on a much larger external hand corpus. This suggests that pose diversity from outside the dataset matters, not just re-rendering the existing poses.

### Gaps
- No paper was found that samples MANO PCA or interpolates MANO hand poses to augment an SLR or fingerspelling recogniser and measures the gain. Neural Sign Actors and other MANO/SMPL-X avatar papers were not re-checked for recogniser-training results in this pass.
- The UCD thesis and the LSFB synthetic-pretraining numbers were not retrieved.

## Q4. Which handshape or phonology classifiers from keypoints could serve as an invariance check?

### Takeaway
Kezar et al.'s Sem-Lex models are the most directly usable. They pair an SL-GCN on pose keypoints with one head per ASL-LEX phonological feature, and their per-feature test accuracies include the exact features we want to preserve: selected fingers about 91%, flexion about 81%, spread about 88%, thumb position about 91.5%, and the 58-class handshape feature 75–77%. Code for the phonology models is public. These accuracies set a noise floor: an invariance check cannot detect class flips below the classifier's own error rate, which is about 9–25% depending on the feature.

### Cited Findings
- **Sem-Lex Benchmark (Kezar et al., ASSETS 2023, arXiv 2310.00196).**
  - Data: more than 84k isolated-sign videos from deaf signers, aligned to ASL-LEX, SignBank and ASL Citizen.
  - Model: SL-GCN on pose keypoints, with one FC head per phonological feature.
  - Test accuracy per feature (Fine-tune / Multitask, per Table 5; class counts from Table 1):

    | Feature | Classes | Fine-tune | Multitask |
    |---|---|---|---|
    | Selected Fingers | 8 | 0.911 | 0.902 |
    | Flexion | 8 | 0.812 | 0.810 |
    | Spread | 3 | 0.884 | 0.880 |
    | Spread Change | 3 | 0.903 | 0.895 |
    | Thumb Position | 2 | 0.915 | 0.915 |
    | Thumb Contact | 3 | 0.917 | 0.911 |
    | Handshape | 58 | 0.774 | 0.747 |
    | Nondominant Handshape | 56 | 0.835 | 0.817 |
    | Wrist Twist | 2 | 0.924 | 0.926 |
    | Average | | 0.858 | 0.850 |

  - Phonological features as auxiliary targets improve ISR by **+6% (few-shot)** and **+2% (overall)**.
  - The paper describes no augmentation, and model release is not stated; the data is at github.com/leekezar/SemLex.

  — VERIFIED — [ar5iv 2310.00196](https://ar5iv.labs.arxiv.org/html/2310.00196)
- **Kezar et al., "Exploring Strategies for Modeling Sign Language Phonology" (arXiv 2310.00195, ESANN 2023).**
  - Setup: SL-GCN on the 16 ASL-LEX 2.0 phoneme types, gloss-pretrained, built on OpenHands; 65,935 videos.
  - Curriculum learning averages **86.8%**, against 85.8% / 85.0% for the other two strategies.
  - Curriculum per-feature accuracy: Selected Fingers 90.6, Flexion 83.2, Spread 88.8, Thumb Position 91.8, Handshape 76.9.
  - Code: github.com/leekezar/Modeling-ASL-Phonology.

  — VERIFIED (ar5iv; the page digest warned that the extracted table's column headers were misaligned, and the multitask/fine-tune column order conflicts with the Sem-Lex Table 5 labels above) — [ar5iv 2310.00195](https://ar5iv.labs.arxiv.org/html/2310.00195); conflict with [ar5iv 2310.00196](https://ar5iv.labs.arxiv.org/html/2310.00196)
- **WLASL-LEX (Tavella et al., 2022).** It augments WLASL with phonological feature estimates and was used by Sem-Lex for comparison. Its classifiers were not re-read here. — VERIFIED-SECONDARY (cited inside Sem-Lex) — [ar5iv 2310.00196](https://ar5iv.labs.arxiv.org/html/2310.00196)

### Inferences
- A practical invariance check is to train or download the Kezar SL-GCN phonology heads, feed original and jittered clips, and require that the predicted selected-fingers, flexion, spread and thumb classes do not change more often than they disagree across the original clip's own frames or near-duplicates. The headline tolerance would come from the flip-rate gap, not the raw flip rate, because flexion (8 classes, about 81%) is already noisy.
- Flexion is the weakest of our four preserved features. Jitter along the flexion axis needs the tightest bounds, or a check against an independent geometric flexion measure (e.g., MANO joint angles binned to ASL-LEX flexion classes), not the classifier alone.
- These models take MediaPipe/OpenPose-style 2-D/3-D keypoints, not MANO parameters. Our SMPL-X/MANO clips would have to be projected to the matching keypoint layout. That projection adds a domain gap to the invariance check.

### Gaps
- Pretrained Sem-Lex phonology checkpoints were not confirmed to be downloadable. Only the code repositories are confirmed.
- The exact multitask/fine-tune column assignment in 2310.00195 needs checking against the PDF.
- PopSign and ASL Citizen handshape-specific classifiers were not checked in this pass.
