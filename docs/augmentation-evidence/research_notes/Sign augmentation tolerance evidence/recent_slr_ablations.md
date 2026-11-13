# Recent (2023–2026) SLR augmentation ablations, mirroring, view, temporal and synthetic-data evidence

Scope: new evidence beyond the sources already covered (SPOTER, OpenHands, ASL Citizen, SignBERT/+, Pose-TGCN, SAM-SLR, NLA-SLR, Laines 2023, GISLR 1st place, Spoken2Sign, Kezar 2023, Shi 2023, AS-CAL, STARS, ST-GCN, SGNify, SignAvatars, HandCraft). The aim is to test KineticAugment's Tier 1: yaw ±20° / pitch ±5° view rotation, mirror with L/R swap p=0.5, duration 0.8–1.25×, arm-joint rotation ±4° p=0.3, scene scale 0.85–1.15, and uncentring ±0.1.

Labels: **VERIFIED** means I read the primary text (PDF or HTML) myself. **VERIFIED-SECONDARY** means a fetched summary or search-engine extract of the primary source, without my reading the table directly. **EXISTS** means the paper or resource exists but I did not check its numbers. **UNVERIFIED** means a claim from a secondary blog or search summary only.

Overall caveat: I found **no** 2023–2026 paper that ablates augmentations per type on **3-D** skeleton input for isolated SLR at large vocabulary (WLASL2000, MSASL1000, ASL Citizen, Sem-Lex). The new per-augmentation ablations are all on small vocabularies (WLASL100, 20 classes), on 2-D keypoints, on RGB, or on translation rather than recognition.

## Q1. 2023–2026 pose-based SLR papers with per-augmentation ablations (ranges and deltas)

### Takeaway
On 2-D keypoints, newer ablations show small or statistically null gains from rotation, scale, noise and frame-drop on WLASL100 (Woods & Rana 2023; Roh et al. 2024). Shear and rotation helped in a semi-supervised 20-class setting (SSLR 2025). In a large pose-based translation study, in-plane rotation, perspective, and shoulder and wrist rotation *hurt*, while elbow rotation, shear and noise were neutral or helpful (Železný et al. 2025). Large-vocabulary skeleton SOTA papers (DSTA-SLR 2024, LA-Sign 2026) report no augmentation ablations.

### Cited Findings
- **Roh et al. 2024, "Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for ISLR" (LREC-COLING 2024 SignLang workshop), VERIFIED.**
  - Input is 2-D MediaPipe keypoints on WLASL-100.
  - Augmentation copied from SPOTER: in-plane rotation uniform in [−13°, +13°], plus Gaussian noise N(0, 10⁻³) per keypoint.
  - Table 5 top-1: none 82.17, Gaussian only 82.24 (+0.07), rotation only 82.63 (+0.46), both 83.26 (+1.09).
  - On AUTSL-100 the same augmentation gave 91.15 → 91.66 (+0.51).
  - The authors list speed and joint-rotation augmentation as future work.
  - Source: [ACL Anthology](https://aclanthology.org/2024.signlang-1.36.pdf)
- **Woods & Rana 2023, "Constraints on Optimising Encoder-Only Transformers for Modelling Sign Language with Human Pose Estimation Keypoint Data" (J. Imaging 9(11):238), VERIFIED-SECONDARY (fetched full-text HTML via summarizer).**
  - Setup: 2-D OpenPose with 54 keypoints, on WLASL-alt (a corrected 100-class WLASL; 1,842/418/403 train/val/test). Baseline test top-1 is 0.7926 ± 0.0075, and every sweep ran several times.
  - **Rotation, maximum angle:** 1° 0.7884, 2° 0.7912, 4° 0.7894, **8° 0.7969 (best, +0.43 pt)**, 16° 0.7925, 32° 0.7900.
    - The authors conclude there is "no measurable effect": the uncertainty is about ±0.56 pt. They contrast this with SPOTER's reported +2.27 pt.
  - **Scale:**
    - Both axes: 0.08 gave 0.7963 and 0.32 gave 0.7981.
    - x-axis only: 0.32 gave 0.8028.
    - The authors judge every scale setting to be within noise.
  - **Noise:**
    - Flat up to λ = 0.256 (0.7887–0.7968).
    - **λ = 0.512 hurts: 0.7588, −3.4 pt.**
  - **Frame manipulation, maximum frames:**
    - Frame drop (speed-up) from 1 to 32 frames: 0.7862–0.7937, flat.
    - Trim-start 32 gave 0.7748 (−1.8 pt), and offset-pad 32 gave 0.7720 (−2.1 pt). These large temporal crops or offsets hurt.
  - Flip, squeeze and perspective were **not** swept.
  - Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10672608)
- **Algafri, Luqman, Alyami, Laradji 2025, "SSLR: A Semi-Supervised Learning Method for ISLR" (arXiv 2504.16640), VERIFIED.**
  - Setup: pose keypoints on WLASL, transformer backbone.
  - Augmentations:
    - Gaussian noise.
    - In-plane rotation up to 13°, following SPOTER.
    - Arm-level rotation of ±4° for each arm relative to the other.
    - Shear or squeeze of up to 15% from the left and right sides.
  - Table 3 ablation (20 classes, 75% labeled), val/test:
    - No augmentation, no normalization: 29.8 / 46.2.
    - Normalization only: 61.9 / 58.5.
    - Plus Gaussian noise: 58.3 / 56.9, so **noise alone −1.6 test**.
    - Plus rotation: 60.7 / 58.5.
    - Plus shear: **63.1 / 63.1 (+4.6 test over normalization only)**.
  - The authors say rotation and shear "had a significant impact".
  - Caveat: the test set is tiny (20 classes) and no variance is reported.
  - Source: [arXiv](https://arxiv.org/abs/2504.16640)
- **Železný, Straka, Javorek, Valach, Hrúz, Gruber 2025, "Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights" (arXiv 2507.01532), VERIFIED-SECONDARY (fetched HTML).**
  - Setup: train on YouTube-ASL, evaluate zero-shot on How2Sign. The task is translation, not ISLR.
  - Ablation at medium strength, BLEU-1 / BLEU-4 (baseline 17.47 / 2.17):

    | Augmentation (medium strength) | BLEU-1 | BLEU-4 |
    |---|---|---|
    | In-plane rotate ±4.5°, p=0.75 | 15.30 (worst) | 1.61 |
    | Shear ±4.5° | 17.19 | 2.20 |
    | Perspective ±0.11 | 16.07 | 2.17 |
    | Shoulder rotation ±7.5°, p=0.56 | 16.39 | 2.17 |
    | **Elbow rotation ±7.5°** | **17.48** | **2.28** |
    | Wrist rotation ±7.5° | 16.05 | 2.20 |
    | Noise σ=1.5 | 17.45 | 2.41 |

  - Authors' explanations:
    - Rotation is "not very common in real-world data examples."
    - Shoulder and wrist rotation "can easily change the meaning of signs."
  - The final protocol keeps only shear, elbow rotation and noise. Even then, only "medium" beat the baseline (B-4 2.46), while light scored 1.71 and heavy 2.29.
  - Source: [arXiv](https://arxiv.org/abs/2507.01532)
- **Hu et al. 2024, "Dynamic Spatial-Temporal Aggregation for Skeleton-Aware SLR" (DSTA-SLR, arXiv 2403.12519), VERIFIED-SECONDARY.**
  - Evaluated on WLASL100/300/1000/2000, MSASL100–1000, SLR500 and NMFs-CSL.
  - Augmentation: random noise with "maximum value of 20" (no units given), 50% horizontal flip, and random 120-of-150-frame temporal crop.
  - **No augmentation ablation.**
  - Source: [arXiv](https://arxiv.org/abs/2403.12519)
- **Pu, Lim, Chong, Loy 2026, "LA-Sign: Looped Transformers with Geometry-aware Alignment for Skeleton-based SLR" (arXiv 2603.29057), VERIFIED-SECONDARY.**
  - Setup: 2-D RTMPose with 69 keypoints, on WLASL2000/300 and MSASL1000/200, with YouTube-ASL pre-training.
  - **It describes no data augmentation and gives no augmentation ablation.**
  - Latent noise or temporal injection during looping *reduced* accuracy (e.g., internal noise 64.73 → 63.35 P-I). This is a latent perturbation, not an input augmentation.
  - Source: [arXiv](https://arxiv.org/abs/2603.29057)
- **Kvanchiani et al. (SberDevices) 2024/2025, "Training Strategies for ISLR" (arXiv 2412.11553v2), VERIFIED-SECONDARY.** This is **RGB**, not pose, but it is the clearest recent per-augmentation temporal ablation.
  - Augmentations:
    - Boundary shift [−5, 5] frames.
    - Random drop: 10% of frames, p=0.5.
    - Random add: 30%, p=0.25.
    - **Speed-up ×2, p=0.25, and slow-down ×2, p=0.25**.
    - MixUp α=1 and CutMix α=0.8, for transformers only.
  - SlovoExt ablation with MViTv2-S (full model 87.31 top-1); removing each component gives:

    | Removed component | Change (pt) |
    |---|---|
    | All video (temporal) augmentations | −0.54 |
    | Slow-down | −0.66 |
    | Speed-up | −0.32 |
    | Random add | −0.35 |
    | Random drop | −0.30 |
    | Boundary shift | −0.31 |
    | CutMix + MixUp | −1.89 |

  - Source: [arXiv](https://arxiv.org/abs/2412.11553)
- **GISLR (Kaggle/PopSign) 2nd and 3rd place, UNVERIFIED** (Chinese-language secondary write-up seen only through a search summary).
  - 2nd place reportedly used:
    - random affine
    - random interpolation
    - pose flip
    - "finger-tree rotation"
  - 3rd place reportedly used:
    - global rotation, shift, scale and shear
    - flip "for some signs"
    - random frame dropping
  - No ablation numbers were seen.
  - Sources: [jingsailian](https://jingsailian.com/zlk/361776.html); [NVIDIA blog](https://blogs.nvidia.com/blog/2023/10/23/nvidia-life-kaggle-ai-contest-sign-language-recognition/)

### Inferences
- In 2-D keypoint ISLR on about 100 classes, single geometric augmentations are worth roughly 0–1 pt each and are often within run-to-run noise (Woods & Rana; Roh). The augmentation "stack" matters more than any single one. SSLR's +4.6 pt came from adding shear on top of rotation.
- **Arm-joint rotation:** ±4° per arm is the SPOTER and SSLR setting and is not reported to hurt. In Železný 2025, ±7.5° shoulder and wrist rotation *hurt* translation, while ±7.5° elbow rotation was neutral or slightly positive. Tier 1's ±4° at p=0.3 is below the magnitude seen to hurt, which supports keeping it small. If anything should be restricted, it is shoulder and wrist.
- **Scale 0.85–1.15** is well inside the range Woods & Rana found harmless (up to ±0.32 per axis).
- **Noise** hurts once it is large (Woods λ=0.512; SSLR noise-only −1.6), and Tier 1 does not include joint noise. That is consistent with the evidence.
- The strongest recent large-vocabulary skeleton SLR models either use flip, noise and crop without ablating them (DSTA-SLR) or use no input augmentation (LA-Sign, which relies on YouTube-ASL pre-training). So there is no large-vocabulary evidence that Tier 1 is either needed or harmful.

### Gaps
- No 2023–2026 per-augmentation ablation was found for **3-D skeletons** at WLASL2000, MSASL1000, ASL Citizen, Sem-Lex or PopSign scale.
- Not checked:
  - Uni-Sign (ICLR 2025)
  - MASA
  - SignCLIP
  - the ASL Citizen follow-ups
  - the NeurIPS 2023 PopSign baselines
  - arXiv 2511.05772 (skeletal graph+RNN SLR)
  - the GISLR 2nd- and 3rd-place Kaggle write-ups, whose primary posts were not read (Kaggle discussions 406306 and 406684)
- Mixup on skeletons for ISLR: I found no pose-specific ablation. The only number is Kvanchiani's RGB CutMix+MixUp −1.89 when removed.

## Q2. Does mirroring ever hurt sign recognition? Any per-class analysis?

### Takeaway
Yes, at least once, measured on RGB. Horizontal flip **lowered WLASL accuracy by 1.3 pt** and raised AUTSL by only 0.11 pt (Kvanchiani et al.). The same authors blacklist laterally-meaningful RSL signs ("left", "right", "heart", "liver") from flipping. NGT200 also warns that directional signs need caution. I found no published per-class analysis of flip harm.

### Cited Findings
- **Kvanchiani et al. 2024/25 (RGB), VERIFIED-SECONDARY.**
  - WLASL: "the metric with a flip is lower by 1.3%", so flip is disabled for WLASL.
  - AUTSL: flip "higher by 0.11%", so flip is enabled.
  - Slovo/SlovoExt: flip at p=0.5, with four class IDs [739, 635, 148, 636] blacklisted. These are signs such as "left", "right", "heart" and "liver" whose meaning depends on hand or side.
  - The authors note they could not determine which WLASL signs are non-mirrorable, so they tested flip empirically.
  - Absolute accuracies with and without flip are not given, and there is no per-class breakdown.
  - Source: [arXiv](https://arxiv.org/abs/2412.11553)
- **Ranum et al. 2024, NGT200 (GRaM workshop @ ICML 2024), VERIFIED.**
  - Vocabulary handedness: 122 one-handed signs, 63 symmetric two-handed (2s), and 15 asymmetric two-handed (2a).
  - "signers may not consistently use their dominant hand… suggesting flip symmetries as a potential data augmentation technique. However, caution is needed as some signs encode directionality". The paper did not test flip.
  - Source: [arXiv](https://arxiv.org/abs/2409.15284)
- **Pose-based (2-D) models that use flip p=0.5 on WLASL2000/MSASL** without reporting harm, and without ablation: DSTA-SLR 2024, VERIFIED-SECONDARY — [arXiv](https://arxiv.org/abs/2403.12519)
- **Laines 2023 (already covered):** removing flip costs about 1 pt on WLASL100, which runs the other way from the RGB WLASL result. Seen only in a search snippet here — [CVPRW](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/papers/Laines_Isolated_Sign_Language_Recognition_Based_on_Tree_Structure_Skeleton_Images_CVPRW_2023_paper.pdf)
- **GISLR 3rd place** reportedly applied flip only "for some signs". UNVERIFIED — [jingsailian](https://jingsailian.com/zlk/361776.html)

### Inferences
- On WLASL, the evidence on flip conflicts: −1.3 pt for RGB (Kvanchiani) versus about −1 pt when removed for 2-D pose (Laines). Flip's net effect is therefore dataset- and model-dependent and small, roughly ±1 pt.
- Mirror p=0.5 should not be called universally "safe". Tier 1 should at least carry a blacklist hook for laterally-meaningful or indexing signs, and ideally an ablation on the target dataset.
- Kvanchiani's RGB flip does not swap left/right hand semantics in the label space. KineticAugment's mirror *with* L/R swap is the skeleton analogue. Neither fixes signs whose meaning depends on absolute side (LEFT, RIGHT, indexing toward a referent placed on one side).

### Gaps
- I found no per-class or per-sign-category analysis of where flip hurts: directional verbs, indexing, LEFT/RIGHT, or body-anchored lateral locations such as the heart.
- I found no study on datasets with uniformly right-handed signers showing flip harms or helps. ASL Citizen and Sem-Lex handedness effects were not found.

## Q3. View-rotation magnitude: gains vs range

### Takeaway
No 2023–2026 study sweeps **3-D yaw** magnitude for SLR.

The only recent sweep is 2-D in-plane rotation on WLASL100, and it is flat from 1° to 32° (Woods & Rana 2023). In translation, ±4.5° in-plane rotation hurt (Železný 2025).

NGT200's real multi-view data shows that a ±25° view change is a severe domain shift for 2-D pose models. Single-view training loses more than 50% relative accuracy on a ±25° view, and adding views, including avatar-rendered views, recovers a lot of it. That indirectly supports yaw augmentation of about 20–25° when test views may vary, but it is not a range ablation.

### Cited Findings
- **Woods & Rana 2023, in-plane rotation sweep (2-D, WLASL-alt-100):** 0° 0.7926; 1° 0.7884; 2° 0.7912; 4° 0.7894; 8° 0.7969; 16° 0.7925; 32° 0.7900. There is no measurable effect across ranges and no harm even at 32°. VERIFIED-SECONDARY — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10672608)
- **Roh 2024:** ±13° rotation gives +0.46 pt alone on WLASL-100 (82.17 → 82.63). VERIFIED — [ACL Anthology](https://aclanthology.org/2024.signlang-1.36.pdf)
- **Železný 2025:** in-plane rotation of ±4.5° at p=0.75 gave the largest drop of any augmentation on YouTube-ASL → How2Sign translation (B-1 17.47 → 15.30). Perspective ±0.11 also hurt (16.07). VERIFIED-SECONDARY — [arXiv](https://arxiv.org/abs/2507.01532)
- **NGT200 (Ranum et al. 2024), VERIFIED.** Setup: cameras at −25°, 0° and +25°; 200 NGT signs; 3 human signers plus 1 synthetic; MediaPipe poses; SL-GCN.
  - Single-view training generalises badly to other views (Table 4, top-1). For example, front-only F¹²³ᴬ scores 0.09 on the front view but 0.03 on both left and right. The paper reports a relative drop of more than 50% on novel views.
  - Multi-view training (Table 5):
    - The front view rises to 0.35 with LF, 0.42 with FR and 0.49 with LFR.
    - The left view rises to 0.46 with LFR, and the right view to 0.47.
  - An SE(2)-equivariant temporal-PONITA adds +0.08 to +0.22 absolute over SL-GCN (e.g., LFR → R123: 0.55).
  - Source: [arXiv](https://arxiv.org/abs/2409.15284)
- **Action-recognition analogue (context, 2022, not SLR):** in an asymmetric-augmentation SSL paper on NTU-60, removing rotation lowered accuracy from 79.0 / 84.6 to 78.8 / 84.3 (xsub / xview). VERIFIED-SECONDARY (search extract) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9698072/)

### Inferences
- For frontal-camera datasets (WLASL, YouTube-ASL, How2Sign), view or rotation augmentation buys little and can hurt translation. For data with real view variation (NGT200), view coverage dominates.
- Tier 1's yaw ±20° is therefore defensible mainly as robustness insurance. The evidence does not say it raises accuracy on frontal test sets.
- Pitch ±5° is small enough that the in-plane evidence (flat to 32° in 2-D) suggests low risk. However, Železný's harm at ±4.5° in-plane rotation, which acts like camera roll, is a caution against adding roll.

### Gaps
- No ±10° / ±30° / ±45° 3-D yaw ablation was found for sign recognition. NTU action-recognition range sweeps with explicit degree ranges were not found in this pass.
- Spoken2Sign's ±20° is already covered, and I found no newer source sweeping it.

## Q4. Temporal: speed ranges beyond 0.8–1.25, extreme-range harm, frame-drop and interpolation

### Takeaway
×2 speed-up and ×2 slow-down (each p=0.25) helped modestly in RGB ISLR: removing them costs 0.32 and 0.66 pt on SlovoExt. A 0.5 duration scale hurt pose-based translation pre-training relative to no change. Frame-drop of up to 32 frames was neutral on 2-D WLASL100, but large start-trims or offset padding (32 frames) cost about 2 pt.

### Cited Findings
- **Kvanchiani et al. (RGB, SlovoExt, MViTv2-S), VERIFIED-SECONDARY.**
  - The augmentations were speed-up ×2, slow-down ×2, random drop 10%, random add 30%, and boundary shift [−5, 5].
  - Removal deltas: slow-down −0.66, speed-up −0.32, drop −0.30, add −0.35, shift −0.31, all video augmentations −0.54. The full model scores 87.31.
  - Source: [arXiv](https://arxiv.org/abs/2412.11553)
- **Woods & Rana 2023 (2-D, WLASL-alt-100), frame manipulation sweep, VERIFIED-SECONDARY.**
  - Drop (speed-up) of up to 1–32 frames: 0.7862–0.7937 against a 0.7926 baseline, so neutral.
  - Trim-start of 32 frames: 0.7748 (−1.8 pt).
  - Offset-pad of 32 frames: 0.7720 (−2.1 pt).
  - Offset-copy of 32 frames: 0.7954, neutral.
  - Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10672608)
- **Walsh, Ivashechkin, Bowden 2025, "Using SLP as Data Augmentation to enhance SLT" (IVA Adjunct '25 / SLTAT; arXiv 2506.09643), VERIFIED.** The setup is skeleton sign-stitched pre-training for PHOENIX14T pose-to-text, using duration resampling.
  - Test BLEU-1 / BLEU-4 by duration scale:

    | Duration scale | BLEU-1 | BLEU-4 |
    |---|---|---|
    | None (1.0) | 37.86 | 13.68 |
    | **0.5** | **35.75** | **12.27** |
    | 0.7 | 37.23 | 14.29 |
    | 1.1 | 36.24 | 13.27 |
    | 1.5 | 38.61 | 14.71 |
    | 0.7 + 1.0 + 1.5 mixed | 37.60 | 12.65 |

  - On the dev set, the shorter or slower settings won on BLEU-2 to BLEU-4. The authors suspect that "speed augmentation might introduce bias for some signs".
  - The paper's wording on direction is ambiguous: 1.5 is described as "increasing the speed by 1.5 times… by resampling the sequence to the desired number of frames."
  - Source: [Surrey PDF](https://personalpages.surrey.ac.uk/r.bowden/publications/2025/Walsh_SLTAT2025pp.pdf), [arXiv](https://arxiv.org/abs/2506.09643)
- **Walsh 2025, sign-order permutation:** permuting up to 3 glosses is neutral or slightly positive, while permuting more than 3 hurts. This applies to continuous signing only. VERIFIED — [Surrey PDF](https://personalpages.surrey.ac.uk/r.bowden/publications/2025/Walsh_SLTAT2025pp.pdf)

### Inferences
- Tier 1's 0.8–1.25× is conservative relative to evidence that ×2 in both directions still helps, though that evidence is RGB and continuous-signing ISLR.
- The one measured harm at a single fixed extreme is a 0.5 scale (Walsh). Since there the duration factor is applied to every pre-training sequence, it is weak evidence that a 2× change across the whole corpus degrades results.
- Large temporal crops that remove sign onset (trim-start or offset-pad of 32 frames) are what hurt in pose ISLR. Uniform resampling does not.

### Gaps
- No pose-based ISLR paper was found that sweeps a uniform speed factor (for example 0.5 / 0.8 / 1.25 / 1.5 / 2.0) on WLASL2000 or ASL Citizen.
- No temporal-interpolation ablation (as opposed to resampling) was found.

## Q5. Synthetic and avatar-generated sign data: measured gains on real test sets

### Takeaway
The only recent *isolated* recognition result found is NGT200. Adding a Ready-Player-Me avatar driven by 3D-LEX motion-capture animations, rendered at three views and passed through MediaPipe, raised unseen-signer accuracy substantially on a tiny dataset; for example, on the front view, 0.14 → 0.43 with all avatar views. Avatar data was roughly as useful as one extra human front-view signer.

For translation, synthetic skeleton "sign stitching" helped only as **pre-training** (B-4 11.32 → 13.68 on PHOENIX14T). Naively **mixing** synthetic with real data **hurt** slightly. Photo-realistic avatar (SignGAN or SignSplat) data mixed with real data gave small gains in RGB SLT.

I found no SMPL-X- or diffusion-generated augmentation paper reporting ISLR gains on large real test sets.

### Cited Findings
- **NGT200 (Ranum et al. 2024), VERIFIED.**
  - Synthetic data pipeline (App. C): 3D-LEX animation files for the 200 overlapping signs drive a Ready Player Me avatar in BabylonJS, which is rendered at −25°, 0° and +25°. MediaPipe Holistic then extracts 2-D poses.
  - The dataset is tiny: 600 poses per signer, i.e., one example per sign per view.
  - Table 6: SL-GCN trained on signers 1 and 2 (all views), tested on unseen signer 3, top-1 averaged over 10 runs:

    | Training data | Left | Front | Right |
    |---|---|---|---|
    | LFR¹² (baseline) | 0.03 | 0.14 | 0.14 |
    | \+ avatar front view (FA) | 0.22* | 0.27 | 0.09* |
    | \+ human SignBank front view (FSb) | 0.10 | 0.28 | 0.26 |
    | \+ all 3 avatar views (LFR¹²ᴬ) | **0.19** | **0.43** | **0.38** |
    | \+ all 3 avatar views + FSb | 0.32 | 0.48 | 0.43 |

    \*In the extracted text the left and right labels of the "+ avatar front view" row appear in the order R, F, L, so its left and right values may be swapped.
  - The authors: the human-vs-synthetic difference is "marginal", and human data gives "a slightly higher improvement".
  - They also note they did not evaluate whether synthetic data "might hinder" performance or introduce "false motion patterns".
  - Source: [arXiv](https://arxiv.org/abs/2409.15284)
- **Walsh et al. 2025 (SLT, PHOENIX14T), VERIFIED.**
  - Pose-to-text test results:

    | Training data | BLEU-1 | BLEU-4 |
    |---|---|---|
    | Real data (GT) | 32.41 | 11.32 |
    | Stitched synthetic only | 16.74 | 3.49 |
    | GT + stitched, mixed | 31.84 | 10.73 (slightly worse) |
    | **Stitched pre-train, then GT fine-tune** | **37.86** | **13.68 (≥15% relative on all metrics)** |

  - Video-to-text with GFSLT (test BLEU-4): PHIX 19.53; PHIX+GAN 20.77; PHIX+Splat 21.79.
    - VLP pre-training alone reaches 21.35, so the avatar gain over a VLP baseline is small.
    - GAN pre-train then PHIX gives 20.28, and Splat pre-train then PHIX gives 19.47.
  - Headline claim: "up to 19%" improvement.
  - Source: [Surrey PDF](https://personalpages.surrey.ac.uk/r.bowden/publications/2025/Walsh_SLTAT2025pp.pdf)
- **Related generators that report no ISLR gains from use as training data (EXISTS, numbers not checked):**
  - Signs as Tokens (SOKE): [arXiv 2411.17799](https://arxiv.org/abs/2411.17799)
  - SignAvatar / ASL3DWord (FG 2024): [arXiv 2405.07974](https://arxiv.org/abs/2405.07974)
  - DiffSign, which retargets onto SMPL-X avatars for video generation: [arXiv 2412.03878](https://arxiv.org/abs/2412.03878)
  - DexAvatar: [arXiv 2512.21054](https://arxiv.org/abs/2512.21054)

### Inferences
- Avatar or retargeted data pays off most as **signer and view diversity** when real signers are few: NGT200 had two training signers. With large real corpora, the expected marginal gain is unproven.
- Walsh's mixed-training harm against pre-training gain suggests that synthetic or heavily-augmented skeletons may work better as a pre-training or curriculum stage than as 50/50 mixing. This is relevant if KineticAugment adds synthetic-signer variants.

### Gaps
- No paper found trains ISLR on SMPL-X or MANO-retargeted or diffusion-generated isolated signs and reports gains on WLASL2000, ASL Citizen, Sem-Lex or BOBSL.
- The specific named items "SynthSL" and "synthetic BSL from BOBSL" were not found in this pass.

## Q6. Pose-based SLR robustness and benchmark studies (cross-signer, cross-view)

### Takeaway
NGT200 is the clearest recent cross-view and cross-signer pose benchmark.

- A ±25° camera change roughly halves or worse accuracy for 2-D pose models.
- An unseen signer costs a further 6–27% relative accuracy at equal training size (left −27%, front −6%, right −9%).
- SE(2)-equivariant models improve robustness.

### Cited Findings
- **NGT200, VERIFIED.**
  - A single-view-trained SL-GCN loses more than 50% relative accuracy on novel views (Table 4).
  - Extra front-view data (SignBank) raised front-view accuracy by 122% relative but "only marginally" improved side views.
  - At equal training size, an unseen signer costs 27% (left), 6% (front) and 9% (right) relative accuracy.
  - Temporal-PONITA, which is SE(2)-equivariant, beats SL-GCN by +0.08 to +0.22 absolute.
  - Source: [arXiv](https://arxiv.org/abs/2409.15284)
- **Roh 2024:** MediaPipe fails to detect almost 50% of hands in WLASL. Their keypoint reconstruction and anchor preprocessing alone gives +6.05 pt, far more than augmentation (+1.09 pt). VERIFIED — [ACL Anthology](https://aclanthology.org/2024.signlang-1.36.pdf)
- **SSLR 2025:** signing-space normalization alone gives +12.3 pt test (46.2 → 58.5), against about +4.6 pt from augmentation. VERIFIED — [arXiv](https://arxiv.org/abs/2504.16640)

### Inferences
- Normalisation and pose cleaning (missing-hand reconstruction) have repeatedly outweighed geometric augmentation by 3–10×. For example, Tier 1's uncentring ±0.1 matters only if the downstream model is not already root-normalised.
- For cross-view robustness, real or avatar multi-view data and equivariant architectures have measured benefits. The benefit of yaw augmentation alone is unmeasured.

### Gaps
- No large-scale cross-signer robustness benchmark for pose ISLR was found beyond NGT200 (2023–2026), e.g., on ASL Citizen's signer-independent split with augmentation ablations.
- Not checked: robustness studies under keypoint-estimator change (MediaPipe vs RTMPose vs SMPL-X fits).
