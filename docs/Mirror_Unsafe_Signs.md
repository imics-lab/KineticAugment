# Signs that cannot be mirrored

**Date:** 2026-11-02. **Evidence:**
[mirror-unsafe sign notes](<augmentation-evidence/research_notes/Mirror-unsafe signs/mirror_unsafe_signs.md>),
where each claim has its URL and a verification label. **Blacklist:**
[`configs/mirror_blacklist.yaml`](../configs/mirror_blacklist.yaml). **Clip → gloss map:**
`scripts/map_clip_glosses.py`, which writes `data/wlasl/clip_glosses.csv`.

Mirroring reflects the whole skeleton left to right and swaps the SMPL-X sides and the MANO
hands, so the dominant hand becomes the other hand (Tier 1, item 2 in
[Augmentation_Grounding.md](Augmentation_Grounding.md)). It is safe for almost every sign:
signs are specified for the dominant and non-dominant hand, not for left and right, and
left-handed signers mirror them (Battison 1978, via Shield & Meier 2018). This page lists the
exceptions and checks our clips against them.

## 1. Which signs cannot be mirrored

**Absolute direction (strong evidence).** A few signs mean a direction in the world, and
left-handed signers keep it: LEFT and WEST still move to the signer's left, RIGHT and EAST to the
right (Lifeprint, Handspeak). Shield & Meier 2018 name EAST, WEST, RIGHT and LEFT as the
exceptions to mirroring, in a footnote credited to a reviewer rather than to a study. A mirrored
LEFT is therefore a wrong or ill-formed sign.

- **Relabelling does not rescue them.** A mirrored LEFT is an L handshape moving right, which is
  not RIGHT: the pairs are signed with different letter handshapes (L/R, E/W; confirmed in
  ASL-LEX). So no WLASL gloss can be mirrored and given the opposite label.
- **NORTH and SOUTH are safe.** They move vertically, which mirroring does not change.
- **NORTHWEST** is excluded by inference from WEST. No source covers it.
- **Turn-left, turn-right and lane-change classifiers** keep their direction too, but they are not
  separate WLASL glosses. `turn` is a different sign and is safe.

**Caution (weak evidence).** HEART names an organ on the left side, and the only flip blacklist
published in sign recognition, for Russian Sign Language's Slovo dataset, excludes it
(Kvanchiani et al. 2024). ASL sources disagree on where HEART is signed (mid-chest in Handspeak,
left chest in ASL Bloom), and none says what left-handers do. HEART and HEART ATTACK are excluded
because excluding them is cheap. Watching the left-handed WLASL clips of `heart`, or asking a Deaf
left-handed signer, would settle it.

**Safe, although they look lateral.**

| Signs | Why mirroring keeps them |
|---|---|
| TIME, DOCTOR, NURSE, HOSPITAL | Made on the non-dominant wrist or arm, which swaps with the hands |
| BIRTHDAY, LAZY, CHARACTER, POLICE | Chest-side signs; Lifeprint says left-handers mirror them |
| ME, YOU, SHE, THEY and other pointing | Citation-form loci are not fixed sides |
| GIVE, SHOW, HELP, TELL, ASK, INFORM | Citation forms move forward and back, not sideways |
| ONE to TEN, J, fingerspelling | Lifeprint describes left-handed forms as mirrored |

## 2. The blacklist

| Category | WLASL glosses | WLASL clips | Evidence |
|---|---|---|---|
| Absolute side | `left`, `right`, `east`, `west`, `northwest` | 62 | strong (dictionaries agree, a peer-reviewed footnote) |
| Caution | `heart`, `heart attack` | 21 | weak (one cross-language precedent) |

That is 83 of WLASL's 21,083 clips. `right` is the only one inside WLASL100 (rank 95).

- **`right` mixes two senses.** Its source URLs show 2 clips of the direction sign and 4 of
  RIGHT meaning "correct"; the other 12 are unlabelled. The "correct" sense could be mirrored,
  but the whole gloss is blocked until someone checks the 12 clips by eye.
- **ASL-LEX cannot find these signs for us.** It has no left/right field, so the categories come
  from each gloss's meaning (rules in the notes, §4).

## 3. Our clips

`data/metsis_new_unseen_sign_videos/` holds 334 clips, and every one is in WLASL. They cover 39
glosses, all in WLASL100 (ranks 0–38). **None is on the blacklist, so every clip can be
mirrored.**

| Gloss | Clips | Gloss | Clips | Gloss | Clips |
|---|---|---|---|---|---|
| all | 6 | deaf | 9 | many | 9 |
| bed | 9 | dog | 8 | mother | 8 |
| before | 13 | drink | 11 | no | 9 |
| black | 8 | family | 9 | now | 8 |
| blue | 6 | fine | 7 | orange | 7 |
| book | 5 | finish | 8 | table | 4 |
| bowling | 11 | fish | 1 | thanksgiving | 12 |
| can | 7 | go | 13 | thin | 13 |
| candy | 9 | help | 10 | walk | 8 |
| chair | 5 | hot | 9 | what | 9 |
| clothes | 4 | like | 8 | who | 11 |
| computer | 13 | woman | 8 | year | 7 |
| cool | 12 | yes | 10 | cousin | 10 |

A few of these look lateral but are dominance-relative, so mirroring keeps them:

- COMPUTER is made on the non-dominant forearm.
- DEAF moves between the ear and the mouth on the dominant side.
- COUSIN is signed beside the head on the dominant side.
- BOWLING swings the dominant arm.
- GO and HELP are directional verbs, but their citation forms move forward.

To look up any clip, open `data/wlasl/clip_glosses.csv`. It gives each clip's gloss, WLASL rank,
signer, split and variation, and whether it may be mirrored. Rebuild it with:

```sh
python scripts/map_clip_glosses.py
```

It needs `data/wlasl/WLASL_v0.3.json`, which is not committed (see
[`data/wlasl/index.md`](../data/wlasl/index.md)).

## 4. Limits

- **The evidence is mostly dictionaries.** No peer-reviewed inventory of non-mirrored ASL signs was
  found. The strong list rests on Lifeprint, Handspeak and one footnote.
- **Mirroring can still cost accuracy without changing any sign.** Flipping cost 1.3 points on
  WLASL in one RGB study (Kvanchiani et al. 2024), and nobody has traced that loss to particular
  glosses. The blacklist prevents wrong labels; the ablation in [`TODO.md`](../TODO.md), with
  per-class deltas, is what shows whether mirroring helps.
- **Some WLASL signers may already be left-handed.** For the absolute-side glosses their clips
  still move in the correct direction, so a flip is wrong whichever hand the signer used.
