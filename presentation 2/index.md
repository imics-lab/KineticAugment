# Presentation 2 index

Week 2 progress presentation for the professor (2026-10-13), with the sample clips it shows.

- `Week_2_Progress_Deck.pptx` — 21-slide deck: round trip on unseen clips, error make-up and smoothing, the augmentation evidence base and first trial, and V1.3 face results from `feat/face-v1.3`.
- `63229_face_augmentations.mp4` — Clip 63229 (hand to chin): source video, the 3-D copy with the face attached, and six augmentations at their safe ranges (camera turn 20°, mirror, 85% size, shift, 1.25× speed, arm joints 4°). Played at half speed.
- `05631_face_augmentations.mp4` — Clip 05631, where the signer's palm rests on their cheek: the same layout, showing that the touch is lost in 3-D (the hand ends up 15–18 cm from the face). Played at quarter speed, since the clip is only 24 frames.
- `17083_face_augmentations.mp4` — The same layout for clip 17083, where the mouth movement shows the face carried through each augmentation. Played at half speed.
- `63669_face_augmentations.mp4` — Clip 63669 (hand raised near the chin), in a wider layout: the 2-D input, the round trip and eight augmentations (yaw 25°, mirror, 85% size, off-centre shift, elbows +15° / shoulders +10°, body shape betas[0] +2, head turn 20°, time warp k=0.5). It also shows a per-augmentation table of face and hand-to-face measurements and a hand-to-face distance strip. 40 frames, played at normal speed. `face_augment_samples.py` does not produce this layout.
- `17083_first_trial_body_only.mp4` — The first augmentation trial (body and hands, no face), whose stills are on slides 11 and 12. Played at half speed.
- `face_augment_samples.py` — Script that renders the three face clips. It needs the face code on `feat/face-v1.3`: run it from that worktree's root with `PYTHONPATH=src:scripts .venv/bin/python <path>/face_augment_samples.py --clips 63229 17083 05631 --out <dir>`; clip 05631 was rendered with an extra `--note` caption about the lost touch.
- `index.md` — This index.
