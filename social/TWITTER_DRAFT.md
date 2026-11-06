# Twitter/X draft — not posted

Attach `media/twitter_demo.mp4` to the first post. It is a selected gain, labeled simulation and 2× playback. The repository contains all ten paired outcomes.

## Main post

I changed one learned weight in G0.5—no training.

On 10 paired simulation trials, mug-hanging success went from 1/10 to 6/10: five gains, zero regressions.

Small pilot, not a reliability claim. Video shows one selected gain.

Code + results: https://github.com/afinitus/g05-weight-space-programming

## Optional thread

**2/** The robot could often pick up and transfer the mug, then miss the rack when releasing it. I used forward-only probes to find an output feature that influenced the approach more than the earlier regrasp. No gradients or optimizer steps.

**3/** The edit changes one matrix coefficient:
`action_expert.output_proj.weight[10,168] += 0.2`

Six separate development scenes selected the edit. It was frozen before reviewing the ten paired test outcomes. Same starts, cameras, seeds and scoring.

**4/** Success meant hanging on the rack for one second after release, with no robot contact. Every 4 ms of physics was logged. A development candidate with a suspicious rack-contact event was rejected. Passing these checks does not prove physical realism.

**5/** Limits: ten test scenes; exact paired p=0.0625; simulation only. Two failed trials still have contact-dynamics cautions. This uses released G0.5 flow matching + ALOHA-AgileX in RoboTwin/SAPIEN—not an official benchmark reproduction.

**6/** The interesting part is the intervention: one coefficient caused a useful behavioral change without training. It is not proof that the feature means “hang the mug,” or that weight editing beats optimized action calibration. Full traces, patch and failures are in the repo.

## Short alternative

One learned weight changed. No training.

G0.5 mug hanging: 1/10 → 6/10 on the same ten held-out simulation scenes.

Five gains, no regressions. Small pilot; selected gain shown. Full results + physics caveats:
https://github.com/afinitus/g05-weight-space-programming

## Before posting

- Use “one learned weight,” not “one neuron” or “one line fixed robotics.”
- Keep “simulation,” “ten trials,” and “selected gain” visible.
- Do not claim statistical significance at the conventional 0.05 level, real-world reliability, or an official Galaxea result.
- Do not call this fine-tuning, an ActionCodec edit, or an autoregressive-policy experiment.
- This is a draft for the account owner to review; nothing has been posted to X.
