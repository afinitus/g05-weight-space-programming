# G0.5 Hanging Mug: one learned-weight edit

Completed 2026-11-06T19:24:22.866527+00:00.

## Measured outcome

On ten frozen, paired held-out RoboTwin Hanging Mug configurations, original G0.5 completed **1/10 supported hangs**, and the frozen one-weight edit completed **6/10**, a **+50 percentage-point** change. Development improved from1/6 to6/6, repeated exactly before test evaluation. This is a small simulation pilot, not a100-episode reliability study or a hardware result.

| Metric | Original G0.5 | Edited G0.5 |
|---|---:|---:|
| Held-out episodes |10|10|
| Supported hangs for1second |1|6|
| Success rate |10%|60%|
| Native criterion reached at any point |1|6|
| Mean planar handle error at first right release |4.00 cm (9/10 releases observed)|3.40 cm (9/10 releases observed)|
| Learned scalars changed |0|1|

Both succeeded: 1; both failed: 4; only edited succeeded: 5; only original succeeded: 0. Exact two-sided McNemar p=0.0625. Wilson95% intervals: original 1.8%–40.4%, edited 31.3%–83.2%. Paired bootstrap95% interval for the success difference: 20.0%–80.0%;100,000 resamples, seed271828. Small-n intervals are broad; the development search is not an unbiased performance estimate.

Release error is a diagnostic, not a success criterion: first crossing of normalized right-gripper opening0.2 after action150, handle functional point relative to the rack's native target midpoint, xy norm. A crossing can be an unsuccessful release. Missing releases are reported, not imputed or dropped from success denominators.

## Task and model

Native RoboTwin/SAPIEN task `hanging_mug`, ALOHA-AgileX dual arm. Official instruction: “Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.” The released `g05-robotwin20` **continuous flow-matching** checkpoint is used, with10 integration steps,8 executed actions per chunk, BF16 backbone and FP32 action output projection. This is not the paper's autoregressive benchmark configuration and not the original R1 Lite Coke-can task. No ActionCodec claim is made for this FM experiment.

Inputs are the original head and two wrist RGB images, supported14D joint/gripper state, and instruction. Simulator coordinates never enter policy input or action corrections. The policy controls the complete task. No scripted grasp, release, handoff or rack approach replaces its actions.

Native success checks handle xy within2cm of the rack midpoint, handle z>0.86m, and right gripper open. Our primary outcome additionally requires a full1second with native geometry true throughout, rack contact at least95% of250 physics steps, and zero robot contact. Robot commands are held during this post-release verification. If the hold fails, the original policy continues until900 actions.

We record native success during control but prevent native early termination from cutting a command midway through gripper opening. The policy finishes its command and continues until unsupported rack contact permits the hold check. This uniform correction was made before weight search; initial integration results are archived separately. This is an evaluation protocol change relative to native instantaneous termination, not a learned edit.

Six development and ten test configurations are the first16 expert-feasible scenes among38 consecutively screened seeds200000–200037. Feasibility requires the native expert to complete the same supported hold; it is not conditioned on G0.5 success. This restricts the evaluated distribution. Scene seeds randomize mug/rack poses and native mug assets. Splits were frozen before policy testing. Every paired test has exactly equal initial physical states and byte-identical images from all three policy cameras. RNG seeds, action horizon, rendering, physics and model settings match.

## Diagnosis and causal intervention

Baseline development succeeded1/6. First right releases showed a repeatable negative-x handle displacement relative to the rack, approximately−2.5 to−4.0cm. We traced the physical right-base joint to padded output row10, decoded action index7. Constant row10 bias edits shifted approach trajectories but also disrupted earlier manipulation; a +0.16 bias obtained4/6 but caused a suspicious7.4m/s rack contact event and was rejected by the physics gate.

We inspected1024 output-projection input features across twelve cached development observations: six during regrasping and six during rack approach. Feature168 was much stronger during approach. Two matrix-coefficient candidates were added after forward-only causal tests; the bounded rollout budget increased from8 to10 before held-out outcome review. The feature is not claimed to be a semantic “hanging” neuron.

For the selected coefficient, forward intervention on fixed cached observations produced mean absolute right-base changes of0.001003rad during regrasping and0.014496rad during approach. The mean approach shift was negative. Other joint changes reached0.011450rad; gripper changes reached0.000694 in their normalized units. Effects are therefore not confined to one physical action coordinate: flow integration and later closed-loop observations propagate the intervention. Repeated baseline forward calls were identical and all probed outputs were finite. See `results/weight_forward_sensitivity.json` and `results/activation_diagnosis.json`.

### Exact edit

- Tensor: `model.action_expert.output_proj.weight`, shape20×1024.
- Index: `[10,168]`, flattened index10408.
- Original FP32 value: **0.06640625**.
- Edited FP32 value: **0.26640623807907104**.
- Requested delta: +0.2; actual delta, L1 and L2 norms: **0.19999998807907104**.
- Changed scalars: **1 of2,859,995,988** unique model parameters, **3.4965084014e-08%**.
- This is sparse support, not a small relative amplitude: the requested delta is about13.7 times the output row's coefficient standard deviation and about3.01 times this coefficient's original magnitude.
- Original checkpoint SHA256: `300d24c691f8d76846a218b9bd670312cf577b701ea5863f221af1b8d1c0d637`, verified unchanged after search.
- Frozen patch SHA256: `d2f4df4c42b09cebdc01b271d52ef361df46e55dd11aee375a9316c4a99739dc` at `2026-11-06T19:08:13.477577+00:00`.

All edits are direct inference-time PyTorch parameter assignments under disabled gradients. No backward pass, optimizer step, supervised training, RL training or codec retraining was used. Reversible sparse manifests preserve old values and deltas; original checkpoint files were never edited.

### Complete candidate ledger

Fractions below use **completed development episodes**. Pruned candidates were not all evaluated on all six configurations. All completed failures remain; interrupted partial trajectories are marked unscored. No uncompleted candidate could beat the eligible6/6 winner because each had already failed at least one scene, or was physics-ineligible.

| Candidate | Output-projection support | Requested delta | Successes/completed | Rack flags | Status |
|---|---|---:|---:|---:|---|
| right_base_feature168_p0p20 | weight[10, 168] | +0.2 | 6/6 | 0 | complete |
| right_base_feature200_p0p08 | weight[10, 200] | +0.08 | 1/2 | 0 | pruned |
| right_base_m0p32 | bias[10] | -0.32 | 0/4 | 0 | pruned |
| right_base_p0p08 | bias[10] | +0.08 | 3/4 | 0 | pruned |
| right_base_p0p12 | bias[10] | +0.12 | 3/4 | 0 | pruned |
| right_base_p0p16 | bias[10] | +0.16 | 4/6 | 1 | pruned |
| right_base_p0p20 | bias[10] | +0.2 | 1/3 | 0 | pruned |
| right_base_p0p24 | bias[10] | +0.24 | 0/3 | 0 | pruned |
| right_base_p0p32 | bias[10] | +0.32 | 0/5 | 0 | pruned |
| right_base_p0p64 | bias[10] | +0.64 | 0/4 | 0 | pruned |

The winner was rerun on the original four worker assignments. All six action trajectories, mug/rack/joint arrays and every4ms physics sample matched screening exactly. It was frozen before reviewing held-out baseline outcomes; test outcomes did not select or revise the edit.

## Remaining edited-policy failures

| Seed | Observed stage | Maximum mug rise(cm) | First-release handle xy error(cm) |
|---|---|---:|---|
| 200014 | lift achieved but supported hang not achieved | 31.94 | [-3.23, -2.17] |
| 200016 | no5cm lift achieved | 0.27 | No release observed |
| 200021 | lift achieved but supported hang not achieved | 32.04 | [-3.87, -1.82] |
| 200027 | lift achieved but supported hang not achieved | 31.70 | [-5.48, -14.9] |

These are pose-based stage descriptions, not complete causal diagnoses. A correction aimed at the right-arm rack approach cannot be assumed to fix an initial pickup failure. No failing test scene was removed or used for retuning. Full stage metrics for both models are in `results/failure_modes.json`.

## Action-calibration controls

These controls change decoded action index7, not learned weights, and use only development scenes:

- Constant decoded right-base offset -0.03 rad: **0/6**. Initial policy images match original: True.
- Constant decoded right-base offset -0.0145 rad: **2/6**. Initial policy images match original: True.

The−0.03rad dose matches an earlier bias probe approximately. The−0.0145rad dose matches the selected edit's average cached late-approach effect; it is applied globally rather than through a learned feature. These are two limited controls, not an optimized calibration search or a held-out comparison. Their results cannot establish broad superiority of weight-space programming.

## Physics validity and video audit

All episodes log mug positions, speeds, penetration, contact impulses, rack/robot/table/ground contact every4ms. Final20-trial rack audit: **0 flagged episodes** under the development gate (rack-contact speed or velocity jump>3m/s while mug origin z>0.8m). Maximum rack-contact speed=1.1363m/s; maximum rack-contact velocity jump=1.2054m/s; maximum mug contact penetration=8.492mm. Flags and full traces remain in `results/final_physics_audit.json`; no flagged test episode is discarded.

The failed edited trial200027 reached2.998m/s during robot contact (not rack contact), just below the3m/s audit threshold, and later ended on the ground. Failed trial200014 reached8.49mm contact penetration. These remain reasons for caution about failure physics; neither is counted as an improvement. All six edited successes had peak speeds at most1.137m/s. This audit can detect the previously observed basket-style high-speed contact issue, not every possible simulator error. Native mug mass is10g. Contact solver parameters and assets are inherited from RoboTwin; they have not been validated against hardware. Passing a contact-speed audit and deterministic replay do not prove realism. This experiment is separate from the earlier basket experiment, whose skill-improvement interpretation was withdrawn after suspicious contact dynamics.

All scored videos capture15 frames per simulated second. Unlike action-index video sampling, this preserves falls between variable-duration actions. Full raw videos are stored beside each `result.json`; website copies are compressed. Any close presentation view is explicitly a replay of recorded policy actions, verified against every recorded state and physics sample, not a new policy rollout. Cropped example clips do not replace full episodes or the complete score table.

## Reproduction and provenance

See [REPRODUCING.md](docs/REPRODUCING.md) and [configs/provenance.json](configs/provenance.json) for the pinned sources, assets, checkpoint hash and environment requirements. The public copy makes instance-specific paths configurable; it does not represent a fresh rerun. All20valid paired test outcomes and full physics/action traces are included. All completed development candidate results are retained. Integration-only runs were excluded before scoring; the original evidence archive is retained privately.

## Limitations

This is a personal, independent simulation experiment, not an official Galaxea result. No affiliation or endorsement is implied. The ten test scenes are expert-feasible and were frozen before policy testing. This does not establish reliability across100scenes, other tasks or physical hardware. Do not retune on these ten scenes and describe them as held-out again.

The original model and simulator assets are not redistributed. Obtain them from their respective upstream sources under their applicable terms.
