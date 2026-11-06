# What the demo shows

The task is more than picking up a mug: the left arm picks it up and reorients it, places it near the middle, and the right arm picks it up again and hangs it on a rack.

The original G0.5 policy often completed the early stages but released the mug with its handle misaligned to the rack. We tested small direct parameter interventions using only forward passes, then evaluated candidates on six development scenes. Constant bias edits could help the approach but also disrupted earlier handling. One candidate produced a suspicious contact event and was rejected.

We then inspected features entering the action expert's final projection. Feature168 was more active during the rack approach than during regrasping. Increasing its connection to output row10 changed the decoded right-base joint more during approach. This offered a state-dependent effect compared with a constant action correction. The feature's semantic meaning remains unknown; other action coordinates also changed through flow integration and feedback.

The edit changes exactly one FP32 matrix coefficient:

- Tensor: `model.action_expert.output_proj.weight`.
- Index: `[10,168]`.
- Before: `0.06640625`.
- After: `0.26640623807907104`.
- Requested delta: `+0.2`.

“One weight” describes sparse support. The perturbation is not small relative to the original coefficient: the delta is about three times its initial magnitude.

The selected candidate completed6/6 development scenes, versus1/6 original. An independent confirmation reproduced every logged action and physics sample. We froze the patch, then evaluated on ten separate paired test scenes: **1/10 original,6/10 edited**. Five improvements, no regressions.

In the selected video pair, seed200017, planar handle error at first release changed from about4.9cm to0.5cm, followed by one second of rack-supported hanging without robot contact. This example is selected to illustrate a gain; it is not the whole evaluation.

The result is a useful causal intervention within this simulator, not proof of a complete internal mechanism. Exact paired p=0.0625, and only ten scenes were tested. No hardware was tested. Contact audits did not flag final rack events, but failed trajectories retain physics cautions. Broader evaluation and stronger calibration controls are needed before making larger claims.
