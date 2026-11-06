# Demo provenance

`twitter_demo.mp4`: approximately39seconds,1280×720 H.264,30fps, no audio. The recorded comparison plays at2×, followed by a4second explanation card. `comparison_1x.mp4` preserves the comparison at1×.

The selected held-out gain is seed200017. Both views replay the actual recorded policy actions; all saved states and every4ms physics sample matched the original rollouts exactly. Simulation time55–125s is shown. The edited episode finishes at92.09s; its final frame is then held and explicitly labeled. It is not an uninterrupted live inference recording.

This is one selected gain, not all ten episodes. The aggregate1/10→6/10 result comes from all ten paired scenes. `replay_audit.json` records the source audits. Full test traces and outcomes are in `results/`. No generated robot footage or scripted replacement grasp is used.
