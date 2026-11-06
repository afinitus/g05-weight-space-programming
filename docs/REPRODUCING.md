# Reproducing the experiment

## Recalculate the published result (no GPU)

Use Python with NumPy, SciPy and Matplotlib installed:

```bash
python src/final_analysis.py
python src/audit_final.py
python src/failure_modes.py
python src/plot_final.py
```

These scripts read the committed traces and regenerate the statistics/plots. Expected result: baseline1/10, edited6/10, five gains, zero losses, exact McNemar p=0.0625. This recomputes evidence; it does not rerun the neural policy.

## Run new policy rollouts

The original experiment used separate model (Python3.11) and simulator (Python3.10) environments on RTX PRO6000 Blackwell GPUs. Full fresh-machine installation is **not automated or validated by this public packaging step**. Install the official dependencies, model and assets first. Use CUDA-compatible PyTorch/cuRobo builds for your hardware.

Pinned upstream sources:

- GalaxeaVLA: `89f2322b4ad016e192437adc1a2c253b05bab246`.
- RoboTwin: `13c3c47ff4312dd62484bcd51be034af55c062d1`.
- cuRobo: `36ea382dabbde2a2431951fc9048f7e5ffd8106b`.
- Hugging Face `OpenGalaxea/G05`, revision`e312be81e90c56a55bcb26b57429bd39a335b449`, `g05-robotwin20/`.
- Assets: `TianxingChen/RoboTwin2.0`, revision`3dc3b798668feb99ac61cc9086d84cbcc3d79186`, `embodiments.zip` and `objects.zip`, extracted into RoboTwin's assets directory.

Original checkpoint SHA256:
`300d24c691f8d76846a218b9bd670312cf577b701ea5863f221af1b8d1c0d637`.

Apply `configs/robotwin_local.patch` for the original OptiX denoiser setting. The cuRobo helper-name compatibility patch is provided for the original CUDA build; review applicability to your environment. Neither changes task scoring or learned weights. No host driver modifications are needed by these scripts.

Set absolute source paths and create a separate output directory to avoid mixing new rollouts with published evidence:

```bash
export ROBOTWIN_ROOT=/absolute/path/to/RoboTwin
export GALAXEA_VLA_ROOT=/absolute/path/to/GalaxeaVLA
export G05_EXPERIMENT_ROOT=/absolute/path/to/new-mug-run
mkdir -p "$G05_EXPERIMENT_ROOT"/results "$G05_EXPERIMENT_ROOT"/logs "$G05_EXPERIMENT_ROOT"/videos
cp -r configs patches "$G05_EXPERIMENT_ROOT"/
# Place the authorized checkpoint and dataset_stats.json at:
# $G05_EXPERIMENT_ROOT/checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt
# $G05_EXPERIMENT_ROOT/checkpoints/g05-robotwin20/dataset_stats.json
```

From the repository, using the model environment in terminal1:

```bash
CUDA_VISIBLE_DEVICES=0 python src/model_server.py --port 18300
```

Using the simulator environment in terminal2, on the same host:

```bash
export CUDA_VISIBLE_DEVICES=1
export ROBOTWIN_RENDER_DEVICE=cuda:1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
python src/run.py eval --split dev --port 18300
python src/run.py eval --split test --port 18300
python src/run.py eval --split test --port 18300 \
  --patch "$G05_EXPERIMENT_ROOT/patches/right_base_feature168_p0p20.json"
```

Run these sequentially. Each evaluation restores or applies its patch, resets policy RNG by episode seed, and skips completed result files. Do not patch/reset an occupied server. For parallel workers, give each a distinct model server/port and use `--worker W --workers N`. Model and renderer GPU selections are separate; verify both on your node.

`src/activation_server.py` adds a forward-feature endpoint. The corresponding diagnostics use port18400 and cached original-development observations; they require an **exclusive** diagnostic server and newly generated observation caches. `probe_weights.py` writes candidate patches, so run it in a separate experimental output directory. Do not modify the frozen public patch. Development search was an adaptive reasoning process, not a packaged automatic optimizer.

The current public script copy only replaces private instance paths with configurable roots. This packaging has been checked by recompiling the Python files and reproducing the stored statistics; **new GPU rollouts have not been rerun from a clean installation**.

## Evaluation details

The model executes eight actions per chunk, with ten flow-matching integration steps, BF16 backbone and FP32 output projection. Maximum900 policy actions. The policy gets its supported three RGB cameras and14D proprioception, not simulator object coordinates.

Native success is recorded but does not stop a command midway through gripper opening. At an action boundary, native geometry plus rack contact and no robot contact triggers250physics steps of hold verification. Native geometry must persist throughout, rack contact must occur in at least95% of steps, and robot contact must remain absent. Failed holds return to policy control. This applies uniformly to all conditions.

The six development and ten test scenes were frozen before policy evaluation, drawn from the first16 expert-feasible scenes among38 screened seeds. This is a constrained scene distribution. Do not tune a new edit using the committed ten test outcomes and then call those scenes held-out.
