# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Equivariant Diffusion Policy (equidiff) with ManiFlow integration for robotic manipulation. Trains diffusion-based and consistency flow-based policies on MimicGen/Robomimic tasks using Hydra-configured workspaces.

- **Language:** Python 3.9
- **Package Manager:** Conda/Mamba
- **Config System:** Hydra + OmegaConf
- **Core Dependencies:** PyTorch 2.1.0 (CUDA 11.8), pytorch3d 0.7.5, hydra-core 1.2.0, wandb, robomimic, robosuite
- **Main Package:** `equi_diffpo`

### Success Criteria

Success rate is measured by `test_mean_score` in logs and WandB. Checkpoint TopK selection uses `test_mean_score` (higher is better).

## Common Commands

### Setup
```bash
mamba env create -f conda_environment.yaml
conda activate equidiff
pip install -e .
./setup_maniflow_policy.sh   # installs timm, verifies ManiFlow imports
```

Prerequisites: `sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf gfortran`

MimicGen must be installed separately:
```bash
git clone https://github.com/NVlabs/mimicgen_environments.git
cd mimicgen_environments && git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9 && pip install -e .
```

### Training
```bash
# ManiFlow point cloud policy (baseline)
python train.py --config-name=train_maniflow_pointcloud_workspace task_name=stack_d1 n_demo=100

# ManiFlow image policy
python train.py --config-name=train_maniflow_image_workspace task_name=stack_d1 n_demo=100

# SE3 manifold policy (rotation-aware variant)
python train.py --config-name=train_maniflow_se3_pointcloud_workspace task_name=stack_d1 n_demo=100

# Relative rotation policy
python train.py --config-name=train_maniflow_relrot_pointcloud_workspace task_name=stack_d1 n_demo=100

# Equivariant diffusion (original equidiff)
python train.py --config-name=train_equi_diffusion_unet_abs task_name=stack_d1 n_demo=100

# Run MuJoCo rendering on CPU to save GPU memory
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python train.py --config-name=...

# Reduce memory usage
python train.py --config-name=... policy.n_layer=6 policy.n_emb=512 dataloader.batch_size=64
```

Available task names: `stack_d1`, `stack_three_d1`, `square_d2`, `threading_d2`, `coffee_d2`, `three_piece_assembly_d2`, `hammer_cleanup_d1`, `mug_cleanup_d1`, `kitchen_d1`, `nut_assembly_d0`, `pick_place_d0`, `coffee_preparation_d1`.

### Dataset Preparation
```bash
# Generate point cloud / voxel observations from raw dataset
python equi_diffpo/scripts/dataset_states_to_obs.py \
  --input data/robomimic/datasets/<task>/<task>.hdf5 \
  --output data/robomimic/datasets/<task>/<task>_voxel.hdf5 \
  --num_workers=24

# Convert relative to absolute action space
python equi_diffpo/scripts/robomimic_dataset_conversion.py \
  -i data/robomimic/datasets/<task>/<task>_voxel.hdf5 \
  -o data/robomimic/datasets/<task>/<task>_voxel_abs.hdf5 \
  -n 12
```

Datasets go in `equi_diffpo/data/robomimic/datasets/<task_name>/`.

### Verification
```bash
python test_maniflow_import.py    # Tests imports, policy creation, forward pass
```

## Architecture

### Execution Flow

```
train.py (Hydra entry) → OmegaConf resolvers → Workspace.__init__(cfg) → Workspace.run()
```

1. `train.py` registers custom OmegaConf resolvers (`get_max_steps`, `get_ws_x_center`, `get_ws_y_center`, `eval`)
2. Hydra loads config from `equi_diffpo/config/`, which composes with task defaults in `equi_diffpo/config/task/`
3. The config's `_target_` field specifies which Workspace class to instantiate
4. `workspace.run()` executes the full training loop: data loading, training epochs, validation, rollouts, checkpointing, WandB logging

### Workspace Pattern

Each policy variant has a paired workspace class and YAML config:

| Config YAML | Workspace Class | Policy Class |
|-------------|----------------|--------------|
| `train_maniflow_pointcloud_workspace` | `TrainManiFlowPointcloudWorkspace` | `ManiFlowTransformerPointcloudPolicy` |
| `train_maniflow_image_workspace` | `TrainManiFlowImageWorkspace` | `ManiFlowTransformerImagePolicy` |
| `train_maniflow_se3_pointcloud_workspace` | `TrainManiFlowSE3PointcloudWorkspace` | `ManiFlowSE3PointcloudPolicy` |
| `train_maniflow_relrot_pointcloud_workspace` | `TrainManiFlowRelRotPointcloudWorkspace` | `ManiFlowRelRotPointcloudPolicy` |
| `train_equi_diffusion_unet_abs` | `TrainEquiWorkspace` | Equivariant U-Net |

All workspaces inherit from `BaseWorkspace` which provides checkpoint save/load, snapshot serialization, and threaded saving.

The training loop in each workspace follows the same structure:
1. Resume from checkpoint if available
2. Instantiate dataset, normalizer, optimizer, LR scheduler, optional EMA
3. Epoch loop: train batches → validation → env rollout → checkpoint (TopK + latest)
4. All metrics logged to WandB and `logs.json.txt`

### Key Modules

**`equi_diffpo/policy/maniflow/`** — ManiFlow policy variants (the main focus of development):
- `maniflow_pointcloud_policy.py` — Baseline: 10D state + 10D velocity in Euclidean space, Gaussian noise, linear interpolation
- `maniflow_se3_pointcloud_policy.py` — SE(3) manifold: 10D state (6D rotation repr) + 7D velocity (angular velocity), SO(3) geodesic interpolation, rotation-aware ODE updates
- `maniflow_relrot_pointcloud_policy.py` — Relative rotation: direct rotation prediction variant
- `maniflow_image_policy.py` — RGB image observations with TimmObsEncoder
- `maniflow_right_triv_pointcloud_policy.py` — Right trivialization variant

**`equi_diffpo/model/diffusion/`** — Core neural architectures:
- `ditx.py` / `ditx_block.py` — DiTX transformer (ManiFlow's backbone)
- `ema_model.py` — Exponential Moving Average wrapper

**`equi_diffpo/model/vision_2d/`** — Image encoders (timm-based: ResNet, ViT, EfficientNet)

**`equi_diffpo/model/vision_3d/`** — Point cloud encoders (PointNet, DP3Encoder)

**`equi_diffpo/dataset/`** — HDF5 dataset loaders for Robomimic format (image, point cloud, voxel, low-dim variants)

**`equi_diffpo/env_runner/`** — Episode rollout runners for evaluation in MuJoCo environments

**`equi_diffpo/model/common/normalizer.py`** — `LinearNormalizer` used by all policies for data normalization

### Policy Interface

All policies implement:
- `compute_loss(batch, ema_model=None) → (loss, loss_dict)` — Training loss
- `predict_action(obs_dict) → action_dict` — Inference
- `set_normalizer(normalizer)` — Sets data normalizer
- `get_optimizer(...)` — Returns optimizer

### SE3 vs Baseline Policy (Key Distinction)

| Aspect | Baseline (pointcloud) | SE3 |
|--------|----------------------|-----|
| State representation | 10D state + 10D velocity | 10D state [pos(3), rot_6d(6), gripper(1)] + 7D velocity [pos_vel(3), omega(3), gripper_vel(1)] |
| Noise sampling | All Gaussian | Position: Gaussian, Rotation: SO(3) uniform, Gripper: Gaussian |
| Interpolation | All linear | Position/gripper: linear, Rotation: SO(3) geodesic |
| ODE update | `x = x + v * dt` | Position: `pos += v * dt`, Rotation: `R = R @ exp(dt * omega)`, Gripper: `g += v * dt` |
| Normalization | All dimensions | Only position and gripper; rotation keeps raw 6D representation |
| Training loss | Flow + Consistency dual loss | Flow only, separate losses for position/rotation/gripper |

### Configuration

Configs in `equi_diffpo/config/` compose with task defaults (`equi_diffpo/config/task/`):
- `mimicgen_pc_abs.yaml` — Point cloud + absolute actions
- `mimicgen_abs.yaml` — Image + absolute actions
- `mimicgen_rel.yaml` — Image + relative actions
- `mimicgen_voxel_abs.yaml` — Voxel + absolute actions

Custom OmegaConf resolvers defined in `train.py`:
- `${get_max_steps:task_name}` — Task-specific max episode steps
- `${get_ws_x_center:task_name}` — Workspace center X (-0.2 for kitchen/hammer_cleanup, 0 otherwise)
- `${eval:'python_expr'}` — Arbitrary Python expression evaluation

Key config sections: `policy` (architecture params), `training` (epochs, device, EMA, rollout frequency), `dataloader`, `optimizer`, `logging` (WandB), `checkpoint` (TopK by `test_mean_score`).

### Outputs

Training outputs go to `data/outputs/<date>/<time>_<config>_<task>/`:
- `checkpoints/latest.ckpt` — Latest checkpoint
- `checkpoints/epoch=XXXX-test_mean_score=X.XXX.ckpt` — TopK checkpoints
- `logs.json.txt` — JSON-lines training log
- WandB logs uploaded to project `maniflow_pointcloud_<task_name>` (or equivalent)
