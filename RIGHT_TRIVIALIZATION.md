# Right Trivialization Manifold Flow Matching

## Overview

This policy implements **right trivialization** for flow matching on SO(3), where angular velocities are expressed in the **body frame** (local coordinates) instead of the spatial frame (world coordinates).

## 🎯 What is Trivialization?

**Trivialization** is how we represent tangent vectors (velocities) on a manifold using a consistent coordinate system.

### Visual Intuition

Imagine a rotating robot arm:

```
         ┌─────┐
         │ EE  │  ← End effector
         └──┬──┘
            │
         ┌──┴──┐
         │ ARM │
         └─────┘
            ↓
        ========  ← Base
```

**Question:** How do we describe the angular velocity of the end effector?

1. **LEFT Trivialization (Spatial/World Frame):**
   - "The end effector is rotating around the Z-axis of the **world frame** at 0.5 rad/s"
   - ω is expressed relative to fixed world coordinates
   - Natural for vision-based systems

2. **RIGHT Trivialization (Body/Local Frame):**
   - "The end effector is rotating around its **own local Z-axis** at 0.5 rad/s"
   - ω is expressed relative to the current orientation
   - Natural for robot control (joint velocities)

---

## 📐 Mathematical Formulation

### Left Trivialization (SE3 policy)

**Velocity in spatial frame:**
```
dR/dt = R @ [ω_spatial]_×
```

**Update rule:**
```python
R_{t+dt} = R_t @ exp(dt * ω_spatial)  # Post-multiply
```

**Velocity computation:**
```python
ω_spatial = R_0 @ log(R_0^T @ R_1)  # Transform to spatial frame
```

### Right Trivialization (This policy)

**Velocity in body frame:**
```
dR/dt = [ω_body]_× @ R
```

**Update rule:**
```python
R_{t+dt} = exp(dt * ω_body) @ R_t  # PRE-multiply!
```

**Velocity computation:**
```python
ω_body = log(R_0^T @ R_1)  # Already in body frame!
```

---

## 🔄 Key Difference

| Aspect | LEFT Trivialization | RIGHT Trivialization |
|--------|---------------------|----------------------|
| **Frame** | Spatial/World | Body/Local |
| **Update** | `R_new = R_old @ exp(dt*ω)` | `R_new = exp(dt*ω) @ R_old` |
| **Multiplication** | Post-multiply (right) | PRE-multiply (left) ⚠️ |
| **Velocity** | `ω = R @ log(R₀ᵀR₁)` | `ω = log(R₀ᵀR₁)` |
| **Intuition** | "Rotate around world axes" | "Rotate around body axes" |
| **Use case** | Vision, global planning | Robot control, manipulation |

**Important naming note:**
- RIGHT trivialization → PRE-multiplication (confusing but standard!)
- LEFT trivialization → POST-multiplication

---

## 🤖 Why Right Trivialization for Robotics?

### 1. **Natural for Robot Control**

Robots often think in body frame:
```python
# Body frame (natural for robot)
"Move forward 0.1m, rotate left 30°"  # Relative to current pose

# Spatial frame (less intuitive)
"Move to position (x, y, z) in world, orient to (θ_world)"
```

### 2. **Joint Velocities are Body Velocities**

Most robots receive commands as:
```python
joint_velocities = [ω_1, ω_2, ω_3, ...]  # In joint frames (body)
```

### 3. **Consistency with Robot Kinematics**

Forward kinematics typically computes:
```python
T_world_ee = T_base @ T_1 @ T_2 @ ... @ T_n  # PRE-multiply chain!
```

Right trivialization matches this natural chain multiplication.

### 4. **More Stable for Manipulation**

When grasping, object-relative (body frame) velocities are more meaningful:
```python
# Body frame: "Rotate 5° around gripper Z-axis"
# Spatial frame: "Rotate 5° around world Z... wait, where is world Z relative to gripper?"
```

---

## 📊 Comparison of All Three Policies

| Policy | Frame | Representation | Update Rule | Matches Env |
|--------|-------|----------------|-------------|-------------|
| **SE3 (Left Triv)** | Spatial | 7D: [pos, ω_spatial, grip] | `R @ exp(dt*ω)` | ❌ (10D) |
| **RelRot** | N/A | 10D: [pos, ΔR_6d, grip] | `R @ ΔR^dt` | ✅ (10D) |
| **Right Triv** | Body | 7D: [pos, ω_body, grip] | `exp(dt*ω) @ R` | ❌ (10D) |

---

## 🔬 When to Use Which?

### Use **LEFT Trivialization** (SE3 policy) when:
- ✅ Working with global path planning
- ✅ Vision-based control (camera frame is world-fixed)
- ✅ Need spatial frame semantics
- ✅ Debugging spatial trajectories

### Use **RIGHT Trivialization** (this policy) when:
- ✅ Robot manipulation tasks (grasp, place, etc.)
- ✅ Working with joint-space controllers
- ✅ Object-relative operations
- ✅ Need body frame semantics
- ✅ Potentially better numerical stability for rotation sequences

### Use **RelRot** when:
- ✅ Want simplest implementation (no exp/log in inference)
- ✅ Match environment format directly
- ✅ Don't need explicit velocity semantics
- ✅ Fastest inference speed

---

## 💻 Implementation Details

### Core Functions

#### 1. **Interpolation** (with right trivialization)

```python
def interpolate(self, noise, target, t):
    # Position: same as left triv
    pos_t = (1-t) * pos_0 + t * pos_1

    # Rotation: RIGHT trivialization geodesic
    R_delta = R_0^T @ R_1
    R_t = exp(t * log(R_delta)) @ R_0  # PRE-multiply!
```

**Difference from left triv:**
```python
# Left:  R_t = R_0 @ exp(t * log(R_delta))  # POST-multiply
# Right: R_t = exp(t * log(R_delta)) @ R_0  # PRE-multiply
```

#### 2. **Body Velocity Computation**

```python
def get_body_velocity(self, x_0, x_1):
    R_0 = exp_so3(x_0.rotation)
    R_1 = exp_so3(x_1.rotation)

    # Body velocity (right triv)
    ω_body = log(R_0^T @ R_1)  # Simpler than left triv!

    # Left triv would be:
    # ω_spatial = R_0 @ log(R_0^T @ R_1)  # Extra rotation
```

#### 3. **ODE Integration**

```python
def sample_ode(self, x0, N):
    for i in range(N):
        ω_body = network(x, t)  # Predict body velocity

        # RIGHT trivialization update
        R_current = exp_so3(x.rotation)
        dR = exp_so3(dt * ω_body)
        R_new = dR @ R_current  # PRE-multiply!
```

**Critical difference:**
```python
# Left:  R_new = R_current @ dR  # POST
# Right: R_new = dR @ R_current  # PRE
```

---

## 🧪 Training

### Quick Start

```bash
conda activate equidiff

# Train with right trivialization
python train.py --config-name=train_maniflow_right_triv_pointcloud_workspace
```

### Compare All Three

```bash
# Terminal 1: Left trivialization (spatial frame)
python train.py --config-name=train_maniflow_se3_pointcloud_workspace

# Terminal 2: Right trivialization (body frame)
python train.py --config-name=train_maniflow_right_triv_pointcloud_workspace

# Terminal 3: Relative rotation (no frame concept)
python train.py --config-name=train_maniflow_relrot_pointcloud_workspace
```

---

## 📈 Expected Differences

### Training Behavior

**Hypothesis:**
- **Right triv** may converge faster for manipulation (body velocities more consistent)
- **Left triv** may be better for path following (spatial trajectories smoother)
- **RelRot** should be most stable (no velocity frame issues)

### Inference

All three should produce similar final actions, but:
- **Right/Left triv** have explicit velocity semantics
- **RelRot** is fastest (no exp/log in ODE)
- **Right triv** may have better numerical properties for long rotation sequences

---

## 🔍 Debugging Tips

### Check if Right Triv is Actually Being Used

```python
# In ODE sampling, you should see PRE-multiplication:
R_new = exp(dt * ω_body) @ R_current  # ✅ Correct (right triv)
R_new = R_current @ exp(dt * ω_body)  # ❌ Wrong (left triv)
```

### Verify Body Frame Velocities

```python
# Body velocity should be independent of current rotation
# Test: If R_0 = I, then ω_body = log(R_1)
```

### Compare with SE3 Policy

```python
# Both should reach same final pose, but with different paths
# Right triv: rotates in body frame (may spiral)
# Left triv: rotates in spatial frame (may arc)
```

---

## 📚 Mathematical Background

### Tangent Space Identification

**SO(3) tangent space at identity:** `TₑSO(3) ≅ so(3)` (3×3 skew-symmetric matrices)

**Right trivialization map:**
```
TᵣSO(3) → so(3)
v ↦ R^T @ v  (pull back to identity)
```

**Left trivialization map:**
```
TᵣSO(3) → so(3)
v ↦ v @ R^T  (pull back to identity)
```

### Exponential Map

```
exp: so(3) → SO(3)
[ω]_× ↦ I + sin(‖ω‖)/‖ω‖ [ω]_× + (1-cos(‖ω‖))/‖ω‖² [ω]_×²
```

### Geodesic Equation

**Right trivialization:**
```
R(t) = exp(t * log(R_0^T @ R_1)) @ R_0
```

**Left trivialization:**
```
R(t) = R_0 @ exp(t * log(R_0^T @ R_1))
```

Both are valid geodesics! They connect R_0 to R_1, but traverse the manifold differently.

---

## 🎯 Summary

**Right Trivialization Policy:**
- ✅ Body frame velocities (natural for robots)
- ✅ PRE-multiplication update rule
- ✅ Simpler velocity computation (no extra rotation)
- ✅ Good for manipulation tasks
- ✅ Same 7D → 10D conversion as SE3

**When to use:**
- Robot manipulation
- Body-relative control
- Consistency with robot kinematics
- Potentially better stability

**Files created:**
- `equi_diffpo/policy/maniflow/maniflow_right_triv_pointcloud_policy.py`
- `equi_diffpo/workspace/train_maniflow_right_triv_pointcloud_workspace.py`
- `equi_diffpo/config/train_maniflow_right_triv_pointcloud_workspace.yaml`

---

**You now have THREE manifold flow policies to choose from!** 🎉

Choose based on your specific needs:
1. **SE3** (left triv) - Spatial frame, standard
2. **RelRot** - No frame, simplest/fastest
3. **Right Triv** - Body frame, natural for robots ⭐

Happy experimenting! 🚀
