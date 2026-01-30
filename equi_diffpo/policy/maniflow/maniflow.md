
## juyi

policy/maniflow/maniflow_pointcloud_policy 是 baseline, se3_pointcloud_policy是改了以后的.

Baseline: 状态和速度都是10D，直接在欧几里得空间操作, 高斯噪声采样. 所有维度线性插值

SE3: 状态是10D (6D旋转表示)，速度是7D (角速度表示)，在SE(3)流形上操作,旋转部分用SO(3)流形上的均匀采样，位置和夹爪用线性插值 旋转不是测地线,旋转是lie group插值


Lie group插值和测地线的区别在于：两者公式 R_t = R_0 @ exp(t * log(R_0^T @ R_1)) 实际上是一样的，但语义不同——测地线强调的是流形上最短路径，而Lie group插值强调的是通过指数映射在李群上插值的代数结构。你的实现用的是右不变(right trivialization)的李群框架，所以用"Lie group插值"描述更准确。


# SE3 Policy vs Baseline Policy

## 核心区别

### 动作表示
- **Baseline**: 10D状态 + 10D速度
- **SE3**: 10D状态 `[pos(3), rot_6d(6), gripper(1)]` + 7D速度 `[pos_vel(3), omega(3), gripper_vel(1)]`

### 噪声采样
- **Baseline**: 全部高斯噪声
- **SE3**: 位置高斯 + 旋转SO(3)均匀采样 + 夹爪高斯

### 插值方法
- **Baseline**: 全部线性插值
- **SE3**: 位置线性 + 旋转SO(3) Lie group插值 + 夹爪线性

### 速度计算
- **Baseline**: 简单差分 `v = x1 - x0`
- **SE3**: 位置差分 + 角速度 `omega = log(R0^T @ R1)` + 夹爪差分

### ODE更新
- **Baseline**: `x = x + v * dt`
- **SE3**: 位置 `pos += v * dt` + 旋转 `R = R @ exp(dt * omega)` + 夹爪 `g += v * dt`

### 归一化
- **Baseline**: 全部维度归一化
- **SE3**: 只归一化位置和夹爪，旋转保持6D原始表示

### 训练损失
- **Baseline**: Flow + Consistency双损失
- **SE3**: 只用Flow，分别计算位置、旋转、夹爪损失

### 理论基础
- **Baseline**: 欧几里得空间 (R^10)
- **SE3**: SE(3)流形 (R^3 × SO(3) × R)

## 优势
SE3在流形上操作，旋转插值更自然，避免欧几里得空间中的不连续性问题

---

## 代码修改位置详解

### 1. 类初始化 `__init__`

| 位置 | Baseline | SE3 | 说明 |
|------|----------|-----|------|
| 动作维度 | `action_dim = action_shape[0]` | `action_dim_internal = 10` (状态) + `action_dim_velocity = 7` (速度) | 状态和速度解耦 |
| 模型输出维度 | `output_dim=action_dim` (10D) | `output_dim=self.action_dim_velocity` (7D) | 模型只预测7D速度 |
| 训练参数 | `flow_batch_ratio`, `consistency_batch_ratio` | 移除consistency相关参数 | 只用flow loss |
| 新增参数 | - | `se3_noise_std_pos`, `se3_noise_std_rot` | SO(3)采样控制 |

### 2. 噪声采样

| 方法 | Baseline | SE3 |
|------|----------|-----|
| 位置 | `torch.randn_like(actions)` | `torch.randn(B, H, 3) * se3_noise_std_pos` |
| 旋转 | 同上 (高斯) | `sample_uniform_SO3()` → `matrix_to_rotation_6d()` |
| 函数 | 无专门函数 | **新增** `sample_se3_noise()` (L194-222) |

### 3. 插值方法

| 方法 | Baseline | SE3 |
|------|----------|-----|
| 实现 | `linear_interpolate()` (L290-317) | **新增** `se3_interpolate()` (L224-268) |
| 位置 | `(1-t)*x0 + t*x1` | 同baseline |
| 旋转 | 同上 (线性) | `R_t = R_0 @ exp(t * log(R_0^T @ R_1))` (Lie group) |
| 夹爪 | 同上 | 同baseline |

### 4. 速度计算

| 方法 | Baseline | SE3 |
|------|----------|-----|
| 实现 | 直接差分 `v = x1 - x0` | **新增** `get_se3_velocity()` (L270-301) |
| 位置 | `target - noise` | 同baseline |
| 旋转 | 同上 (欧氏差分) | `omega = log(R_0^T @ R_1)` (右不变) |
| 输出 | 10D velocity | 7D `[pos_vel(3), omega(3), gripper_vel(1)]` |

### 5. ODE采样 `sample_ode()`

| 位置 | Baseline (L385-403) | SE3 (L405-468) | 说明 |
|------|---------------------|----------------|------|
| 状态更新 | `x = x + pred * dt` | 分别更新pos/rot/gripper | 流形更新 |
| 位置 | 同上 | `pos_new = pos + dt * v_pos` | 线性 |
| 旋转 | 同上 | `R_new = R @ exp(dt * omega)` | 指数映射 |
| 正交化 | - | **新增** SVD正交化 (L449-454) | 数值稳定 |

### 6. 归一化 `set_normalizer()`

| Baseline (L226-227) | SE3 (L382-423) |
|---------------------|----------------|
| 直接加载全部normalizer | 修改action normalizer |
| 所有维度归一化 | 位置+夹爪归一化，旋转保持identity |
| - | `scale[3:9]=1.0, offset[3:9]=0.0` |

### 7. 训练目标 `get_flow_velocity()`

| Baseline (L319-357) | SE3 (L426-466) |
|---------------------|----------------|
| 调用 `linear_interpolate()` | 调用 `se3_interpolate()` |
| 速度 = `x1 - x0` | 调用 `get_se3_velocity()` |
| - | **移除** `get_consistency_velocity()` |

### 8. 损失计算 `compute_loss()`

| Baseline (L405-481) | SE3 (L470-521) |
|---------------------|----------------|
| flow_loss + consistency_loss | 只用 flow_loss |
| 整体MSE | **分开** `loss_pos + loss_rot + loss_gripper` |
| 需要 `ema_model` 参数 | 不需要 |
| 旋转速度归一化 | `v_target[..., 3:6] /= 3.14` |

### 9. 新增依赖

```python
# se3_utils.py 中的函数
from equi_diffpo.common.se3_utils import (
    sample_uniform_SO3,   # SO(3)均匀采样
    exp_so3,              # so(3) → SO(3) 指数映射
    log_SO3,              # SO(3) → so(3) 对数映射
    inv_SO3,              # SO(3)逆
    bracket_so3,          # 反对称矩阵 → 向量
    rotation_6d_to_matrix,
    matrix_to_rotation_6d
)
```

---

## 修改策略总结

1. **状态表示不变** (10D)，**速度预测改为7D**
2. **旋转部分在SO(3)流形操作**：采样、插值、更新都用指数/对数映射
3. **归一化只作用于位置和夹爪**，6D旋转保持原始
4. **移除consistency training**，简化为纯flow matching
5. **损失分开计算**，便于调试
6. **推理时正交化旋转矩阵**，保证数值稳定



---

# OpenPi SE(3) 改造方案详解

## OpenPi vs SE3 Policy 架构对比

| 特性 | OpenPi (当前) | SE3 Policy (目标) |
|------|--------------|------------------|
| **几何空间** | R^32 欧几里得 (padded) | SE(3) = R^3 × SO(3) × R |
| **状态维度** | 32D (action_dim) | 10D: [pos(3), rot_6d(6), gripper(1)] |
| **速度维度** | 32D (同状态) | 7D: [pos_vel(3), omega(3), gripper_vel(1)] |
| **模型输出** | v_t (32D) | v_t (7D) |
| **环境动作** | 7D (action_env_dim) | 10D → 7D 转换 |
| **噪声方法** | flow_sde / flow_noise / flow_cps | SE(3) 流形采样 |

---

## 核心文件修改

### 文件路径: `rlinf/models/embodiment/openpi/openpi_action_model.py`

---

## 1. 配置类 `OpenPi0Config` (L34-66)

```python
# 当前
noise_method: str = "flow_sde"
action_env_dim: int = 7

# SE3修改 - 新增配置项
action_dim_internal: int = 10  # 10D: [pos(3), rot_6d(6), gripper(1)]
action_dim_velocity: int = 7   # 7D: [pos_vel(3), omega(3), gripper_vel(1)]
noise_method: str = "flow_se3" # 新增方法
se3_noise_std_pos: float = 1.0
se3_noise_std_rot: float = 1.0
```

---

## 2. 新增 SE3 工具导入

```python
# 在文件头部添加
from rlinf.models.embodiment.openpi.se3_utils import (
    sample_uniform_SO3, exp_so3, log_SO3, inv_SO3, bracket_so3,
    rotation_6d_to_matrix, matrix_to_rotation_6d
)
```

---

## 3. 噪声采样 (L390-391) → `sample_se3_noise()`

```python
# 当前
noise = self.sample_noise(actions_shape, device)  # 全高斯 32D

# SE3 新增方法
def sample_se3_noise(self, batch_size, horizon, device):
    """Sample noise on SE(3) manifold."""
    # 位置: 高斯
    pos_noise = torch.randn(batch_size, horizon, 3, device=device) * self.config.se3_noise_std_pos
    
    # 旋转: SO(3) 均匀采样 → 6D
    R_noise = sample_uniform_SO3(batch_size * horizon, device)
    rot_noise_6d = matrix_to_rotation_6d(R_noise).reshape(batch_size, horizon, 6)
    
    # 夹爪: 高斯
    gripper_noise = torch.randn(batch_size, horizon, 1, device=device)
    
    # Padding 到 32D
    noise_10d = torch.cat([pos_noise, rot_noise_6d, gripper_noise], dim=-1)
    noise_32d = F.pad(noise_10d, (0, 22), value=0)  # padding
    return noise_32d
```

---

## 4. ODE更新 `sample_mean_var_val()` (L500-603)

```python
# 当前 (L570-575) - 欧氏空间线性更新
x0_pred = x_t - v_t * t_input
x1_pred = x_t + v_t * (1 - t_input)
x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
x_t_next = x_t_mean + noise * x_t_std

# SE3修改 - 分别处理 pos/rot/gripper
# 位置 (线性)
pos_mean = x_t[:, :, :3] + delta * v_t[:, :, :3]

# 旋转 (SO(3) 指数映射)
rot_6d = x_t[:, :, 3:9].reshape(-1, 6)
R_current = rotation_6d_to_matrix(rot_6d)
omega = v_t[:, :, 3:6].reshape(-1, 3)
dR = exp_so3(delta * omega)
R_new = R_current @ dR  # RIGHT trivialization
# SVD 正交化
U, _, Vt = torch.linalg.svd(R_new)
det_sign = torch.det(U @ Vt).unsqueeze(-1).unsqueeze(-1)
S = torch.eye(3, device=R_new.device).unsqueeze(0).expand_as(R_new).clone()
S[:, 2, 2] = det_sign.squeeze()
R_new = U @ S @ Vt
rot_6d_new = matrix_to_rotation_6d(R_new).reshape(batch_size, horizon, 6)

# 夹爪 (线性)
gripper_mean = x_t[:, :, 9:10] + delta * v_t[:, :, 6:7]

# 组合
x_t_mean = torch.cat([pos_mean, rot_6d_new, gripper_mean], dim=-1)
```

---

## 5. SE3 插值方法 (训练用)

```python
def se3_interpolate(self, noise, target, t):
    """Geodesic interpolation on SE(3) manifold."""
    batch_size, horizon, _ = noise.shape
    
    # 位置: 线性插值
    pos_t = noise[:, :, :3] + t * (target[:, :, :3] - noise[:, :, :3])
    
    # 旋转: SO(3) Lie group插值
    rot_6d_0 = noise[:, :, 3:9].reshape(-1, 6)
    rot_6d_1 = target[:, :, 3:9].reshape(-1, 6)
    R_0 = rotation_6d_to_matrix(rot_6d_0)
    R_1 = rotation_6d_to_matrix(rot_6d_1)
    R_delta = inv_SO3(R_0) @ R_1
    log_R_delta = log_SO3(R_delta)
    omega = bracket_so3(log_R_delta)  # body frame angular velocity
    t_flat = t.expand(-1, horizon, 1).reshape(-1, 1)
    R_t = R_0 @ exp_so3(t_flat * omega)
    rot_6d_t = matrix_to_rotation_6d(R_t).reshape(batch_size, horizon, 6)
    
    # 夹爪: 线性插值
    gripper_t = noise[:, :, 9:10] + t * (target[:, :, 9:10] - noise[:, :, 9:10])
    
    return torch.cat([pos_t, rot_6d_t, gripper_t], dim=-1)
```

---

## 6. SE3 速度计算 (训练目标)

```python
def get_se3_velocity(self, noise, target):
    """Compute velocity in SE(3) tangent space (RIGHT trivialization)."""
    batch_size, horizon, _ = noise.shape
    
    # 位置速度: 差分
    vel_pos = target[:, :, :3] - noise[:, :, :3]
    
    # 角速度: body frame
    # omega_body = log(R_0^T @ R_1)
    R_0 = rotation_6d_to_matrix(noise[:, :, 3:9].reshape(-1, 6))
    R_1 = rotation_6d_to_matrix(target[:, :, 3:9].reshape(-1, 6))
    R_delta = inv_SO3(R_0) @ R_1
    log_R_delta = log_SO3(R_delta)
    vel_ang = bracket_so3(log_R_delta).reshape(batch_size, horizon, 3)
    
    # 夹爪速度: 差分
    vel_gripper = target[:, :, 9:10] - noise[:, :, 9:10]
    
    return torch.cat([vel_pos, vel_ang, vel_gripper], dim=-1)  # 7D
```

---

## 7. Log Probability 修改 (L672-749)

```python
# 当前: 32D 统一计算
log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)

# SE3修改: 分开计算
log_prob_pos = self.get_logprob_norm(chains_next[..., :3], x_t_mean[..., :3], x_t_std[..., :3])
log_prob_rot = self.get_logprob_norm(chains_next[..., 3:9], x_t_mean[..., 3:9], x_t_std[..., 3:9])
log_prob_gripper = self.get_logprob_norm(chains_next[..., 9:10], x_t_mean[..., 9:10], x_t_std[..., 9:10])
log_probs = log_prob_pos + log_prob_rot + log_prob_gripper
```

---

## 8. 动作输出转换 (10D → 7D)

```python
def action_10d_to_7d(action_10d):
    """Convert 10D internal action to 7D environment action."""
    pos = action_10d[..., :3]
    rot_6d = action_10d[..., 3:9]
    gripper = action_10d[..., 9:10]
    
    # 6D → axis-angle
    R = rotation_6d_to_matrix(rot_6d.reshape(-1, 6))
    log_R = log_SO3(R)
    axis_angle = bracket_so3(log_R)
    
    return torch.cat([pos.reshape(-1, 3), axis_angle, gripper.reshape(-1, 1)], dim=-1)
```

---

## 改造策略对比

### 方案A: 最小修改 (推荐)

| 修改项 | 说明 |
|--------|------|
| `sample_se3_noise()` | 新增，采样10D后padding到32D |
| `sample_mean_var_val()` | 只修改前10D的更新逻辑 |
| 模型输出 | 保持32D，只使用前7D作为速度 |

**优点**: 改动小，兼容现有代码  
**缺点**: 多余计算

---

## 新增依赖文件

将 `maniflow_mimicgen/equi_diffpo/common/se3_utils.py` 复制到:
```
rlinf/models/embodiment/openpi/se3_utils.py
```

核心函数:
- `sample_uniform_SO3()` - SO(3) 均匀采样
- `exp_so3()` / `log_SO3()` - 指数/对数映射
- `rotation_6d_to_matrix()` / `matrix_to_rotation_6d()` - 6D ↔ 矩阵

---

## 修改优先级

| 优先级 | 修改项 |
|--------|--------|
| **高** | 噪声采样 `sample_se3_noise()` |
| **高** | ODE更新 `sample_mean_var_val()` 旋转部分 |
| **中** | 插值 `se3_interpolate()` |
| **中** | 速度计算 `get_se3_velocity()` |
| **低** | Log probability / Entropy 计算 |

---

## 潜在问题

1. **VLM 兼容性** - 需确认32D是硬编码还是可配置 ->  32D 是 需要 padding 的. 
2. **归一化** - 6D旋转需要identity归一化
3. **数值稳定性** - SO(3) 映射在奇点需特殊处理
4. **训练稳定性** - 可能需调整学习率