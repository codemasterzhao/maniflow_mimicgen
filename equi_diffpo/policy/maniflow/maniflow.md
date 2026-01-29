
## juyi

policy/maniflow/maniflow_pointcloud_policy 是 baseline, se3_pointcloud_policy是改了以后的.

Baseline: 状态和速度都是10D，直接在欧几里得空间操作, 高斯噪声采样. 所有维度线性插值

SE3: 状态是10D (6D旋转表示)，速度是7D (角速度表示)，在SE(3)流形上操作,旋转部分用SO(3)流形上的均匀采样, 旋转用SO(3)测地线，位置和夹爪用线性插值


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
- **SE3**: 位置线性 + 旋转SO(3)测地线 + 夹爪线性

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
| 旋转 | 同上 (线性) | `R_t = R_0 @ exp(t * log(R_0^T @ R_1))` (测地线) |
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
