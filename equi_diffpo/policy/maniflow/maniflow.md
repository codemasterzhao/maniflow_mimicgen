
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
