"""
验证脚本：测试SE(3) rigid policy的采样轨迹
给定R0和R1，查看插值轨迹和速度场

使用方法:
    python verify_rigid_se3_trajectory.py [test_case]
    
    参数:
        test_case: 可选，测试场景
            - "default": R0=单位矩阵, R1=绕z轴旋转90度 (默认)
            - "x_axis": R0=单位矩阵, R1=绕x轴旋转180度
            - "large_angle": R0=单位矩阵, R1=绕[1,1,1]轴旋转270度

输出:
    - 控制台输出: 详细的轨迹信息和验证结果
    - 可视化图表: 保存到 output/se3_trajectory_verification.png
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
# 获取当前脚本所在目录，然后向上找到项目根目录
current_dir = Path(__file__).resolve().parent  # equi_diffpo/policy/maniflow/
project_root = current_dir.parent.parent.parent  # 项目根目录
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入必要的工具函数
from equi_diffpo.common.se3_utils import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    log_SO3,
    bracket_so3,
    exp_so3,
    obtain_rigid_SE3_group,
    inv_SE3,
    log_SE3,
    bracket_se3,
    exp_se3,
)

# 不再需要导入policy类，我们直接实现所需的方法


def rotation_matrix_from_axis_angle(axis, angle):
    """
    从轴角表示创建旋转矩阵
    
    Args:
        axis: [3] 归一化的旋转轴
        angle: 旋转角度（弧度）
    
    Returns:
        R: [3, 3] 旋转矩阵
    """
    device = axis.device
    axis = axis / (torch.norm(axis) + 1e-8)
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=axis.dtype, device=device)
    
    R = torch.eye(3, dtype=axis.dtype, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


def rotation_to_axis_angle(R):
    """
    将旋转矩阵转换为轴角表示
    
    Args:
        R: [3, 3] 旋转矩阵
    
    Returns:
        axis: [3] 旋转轴（归一化）
        angle: 旋转角度（弧度）
    """
    trace = torch.trace(R)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))
    
    # 处理小角度情况
    device = R.device
    if angle < 1e-6:
        axis = torch.tensor([1., 0., 0.], dtype=R.dtype, device=device)
        return axis, angle
    
    # 计算旋转轴
    log_R = log_SO3(R.unsqueeze(0))
    w = bracket_so3(log_R).squeeze(0)
    axis = w / (torch.norm(w) + 1e-8)
    return axis, angle


def verify_se3_interpolation(test_case="default"):
    """
    验证SE(3)流形上的插值轨迹
    
    Args:
        test_case: 测试场景
            - "default": R0=单位矩阵, R1=绕z轴90度
            - "x_axis": R0=单位矩阵, R1=绕x轴180度
            - "large_angle": R0=单位矩阵, R1=绕任意轴大角度旋转
    """
    print("=" * 80)
    print("SE(3) Rigid Policy 轨迹验证")
    print("=" * 80)
    
    # 强制使用CPU
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"测试场景: {test_case}\n")
    
    # ========== 创建测试用的R0和R1 ==========
    print("创建测试旋转矩阵 R0 和 R1...")
    
    # R0: 单位矩阵（初始姿态）
    R0 = torch.eye(3, device=device)
    
    # 根据测试场景创建R1
    if test_case == "default":
        # R1: 绕z轴旋转90度
        axis = torch.tensor([0., 0., 1.], device=device)  # z轴
        angle = torch.tensor(np.pi / 2, device=device)  # 90度
        R1 = rotation_matrix_from_axis_angle(axis, angle)
        print("测试场景: R0=单位矩阵, R1=绕z轴旋转90度")
    elif test_case == "x_axis":
        # R1: 绕x轴旋转180度
        axis = torch.tensor([1., 0., 0.], device=device)  # x轴
        angle = torch.tensor(np.pi, device=device)  # 180度
        R1 = rotation_matrix_from_axis_angle(axis, angle)
        print("测试场景: R0=单位矩阵, R1=绕x轴旋转180度")
    elif test_case == "large_angle":
        # R1: 绕任意轴大角度旋转
        axis = torch.tensor([1., 1., 1.], device=device)  # 任意轴
        axis = axis / torch.norm(axis)  # 归一化
        angle = torch.tensor(np.pi * 1.5, device=device)  # 270度
        R1 = rotation_matrix_from_axis_angle(axis, angle)
        print("测试场景: R0=单位矩阵, R1=绕[1,1,1]轴旋转270度")
    else:
        raise ValueError(f"未知的测试场景: {test_case}")
    
    print(f"R0 (单位矩阵):\n{R0}\n")
    print(f"R1:\n{R1}\n")
    
    # 计算R0到R1的旋转角度
    R1_axis, R1_angle = rotation_to_axis_angle(R1)
    print(f"R1 的轴角表示: 轴={R1_axis.cpu().numpy()}, 角度={R1_angle.cpu().item()*180/np.pi:.2f}度\n")
    
    # ========== 转换为10D状态表示 ==========
    print("转换为10D状态表示...")
    
    # 位置（可以设置为0或任意值）
    pos_0 = torch.tensor([0., 0., 0.], device=device)
    pos_1 = torch.tensor([1., 0., 0.], device=device)  # 位置变化，测试线性轨迹
    
    # 旋转矩阵转换为6D表示
    rot_6d_0 = matrix_to_rotation_6d(R0.unsqueeze(0)).squeeze(0)  # [6]
    rot_6d_1 = matrix_to_rotation_6d(R1.unsqueeze(0)).squeeze(0)  # [6]
    
    # Gripper状态
    gripper_0 = torch.tensor([0.], device=device)
    gripper_1 = torch.tensor([1.], device=device)  # 从关闭到打开
    
    # 组合成10D状态: [pos(3), rot_6d(6), gripper(1)]
    state_0 = torch.cat([pos_0, rot_6d_0, gripper_0], dim=0)  # [10]
    state_1 = torch.cat([pos_1, rot_6d_1, gripper_1], dim=0)  # [10]
    
    print(f"State 0 (10D): pos={state_0[:3].cpu().numpy()}, rot_6d={state_0[3:9].cpu().numpy()}, gripper={state_0[9].cpu().item()}")
    print(f"State 1 (10D): pos={state_1[:3].cpu().numpy()}, rot_6d={state_1[3:9].cpu().numpy()}, gripper={state_1[9].cpu().item()}\n")
    
    # ========== 定义ground truth速度场计算函数 ==========
    def get_se3_velocity_gt(noise, target):
        """
        计算SE(3)流形上的ground truth速度（不使用神经网络）
        
        Args:
            noise: [B, H, 10] 初始状态 (pos 3, rot_6d 6, gripper 1)
            target: [B, H, 10] 目标状态
        
        Returns:
            velocity: [B, H, 7] 速度 [v(3), omega(3), gripper_vel(1)]
        """
        batch_size, horizon, _ = noise.shape
        
        # Flatten
        noise_flat = noise.reshape(-1, 10)
        target_flat = target.reshape(-1, 10)
        
        # 计算SE(3)速度
        rot_6d_0 = noise_flat[:, 3:9]
        rot_6d_1 = target_flat[:, 3:9]
        
        R_0 = rotation_6d_to_matrix(rot_6d_0)
        R_1 = rotation_6d_to_matrix(rot_6d_1)
        Rt_0 = obtain_rigid_SE3_group(noise_flat[:, :3], R_0)
        Rt_1 = obtain_rigid_SE3_group(target_flat[:, :3], R_1)
        Rt_delta = inv_SE3(Rt_0) @ Rt_1
        log_Rt_delta = log_SE3(Rt_delta)
        vel_se3 = bracket_se3(log_Rt_delta)  # [B*H, 6] - [v(3), omega(3)]
        
        # Gripper velocity
        vel_gripper = target_flat[:, 9:10] - noise_flat[:, 9:10]
        
        # Combine [v(3), omega(3), gripper_vel(1)] = 7D
        velocity_flat = torch.cat([vel_se3, vel_gripper], dim=-1)
        velocity = velocity_flat.reshape(batch_size, horizon, 7)
        
        return velocity
    
    # ========== 定义ODE积分采样函数（使用ground truth速度场） ==========
    def sample_ode_ground_truth(x0, target, N=500):
        """
        使用ground truth速度场进行ODE积分采样
        
        Args:
            x0: [B, H, 10] 初始状态
            target: [B, H, 10] 目标状态
            N: 积分步数
        
        Returns:
            traj: 轨迹列表，每个元素是 [B, H, 10]
        """
        dt = 1. / N
        traj = []
        x = x0.detach().clone()
        batch_size, horizon, _ = x.shape
        traj.append(x.detach().clone())
        
        for i in range(N):
            # 计算从当前位置到目标的ground truth速度
            v_gt = get_se3_velocity_gt(x0, target)  # [B, H, 7]
            
            # 更新状态
            x_flat = x.reshape(-1, 10)  # [B*H, 10]
            v_flat = v_gt.reshape(-1, 7)  # [B*H, 7]
            
            # 更新旋转和位置（使用SE(3)指数映射）
            rot_6d_current = x_flat[:, 3:9]
            R_current = rotation_6d_to_matrix(rot_6d_current)
            Rt_current = obtain_rigid_SE3_group(x_flat[:, :3], R_current)
            
            # 速度场的前6维是SE(3)速度 [v(3), omega(3)]
            w_vel = v_flat[:, :6]
            
            # 使用指数映射更新
            dR = exp_se3(dt * w_vel)
            Rt_new = Rt_current @ dR
            
            # 正交化防止数值漂移
            R_new = Rt_new[..., :3, :3]
            # U, _, Vt = torch.linalg.svd(R_new)
            # det_sign = torch.det(U @ Vt).unsqueeze(-1).unsqueeze(-1)
            # S = torch.eye(3, device=R_new.device, dtype=R_new.dtype).unsqueeze(0).expand(R_new.shape[0], -1, -1).clone()
            # S[:, 2, 2] = det_sign.squeeze(-1).squeeze(-1)
            # R_new = U @ S @ Vt
            pos_new = Rt_new[..., :3, 3]
            
            # 转换回6D表示
            rot_6d_new = matrix_to_rotation_6d(R_new)
            
            # 更新gripper
            gripper_new = x_flat[:, 9:10] + dt * v_flat[:, 6:7]
            
            # 组合 [pos(3), rot_6d(6), gripper(1)] = 10D
            x_new_flat = torch.cat([pos_new, rot_6d_new, gripper_new], dim=-1)
            x = x_new_flat.reshape(batch_size, horizon, 10)
            
            traj.append(x.detach().clone())
        
        return traj
    
    # ========== 使用ground truth速度场进行ODE积分采样 ==========
    print("\n" + "=" * 80)
    print("使用Ground Truth速度场进行ODE积分采样")
    print("=" * 80)
    
    # 扩展为 [B, H, 10] 格式
    batch_size = 1
    horizon = 1
    x0 = state_0.unsqueeze(0).unsqueeze(0).expand(batch_size, horizon, -1)  # [1, 1, 10]
    target = state_1.unsqueeze(0).unsqueeze(0).expand(batch_size, horizon, -1)  # [1, 1, 10]
    
    # ODE积分步数
    num_steps = 50
    print(f"ODE积分步数: {num_steps}\n")
    
    # 使用ground truth速度场进行采样
    traj = sample_ode_ground_truth(x0, target, N=num_steps)
    
    # 提取轨迹信息
    trajectory = []
    rotation_angles = []
    rotation_axes = []
    positions = []
    gripper_states = []
    
    for i, state_t in enumerate(traj):
        state_flat = state_t.squeeze(0).squeeze(0)  # [10]
        
        # 提取位置、旋转、gripper
        pos_t = state_flat[:3]
        rot_6d_t = state_flat[3:9]
        gripper_t = state_flat[9]
        
        # 转换旋转矩阵并计算角度
        R_t = rotation_6d_to_matrix(rot_6d_t.unsqueeze(0)).squeeze(0)  # [3, 3]
        axis_t, angle_t = rotation_to_axis_angle(R_t)
        
        trajectory.append(state_flat.cpu().numpy())
        rotation_angles.append(angle_t.cpu().item())
        rotation_axes.append(axis_t.cpu().numpy())
        positions.append(pos_t.cpu().numpy())
        gripper_states.append(gripper_t.cpu().item())
        
        t = i / num_steps
        if i == 0 or i == num_steps // 2 or i == num_steps:
            print(f"t={t:.3f}: 角度={angle_t.item()*180/np.pi:.2f}度, "
                  f"轴={axis_t.cpu().numpy()}, gripper={gripper_t.item():.3f}")
    
    trajectory = np.array(trajectory)
    rotation_angles = np.array(rotation_angles)
    rotation_axes = np.array(rotation_axes)
    positions = np.array(positions)
    gripper_states = np.array(gripper_states)
    
    # ========== 测试初始速度场 ==========
    print("\n" + "=" * 80)
    print("初始Ground Truth速度场")
    print("=" * 80)
    
    velocity = get_se3_velocity_gt(x0, target)  # [1, 1, 7]
    vel_flat = velocity.squeeze(0).squeeze(0)  # [7]
    
    print(f"速度 (7D):")
    print(f"  线性速度 v={vel_flat[:3].cpu().numpy()}")
    print(f"  角速度 omega={vel_flat[3:6].cpu().numpy()}")
    print(f"  Gripper速度={vel_flat[6].cpu().item()}")
    
    # 计算速度的模长
    v_magnitude = torch.norm(vel_flat[:3]).cpu().item()
    omega_magnitude = torch.norm(vel_flat[3:6]).cpu().item()
    print(f"\n速度模长:")
    print(f"  线性速度模长: {v_magnitude:.4f}")
    print(f"  角速度模长: {omega_magnitude:.4f} rad/s")
    print(f"  角速度模长 (度/秒): {omega_magnitude * 180 / np.pi:.2f} deg/s")
    
    # ========== 可视化 ==========
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    t_array = np.linspace(0, 1, len(trajectory))
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SE(3) Rigid Policy 轨迹验证', fontsize=16, fontweight='bold')
    
    # 1. 旋转角度随时间变化
    ax1 = axes[0, 0]
    ax1.plot(t_array, rotation_angles * 180 / np.pi, 'b-', linewidth=2, label='旋转角度')
    ax1.axhline(y=R1_angle.item() * 180 / np.pi, color='r', linestyle='--', label='目标角度')
    ax1.set_xlabel('时间 t')
    ax1.set_ylabel('旋转角度 (度)')
    ax1.set_title('旋转角度随时间变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 旋转轴分量随时间变化
    ax2 = axes[0, 1]
    ax2.plot(t_array, rotation_axes[:, 0], 'r-', label='x轴分量', linewidth=2)
    ax2.plot(t_array, rotation_axes[:, 1], 'g-', label='y轴分量', linewidth=2)
    ax2.plot(t_array, rotation_axes[:, 2], 'b-', label='z轴分量', linewidth=2)
    ax2.set_xlabel('时间 t')
    ax2.set_ylabel('旋转轴分量')
    ax2.set_title('旋转轴分量随时间变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Gripper状态随时间变化
    ax3 = axes[1, 0]
    ax3.plot(t_array, gripper_states, 'g-', linewidth=2, label='Gripper状态')
    ax3.set_xlabel('时间 t')
    ax3.set_ylabel('Gripper值')
    ax3.set_title('Gripper状态随时间变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 旋转6D表示的分量
    ax4 = axes[1, 1]
    for i in range(6):
        ax4.plot(t_array, trajectory[:, 3+i], label=f'rot_6d[{i}]', linewidth=1.5)
    ax4.set_xlabel('时间 t')
    ax4.set_ylabel('6D旋转分量')
    ax4.set_title('6D旋转表示的分量')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "se3_trajectory_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    # ========== 验证性质 ==========
    print("\n" + "=" * 80)
    print("验证轨迹性质")
    print("=" * 80)
    
    # 验证起始和结束状态
    state_start = trajectory[0]
    state_end = trajectory[-1]
    
    print(f"起始状态 (t=0):")
    print(f"  位置: {state_start[:3]}")
    print(f"  旋转6D: {state_start[3:9]}")
    print(f"  Gripper: {state_start[9]:.3f}")
    
    print(f"\n结束状态 (t=1):")
    print(f"  位置: {state_end[:3]}")
    print(f"  旋转6D: {state_end[3:9]}")
    print(f"  Gripper: {state_end[9]:.3f}")
    
    # 验证旋转矩阵的正交性
    print(f"\n验证旋转矩阵的正交性...")
    R_end = rotation_6d_to_matrix(torch.tensor(state_end[3:9]).unsqueeze(0).to(device)).squeeze(0)
    R_end_orthogonal = torch.allclose(R_end @ R_end.T, torch.eye(3).to(device), atol=1e-4)
    R_end_det = torch.det(R_end).cpu().item()
    print(f"  R(t=1) 是否正交: {R_end_orthogonal}")
    print(f"  det(R(t=1)): {R_end_det:.6f} (应该接近1.0)")
    
    # ========== 旋转矩阵对比 ==========
    print(f"\n" + "=" * 80)
    print("旋转矩阵对比 (6D向量 -> 旋转矩阵)")
    print("=" * 80)
    
    # 起始和结束的旋转矩阵
    R_start = rotation_6d_to_matrix(torch.tensor(state_start[3:9]).unsqueeze(0).to(device)).squeeze(0)
    
    print(f"\n起始旋转矩阵 R(t=0):")
    print(R_start.cpu().numpy())
    
    print(f"\n结束旋转矩阵 R(t=1) [从6D转换]:")
    print(R_end.cpu().numpy())
    
    print(f"\n目标旋转矩阵 R1 [ground truth]:")
    print(R1.cpu().numpy())
    
    # 计算旋转矩阵误差
    R_diff = R_end @ R1.T  # R_end相对于R1的旋转
    R_diff_angle = rotation_to_axis_angle(R_diff)[1]
    
    # Frobenius norm误差
    R_frobenius_error = torch.norm(R_end - R1, p='fro').cpu().item()
    
    # 角度误差
    R_angle_error_deg = R_diff_angle.cpu().item() * 180 / np.pi
    
    print(f"\n旋转矩阵误差分析:")
    print(f"  Frobenius范数误差: {R_frobenius_error:.6f}")
    print(f"  角度误差: {R_angle_error_deg:.6f} 度")
    print(f"  (R_end相对于R1的旋转角度)")
    
    # 验证是否到达目标
    target_state = state_1.cpu().numpy()
    pos_error = np.linalg.norm(state_end[:3] - target_state[:3])
    rot_6d_error = np.linalg.norm(state_end[3:9] - target_state[3:9])
    gripper_error = abs(state_end[9] - target_state[9])
    
    print(f"\n其他误差分析:")
    print(f"  位置误差: {pos_error:.6f}")
    print(f"  旋转6D误差: {rot_6d_error:.6f}")
    print(f"  Gripper误差: {gripper_error:.6f}")
    
    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # 可以选择测试场景
    if len(sys.argv) > 1:
        test_case = sys.argv[1]
    else:
        test_case = "default"
    
    verify_se3_interpolation(test_case=test_case)
    
    print("\n提示: 可以运行以下命令测试不同场景:")
    print("  python verify_rigid_se3_trajectory.py default      # 绕z轴90度")
    print("  python verify_rigid_se3_trajectory.py x_axis      # 绕x轴180度")
    print("  python verify_rigid_se3_trajectory.py large_angle # 大角度旋转")

