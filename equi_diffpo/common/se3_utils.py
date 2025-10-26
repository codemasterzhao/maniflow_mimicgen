"""
SE(3) manifold operations for flow matching.
Adapted from EquiGraspFlow for manifold-based flow matching in SE(3).
"""

import torch
import numpy as np

EPS = 1e-4


def is_SO3(R):
    """Check if R is a valid SO(3) rotation matrix."""
    test_0 = torch.allclose(R @ R.transpose(1, 2), torch.eye(3).repeat(len(R), 1, 1).to(R), atol=EPS)
    test_1 = torch.allclose(R.transpose(1, 2) @ R, torch.eye(3).repeat(len(R), 1, 1).to(R), atol=EPS)
    return test_0 and test_1


def inv_SO3(R):
    """Compute inverse of SO(3) rotation matrix."""
    assert R.shape[1:] == (3, 3), f"inv_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"
    return R.transpose(1, 2)


def bracket_so3(w):
    """
    Bracket operation for so(3) Lie algebra.
    Converts between vector and matrix representations.
    """
    # vector -> matrix
    if w.shape[1:] == (3,):
        zeros = w.new_zeros(len(w))
        out = torch.stack([
            torch.stack([zeros, -w[:, 2], w[:, 1]], dim=1),
            torch.stack([w[:, 2], zeros, -w[:, 0]], dim=1),
            torch.stack([-w[:, 1], w[:, 0], zeros], dim=1)
        ], dim=1)
    # matrix -> vector
    elif w.shape[1:] == (3, 3):
        out = torch.stack([w[:, 2, 1], w[:, 0, 2], w[:, 1, 0]], dim=1)
    else:
        raise ValueError(f"bracket_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w.shape)}")
    return out


def log_SO3(R):
    """
    Logarithm map from SO(3) to so(3).
    Returns the matrix representation of the Lie algebra element.
    """
    n = R.shape[0]
    assert R.shape == (n, 3, 3), f"log_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"

    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    w_mat = torch.zeros_like(R)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS))

    is_regular = (tr_R + 1 > EPS)
    is_singular = (tr_R + 1 <= EPS)

    theta = theta.unsqueeze(1).unsqueeze(2)

    if is_regular.any():
        w_mat_regular = (1 / (2 * torch.sin(theta[is_regular]) + EPS)) * (R[is_regular] - R[is_regular].transpose(1, 2)) * theta[is_regular]
        w_mat[is_regular] = w_mat_regular

    if is_singular.any():
        w_mat_singular = (R[is_singular] - torch.eye(3).to(R)) / 2
        w_vec_singular_raw = torch.sqrt(torch.diagonal(w_mat_singular, dim1=1, dim2=2) + 1)
        # Use torch.where to avoid inplace operation
        w_vec_singular = torch.where(torch.isnan(w_vec_singular_raw),
                                     torch.zeros_like(w_vec_singular_raw),
                                     w_vec_singular_raw)

        w_1 = w_vec_singular[:, 0]
        w_2 = w_vec_singular[:, 1] * (torch.sign(w_mat_singular[:, 0, 1]) + (w_1 == 0))
        w_3 = w_vec_singular[:, 2] * torch.sign(4 * torch.sign(w_mat_singular[:, 0, 2]) + 2 * (w_1 == 0) * torch.sign(w_mat_singular[:, 1, 2]) + 1 * (w_1 == 0) * (w_2 == 0))

        w_vec_singular = torch.stack([w_1, w_2, w_3], dim=1)
        w_mat[is_singular] = bracket_so3(w_vec_singular) * torch.pi

    return w_mat


def exp_so3(w_vec):
    """
    Exponential map from so(3) to SO(3).
    Input can be either vector (N, 3) or matrix (N, 3, 3) representation.
    """
    if w_vec.shape[1:] == (3, 3):
        w_vec = bracket_so3(w_vec)
    elif w_vec.shape[1:] != (3,):
        raise ValueError(f"exp_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w_vec.shape)}")

    # Create identity matrices (not connected to w_vec's grad graph)
    R = torch.eye(3, device=w_vec.device, dtype=w_vec.dtype).unsqueeze(0).repeat(len(w_vec), 1, 1)
    theta = w_vec.norm(dim=1)

    is_regular = theta > EPS

    if is_regular.any():
        w_vec_regular = w_vec[is_regular]
        theta_regular = theta[is_regular].unsqueeze(1)
        w_mat_hat_regular = bracket_so3(w_vec_regular / theta_regular)
        theta_regular = theta_regular.unsqueeze(2)
        R_regular = torch.eye(3, device=w_vec.device, dtype=w_vec.dtype).unsqueeze(0).repeat(len(w_vec_regular), 1, 1) + \
                    torch.sin(theta_regular) * w_mat_hat_regular + \
                    (1 - torch.cos(theta_regular)) * w_mat_hat_regular @ w_mat_hat_regular

        # Avoid inplace: create new tensor
        R = R.clone()
        R[is_regular] = R_regular

    # Orthogonalize to ensure R is a valid rotation matrix (differentiable)
    # Use SVD: R = U @ S @ V^T, nearest rotation is U @ V^T
    U, _, Vt = torch.linalg.svd(R)
    # Ensure proper rotation (det = +1), not reflection (det = -1)
    det_sign = torch.det(U @ Vt).unsqueeze(-1).unsqueeze(-1)
    # Create diagonal correction matrix
    S = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], -1, -1).clone()
    S[:, 2, 2] = det_sign.squeeze(-1).squeeze(-1)
    R = U @ S @ Vt

    return R


def sample_uniform_SO3(num_samples, device):
    """
    Sample random rotations uniformly from SO(3) using quaternion method.
    """
    # Sample uniform quaternions
    u1, u2, u3 = torch.rand(3, num_samples, device=device)

    q0 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
    q1 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
    q2 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
    q3 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

    # Convert quaternion to rotation matrix
    R = torch.zeros(num_samples, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (q2**2 + q3**2)
    R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)
    R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1 - 2 * (q1**2 + q3**2)
    R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return R


def get_se3_trajectory(x_0, x_1, t):
    """
    Compute SE(3) geodesic trajectory and velocity.

    Args:
        x_0: Initial SE(3) state [B, 7] (3 pos + 4 quat or 3 pos + 3 rot_vec)
        x_1: Target SE(3) state [B, 7]
        t: Time parameter [B, 1]

    Returns:
        x_t: Interpolated state [B, 7]
        u_t: Velocity in se(3) algebra [B, 6] (3 angular + 3 linear)
    """
    # Extract positions and rotations
    # Assuming x is [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper]
    p_0 = x_0[:, :3]
    p_1 = x_1[:, :3]

    # For rotation, we'll use axis-angle representation (3D)
    w_0 = x_0[:, 3:6]
    w_1 = x_1[:, 3:6]

    # Convert to rotation matrices
    R_0 = exp_so3(w_0)
    R_1 = exp_so3(w_1)

    # Geodesic interpolation on SO(3)
    # R_t = R_0 @ exp(t * log(R_0^T @ R_1))
    R_delta = inv_SO3(R_0) @ R_1
    log_R_delta = log_SO3(R_delta)
    w_delta = bracket_so3(log_R_delta)  # Convert to vector form

    # Interpolate rotation
    t_expanded = t.unsqueeze(2)  # [B, 1, 1]
    R_t = R_0 @ exp_so3(t.squeeze(1) * w_delta)

    # Linear interpolation for translation
    p_t = p_0 + t * (p_1 - p_0)

    # Convert R_t back to axis-angle
    log_R_t = log_SO3(torch.bmm(inv_SO3(R_0), R_t))
    w_t = bracket_so3(log_R_t)

    # Construct x_t (without gripper for now)
    x_t = torch.cat([p_t, w_t], dim=1)

    # Compute velocity u_t in se(3)
    # Angular velocity (in spatial frame)
    u_angular = torch.einsum('bij,bj->bi', R_0, w_delta)

    # Linear velocity
    u_linear = p_1 - p_0

    u_t = torch.cat([u_angular, u_linear], dim=1)

    return x_t, u_t


def rotation_6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to rotation matrix.
    The 6D representation consists of the first two rows of a rotation matrix.

    Args:
        rot_6d: [B, 6] tensor representing first two rows of rotation matrix

    Returns:
        R: [B, 3, 3] rotation matrix
    """
    batch_size = rot_6d.shape[0]

    # First row - clone to avoid inplace operations on the input
    x_raw = rot_6d[:, :3].clone()
    # Normalize (store norm separately to avoid modifying intermediate tensors)
    x_norm = torch.norm(x_raw, dim=1, keepdim=True).clone()
    x = x_raw / (x_norm + 1e-8)

    # Second row (before Gram-Schmidt) - clone to avoid inplace operations
    y_raw = rot_6d[:, 3:6].clone()

    # Gram-Schmidt orthogonalization
    # Remove component of y in direction of x
    projection = torch.sum(x * y_raw, dim=1, keepdim=True).clone()
    y_ortho = y_raw - (projection * x)
    # Normalize
    y_norm = torch.norm(y_ortho, dim=1, keepdim=True).clone()
    y = y_ortho / (y_norm + 1e-8)

    # Third row (cross product)
    z = torch.cross(x, y, dim=1)

    # Stack into rotation matrix
    R = torch.stack([x, y, z], dim=1)  # [B, 3, 3]

    return R


def matrix_to_rotation_6d(R):
    """
    Convert rotation matrix to 6D rotation representation.

    Args:
        R: [B, 3, 3] rotation matrix

    Returns:
        rot_6d: [B, 6] tensor representing first two rows of rotation matrix
    """
    # Take first two rows
    rot_6d = R[:, :2, :].reshape(-1, 6)
    return rot_6d


def action_10d_to_7d(action_10d):
    """
    Convert action from 10D to 7D representation.

    10D format: [x, y, z, rot_6d (6 dims), gripper]
    - indices 0-2: x, y, z position
    - indices 3-8: 6D rotation representation (first two rows of rotation matrix)
    - index 9: gripper

    7D format: [x, y, z, axis_angle (3 dims), gripper]
    - indices 0-2: x, y, z position
    - indices 3-5: axis-angle representation (for angular velocity)
    - index 6: gripper

    Args:
        action_10d: [..., 10] tensor

    Returns:
        action_7d: [..., 7] tensor
    """
    # Handle different input shapes
    original_shape = action_10d.shape[:-1]
    action_10d_flat = action_10d.reshape(-1, 10)
    batch_size = action_10d_flat.shape[0]

    # Extract components
    pos = action_10d_flat[:, :3]  # [B, 3]
    rot_6d = action_10d_flat[:, 3:9]  # [B, 6]
    gripper = action_10d_flat[:, 9:10]  # [B, 1]

    # Convert 6D rotation to rotation matrix
    R = rotation_6d_to_matrix(rot_6d)  # [B, 3, 3]

    # Convert rotation matrix to axis-angle using log map
    log_R = log_SO3(R)  # [B, 3, 3]
    axis_angle = bracket_so3(log_R)  # [B, 3]

    # Combine into 7D action
    action_7d_flat = torch.cat([pos, axis_angle, gripper], dim=-1)  # [B, 7]

    # Reshape back to original shape
    action_7d = action_7d_flat.reshape(*original_shape, 7)

    return action_7d


def action_7d_to_10d(action_7d):
    """
    Convert action from 7D to 10D representation.

    7D format: [x, y, z, axis_angle (3 dims), gripper]
    - indices 0-2: x, y, z position
    - indices 3-5: axis-angle representation (angular velocity)
    - index 6: gripper

    10D format: [x, y, z, rot_6d (6 dims), gripper]
    - indices 0-2: x, y, z position
    - indices 3-8: 6D rotation representation (first two rows of rotation matrix)
    - index 9: gripper

    Args:
        action_7d: [..., 7] tensor

    Returns:
        action_10d: [..., 10] tensor
    """
    # Handle different input shapes
    original_shape = action_7d.shape[:-1]
    action_7d_flat = action_7d.reshape(-1, 7)
    batch_size = action_7d_flat.shape[0]

    # Extract components
    pos = action_7d_flat[:, :3]  # [B, 3]
    axis_angle = action_7d_flat[:, 3:6]  # [B, 3]
    gripper = action_7d_flat[:, 6:7]  # [B, 1]

    # Convert axis-angle to rotation matrix
    R = exp_so3(axis_angle)  # [B, 3, 3]

    # Convert rotation matrix to 6D representation
    rot_6d = matrix_to_rotation_6d(R)  # [B, 6]

    # Combine into 10D action
    action_10d_flat = torch.cat([pos, rot_6d, gripper], dim=-1)  # [B, 10]

    # Reshape back to original shape
    action_10d = action_10d_flat.reshape(*original_shape, 10)

    return action_10d
