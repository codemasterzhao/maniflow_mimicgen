"""
SO(3) manifold utilities for flow matching.
Provides exp_so3 (axis-angle -> rotation matrix) and log_SO3 (rotation matrix -> axis-angle),
plus rotation_6d <-> matrix conversions.
"""

import torch

EPS = 1e-4


def bracket_so3(w):
    """
    Convert between so(3) vector (N, 3) and skew-symmetric matrix (N, 3, 3).
    """
    if w.shape[-1:] == (3,) and len(w.shape) == 2:
        zeros = w.new_zeros(len(w))
        out = torch.stack([
            torch.stack([zeros, -w[:, 2], w[:, 1]], dim=1),
            torch.stack([w[:, 2], zeros, -w[:, 0]], dim=1),
            torch.stack([-w[:, 1], w[:, 0], zeros], dim=1)
        ], dim=1)
    elif w.shape[-2:] == (3, 3):
        out = torch.stack([w[:, 2, 1], w[:, 0, 2], w[:, 1, 0]], dim=1)
    else:
        raise ValueError(f"bracket_so3: unsupported shape {tuple(w.shape)}")
    return out


def log_SO3(R):
    """
    Logarithm map SO(3) -> so(3). Returns skew-symmetric matrix (N, 3, 3).
    Use bracket_so3 to convert to vector (N, 3).
    """
    n = R.shape[0]
    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    w_mat = torch.zeros_like(R)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS))

    is_regular = (tr_R + 1 > EPS)
    is_singular = (tr_R + 1 <= EPS)

    theta = theta.unsqueeze(1).unsqueeze(2)

    if is_regular.any():
        w_mat_regular = (1 / (2 * torch.sin(theta[is_regular]) + EPS)) * \
            (R[is_regular] - R[is_regular].transpose(1, 2)) * theta[is_regular]
        w_mat[is_regular] = w_mat_regular

    if is_singular.any():
        w_mat_singular = (R[is_singular] - torch.eye(3).to(R)) / 2
        w_vec_singular_raw = torch.sqrt(torch.diagonal(w_mat_singular, dim1=1, dim2=2) + 1)
        w_vec_singular = torch.where(
            torch.isnan(w_vec_singular_raw),
            torch.zeros_like(w_vec_singular_raw),
            w_vec_singular_raw)

        w_1 = w_vec_singular[:, 0]
        w_2 = w_vec_singular[:, 1] * (torch.sign(w_mat_singular[:, 0, 1]) + (w_1 == 0))
        w_3 = w_vec_singular[:, 2] * torch.sign(
            4 * torch.sign(w_mat_singular[:, 0, 2]) +
            2 * (w_1 == 0) * torch.sign(w_mat_singular[:, 1, 2]) +
            1 * (w_1 == 0) * (w_2 == 0))

        w_vec_singular = torch.stack([w_1, w_2, w_3], dim=1)
        w_mat[is_singular] = bracket_so3(w_vec_singular) * torch.pi

    return w_mat


def exp_so3(w_vec):
    """
    Exponential map so(3) -> SO(3).
    Input: (N, 3) axis-angle vector or (N, 3, 3) skew-symmetric matrix.
    Output: (N, 3, 3) rotation matrix.
    """
    if w_vec.shape[-2:] == (3, 3):
        w_vec = bracket_so3(w_vec)
    elif w_vec.shape[-1:] != (3,) or len(w_vec.shape) != 2:
        raise ValueError(f"exp_so3: unsupported shape {tuple(w_vec.shape)}")

    R = torch.eye(3, device=w_vec.device, dtype=w_vec.dtype).unsqueeze(0).repeat(len(w_vec), 1, 1)
    theta = w_vec.norm(dim=1)
    is_regular = theta > EPS

    if is_regular.any():
        w_reg = w_vec[is_regular]
        theta_reg = theta[is_regular].unsqueeze(1)
        w_hat = bracket_so3(w_reg / theta_reg)
        theta_reg = theta_reg.unsqueeze(2)
        R_reg = torch.eye(3, device=w_vec.device, dtype=w_vec.dtype).unsqueeze(0).repeat(len(w_reg), 1, 1) + \
            torch.sin(theta_reg) * w_hat + \
            (1 - torch.cos(theta_reg)) * w_hat @ w_hat
        R = R.clone()
        R[is_regular] = R_reg

    # SVD orthogonalization
    U, _, Vt = torch.linalg.svd(R)
    det_sign = torch.det(U @ Vt).unsqueeze(-1).unsqueeze(-1)
    S = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], -1, -1).clone()
    S[:, 2, 2] = det_sign.squeeze(-1).squeeze(-1)
    R = U @ S @ Vt

    return R


def rotation_6d_to_matrix(rot_6d):
    """Convert 6D rotation representation to rotation matrix. (N, 6) -> (N, 3, 3)"""
    x_raw = rot_6d[:, :3].clone()
    x = x_raw / (torch.norm(x_raw, dim=1, keepdim=True) + 1e-8)

    y_raw = rot_6d[:, 3:6].clone()
    projection = torch.sum(x * y_raw, dim=1, keepdim=True)
    y_ortho = y_raw - projection * x
    y = y_ortho / (torch.norm(y_ortho, dim=1, keepdim=True) + 1e-8)

    z = torch.cross(x, y, dim=1)
    R = torch.stack([x, y, z], dim=1)
    return R


def matrix_to_rotation_6d(R):
    """Convert rotation matrix to 6D rotation representation. (N, 3, 3) -> (N, 6)"""
    return R[:, :2, :].reshape(-1, 6)


def action_10d_to_7d(action_10d):
    """
    Convert 10D action [pos(3), rot_6d(6), gripper(1)] to 7D [pos(3), axis_angle(3), gripper(1)].
    Works with arbitrary leading dimensions (B,), (B, H), etc.
    """
    original_shape = action_10d.shape[:-1]
    flat = action_10d.reshape(-1, 10)

    pos = flat[:, :3]
    rot_6d = flat[:, 3:9]
    gripper = flat[:, 9:10]

    R = rotation_6d_to_matrix(rot_6d)
    log_R = log_SO3(R)  # (N, 3, 3) skew-symmetric
    axis_angle = bracket_so3(log_R)  # (N, 3) vector

    action_7d = torch.cat([pos, axis_angle, gripper], dim=-1)
    return action_7d.reshape(*original_shape, 7)


def action_7d_to_10d(action_7d):
    """
    Convert 7D action [pos(3), axis_angle(3), gripper(1)] to 10D [pos(3), rot_6d(6), gripper(1)].
    Works with arbitrary leading dimensions (B,), (B, H), etc.
    """
    original_shape = action_7d.shape[:-1]
    flat = action_7d.reshape(-1, 7)

    pos = flat[:, :3]
    axis_angle = flat[:, 3:6]
    gripper = flat[:, 6:7]

    R = exp_so3(axis_angle)  # (N, 3, 3)
    rot_6d = matrix_to_rotation_6d(R)  # (N, 6)

    action_10d = torch.cat([pos, rot_6d, gripper], dim=-1)
    return action_10d.reshape(*original_shape, 10)
