"""
Manifold Flow Matching Policy with Relative Rotation Prediction for Point Cloud Observations.

Key difference from SE3 policy:
- Network predicts RELATIVE ROTATION (6D or quaternion) instead of angular velocity
- Update rule: R_{t+dt} = R_t @ ΔR (simple multiplication, no exp map needed!)
- More direct and potentially more stable
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.common.model_util import print_params
from equi_diffpo.model.vision_3d.pointnet_extractor import DP3Encoder
from equi_diffpo.model.diffusion.ditx import DiTX
from equi_diffpo.model.common.sample_util import *
from equi_diffpo.common.se3_utils import (
    sample_uniform_SO3,
    log_SO3,
    bracket_so3,
    inv_SO3,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)


class ManiFlowRelRotPointcloudPolicy(BaseImagePolicy):
    """
    ManiFlow policy with Relative Rotation prediction.

    Key features:
    1. Network predicts relative rotation ΔR in 6D representation
    2. Action dimension is 10 (3 pos + 6 rot_6d + 1 gripper) - same as environment!
    3. Update: R_{t+dt} = R_t @ ΔR (simple, no exp/log needed during inference)
    4. More direct and stable than velocity-based approach
    """

    def __init__(self,
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_target_t_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            encoder_type="DP3Encoder",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            downsample_points=False,
            pre_norm_modality=False,
            language_conditioned=False,
            # flow training parameters
            denoise_timesteps=10,
            sample_t_mode_flow="beta",
            sample_target_t_mode="relative",
            # Noise parameters
            noise_std_pos=1.0,
            noise_std_rot=1.0,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape

        # Action dimension: [pos(3), rot_6d(6), gripper(1)] = 10D (same as environment!)
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        assert action_dim == 10, f"Expected 10D actions [pos(3), rot_6d(6), gripper(1)], got {action_dim}"
        self.action_dim = action_dim

        cprint(f"[RelRot Policy] Action dim: {self.action_dim} [pos(3) + rot_6d(6) + gripper(1)]", "cyan")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # create observation encoder
        self.encoder_type = encoder_type
        if encoder_type == "DP3Encoder":
            obs_encoder = DP3Encoder(
                observation_space=obs_dict,
                img_crop_shape=crop_shape,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                downsample_points=downsample_points,
            )
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")

        # create ManiFlow model with 10D action dimension
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = self.action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.action_dim
            global_cond_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        cprint(f"[ManiFlowRelRotPolicy] Using DiTX with relative rotation prediction", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=self.action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            pre_norm_modality=pre_norm_modality,
            language_conditioned=language_conditioned,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_target_t_mode = sample_target_t_mode

        # Noise parameters
        self.noise_std_pos = noise_std_pos
        self.noise_std_rot = noise_std_rot

        cprint(f"[ManiFlowRelRotPolicy] Initialized with:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - noise_std_pos: {self.noise_std_pos}", "yellow")
        cprint(f"  - noise_std_rot: {self.noise_std_rot}", "yellow")

        print_params(self)

    def sample_noise(self, batch_size, horizon, device):
        """
        Sample noise for position, rotation (6D), and gripper.

        Returns:
            noise: [B, H, 10] tensor
        """
        # Sample position noise (Gaussian)
        pos_noise = torch.randn(batch_size, horizon, 3, device=device) * self.noise_std_pos

        # Sample rotation noise (uniform on SO(3), convert to 6D)
        total_samples = batch_size * horizon
        R_noise = sample_uniform_SO3(total_samples, device)
        rot_6d_noise = matrix_to_rotation_6d(R_noise)  # [B*H, 6]
        rot_6d_noise = rot_6d_noise.reshape(batch_size, horizon, 6) * self.noise_std_rot

        # Sample gripper noise (Gaussian)
        gripper_noise = torch.randn(batch_size, horizon, 1, device=device)

        # Combine: [B, H, 10]
        noise = torch.cat([pos_noise, rot_6d_noise, gripper_noise], dim=-1)
        return noise

    def interpolate(self, noise, target, timestep):
        """
        Interpolate between noise and target.

        For position and gripper: linear interpolation
        For rotation: geodesic interpolation on SO(3)

        Args:
            noise: [B, H, 10] initial state
            target: [B, H, 10] target state
            timestep: [B, 1, 1] interpolation parameter

        Returns:
            interpolated: [B, H, 10] interpolated state
        """
        batch_size, horizon, _ = noise.shape

        # Flatten for easier processing
        noise_flat = noise.reshape(-1, 10)  # [B*H, 10]
        target_flat = target.reshape(-1, 10)
        t_flat = timestep.expand(batch_size, horizon, 1).reshape(-1, 1)  # [B*H, 1]

        # Position: linear interpolation
        pos_t = noise_flat[:, :3] + t_flat * (target_flat[:, :3] - noise_flat[:, :3])

        # Rotation: geodesic interpolation on SO(3)
        rot_6d_0 = noise_flat[:, 3:9]  # [B*H, 6]
        rot_6d_1 = target_flat[:, 3:9]

        # Convert to rotation matrices
        R_0 = rotation_6d_to_matrix(rot_6d_0)  # [B*H, 3, 3]
        R_1 = rotation_6d_to_matrix(rot_6d_1)

        # Geodesic interpolation: R_t = R_0 @ exp(t * log(R_0^T @ R_1))
        R_delta = inv_SO3(R_0) @ R_1
        log_R_delta = log_SO3(R_delta)

        # Scale by t
        log_R_t = t_flat.unsqueeze(-1) * log_R_delta  # [B*H, 3, 3]

        # Reconstruct: R_t = R_0 @ exp(log_R_t)
        # For small angles, exp(A) ≈ I + A + 0.5*A^2
        # But for robustness, use proper exponential
        from equi_diffpo.common.se3_utils import exp_so3
        w_t = bracket_so3(log_R_t)  # [B*H, 3]
        R_t = R_0 @ exp_so3(w_t)

        # Convert back to 6D
        rot_6d_t = matrix_to_rotation_6d(R_t)  # [B*H, 6]

        # Gripper: linear interpolation
        gripper_t = noise_flat[:, 9:10] + t_flat * (target_flat[:, 9:10] - noise_flat[:, 9:10])

        # Combine
        interpolated_flat = torch.cat([pos_t, rot_6d_t, gripper_t], dim=-1)
        interpolated = interpolated_flat.reshape(batch_size, horizon, 10)

        return interpolated

    def get_relative_rotation_target(self, noise, target):
        """
        Compute the relative rotation target: ΔR and Δpos.

        This is what the network should predict:
        - Δpos = target_pos - noise_pos (position difference)
        - ΔR in 6D representation (rotation difference)

        Args:
            noise: [B, H, 10] initial state
            target: [B, H, 10] target state

        Returns:
            relative_action: [B, H, 10] target for network to predict
        """
        batch_size, horizon, _ = noise.shape

        # Flatten
        noise_flat = noise.reshape(-1, 10)
        target_flat = target.reshape(-1, 10)

        # Position difference
        delta_pos = target_flat[:, :3] - noise_flat[:, :3]  # [B*H, 3]

        # Relative rotation: ΔR = R_0^T @ R_1
        rot_6d_0 = noise_flat[:, 3:9]
        rot_6d_1 = target_flat[:, 3:9]

        R_0 = rotation_6d_to_matrix(rot_6d_0)
        R_1 = rotation_6d_to_matrix(rot_6d_1)

        # Relative rotation
        delta_R = inv_SO3(R_0) @ R_1  # [B*H, 3, 3]
        delta_rot_6d = matrix_to_rotation_6d(delta_R)  # [B*H, 6]

        # Gripper difference
        delta_gripper = target_flat[:, 9:10] - noise_flat[:, 9:10]

        # Combine
        relative_action_flat = torch.cat([delta_pos, delta_rot_6d, delta_gripper], dim=-1)
        relative_action = relative_action_flat.reshape(batch_size, horizon, 10)

        return relative_action

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            vis_cond=None,
            lang_cond=None,
            **kwargs):
        """
        Sample from the flow using ODE integration.
        """
        batch_size, horizon, _ = condition_data.shape

        # Sample noise
        noise = self.sample_noise(batch_size, horizon, condition_data.device)

        ode_traj = self.sample_ode(
            x0=noise,
            N=self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **kwargs)

        return ode_traj[-1]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict action from observation."""
        # normalize input
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim

        # build input
        device = self.device
        dtype = self.dtype

        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # Encode observation
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]).to(device))
        nobs_features = self.obs_encoder(this_nobs)
        vis_cond = nobs_features.reshape(B, -1, Do)

        # Empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # Run sampling
        nsample = self.conditional_sample(
            cond_data,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **self.kwargs)

        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer (10D action space, same as environment)."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=weight_decay)

        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[ManiFlowRelRotPolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def sample_t(self, batch_size, mode="uniform"):
        """Sample t for flow matching."""
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f"Unsupported sample_t_mode {mode}")
        return t

    def get_flow_targets(self, actions, **model_kwargs):
        """
        Get flow targets for training.

        Network predicts relative rotation (not velocity!).
        """
        target_dict = {}

        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        batch_size = actions.shape[0]
        device = actions.device

        # Sample t for flow
        t_flow = self.sample_t(batch_size, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((batch_size,), device=device)

        # Get target timestep
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = t_flow.squeeze()

        # Sample noise
        x_0_flow = self.sample_noise(batch_size, actions.shape[1], device)
        x_1_flow = actions.to(device)

        # Interpolate to get x_t
        x_t_flow = self.interpolate(x_0_flow, x_1_flow, t_flow)

        # Target: relative rotation from x_0 to x_1
        # This is what network should predict at x_t
        relative_target = self.get_relative_rotation_target(x_0_flow, x_1_flow)

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['relative_target'] = relative_target
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict

    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        """
        Sample from the learned flow using Euler method.

        Update rule: x_{t+dt} = apply_relative_action(x_t, predicted_relative_action * dt)
        """
        if N is None:
            N = self.num_inference_steps
        dt = 1. / N
        traj = []
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt

            # Predict relative action
            pred_relative = self.model(x, ti, target_t=target_t, **model_kwargs)

            # Apply relative action to update x
            x = self.apply_relative_action(x, pred_relative, dt)

            traj.append(x.detach().clone())

        return traj

    def apply_relative_action(self, x, relative_action, dt):
        """
        Apply predicted relative action to current state.

        x: [B, H, 10] current state
        relative_action: [B, H, 10] predicted [Δpos, ΔR_6d, Δgripper]
        dt: scalar time step

        Returns: [B, H, 10] updated state
        """
        batch_size, horizon, _ = x.shape

        # Flatten
        x_flat = x.reshape(-1, 10)
        rel_flat = relative_action.reshape(-1, 10)

        # Update position
        pos_new = x_flat[:, :3] + dt * rel_flat[:, :3]

        # Update rotation: R_new = R_current @ ΔR
        rot_6d_current = x_flat[:, 3:9]
        delta_rot_6d = rel_flat[:, 3:9]

        R_current = rotation_6d_to_matrix(rot_6d_current)
        delta_R = rotation_6d_to_matrix(delta_rot_6d)

        # Apply relative rotation (scaled by dt for smooth integration)
        # For small dt, ΔR ≈ I + dt * skew(ω)
        # So we use: R_new = R_current @ (I + (ΔR - I) * dt)
        # Which is approximately: R_current @ exp(dt * log(ΔR))

        # Simple approach: R_new = R_current @ ΔR^dt
        # For dt=1/N, this gives smooth interpolation
        from equi_diffpo.common.se3_utils import exp_so3
        log_delta_R = log_SO3(delta_R)
        w_delta = bracket_so3(log_delta_R)
        scaled_delta_R = exp_so3(dt * w_delta)

        R_new = R_current @ scaled_delta_R
        rot_6d_new = matrix_to_rotation_6d(R_new)

        # Update gripper
        gripper_new = x_flat[:, 9:10] + dt * rel_flat[:, 9:10]

        # Combine
        x_new_flat = torch.cat([pos_new, rot_6d_new, gripper_new], dim=-1)
        x_new = x_new_flat.reshape(batch_size, horizon, 10)

        return x_new

    def compute_loss(self, batch, **kwargs):
        """Compute loss for relative rotation flow matching."""
        # Normalize input
        nobs = self.normalizer.normalize(batch['obs'])

        # Actions are already 10D: [pos(3), rot_6d(6), gripper(1)]
        actions = batch['action'].to(self.device)
        nactions = self.normalizer['action'].normalize(actions)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]

        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # Encode observations
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]).to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)

        # Get flow targets (no gradients needed for targets)
        with torch.no_grad():
            flow_target_dict = self.get_flow_targets(
                nactions,
                vis_cond=vis_cond,
                lang_cond=lang_cond
            )

        # Predict relative action
        relative_pred = self.model(
            sample=flow_target_dict['x_t'],
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond,
            lang_cond=flow_target_dict['lang_cond']
        )

        # Compute loss
        relative_target = flow_target_dict['relative_target']

        # Separate loss for different components
        loss_pos = F.mse_loss(relative_pred[..., :3], relative_target[..., :3])
        loss_rot = F.mse_loss(relative_pred[..., 3:9], relative_target[..., 3:9])
        loss_gripper = F.mse_loss(relative_pred[..., 9:10], relative_target[..., 9:10])

        # Combined loss
        loss = loss_pos + loss_rot + loss_gripper

        loss_dict = {
            'loss_pos': loss_pos.item(),
            'loss_rot': loss_rot.item(),
            'loss_gripper': loss_gripper.item(),
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
