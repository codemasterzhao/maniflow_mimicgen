"""
Manifold Flow Matching with RIGHT Trivialization for Point Cloud Observations.

Key difference from left trivialization:
- Velocities are in BODY FRAME (right trivialization) instead of spatial frame
- Update rule: R_{t+dt} = exp(dt * ω_body) @ R_t (pre-multiplication!)
- More intuitive for robotics: body velocities are often more natural
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
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
    exp_so3,
    log_SO3,
    inv_SO3,
    bracket_so3,
    action_10d_to_7d,
    action_7d_to_10d,
)


class ManiFlowRightTrivPointcloudPolicy(BaseImagePolicy):
    """
    ManiFlow with Right Trivialization on SO(3).

    Key features:
    1. Velocities in BODY FRAME (right trivialization)
    2. Action: [pos(3), ω_body(3), gripper(1)] = 7D
    3. Update: R_{t+dt} = exp(dt * ω_body) @ R_t (PRE-multiplication!)
    4. Natural for robotics: body angular velocities
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

        # Internal action: 7D [pos(3), ω_body(3), gripper(1)]
        self.action_dim_internal = 7

        # Environment expects 10D
        if len(action_shape) == 1:
            action_dim_env = action_shape[0]
        elif len(action_shape) == 2:
            action_dim_env = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        self.action_dim_env = action_dim_env
        cprint(f"[Right Triv Policy] Internal: 7D, Environment: {action_dim_env}D", "cyan")

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

        # create ManiFlow model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = self.action_dim_internal + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.action_dim_internal
            global_cond_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        cprint(f"[ManiFlowRightTrivPolicy] Using RIGHT trivialization (body frame)", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=self.action_dim_internal,
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

        cprint(f"[ManiFlowRightTrivPolicy] Initialized:", "yellow")
        cprint(f"  - Trivialization: RIGHT (body frame)", "yellow")
        cprint(f"  - Update rule: R_new = exp(dt*ω_body) @ R_old", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")

        print_params(self)

    def sample_noise(self, batch_size, horizon, device):
        """
        Sample noise on manifold.

        Returns:
            noise: [B, H, 7] tensor (pos, ω_body, gripper)
        """
        # Position noise
        pos_noise = torch.randn(batch_size, horizon, 3, device=device) * self.noise_std_pos

        # Rotation noise (uniform on SO(3), convert to body angular velocity)
        total_samples = batch_size * horizon
        R_noise = sample_uniform_SO3(total_samples, device)
        log_R_noise = log_SO3(R_noise)
        rot_noise = bracket_so3(log_R_noise)
        rot_noise = rot_noise.reshape(batch_size, horizon, 3) * self.noise_std_rot

        # Gripper noise
        gripper_noise = torch.randn(batch_size, horizon, 1, device=device)

        # Combine
        noise = torch.cat([pos_noise, rot_noise, gripper_noise], dim=-1)
        return noise

    def interpolate(self, noise, target, timestep):
        """
        Geodesic interpolation on SE(3) with right trivialization.

        Args:
            noise: [B, H, 7] initial state
            target: [B, H, 7] target state
            timestep: [B, 1, 1] interpolation parameter

        Returns:
            interpolated: [B, H, 7]
        """
        batch_size, horizon, _ = noise.shape

        # Flatten
        noise_flat = noise.reshape(-1, 7)
        target_flat = target.reshape(-1, 7)
        t_flat = timestep.expand(batch_size, horizon, 1).reshape(-1, 1)

        # Position: linear interpolation
        pos_t = noise_flat[:, :3] + t_flat * (target_flat[:, :3] - noise_flat[:, :3])

        # Rotation: geodesic interpolation (RIGHT trivialization aware)
        w_0 = noise_flat[:, 3:6]
        w_1 = target_flat[:, 3:6]

        R_0 = exp_so3(w_0)
        R_1 = exp_so3(w_1)

        # Geodesic: R_t = exp(t * log(R_delta)) @ R_0  (RIGHT version!)
        # where R_delta = R_0^T @ R_1
        R_delta = inv_SO3(R_0) @ R_1
        log_R_delta = log_SO3(R_delta)
        w_delta = bracket_so3(log_R_delta)

        # Interpolate in body frame
        R_t = exp_so3(t_flat.squeeze(1) * w_delta) @ R_0  # PRE-multiply!
        log_R_t = log_SO3(R_t)
        w_t = bracket_so3(log_R_t)

        # Gripper: linear interpolation
        gripper_t = noise_flat[:, 6:7] + t_flat * (target_flat[:, 6:7] - noise_flat[:, 6:7])

        # Combine
        interpolated_flat = torch.cat([pos_t, w_t, gripper_t], dim=-1)
        interpolated = interpolated_flat.reshape(batch_size, horizon, 7)

        return interpolated

    def get_body_velocity(self, noise, target):
        """
        Compute velocity in BODY FRAME (right trivialization).

        This is the key difference from left trivialization!

        Args:
            noise: [B, H, 7] initial state
            target: [B, H, 7] target state

        Returns:
            velocity: [B, H, 7] velocity in body frame
        """
        batch_size, horizon, _ = noise.shape

        # Flatten
        noise_flat = noise.reshape(-1, 7)
        target_flat = target.reshape(-1, 7)

        # Position velocity (same for both trivializations)
        vel_pos = target_flat[:, :3] - noise_flat[:, :3]

        # Angular velocity in BODY FRAME (right trivialization)
        w_0 = noise_flat[:, 3:6]
        w_1 = target_flat[:, 3:6]

        R_0 = exp_so3(w_0)
        R_1 = exp_so3(w_1)

        # Right trivialization: velocity in body frame
        # ω_body = log(R_0^T @ R_1)
        R_delta = inv_SO3(R_0) @ R_1
        log_R_delta = log_SO3(R_delta)
        vel_ang_body = bracket_so3(log_R_delta)  # This is ω_body!

        # Note: For LEFT trivialization, we would do:
        # vel_ang_spatial = torch.einsum('bij,bj->bi', R_0, w_delta)
        # But we're using RIGHT, so we keep it in body frame

        # Gripper velocity
        vel_gripper = target_flat[:, 6:7] - noise_flat[:, 6:7]

        # Combine
        velocity_flat = torch.cat([vel_pos, vel_ang_body, vel_gripper], dim=-1)
        velocity = velocity_flat.reshape(batch_size, horizon, 7)

        return velocity

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            vis_cond=None,
            lang_cond=None,
            **kwargs):
        """Sample from the flow using ODE integration."""
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
        Da = self.action_dim_internal
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

        # Unnormalize prediction (7D)
        naction_pred = nsample[..., :Da]
        action_pred_7d = self.normalizer['action'].unnormalize(naction_pred)

        # Convert 7D to 10D for environment
        action_pred = action_7d_to_10d(action_pred_7d)

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
        """Set normalizer (same as SE3 left triv - converts 10D to 7D)."""
        # Copy the full normalizer
        self.normalizer.load_state_dict(normalizer.state_dict())

        # Extract 7D action normalizer from 10D
        action_normalizer_10d = normalizer['action']

        from equi_diffpo.model.common.normalizer import LinearNormalizer as LN
        action_normalizer_7d = LN()

        # Note: 10D has [pos(0-2), rot_6d(3-8), gripper(9)]
        #       7D has [pos(0-2), ω_body(3-5), gripper(6)]

        if hasattr(action_normalizer_10d.params_dict, 'min'):
            min_7d = torch.zeros_like(action_normalizer_10d.params_dict['min'][..., :7])
            max_7d = torch.zeros_like(action_normalizer_10d.params_dict['max'][..., :7])
            # Position
            min_7d[..., :3] = action_normalizer_10d.params_dict['min'][..., :3]
            max_7d[..., :3] = action_normalizer_10d.params_dict['max'][..., :3]
            # Rotation (body angular velocity, typically [-π, π])
            min_7d[..., 3:6] = -3.14159
            max_7d[..., 3:6] = 3.14159
            # Gripper
            min_7d[..., 6:7] = action_normalizer_10d.params_dict['min'][..., 9:10]
            max_7d[..., 6:7] = action_normalizer_10d.params_dict['max'][..., 9:10]
            action_normalizer_7d.params_dict['min'] = min_7d
            action_normalizer_7d.params_dict['max'] = max_7d

        if hasattr(action_normalizer_10d.params_dict, 'mean'):
            mean_7d = torch.zeros_like(action_normalizer_10d.params_dict['mean'][..., :7])
            std_7d = torch.ones_like(action_normalizer_10d.params_dict['std'][..., :7])
            # Position
            mean_7d[..., :3] = action_normalizer_10d.params_dict['mean'][..., :3]
            std_7d[..., :3] = action_normalizer_10d.params_dict['std'][..., :3]
            # Rotation (zero mean, unit std)
            mean_7d[..., 3:6] = 0.0
            std_7d[..., 3:6] = 1.0
            # Gripper
            mean_7d[..., 6:7] = action_normalizer_10d.params_dict['mean'][..., 9:10]
            std_7d[..., 6:7] = action_normalizer_10d.params_dict['std'][..., 9:10]
            action_normalizer_7d.params_dict['mean'] = mean_7d
            action_normalizer_7d.params_dict['std'] = std_7d

        if hasattr(action_normalizer_10d.params_dict, 'scale'):
            scale_7d = torch.ones_like(action_normalizer_10d.params_dict['scale'][..., :7])
            offset_7d = torch.zeros_like(action_normalizer_10d.params_dict['offset'][..., :7])
            # Position
            scale_7d[..., :3] = action_normalizer_10d.params_dict['scale'][..., :3]
            offset_7d[..., :3] = action_normalizer_10d.params_dict['offset'][..., :3]
            # Rotation (identity)
            scale_7d[..., 3:6] = 1.0
            offset_7d[..., 3:6] = 0.0
            # Gripper
            scale_7d[..., 6:7] = action_normalizer_10d.params_dict['scale'][..., 9:10]
            offset_7d[..., 6:7] = action_normalizer_10d.params_dict['offset'][..., 9:10]
            action_normalizer_7d.params_dict['scale'] = scale_7d
            action_normalizer_7d.params_dict['offset'] = offset_7d

        # Replace action normalizer
        self.normalizer['action'] = action_normalizer_7d

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
            cprint(f"[ManiFlowRightTrivPolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
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
        Get flow targets for training with RIGHT trivialization.
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

        # Interpolate
        with torch.no_grad():
            x_t_flow = self.interpolate(x_0_flow, x_1_flow, t_flow)
            # Compute velocity in BODY FRAME (right trivialization!)
            v_t_flow = self.get_body_velocity(x_0_flow, x_1_flow)

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict

    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        """
        Sample using RIGHT trivialization ODE.

        Update rule: R_new = exp(dt * ω_body) @ R_old  (PRE-multiply!)
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

            # Predict velocity in body frame
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)

            # Update using RIGHT trivialization
            batch_size, horizon, _ = x.shape
            x_flat = x.reshape(-1, 7)
            pred_flat = pred.reshape(-1, 7)

            # Update position (same for both trivializations)
            pos_new = x_flat[:, :3] + dt * pred_flat[:, :3]

            # Update rotation with RIGHT trivialization: R_new = exp(dt*ω_body) @ R_old
            w_current = x_flat[:, 3:6]
            w_vel_body = pred_flat[:, 3:6]  # This is ω_body!

            R_current = exp_so3(w_current)
            dR = exp_so3(dt * w_vel_body)  # Rotation increment
            R_new = dR @ R_current  # PRE-multiply! (right trivialization)

            log_R_new = log_SO3(R_new)
            w_new = bracket_so3(log_R_new)

            # Update gripper
            gripper_new = x_flat[:, 6:7] + dt * pred_flat[:, 6:7]

            # Combine
            x_new_flat = torch.cat([pos_new, w_new, gripper_new], dim=-1)
            x = x_new_flat.reshape(batch_size, horizon, 7)

            traj.append(x.detach().clone())

        return traj

    def compute_loss(self, batch, **kwargs):
        """Compute loss for flow matching with right trivialization."""
        # Normalize input
        nobs = self.normalizer.normalize(batch['obs'])

        # Convert actions from 10D to 7D
        actions_10d = batch['action'].to(self.device)
        actions_7d = action_10d_to_7d(actions_10d)
        nactions = self.normalizer['action'].normalize(actions_7d)

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

        # Get flow targets
        flow_target_dict = self.get_flow_targets(
            nactions,
            vis_cond=vis_cond,
            lang_cond=lang_cond
        )

        # Predict velocity
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'],
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond,
            lang_cond=flow_target_dict['lang_cond']
        )

        # Compute loss
        v_flow_target = flow_target_dict['v_target']

        # Separate loss components
        loss_pos = F.mse_loss(v_flow_pred[..., :3], v_flow_target[..., :3])
        loss_rot = F.mse_loss(v_flow_pred[..., 3:6], v_flow_target[..., 3:6])
        loss_gripper = F.mse_loss(v_flow_pred[..., 6:7], v_flow_target[..., 6:7])

        # Combined loss
        loss = loss_pos + loss_rot + loss_gripper

        loss_dict = {
            'loss_pos': loss_pos.item(),
            'loss_rot': loss_rot.item(),
            'loss_gripper': loss_gripper.item(),
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
