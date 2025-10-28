"""
SE(3) Manifold Flow Matching Policy for Point Cloud Observations.
Implements flow matching on SE(3) manifold for robot manipulation tasks.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from equi_diffpo.common.se3_utils import rotation_6d_to_matrix, matrix_to_rotation_6d
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.common.model_util import print_params
from equi_diffpo.model.vision_3d.pointnet_extractor import DP3Encoder
from equi_diffpo.model.diffusion.ditx import DiTX
from equi_diffpo.model.common.sample_util import *
from equi_diffpo.common.se3_utils import (
    sample_uniform_SO3,
    exp_se3,
    log_SE3,
    inv_SE3,
    bracket_se3,
    obtain_rigid_SE3_group,
    exp_SO3,
    log_SO3,
    inv_SO3,
    bracket_so3,
    get_se3_trajectory,
    action_7d_to_10d,
    action_10d_to_7d
)


class ManiFlowRigidSE3PointcloudPolicy(BaseImagePolicy):
    """
    ManiFlow policy with SE(3) manifold operations using RIGHT trivialization (body frame).

    Key features:
    1. Noise sampling on SE(3) manifold instead of Gaussian
    2. State representation: 10D [pos(3), rot_6d(6), gripper(1)]
    3. Model prediction: 7D velocity [pos_vel(3), omega_body(3), gripper_vel(1)]
    4. Geodesic interpolation on SO(3) for rotation
    5. RIGHT trivialization: omega_body = log(R_0^T @ R_1), update: R_new = R @ exp(dt * omega_body)
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
            # flow training parameters (no consistency for simplicity)
            denoise_timesteps=10,
            sample_t_mode_flow="beta",
            sample_target_t_mode="relative",  # relative, absolute
            # SE(3) specific parameters
            se3_noise_std_pos=1.0,  # std for position noise
            se3_noise_std_rot=1.0,  # std for rotation noise (in axis-angle)
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape

        # State representation: 10D with 6D rotation
        # Model predicts: 7D velocity (angular velocity for rotation)
        self.action_dim_internal = 10  # State: [pos(3), rot_6d(6), gripper(1)]
        self.action_dim_velocity = 7   # Velocity: [pos_vel(3), omega(3), gripper_vel(1)]

        # But the environment expects 10D actions
        if len(action_shape) == 1:
            action_dim_env = action_shape[0]
        elif len(action_shape) == 2:
            action_dim_env = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        self.action_dim_env = action_dim_env
        cprint(f"[SE3 Policy] Internal action dim: {self.action_dim_internal}, Environment action dim: {action_dim_env}", "cyan")

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
        cprint(f"[Encoder_type] {encoder_type}", "yellow")

        # create ManiFlow model with 7D action dimension
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = self.action_dim_internal + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.action_dim_internal
            global_cond_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        cprint(f"[ManiFlowSE3Policy] Using DiTX model with SE(3) manifold operations", "red")
        cprint(f"[ManiFlowSE3Policy] State: 10D (6D rotation), Prediction: 7D (body frame angular velocity)", "cyan")
        cprint(f"[ManiFlowSE3Policy] RIGHT trivialization: omega_body = log(R_0^T @ R_1)", "cyan")
        model = DiTX(
            input_dim=input_dim,
            output_dim=self.action_dim_velocity,  # Output 7D velocity (pos_vel, omega, gripper_vel)
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
        assert self.sample_target_t_mode in ["absolute", "relative"], \
            "sample_target_t_mode must be either 'absolute' or 'relative'"

        # SE(3) specific parameters
        self.se3_noise_std_pos = se3_noise_std_pos
        self.se3_noise_std_rot = se3_noise_std_rot

        cprint(f"[ManiFlowSE3Policy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - denoise_timesteps: {self.denoise_timesteps}", "yellow")
        cprint(f"  - sample_t_mode_flow: {self.sample_t_mode_flow}", "yellow")
        cprint(f"  - sample_target_t_mode: {self.sample_target_t_mode}", "yellow")
        cprint(f"  - se3_noise_std_pos: {self.se3_noise_std_pos}", "yellow")
        cprint(f"  - se3_noise_std_rot: {self.se3_noise_std_rot}", "yellow")

        print_params(self)

    def sample_se3_noise(self, batch_size, horizon, device):
        """
        Sample noise on SE(3) manifold.

        Returns:
            noise: [B, H, 10] tensor with SE(3) noise
                   (3 for position, 6 for rotation in 6D, 1 for gripper)
        """
        from equi_diffpo.common.se3_utils import matrix_to_rotation_6d

        # Sample position noise (Gaussian in R^3)
        pos_noise = torch.randn(batch_size, horizon, 3, device=device) * self.se3_noise_std_pos

        # Sample rotation noise (uniform on SO(3) then convert to 6D)
        total_samples = batch_size * horizon
        R_noise = sample_uniform_SO3(total_samples, device)

        # Convert to 6D rotation representation
        rot_noise_6d = matrix_to_rotation_6d(R_noise)  # [B*H, 6]
        rot_noise_6d = rot_noise_6d.reshape(batch_size, horizon, 6)

        # Sample gripper noise (Gaussian)
        gripper_noise = torch.randn(batch_size, horizon, 1, device=device)

        # Combine [pos(3), rot_6d(6), gripper(1)] = 10D
        noise = torch.cat([pos_noise, rot_noise_6d, gripper_noise], dim=-1)

        return noise

    def se3_interpolate(self, noise, target, timestep):
        """
        Interpolate on rigid SE(3) manifold between noise and target. 

        Args:
            noise: [B, H, 10] initial state (pos 3, rot_6d 6, gripper 1)
            target: [B, H, 10] target state
            timestep: [B, 1, 1] interpolation parameter

        Returns:
            interpolated: [B, H, 10] interpolated state
        """
        from equi_diffpo.common.se3_utils import rotation_6d_to_matrix, matrix_to_rotation_6d, obtain_rigid_SE3_group

        batch_size, horizon, _ = noise.shape

        # Flatten batch and horizon dimensions
        noise_flat = noise.reshape(-1, 10)  # [B*H, 10]
        target_flat = target.reshape(-1, 10)  # [B*H, 10]
        t_flat = timestep.expand(batch_size, horizon, 1).reshape(-1, 1)  # [B*H, 1]

        # For position (linear interpolation)
        # pos_t = noise_flat[:, :3] + t_flat * (target_flat[:, :3] - noise_flat[:, :3])

        # For rotation (geodesic interpolation on SO(3))
        rot_6d_0 = noise_flat[:, 3:9]
        rot_6d_1 = target_flat[:, 3:9]

        R_0 = rotation_6d_to_matrix(rot_6d_0)
        R_1 = rotation_6d_to_matrix(rot_6d_1)
        
        Rt_0 = obtain_rigid_SE3_group(noise_flat[:, :3], R_0)
        Rt_1 = obtain_rigid_SE3_group(target_flat[:, :3], R_1)
        # Geodesic interpolation: R_t = R_0 @ exp(t * log(R_0^T @ R_1))
        # This is the geodesic from R_0 to R_1 on SO(3)
        Rt_delta = inv_SE3(Rt_0) @ Rt_1
        log_Rt_delta = log_SE3(Rt_delta)
        w_delta = bracket_se3(log_Rt_delta)  # Body frame angular velocity (constant along geodesic)

        Rt_t = Rt_0 @ exp_se3(t_flat * w_delta)  # Geodesic at time t
        pos_t = Rt_t[..., :3, 3]
        R_t = Rt_t[..., :3, :3]
        rot_6d_t = matrix_to_rotation_6d(R_t)

        # For gripper (linear interpolation)
        gripper_t = noise_flat[:, 9:10] + t_flat * (target_flat[:, 9:10] - noise_flat[:, 9:10])

        # Combine [pos(3), rot_6d(6), gripper(1)] = 10D
        interpolated_flat = torch.cat([pos_t, rot_6d_t, gripper_t], dim=-1)
        interpolated = interpolated_flat.reshape(batch_size, horizon, 10)

        return interpolated

    def get_se3_velocity(self, noise, target):
        """
        Compute velocity on SE(3) manifold using RIGHT trivialization (body frame).

        Args:
            noise: [B, H, 10] initial state (pos 3, rot_6d 6, gripper 1)
            target: [B, H, 10] target state

        Returns:
            velocity: [B, H, 7] velocity in tangent space (pos_vel 3, omega_body 3, gripper_vel 1)
        """
        from equi_diffpo.common.se3_utils import rotation_6d_to_matrix, obtain_rigid_SE3_group

        batch_size, horizon, _ = noise.shape

        # Flatten
        noise_flat = noise.reshape(-1, 10)
        target_flat = target.reshape(-1, 10)

        # Position velocity (simple difference)
        # vel_pos = target_flat[:, :3] - noise_flat[:, :3]

        # Angular velocity in BODY frame (right trivialization)
        rot_6d_0 = noise_flat[:, 3:9]
        rot_6d_1 = target_flat[:, 3:9]

        R_0 = rotation_6d_to_matrix(rot_6d_0)
        R_1 = rotation_6d_to_matrix(rot_6d_1)
        Rt_0 = obtain_rigid_SE3_group(noise_flat[:, :3], R_0)
        Rt_1 = obtain_rigid_SE3_group(target_flat[:, :3], R_1)
        Rt_delta = inv_SE3(Rt_0) @ Rt_1
        log_Rt_delta = log_SE3(Rt_delta)
        vel_ang = bracket_se3(log_Rt_delta)  # Body frame angular velocity (constant along geodesic)

        # Gripper velocity
        vel_gripper = target_flat[:, 9:10] - noise_flat[:, 9:10]

        # Combine [omega_body(3), gripper_vel(1)] = 7D
        velocity_flat = torch.cat([vel_ang, vel_gripper], dim=-1)
        velocity = velocity_flat.reshape(batch_size, horizon, 7)

        return velocity

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            vis_cond=None,
            lang_cond=None,
            **kwargs):
        """
        Sample from the flow using ODE integration on SE(3) manifold.
        """
        batch_size, horizon, _ = condition_data.shape

        # Sample noise on SE(3) manifold
        noise = self.sample_se3_noise(batch_size, horizon, condition_data.device)

        ode_traj = self.sample_ode(
            x0=noise,
            N=self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **kwargs)

        return ode_traj[-1]  # Return the last state

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict action from observation.
        """
        # normalize input
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim_internal  # Use 10D [pos(3), rot_6d(6), gripper(1)]
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

        # Unnormalize prediction (10D)
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Actions are already 10D, no conversion needed!

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
        """
        Set normalizer. Only normalize position and gripper, leave 6D rotation unnormalized.
        Actions: [pos(3), rot_6d(6), gripper(1)] = 10D
        """
        # Copy the full normalizer (for observations)
        self.normalizer.load_state_dict(normalizer.state_dict())

        # Modify action normalizer to only normalize pos and gripper, not 6D rotation
        action_normalizer_10d = normalizer['action']
        from equi_diffpo.model.common.normalizer import LinearNormalizer as LN
        action_normalizer_custom = LN()

        # 10D: [pos(0-2), rot_6d(3-8), gripper(9)]
        # Normalize: pos and gripper
        # Identity: rot_6d (leave unchanged)

        if hasattr(action_normalizer_10d.params_dict, 'min'):
            min_custom = action_normalizer_10d.params_dict['min'].clone()
            max_custom = action_normalizer_10d.params_dict['max'].clone()
            # Set identity range for 6D rotation (will result in no normalization)
            min_custom[..., 3:9] = -1.0
            max_custom[..., 3:9] = 1.0
            action_normalizer_custom.params_dict['min'] = min_custom
            action_normalizer_custom.params_dict['max'] = max_custom

        if hasattr(action_normalizer_10d.params_dict, 'mean'):
            mean_custom = action_normalizer_10d.params_dict['mean'].clone()
            std_custom = action_normalizer_10d.params_dict['std'].clone()
            # Set identity normalization for 6D rotation
            mean_custom[..., 3:9] = 0.0
            std_custom[..., 3:9] = 1.0
            action_normalizer_custom.params_dict['mean'] = mean_custom
            action_normalizer_custom.params_dict['std'] = std_custom

        if hasattr(action_normalizer_10d.params_dict, 'scale'):
            scale_custom = action_normalizer_10d.params_dict['scale'].clone()
            offset_custom = action_normalizer_10d.params_dict['offset'].clone()
            # Set identity normalization for 6D rotation
            scale_custom[..., 3:9] = 1.0
            offset_custom[..., 3:9] = 0.0
            action_normalizer_custom.params_dict['scale'] = scale_custom
            action_normalizer_custom.params_dict['offset'] = offset_custom

        # Replace action normalizer with custom version
        self.normalizer['action'] = action_normalizer_custom

        cprint("[ManiFlowSE3Policy] Action normalizer: normalizing pos & gripper, identity for 6D rotation", "cyan")

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
            cprint(f"[ManiFlowSE3Policy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
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

    def sample_t(self, batch_size, mode="discrete"):
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

    def get_flow_velocity(self, actions, **model_kwargs):
        """
        Get flow velocity targets for training with SE(3) manifold operations.
        """
        target_dict = {}

        # vis_cond = model_kwargs.get('vis_cond', None)
        # lang_cond = model_kwargs.get('lang_cond', None)
        batch_size = actions.shape[0]
        device = actions.device

        # Sample t for flow
        t_flow = self.sample_t(batch_size, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((batch_size,), device=device)

        # Get target timestep
        # if self.sample_target_t_mode == "absolute":
        #     target_t_flow = t_flow.squeeze() + dt_flow
        # elif self.sample_target_t_mode == "relative":
        #     target_t_flow = t_flow.squeeze()
        target_t_flow = t_flow.squeeze()
        # Sample noise on SE(3) manifold
        x_0_flow = self.sample_se3_noise(batch_size, actions.shape[1], device)
        x_1_flow = actions.to(device)

        # Interpolate on SE(3) manifold
        # Use no_grad for target generation - we don't need gradients on targets
        with torch.no_grad():
            x_t_flow = self.se3_interpolate(x_0_flow, x_1_flow, t_flow)
            # Compute velocity on SE(3) manifold
            v_t_flow = self.get_se3_velocity(x_0_flow, x_1_flow)

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        # target_dict['vis_cond'] = vis_cond
        # target_dict['lang_cond'] = lang_cond

        return target_dict

    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        """Sample from the learned flow using Euler method on rigid SE(3) manifold.

        State x is 10D: [pos(3), rot_6d(6), gripper(1)]
        Model predicts 7D velocity: [omega_body(3), gripper_vel(1)]
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
            target_t = ti

            # Predict velocity (7D: pos_vel, omega, gripper_vel)
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)
            
            # Unnormalize
            pred[...,3:6]=pred[...,3:6]*3.14


            # Update using SE(3) exponential map
            batch_size, horizon, _ = x.shape
            x_flat = x.reshape(-1, 10)  # [B*H, 10]
            pred_flat = pred.reshape(-1, 7)  # [B*H, 7]

            # Update position
            # pos_new = x_flat[:, :3] + dt * pred_flat[:, :3]

            # Update rotation using exponential map
            # Current rotation from 6D representation
            rot_6d_current = x_flat[:, 3:9]
            R_current = rotation_6d_to_matrix(rot_6d_current)
            Rt_current = obtain_rigid_SE3_group(x_flat[:, :3], R_current)

            # Angular velocity from prediction (body frame, right trivialization)
            w_vel = pred_flat[:, 3:6]
            dR = exp_se3(dt * w_vel)

            # Apply rotation update: R_new = R_current @ dR (RIGHT trivialization)
            # This corresponds to: dR/dt = R @ [omega_body]_×
            Rt_new = Rt_current @ dR

            # Orthogonalize after multiplication to prevent numerical drift
            R_new = Rt_new[..., :3, :3]
            U, _, Vt = torch.linalg.svd(R_new)
            det_sign = torch.det(U @ Vt).unsqueeze(-1).unsqueeze(-1)
            S = torch.eye(3, device=R_new.device, dtype=R_new.dtype).unsqueeze(0).expand(R_new.shape[0], -1, -1).clone()
            S[:, 2, 2] = det_sign.squeeze(-1).squeeze(-1)
            R_new = U @ S @ Vt
            pos_new = Rt_new[..., :3, 3]
            # Convert back to 6D representation
            rot_6d_new = matrix_to_rotation_6d(R_new)
            gripper_new = x_flat[:, 9:10] + dt * pred_flat[:, 6:7]

            # Combine [pos(3), rot_6d(6), gripper(1)] = 10D
            x_new_flat = torch.cat([pos_new, rot_6d_new, gripper_new], dim=-1)
            x = x_new_flat.reshape(batch_size, horizon, 10)

            traj.append(x.detach().clone())

        return traj

    def compute_loss(self, batch, **kwargs):
        """Compute loss for SE(3) manifold flow matching."""
        # Normalize input
        nobs = self.normalizer.normalize(batch['obs'])

        # Keep actions in 10D: [x, y, z, rot_6d (6 dims), gripper]
        # State representation is 10D (6D rotation)
        actions_10d = batch['action'].to(self.device)

        # Normalize the 10D actions
        nactions = self.normalizer['action'].normalize(actions_10d)

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

        # Get flow velocity targets with SE(3) operations
        flow_target_dict = self.get_flow_velocity(
            nactions
        )

        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'],
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond,
            lang_cond=lang_cond
        )

        # Compute loss
        v_flow_target = flow_target_dict['v_target']

        # Normalize
        v_flow_target[..., 3:6]=v_flow_target[..., 3:6]/3.14


        # Separate loss for position, rotation, and gripper
        loss_pos = F.mse_loss(v_flow_pred[..., :3], v_flow_target[..., :3])
        loss_rot = F.mse_loss(v_flow_pred[..., 3:6], v_flow_target[..., 3:6])
        loss_gripper = F.mse_loss(v_flow_pred[..., 6:7], v_flow_target[..., 6:7])

        # Combined loss (you can weight these differently)
        loss = loss_pos + loss_rot + loss_gripper

        loss_dict = {
            'loss_pos': loss_pos.item(),
            'loss_rot': loss_rot.item(),
            'loss_gripper': loss_gripper.item(),
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
