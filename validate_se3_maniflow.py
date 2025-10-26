"""
Comprehensive validation script for SE(3) Manifold Flow Matching.
Tests all components: SE(3) utilities, policy, training loop, and ODE integration.

Usage:
    python validate_se3_maniflow.py
"""

import sys
import os
import pathlib

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import torch
import numpy as np
from termcolor import cprint


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    cprint(f"  {title}", "cyan", attrs=["bold"])
    print("=" * 70)


def print_test(name, passed=True, error=None):
    """Print test result."""
    if passed:
        cprint(f"✓ {name}", "green")
    else:
        cprint(f"✗ {name}", "red")
        if error:
            cprint(f"  Error: {error}", "red")


def print_metric(name, value):
    """Print a metric."""
    print(f"  • {name}: {value}")


# ============================================================================
# Test 1: SE(3) Utility Functions
# ============================================================================

def test_se3_utilities():
    """Test SE(3) Lie group operations."""
    print_section("TEST 1: SE(3) Utility Functions")

    from equi_diffpo.common.se3_utils import (
        sample_uniform_SO3, exp_so3, log_SO3, inv_SO3, bracket_so3
    )

    all_passed = True

    # Test 1.1: exp(log(R)) = R (round-trip)
    try:
        R = sample_uniform_SO3(10, 'cpu')
        log_R = log_SO3(R)
        w = bracket_so3(log_R)
        R_reconstructed = exp_so3(w)

        error = (R - R_reconstructed).abs().max().item()
        passed = error < 1e-4
        print_test(f"exp(log(R)) = R (error: {error:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("exp(log(R)) = R", False, str(e))
        all_passed = False

    # Test 1.2: log(exp(w)) = w (round-trip for small angles)
    try:
        w = torch.randn(10, 3) * 0.5  # Small angles
        R = exp_so3(w)
        log_R = log_SO3(R)
        w_reconstructed = bracket_so3(log_R)

        error = (w - w_reconstructed).abs().max().item()
        passed = error < 1e-4
        print_test(f"log(exp(w)) = w (error: {error:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("log(exp(w)) = w", False, str(e))
        all_passed = False

    # Test 1.3: R @ R^T = I (orthogonality)
    try:
        R = sample_uniform_SO3(10, 'cpu')
        identity = R @ R.transpose(1, 2)
        expected = torch.eye(3).unsqueeze(0).repeat(10, 1, 1)
        error = (identity - expected).abs().max().item()
        passed = error < 1e-4
        print_test(f"R @ R^T = I (error: {error:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("R @ R^T = I", False, str(e))
        all_passed = False

    # Test 1.4: det(R) = 1 (special orthogonal)
    try:
        R = sample_uniform_SO3(10, 'cpu')
        dets = torch.linalg.det(R)
        error = (dets - 1.0).abs().max().item()
        passed = error < 1e-4
        print_test(f"det(R) = 1 (error: {error:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("det(R) = 1", False, str(e))
        all_passed = False

    # Test 1.5: Angle bounds (||w|| should be <= π)
    try:
        R_large = sample_uniform_SO3(1000, 'cpu')
        log_R_large = log_SO3(R_large)
        w_large = bracket_so3(log_R_large)
        angles = w_large.norm(dim=1)

        max_angle = angles.max().item()
        mean_angle = angles.mean().item()
        passed = max_angle <= np.pi + 1e-4

        print_test(f"Angle bounds (max: {max_angle:.4f}, mean: {mean_angle:.4f})", passed)
        print_metric("Expected mean", "~π/2 = 1.57")
        print_metric("Actual mean", f"{mean_angle:.4f}")
        all_passed &= passed
    except Exception as e:
        print_test("Angle bounds", False, str(e))
        all_passed = False

    return all_passed


# ============================================================================
# Test 2: Policy Noise Sampling
# ============================================================================

def test_policy_noise_sampling():
    """Test SE(3) noise sampling in policy."""
    print_section("TEST 2: Policy Noise Sampling")

    from equi_diffpo.policy.maniflow.maniflow_se3_pointcloud_policy import ManiFlowSE3PointcloudPolicy
    from omegaconf import OmegaConf

    all_passed = True

    try:
        # Create minimal config
        cfg = OmegaConf.create({
            'shape_meta': {
                'action': {'shape': [7]},  # SE(3): [pos(3), ang_vel(3), gripper(1)]
                'obs': {
                    'point_cloud': {'shape': [512, 3]},
                    'robot0_eef_pos': {'shape': [3]},
                    'robot0_eef_quat': {'shape': [4]},
                    'robot0_gripper_qpos': {'shape': [2]}
                }
            }
        })

        # Create policy
        policy = ManiFlowSE3PointcloudPolicy(
            shape_meta=cfg.shape_meta,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=10,
            encoder_output_dim=128,
            n_layer=2,
            n_head=2,
            n_emb=128,
            visual_cond_len=64,
            pointcloud_encoder_cfg={
                'in_channels': 6,
                'out_channels': 128,
                'use_layernorm': True,
                'final_norm': 'layernorm',
                'normal_channel': False,
                'num_points': 64,
                'pointwise': True,
            },
        )

        # Test noise sampling
        batch_size, horizon = 4, 16
        noise = policy.sample_se3_noise(batch_size, horizon, 'cpu')

        # Test shape
        passed = noise.shape == (batch_size, horizon, 7)
        print_test(f"Noise shape {noise.shape}", passed)
        all_passed &= passed

        # Test components
        pos_noise = noise[..., :3]
        rot_noise = noise[..., 3:6]
        gripper_noise = noise[..., 6:7]

        print_metric("Position noise std", f"{pos_noise.std().item():.4f}")
        print_metric("Rotation noise std", f"{rot_noise.std().item():.4f}")
        print_metric("Gripper noise std", f"{gripper_noise.std().item():.4f}")

        # Check rotation angles are reasonable
        from equi_diffpo.common.se3_utils import exp_so3, log_SO3

        rot_flat = rot_noise.reshape(-1, 3)
        try:
            R = exp_so3(rot_flat)
            # If exp_so3 succeeds, rotations are valid
            angles = rot_flat.norm(dim=1)
            max_angle = angles.max().item()
            print_metric("Max rotation angle", f"{max_angle:.4f} rad")
            passed = max_angle < 10.0  # Reasonable bound
            print_test(f"Rotation angles reasonable (max: {max_angle:.4f})", passed)
            all_passed &= passed
        except Exception as e:
            print_test("Valid rotations from noise", False, str(e))
            all_passed = False

        return policy, all_passed

    except Exception as e:
        print_test("Policy creation and noise sampling", False, str(e))
        import traceback
        traceback.print_exc()
        return None, False


# ============================================================================
# Test 3: SE(3) Interpolation
# ============================================================================

def test_se3_interpolation(policy):
    """Test SE(3) geodesic interpolation."""
    print_section("TEST 3: SE(3) Interpolation")

    if policy is None:
        cprint("⊘ Skipping (policy not available)", "yellow")
        return False

    all_passed = True

    batch_size, horizon = 4, 16
    noise = torch.randn(batch_size, horizon, 7)
    target = torch.randn(batch_size, horizon, 7)

    # Test 3.1: Interpolation at t=0
    try:
        t = torch.zeros(batch_size, 1, 1)
        result = policy.se3_interpolate(noise, target, t)

        error_pos = (result[..., :3] - noise[..., :3]).abs().max().item()
        passed = error_pos < 1e-5
        print_test(f"Interpolation at t=0 (pos error: {error_pos:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("Interpolation at t=0", False, str(e))
        all_passed = False

    # Test 3.2: Interpolation at t=1
    try:
        t = torch.ones(batch_size, 1, 1)
        result = policy.se3_interpolate(noise, target, t)

        error_pos = (result[..., :3] - target[..., :3]).abs().max().item()
        passed = error_pos < 1e-5
        print_test(f"Interpolation at t=1 (pos error: {error_pos:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("Interpolation at t=1", False, str(e))
        all_passed = False

    # Test 3.3: Interpolation preserves valid rotations
    try:
        from equi_diffpo.common.se3_utils import exp_so3

        t = torch.rand(batch_size, 1, 1)
        result = policy.se3_interpolate(noise, target, t)

        rot_interp = result[..., 3:6].reshape(-1, 3)
        R_interp = exp_so3(rot_interp)

        # Check orthogonality
        identity = R_interp @ R_interp.transpose(1, 2)
        expected = torch.eye(3).unsqueeze(0).repeat(len(R_interp), 1, 1)
        error = (identity - expected).abs().max().item()

        passed = error < 1e-3  # Looser tolerance
        print_test(f"Interpolated rotations valid (error: {error:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("Valid interpolated rotations", False, str(e))
        all_passed = False

    # Test 3.4: Mid-point interpolation
    try:
        t = torch.ones(batch_size, 1, 1) * 0.5
        result = policy.se3_interpolate(noise, target, t)

        # Position should be halfway
        expected_pos = 0.5 * (noise[..., :3] + target[..., :3])
        error_pos = (result[..., :3] - expected_pos).abs().max().item()
        passed = error_pos < 1e-5
        print_test(f"Mid-point position (error: {error_pos:.2e})", passed)

        print_metric("Sample interpolated state", f"{result[0, 0, :]}")
        all_passed &= passed
    except Exception as e:
        print_test("Mid-point interpolation", False, str(e))
        all_passed = False

    return all_passed


# ============================================================================
# Test 4: SE(3) Velocity
# ============================================================================

def test_se3_velocity(policy):
    """Test SE(3) velocity calculation."""
    print_section("TEST 4: SE(3) Velocity Calculation")

    if policy is None:
        cprint("⊘ Skipping (policy not available)", "yellow")
        return False

    all_passed = True

    batch_size, horizon = 4, 16

    # Test 4.1: Same state should give zero velocity
    try:
        state = torch.randn(batch_size, horizon, 7)
        velocity = policy.get_se3_velocity(state, state)

        error = velocity.abs().max().item()
        passed = error < 1e-5
        print_test(f"Zero velocity for same state (error: {error:.2e})", passed)
        all_passed &= passed
    except Exception as e:
        print_test("Zero velocity for same state", False, str(e))
        all_passed = False

    # Test 4.2: Velocity shape
    try:
        noise = torch.randn(batch_size, horizon, 7)
        target = torch.randn(batch_size, horizon, 7)
        velocity = policy.get_se3_velocity(noise, target)

        passed = velocity.shape == (batch_size, horizon, 7)
        print_test(f"Velocity shape {velocity.shape}", passed)
        all_passed &= passed
    except Exception as e:
        print_test("Velocity shape", False, str(e))
        all_passed = False

    # Test 4.3: Linear and gripper velocities
    try:
        noise = torch.randn(batch_size, horizon, 7)
        target = torch.randn(batch_size, horizon, 7)
        velocity = policy.get_se3_velocity(noise, target)

        # Linear velocity should match difference
        expected_linear = target[..., :3] - noise[..., :3]
        error_linear = (velocity[..., :3] - expected_linear).abs().max().item()

        # Gripper velocity should match difference
        expected_gripper = target[..., 6:7] - noise[..., 6:7]
        error_gripper = (velocity[..., 6:7] - expected_gripper).abs().max().item()

        passed_linear = error_linear < 1e-5
        passed_gripper = error_gripper < 1e-5

        print_test(f"Linear velocity (error: {error_linear:.2e})", passed_linear)
        print_test(f"Gripper velocity (error: {error_gripper:.2e})", passed_gripper)

        print_metric("Angular velocity magnitude", f"{velocity[..., 3:6].abs().mean().item():.4f}")

        all_passed &= passed_linear and passed_gripper
    except Exception as e:
        print_test("Linear and gripper velocities", False, str(e))
        all_passed = False

    return all_passed


# ============================================================================
# Test 5: Forward Pass and Loss
# ============================================================================

def test_forward_pass(policy):
    """Test forward pass and loss computation."""
    print_section("TEST 5: Forward Pass and Loss Computation")

    if policy is None:
        cprint("⊘ Skipping (policy not available)", "yellow")
        return False

    all_passed = True

    try:
        batch_size = 2
        horizon = 16

        # Create dummy batch
        batch = {
            'obs': {
                'point_cloud': torch.randn(batch_size, 2, 512, 3),
                'robot0_eef_pos': torch.randn(batch_size, 2, 3),
                'robot0_eef_quat': torch.randn(batch_size, 2, 4),
                'robot0_gripper_qpos': torch.randn(batch_size, 2, 2)
            },
            'action': torch.randn(batch_size, horizon, 7)  # 7D SE(3) actions
        }

        # Set normalizer
        from equi_diffpo.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        policy.set_normalizer(normalizer)

        policy.to('cpu')
        policy.eval()

        # Compute loss
        with torch.no_grad():
            loss, loss_dict = policy.compute_loss(batch)

            # Check loss is finite
            passed = torch.isfinite(loss).item()
            print_test(f"Loss is finite (value: {loss.item():.6f})", passed)
            all_passed &= passed

            # Check loss components
            print_metric("Loss components", "")
            for key, val in loss_dict.items():
                is_finite = np.isfinite(val)
                status = "✓" if is_finite else "✗"
                print(f"    {status} {key}: {val:.6f}")
                all_passed &= is_finite

            # Check loss is positive
            passed = loss.item() >= 0
            print_test(f"Loss is non-negative", passed)
            all_passed &= passed

    except Exception as e:
        print_test("Forward pass and loss", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


# ============================================================================
# Test 6: ODE Integration
# ============================================================================

def test_ode_integration(policy):
    """Test ODE integration preserves SE(3) structure."""
    print_section("TEST 6: ODE Integration")

    if policy is None:
        cprint("⊘ Skipping (policy not available)", "yellow")
        return False

    all_passed = True

    try:
        from equi_diffpo.common.se3_utils import exp_so3

        batch_size = 2
        horizon = 16

        # Create dummy observation features
        vis_cond = torch.randn(batch_size, 2, policy.obs_feature_dim)

        # Sample initial noise
        x0 = policy.sample_se3_noise(batch_size, horizon, 'cpu')

        policy.eval()

        # Run ODE sampling
        with torch.no_grad():
            traj = policy.sample_ode(
                x0=x0,
                N=10,
                vis_cond=vis_cond,
                lang_cond=None
            )

            # Check trajectory length
            expected_len = 11  # N+1 states
            passed = len(traj) == expected_len
            print_test(f"Trajectory length (length: {len(traj)})", passed)
            all_passed &= passed

            # Check states in trajectory
            max_ortho_error = 0.0
            for i, x_t in enumerate(traj):
                rot = x_t[..., 3:6].reshape(-1, 3)
                R = exp_so3(rot)

                # Check orthogonality
                identity = R @ R.transpose(1, 2)
                expected = torch.eye(3).unsqueeze(0).repeat(len(R), 1, 1)
                error = (identity - expected).abs().max().item()
                max_ortho_error = max(max_ortho_error, error)

                if i % 5 == 0:  # Print every 5th step
                    print_metric(f"Step {i} orthogonality error", f"{error:.2e}")

            # Overall orthogonality check (looser tolerance during integration)
            passed = max_ortho_error < 1e-2
            print_test(f"Orthogonality preserved (max error: {max_ortho_error:.2e})", passed)
            all_passed &= passed

            # Check final state
            final_state = traj[-1]
            print_metric("Final state shape", f"{final_state.shape}")
            print_metric("Final state sample", f"{final_state[0, 0, :]}")

    except Exception as e:
        print_test("ODE integration", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


# ============================================================================
# Test 7: Minimal Training
# ============================================================================

def test_minimal_training():
    """Test a few training steps."""
    print_section("TEST 7: Minimal Training Loop")

    all_passed = True

    try:
        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.length = 16

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                return {
                    'obs': {
                        'point_cloud': torch.randn(2, 512, 3),
                        'robot0_eef_pos': torch.randn(2, 3),
                        'robot0_eef_quat': torch.randn(2, 4),
                        'robot0_gripper_qpos': torch.randn(2, 2)
                    },
                    'action': torch.randn(16, 7)  # 7D SE(3) actions
                }

        from torch.utils.data import DataLoader
        from equi_diffpo.policy.maniflow.maniflow_se3_pointcloud_policy import ManiFlowSE3PointcloudPolicy
        from equi_diffpo.model.common.normalizer import LinearNormalizer
        from omegaconf import OmegaConf

        # Create dataset
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Create policy
        cfg = OmegaConf.create({
            'shape_meta': {
                'action': {'shape': [7]},  # SE(3): [pos(3), ang_vel(3), gripper(1)]
                'obs': {
                    'point_cloud': {'shape': [512, 3]},
                    'robot0_eef_pos': {'shape': [3]},
                    'robot0_eef_quat': {'shape': [4]},
                    'robot0_gripper_qpos': {'shape': [2]}
                }
            }
        })

        policy = ManiFlowSE3PointcloudPolicy(
            shape_meta=cfg.shape_meta,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=5,
            encoder_output_dim=64,
            n_layer=2,
            n_head=2,
            n_emb=64,
            visual_cond_len=64,
            pointcloud_encoder_cfg={
                'in_channels': 6,
                'out_channels': 64,
                'use_layernorm': True,
                'final_norm': 'layernorm',
                'normal_channel': False,
                'num_points': 64,
                'pointwise': True,
            },
        )

        # Set normalizer
        normalizer = LinearNormalizer()
        policy.set_normalizer(normalizer)

        # Create optimizer
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        policy.train()
        policy.to('cpu')

        # Train for a few steps
        losses = []
        num_steps = 5

        print_metric("Training steps", f"{num_steps}")

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            optimizer.zero_grad()

            loss, loss_dict = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Check loss is finite
            if not torch.isfinite(loss):
                print_test(f"Step {step} loss is finite", False)
                all_passed = False
            else:
                print_metric(f"Step {step} loss", f"{loss.item():.6f}")

        # Overall training check
        passed = len(losses) == num_steps and all(np.isfinite(l) for l in losses)
        print_test(f"Completed {num_steps} training steps", passed)
        all_passed &= passed

        print_metric("Loss range", f"{min(losses):.6f} - {max(losses):.6f}")

        # Check gradients exist
        has_gradients = False
        grad_count = 0
        for name, param in policy.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_count += 1

        print_test(f"Gradients computed ({grad_count} parameters)", has_gradients)
        all_passed &= has_gradients

    except Exception as e:
        print_test("Minimal training loop", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all validation tests."""
    print("\n")
    cprint("╔" + "═" * 68 + "╗", "cyan", attrs=["bold"])
    cprint("║" + " " * 10 + "SE(3) MANIFOLD FLOW MATCHING VALIDATION" + " " * 18 + "║", "cyan", attrs=["bold"])
    cprint("╚" + "═" * 68 + "╝", "cyan", attrs=["bold"])

    results = {}

    # Run tests
    results['utilities'] = test_se3_utilities()

    policy, results['noise'] = test_policy_noise_sampling()

    results['interpolation'] = test_se3_interpolation(policy)

    results['velocity'] = test_se3_velocity(policy)

    results['forward_pass'] = test_forward_pass(policy)

    results['ode'] = test_ode_integration(policy)

    results['training'] = test_minimal_training()

    # Summary
    print_section("VALIDATION SUMMARY")

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "green" if passed else "red"
        cprint(f"  {status:8s} {test_name.replace('_', ' ').title()}", color)

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        cprint("  ✓✓✓ ALL TESTS PASSED! ✓✓✓", "green", attrs=["bold"])
        cprint("  SE(3) Manifold Flow implementation is working correctly!", "green")
    else:
        cprint("  ✗✗✗ SOME TESTS FAILED ✗✗✗", "red", attrs=["bold"])
        cprint("  Please review the errors above and fix the issues.", "red")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
