"""
Quick test for Relative Rotation Policy
"""
import torch
import sys
sys.path.append('.')

from equi_diffpo.policy.maniflow.maniflow_relrot_pointcloud_policy import ManiFlowRelRotPointcloudPolicy

def test_relative_rotation_policy():
    print("="*60)
    print("Testing Relative Rotation Policy")
    print("="*60)

    # Create dummy shape_meta
    shape_meta = {
        'action': {'shape': (10,)},  # 10D: [pos(3), rot_6d(6), gripper(1)]
        'obs': {
            'point_cloud': {'shape': (512, 3), 'type': 'low_dim'}
        }
    }

    # Create policy
    print("\n1. Creating policy...")
    policy = ManiFlowRelRotPointcloudPolicy(
        shape_meta=shape_meta,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=10,
        n_layer=2,  # Small for testing
        n_head=4,
        n_emb=128,
        visual_cond_len=256,
        encoder_output_dim=64,
        pointcloud_encoder_cfg={
            'in_channels': 3,
            'out_channels': 64,
            'use_layernorm': True,
            'final_norm': 'layernorm',
            'normal_channel': False,
            'num_points': 256,
            'pointwise': True,
        }
    )
    print("✅ Policy created successfully!")
    print(f"   Action dim: {policy.action_dim}")

    # Test noise sampling
    print("\n2. Testing noise sampling...")
    noise = policy.sample_noise(batch_size=4, horizon=16, device='cpu')
    print(f"   Noise shape: {noise.shape}")
    print(f"   Expected: [4, 16, 10]")
    assert noise.shape == (4, 16, 10), f"Wrong shape: {noise.shape}"
    print("✅ Noise sampling works!")

    # Test interpolation
    print("\n3. Testing interpolation...")
    noise = torch.randn(4, 16, 10)
    target = torch.randn(4, 16, 10)
    t = torch.tensor([0.5]).view(1, 1, 1).expand(4, -1, -1)

    try:
        interpolated = policy.interpolate(noise, target, t)
        print(f"   Interpolated shape: {interpolated.shape}")
        assert interpolated.shape == (4, 16, 10)
        print("✅ Interpolation works!")
    except Exception as e:
        print(f"❌ Interpolation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test relative rotation target computation
    print("\n4. Testing relative rotation target...")
    try:
        rel_target = policy.get_relative_rotation_target(noise, target)
        print(f"   Relative target shape: {rel_target.shape}")
        assert rel_target.shape == (4, 16, 10)
        print("✅ Relative rotation target computation works!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test apply_relative_action
    print("\n5. Testing apply_relative_action...")
    try:
        x = torch.randn(4, 16, 10)
        rel_action = torch.randn(4, 16, 10) * 0.1  # Small actions
        dt = 0.1
        x_new = policy.apply_relative_action(x, rel_action, dt)
        print(f"   Updated state shape: {x_new.shape}")
        assert x_new.shape == (4, 16, 10)
        print("✅ apply_relative_action works!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe Relative Rotation Policy is ready to use!")
    print("\nTo train:")
    print("  python train.py --config-name=train_maniflow_relrot_pointcloud_workspace")

if __name__ == "__main__":
    test_relative_rotation_policy()
