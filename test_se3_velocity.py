"""
Test script to verify SE3 velocity computation with gradient flow
"""
import torch
import sys
sys.path.append('.')

from equi_diffpo.common.se3_utils import exp_so3, log_SO3, bracket_so3, inv_SO3

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def test_se3_velocity_gradient():
    """Test the exact code path from get_se3_velocity"""
    print("Testing SE3 velocity computation with gradient flow...")

    batch_size = 5

    # Create sample axis-angle vectors (requires_grad=True)
    w_0 = torch.randn(batch_size, 3, requires_grad=True) * 0.1
    w_1 = torch.randn(batch_size, 3, requires_grad=True) * 0.1

    print(f"w_0 shape: {w_0.shape}, requires_grad: {w_0.requires_grad}")
    print(f"w_1 shape: {w_1.shape}, requires_grad: {w_1.requires_grad}")

    # This is the exact code from get_se3_velocity
    print("\nConverting to rotation matrices...")
    R_0 = exp_so3(w_0)
    R_1 = exp_so3(w_1)

    print(f"R_0 shape: {R_0.shape}")
    print(f"R_1 shape: {R_1.shape}")

    # Velocity in spatial frame
    print("\nComputing angular velocity...")
    R_delta = inv_SO3(R_0) @ R_1
    log_R_delta = log_SO3(R_delta)
    w_delta = bracket_so3(log_R_delta)
    vel_ang = torch.einsum('bij,bj->bi', R_0, w_delta)

    print(f"Angular velocity shape: {vel_ang.shape}")

    # Compute loss and backward
    print("\nComputing loss and backpropagating...")
    loss = vel_ang.sum()

    try:
        loss.backward()
        print(f"✅ Gradient computed successfully (no inplace operation errors)!")

        # Check if gradients exist
        if w_0.grad is not None:
            print(f"w_0 gradient shape: {w_0.grad.shape}")
            print(f"w_0 gradient max: {w_0.grad.abs().max().item():.6f}")
            assert not torch.isnan(w_0.grad).any(), "w_0 gradients contain NaN!"
        else:
            print("Note: w_0.grad is None (may not be a leaf tensor)")

        if w_1.grad is not None:
            print(f"w_1 gradient shape: {w_1.grad.shape}")
            print(f"w_1 gradient max: {w_1.grad.abs().max().item():.6f}")
            assert not torch.isnan(w_1.grad).any(), "w_1 gradients contain NaN!"
        else:
            print("Note: w_1.grad is None (may not be a leaf tensor)")

    except RuntimeError as e:
        print(f"❌ Gradient computation failed: {e}")
        raise

    print("\n✅ All SE3 velocity tests passed (no inplace operation errors)!")

if __name__ == "__main__":
    test_se3_velocity_gradient()
