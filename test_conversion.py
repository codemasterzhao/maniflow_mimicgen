"""
Test script to verify 10D <-> 7D conversion with gradient flow
"""
import torch
import sys
sys.path.append('.')

from equi_diffpo.common.se3_utils import action_10d_to_7d, action_7d_to_10d

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def test_conversion_with_gradients():
    print("Testing 10D -> 7D -> 10D conversion with gradient flow...")

    # Create sample 10D actions with gradient tracking
    batch_size, horizon = 5, 10
    action_10d = torch.randn(batch_size, horizon, 10, requires_grad=True)

    print(f"Input shape: {action_10d.shape}")

    # Convert to 7D
    print("Converting 10D -> 7D...")
    action_7d = action_10d_to_7d(action_10d)
    print(f"7D shape: {action_7d.shape}")

    # Convert back to 10D
    print("Converting 7D -> 10D...")
    action_10d_reconstructed = action_7d_to_10d(action_7d)
    print(f"Reconstructed 10D shape: {action_10d_reconstructed.shape}")

    # Compute loss and backward
    print("Computing loss and backpropagating...")
    loss = action_10d_reconstructed.sum()
    loss.backward()

    print(f"✅ Gradient computed successfully!")
    print(f"Input gradient shape: {action_10d.grad.shape}")
    print(f"Gradient max: {action_10d.grad.abs().max().item():.6f}")
    print(f"Gradient mean: {action_10d.grad.abs().mean().item():.6f}")

    # Check if gradients are not NaN
    assert not torch.isnan(action_10d.grad).any(), "Gradients contain NaN!"

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_conversion_with_gradients()
