"""
Test to verify torch.no_grad() is actually working
"""
import torch

print("Testing torch.no_grad() behavior...\n")

# Test 1: Without no_grad
x = torch.randn(5, 3, requires_grad=True)
print("Test 1: WITHOUT torch.no_grad()")
print(f"  Input x requires_grad: {x.requires_grad}")

y = x * 2
print(f"  Output y requires_grad: {y.requires_grad}")
print(f"  Output y grad_fn: {y.grad_fn}")

# Test 2: With no_grad
x2 = torch.randn(5, 3, requires_grad=True)
print("\nTest 2: WITH torch.no_grad()")
print(f"  Input x2 requires_grad: {x2.requires_grad}")

with torch.no_grad():
    y2 = x2 * 2
    print(f"  Inside no_grad - Output y2 requires_grad: {y2.requires_grad}")
    print(f"  Inside no_grad - Output y2 grad_fn: {y2.grad_fn}")

print(f"  After no_grad - Output y2 requires_grad: {y2.requires_grad}")

# Test 3: Check memory usage difference
print("\n" + "="*60)
print("Memory Test: Simulating SE3 target generation")
print("="*60)

batch_size = 100
horizon = 10

# Simulate with grad tracking
x_input = torch.randn(batch_size, horizon, 7, requires_grad=True)
print(f"\nInput tensor requires_grad: {x_input.requires_grad}")

# Version 1: WITH grad tracking (wasteful)
result_with_grad = x_input * 2 + torch.randn(batch_size, horizon, 7)
print(f"\nVERSION 1 (wasteful): Result requires_grad: {result_with_grad.requires_grad}")
print(f"                      Result grad_fn: {result_with_grad.grad_fn is not None}")
print(f"                      PyTorch is tracking gradients: {result_with_grad.grad_fn is not None}")

# Version 2: WITHOUT grad tracking (optimized)
with torch.no_grad():
    result_no_grad = x_input.detach() * 2 + torch.randn(batch_size, horizon, 7)
print(f"\nVERSION 2 (optimized): Result requires_grad: {result_no_grad.requires_grad}")
print(f"                       Result grad_fn: {result_no_grad.grad_fn is not None}")
print(f"                       PyTorch is tracking gradients: {result_no_grad.grad_fn is not None}")

print("\n" + "="*60)
if result_no_grad.grad_fn is None and result_with_grad.grad_fn is not None:
    print("✅ torch.no_grad() IS WORKING - no computation graph built!")
else:
    print("❌ torch.no_grad() NOT WORKING - computation graph still built!")
print("="*60)
