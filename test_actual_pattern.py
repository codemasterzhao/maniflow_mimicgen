import torch

print("Testing the EXACT pattern from your policy...\n")

def dummy_function(x, y):
    return x + y * 2

actions = torch.randn(5, 10, 7, requires_grad=True)
x_0_flow = torch.randn(5, 10, 7)
x_1_flow = actions

print(f"x_1_flow requires_grad: {x_1_flow.requires_grad}")

print("\nWITHOUT torch.no_grad():")
result_old = dummy_function(x_0_flow, x_1_flow)
print(f"  result requires_grad: {result_old.requires_grad}")
print(f"  result grad_fn: {result_old.grad_fn is not None}")

print("\nWITH torch.no_grad():")
with torch.no_grad():
    result_new = dummy_function(x_0_flow, x_1_flow)
print(f"  result requires_grad: {result_new.requires_grad}")
print(f"  result grad_fn: {result_new.grad_fn is not None}")

if not result_new.requires_grad and result_old.requires_grad:
    print("\n✅ torch.no_grad() IS WORKING!")
else:
    print("\n❌ torch.no_grad() NOT WORKING!")
