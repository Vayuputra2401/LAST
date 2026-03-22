import torch
import torch.nn as nn
from src.models.shiftfuse_zero import (
    build_shiftfuse_zero,
    build_shiftfuse_zero_late,
    build_shiftfuse_zero_midfusion
)

def count_flops(model, input_size=(2, 3, 300, 25)):
    """Very basic FLOPs estimator for Conv2d and Linear layers.
    Note: Multi-body (M=2) inputs are handled in the forward pass.
    """
    total_flops = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            # batch * out_ch * h * w * in_ch * k_h * k_w
            batch_size = output.shape[0]
            out_channels = output.shape[1]
            out_h, out_w = output.shape[2], output.shape[3]
            in_channels = module.in_channels
            kernel_h, kernel_w = module.kernel_size
            flops = batch_size * out_channels * out_h * out_w * (in_channels * kernel_h * kernel_w)
            total_flops += flops

        elif isinstance(module, nn.Linear):
            # batch * in_features * out_features
            batch_size = output.shape[0]
            flops = batch_size * module.in_features * module.out_features
            total_flops += flops
        
        elif isinstance(module, nn.BatchNorm2d):
            # batch * ch * h * w (roughly 2 ops per element: shift and scale)
            flops = output.numel() * 2
            total_flops += flops

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(m.register_forward_hook(hook_fn))

    # Create dummy input based on variant's stream requirements
    B, C, T, V = input_size
    dummy_x = torch.randn(input_size)
    
    with torch.no_grad():
        if isinstance(model, nn.Module):
            # Try to determine stream names
            if hasattr(model, 'stream_names'):
                s_names = model.stream_names
            elif hasattr(model, 'STREAM_NAMES'):
                s_names = model.STREAM_NAMES
            else:
                s_names = ['joint', 'bone', 'velocity']
            
            # For Nano/Small, build_shiftfuse_zero expects a dict
            stream_dict = {name: dummy_x for name in s_names}
            try:
                # Force forward pass
                _ = model(stream_dict)
            except Exception as e:
                print(f"Error in forward pass: {e}")

    for h in hooks:
        h.remove()
    
    return total_flops

print("ShiftFuse-Zero FLOPs Audit:")
print("-" * 50)

# Setup dummy input (Batch=1, T=300, V=25)
input_size = (1, 3, 300, 25)

# 1. Nano
nano = build_shiftfuse_zero(variant='nano_tiny_efficient')
nano_flops = count_flops(nano, input_size)
print(f"Nano:  {nano_flops/1e9:.2f}G FLOPs  ({sum(p.numel() for p in nano.parameters()):,})")

# 2. Small
small = build_shiftfuse_zero_late(variant='small_late_efficient_bb')
small_flops = count_flops(small, input_size)
print(f"Small: {small_flops/1e9:.2f}G FLOPs  ({sum(p.numel() for p in small.parameters()):,})")

# 3. Large
large = build_shiftfuse_zero_midfusion()
large_flops = count_flops(large, input_size)
print(f"Large: {large_flops/1e9:.2f}G FLOPs  ({sum(p.numel() for p in large.parameters()):,})")

print("\n(Baseline EfficientGCN-B4: 8.36G FLOPs, 1.10M params)")
