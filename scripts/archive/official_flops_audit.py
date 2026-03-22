import torch
import torch.nn as nn
from src.models.shiftfuse_zero import (
    build_shiftfuse_zero,
    build_shiftfuse_zero_late,
    build_shiftfuse_zero_midfusion
)

def count_flops_official(model, input_shape, device):
    """Refined FLOPs estimator from scripts/train.py."""
    model = model.to(device)
    model.eval()

    x_single = torch.randn(1, *input_shape).to(device)
    # Most variants use 3 or 4 streams
    x = {
        'joint':         x_single,
        'velocity':      x_single,
        'bone':          x_single,
        'bone_velocity': x_single,
    }

    flops = 0
    try:
        from torch.profiler import profile, ProfilerActivity
        # Note: Profiling with FLOPs requires torch.profiler
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            model(x)
        for evt in prof.key_averages():
            if evt.flops:
                flops += evt.flops
    except Exception as e:
        print(f"  [Profiler Error] {e}")
        # Fallback to a very rough estimate if profiler fails
        flops = sum(p.numel() for p in model.parameters()) * 2
        print(f"  [Fallback Applied] Using params*2 estimate.")

    return flops

# Define input shape (C, T, V, M=1) - using T=64 as per config
input_shape = (3, 64, 25, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Official ShiftFuse-Zero FLOPs Audit (T=64, M=2):")
print("-" * 60)

# 1. Nano
nano = build_shiftfuse_zero(variant='nano_tiny_efficient')
f_nano = count_flops_official(nano, input_shape, device)
print(f"Nano:  {f_nano/1e9:.3f} GFLOPs  ({sum(p.numel() for p in nano.parameters()):,})")

# 2. Small
small = build_shiftfuse_zero_late(variant='small_late_efficient_bb')
f_small = count_flops_official(small, input_shape, device)
print(f"Small: {f_small/1e9:.3f} GFLOPs  ({sum(p.numel() for p in small.parameters()):,})")

# 3. Large
large = build_shiftfuse_zero_midfusion()
f_large = count_flops_official(large, input_shape, device)
print(f"Large: {f_large/1e9:.3f} GFLOPs  ({sum(p.numel() for p in large.parameters()):,})")
