import torch
from src.models.shiftfuse_zero import (
    build_shiftfuse_zero,
    build_shiftfuse_zero_late,
    build_shiftfuse_zero_midfusion
)

print("ShiftFuse-Zero Parameter Confirmation:")
print("-" * 40)

# 1. Nano
nano = build_shiftfuse_zero(variant='nano_tiny_efficient')
print(f"Nano:  {sum(p.numel() for p in nano.parameters()):,}")

# 2. Small
small = build_shiftfuse_zero_late(variant='small_late_efficient_bb', cross_stream=True)
print(f"Small: {sum(p.numel() for p in small.parameters()):,}")

# 3. Large
large = build_shiftfuse_zero_midfusion()
print(f"Large: {sum(p.numel() for p in large.parameters()):,}")
