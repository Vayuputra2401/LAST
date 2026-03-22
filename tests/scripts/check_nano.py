import sys
import torch
from src.models.shiftfuse_zero import ShiftFuseZero

model = ShiftFuseZero(variant='nano_tiny_efficient', num_classes=60)
total_params = sum(p.numel() for p in model.parameters())

print(f"Nano new params: {total_params}")
