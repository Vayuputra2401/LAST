import torch
import torch.nn as nn
from src.models.shiftfuse_zero import EfficientZeroBlock, Graph, normalize_symdigraph_full

def manual_flops_block(c_in, c_out, T=64, V=25, K=3):
    """
    Manual FLOPs estimation for one EfficientZeroBlock.
    """
    # 1. GCN (A_gcn @ x) -> (K * V * V * C)
    # Each partition: V * V * C_in (matrix mul)
    gcn_mm_flops = K * V * V * c_in * T
    
    # 2. GCN Conv1x1 (K weights, merged) -> (C_in * C_out * V)
    # Actually K weights: K * C_in * C_out * V
    gcn_conv_flops = K * c_in * c_out * V * T
    
    # 3. STCAttention
    # Temporal pool + FC: 2 * C^2 (roughly)
    stc_flops = (c_in * T * V) + (2 * c_in * c_in)
    
    # 4. DS-TCN (9x1 DW + 1x1 PW)
    tcn_dw_flops = c_out * 9 * T * V
    tcn_pw_flops = c_out * c_out * T * V
    
    # 5. BN + DropPath + Residual
    misc_flops = 10 * c_out * T * V # rough estimate
    
    total = gcn_mm_flops + gcn_conv_flops + stc_flops + tcn_dw_flops + tcn_pw_flops + misc_flops
    return {
        'total': total,
        'gcn': gcn_mm_flops + gcn_conv_flops,
        'tcn': tcn_dw_flops + tcn_pw_flops,
        'ratio_tcn_gcn': (tcn_dw_flops + tcn_pw_flops) / (gcn_mm_flops + gcn_conv_flops + 1e-6)
    }

print("Manual FLOPs Audit for Large Block (C=192, T=64, V=25):")
stats = manual_flops_block(192, 192)
print(f"Total Block FLOPs: {stats['total']/1e6:.2f}M")
print(f"GCN: {stats['gcn']/1e6:.2f}M")
print(f"TCN: {stats['tcn']/1e6:.2f}M")

# Full TCN estimate for comparison
full_tcn_flops = 192 * 192 * 9 * 64 * 25
print(f"Full TCN (9x1) Baseline: {full_tcn_flops/1e6:.2f}M")
print(f"Speedup vs Full TCN Block: {full_tcn_flops / stats['tcn']:.2f}x")

# Final Large estimate (15 blocks + stems)
# Assuming ~15 layers of 70-100M
total_est = 15 * stats['total']
print(f"\nEstimated Model FLOPs (15 blocks): {total_est/1e9:.2f}G")
