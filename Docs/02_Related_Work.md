# 02 — Related Work

## SOTA Benchmark (NTU RGB+D 60, X-Sub protocol)

| Method        | Params | Top-1 (%) | Notes                                       |
|---------------|--------|-----------|---------------------------------------------|
| EfficientGCN-B0 | ~150K | 88.3     | Lightest; multi-scale parallel TCN          |
| EfficientGCN-B1 | ~300K | 89.4     | Slightly wider channels                     |
| EfficientGCN-B4 | ~2M   | 91.7     | Full-size EfficientGCN                      |
| Shift-GCN     | 2.8M   | 90.7     | Shift ops for efficiency                    |
| CTR-GCN       | 1.7M   | 92.4     | Channel-wise topology refinement            |
| InfoGCN       | 1.5M   | 93.0     | Mutual information maximization             |
| HD-GCN        | 3.3M   | 93.6     | Hierarchical directed graph                 |
| **LAST-E-base** | **364K** | *TBD* | Ours — target >91.7% at <2M tier          |
| **LAST-v2-base** | **9.2M** | *TBD* | Ours — target >93%                       |

---

## What LAST-E Borrows from EfficientGCN

**Multi-scale parallel TCN principle** — EfficientGCN demonstrated that splitting the temporal
channel budget into parallel branches with different dilation rates improves temporal coverage
at lower parameter cost than a single wide convolution. LAST-E's `MultiScaleTCN` applies this:
branch1 (C//2, dilation=1, pad=4) + branch2 (C//2, dilation=2, pad=8) → concatenate.

The C²/2 savings (vs. a single C×C branch) per block is the same mechanism EfficientGCN uses,
independently rediscovered and confirmed to apply cleanly to LightGCNBlock.

---

## What LAST-E Improves Over EfficientGCN

### DirectionalGCNConv vs. EfficientGCN's K-subset sum

EfficientGCN computes a separate full Conv2d per subset and sums results:
```
out = sum_{k=0}^{K-1} Conv2d_k(A_k @ x)
```
This costs K × C_in × C_out pointwise multiplies per block — expensive at larger widths.

LAST-E's `DirectionalGCNConv` keeps K=3 directed adjacency buffers (A_0: centripetal,
A_1: centrifugal, A_2: self-loops) but applies a **single shared Conv2d** with per-channel
alpha blending:
```
feat_k = alpha[k, :] * (A_k @ x)   # alpha: (K, C_in), zero-initialized
out = Conv2d(sum_k feat_k)
```
This preserves directional structure information (unlike summing first then convolving) while
keeping a single Conv2d weight matrix.

### Per-Channel StreamFusion vs. Scalar Blending

Most multi-stream methods use scalar fusion weights or simple concatenation. LAST-E's
`StreamFusion` learns a (3, C₀) weight matrix per-variant, with softmax normalization across
streams per channel. This allows the network to prefer different streams for different feature
channels — e.g., velocity-dominant channels for motion-sensitive joints.

### Why NOT K-separate full Conv2d per subset

For LAST-E-base (C=160 at last stage): K=3 full Conv2d would cost 3 × 160² = 76,800 params
per GCN layer vs. 160² = 25,600 with our shared Conv2d. At 4 blocks in stage 3, that is
~204K extra params just for GCN layers — nearly doubling total model size.

---

## Other Related Methods (brief)

- **ST-GCN** (Yan et al., 2018): first GCN-based skeleton action recognition; fixed topology
- **AGCN** (Shi et al., 2019): adaptive (learned) adjacency; basis for AdaptiveGraphConv in LAST-v2
- **2s-AGCN**: two-stream (joint + bone); predecessor to MIB 3-stream setup
- **MS-G3D** (Liu et al., 2020): disentangled multi-scale graph convolutions
- **PoseConv3D** (Duan et al., 2022): heatmap-based 3D conv; very different approach, higher cost
