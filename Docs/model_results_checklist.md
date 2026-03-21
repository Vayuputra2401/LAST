# ShiftFuse-Zero Model Results Verification Checklist

This document contains a comprehensive list of all results and metrics added to the paper regarding the 3 ShiftFuse-Zero models (Nano, Small, Large), with verification ticks `[x]` for mathematically sound and consistent results, and `[ ]` (unverified) for those containing inconsistencies in the paper text.

## General Model Specifications & Training
- [ ] **Nano Params:** 97K parameters *(Unverified: Stated as 97K in Tables 1 & 2 but 0.10M / 100K in Tables 3, 4, 5)*
- [ ] **Small Params:** 267K parameters *(Unverified: Stated as 267K in Tables 1 & 2 but 0.25M / 250K in Text & Tables 3, 4, 5)*
- [x] **Large Params:** 1.67M parameters
- [x] **Nano Memory (f32/int8):** 388 KB / 97 KB
- [x] **Small Memory (f32/int8):** 1.07 MB / 267 KB
- [x] **Large Memory (f32/int8):** 6.38 MB / 1.67 MB
- [x] **Training Time (A100):** Nano = 2.5h, Small = 6h, Large = 4.75h

## Efficiency Metrics (GFLOPs & CPU Latency)
- [x] **Nano GFLOPs & Latency:** 0.26 GFLOPs, 11 ms
- [x] **Small GFLOPs & Latency:** 0.69 GFLOPs, 24 ms
- [x] **Large GFLOPs & Latency:** 4.71 GFLOPs, 89 ms

## Accuracy per Million Parameters (Acc/M on NTU-60 X-Sub)
- [ ] **Nano Acc/M:** 856 *(Unverified: Calculated using 0.10M (85.6/0.10). If using the exact 97K, it should be 882.5)*
- [ ] **Nano+KD Acc/M:** 885 *(Unverified: Calculated using 0.10M. If using exact 97K, it should be 912.4)*
- [ ] **Small+KD Acc/M:** 359 *(Unverified: Calculated using 0.25M. If using exact 267K, it should be 89.8/0.267 = 336.3)*
- [x] **Large Acc/M:** 55.4 *(Verified: 92.5/1.67 = 55.38)*

## Main Accuracy Results
### NTU-60 (X-Sub / X-View)
- [x] **SFZ-Nano:** 85.6% / 90.1%
- [x] **SFZ-Nano+KD:** 88.5% / 93.2%
- [x] **SFZ-Small:** 87.6% / 92.8%
- [x] **SFZ-Small+KD:** 89.8% / 94.7%
- [x] **SFZ-Large:** 92.5% / 96.4%

### NTU-120 (X-Sub / X-Set)
- [x] **SFZ-Nano:** 79.6% / 81.2%
- [x] **SFZ-Nano+KD:** 82.9% / 84.5%
- [x] **SFZ-Small:** 83.1% / 84.7%
- [x] **SFZ-Small+KD:** 85.5% / 87.0%
- [x] **SFZ-Large:** 88.5% / 90.1%

## Ablation Studies
### Cumulative Component Ablation (Nano, NTU-60 X-Sub)
- [x] **Baseline (no zero-param, TLA, KD):** 82.8%
- [x] **+ BRASP:** 83.9% (+1.1) *(Verified: 82.8 + 1.1 = 83.9)*
- [x] **+ SGPShift:** 84.6% (+1.8 cumulative) *(Verified: 82.8 + 1.8 = 84.6)*
- [x] **+ TLA:** 85.6% (+2.8 cumulative) *(Verified: 82.8 + 2.8 = 85.6)*
- [x] **+ KD:** 88.5% (+5.7 cumulative) *(Verified: 82.8 + 5.7 = 88.5)*

### GCN Partition Count Ablation (Nano, NTU-60 X-Sub)
- [x] **K=1:** 83.4%
- [x] **K=2:** 84.2% (+0.8) *(Verified: 83.4 + 0.8 = 84.2)*
- [x] **K=3:** 84.6% (+1.2) *(Verified: 83.4 + 1.2 = 84.6)*
- [ ] **K=3 + Shared Learned Matrix ($A_l$):** 85.6% (+1.6) *(Unverified: Math error. 85.6 - 83.4 = 2.2, not 1.6. Text also states $A_l$ adds +0.4 over K=3, making it 85.0. 85.6 includes TLA which isn't accounted for in the +1.6 delta.)*

### EfficientZero Block Component Ablation (Nano, NTU-60 X-Sub)
- [x] **Full Block:** 85.6% 
- [x] **- STC-Attention:** 85.0% (-0.6) *(Verified: 85.6 - 0.6 = 85.0)*
- [x] **- Joint Embedding:** 85.1% (-0.5) *(Verified: 85.6 - 0.5 = 85.1)*
- [x] **Replace DS-TCN with standard TCN:** 84.8% (-0.8, +12% params) *(Verified: 85.6 - 0.8 = 84.8; 109K / 97K ≈ +12.4%)*
- [x] **- DropPath (rate=0):** 84.9% (-0.7) *(Verified: 85.6 - 0.7 = 84.9)*

### DropPath Rate Sensitivity (Highest Accuracy)
- [x] **Nano Optimal ($p_d=0.10$):** 85.6%
- [x] **Small Optimal ($p_d=0.10$):** 87.6%
- [x] **Large Optimal ($p_d=0.20$):** 92.5%

### Fusion Strategy Ablation (Large, NTU-60 X-Sub)
- [x] **Early Fusion:** 89.8% (0.67M params)
- [x] **Late Fusion:** 91.9% (1.93M params)
- [x] **Mid-Network Fusion (Ours):** 92.5% (1.67M params) *(Verified: 1.93M - 1.67M = 0.26M lower params than late fusion as mentioned in text)*

### Cross-Stream Fusion (Small)
- [x] **Replace CSF with plain concatenation:** Costs -0.4pp
- [x] **Remove CSF entirely:** Costs -0.8pp 

### TLA Landmark Sensitivity
- [x] **Nano:** Saturates at $K=8$
- [x] **Large:** Benefits from $K=14$
