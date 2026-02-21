# 07 — Experiments

## Status: Code Verified, GPU Runs Pending

All model variants pass integration tests. No GPU training runs have completed yet.
This file is the living results table — fill in as training runs complete.

---

## Confirmed: Parameter Counts (No GPU Needed)

All param counts verified by `tests/test_model_integration.py`:

| Model          | Params    | All tests | EfficientGCN target |
|----------------|-----------|-----------|---------------------|
| LAST-E-nano    | 92,358    | PASS      | <150K (B0) ✓        |
| LAST-E-small   | 177,646   | PASS      | <300K (B1) ✓        |
| LAST-E-base    | 363,958   | PASS      | <2M (B4) ✓          |
| LAST-E-large   | 644,094   | PASS      | <2M (B4) ✓          |
| LAST-v2-base   | 9,217,256 | PASS      | — (teacher)         |

To re-verify: `python tests/test_model_integration.py`

---

## Pending: NTU60 X-Sub Accuracy (fill in as runs complete)

### Standalone Training

| Model          | NTU60 xsub Top-1 | NTU60 xsub Top-5 | Epochs | Run   |
|----------------|------------------|------------------|--------|-------|
| LAST-v2-base   | —                | —                | 70     | GCP   |
| LAST-E-base    | —                | —                | 70     | Kaggle|
| LAST-E-small   | —                | —                | 70     | Kaggle|
| LAST-E-nano    | —                | —                | 70     | Kaggle|
| LAST-E-large   | —                | —                | 70     | Kaggle|

### Post-Distillation (LAST-v2-base → LAST-E)

| Model          | NTU60 xsub Top-1 | Delta vs standalone | α    | τ   |
|----------------|------------------|---------------------|------|-----|
| LAST-E-nano    | —                | —                   | 0.7  | 4.0 |
| LAST-E-small   | —                | —                   | 0.6  | 4.0 |
| LAST-E-base    | —                | —                   | 0.5  | 4.0 |
| LAST-E-large   | —                | —                   | 0.5  | 4.0 |

---

## Pending: NTU120 (after NTU60 confirmed)

| Model        | NTU120 xsub Top-1 | NTU120 xset Top-1 |
|--------------|-------------------|-------------------|
| LAST-v2-base | —                 | —                 |
| LAST-E-base  | —                 | —                 |

---

## Expected Targets (Architecture Analysis)

Based on architecture comparison with EfficientGCN and similar efficiency-accuracy scaling:

| Model        | Standalone est. | Post-distill est. | EfficientGCN ref |
|--------------|-----------------|-------------------|------------------|
| LAST-E-nano  | 85–87%          | 87–89%            | B0: 88.3%        |
| LAST-E-small | 87–89%          | 89–91%            | B1: 89.4%        |
| LAST-E-base  | 89–91%          | 91–92%            | B4: 91.7%        |
| LAST-E-large | 91–93%          | 92–93%            | B4: 91.7%        |

These are estimates only. Actual numbers from training will replace these.

---

## Ablation Plan

Run after main results:

| Experiment                   | Purpose                                            |
|------------------------------|----------------------------------------------------|
| No ST_JointAtt (nano)        | Measure spatial attention contribution             |
| SingleScaleTCN vs MultiScale | Verify C²/2 savings don't hurt accuracy            |
| StreamFusion vs late logit sum | Verify early fusion > late fusion for student    |
| α sweep (nano: 0.5, 0.6, 0.7, 0.8) | Sensitivity of distillation weight         |
| τ sweep (nano: 3, 4, 6)      | Temperature sensitivity                            |

---

## SOTA Comparison (NTU60 xsub, update as results arrive)

| Method           | Params  | Top-1  |
|------------------|---------|--------|
| EfficientGCN-B0  | ~150K   | 88.3%  |
| EfficientGCN-B1  | ~300K   | 89.4%  |
| EfficientGCN-B4  | ~2M     | 91.7%  |
| Shift-GCN        | 2.8M    | 90.7%  |
| CTR-GCN          | 1.7M    | 92.4%  |
| InfoGCN          | 1.5M    | 93.0%  |
| HD-GCN           | 3.3M    | 93.6%  |
| **LAST-E-nano**  | 92K     | *TBD*  |
| **LAST-E-small** | 178K    | *TBD*  |
| **LAST-E-base**  | 364K    | *TBD*  |
| **LAST-E-large** | 644K    | *TBD*  |
| **LAST-v2-base** | 9.2M    | *TBD*  |
