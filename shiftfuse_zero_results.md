# ShiftFuse-Zero Results (NTU-60 xsub)

## nano_tiny_efficient — 97K params
- Run: 2026-03-18, 400 epochs, ~55s/ep (~6.1h)
- Config: mixup=0.1, cutmix=0.0, drop_path=0.10, wd=0.0001, lr=0.1 cosine
- **Best Val Top-1: 85.61%** (Top-5: 97.6%)
- Notes: fully converged by ep400, LR at min_lr=0.0001 at end; 300ep sufficient

## small_late_efficient_bb — 267K params
- **Best Val Top-1: 87.58%**
- Notes: late fusion 2-backbone + CrossStreamFusion + TLA

## large_late_efficient — 1.1M params
- Status: TRAINING (started ~2026-03-21, 150 epochs, 114s/ep, ~4.75h)
- Expected: 92%+
