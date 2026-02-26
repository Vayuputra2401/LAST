# Work Left — Post-Training Roadmap

> **Note:** Detailed plans for architecture, preprocessing, and training have been moved to dedicated experiment documents. See `Experiment-LAST-Base.md`, `Experiment-LAST-Lite.md`, `experiment-preprocessing.md`, and `experiment-training.md` for full specifications and timelines.

---

## 1. Implement Preprocessing Updates
**Reference:** `experiment-preprocessing.md`
- [x] Merge multi-stream generation into `preprocess_data.py`.
- [x] Update preprocessing to use uniform subsampling (T=64) instead of random crop.
- [ ] Run preprocessing pipeline to generate 4 streams (joint, velocity, bone, bone_velocity).
- [ ] Run quality verification scripts.

## 2. Implement LAST-Base (SOTA Research Model)
**Reference:** `Experiment-LAST-Base.md`
- [ ] Implement `CrossTemporalPrototypeGCN` block.
- [ ] Implement `FreqTemporalGate` and `PartitionedTemporalAttention`.
- [ ] Assemble 4-stream ensemble architecture.
- [ ] Run standalone training (Phase 1).
- [ ] Conduct LAST-Base ablation studies (L13-L20).

## 3. Implement LAST-Lite (Edge-Deployable Model)
**Reference:** `Experiment-LAST-Lite.md`
- [ ] Implement `ShiftFuseBlock` with `BRASP` (Body-Region-Aware Shift Patterns).
- [ ] Implement `FDCR` (Frozen DCT Frequency Routing).
- [ ] Assemble small and nano variants.
- [ ] Run standalone training baselines.
- [ ] Conduct LAST-Lite ablation studies (L0-L9).

## 4. Knowledge Distillation & Pretraining pipelines
**Reference:** `experiment-training.md`
- [ ] Write distillation training loop (`distill_trainer.py`).
- [ ] Execute Phase 2 Distillation: existing model → LAST-Lite.
- [ ] Execute Phase 2 Distillation: LAST-Base → LAST-Lite.
- [ ] Evaluate if MaskCLR Pretraining is needed (if small < 88%).

## 5. Quantization & Edge Deployment
**Reference:** `experiment-training.md`
- [ ] Implement ONNX export script with dynamic batching.
- [ ] Implement INT8 Post-Training Quantization with calibration.
- [ ] Deploy Edge demo.
