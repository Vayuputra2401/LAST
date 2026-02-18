# SOTA Training Optimizations

Based on research into State-of-the-Art (SOTA) methods like **CTR-GCN** and **EfficientGCN**, we have optimized the training pipeline (`trainer.py` and `default.yaml`).

## 1. Optimizer: SGD + Nesterov
*   **Choice**: Switched default from `AdamW` to `SGD`.
*   **Reasoning**: Research shows SGD with Nesterov momentum generalizes better for Skeleton GCNs than Adam/AdamW, which can overshoot or settle in sharp minima.
*   **Configuration**:
    *   `optimizer: sgd`
    *   `momentum: 0.9`
    *   `nesterov: true`
    *   `lr: 0.1` (Standard high starting LR for SGD)

## 2. Weight Decay Filtering
*   **Implementation**: Updated `Trainer` class in `trainer.py`.
*   **Change**: We now separate parameters into two groups:
    *   **Decay (0.0004)**: Convolutional and Linear weights.
    *   **No Decay (0.0)**: Biases, Batch Normalization weights, and offsets.
*   **Benefit**: Applying weight decay to BN parameters hurts performance. This filtering is a critical SOTA detail often missed in basic implementations.

## 3. Learning Rate Scheduler
*   **Strategy**: Cosine Annealing with Linear Warmup.
*   **Warmup**: 5 Epochs. Starts at `0.01` and ramps linearly to `0.1`.
*   **Benefit**: High learning rates (`0.1`) can cause instability at the start. Warmup allows the gradients to stabilize before aggressive optimization begins.

## 4. Regularization
*   **Label Smoothing**: Enabled (0.1). Prevents the model from becoming over-confident on noisy skeleton data.
*   **Gradient Clipping**: Set to `5.0`. Essential for GCNs as graph aggregations can sometimes cause exploding gradients.

## Summary of Defaults (`default.yaml`)
```yaml
optimizer: "sgd"
lr: 0.1
weight_decay: 0.0004
epochs: 70
warmup_epochs: 5
```
This recipe closely matches the official configurations of top-performing skeleton action recognition models.
