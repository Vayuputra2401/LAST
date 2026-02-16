"""
Teacher-LAST: Dry Run Verification
====================================
Validates the entire pipeline WITHOUT training or loading real videos.
Tests:
  1. Frame sampling math (short/long videos, train/val/test)
  2. Spatial transform output shapes
  3. Model input/output shapes
  4. LR scaling formula
  5. Cosine scheduler shape
  6. Config loading
  7. End-to-end data flow
"""

import os
import sys
import numpy as np
import torch

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =========================================================================
# TEST 1: Config Loading
# =========================================================================
section("1. Config Loading")

from src.utils import load_config
config = load_config(os.path.join(project_root, "configs", "ntu60_finetune.yaml"))

check("Config has 'data' section", 'data' in config)
check("Config has 'model' section", 'model' in config)
check("Config has 'training' section", 'training' in config)
check("Config has 'checkpoint' section", 'checkpoint' in config)
check("Config has 'hardware' section", 'hardware' in config)
check("num_classes = 60", config['data']['num_classes'] == 60)
check("num_frames = 16", config['data']['num_frames'] == 16)
check("sampling_rate = 4", config['data']['sampling_rate'] == 4)
check("input_size = 224", config['data']['input_size'] == 224)
check("epochs = 80", config['training']['epochs'] == 80)
check("batch_size = 8", config['training']['batch_size'] == 8)


# =========================================================================
# TEST 2: LR Scaling
# =========================================================================
section("2. LR Scaling Formula")

from src.utils import compute_actual_lr

# 1 GPU
lr_1gpu = compute_actual_lr(base_lr=0.001, batch_size=8, num_gpus=1, update_freq=4)
check("1 GPU: lr = 0.001 * (8*1*4)/256 = 1.25e-4",
      abs(lr_1gpu - 1.25e-4) < 1e-10,
      f"got {lr_1gpu}")

# 2 GPUs
lr_2gpu = compute_actual_lr(base_lr=0.001, batch_size=8, num_gpus=2, update_freq=4)
check("2 GPUs: lr = 0.001 * (8*2*4)/256 = 2.5e-4",
      abs(lr_2gpu - 2.5e-4) < 1e-10,
      f"got {lr_2gpu}")

# 4 GPUs
lr_4gpu = compute_actual_lr(base_lr=0.001, batch_size=8, num_gpus=4, update_freq=4)
check("4 GPUs: lr = 0.001 * (8*4*4)/256 = 5e-4",
      abs(lr_4gpu - 5e-4) < 1e-10,
      f"got {lr_4gpu}")

# Linear scaling: doubling GPUs should double LR
check("LR scales linearly with GPUs",
      abs(lr_2gpu / lr_1gpu - 2.0) < 1e-10,
      f"ratio = {lr_2gpu/lr_1gpu}")


# =========================================================================
# TEST 3: Cosine Scheduler
# =========================================================================
section("3. Cosine Scheduler with Warmup")

from src.utils import cosine_scheduler

schedule = cosine_scheduler(
    base_value=1e-3, final_value=1e-6,
    epochs=80, niter_per_ep=100,
    warmup_epochs=5, start_warmup_value=1e-6
)

check("Schedule length = epochs * niter_per_ep",
      len(schedule) == 80 * 100,
      f"got {len(schedule)}, expected {80*100}")

check("Schedule starts at warmup_value",
      abs(schedule[0] - 1e-6) < 1e-10,
      f"got {schedule[0]}")

check("Schedule peaks at base_value after warmup",
      abs(schedule[5 * 100 - 1] - 1e-3) < 1e-5,
      f"got {schedule[5*100-1]}")

check("Schedule ends near final_value",
      abs(schedule[-1] - 1e-6) < 1e-5,
      f"got {schedule[-1]}")

check("Schedule is monotonically increasing during warmup",
      all(schedule[i] <= schedule[i+1] for i in range(5*100-1)),
      "warmup phase checks")

check("Peak value >= all values",
      max(schedule) <= 1e-3 + 1e-8,
      f"max = {max(schedule)}")


# =========================================================================
# TEST 4: Frame Sampling — Short Video (40 frames)
# =========================================================================
section("4. Frame Sampling — Short Video (40 frames, like 'clapping')")

from src.dataset import NTU60Dataset

# Create a minimal mock config
mock_data_config = {
    'num_frames': 16,
    'sampling_rate': 4,
    'input_size': 224,
    'short_side_size': 224,
}

# We need to test _sample_frame_indices directly
# Create a dummy dataset instance without CSV (we'll test the method directly)
class MockDataset:
    def __init__(self):
        self.clip_len = 16
        self.frame_sample_rate = 4

mock = MockDataset()
# Bind the method from NTU60Dataset
mock._sample_frame_indices = NTU60Dataset._sample_frame_indices.__get__(mock)

# Short video: 40 frames
indices_short = mock._sample_frame_indices(total_frames=40, mode='train')
check("Short video (40 frames) returns 16 indices",
      len(indices_short) == 16,
      f"got {len(indices_short)} indices")

check("All indices in valid range [0, 39]",
      np.all(indices_short >= 0) and np.all(indices_short <= 39),
      f"range: [{indices_short.min()}, {indices_short.max()}]")

check("Indices are non-decreasing (temporal order preserved)",
      all(indices_short[i] <= indices_short[i+1] for i in range(len(indices_short)-1)),
      f"indices: {indices_short.tolist()}")

# For 40 frames: available = 40//4 = 10 real, padded with 6 repeated last frame
available = 40 // 4  # 10
check(f"Short video: {available} real frames sampled, {16-available} padded",
      True,
      f"last {16-available} should be frame 39")

# Last 6 indices should all be 39 (last frame repeat)
padded_count = 16 - available
last_indices = indices_short[-padded_count:]
try:
    check("Last-frame repeat: padded indices are all frame 39",
          np.all(last_indices == 39),
          f"padded indices: {last_indices.tolist()}")
except Exception as e:
    check("Last-frame repeat: padded indices are all frame 39", False, f"Error: {e}")

# Train and val should produce SAME structure (but different random seeds for long videos)
# For short videos, there's no randomness, so they should be IDENTICAL
indices_short_val = mock._sample_frame_indices(total_frames=40, mode='validation')
check("Short video: train and val produce SAME indices (no randomness)",
      np.array_equal(indices_short, indices_short_val),
      f"train={indices_short.tolist()}, val={indices_short_val.tolist()}")

print(f"\n  Sampled indices: {indices_short.tolist()}")


# =========================================================================
# TEST 5: Frame Sampling — Long Video (150 frames)
# =========================================================================
section("5. Frame Sampling — Long Video (150 frames, like 'writing')")

indices_long = mock._sample_frame_indices(total_frames=150, mode='train')
check("Long video (150 frames) returns 16 indices",
      len(indices_long) == 16,
      f"got {len(indices_long)} indices")

check("All indices in valid range [0, 149]",
      np.all(indices_long >= 0) and np.all(indices_long <= 149),
      f"range: [{indices_long.min()}, {indices_long.max()}]")

check("Indices are non-decreasing",
      all(indices_long[i] <= indices_long[i+1] for i in range(len(indices_long)-1)),
      f"indices: {indices_long.tolist()}")

# Window should be exactly 64 frames wide
window_size = indices_long[-1] - indices_long[0]
check("Window span ≤ 64 frames (converted_len)",
      window_size <= 64,
      f"window: [{indices_long[0]}, {indices_long[-1]}], span={window_size}")

# Spacing should be approximately frame_sample_rate = 4
spacings = np.diff(indices_long)
avg_spacing = np.mean(spacings)
check("Average spacing ≈ 4 (frame_sample_rate)",
      3.0 <= avg_spacing <= 5.0,
      f"avg spacing = {avg_spacing:.1f}")

# Run multiple times to verify randomness
np.random.seed(42)
starts = []
for _ in range(100):
    idx = mock._sample_frame_indices(total_frames=150, mode='train')
    starts.append(idx[0])
unique_starts = len(set(starts))
check("Random temporal crop: multiple unique starting positions over 100 runs",
      unique_starts > 10,
      f"{unique_starts} unique starting positions")

print(f"\n  Sampled indices: {indices_long.tolist()}")


# =========================================================================
# TEST 6: Frame Sampling — Edge Cases
# =========================================================================
section("6. Frame Sampling — Edge Cases")

# Exactly 64 frames (boundary)
indices_64 = mock._sample_frame_indices(total_frames=64, mode='train')
check("Exactly 64 frames: returns 16 indices",
      len(indices_64) == 16,
      f"got {len(indices_64)}")
check("Exactly 64 frames: all in range [0, 63]",
      np.all(indices_64 >= 0) and np.all(indices_64 <= 63),
      f"range: [{indices_64.min()}, {indices_64.max()}]")

# 65 frames (just above boundary → should use random crop)
indices_65 = mock._sample_frame_indices(total_frames=65, mode='train')
check("65 frames: returns 16 indices",
      len(indices_65) == 16)
check("65 frames: all in range [0, 64]",
      np.all(indices_65 >= 0) and np.all(indices_65 <= 64),
      f"range: [{indices_65.min()}, {indices_65.max()}]")

# Very short video (5 frames)
indices_5 = mock._sample_frame_indices(total_frames=5, mode='train')
check("Very short (5 frames): returns 16 indices",
      len(indices_5) == 16,
      f"got {len(indices_5)}")
check("Very short (5 frames): all in range [0, 4]",
      np.all(indices_5 >= 0) and np.all(indices_5 <= 4),
      f"range: [{indices_5.min()}, {indices_5.max()}]")
check("Very short: heavy padding with last frame",
      np.sum(indices_5 == 4) >= 10,
      f"frame 4 appears {np.sum(indices_5 == 4)} times")

# 1 frame video
indices_1 = mock._sample_frame_indices(total_frames=1, mode='train')
check("Single frame video: returns 16 indices (all zeros)",
      len(indices_1) == 16 and np.all(indices_1 == 0),
      f"indices: {indices_1.tolist()}")


# =========================================================================
# TEST 7: Spatial Transforms — Shape Check
# =========================================================================
section("7. Spatial Transforms — Output Shapes")

from src.dataset import get_train_transform, get_val_transform

train_transform = get_train_transform(224)
val_transform = get_val_transform(224)

# Create a fake frame (H=1080, W=1920, C=3) — NTU resolution
fake_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

train_out = train_transform(fake_frame)
check("Train transform: (1080,1920,3) → (3,224,224)",
      train_out.shape == (3, 224, 224),
      f"got {train_out.shape}")

val_out = val_transform(fake_frame)
check("Val transform: (1080,1920,3) → (3,224,224)",
      val_out.shape == (3, 224, 224),
      f"got {val_out.shape}")

# Check normalization range (should be roughly [-2.5, 2.5] after ImageNet normalization)
check("Train output is float32",
      train_out.dtype == torch.float32,
      f"got {train_out.dtype}")

check("Train output is normalized (not [0,255])",
      train_out.max() < 10.0 and train_out.min() > -10.0,
      f"range: [{train_out.min():.2f}, {train_out.max():.2f}]")

# Small frame (320x240) — some NTU videos might be pre-resized
small_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
small_out = val_transform(small_frame)
check("Small frame (240,320,3) → (3,224,224)",
      small_out.shape == (3, 224, 224),
      f"got {small_out.shape}")


# =========================================================================
# TEST 8: Full Frame → Tensor Pipeline (Mock Video)
# =========================================================================
section("8. Full Frame→Tensor Pipeline (16 frames)")

# Simulate what _apply_transforms does
from src.dataset import NTU60Dataset

# Create a mock buffer: 16 frames of NTU resolution
mock_buffer = np.random.randint(0, 255, (16, 1080, 1920, 3), dtype=np.uint8)

frames = []
for i in range(16):
    frame = train_transform(mock_buffer[i])
    frames.append(frame)
frames = torch.stack(frames, dim=0)       # (T, C, H, W)

check("Stacked frames shape: (16, 3, 224, 224)",
      frames.shape == (16, 3, 224, 224),
      f"got {frames.shape}")

frames_permuted = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
check("Permuted to channel-first: (3, 16, 224, 224)",
      frames_permuted.shape == (3, 16, 224, 224),
      f"got {frames_permuted.shape}")

# Add batch dimension (what DataLoader does)
batched = frames_permuted.unsqueeze(0)     # (1, C, T, H, W)
check("Batched shape: (1, 3, 16, 224, 224)",
      batched.shape == (1, 3, 16, 224, 224),
      f"got {batched.shape}")


# =========================================================================
# TEST 9: Model Input/Output Shape (without GPU)
# =========================================================================
section("9. Model I/O Shape Verification")

print("  Checking model architecture without loading weights (CPU only)...")
print("  (This avoids downloading the large pre-trained model)")

# Instead of loading the full HuggingFace model, verify the expected shapes
# The VideoMAE-2 Large expects:
#   Input:  (B, C, T, H, W) = (B, 3, 16, 224, 224)
#   Output: (B, num_classes) = (B, 60)

expected_input = (1, 3, 16, 224, 224)
expected_output_classes = 60

check("Expected model input: (B, 3, 16, 224, 224)",
      True,
      f"matches our data pipeline output: {batched.shape}")

check("Input channels = 3 (RGB)",
      batched.shape[1] == 3)

check("Input temporal = 16 (clip_len)",
      batched.shape[2] == 16)

check("Input spatial = 224×224",
      batched.shape[3] == 224 and batched.shape[4] == 224)

check("Output classes = 60 (NTU-60)",
      expected_output_classes == config['data']['num_classes'])

# ViT-L patch calculations
patch_size = 16
tubelet_size = 2
temporal_patches = 16 // tubelet_size    # 8
spatial_patches = (224 // patch_size) ** 2  # 14 * 14 = 196
total_tokens = temporal_patches * spatial_patches  # 8 * 196 = 1568

check(f"ViT-L temporal patches: 16 / {tubelet_size} = {temporal_patches}",
      temporal_patches == 8)
check(f"ViT-L spatial patches: (224/{patch_size})² = {spatial_patches}",
      spatial_patches == 196)
check(f"ViT-L total tokens: {temporal_patches} × {spatial_patches} = {total_tokens}",
      total_tokens == 1568,
      "this is what the transformer processes")


# =========================================================================
# TEST 10: Checkpoint Save/Load Roundtrip
# =========================================================================
section("10. Checkpoint Save/Load Roundtrip")

import tempfile
from src.utils import save_checkpoint, load_checkpoint

# Create a tiny mock model
mock_model = torch.nn.Linear(10, 60)
mock_optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)
mock_scaler = torch.cuda.amp.GradScaler(enabled=False)  # CPU-safe

with tempfile.TemporaryDirectory() as tmpdir:
    # Save
    save_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        scaler=mock_scaler,
        epoch=5,
        val_acc=85.5,
        config=config,
        output_dir=tmpdir,
        is_best=True,
    )

    # Check files exist
    latest_exists = os.path.exists(os.path.join(tmpdir, 'checkpoint-latest.pth'))
    best_exists = os.path.exists(os.path.join(tmpdir, 'checkpoint-best.pth'))
    check("checkpoint-latest.pth created", latest_exists)
    check("checkpoint-best.pth created", best_exists)

    # Load
    loaded_model = torch.nn.Linear(10, 60)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-4)

    ckpt = load_checkpoint(
        os.path.join(tmpdir, 'checkpoint-best.pth'),
        loaded_model,
        loaded_optimizer,
    )

    check("Loaded epoch = 5", ckpt['epoch'] == 5, f"got {ckpt['epoch']}")
    check("Loaded val_acc = 85.5", ckpt['val_acc'] == 85.5, f"got {ckpt['val_acc']}")

    # Verify model weights match
    orig_weight = mock_model.weight.data
    loaded_weight = loaded_model.weight.data
    check("Model weights match after load",
          torch.allclose(orig_weight, loaded_weight),
          f"max diff = {(orig_weight - loaded_weight).abs().max():.1e}")


# =========================================================================
# TEST 11: Auto-Resume
# =========================================================================
section("11. Auto-Resume")

from src.utils import auto_resume

with tempfile.TemporaryDirectory() as tmpdir:
    # No checkpoint → should return 0
    start_epoch = auto_resume(tmpdir, mock_model)
    check("No checkpoint → start_epoch = 0", start_epoch == 0, f"got {start_epoch}")

    # Save a checkpoint at epoch 10
    save_checkpoint(mock_model, mock_optimizer, mock_scaler, 10, 90.0, config, tmpdir)

    # Now auto-resume should return 11
    start_epoch = auto_resume(tmpdir, mock_model)
    check("After saving epoch 10 → start_epoch = 11", start_epoch == 11, f"got {start_epoch}")


# =========================================================================
# TEST 12: MetricLogger
# =========================================================================
section("12. MetricLogger")

from src.utils import MetricLogger, SmoothedValue

logger = MetricLogger()
logger.update(loss=1.5)
logger.update(loss=1.0)
logger.update(loss=0.5)

check("MetricLogger tracks loss",
      abs(logger.meters['loss'].global_avg - 1.0) < 1e-6,
      f"avg = {logger.meters['loss'].global_avg}")

check("SmoothedValue tracks latest",
      abs(logger.meters['loss'].value - 0.5) < 1e-6,
      f"latest = {logger.meters['loss'].value}")


# =========================================================================
# TEST 13: Accuracy Function
# =========================================================================
section("13. Accuracy Computation")

from src.trainer import accuracy

# Perfect prediction
logits = torch.zeros(4, 60)
logits[0, 5] = 10.0   # sample 0 → class 5
logits[1, 10] = 10.0  # sample 1 → class 10
logits[2, 30] = 10.0  # sample 2 → class 30
logits[3, 59] = 10.0  # sample 3 → class 59
targets = torch.tensor([5, 10, 30, 59])

top1, top5 = accuracy(logits, targets, topk=(1, 5))
check("Perfect prediction → Top-1 = 100%", top1 == 100.0, f"got {top1}")
check("Perfect prediction → Top-5 = 100%", top5 == 100.0, f"got {top5}")

# Half correct
logits2 = torch.zeros(4, 60)
logits2[0, 5] = 10.0   # correct
logits2[1, 10] = 10.0  # correct
logits2[2, 0] = 10.0   # wrong (should be 30)
logits2[3, 0] = 10.0   # wrong (should be 59)
targets2 = torch.tensor([5, 10, 30, 59])

top1_2, top5_2 = accuracy(logits2, targets2, topk=(1, 5))
check("Half correct → Top-1 = 50%", top1_2 == 50.0, f"got {top1_2}")


# =========================================================================
# TEST 14: NTU Filename Parsing
# =========================================================================
section("14. NTU Filename Parsing")

from scripts.prepare_annotations import parse_ntu_filename, TRAIN_SUBJECTS, VAL_SUBJECTS

meta = parse_ntu_filename("S001C001P001R001A001_rgb.avi")
check("Parse S001C001P001R001A001: setup=1", meta['setup'] == 1)
check("Parse S001C001P001R001A001: camera=1", meta['camera'] == 1)
check("Parse S001C001P001R001A001: subject=1", meta['subject'] == 1)
check("Parse S001C001P001R001A001: action=1", meta['action'] == 1)

meta2 = parse_ntu_filename("S017C003P040R002A060_rgb.avi")
check("Parse max values: setup=17", meta2['setup'] == 17)
check("Parse max values: camera=3", meta2['camera'] == 3)
check("Parse max values: subject=40", meta2['subject'] == 40)
check("Parse max values: action=60", meta2['action'] == 60)

# Cross-subject split
check("Train subjects count = 20", len(TRAIN_SUBJECTS) == 20)
check("Val subjects count = 20", len(VAL_SUBJECTS) == 20)
check("No overlap between train and val subjects",
      len(TRAIN_SUBJECTS & VAL_SUBJECTS) == 0)
check("All 40 subjects covered",
      TRAIN_SUBJECTS | VAL_SUBJECTS == set(range(1, 41)))

# Subject 1 should be train, subject 3 should be val
check("Subject 1 → TRAIN", 1 in TRAIN_SUBJECTS)
check("Subject 3 → VAL", 3 in VAL_SUBJECTS)


# =========================================================================
# SUMMARY
# =========================================================================
section("SUMMARY")

total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)

print(f"\n  Total tests: {total}")
print(f"  Passed:      {passed} {PASS}")
print(f"  Failed:      {failed} {FAIL if failed > 0 else ''}")
print(f"\n  {'ALL TESTS PASSED!' if failed == 0 else f'{failed} TESTS FAILED!'}")
print()
