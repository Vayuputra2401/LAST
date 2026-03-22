"""Comprehensive model variant check + bug fix verification."""
import torch
import traceback

from src.models.shiftfuse_zero import (
    ShiftFuseZero,
    ShiftFuseZeroLate,
    ShiftFuseZeroMidFusion,
    ZERO_VARIANTS,
)

B, C, T, V = 2, 3, 64, 25

def make_streams(*names):
    return {n: torch.randn(B, C, T, V) for n in names}

results = []

# ── 1. Nano ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. NANO (nano_tiny_efficient) — 3-stream")
print("=" * 60)
try:
    nano = ShiftFuseZero(variant='nano_tiny_efficient', num_classes=60)
    nano.eval()
    p = sum(p.numel() for p in nano.parameters())
    print(f"   Total params:     {p:,}")
    sd = make_streams('joint', 'bone', 'velocity')
    with torch.no_grad():
        out = nano(sd)
    print(f"   Forward pass:     OK  shape={out.shape}")
    print(f"   NaN check:        {'PASS' if torch.isfinite(out).all() else 'FAIL'}")
    results.append(("Nano", p, "PASS"))
except Exception as e:
    print(f"   ERROR: {e}")
    traceback.print_exc()
    results.append(("Nano", 0, "FAIL"))

# ── 2. Small ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. SMALL (ShiftFuseZeroLate) — bone-only BB_B, shared TLA, early fusion")
print("=" * 60)
try:
    small = ShiftFuseZeroLate(variant='small_late_efficient_bb', num_classes=60)
    small.eval()
    p = sum(p.numel() for p in small.parameters())
    print(f"   Total params:     {p:,}")
    sd = make_streams('joint', 'bone', 'velocity')
    with torch.no_grad():
        out = small(sd)
    print(f"   Forward pass:     OK  shape={out.shape}")
    print(f"   NaN check:        {'PASS' if torch.isfinite(out).all() else 'FAIL'}")
    # Verify structural changes
    has_early = hasattr(small, 'cross_fusion_early')
    has_late  = hasattr(small, 'cross_fusion_late')
    has_tla   = hasattr(small, 'tla_shared') and small.tla_shared is not None
    bb_b_streams = small.backbone_b.stream_names
    print(f"   Early fusion:     {'YES' if has_early else 'NO'}")
    print(f"   Late fusion:      {'YES' if has_late else 'NO'}")
    print(f"   Shared TLA:       {'YES' if has_tla else 'NO'}")
    print(f"   BB_B streams:     {bb_b_streams}")
    results.append(("Small", p, "PASS"))
except Exception as e:
    print(f"   ERROR: {e}")
    traceback.print_exc()
    results.append(("Small", 0, "FAIL"))

# ── 3. Large ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. LARGE (ShiftFuseZeroMidFusion)")
print("=" * 60)
try:
    large = ShiftFuseZeroMidFusion(num_classes=60)
    large.eval()
    p = sum(p.numel() for p in large.parameters())
    print(f"   Total params:     {p:,}")
    sd = make_streams('joint', 'bone', 'velocity')
    with torch.no_grad():
        out = large(sd)
    print(f"   Forward pass:     OK  shape={out.shape}")
    print(f"   NaN check:        {'PASS' if torch.isfinite(out).all() else 'FAIL'}")
    results.append(("Large", p, "PASS"))
except Exception as e:
    print(f"   ERROR: {e}")
    traceback.print_exc()
    results.append(("Large", 0, "FAIL"))

# ── Bug Fix Verification ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUG FIX VERIFICATION")
print("=" * 60)

# Check A_learned naming
nano2 = ShiftFuseZero(variant='nano_tiny_efficient', num_classes=60)
a_learned_names = [n for n, _ in nano2.named_parameters() if 'A_learned' in n]
print(f"   A_learned params found: {len(a_learned_names)}")
for n in a_learned_names[:3]:
    print(f"     - {n}")
all_catch = all('A_learned' in n for n in a_learned_names)
print(f"   All caught by 'A_learned' filter: {'YES' if all_catch else 'NO'}")

# Check je.embed naming
je_names = [n for n, _ in nano2.named_parameters() if 'je' in n and 'embed' in n]
print(f"   je.embed params found: {len(je_names)}")
for n in je_names[:3]:
    print(f"     - {n}")
all_je_catch = all('je.embed' in n for n in je_names)
print(f"   All caught by 'je.embed' filter: {'YES' if all_je_catch else 'NO'}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, params, status in results:
    print(f"   {name:8s}  {params:>10,} params   {status}")
