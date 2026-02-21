"""
Integration tests for LAST-v2 and LAST-E models.

Tests forward pass shapes, parameter counts, and MIB dict interface.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.models.last_v2 import LAST_v2
from src.models.last_e import (
    LAST_E,
    create_last_e_nano,
    create_last_e_small,
    create_last_e_base,
    create_last_e_large,
)


def _make_mib(batch=2, C=3, T=64, V=25):
    """Create a MIB dict with 4D tensors (B, C, T, V)."""
    return {
        'joint':    torch.randn(batch, C, T, V),
        'velocity': torch.randn(batch, C, T, V),
        'bone':     torch.randn(batch, C, T, V),
    }


def _param_breakdown(model):
    """Print a detailed param breakdown per component."""
    sections = {
        'fusion':  model.fusion,
        'stage1':  model.stage1,
        'stage2':  model.stage2,
        'stage3':  model.stage3,
        'fc':      model.fc,
    }
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  {'Section':<10} {'Params':>10}  {'%':>6}")
    print(f"  {'-'*30}")
    for name, mod in sections.items():
        p = sum(x.numel() for x in mod.parameters())
        print(f"  {name:<10} {p:>10,}  {100*p/total:>5.1f}%")
    print(f"  {'-'*30}")
    print(f"  {'TOTAL':<10} {total:>10,}")

    # Per-block breakdown for stage3
    print(f"\n  Stage3 per-block breakdown:")
    for i, blk in enumerate(model.stage3):
        gcn_p = sum(x.numel() for x in blk.gcn.parameters())
        tcn_p = sum(x.numel() for x in blk.tcn.parameters())
        att_p = sum(x.numel() for x in blk.st_att.parameters()) if blk.st_att else 0
        res_p = sum(x.numel() for x in blk.residual_path.parameters()) \
                if hasattr(blk.residual_path, 'parameters') else 0
        block_total = gcn_p + tcn_p + att_p + res_p
        print(f"    block {i}: total={block_total:,}  gcn={gcn_p:,}  "
              f"tcn={tcn_p:,}  att={att_p:,}  res={res_p:,}")
    return total


# ── LAST-v2 Tests ─────────────────────────────────────────────────────────────

class TestLASTv2:
    def test_base_forward(self):
        model = LAST_v2(num_classes=60, variant='base')
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60), f"Expected (2,60), got {out.shape}"
        assert not torch.isnan(out).any()

    def test_small_forward(self):
        model = LAST_v2(num_classes=60, variant='small')
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60)

    def test_single_stream(self):
        model = LAST_v2(num_classes=60, variant='small')
        model.eval()
        x = torch.randn(2, 3, 64, 25)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60)

    def test_ntu120(self):
        model = LAST_v2(num_classes=120, variant='small')
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 120)

    def test_count_parameters(self):
        model = LAST_v2(num_classes=60, variant='base')
        params = model.count_parameters()
        assert params > 0
        print(f"\nLAST-v2-Base params: {params:,}")


# ── LAST-E Tests ──────────────────────────────────────────────────────────────

class TestLASTE:
    def test_nano_forward(self):
        model = create_last_e_nano(num_classes=60)
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60), f"Expected (2,60), got {out.shape}"
        assert not torch.isnan(out).any()

    def test_base_forward(self):
        model = create_last_e_base(num_classes=60)
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60), f"Expected (2,60), got {out.shape}"
        assert not torch.isnan(out).any()

    def test_small_forward(self):
        model = create_last_e_small(num_classes=60)
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60)

    def test_large_forward(self):
        model = create_last_e_large(num_classes=60)
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 60)

    def test_ntu120(self):
        model = create_last_e_base(num_classes=120)
        model.eval()
        x = _make_mib()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 120)

    def test_multiscale_tcn_output_shape(self):
        """Both TCN branches must produce same T_out — concat cannot be mismatched."""
        from src.models.blocks.light_gcn import MultiScaleTCN
        for C in [24, 32, 40, 48, 64, 96, 128, 160, 192]:
            for stride in [1, 2]:
                tcn = MultiScaleTCN(C, stride=stride)
                x = torch.randn(2, C, 64, 25)
                out = tcn(x)
                expected_T = 64 // stride
                assert out.shape == (2, C, expected_T, 25), \
                    f"C={C} stride={stride}: expected (2,{C},{expected_T},25) got {out.shape}"

    def test_directional_gcn_subset_buffers(self):
        """K separate adjacency buffers must all exist and be non-zero."""
        from src.models.blocks.light_gcn import DirectionalGCNConv
        from src.models.graph import Graph
        import torch
        graph = Graph(layout='ntu-rgb+d', strategy='spatial')
        A = torch.tensor(graph.A, dtype=torch.float32)
        K = A.shape[0]
        gcn = DirectionalGCNConv(40, 40, A)
        for k in range(K):
            buf = getattr(gcn, f'A_{k}')
            assert buf.shape == (25, 25), f"A_{k} wrong shape: {buf.shape}"
            assert buf.abs().sum() > 0,   f"A_{k} is all-zeros"
        # alpha init: softmax should give near-1/K for each channel
        alpha_w = torch.softmax(gcn.alpha, dim=0)
        assert alpha_w.shape == (K, 40)
        assert (alpha_w - 1.0/K).abs().max() < 1e-5, "alpha not uniform at init"

    def test_no_nan_gradients(self):
        model = create_last_e_base(num_classes=60)
        model.train()
        x = _make_mib()
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            LAST_E(num_classes=60, variant='xlarge')

    # ── Param count tests — updated bounds for v2 architecture ────────────────

    def test_nano_param_count(self):
        model = create_last_e_nano(num_classes=60)
        params = _param_breakdown(model)
        print(f"\nLAST-E-Nano  total: {params:,}  (EfficientGCN-B0 target: <150K)")
        # v2: ~91K after TCN savings + directional GCN additions
        assert 70_000 < params < 130_000, f"Unexpected param count: {params:,}"

    def test_small_param_count(self):
        model = create_last_e_small(num_classes=60)
        params = _param_breakdown(model)
        print(f"\nLAST-E-Small total: {params:,}  (EfficientGCN-B1 target: <300K)")
        # v2: ~177K
        assert 130_000 < params < 250_000, f"Unexpected param count: {params:,}"

    def test_base_param_count(self):
        model = create_last_e_base(num_classes=60)
        params = _param_breakdown(model)
        print(f"\nLAST-E-Base  total: {params:,}  (EfficientGCN-B4 target: <2M)")
        # v2: ~364K
        assert 250_000 < params < 500_000, f"Unexpected param count: {params:,}"

    def test_large_param_count(self):
        model = create_last_e_large(num_classes=60)
        params = _param_breakdown(model)
        print(f"\nLAST-E-Large total: {params:,}  (EfficientGCN-B4 target: <2M)")
        # v2: ~645K
        assert 500_000 < params < 800_000, f"Unexpected param count: {params:,}"


if __name__ == '__main__':
    print("=" * 60)
    print("LAST-v2 tests")
    print("=" * 60)
    t = TestLASTv2()
    t.test_base_forward()
    t.test_small_forward()
    t.test_single_stream()
    t.test_ntu120()
    t.test_count_parameters()
    print("\nLAST-v2 tests passed.")

    print()
    print("=" * 60)
    print("LAST-E tests")
    print("=" * 60)
    t = TestLASTE()

    print("\n--- Forward passes ---")
    t.test_nano_forward()
    print("  nano  OK")
    t.test_base_forward()
    print("  base  OK")
    t.test_small_forward()
    print("  small OK")
    t.test_large_forward()
    print("  large OK")
    t.test_ntu120()
    print("  ntu120 OK")

    print("\n--- Architecture sanity checks ---")
    t.test_multiscale_tcn_output_shape()
    print("  MultiScaleTCN output shapes OK (all C/stride combos)")
    t.test_directional_gcn_subset_buffers()
    print("  DirectionalGCNConv buffers OK (K buffers, uniform alpha init)")

    print("\n--- Gradient sanity ---")
    t.test_no_nan_gradients()
    print("  No NaN gradients OK")

    print("\n--- Param counts ---")
    t.test_nano_param_count()
    t.test_small_param_count()
    t.test_base_param_count()
    t.test_large_param_count()

    print()
    print("=" * 60)
    print("ALL LAST-E TESTS PASSED")
    print("=" * 60)
