"""
Integration tests for ShiftFuse-GCN (LAST-Lite v9 CleanFuse).

Covers:
  - Forward pass shape: dict input → (B, num_classes)  [eval]
  - Training forward: returns ([logit_j, logit_v, logit_b, logit_bv], ib_loss)
  - Tensor (non-dict) input handling
  - 5D input handling (B, C, T, V, M) → takes M=0 body
  - Param count reporting (no hard budget limit in v9 — param-free design)
  - BodyRegionShift: zero learnable parameters
  - FrozenDCTGate: dct/idct buffers are not trainable (require_grad=False)
  - JointEmbedding: correct output shape (no-op at zero-init)
  - FrameDynamicsGate: correct output shape
  - CTRLightGCN: output shape, param count, per-group adjacency parameters
  - MultiScaleAdaptiveGCN: K-subset separate buffers, output shape, param count
  - BilateralSymmetryEncoding: antisymmetric injection, torso unchanged
  - DropPath: stochastic depth (0 params, identity at eval)
  - pw_conv conditional: Identity when in==out, Sequential when channels differ
  - TemporalAttention gate init: 0.0 (sigmoid(0)=0.5, active from epoch 1)
  - All individual blocks: input → output shape preserved
  - Dry run: data shape trace through entire model
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

from src.models.shiftfuse_gcn import (
    LAST_Lite,
    create_shiftfuse_nano,
    create_shiftfuse_small,
)
from src.models.blocks.body_region_shift import BodyRegionShift
from src.models.blocks.frozen_dct_gate import FrozenDCTGate
from src.models.blocks.joint_embedding import JointEmbedding
from src.models.blocks.frame_dynamics_gate import FrameDynamicsGate
from src.models.blocks.bilateral_symmetry import BilateralSymmetryEncoding
from src.models.blocks.static_gcn import StaticGCN
from src.models.blocks.ctr_light_gcn import CTRLightGCN
from src.models.blocks.adaptive_ctr_gcn import MultiScaleAdaptiveGCN
from src.models.blocks.ep_sep_tcn import MultiScaleEpSepTCN
from src.models.blocks.drop_path import DropPath
from src.models.graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, C, T, V = 2, 3, 64, 25
NUM_CLASSES = 60


def _mib(batch=B, c=C, t=T, v=V):
    return {
        'joint':         torch.randn(batch, c, t, v),
        'velocity':      torch.randn(batch, c, t, v),
        'bone':          torch.randn(batch, c, t, v),
        'bone_velocity': torch.randn(batch, c, t, v),
    }


# ---------------------------------------------------------------------------
# Block-level tests
# ---------------------------------------------------------------------------

class TestBodyRegionShift:
    def _make_shift(self, channels=32):
        from src.models.graph import Graph
        import numpy as np
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A_flat = torch.tensor((g.A.sum(0) > 0).astype('float32'))
        return BodyRegionShift(channels, A_flat)

    def test_zero_params(self):
        shift = self._make_shift(32)
        n_params = sum(p.numel() for p in shift.parameters())
        assert n_params == 0, f"BodyRegionShift should have 0 params, got {n_params}"

    def test_output_shape(self):
        shift = self._make_shift(32)
        x = torch.randn(B, 32, T, V)
        out = shift(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_shift_indices_buffer_no_grad(self):
        shift = self._make_shift(32)
        assert not shift.shift_indices.requires_grad


class TestFrozenDCTGate:
    def test_output_shape(self):
        gate = FrozenDCTGate(channels=48, T=T)
        x = torch.randn(B, 48, T, V)
        out = gate(x)
        assert out.shape == x.shape

    def test_dct_buffers_no_grad(self):
        gate = FrozenDCTGate(channels=48, T=T)
        assert not gate.dct.requires_grad
        assert not gate.idct.requires_grad

    def test_freq_mask_is_param(self):
        gate = FrozenDCTGate(channels=48, T=T)
        assert gate.freq_mask.requires_grad

    def test_param_count(self):
        C_test, T_test = 48, 64
        gate = FrozenDCTGate(channels=C_test, T=T_test)
        n = sum(p.numel() for p in gate.parameters())
        assert n == C_test * T_test, f"Expected {C_test * T_test}, got {n}"

    def test_dct_near_identity_at_init(self):
        # Init +4.0 → sigmoid(4.0) ≈ 0.982 → output ≈ 0.982x (near-identity, no internal residual)
        gate = FrozenDCTGate(channels=8, T=16)
        x = torch.randn(1, 8, 16, 4)
        out = gate(x)
        # Output should be close to x (within 5%) since mask ≈ 0.982
        rel_err = (out - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.05, f"Expected near-identity at init, rel_err={rel_err:.4f}"


class TestJointEmbedding:
    def test_output_shape(self):
        embed = JointEmbedding(channels=48, num_joints=V)
        x = torch.randn(B, 48, T, V)
        out = embed(x)
        assert out.shape == x.shape

    def test_zero_init_is_identity(self):
        embed = JointEmbedding(channels=16, num_joints=V)
        x = torch.randn(B, 16, T, V)
        out = embed(x)
        assert torch.allclose(out, x), "Zero-init embed should not change input"

    def test_param_count(self):
        C_test = 48
        embed = JointEmbedding(channels=C_test, num_joints=V)
        n = sum(p.numel() for p in embed.parameters())
        assert n == V * C_test, f"Expected {V * C_test}, got {n}"


class TestStaticGCN:
    def _make_gcn(self, channels=48):
        g   = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A   = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        return StaticGCN(channels=channels, A=A, num_joints=V)

    def test_output_shape(self):
        gcn = self._make_gcn(48)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_param_count(self):
        C_test = 48
        gcn = self._make_gcn(C_test)
        n   = sum(p.numel() for p in gcn.parameters())
        expected = C_test * C_test + 2 * C_test + V * V   # conv + BN + A_learned
        assert n == expected, f"Expected {expected}, got {n}"

    def test_near_identity_at_zero_init(self):
        # A_learned = zeros → A_learned_norm = 0 → x_agg comes only from static A
        # After static aggregation and zero-init conv, BN(0) = 0 → out ≈ x (residual)
        gcn = self._make_gcn(16)
        # Freeze conv weight to zero to isolate the zero-init A_learned effect
        with torch.no_grad():
            gcn.conv.weight.zero_()
        x   = torch.randn(1, 16, T, V)
        out = gcn(x)
        # With conv=0: BN(0) → ~0, so out ≈ x + 0 = x
        assert torch.allclose(out, x, atol=1e-4), "Expected near-identity with zero conv"

    def test_A_learned_is_parameter(self):
        gcn = self._make_gcn(32)
        assert gcn.A_learned.requires_grad, "A_learned must be a trainable parameter"

    def test_A_buffer_no_grad(self):
        gcn = self._make_gcn(32)
        assert not gcn.A.requires_grad, "Static A must be a buffer (no grad)"


class TestCTRLightGCN:
    def _make_gcn(self, channels=48, num_groups=4):
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        return CTRLightGCN(channels=channels, A=A, num_joints=V, num_groups=num_groups)

    def test_output_shape(self):
        gcn = self._make_gcn(48, num_groups=4)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_param_count(self):
        C_test, G = 48, 4
        gcn = self._make_gcn(C_test, G)
        n   = sum(p.numel() for p in gcn.parameters())
        # G × (C//G)² conv + 2C BN + G × V² adjacency
        expected = G * (C_test // G) ** 2 + 2 * C_test + G * V * V
        assert n == expected, f"Expected {expected}, got {n}"

    def test_A_group_is_parameter(self):
        gcn = self._make_gcn(48, 4)
        for i, ag in enumerate(gcn.A_group):
            assert ag.requires_grad, f"A_group[{i}] must be trainable"

    def test_A_physical_buffer_no_grad(self):
        gcn = self._make_gcn(48, 4)
        assert not gcn.A_physical.requires_grad, "A_physical must be a buffer"

    def test_nano_groups(self):
        gcn = self._make_gcn(32, num_groups=2)
        x   = torch.randn(B, 32, T, V)
        out = gcn(x)
        assert out.shape == x.shape

    def test_A_g_normalised(self):
        """Row-normalisation: A_physical is normalised at __init__ (not per-forward).
        This avoids the unstable normalisation Jacobian that caused NaN explosion."""
        gcn = self._make_gcn(32, num_groups=2)
        # A_physical should have row sums = 1.0 immediately after construction
        row_sums = gcn.A_physical.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(V), atol=1e-5), \
            f"A_physical row sums not ≈ 1.0 at init: {row_sums}"
        # At init A_group = 0 → A_g = A_physical, so A_g is also unit-scale
        with torch.no_grad():
            A_g = gcn.A_physical + gcn.A_group[0]
        assert torch.allclose(A_g.sum(dim=-1), torch.ones(V), atol=1e-5), \
            "A_g row sums not ≈ 1.0 at init (A_group should be zero)"


class TestMultiScaleAdaptiveGCN:
    """Tests for K-subset MultiScaleAdaptiveGCN (v7 replacement for AdaptiveCTRGCN)."""

    def _make_gcn(self, channels=48, num_groups=4):
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        return MultiScaleAdaptiveGCN(
            channels=channels, A=A, num_joints=V, num_groups=num_groups
        )

    def test_output_shape(self):
        gcn = self._make_gcn(48, 4)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_k_separate_buffers(self):
        """K=3 subsets must be stored as separate buffers A_0, A_1, A_2."""
        gcn = self._make_gcn(48, 4)
        assert hasattr(gcn, 'A_0'), "MultiScaleAdaptiveGCN must have buffer A_0"
        assert hasattr(gcn, 'A_1'), "MultiScaleAdaptiveGCN must have buffer A_1"
        assert hasattr(gcn, 'A_2'), "MultiScaleAdaptiveGCN must have buffer A_2"
        # Buffers must NOT require grad
        assert not gcn.A_0.requires_grad
        assert not gcn.A_1.requires_grad
        assert not gcn.A_2.requires_grad

    def test_k_subsets_row_normalised(self):
        gcn = self._make_gcn(48, 4)
        for k in range(gcn.K):
            Ak = getattr(gcn, f'A_{k}')
            row_sums = Ak.sum(dim=-1)
            assert (row_sums <= 1.0 + 1e-4).all(), \
                f"A_{k} row sums exceed 1.0"

    def test_k_independent_convs(self):
        """K independent group convolutions — one per subset."""
        gcn = self._make_gcn(48, 4)
        assert len(gcn.convs) == gcn.K, \
            f"Expected {gcn.K} convs, got {len(gcn.convs)}"

    def test_alpha_is_parameter(self):
        gcn = self._make_gcn(48, 4)
        assert gcn.alpha.requires_grad, "alpha must be a trainable parameter"
        assert gcn.alpha.shape == (1,)

    def test_backwards_compat_alias(self):
        """AdaptiveCTRGCN alias must still import and instantiate."""
        from src.models.blocks.adaptive_ctr_gcn import AdaptiveCTRGCN
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        gcn = AdaptiveCTRGCN(channels=48, A=A, num_joints=V)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape


class TestMultiScaleEpSepTCN:
    """Tests for 4-branch dilated TCN (v8: TSM + d=2 + d=4 + MaxPool)."""

    def test_output_shape_stride1(self):
        tcn = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert out.shape == (B, 48, T, V), f"Expected (B,48,T,V), got {out.shape}"

    def test_output_shape_stride2(self):
        tcn = MultiScaleEpSepTCN(channels=48, stride=2, num_branches=4)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert out.shape == (B, 48, T // 2, V), f"Expected T//2, got {out.shape}"

    def test_tsm_zero_params(self):
        """TSM branch contributes 0 learnable params (only BN shared in tsm_downsample)."""
        tcn = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        # tsm_downsample is BN(C//4) = 2*(48//4) = 24 params (gamma+beta)
        tsm_params = sum(p.numel() for p in tcn.tsm_downsample.parameters())
        assert tsm_params == 2 * (48 // 4), f"TSM BN params: {tsm_params}"

    def test_4branch_fewer_params_than_3branch(self):
        """4-branch (TSM replaces d=1 conv) should have fewer params than 3-branch legacy."""
        t4 = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        t3 = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=3)
        n4 = sum(p.numel() for p in t4.parameters())
        n3 = sum(p.numel() for p in t3.parameters())
        assert n4 < n3, f"4-branch ({n4}) should use fewer params than 3-branch ({n3})"

    def test_dilation4_receptive_field(self):
        """d=4 branch must produce non-zero output (actual computation)."""
        tcn = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert not torch.allclose(out, torch.zeros_like(out)), "Output should be non-zero"

    def test_channels_div4_assertion(self):
        """Should raise AssertionError when channels not divisible by 4."""
        import pytest
        with pytest.raises(AssertionError):
            MultiScaleEpSepTCN(channels=50, stride=1, num_branches=4)


class TestDropPath:
    def test_output_shape(self):
        dp  = DropPath(drop_prob=0.1)
        x   = torch.randn(B, 48, T, V)
        dp.train()
        out = dp(x)
        assert out.shape == x.shape

    def test_zero_params(self):
        dp = DropPath(0.1)
        n  = sum(p.numel() for p in dp.parameters())
        assert n == 0, f"DropPath should have 0 params, got {n}"

    def test_identity_at_eval(self):
        dp = DropPath(drop_prob=0.99)  # would drop almost everything in train
        dp.eval()
        x   = torch.randn(B, 48, T, V)
        out = dp(x)
        assert torch.allclose(out, x), "DropPath should be identity at eval"

    def test_identity_when_prob_zero(self):
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x   = torch.randn(B, 48, T, V)
        out = dp(x)
        assert torch.allclose(out, x), "DropPath(0.0) should be identity in train too"


class TestFrameDynamicsGate:
    def test_output_shape(self):
        gate = FrameDynamicsGate(channels=48, T=T)
        x = torch.randn(B, 48, T, V)
        out = gate(x)
        assert out.shape == x.shape

    def test_init_near_identity(self):
        gate = FrameDynamicsGate(channels=16, T=T)
        x = torch.ones(1, 16, T, V)
        out = gate(x)
        # Init 2.0 → sigmoid(2.0) ≈ 0.880 → near-identity (12% suppression)
        expected_gate = torch.sigmoid(torch.tensor(2.0))
        assert torch.allclose(out, x * expected_gate, atol=1e-5)

    def test_param_count(self):
        C_test, T_test = 48, 64
        gate = FrameDynamicsGate(channels=C_test, T=T_test)
        n = sum(p.numel() for p in gate.parameters())
        assert n == C_test * T_test, f"Expected {C_test * T_test}, got {n}"


class TestBilateralSymmetryEncoding:
    def test_output_shape(self):
        bse = BilateralSymmetryEncoding(channels=48)
        x = torch.randn(B, 48, T, V)
        out = bse(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_param_count(self):
        C_test = 48
        bse = BilateralSymmetryEncoding(channels=C_test)
        n = sum(p.numel() for p in bse.parameters())
        expected = 2 * C_test + 1  # sym_weight + sym_vel_weight + gate
        assert n == expected, f"Expected {expected}, got {n}"

    def test_near_identity_at_init(self):
        bse = BilateralSymmetryEncoding(channels=16)
        x = torch.randn(B, 16, T, V)
        out = bse(x)
        # gate=sigmoid(-2)≈0.12 and weights=0 → mod≈0 → out≈x
        assert torch.allclose(out, x, atol=1e-6), "BSE should be near-identity at init"

    def test_antisymmetric_injection(self):
        bse = BilateralSymmetryEncoding(channels=8)
        # Set non-zero weights to test antisymmetry
        with torch.no_grad():
            bse.sym_weight.fill_(1.0)
            bse.gate.fill_(10.0)  # sigmoid(10)≈1
        x = torch.randn(1, 8, 16, V)
        out = bse(x)
        # Check: modification to left joints = -modification to right joints
        left_mod = out[:, :, :, bse.LEFT_JOINTS] - x[:, :, :, bse.LEFT_JOINTS]
        right_mod = out[:, :, :, bse.RIGHT_JOINTS] - x[:, :, :, bse.RIGHT_JOINTS]
        assert torch.allclose(left_mod, -right_mod, atol=1e-5), \
            "BSE should inject antisymmetrically"

    def test_torso_unchanged(self):
        bse = BilateralSymmetryEncoding(channels=8)
        with torch.no_grad():
            bse.sym_weight.fill_(1.0)
            bse.gate.fill_(10.0)
        x = torch.randn(1, 8, 16, V)
        out = bse(x)
        torso = [0, 1, 2, 3, 20]
        assert torch.allclose(out[:, :, :, torso], x[:, :, :, torso]), \
            "BSE should not modify torso/midline joints"


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestLASTLite:
    def test_nano_forward_dict(self):
        model = create_shiftfuse_nano(NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert out.shape == (B, NUM_CLASSES), f"Expected ({B},{NUM_CLASSES}), got {out.shape}"

    def test_small_forward_dict(self):
        model = create_shiftfuse_small(NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert out.shape == (B, NUM_CLASSES)

    def test_tensor_input(self):
        model = create_shiftfuse_small(NUM_CLASSES)
        model.eval()
        x = torch.randn(B, C, T, V)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, NUM_CLASSES)

    def test_5d_input(self):
        # (B, C, T, V, M=2) — should take M=0 body
        model = create_shiftfuse_small(NUM_CLASSES)
        model.eval()
        x = torch.randn(B, C, T, V, 2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, NUM_CLASSES)

    def test_5d_dict_input(self):
        model = create_shiftfuse_small(NUM_CLASSES)
        model.eval()
        x = {k: torch.randn(B, C, T, V, 2) for k in ['joint', 'velocity', 'bone', 'bone_velocity']}
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, NUM_CLASSES)

    def test_nano_param_count_report(self):
        """v9: no hard param budget — just verify model builds and count is positive."""
        model = create_shiftfuse_nano(NUM_CLASSES)
        n = model.count_parameters()
        assert n > 0, "nano should have at least some parameters"
        print(f"\n  nano params: {n:,}")

    def test_small_param_count_report(self):
        """v9: no hard param budget — just verify model builds and count is positive."""
        model = create_shiftfuse_small(NUM_CLASSES)
        n = model.count_parameters()
        assert n > 0, "small should have at least some parameters"
        print(f"\n  small params: {n:,}")

    def test_pw_conv_is_identity_within_stage(self):
        """Within-stage blocks (same channels, stride=1) must have Identity pw_conv."""
        model = create_shiftfuse_small(NUM_CLASSES)
        # small blocks=[1,2,3]; stage 1 has 1 block (transition), stage 2 has 2 blocks
        # Stage 2 block index 1 (second block, in==out==channels[1], stride=1)
        stage2_blocks = model.stages[1]  # ModuleList
        if len(stage2_blocks) > 1:
            blk = stage2_blocks[1]
            assert isinstance(blk.pw_conv, torch.nn.Identity), \
                "Within-stage block (in==out) must have Identity pw_conv"

    def test_pw_conv_is_conv_at_stage_transition(self):
        """Stage-transition blocks (in≠out) must have Sequential pw_conv (Conv+BN+Act)."""
        model = create_shiftfuse_small(NUM_CLASSES)
        # Stage 1 block 0: stem_channels → channels[0] (24→64), always a transition
        blk = model.stages[0][0]
        assert isinstance(blk.pw_conv, torch.nn.Sequential), \
            "Transition block (in≠out) must have Sequential pw_conv"

    def test_temporal_attn_gate_init(self):
        """TemporalAttention gate must be initialized to 0.0 (sigmoid(0)=0.5)."""
        from src.models.blocks.temporal_attention import LightweightTemporalAttention
        attn = LightweightTemporalAttention(channels=64)
        assert float(attn.gate.item()) == 0.0, \
            f"Gate must be 0.0 at init, got {attn.gate.item()}"

    def test_train_forward_multihead(self):
        """Training forward returns (list of 4 logit tensors, ib_loss scalar)."""
        model = create_shiftfuse_small(NUM_CLASSES)
        model.train()
        logits, ib_loss = model(_mib())
        assert isinstance(logits, list), "Training output[0] must be a list"
        assert len(logits) == 4, f"Expected 4 heads, got {len(logits)}"
        for i, l in enumerate(logits):
            assert l.shape == (B, NUM_CLASSES), \
                f"Head {i}: expected ({B},{NUM_CLASSES}), got {l.shape}"
        assert isinstance(ib_loss, torch.Tensor), "ib_loss must be a tensor"
        assert ib_loss.ndim == 0, "ib_loss must be a scalar"

    def test_stream_weights_param(self):
        """stream_weights must exist as a trainable parameter with shape (4,)."""
        model = create_shiftfuse_nano(NUM_CLASSES)
        assert hasattr(model, 'stream_weights'), "Model must have stream_weights"
        assert model.stream_weights.shape == (4,)
        assert model.stream_weights.requires_grad

    def test_no_nan_output(self):
        model = create_shiftfuse_small(NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_variant_invalid(self):
        try:
            LAST_Lite(num_classes=60, variant='invalid_variant')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Main — dry run with full shape trace
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    SEP = '=' * 60

    # ── Param report ────────────────────────────────────────────────
    print(SEP)
    print('PARAM COUNT (v9: CleanFuse — EfficientGCN B0 channel ratio, no param limit)')
    print(SEP)
    nano  = create_shiftfuse_nano(60)
    small = create_shiftfuse_small(60)
    print(f'  nano  params: {nano.count_parameters():>10,}  (channels [32,64,128], 1:2:4 ratio)')
    print(f'  small params: {small.count_parameters():>10,}  (channels [64,128,256], 1:2:4 ratio)')
    print()

    # ── Block-level param breakdown ─────────────────────────────────
    print(SEP)
    print('BLOCK PARAM BREAKDOWN (small variant, stage channels=[64,128,256])')
    print(SEP)
    g     = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
    A_sym = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
    A_flat= torch.tensor((g.A.sum(0) > 0).astype('float32'))
    C_ex, T_ex, V_ex = 48, 64, 25

    shift = BodyRegionShift(C_ex, A_flat)
    dct_g = FrozenDCTGate(C_ex, T_ex)
    je    = JointEmbedding(C_ex, V_ex)
    bse   = BilateralSymmetryEncoding(C_ex)
    fg    = FrameDynamicsGate(C_ex, T_ex)
    sgcn  = StaticGCN(C_ex, A_sym, V_ex)

    print(f'  BodyRegionShift   : {sum(p.numel() for p in shift.parameters()):>8,}  (expected 0)')
    print(f'  FrozenDCTGate     : {sum(p.numel() for p in dct_g.parameters()):>8,}  (expected {C_ex*T_ex})')
    print(f'  JointEmbedding    : {sum(p.numel() for p in je.parameters()):>8,}  (expected {V_ex*C_ex})')
    print(f'  BSE               : {sum(p.numel() for p in bse.parameters()):>8,}  (expected {2*C_ex+1})')
    print(f'  FrameDynamicsGate : {sum(p.numel() for p in fg.parameters()):>8,}  (expected {C_ex*T_ex})')
    print(f'  StaticGCN (C=48)  : {sum(p.numel() for p in sgcn.parameters()):>8,}  (expected {C_ex*C_ex + 2*C_ex + V_ex*V_ex})')
    print()

    # ── Input/output shape trace ─────────────────────────────────────
    print(SEP)
    print('SHAPE TRACE — small variant, B=2, C=3, T=64, V=25')
    print(SEP)
    B_ex = 2
    streams_in = {k: torch.randn(B_ex, 3, T_ex, V_ex)
                  for k in ['joint', 'velocity', 'bone', 'bone_velocity']}

    print(f'  Input  (per stream): {streams_in["joint"].shape}  x4 streams')

    small.eval()
    # Hook to capture intermediate shapes
    shapes = {}
    def make_hook(name):
        def hook(module, inp, out):
            shapes[name] = tuple(out.shape) if isinstance(out, torch.Tensor) else '(tuple)'
        return hook

    small.fusion.register_forward_hook(make_hook('MultiStreamStem'))
    # Stage hooks
    for si, stage in enumerate(small.stages):
        for bi, blk in enumerate(stage):
            blk.register_forward_hook(make_hook(f'Stage{si+1}_Block{bi+1}'))
        small.stage_gcns[si].register_forward_hook(make_hook(f'Stage{si+1}_AdaptiveGCN'))

    with torch.no_grad():
        logits = small(streams_in)

    print(f'  MultiStreamStem    : {shapes["MultiStreamStem"]}  (4B stacked)')
    for si in range(len(small.stages)):
        for bi in range(len(small.stages[si])):
            k = f'Stage{si+1}_Block{bi+1}'
            if k in shapes:
                print(f'  {k:<22}: {shapes[k]}')
        k2 = f'Stage{si+1}_AdaptiveGCN'
        if k2 in shapes:
            print(f'  {k2:<22}: {shapes[k2]}')
    print(f'  Output (logits)    : {tuple(logits.shape)}')
    print()

    # ── NaN / Inf check ─────────────────────────────────────────────
    print(SEP)
    print('NUMERICAL CHECKS')
    print(SEP)
    assert not torch.isnan(logits).any(), "FAIL: NaN in output"
    assert not torch.isinf(logits).any(), "FAIL: Inf in output"
    print('  No NaN/Inf in output: PASS')
    print(f'  Output range: [{logits.min():.4f}, {logits.max():.4f}]')
    print()

    # ── GCN sharing verification ─────────────────────────────────────
    print(SEP)
    print('WEIGHT SHARING CHECK')
    print(SEP)
    for si, stage in enumerate(small.stages):
        gcn_ref = small.stage_gcns[si]
        for bi, blk in enumerate(stage):
            # blk.gcn is set via object.__setattr__ — access directly
            blk_gcn = object.__getattribute__(blk, 'gcn')
            shared  = blk_gcn is gcn_ref
            print(f'  Stage{si+1} Block{bi+1} gcn is stage_gcns[{si}]: {shared}')
    print()
    print('All checks passed.')
