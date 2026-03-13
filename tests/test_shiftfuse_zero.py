"""
Tests for ShiftFuse-Zero architecture.

Covers:
  - SGPShift: zero params, correct output shape, no NaN
  - ZeroGCNBlock: shape correctness at stride=1 and stride=2
  - ShiftFuseZero: full forward pass (train + eval), param count, no NaN
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn

from src.models.blocks.sgp_shift import SGPShift
from src.models.graph import Graph, normalize_symdigraph_full
from src.models.shiftfuse_zero import (
    ZeroGCNBlock,
    ShiftFuseZero,
    build_shiftfuse_zero,
    ZERO_VARIANTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def graph_data():
    """Precompute graph adjacencies once for the whole module."""
    graph = Graph('ntu-rgb+d', 'semantic_bodypart', max_hop=2, raw_partitions=True)
    A_raw = graph.A
    A_sym = normalize_symdigraph_full(A_raw)
    A_flat  = torch.tensor((A_raw.sum(0) > 0).astype('float32'))
    A_intra = torch.tensor(A_sym[0], dtype=torch.float32)
    A_inter = torch.tensor(A_sym[1], dtype=torch.float32)
    return A_flat, A_intra, A_inter


@pytest.fixture(scope='module')
def nano_model():
    return build_shiftfuse_zero('nano', num_classes=60)


@pytest.fixture
def stream_dict_batch():
    """Fake 4-stream batch: (B=2, 3, T=64, V=25)."""
    B, T, V = 2, 64, 25
    return {
        'joint':         torch.randn(B, 3, T, V),
        'velocity':      torch.randn(B, 3, T, V),
        'bone':          torch.randn(B, 3, T, V),
        'bone_velocity': torch.randn(B, 3, T, V),
    }


@pytest.fixture
def stream_dict_batch_m2():
    """4-stream batch with M=2 bodies: (B=2, 3, T=64, V=25, M=2)."""
    B, T, V = 2, 64, 25
    return {
        'joint':         torch.randn(B, 3, T, V, 2),
        'velocity':      torch.randn(B, 3, T, V, 2),
        'bone':          torch.randn(B, 3, T, V, 2),
        'bone_velocity': torch.randn(B, 3, T, V, 2),
    }


# ---------------------------------------------------------------------------
# SGPShift tests
# ---------------------------------------------------------------------------

class TestSGPShift:

    def test_zero_params(self, graph_data):
        """SGPShift must have exactly 0 learnable parameters."""
        A_flat, A_intra, A_inter = graph_data
        sgp = SGPShift(channels=40, A_intra=A_intra, A_inter=A_inter)
        n_params = sum(p.numel() for p in sgp.parameters())
        assert n_params == 0, f"Expected 0 params, got {n_params}"

    def test_output_shape(self, graph_data):
        """Output shape must equal input shape."""
        A_flat, A_intra, A_inter = graph_data
        sgp = SGPShift(channels=40, A_intra=A_intra, A_inter=A_inter)
        x = torch.randn(2, 40, 64, 25)
        out = sgp(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_no_nan(self, graph_data):
        """Output must not contain NaN."""
        A_flat, A_intra, A_inter = graph_data
        sgp = SGPShift(channels=40, A_intra=A_intra, A_inter=A_inter)
        x = torch.randn(2, 40, 64, 25)
        out = sgp(x)
        assert torch.isfinite(out).all(), "SGPShift output contains NaN or Inf"

    def test_identity_group_unchanged(self, graph_data):
        """Identity channels (group 2) must pass through unchanged."""
        A_flat, A_intra, A_inter = graph_data
        C = 12
        sgp = SGPShift(channels=C, A_intra=A_intra, A_inter=A_inter)
        x = torch.arange(C * 25, dtype=torch.float).reshape(1, C, 1, 25)
        out = sgp(x)
        g0 = C // 3
        g1 = C // 3
        # Identity group: channels g0+g1 : C
        assert torch.allclose(out[:, g0 + g1:, :, :], x[:, g0 + g1:, :, :]), \
            "Identity group was modified — expected pass-through"

    def test_shift_indices_buffer(self, graph_data):
        """shift_indices buffer must have shape (C, V)."""
        A_flat, A_intra, A_inter = graph_data
        C, V = 40, 25
        sgp = SGPShift(channels=C, A_intra=A_intra, A_inter=A_inter, num_joints=V)
        assert sgp.shift_indices.shape == (C, V), \
            f"Expected ({C}, {V}), got {sgp.shift_indices.shape}"

    def test_different_channel_sizes(self, graph_data):
        """SGPShift should work for various channel counts."""
        A_flat, A_intra, A_inter = graph_data
        for C in [32, 40, 64, 80, 128, 160]:
            sgp = SGPShift(channels=C, A_intra=A_intra, A_inter=A_inter)
            x = torch.randn(1, C, 16, 25)
            out = sgp(x)
            assert out.shape == x.shape


# ---------------------------------------------------------------------------
# ZeroGCNBlock tests
# ---------------------------------------------------------------------------

class TestZeroGCNBlock:

    def _make_block(self, graph_data, in_ch, out_ch, stride):
        A_flat, A_intra, A_inter = graph_data
        A_learned = nn.Parameter(torch.zeros(25, 25))
        je  = None
        tla = None
        return ZeroGCNBlock(
            in_channels=in_ch,
            out_channels=out_ch,
            stride=stride,
            A_flat=A_flat,
            A_intra=A_intra,
            A_inter=A_inter,
            A_learned=A_learned,
            je=je,
            tla=tla,
            drop_path_rate=0.0,
        )

    def test_shape_stride1_same_channels(self, graph_data):
        block = self._make_block(graph_data, 40, 40, stride=1)
        x = torch.randn(2, 40, 64, 25)
        out = block(x)
        assert out.shape == (2, 40, 64, 25), f"Unexpected shape: {out.shape}"

    def test_shape_stride1_channel_expansion(self, graph_data):
        """First block of a stage: in_ch=24 → out_ch=40, stride=1."""
        block = self._make_block(graph_data, 24, 40, stride=1)
        x = torch.randn(2, 24, 64, 25)
        out = block(x)
        assert out.shape == (2, 40, 64, 25), f"Unexpected shape: {out.shape}"

    def test_shape_stride2(self, graph_data):
        """Stride-2 block: temporal dim halved."""
        block = self._make_block(graph_data, 40, 80, stride=2)
        x = torch.randn(2, 40, 64, 25)
        out = block(x)
        assert out.shape == (2, 80, 32, 25), f"Unexpected shape: {out.shape}"

    def test_no_nan_stride1(self, graph_data):
        block = self._make_block(graph_data, 40, 40, stride=1)
        x = torch.randn(2, 40, 64, 25)
        out = block(x)
        assert torch.isfinite(out).all(), "NaN in stride=1 block output"

    def test_no_nan_stride2(self, graph_data):
        block = self._make_block(graph_data, 40, 80, stride=2)
        x = torch.randn(2, 40, 64, 25)
        out = block(x)
        assert torch.isfinite(out).all(), "NaN in stride=2 block output"

    def test_with_je(self, graph_data):
        """Block with JointEmbedding attached."""
        A_flat, A_intra, A_inter = graph_data
        A_learned = nn.Parameter(torch.zeros(25, 25))
        from src.models.blocks.joint_embedding import JointEmbedding
        je = JointEmbedding(40, 25)
        block = ZeroGCNBlock(
            in_channels=40, out_channels=40, stride=1,
            A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
            A_learned=A_learned, je=je, drop_path_rate=0.0,
        )
        x = torch.randn(2, 40, 64, 25)
        out = block(x)
        assert out.shape == (2, 40, 64, 25)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ShiftFuseZero (full model) tests
# ---------------------------------------------------------------------------

class TestShiftFuseZero:

    def test_param_count_nano(self, nano_model):
        """Nano variant must have fewer than 180K params."""
        n = sum(p.numel() for p in nano_model.parameters())
        print(f"\n  ShiftFuseZero nano params: {n:,}")
        assert n < 180_000, f"Param count {n:,} exceeds 180K limit"

    def test_forward_eval_shape(self, nano_model, stream_dict_batch):
        """Eval mode forward: output must be (B, num_classes)."""
        nano_model.eval()
        with torch.no_grad():
            out = nano_model(stream_dict_batch)
        assert out.shape == (2, 60), f"Expected (2, 60), got {out.shape}"

    def test_forward_eval_no_nan(self, nano_model, stream_dict_batch):
        nano_model.eval()
        with torch.no_grad():
            out = nano_model(stream_dict_batch)
        assert torch.isfinite(out).all(), "Eval forward contains NaN or Inf"

    def test_forward_train_shape(self, nano_model, stream_dict_batch):
        """Train mode forward: output must be (B, num_classes) logits."""
        nano_model.train()
        out = nano_model(stream_dict_batch)
        assert out.shape == (2, 60), f"Expected (2, 60), got {out.shape}"

    def test_forward_train_no_nan(self, nano_model, stream_dict_batch):
        nano_model.train()
        out = nano_model(stream_dict_batch)
        assert torch.isfinite(out).all(), "Train forward contains NaN or Inf"

    def test_forward_m2_input(self, nano_model, stream_dict_batch_m2):
        """Model must accept M=2 (multi-body) inputs and select body 0."""
        nano_model.eval()
        with torch.no_grad():
            out = nano_model(stream_dict_batch_m2)
        assert out.shape == (2, 60)
        assert torch.isfinite(out).all()

    def test_no_ib_loss_output(self, nano_model, stream_dict_batch):
        """Forward must return a Tensor (not a tuple) — no IB loss."""
        nano_model.train()
        out = nano_model(stream_dict_batch)
        assert isinstance(out, torch.Tensor), \
            f"Expected Tensor, got {type(out)}"

    def test_different_num_classes(self, stream_dict_batch):
        """Model must work with num_classes=120."""
        model = build_shiftfuse_zero('nano', num_classes=120)
        model.eval()
        with torch.no_grad():
            out = model(stream_dict_batch)
        assert out.shape == (2, 120)

    def test_stage_A_learned_in_no_decay(self, nano_model):
        """All stage A_learned parameters must match 'A_learned' in name
        so the trainer's no_decay rule catches them without modification."""
        found = []
        for name, _ in nano_model.named_parameters():
            if 'A_learned' in name:
                found.append(name)
        assert len(found) == 3, \
            f"Expected 3 stage A_learned params (one per stage), found: {found}"

    def test_pool_gate_exists(self, nano_model):
        """pool_gate parameter must exist (trainer no_decay rule matches it)."""
        assert hasattr(nano_model, 'pool_gate'), "pool_gate parameter missing"
        assert isinstance(nano_model.pool_gate, nn.Parameter)

    def test_backward_no_error(self, nano_model, stream_dict_batch):
        """Backward pass must run without error."""
        nano_model.train()
        out = nano_model(stream_dict_batch)
        labels = torch.randint(0, 60, (2,))
        loss = torch.nn.functional.cross_entropy(out, labels)
        loss.backward()   # must not raise

    def test_build_helper(self):
        """build_shiftfuse_zero factory must return a ShiftFuseZero instance."""
        model = build_shiftfuse_zero('nano', num_classes=60)
        assert isinstance(model, ShiftFuseZero)

    def test_se_block_present(self):
        """When use_se=True, every ZeroGCNBlock must contain a ChannelSE module."""
        from src.models.blocks.channel_se import ChannelSE
        # Build model with kwargs overriding use_se
        # Since use_se is usually in ZERO_VARIANTS, we might need a workaround for testing
        # Or we can just modify ZERO_VARIANTS temporarily
        orig_use_se = ZERO_VARIANTS['nano'].get('use_se', False)
        ZERO_VARIANTS['nano']['use_se'] = True
        se_model = build_shiftfuse_zero('nano', num_classes=60)
        
        for stage_blocks in se_model.stages:
            for block in stage_blocks:
                assert hasattr(block, 'se'), "Block missing 'se' attribute"
                assert isinstance(block.se, ChannelSE), \
                    f"Expected ChannelSE, got {type(block.se)}"
                    
        # Restore
        ZERO_VARIANTS['nano']['use_se'] = orig_use_se

    def test_hardswish_activation(self, nano_model):
        """graph_conv must use Hardswish, not ReLU."""
        for stage_blocks in nano_model.stages:
            for block in stage_blocks:
                has_hardswish = any(
                    isinstance(m, nn.Hardswish)
                    for m in block.graph_conv.modules()
                )
                assert has_hardswish, "graph_conv should use Hardswish activation"

    def test_a_learned_warm_init(self):
        """A_learned params should be initialized to 0.01, not 0.0."""
        model = build_shiftfuse_zero('nano', num_classes=60)
        for i in range(3):
            param = getattr(model, f'stage{i}_A_learned')
            expected = torch.full_like(param, 0.01)
            assert torch.allclose(param, expected), \
                f"stage{i}_A_learned should be init to 0.01, got mean={param.mean().item():.4f}"


# ---------------------------------------------------------------------------
# SGPShift buffer device consistency
# ---------------------------------------------------------------------------

class TestSGPShiftDevice:

    def test_buffer_dtype_long(self, graph_data):
        """shift_indices buffer must be dtype=torch.long (needed for gather)."""
        A_flat, A_intra, A_inter = graph_data
        sgp = SGPShift(channels=40, A_intra=A_intra, A_inter=A_inter)
        assert sgp.shift_indices.dtype == torch.long, \
            f"Expected torch.long, got {sgp.shift_indices.dtype}"

    def test_neighbor_cycling_diversity(self, graph_data):
        """With neighbor cycling, not all channels in a group should have
        identical shift indices (more spatial diversity)."""
        A_flat, A_intra, A_inter = graph_data
        C = 40
        sgp = SGPShift(channels=C, A_intra=A_intra, A_inter=A_inter)
        g0 = C // 3
        # Intra group: channels 0..g0-1
        intra_indices = sgp.shift_indices[:g0]  # (g0, V)
        # Check if there's any diversity (not all rows identical)
        # At least some channels should have different shift targets
        num_unique_rows = len(torch.unique(intra_indices, dim=0))
        assert num_unique_rows > 1, \
            f"Expected cycling diversity (>1 unique rows), got {num_unique_rows}"
