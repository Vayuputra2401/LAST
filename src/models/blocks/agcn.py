"""
Adaptive Graph Convolutional Network (A-GCN) Block

Implements spatial modeling with three types of adjacency matrices:
1. Physical adjacency (fixed skeleton topology)
2. Learned adjacency (global trainable matrix)
3. Dynamic adjacency (sample-dependent, computed from input)

Reference: LAST framework - Adaptive spatial modeling component
"""

import torch
import torch.nn as nn
import numpy as np


# NTU RGB+D 25-joint skeleton connections
NTU_SKELETON_BONES = [
    (0, 1),   # SpineBase -> SpineMid
    (1, 20),  # SpineMid -> Spine
    (20, 2),  # Spine -> Neck
    (2, 3),   # Neck -> Head
    (20, 4),  # Spine -> LeftShoulder
    (4, 5),   # LeftShoulder -> LeftElbow
    (5, 6),   # LeftElbow -> LeftWrist
    (6, 7),   # LeftWrist -> LeftHand
    (7, 21),  # LeftHand -> LeftHandTip
    (7, 22),  # LeftHand -> LeftThumb
    (20, 8),  # Spine -> RightShoulder
    (8, 9),   # RightShoulder -> RightElbow
    (9, 10),  # RightElbow -> RightWrist
    (10, 11), # RightWrist -> RightHand
    (11, 23), # RightHand -> RightHandTip
    (11, 24), # RightHand -> RightThumb
    (0, 12),  # SpineBase -> LeftHip
    (12, 13), # LeftHip -> LeftKnee
    (13, 14), # LeftKnee -> LeftAnkle
    (14, 15), # LeftAnkle -> LeftFoot
    (0, 16),  # SpineBase -> RightHip
    (16, 17), # RightHip -> RightKnee
    (17, 18), # RightKnee -> RightAnkle
    (18, 19), # RightAnkle -> RightFoot
]


class AdaptiveGCN(nn.Module):
    """
    Adaptive Graph Convolution Network block.
    
    Combines three adjacency matrices:
    - A_physical: Fixed skeleton topology
    - A_learned: Global trainable matrix
    - A_dynamic: Sample-dependent (computed from input features)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_joints: Number of skeleton joints (25 for NTU RGB+D)
        num_subsets: Number of adjacency matrix subsets (default: 3)
        use_learned: Whether to use learned adjacency (default: True)
        use_dynamic: Whether to use dynamic adjacency (default: True)
        residual: Whether to use residual connection (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 25,
        num_subsets: int = 3,
        use_learned: bool = True,
        use_dynamic: bool = True,
        residual: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.num_subsets = num_subsets
        self.use_learned = use_learned
        self.use_dynamic = use_dynamic
        self.residual = residual
        
        # Physical adjacency matrix (fixed)
        self.register_buffer('A_physical', self._build_physical_adjacency())
        
        # Learned adjacency matrix (global, trainable)
        if self.use_learned:
            # Initialize with identity + small noise
            A_learned = torch.eye(num_joints) + torch.randn(num_joints, num_joints) * 0.01
            self.A_learned = nn.Parameter(A_learned)
        else:
            self.register_buffer('A_learned', torch.zeros(num_joints, num_joints))
        
        # Dynamic adjacency computation
        if self.use_dynamic:
            # Project to embedding space for computing pairwise similarities
            self.embed_dim = max(4, out_channels // 4)
            self.node_embedding = nn.Conv2d(
                in_channels, self.embed_dim, kernel_size=1
            )
        
        # Graph convolution for each subset
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(num_subsets)
        ])
        
        # Residual connection
        if self.residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif self.residual:
            self.residual_conv = nn.Identity()
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def _build_physical_adjacency(self):
        """
        Build physical adjacency matrix from skeleton topology.
        
        Returns:
            Adjacency matrix (V, V) where V = num_joints
        """
        A = np.zeros((self.num_joints, self.num_joints), dtype=np.float32)
        
        # Add edges from bone connections
        for joint1, joint2 in NTU_SKELETON_BONES:
            A[joint1, joint2] = 1
            A[joint2, joint1] = 1  # Undirected graph
        
        # Add self-loops
        A += np.eye(self.num_joints, dtype=np.float32)
        
        # Normalize adjacency matrix: A = D^(-1/2) * A * D^(-1/2)
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_mat = np.diag(D_inv_sqrt)
        A_normalized = D_mat @ A @ D_mat
        
        return torch.from_numpy(A_normalized).float()
    
    def _compute_dynamic_adjacency(self, x):
        """
        Compute sample-dependent adjacency matrix from input features.
        
        Args:
            x: Input tensor (B, C, T, V)
            
        Returns:
            Dynamic adjacency matrix (B, V, V)
        """
        B, C, T, V = x.shape
        
        # Compute node embeddings
        # Average over time to get per-joint features
        x_pooled = x.mean(dim=2)  # (B, C, V)
        x_pooled = x_pooled.unsqueeze(2)  # (B, C, 1, V)
        
        embeddings = self.node_embedding(x_pooled)  # (B, embed_dim, 1, V)
        embeddings = embeddings.squeeze(2)  # (B, embed_dim, V)
        
        # Compute pairwise similarities
        # embeddings: (B, D, V) -> (B, V, D)
        embeddings = embeddings.transpose(1, 2)  # (B, V, D)
        
        # Cosine similarity: (B, V, D) @ (B, D, V) -> (B, V, V)
        norm = torch.norm(embeddings, dim=2, keepdim=True) + 1e-8
        embeddings_norm = embeddings / norm
        A_dynamic = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
        
        # Apply softmax to get attention-like weights
        A_dynamic = torch.softmax(A_dynamic, dim=-1)
        
        return A_dynamic
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V)
               B = batch size
               C = channels (3 for coordinates, or feature channels)
               T = time frames
               V = joints/vertices
               
        Returns:
            Output tensor (B, C_out, T, V)
        """
        B, C, T, V = x.shape
        
        # Collect adjacency matrices
        adjacency_matrices = []
        
        # 1. Physical adjacency (fixed)
        A_phys = self.A_physical.unsqueeze(0).expand(B, -1, -1)  # (B, V, V)
        adjacency_matrices.append(A_phys)
        
        # 2. Learned adjacency (global trainable)
        if self.use_learned:
            # Normalize learned matrix
            A_learned = self.A_learned
            D = torch.sum(A_learned, dim=1)
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            D_mat = torch.diag(D_inv_sqrt)
            A_learned_norm = D_mat @ A_learned @ D_mat
            A_learned_norm = A_learned_norm.unsqueeze(0).expand(B, -1, -1)
            adjacency_matrices.append(A_learned_norm)
        
        # 3. Dynamic adjacency (sample-dependent)
        if self.use_dynamic:
            A_dynamic = self._compute_dynamic_adjacency(x)  # (B, V, V)
            adjacency_matrices.append(A_dynamic)
        
        # Ensure we have exactly num_subsets matrices
        while len(adjacency_matrices) < self.num_subsets:
            # Pad with identity if needed
            A_identity = torch.eye(V, device=x.device).unsqueeze(0).expand(B, -1, -1)
            adjacency_matrices.append(A_identity)
        adjacency_matrices = adjacency_matrices[:self.num_subsets]
        
        # Graph convolution for each subset
        out = 0
        for i, (A, conv) in enumerate(zip(adjacency_matrices, self.conv)):
            # Reshape for matrix multiplication
            # x: (B, C, T, V) -> (B*T, V, C)
            x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
            x_reshaped = x_reshaped.view(B * T, V, C)  # (B*T, V, C)
            
            # Graph convolution: A @ X
            # A: (B, V, V), but we need (B*T, V, V)
            A_expanded = A.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, V, V)
            A_expanded = A_expanded.reshape(B * T, V, V)  # (B*T, V, V)
            
            # Matrix multiplication
            x_conv = torch.bmm(A_expanded, x_reshaped)  # (B*T, V, C)
            
            # Reshape back
            x_conv = x_conv.view(B, T, V, C)  # (B, T, V, C)
            x_conv = x_conv.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
            
            # Apply 1x1 convolution
            out += conv(x_conv)
        
        # Batch norm + activation
        out = self.bn(out)
        
        # Residual connection
        if self.residual:
            residual = self.residual_conv(x) if isinstance(self.residual_conv, nn.Conv2d) else x
            out += residual
        
        out = self.relu(out)
        
        return out


if __name__ == '__main__':
    # Test the A-GCN module
    print("Testing Adaptive GCN...")
    
    # Create module
    agcn = AdaptiveGCN(
        in_channels=64,
        out_channels=128,
        num_joints=25,
        num_subsets=3,
        use_learned=True,
        use_dynamic=True,
        residual=True
    )
    
    # Create dummy input
    x = torch.randn(2, 64, 50, 25)  # B=2, C=64, T=50, V=25
    
    # Forward pass
    out = agcn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: (2, 128, 50, 25)")
    assert out.shape == (2, 128, 50, 25), "Output shape mismatch!"
    
    # Count parameters
    num_params = sum(p.numel() for p in agcn.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    print("\n✓ A-GCN test passed!")
