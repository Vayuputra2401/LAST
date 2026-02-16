"""
Teacher-LAST: VideoMAE-2 Model Wrapper
=======================================
Wraps HuggingFace's VideoMAE-2 Large for fine-tuning on NTU RGB+D 60.

Architecture:
    - Backbone: ViT-L/16 (VideoMAE-2 Large) from HuggingFace
    - Head: Linear classifier (1024 → 60 classes)
    - Pooling: Mean pooling over temporal tokens (no CLS token)

Pre-trained Model:
    MCG-NJU/videomae-large-finetuned-kinetics (Kinetics-400 fine-tuned)
    → We replace the 400-class head with a 60-class head for NTU-60

Layer-wise LR Decay:
    Deeper layers (closer to input) get smaller learning rates.
    This preserves low-level features while adapting high-level ones.
    Official VideoMAE uses layer_decay=0.75 for ViT-Large.

References:
    - VideoMAE V2: https://arxiv.org/abs/2303.16727
    - Official repo: https://github.com/MCG-NJU/VideoMAE
"""

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


# =============================================================================
# VideoMAE-2 Model for NTU-60
# =============================================================================

class VideoMAEForNTU60(nn.Module):
    """
    VideoMAE-2 Large fine-tuned for NTU RGB+D 60 action recognition.
    
    Loads pre-trained VideoMAE-2 from HuggingFace and replaces the
    classification head for 60 action classes.
    
    Args:
        config: Model configuration dictionary with keys:
            - pretrained_path: HuggingFace model name or local path
            - num_classes: Number of output classes (60 for NTU-60)
            - drop_path_rate: Stochastic depth rate
            - fc_drop_rate: Classifier dropout rate
            - use_mean_pooling: Whether to use mean pooling (vs CLS token)
            - init_scale: Scale for classifier weight initialization
    """

    def __init__(self, config):
        super().__init__()

        self.num_classes = config.get('num_classes', 60)
        pretrained_path = config.get('pretrained_path', 'MCG-NJU/videomae-large-finetuned-kinetics')
        fc_drop_rate = config.get('fc_drop_rate', 0.5)
        init_scale = config.get('init_scale', 0.001)

        print(f"[Model] Loading VideoMAE-2 from: {pretrained_path}")
        print(f"[Model] Target classes: {self.num_classes}")

        # Load pre-trained model, replacing the head for our number of classes
        # ignore_mismatched_sizes=True allows loading a 400-class model
        # and replacing the head with a 60-class one
        self.model = VideoMAEForVideoClassification.from_pretrained(
            pretrained_path,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        # The HuggingFace model internally handles:
        # 1. Patch embedding (video → sequence of patch tokens)
        # 2. Positional embedding
        # 3. Transformer encoder blocks
        # 4. Classification head (already replaced via num_labels)

        # Add dropout before classifier if specified
        if fc_drop_rate > 0:
            self.fc_dropout = nn.Dropout(p=fc_drop_rate)
        else:
            self.fc_dropout = nn.Identity()

        # Re-initialize the classifier head with small weights
        # This prevents large gradients from the randomly-initialized head
        # from destabilizing the pre-trained backbone early in training
        if hasattr(self.model, 'classifier'):
            nn.init.trunc_normal_(self.model.classifier.weight, std=init_scale)
            if self.model.classifier.bias is not None:
                nn.init.zeros_(self.model.classifier.bias)
            print(f"[Model] Classifier head initialized (scale={init_scale})")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total_params / 1e6:.1f}M")
        print(f"[Model] Trainable params: {trainable_params / 1e6:.1f}M")

    def forward(self, pixel_values):
        """
        Forward pass.
        
        Args:
            pixel_values: Video tensor (B, C, T, H, W)
                B: batch size
                C: channels (3)
                T: temporal frames (16)
                H, W: spatial resolution (224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    # =========================================================================
    # Layer-wise Learning Rate Decay
    # =========================================================================

    def get_parameter_groups(self, base_lr, weight_decay, layer_decay=0.75):
        """
        Create parameter groups with layer-wise LR decay.
        
        Deeper layers (closer to input) get smaller learning rates.
        This is crucial for fine-tuning — preserves low-level features
        (edges, textures) learned during pre-training while allowing
        high-level features to adapt to NTU-60 actions.
        
        Layer decay formula:
            lr_scale = layer_decay ^ (num_layers - layer_id)
        
        With layer_decay=0.75 and 24 layers (ViT-Large):
            Layer 0  (patch embed): lr * 0.75^24 = lr * 0.00075
            Layer 12 (middle):      lr * 0.75^12 = lr * 0.032
            Layer 23 (last block):  lr * 0.75^1  = lr * 0.75
            Classifier head:        lr * 1.0     (no decay)
        
        Args:
            base_lr: Base learning rate (after linear scaling)
            weight_decay: Weight decay value
            layer_decay: Decay factor per layer (0.75 for ViT-Large)
        
        Returns:
            list: Parameter groups for optimizer
        """
        # Get the number of transformer layers
        num_layers = len(self.model.videomae.encoder.layer)

        parameter_groups = {}
        no_decay = {'bias', 'LayerNorm.weight', 'layernorm.weight',
                    'layer_norm.weight', 'norm.weight'}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Determine layer ID for this parameter
            layer_id = self._get_layer_id(name, num_layers)

            # Compute LR scale based on layer depth
            # Deeper layers (lower ID) get smaller LR
            lr_scale = layer_decay ** (num_layers - layer_id)

            # Check if this parameter should have weight decay
            has_decay = True
            for nd in no_decay:
                if nd in name:
                    has_decay = False
                    break

            # Create group key
            group_key = f"layer_{layer_id}_{'decay' if has_decay else 'no_decay'}"

            if group_key not in parameter_groups:
                parameter_groups[group_key] = {
                    'params': [],
                    'lr': base_lr * lr_scale,
                    'weight_decay': weight_decay if has_decay else 0.0,
                    'lr_scale': lr_scale,
                }

            parameter_groups[group_key]['params'].append(param)

        # Log the learning rates
        print(f"\n[Model] Layer-wise LR decay (layer_decay={layer_decay}):")
        for key in sorted(parameter_groups.keys()):
            group = parameter_groups[key]
            n_params = sum(p.numel() for p in group['params'])
            print(f"  {key}: lr={group['lr']:.6f} (scale={group['lr_scale']:.4f}), "
                  f"wd={group['weight_decay']}, params={n_params}")

        return list(parameter_groups.values())

    def _get_layer_id(self, name, num_layers):
        """
        Determine the layer ID for a parameter name.
        
        Mapping:
            - 'embeddings' or 'patch_embed': Layer 0
            - 'encoder.layer.{N}': Layer N + 1
            - 'classifier' or 'fc_norm' or 'layernorm': Layer num_layers + 1 (head)
        
        Args:
            name: Parameter name string
            num_layers: Total number of transformer layers
        
        Returns:
            int: Layer ID (0 = deepest/input, num_layers+1 = head)
        """
        if 'embeddings' in name or 'patch_embed' in name:
            return 0
        elif 'encoder.layer.' in name:
            # Extract layer number from name like 'model.videomae.encoder.layer.5.xxx'
            try:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                return layer_num + 1
            except (ValueError, IndexError):
                return num_layers  # Fallback to last layer
        else:
            # Head, final norm, classifier → highest layer
            return num_layers + 1


# =============================================================================
# Model Builder
# =============================================================================

def build_model(config):
    """
    Build the VideoMAE-2 model for NTU-60.
    
    Args:
        config: Full configuration dictionary
    
    Returns:
        VideoMAEForNTU60: The model
    """
    model_config = config['model']
    model_config['num_classes'] = config['data']['num_classes']

    model = VideoMAEForNTU60(model_config)

    return model
