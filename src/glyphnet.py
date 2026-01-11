"""
GlyphNetwork: Neural network for generating 128x128 grayscale glyph bitmaps.

Architecture:
- Embedding layers for prev_char and curr_char (27 tokens x 32 dims each)
- MLP fusion module (128 -> 256 -> 512 -> 256 -> 512)
- Transposed convolutional decoder (8x8 -> 128x128)

Inputs:
- prev_char: Previous character [0-26], 0 = start token
- curr_char: Current character [1-26], a-z
- style_z: 64-dimensional style latent vector

Output:
- 128x128 grayscale bitmap with values in [0, 1]
"""

import torch
import torch.nn as nn
from typing import Optional


class GlyphNetwork(nn.Module):
    """
    Neural network for generating character glyphs.

    Combines character context (prev/curr) with style vector to produce
    128x128 grayscale glyph bitmaps.
    """

    def __init__(
        self,
        vocab_size: int = 27,
        char_embed_dim: int = 32,
        style_dim: int = 64,
    ):
        """
        Initialize GlyphNetwork.

        Args:
            vocab_size: Number of tokens (27 = padding + 26 letters)
            char_embed_dim: Dimension per character embedding (32)
            style_dim: Dimension of style latent vector (64)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.char_embed_dim = char_embed_dim
        self.style_dim = style_dim

        # Total input dim to MLP: prev_embed + curr_embed + style_z
        mlp_input_dim = char_embed_dim * 2 + style_dim  # 32 + 32 + 64 = 128

        # Embedding layers
        self.prev_char_embed = nn.Embedding(vocab_size, char_embed_dim)
        self.curr_char_embed = nn.Embedding(vocab_size, char_embed_dim)

        # MLP fusion module: 128 -> 256 -> 512 -> 256 -> 512
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )

        # Decoder: project to spatial then upsample
        # 512 -> 2048 (8x8x32)
        self.decoder_proj = nn.Linear(512, 8 * 8 * 32)

        # Transposed conv decoder: 8x8 -> 128x128
        self.decoder = nn.Sequential(
            # 8x8x32 -> 16x16x64
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16x64 -> 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32x32 -> 64x64x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x64x16 -> 128x128x8
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 128x128x8 -> 128x128x1
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        # Embedding init
        nn.init.normal_(self.prev_char_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.curr_char_embed.weight, mean=0.0, std=0.02)

        # MLP init
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

        # Decoder proj init
        nn.init.kaiming_normal_(self.decoder_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.decoder_proj.bias)

        # Conv init
        for module in self.decoder:
            if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        prev_char: torch.Tensor,
        curr_char: torch.Tensor,
        style_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate glyph bitmap.

        Args:
            prev_char: Previous character indices, shape (B,) or scalar, range [0-26]
            curr_char: Current character indices, shape (B,) or scalar, range [1-26]
            style_z: Style latent vectors, shape (B, 64) or (64,)

        Returns:
            Glyph bitmap, shape (B, 1, 128, 128), values in [0, 1]
        """
        # Handle scalar inputs
        if prev_char.dim() == 0:
            prev_char = prev_char.unsqueeze(0)
        if curr_char.dim() == 0:
            curr_char = curr_char.unsqueeze(0)
        if style_z.dim() == 1:
            style_z = style_z.unsqueeze(0)

        batch_size = prev_char.size(0)

        # Input validation
        if not (0 <= prev_char.min() and prev_char.max() < self.vocab_size):
            raise ValueError(
                f"prev_char must be in [0, {self.vocab_size - 1}], "
                f"got range [{prev_char.min()}, {prev_char.max()}]"
            )
        if not (1 <= curr_char.min() and curr_char.max() < self.vocab_size):
            raise ValueError(
                f"curr_char must be in [1, {self.vocab_size - 1}], "
                f"got range [{curr_char.min()}, {curr_char.max()}]"
            )
        if style_z.size(-1) != self.style_dim:
            raise ValueError(
                f"style_z must have dimension {self.style_dim}, got {style_z.size(-1)}"
            )

        # Embed characters
        prev_embed = self.prev_char_embed(prev_char)  # (B, 32)
        curr_embed = self.curr_char_embed(curr_char)  # (B, 32)

        # Concatenate embeddings with style
        fused = torch.cat([prev_embed, curr_embed, style_z], dim=-1)  # (B, 128)

        # MLP fusion
        features = self.mlp(fused)  # (B, 512)

        # Project to spatial
        spatial = self.decoder_proj(features)  # (B, 2048)
        spatial = spatial.view(batch_size, 32, 8, 8)  # (B, 32, 8, 8)

        # Decode to bitmap
        bitmap = self.decoder(spatial)  # (B, 1, 128, 128)

        return bitmap


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(device: Optional[torch.device] = None) -> GlyphNetwork:
    """
    Create GlyphNetwork instance.

    Args:
        device: Target device (defaults to CPU)

    Returns:
        Initialized GlyphNetwork
    """
    model = GlyphNetwork()
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    # Quick test
    model = GlyphNetwork()
    print(f"GlyphNetwork parameters: {count_parameters(model):,}")

    # Test forward pass
    prev_char = torch.tensor([0])
    curr_char = torch.tensor([1])
    style_z = torch.randn(1, 64)

    output = model(prev_char, curr_char, style_z)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
