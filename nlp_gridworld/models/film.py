"""
models/film.py
--------------
Feature-wise Linear Modulation (FiLM).

Given a language vector z, FiLM conditions a CNN feature map F by:
    F' = gamma(z) * F + beta(z)

gamma and beta are learned affine parameters predicted from the language.
This is more expressive than concatenation and cheaper than cross-attention.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", 2018.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Conditions a feature map on a language vector.

    Args:
        lang_dim      : dimension of the conditioning language vector
        num_channels  : number of channels in the feature map to condition
    """

    def __init__(self, lang_dim: int, num_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(lang_dim, num_channels)
        self.beta_fc  = nn.Linear(lang_dim, num_channels)

        # Initialise gamma to 1 and beta to 0 so the layer starts as identity
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, features: torch.Tensor,
                lang_vec: torch.Tensor) -> torch.Tensor:
        """
        features : (B, C, H, W)
        lang_vec : (B, lang_dim)
        returns  : (B, C, H, W)  — modulated feature map
        """
        gamma = self.gamma_fc(lang_vec).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta  = self.beta_fc(lang_vec).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


class FiLMCNN(nn.Module):
    """
    3-layer CNN where each layer's feature maps are FiLM-conditioned
    by the language vector.

    Args:
        grid_channels : input channels (5 for our env)
        lang_dim      : dimension of the language conditioning vector
        out_channels  : final number of feature-map channels
    """

    def __init__(self, grid_channels: int = 5, lang_dim: int = 128,
                 out_channels: int = 64):
        super().__init__()

        self.conv1  = nn.Conv2d(grid_channels, 32, kernel_size=3, padding=1)
        self.film1  = FiLMLayer(lang_dim, 32)
        self.bn1    = nn.BatchNorm2d(32)

        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.film2  = FiLMLayer(lang_dim, 64)
        self.bn2    = nn.BatchNorm2d(64)

        self.conv3  = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.film3  = FiLMLayer(lang_dim, out_channels)
        self.bn3    = nn.BatchNorm2d(out_channels)

        self.relu   = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, grid: torch.Tensor,
                lang_vec: torch.Tensor) -> torch.Tensor:
        """
        grid     : (B, 5, 8, 8)
        lang_vec : (B, lang_dim)
        returns  : (B, out_channels * 8 * 8)
        """
        x = self.relu(self.bn1(self.film1(self.conv1(grid),    lang_vec)))
        x = self.relu(self.bn2(self.film2(self.conv2(x),       lang_vec)))
        x = self.relu(self.bn3(self.film3(self.conv3(x),       lang_vec)))
        return self.flatten(x)
