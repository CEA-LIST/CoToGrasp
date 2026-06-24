import torch
import torch.nn as nn

class AdaLNDecoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_classes=27):
        super().__init__()
        # Project Z to Scale (gamma) and Shift (beta) for the features
        self.z_proj = nn.Linear(latent_dim, feature_dim * 2) 
        self.mlp_y = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        # Standard LayerNorm (elementwise_affine=False means no learnable params, we provide them)
        self.ln = nn.LayerNorm(feature_dim, elementwise_affine=False)

    def forward(self, y, z):
        """
        :param y: condition (B, y_dim, C_y)
        :param z: (B, latent_size)
        :return x_hat: (B, y_dim)
        """
        # Calculate modulation params from Z: (B, 2*feature_dim) -> (B, 1, 2*feature_dim)
        style = self.z_proj(z).unsqueeze(1) 
        gamma, beta = style.chunk(2, dim=-1)
        # Modulate Y: Normalize Y, then Scale by Gamma, Shift by Beta
        y_modulated = self.ln(y) * (1 + gamma) + beta
        # Classify
        return self.mlp_y(y_modulated)
