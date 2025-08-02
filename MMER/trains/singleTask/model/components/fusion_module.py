import torch
import torch.nn as nn
from typing import Dict


class FusedMultimodalRepresentation(nn.Module):
    """Fused multimodal representation with adaptive weighting"""
    
    def __init__(self, shared_dim: int = 512, lambda_smooth: float = 0.1):
        super(FusedMultimodalRepresentation, self).__init__()
        self.shared_dim = shared_dim
        self.lambda_smooth = lambda_smooth
        self.num_modalities = 3  # text, audio, vision
    
    def forward(self, encoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multimodal representations with adaptive weighting
        
        Args:
            encoded_features: Dict of encoded features for each modality
            
        Returns:
            fused_representation: Fused multimodal embedding
        """
        batch_size = None
        available_modalities = []
        modality_features = []
        
        # Collect available modalities
        for modality, features in encoded_features.items():
            if features is not None:
                if batch_size is None:
                    batch_size = features.size(0)
                available_modalities.append(modality)
                modality_features.append(features)
        
        if not available_modalities:
            # No modalities available, return zero tensor
            return torch.zeros(batch_size, self.shared_dim, device=next(self.parameters()).device)
        
        # Calculate adaptive weights
        num_available = len(available_modalities)
        weights = []
        
        for i in range(num_available):
            weight = (1 + self.lambda_smooth) / (num_available + self.lambda_smooth * self.num_modalities)
            weights.append(weight)
        
        # Weighted fusion
        fused = torch.zeros(batch_size, self.shared_dim, device=modality_features[0].device)
        for i, features in enumerate(modality_features):
            fused += weights[i] * features
        
        return fused 