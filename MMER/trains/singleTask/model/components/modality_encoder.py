import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ModalityEncoder(nn.Module):
    """Modality-specific encoders with lightweight adapters"""
    
    def __init__(self, feature_dims: List[int], shared_dim: int = 256):
        super(ModalityEncoder, self).__init__()
        self.shared_dim = shared_dim
        self.num_modalities = len(feature_dims)
        
        # Lightweight adapters for each modality
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(shared_dim, shared_dim)
            ) for dim in feature_dims
        ])
        
        # Modality-specific classifiers for pseudo-label generation
        self.classifiers = nn.ModuleList([
            nn.Linear(shared_dim, 3)  # 3 classes for emotion recognition
            for _ in range(self.num_modalities)
        ])
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode modality-specific features and generate pseudo-labels
        
        Args:
            features: Dict with keys 'text', 'audio', 'vision' containing modality features
            
        Returns:
            encoded_features: Dict of aligned embeddings
            pseudo_labels: Dict of pseudo-label probabilities
        """
        encoded_features = {}
        pseudo_labels = {}
        
        modality_names = ['text', 'audio', 'vision']
        
        for i, modality in enumerate(modality_names):
            if modality in features and features[modality] is not None:
                # Encode features
                encoded = self.adapters[i](features[modality])
                encoded_features[modality] = encoded
                
                # Generate pseudo-labels
                logits = self.classifiers[i](encoded)
                probs = F.softmax(logits / 0.1, dim=-1)  # temperature = 0.1
                pseudo_labels[modality] = probs
            else:
                encoded_features[modality] = None
                pseudo_labels[modality] = None
        
        return encoded_features, pseudo_labels 