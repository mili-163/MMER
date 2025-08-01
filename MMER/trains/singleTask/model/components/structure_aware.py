import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from .modality_encoder import ModalityEncoder
from .fusion_module import FusedMultimodalRepresentation
from .semantic_graph import CrossModalSemanticGraph
from .gcn_enhancement import StructureAwareGCN


class StructureAwareRepresentationLearning(nn.Module):
    """Complete Structure-aware Representation Learning module"""
    
    def __init__(self, feature_dims: List[int], shared_dim: int = 256, 
                 lambda_smooth: float = 0.1, delta_threshold: float = 0.5, 
                 beta: float = 0.1):
        super(StructureAwareRepresentationLearning, self).__init__()
        
        self.modality_encoder = ModalityEncoder(feature_dims, shared_dim)
        self.fused_representation = FusedMultimodalRepresentation(shared_dim, lambda_smooth)
        self.semantic_graph = CrossModalSemanticGraph(shared_dim, num_classes=3, delta_threshold=delta_threshold)
        self.structure_enhancement = StructureAwareGCN(shared_dim, beta)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Complete forward pass for structure-aware representation learning
        
        Args:
            features: Dict with keys 'text', 'audio', 'vision' containing modality features
            
        Returns:
            enhanced_features: Structure-aware enhanced representations
        """
        # Step 1: Modality encoding and pseudo-label generation
        encoded_features, pseudo_labels = self.modality_encoder(features)
        
        # Step 2: Fused multimodal representation
        fused_representations = self.fused_representation(encoded_features)
        
        # Step 3: Cross-modal semantic graph construction
        adjacency_matrix, node_features = self.semantic_graph(
            encoded_features, pseudo_labels, fused_representations
        )
        
        # Step 4: Structure-aware enhancement via GCN
        batch_size = fused_representations.size(0)
        enhanced_features = self.structure_enhancement(
            adjacency_matrix, node_features, batch_size
        )
        
        return enhanced_features 