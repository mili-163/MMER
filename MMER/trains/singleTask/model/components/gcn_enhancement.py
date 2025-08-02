import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureAwareGCN(nn.Module):
    """Structure-aware representation enhancement via GCNs"""
    
    def __init__(self, shared_dim: int = 512, beta: float = 0.5):
        super(StructureAwareGCN, self).__init__()
        self.shared_dim = shared_dim
        self.beta = beta
        
        # GCN layers
        self.gcn_weight = nn.Linear(shared_dim, shared_dim)
        self.gcn_skip_weight = nn.Linear(shared_dim, shared_dim)
        
    def forward(self, adjacency_matrix: torch.Tensor, node_features: torch.Tensor, 
                batch_size: int) -> torch.Tensor:
        """
        Enhance representations using GCN
        
        Args:
            adjacency_matrix: Graph adjacency matrix
            node_features: Node features
            batch_size: Number of samples
            
        Returns:
            enhanced_features: Structure-enhanced features
        """
        # Normalize adjacency matrix
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        degree_matrix_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(degree_matrix) + 1e-8))
        normalized_adj = degree_matrix_inv_sqrt @ adjacency_matrix @ degree_matrix_inv_sqrt
        
        # GCN forward pass
        gcn_out = torch.relu(normalized_adj @ node_features @ self.gcn_weight.weight.T + 
                            node_features @ self.gcn_skip_weight.weight.T)
        
        # Extract sample features (first batch_size nodes)
        sample_features = gcn_out[:batch_size]
        
        # Residual connection
        original_features = node_features[:batch_size]
        enhanced_features = original_features + self.beta * sample_features
        
        return enhanced_features 