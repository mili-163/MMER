import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class ClassCenterCalculator(nn.Module):
    """Calculate class centers for each modality dynamically"""
    
    def __init__(self, shared_dim: int = 512, num_classes: int = 3):
        super(ClassCenterCalculator, self).__init__()
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        
    def compute_class_centers(self, features: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class centers for a modality
        
        Args:
            features: Encoded features [batch_size, shared_dim]
            pseudo_labels: Pseudo-label probabilities [batch_size, num_classes]
            
        Returns:
            class_centers: Class centers [num_classes, shared_dim]
        """
        batch_size = features.size(0)
        predicted_classes = torch.argmax(pseudo_labels, dim=1)  # [batch_size]
        
        # Initialize class centers
        class_centers = torch.zeros(self.num_classes, self.shared_dim, device=features.device)
        class_counts = torch.zeros(self.num_classes, device=features.device)
        
        # Compute centers by averaging features of same predicted class
        for i in range(batch_size):
            pred_class = predicted_classes[i]
            class_centers[pred_class] += features[i]
            class_counts[pred_class] += 1
        
        # Normalize by counts (avoid division by zero)
        valid_classes = class_counts > 0
        class_centers[valid_classes] /= class_counts[valid_classes].unsqueeze(1)
        
        return class_centers


class CrossModalSemanticGraph(nn.Module):
    """Cross-modal semantic graph construction with proper class center computation"""
    
    def __init__(self, shared_dim: int = 512, num_classes: int = 3, delta_threshold: float = 1.5):
        super(CrossModalSemanticGraph, self).__init__()
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.delta_threshold = delta_threshold
        
        # Class center calculator
        self.class_center_calculator = ClassCenterCalculator(shared_dim, num_classes)
        
        # Initialize learnable class centers as fallback
        self.learnable_class_centers = nn.Parameter(torch.randn(num_classes, shared_dim))
        
    def compute_skl_divergence(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Compute symmetric KL divergence between two probability distributions"""
        kl_12 = F.kl_div(p1.log(), p2, reduction='none').sum(dim=-1)
        kl_21 = F.kl_div(p2.log(), p1, reduction='none').sum(dim=-1)
        return 0.5 * (kl_12 + kl_21)
    
    def compute_sample_edge_weight(self, z_i: torch.Tensor, z_j: torch.Tensor, 
                                 p_i: torch.Tensor, p_j: torch.Tensor,
                                 c_i: torch.Tensor, c_j: torch.Tensor) -> torch.Tensor:
        """Compute edge weight between two samples"""
        # Compute SKL divergence
        skl = self.compute_skl_divergence(p_i, p_j)
        
        # Only create edge if SKL < delta
        mask = skl < self.delta_threshold
        
        # Compute edge weight
        weight = torch.zeros_like(skl)
        valid_mask = mask & (skl < self.delta_threshold)
        
        if valid_mask.any():
            # Distance to class centers
            dist_i_to_cj = torch.exp(-0.5 * torch.norm(z_i - c_j, dim=-1) ** 2)
            dist_j_to_ci = torch.exp(-0.5 * torch.norm(z_j - c_i, dim=-1) ** 2)
            
            # Combined weight
            weight[valid_mask] = (1 - skl[valid_mask] / self.delta_threshold) * dist_i_to_cj[valid_mask] * dist_j_to_ci[valid_mask]
        
        return weight
    
    def compute_modality_class_centers(self, encoded_features: Dict[str, torch.Tensor], 
                                     pseudo_labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute class centers for each modality
        
        Args:
            encoded_features: Dict of encoded features
            pseudo_labels: Dict of pseudo-label probabilities
            
        Returns:
            modality_class_centers: Dict of class centers for each modality
        """
        modality_class_centers = {}
        
        for modality in ['text', 'audio', 'vision']:
            if modality in encoded_features and encoded_features[modality] is not None:
                features = encoded_features[modality]
                labels = pseudo_labels[modality]
                
                # Compute class centers for this modality
                class_centers = self.class_center_calculator.compute_class_centers(features, labels)
                modality_class_centers[modality] = class_centers
            else:
                # Use learnable centers as fallback
                modality_class_centers[modality] = self.learnable_class_centers
        
        return modality_class_centers
    
    def forward(self, encoded_features: Dict[str, torch.Tensor], 
                pseudo_labels: Dict[str, torch.Tensor],
                fused_representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct cross-modal semantic graph
        
        Args:
            encoded_features: Dict of encoded features
            pseudo_labels: Dict of pseudo-label probabilities
            fused_representations: Fused multimodal representations
            
        Returns:
            adjacency_matrix: Graph adjacency matrix
            node_features: Node features (samples + class centers)
        """
        batch_size = fused_representations.size(0)
        num_nodes = batch_size + self.num_classes
        
        # Initialize adjacency matrix
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=fused_representations.device)
        
        # Get available modalities
        available_modalities = [mod for mod, features in encoded_features.items() if features is not None]
        
        if not available_modalities:
            # No modalities available, return identity-like graph
            adjacency_matrix[:batch_size, :batch_size] = torch.eye(batch_size, device=fused_representations.device)
            node_features = torch.cat([fused_representations, self.learnable_class_centers], dim=0)
            return adjacency_matrix, node_features
        
        # Compute class centers for each modality
        modality_class_centers = self.compute_modality_class_centers(encoded_features, pseudo_labels)
        
        # 存储每个模态的边缘权重
        modality_edge_weights = {}
        
        # Compute sample-to-sample edges for each modality
        for modality in available_modalities:
            if modality in pseudo_labels and pseudo_labels[modality] is not None:
                features = encoded_features[modality]
                labels = pseudo_labels[modality]
                class_centers = modality_class_centers[modality]
                
                # 初始化该模态的边缘权重矩阵
                modality_weights = torch.zeros(batch_size, batch_size, device=fused_representations.device)
                
                # Compute edges between all sample pairs
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        # Get predicted class centers
                        pred_i = torch.argmax(labels[i])
                        pred_j = torch.argmax(labels[j])
                        c_i = class_centers[pred_i]
                        c_j = class_centers[pred_j]
                        
                        # Compute edge weight
                        weight = self.compute_sample_edge_weight(
                            features[i], features[j], 
                            labels[i], labels[j],
                            c_i, c_j
                        )
                        
                        # Add to modality weight matrix (symmetric)
                        modality_weights[i, j] = weight
                        modality_weights[j, i] = weight
                
                modality_edge_weights[modality] = modality_weights
        
        # 边缘聚合：按照论文公式 a_ij = (1/M_ij) * Σ_m=1^M_ij w_ij^(m)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # 计算共享模态数量
                    shared_modalities = []
                    for modality in available_modalities:
                        if modality in modality_edge_weights:
                            # 检查该模态中i和j是否都有有效特征
                            if (encoded_features[modality] is not None and 
                                pseudo_labels[modality] is not None):
                                shared_modalities.append(modality)
                    
                    M_ij = len(shared_modalities)
                    
                    if M_ij > 0:
                        # 聚合边缘权重
                        total_weight = 0.0
                        for modality in shared_modalities:
                            total_weight += modality_edge_weights[modality][i, j]
                        
                        # 平均权重
                        avg_weight = total_weight / M_ij
                        adjacency_matrix[i, j] = avg_weight
        
        # Add sample-to-center edges using fused representation
        # Compute fused pseudo-labels and use average class centers
        fused_pseudo_labels = F.softmax(fused_representations @ self.learnable_class_centers.T / 0.1, dim=-1)
        predicted_classes = torch.argmax(fused_pseudo_labels, dim=-1)
        
        # Use average of modality class centers as global centers
        avg_class_centers = torch.stack(list(modality_class_centers.values())).mean(dim=0)
        
        for i in range(batch_size):
            pred_class = predicted_classes[i]
            center_idx = batch_size + pred_class
            
            # Compute weight to class center
            weight = torch.exp(-0.5 * torch.norm(fused_representations[i] - avg_class_centers[pred_class]) ** 2)
            adjacency_matrix[i, center_idx] = weight
            adjacency_matrix[center_idx, i] = weight
        
        # Add self-loops for class centers
        for i in range(self.num_classes):
            center_idx = batch_size + i
            adjacency_matrix[center_idx, center_idx] = 1.0
        
        # Prepare node features (samples + average class centers)
        node_features = torch.cat([fused_representations, avg_class_centers], dim=0)
        
        return adjacency_matrix, node_features 