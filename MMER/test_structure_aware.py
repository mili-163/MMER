#!/usr/bin/env python3
"""
Test script for Structure-aware Representation Learning components
"""

import torch
import torch.nn as nn
import numpy as np
from trains.singleTask.model.components import (
    ModalityEncoder, 
    FusedMultimodalRepresentation,
    CrossModalSemanticGraph,
    StructureAwareGCN,
    StructureAwareRepresentationLearning
)


def test_modality_encoder():
    """Test ModalityEncoder"""
    print("Testing ModalityEncoder...")
    
    # Create encoder
    feature_dims = [768, 5, 20]  # text, audio, vision
    encoder = ModalityEncoder(feature_dims, shared_dim=256)
    
    # Create dummy features
    batch_size = 4
    features = {
        'text': torch.randn(batch_size, 768),
        'audio': torch.randn(batch_size, 5),
        'vision': torch.randn(batch_size, 20)
    }
    
    # Forward pass
    encoded_features, pseudo_labels = encoder(features)
    
    # Check outputs
    assert 'text' in encoded_features
    assert 'audio' in encoded_features
    assert 'vision' in encoded_features
    assert encoded_features['text'].shape == (batch_size, 256)
    assert pseudo_labels['text'].shape == (batch_size, 3)
    
    print("✓ ModalityEncoder test passed")


def test_fusion_module():
    """Test FusedMultimodalRepresentation"""
    print("Testing FusedMultimodalRepresentation...")
    
    # Create fusion module
    fusion = FusedMultimodalRepresentation(shared_dim=256, lambda_smooth=0.1)
    
    # Create dummy encoded features
    batch_size = 4
    encoded_features = {
        'text': torch.randn(batch_size, 256),
        'audio': torch.randn(batch_size, 256),
        'vision': torch.randn(batch_size, 256)
    }
    
    # Forward pass
    fused = fusion(encoded_features)
    
    # Check output
    assert fused.shape == (batch_size, 256)
    
    print("✓ FusedMultimodalRepresentation test passed")


def test_semantic_graph():
    """Test CrossModalSemanticGraph"""
    print("Testing CrossModalSemanticGraph...")
    
    # Create semantic graph
    graph = CrossModalSemanticGraph(shared_dim=256, num_classes=3, delta_threshold=0.5)
    
    # Create dummy inputs
    batch_size = 4
    encoded_features = {
        'text': torch.randn(batch_size, 256),
        'audio': torch.randn(batch_size, 256),
        'vision': torch.randn(batch_size, 256)
    }
    
    pseudo_labels = {
        'text': torch.softmax(torch.randn(batch_size, 3), dim=1),
        'audio': torch.softmax(torch.randn(batch_size, 3), dim=1),
        'vision': torch.softmax(torch.randn(batch_size, 3), dim=1)
    }
    
    fused_representations = torch.randn(batch_size, 256)
    
    # Forward pass
    adjacency_matrix, node_features = graph(encoded_features, pseudo_labels, fused_representations)
    
    # Check outputs
    num_nodes = batch_size + 3  # samples + class centers
    assert adjacency_matrix.shape == (num_nodes, num_nodes)
    assert node_features.shape == (num_nodes, 256)
    
    print("✓ CrossModalSemanticGraph test passed")


def test_gcn_enhancement():
    """Test StructureAwareGCN"""
    print("Testing StructureAwareGCN...")
    
    # Create GCN
    gcn = StructureAwareGCN(shared_dim=256, beta=0.1)
    
    # Create dummy inputs
    batch_size = 4
    num_nodes = batch_size + 3  # samples + class centers
    adjacency_matrix = torch.randn(num_nodes, num_nodes)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Make symmetric
    node_features = torch.randn(num_nodes, 256)
    
    # Forward pass
    enhanced_features = gcn(adjacency_matrix, node_features, batch_size)
    
    # Check output
    assert enhanced_features.shape == (batch_size, 256)
    
    print("✓ StructureAwareGCN test passed")


def test_complete_pipeline():
    """Test complete Structure-aware pipeline"""
    print("Testing complete Structure-aware pipeline...")
    
    # Create complete module
    feature_dims = [768, 5, 20]
    structure_aware = StructureAwareRepresentationLearning(
        feature_dims=feature_dims,
        shared_dim=256,
        lambda_smooth=0.1,
        delta_threshold=0.5,
        beta=0.1
    )
    
    # Create dummy features
    batch_size = 4
    features = {
        'text': torch.randn(batch_size, 768),
        'audio': torch.randn(batch_size, 5),
        'vision': torch.randn(batch_size, 20)
    }
    
    # Forward pass
    enhanced_features = structure_aware(features)
    
    # Check output
    assert enhanced_features.shape == (batch_size, 256)
    
    print("✓ Complete pipeline test passed")


def test_missing_modalities():
    """Test handling of missing modalities"""
    print("Testing missing modalities handling...")
    
    # Create complete module
    feature_dims = [768, 5, 20]
    structure_aware = StructureAwareRepresentationLearning(
        feature_dims=feature_dims,
        shared_dim=256,
        lambda_smooth=0.1,
        delta_threshold=0.5,
        beta=0.1
    )
    
    # Create features with missing modalities
    batch_size = 4
    features = {
        'text': torch.randn(batch_size, 768),
        'audio': None,  # Missing modality
        'vision': torch.randn(batch_size, 20)
    }
    
    # Forward pass
    enhanced_features = structure_aware(features)
    
    # Check output
    assert enhanced_features.shape == (batch_size, 256)
    
    print("✓ Missing modalities test passed")


if __name__ == "__main__":
    print("Running Structure-aware component tests...\n")
    
    try:
        test_modality_encoder()
        test_fusion_module()
        test_semantic_graph()
        test_gcn_enhancement()
        test_complete_pipeline()
        test_missing_modalities()
        
        print("\n🎉 All tests passed! Structure-aware components are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise 