#!/usr/bin/env python3
"""
Test script for Multi-Channel Prompt Distillation components
"""

import torch
import torch.nn as nn
import numpy as np
from trains.singleTask.model.components import (
    LocalChannelDistillation,
    FusionChannelDistillation,
    PromptBasedCompletion,
    MissingAwareClassification,
    MultiChannelPromptDistillation
)


def test_local_channel_distillation():
    """Test Local-Channel Distillation"""
    print("测试 Local-Channel Distillation ...")
    
    # Create distillation module
    distillation = LocalChannelDistillation(temperature=0.1)
    
    # Create dummy data
    batch_size = 8
    embed_dim = 256
    num_classes = 3
    
    modal_features = {
        'text': torch.randn(batch_size, embed_dim),
        'audio': torch.randn(batch_size, embed_dim),
        'vision': torch.randn(batch_size, embed_dim)
    }
    
    local_anchors = {
        'text': torch.randn(num_classes, embed_dim),
        'audio': torch.randn(num_classes, embed_dim),
        'vision': torch.randn(num_classes, embed_dim)
    }
    
    # Forward pass
    loss = distillation(modal_features, local_anchors)
    
    # Check output
    assert isinstance(loss, torch.Tensor)
    print(f"损失值: {loss.item()}")
    assert loss.item() >= 0
    
    print("Local-Channel Distillation 测试通过")


def test_fusion_channel_distillation():
    """Test Fusion-Channel Distillation"""
    print("测试 Fusion-Channel Distillation ...")
    
    # Create distillation module
    distillation = FusionChannelDistillation(temperature=0.1)
    
    # Create dummy data
    batch_size = 8
    embed_dim = 256
    num_classes = 3
    
    fused_features = torch.randn(batch_size, embed_dim)
    global_anchors = torch.randn(num_classes, embed_dim)
    
    # Forward pass
    loss = distillation(fused_features, global_anchors)
    
    # Check output
    assert isinstance(loss, torch.Tensor)
    print(f"损失值: {loss.item()}")
    assert loss.item() >= 0
    
    print("Fusion-Channel Distillation 测试通过")


def test_prompt_based_completion():
    """Test Prompt-Based Completion"""
    print("测试 Prompt-Based Completion ...")
    
    # Create completion module
    completion = PromptBasedCompletion(
        embed_dim=256,
        temperature=0.1,
        llm_model_name="t5-base",
        device="cpu"
    )
    
    # Create dummy data
    batch_size = 8
    embed_dim = 256
    num_classes = 3
    
    modal_features = {
        'text': torch.randn(batch_size, embed_dim),
        'audio': torch.randn(batch_size, embed_dim),
        'vision': torch.randn(batch_size, embed_dim)
    }
    
    global_anchors = torch.randn(num_classes, embed_dim)
    category_prototypes = torch.randn(num_classes, embed_dim)
    
    # Forward pass
    loss, missing_embeddings = completion(modal_features, global_anchors, category_prototypes)
    
    # Check outputs
    assert isinstance(loss, torch.Tensor)
    print(f"损失值: {loss.item()}")
    assert loss.item() >= 0
    assert missing_embeddings.shape == (batch_size, embed_dim)
    
    print("Prompt-Based Completion 测试通过")


def test_missing_aware_classification():
    """Test Missing-Aware Classification"""
    print("测试 Missing-Aware Classification ...")
    
    # Create classification module
    classification = MissingAwareClassification(temperature=0.1, completion_weight=0.5)
    
    # Create dummy data
    batch_size = 8
    embed_dim = 256
    num_classes = 3
    
    fused_features = torch.randn(batch_size, embed_dim)
    missing_embeddings = torch.randn(batch_size, embed_dim)
    global_anchors = torch.randn(num_classes, embed_dim)
    
    modal_features = {
        'text': torch.randn(batch_size, embed_dim),
        'audio': None,  # 模拟缺失模态
        'vision': torch.randn(batch_size, embed_dim)
    }
    
    # Forward pass
    final_probs, compensated_features = classification(
        fused_features, missing_embeddings, modal_features, global_anchors
    )
    
    # Check outputs
    assert final_probs.shape == (batch_size, num_classes)
    assert compensated_features.shape == (batch_size, embed_dim)
    assert torch.allclose(final_probs.sum(dim=1), torch.ones(batch_size))
    
    print("Missing-Aware Classification 测试通过")


def test_complete_prompt_distillation():
    """Test complete Multi-Channel Prompt Distillation"""
    print("测试完整 Multi-Channel Prompt Distillation ...")
    
    # Create distillation module
    distillation = MultiChannelPromptDistillation(
        embed_dim=256,
        num_classes=3,
        temperature=0.1,
        completion_weight=0.5,
        llm_model_name="t5-base",
        lambda_local=1.0,
        lambda_fusion=1.0,
        lambda_prompt=1.0,
        device="cpu"
    )
    
    # Create dummy data
    batch_size = 8
    embed_dim = 256
    num_classes = 3
    
    modal_features = {
        'text': torch.randn(batch_size, embed_dim),
        'audio': torch.randn(batch_size, embed_dim),
        'vision': torch.randn(batch_size, embed_dim)
    }
    
    fused_features = torch.randn(batch_size, embed_dim)
    local_anchors = {
        'text': torch.randn(num_classes, embed_dim),
        'audio': torch.randn(num_classes, embed_dim),
        'vision': torch.randn(num_classes, embed_dim)
    }
    global_anchors = torch.randn(num_classes, embed_dim)
    category_prototypes = torch.randn(num_classes, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    result = distillation(
        modal_features=modal_features,
        fused_features=fused_features,
        local_anchors=local_anchors,
        global_anchors=global_anchors,
        category_prototypes=category_prototypes,
        labels=labels
    )
    
    # Check outputs
    assert 'total_loss' in result
    assert 'cls_loss' in result
    assert 'local_loss' in result
    assert 'fusion_loss' in result
    assert 'prompt_loss' in result
    assert 'final_probs' in result
    assert 'compensated_features' in result
    assert 'missing_embeddings' in result
    
    assert result['final_probs'].shape == (batch_size, num_classes)
    assert result['compensated_features'].shape == (batch_size, embed_dim)
    assert result['missing_embeddings'].shape == (batch_size, embed_dim)
    
    print("完整 Multi-Channel Prompt Distillation 测试通过")


def test_missing_modalities():
    """Test handling of missing modalities"""
    print("测试缺失模态处理 ...")
    
    # Create distillation module
    distillation = MultiChannelPromptDistillation(
        embed_dim=256,
        num_classes=3,
        temperature=0.1,
        completion_weight=0.5,
        llm_model_name="t5-base",
        lambda_local=1.0,
        lambda_fusion=1.0,
        lambda_prompt=1.0,
        device="cpu"
    )
    
    # Create dummy data with missing modalities
    batch_size = 8
    embed_dim = 256
    num_classes = 3
    
    modal_features = {
        'text': torch.randn(batch_size, embed_dim),
        'audio': None,  # 缺失模态
        'vision': torch.randn(batch_size, embed_dim)
    }
    
    fused_features = torch.randn(batch_size, embed_dim)
    local_anchors = {
        'text': torch.randn(num_classes, embed_dim),
        'audio': torch.randn(num_classes, embed_dim),
        'vision': torch.randn(num_classes, embed_dim)
    }
    global_anchors = torch.randn(num_classes, embed_dim)
    category_prototypes = torch.randn(num_classes, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    result = distillation(
        modal_features=modal_features,
        fused_features=fused_features,
        local_anchors=local_anchors,
        global_anchors=global_anchors,
        category_prototypes=category_prototypes,
        labels=labels
    )
    
    # Check that it handles missing modalities gracefully
    assert 'total_loss' in result
    assert result['total_loss'].item() >= 0
    
    print("缺失模态处理测试通过")


if __name__ == "__main__":
    print("开始运行多通道提示蒸馏组件测试 ...\n")
    
    try:
        test_local_channel_distillation()
        test_fusion_channel_distillation()
        test_prompt_based_completion()
        test_missing_aware_classification()
        test_complete_prompt_distillation()
        test_missing_modalities()
        
        print("\n所有多通道提示蒸馏组件测试通过！")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        raise 