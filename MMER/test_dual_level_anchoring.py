#!/usr/bin/env python3
"""
Test script for Dual-Level Semantic Anchoring components
"""

import torch
import torch.nn as nn
import numpy as np
from trains.singleTask.model.components import (
    LLMEncoder,
    PromptLLMEncoder,
    create_llm_encoder,
    LLMCategoryPrototype,
    DualLevelSemanticAnchoring,
    DualLevelSemanticAnchoringModule
)


def test_llm_encoder():
    """Test LLM Encoder"""
    print("测试 LLM Encoder ...")
    
    # Create LLM encoder
    llm_encoder = create_llm_encoder("t5-base", embed_dim=256, device="cpu")
    
    # Create dummy prompt embeddings
    batch_size = 3
    seq_len = 10
    embed_dim = 256
    prompt_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    encoded = llm_encoder(prompt_embeddings)
    
    # Check output
    assert encoded.shape == (batch_size, embed_dim)
    
    print("LLM Encoder 测试通过")


def test_llm_prototype():
    """Test LLM Category Prototype"""
    print("测试 LLM Category Prototype ...")
    
    # Create prototype generator
    prototype_gen = LLMCategoryPrototype(
        num_classes=3,
        prompt_len=4,
        cls_len=2,
        embed_dim=256,
        llm_model_name="t5-base",
        device="cpu"
    )
    
    # Test text prompt generation
    text_prompts = prototype_gen.create_text_prompts("text")
    assert len(text_prompts) == 3
    
    # Test prototype generation
    prototypes = prototype_gen.forward(modality="text", use_text_prompt=True)
    assert prototypes.shape == (3, 256)
    
    # Test all modality prototypes
    all_prototypes = prototype_gen.get_all_modality_prototypes()
    assert len(all_prototypes) == 3
    assert all_prototypes['text'].shape == (3, 256)
    
    print("LLM Category Prototype 测试通过")


def test_semantic_anchoring():
    """Test Dual-Level Semantic Anchoring"""
    print("测试 Dual-Level Semantic Anchoring ...")
    
    # Create semantic anchoring
    anchoring = DualLevelSemanticAnchoring(top_k=4, lambda_entropy=0.1)
    
    # Create dummy data
    batch_size = 8
    num_classes = 3
    embed_dim = 256
    
    z = torch.randn(batch_size, embed_dim)
    p = torch.randn(num_classes, embed_dim)
    alpha = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    
    # Test similarity computation
    sim = anchoring.compute_similarity(z, p, alpha)
    assert sim.shape == (batch_size, num_classes)
    
    # Test local anchors
    local_anchors = anchoring.compute_local_anchors(z, p, alpha)
    assert local_anchors.shape == (num_classes, embed_dim)
    
    # Test global anchors
    global_anchors = anchoring.compute_global_anchors(z, p, alpha)
    assert global_anchors.shape == (num_classes, embed_dim)
    
    print("Dual-Level Semantic Anchoring 测试通过")


def test_dual_level_module():
    """Test complete Dual-Level Semantic Anchoring Module"""
    print("测试完整 Dual-Level Semantic Anchoring Module ...")
    
    # Create module
    module = DualLevelSemanticAnchoringModule(
        num_classes=3,
        embed_dim=256,
        top_k=4,
        lambda_entropy=0.1,
        llm_model_name="t5-base",
        device="cpu"
    )
    
    # Create dummy data
    batch_size = 8
    embed_dim = 256
    
    modal_features = {
        'text': torch.randn(batch_size, embed_dim),
        'audio': torch.randn(batch_size, embed_dim),
        'vision': torch.randn(batch_size, embed_dim)
    }
    
    fused_features = torch.randn(batch_size, embed_dim)
    
    pseudo_labels = {
        'text': torch.softmax(torch.randn(batch_size, 3), dim=1),
        'audio': torch.softmax(torch.randn(batch_size, 3), dim=1),
        'vision': torch.softmax(torch.randn(batch_size, 3), dim=1)
    }
    
    fused_pseudo_label = torch.softmax(torch.randn(batch_size, 3), dim=1)
    
    # Forward pass
    result = module.forward(
        modal_features=modal_features,
        fused_features=fused_features,
        pseudo_labels=pseudo_labels,
        fused_pseudo_label=fused_pseudo_label,
        modality='text'
    )
    
    # Check outputs
    assert 'local_anchors' in result
    assert 'global_anchors' in result
    assert 'category_prototypes' in result
    assert 'all_prototypes' in result
    
    assert result['global_anchors'].shape == (3, embed_dim)
    assert result['category_prototypes'].shape == (3, embed_dim)
    
    print("完整 Dual-Level Semantic Anchoring Module 测试通过")


def test_interface():
    """Test anchor interface for third component"""
    print("测试锚点接口 ...")
    
    # Create module
    module = DualLevelSemanticAnchoringModule(
        num_classes=3,
        embed_dim=256,
        top_k=4,
        lambda_entropy=0.1,
        llm_model_name="t5-base",
        device="cpu"
    )
    
    # Get interface
    interface = module.get_anchor_interface()
    
    # Check interface structure
    assert 'local_anchors' in interface
    assert 'global_anchors' in interface
    assert 'category_prototypes' in interface
    assert 'all_prototypes' in interface
    
    print("锚点接口测试通过")


if __name__ == "__main__":
    print("开始运行双层锚定组件测试 ...\n")
    
    try:
        test_llm_encoder()
        test_llm_prototype()
        test_semantic_anchoring()
        test_dual_level_module()
        test_interface()
        
        print("\n所有双层锚定组件测试通过！")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        raise 