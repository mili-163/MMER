import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .llm_prototype import LLMCategoryPrototype
from .semantic_anchor import DualLevelSemanticAnchoring

class DualLevelSemanticAnchoringModule(nn.Module):
    """
    完整的Dual-Level Semantic Anchoring模块
    整合LLM原型生成和双级语义锚点计算
    """
    def __init__(self, num_classes: int, embed_dim: int, top_k: int = 15, 
                 lambda_entropy: float = 0.1, llm_model_name: str = "t5-base", 
                 prompt_len: int = 4, cls_len: int = 2, device: str = 'cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.device = device
        
        # LLM类别原型生成器
        self.llm_prototype = LLMCategoryPrototype(
            num_classes=num_classes,
            prompt_len=prompt_len,
            cls_len=cls_len,
            embed_dim=embed_dim,
            llm_model_name=llm_model_name,
            device=device
        )
        
        # 双级语义锚点计算器
        self.semantic_anchor = DualLevelSemanticAnchoring(
            top_k=top_k,
            lambda_entropy=lambda_entropy
        )
        
        # 缓存所有模态的原型
        self.cached_prototypes = None
        
    def get_category_prototypes(self, modality: str = 'text') -> torch.Tensor:
        """
        获取类别原型
        Args:
            modality: 模态名称
        Returns:
            prototypes: [num_classes, embed_dim]
        """
        return self.llm_prototype.get_prototype_for_modality(modality)
    
    def get_all_prototypes(self) -> Dict[str, torch.Tensor]:
        """
        获取所有模态的类别原型
        Returns:
            prototypes: {modality: [num_classes, embed_dim]}
        """
        if self.cached_prototypes is None:
            self.cached_prototypes = self.llm_prototype.get_all_modality_prototypes()
        return self.cached_prototypes
    
    def compute_anchors(self, modal_features: Dict[str, torch.Tensor],
                       fused_features: torch.Tensor,
                       pseudo_labels: Dict[str, torch.Tensor],
                       fused_pseudo_label: torch.Tensor,
                       modality: str = 'text') -> Dict[str, torch.Tensor]:
        """
        计算双级语义锚点
        Args:
            modal_features: 各模态特征 {mod: [N, D]}
            fused_features: 融合特征 [N, D]
            pseudo_labels: 各模态伪标签 {mod: [N, C]}
            fused_pseudo_label: 融合伪标签 [N, C]
            modality: 主要模态（用于选择原型）
        Returns:
            anchors: 包含所有锚点的字典
        """
        # 获取类别原型
        category_prototypes = self.get_category_prototypes(modality)
        
        # 计算所有锚点
        anchors = self.semantic_anchor.get_all_anchors(
            modal_features=modal_features,
            fused_features=fused_features,
            category_prototypes=category_prototypes,
            pseudo_labels=pseudo_labels,
            fused_pseudo_label=fused_pseudo_label
        )
        
        return anchors
    
    def forward(self, modal_features: Dict[str, torch.Tensor],
                fused_features: torch.Tensor,
                pseudo_labels: Dict[str, torch.Tensor],
                fused_pseudo_label: torch.Tensor,
                modality: str = 'text') -> Dict[str, torch.Tensor]:
        """
        前向传播，返回所有锚点和原型
        Args:
            modal_features: 各模态特征
            fused_features: 融合特征
            pseudo_labels: 各模态伪标签
            fused_pseudo_label: 融合伪标签
            modality: 主要模态
        Returns:
            result: 包含所有锚点和原型的字典
        """
        # 计算锚点
        anchors = self.compute_anchors(
            modal_features, fused_features, pseudo_labels, fused_pseudo_label, modality
        )
        
        # 获取所有原型
        all_prototypes = self.get_all_prototypes()
        
        # 合并结果
        result = {
            'local_anchors': anchors['local_anchors'],
            'global_anchors': anchors['global_anchors'],
            'category_prototypes': anchors['category_prototypes'],
            'all_prototypes': all_prototypes
        }
        
        return result
    
    def get_anchor_interface(self) -> Dict[str, torch.Tensor]:
        """
        获取锚点接口，供第三部分使用
        Returns:
            interface: 包含所有锚点和原型的接口
        """
        # 这里返回一个示例接口，实际使用时需要传入真实数据
        return {
            'local_anchors': {},  # {mod: [C, D]}
            'global_anchors': None,  # [C, D]
            'category_prototypes': None,  # [C, D]
            'all_prototypes': {}  # {mod: [C, D]}
        } 