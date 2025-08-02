import torch
import torch.nn as nn
from typing import Dict, Any
from .components import (
    StructureAwareRepresentationLearning,
    DualLevelSemanticAnchoringModule,
    MultiChannelPromptDistillation
)

def get_modality_names():
    return ['text', 'audio', 'vision']

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

class MMERMainModel(nn.Module):
    """
    主模型：多模态情感识别，组合结构感知、锚定、蒸馏等模块
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = getattr(args, 'device', get_device())
        self.feature_dims = args.feature_dims
        self.num_classes = getattr(args, 'num_classes', 3)
        self.shared_dim = getattr(args, 'shared_dim', 512)
        self.temperature = getattr(args, 'temperature', 0.7)
        self.lambda_smooth = getattr(args, 'lambda_smooth', 0.1)
        self.delta_threshold = getattr(args, 'delta_threshold', 1.5)
        self.beta = getattr(args, 'beta', 0.5)
        self.top_k = getattr(args, 'top_k', 15)
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.1)
        self.completion_weight = getattr(args, 'completion_weight', 0.3)
        self.lambda_local = getattr(args, 'lambda_local', 1.0)
        self.lambda_fusion = getattr(args, 'lambda_fusion', 1.0)
        self.lambda_prompt = getattr(args, 'lambda_prompt', 0.5)
        self.llm_model_name = getattr(args, 'llm_model_name', 't5-base')
        self.prompt_len = getattr(args, 'prompt_len', 4)
        self.cls_len = getattr(args, 'cls_len', 2)

        # 结构感知表示学习
        self.structure_aware = StructureAwareRepresentationLearning(
            feature_dims=self.feature_dims,
            shared_dim=self.shared_dim,
            lambda_smooth=self.lambda_smooth,
            delta_threshold=self.delta_threshold,
            beta=self.beta
        )
        # 双层锚定
        self.dual_level_anchoring = DualLevelSemanticAnchoringModule(
            num_classes=self.num_classes,
            embed_dim=self.shared_dim,
            top_k=self.top_k,
            lambda_entropy=self.lambda_entropy,
            llm_model_name=self.llm_model_name,
            prompt_len=self.prompt_len,
            cls_len=self.cls_len,
            device=self.device
        )
        # 多通道提示蒸馏
        self.prompt_distillation = MultiChannelPromptDistillation(
            embed_dim=self.shared_dim,
            num_classes=self.num_classes,
            temperature=self.temperature,
            completion_weight=self.completion_weight,
            llm_model_name=self.llm_model_name,
            lambda_local=self.lambda_local,
            lambda_fusion=self.lambda_fusion,
            lambda_prompt=self.lambda_prompt,
            device=self.device
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # 1. 获取输入特征
        features = {mod: batch.get(mod, None) for mod in get_modality_names()}
        labels = batch['labels']['M'].squeeze(-1).long() if 'labels' in batch and 'M' in batch['labels'] else None

        # 2. 结构感知特征
        enhanced_features = self.structure_aware(features)  # [B, D]

        # 3. 编码器输出
        encoded_features, pseudo_labels = self.structure_aware.modality_encoder(features)
        fused_features = self.structure_aware.fused_representation(encoded_features)
        fused_pseudo_label = None
        for v in pseudo_labels.values():
            if v is not None:
                fused_pseudo_label = v
                break
        if fused_pseudo_label is None:
            fused_pseudo_label = torch.zeros(fused_features.size(0), self.num_classes, device=fused_features.device)

        # 4. 锚点
        anchor_result = self.dual_level_anchoring(
            modal_features=encoded_features,
            fused_features=fused_features,
            pseudo_labels=pseudo_labels,
            fused_pseudo_label=fused_pseudo_label,
            modality='text'  # 可根据主模态调整
        )
        local_anchors = anchor_result['local_anchors']
        global_anchors = anchor_result['global_anchors']
        category_prototypes = anchor_result['category_prototypes']

        # 5. 蒸馏与分类
        distill_result = self.prompt_distillation(
            modal_features=encoded_features,
            fused_features=fused_features,
            local_anchors=local_anchors,
            global_anchors=global_anchors,
            category_prototypes=category_prototypes,
            labels=labels
        )

        # 6. 输出
        return {
            'loss': distill_result['total_loss'],
            'predictions': distill_result['final_probs'],
            'cls_loss': distill_result['cls_loss'],
            'local_loss': distill_result['local_loss'],
            'fusion_loss': distill_result['fusion_loss'],
            'prompt_loss': distill_result['prompt_loss']
        }

def create_model(args):
    """
    工厂函数，返回主模型实例
    """
    return MMERMainModel(args)
