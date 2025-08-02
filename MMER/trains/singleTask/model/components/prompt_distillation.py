import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .llm_encoder import create_llm_encoder

class LocalChannelDistillation(nn.Module):
    """
    Local-Channel Distillation
    对齐单模态特征与local anchors以保留模态特定语义
    """
    def __init__(self, temperature: float = 0.7):
        super().__init__()
        self.temperature = temperature
    
    def compute_distribution(self, features: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        计算分类分布
        Args:
            features: [N, D] 样本特征
            anchors: [C, D] 类别锚点
        Returns:
            probs: [N, C] 分类概率
        """
        # 计算相似度
        sim = torch.matmul(features, anchors.t()) / self.temperature  # [N, C]
        # 添加数值稳定性
        sim = torch.clamp(sim, min=-10, max=10)
        probs = F.softmax(sim, dim=1)
        # 确保概率和为1且非零
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs
    
    def forward(self, modal_features: Dict[str, torch.Tensor], 
                local_anchors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算local-channel蒸馏损失
        按照论文公式：L_distill^local = Σ_i=1^N Σ_m=1^M KL(p_i^(m) || stop_grad(p_c^(m)))
        Args:
            modal_features: {mod: [N, D]} 各模态特征
            local_anchors: {mod: [C, D]} 各模态local anchors
        Returns:
            loss: 蒸馏损失
        """
        total_loss = 0.0
        num_modalities = 0
        
        for mod in modal_features:
            if mod in local_anchors and modal_features[mod] is not None:
                features = modal_features[mod]
                anchors = local_anchors[mod]
                
                # 计算学生分布 p_i^(m)
                student_probs = self.compute_distribution(features, anchors)
                
                # 计算教师分布 stop_grad(p_c^(m))
                with torch.no_grad():
                    teacher_probs = self.compute_distribution(features, anchors)
                
                # KL散度损失：KL(p_i^(m) || stop_grad(p_c^(m)))
                loss = F.kl_div(
                    torch.log(student_probs + 1e-8), 
                    teacher_probs, 
                    reduction='batchmean'
                )
                # 确保损失为非负且有限
                loss = torch.clamp(loss, min=0.0, max=100.0)
                total_loss += loss
                num_modalities += 1
        
        return total_loss / max(num_modalities, 1)


class FusionChannelDistillation(nn.Module):
    """
    Fusion-Channel Distillation
    对齐融合特征与global anchors以实现跨模态一致性
    """
    def __init__(self, temperature: float = 0.7):
        super().__init__()
        self.temperature = temperature
    
    def compute_distribution(self, features: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        计算分类分布
        Args:
            features: [N, D] 融合特征
            anchors: [C, D] global anchors
        Returns:
            probs: [N, C] 分类概率
        """
        sim = torch.matmul(features, anchors.t()) / self.temperature  # [N, C]
        # 添加数值稳定性
        sim = torch.clamp(sim, min=-10, max=10)
        probs = F.softmax(sim, dim=1)
        # 确保概率和为1且非零
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs
    
    def forward(self, fused_features: torch.Tensor, global_anchors: torch.Tensor) -> torch.Tensor:
        """
        计算fusion-channel蒸馏损失
        按照论文公式：L_distill^global = Σ_i=1^N KL(p_i^(g) || stop_grad(p_c^(g)))
        Args:
            fused_features: [N, D] 融合特征
            global_anchors: [C, D] global anchors
        Returns:
            loss: 蒸馏损失
        """
        # 计算学生分布 p_i^(g)
        student_probs = self.compute_distribution(fused_features, global_anchors)
        
        # 计算教师分布 stop_grad(p_c^(g))
        with torch.no_grad():
            teacher_probs = self.compute_distribution(fused_features, global_anchors)
        
        # KL散度损失：KL(p_i^(g) || stop_grad(p_c^(g)))
        loss = F.kl_div(
            torch.log(student_probs + 1e-8), 
            teacher_probs, 
            reduction='batchmean'
        )
        # 确保损失为非负且有限
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        return loss


class PromptBasedCompletion(nn.Module):
    """
    Prompt-Based Compensation Distillation
    利用LLM生成的prompt为不完整样本注入高级语义
    """
    def __init__(self, embed_dim: int, temperature: float = 0.7, 
                 llm_model_name: str = "t5-base", device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.device = device
        
        # 创建LLM编码器
        self.llm_encoder = create_llm_encoder(llm_model_name, embed_dim, device)
        
        # 缺失模态占位符
        self.missing_placeholder = nn.Parameter(torch.randn(1, embed_dim, device=device))
        
        # 类别prompt模板
        self.category_prompts = {
            0: "negative emotion",
            1: "neutral emotion", 
            2: "positive emotion"
        }
    
    def construct_input_sequence(self, modal_features: Dict[str, torch.Tensor], 
                               category_prototypes: torch.Tensor) -> torch.Tensor:
        """
        构建输入序列: S_i = [CLS] || {z_i^(m)}_m∈A_i || [H_miss] || T_p^(1) || ... || T_p^(C)
        按照论文公式实现
        Args:
            modal_features: {mod: [N, D]} 可用模态特征
            category_prototypes: [C, D] 类别原型
        Returns:
            sequence: [N, seq_len, D] 输入序列
        """
        batch_size = next(iter(modal_features.values())).size(0)
        num_classes = category_prototypes.size(0)
        
        # 收集可用模态特征 {z_i^(m)}_m∈A_i
        available_features = []
        for mod, features in modal_features.items():
            if features is not None:
                # 确保是3维张量 [N, 1, D]
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                available_features.append(features)
        
        if not available_features:
            # 没有可用模态，使用占位符
            placeholder = self.missing_placeholder.expand(batch_size, 1, -1)
            available_features = [placeholder]
        
        # 拼接特征: [CLS] + {z_i^(m)}_m∈A_i + [H_miss] + T_p^(1) + ... + T_p^(C)
        cls_token = torch.zeros(batch_size, 1, self.embed_dim, device=self.device)  # [CLS]
        missing_token = self.missing_placeholder.expand(batch_size, 1, -1)  # [H_miss]
        
        # 拼接所有部分
        sequence_parts = [cls_token] + available_features + [missing_token]
        
        # 添加类别原型 T_p^(1) + ... + T_p^(C)
        for c in range(num_classes):
            prototype = category_prototypes[c].expand(batch_size, 1, -1)
            sequence_parts.append(prototype)
        
        # 拼接序列
        sequence = torch.cat(sequence_parts, dim=1)  # [N, seq_len, D]
        
        return sequence
    
    def forward(self, modal_features: Dict[str, torch.Tensor], 
                global_anchors: torch.Tensor,
                category_prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算prompt-based补偿蒸馏损失
        按照论文公式：L_distill^prompt = Σ_i KL(q̃_i || p_i^(g))
        Args:
            modal_features: {mod: [N, D]} 各模态特征
            global_anchors: [C, D] global anchors
            category_prototypes: [C, D] 类别原型
        Returns:
            loss: 蒸馏损失
            missing_embeddings: [N, D] 缺失模态嵌入
        """
        batch_size = next(iter(modal_features.values())).size(0)
        
        # 构建输入序列 S_i = [CLS] || {z_i^(m)}_m∈A_i || [H_miss] || T_p^(1) || ... || T_p^(C)
        sequence = self.construct_input_sequence(modal_features, category_prototypes)
        
        # 通过LLM编码器生成缺失模态嵌入 ẑ_i^(miss)
        missing_embeddings = self.llm_encoder(sequence)  # [N, D]
        
        # 计算与global anchors的相似度，得到 q̃_i
        sim = torch.matmul(missing_embeddings, global_anchors.t()) / self.temperature  # [N, C]
        sim = torch.clamp(sim, min=-10, max=10)
        q_probs = F.softmax(sim, dim=1)
        q_probs = torch.clamp(q_probs, min=1e-8, max=1.0)
        q_probs = q_probs / q_probs.sum(dim=1, keepdim=True)
        
        # 计算融合特征的分布 p_i^(g)（作为教师）
        fused_features = torch.stack([f for f in modal_features.values() if f is not None]).mean(dim=0)
        sim_fused = torch.matmul(fused_features, global_anchors.t()) / self.temperature
        sim_fused = torch.clamp(sim_fused, min=-10, max=10)
        p_probs = F.softmax(sim_fused, dim=1)
        p_probs = torch.clamp(p_probs, min=1e-8, max=1.0)
        p_probs = p_probs / p_probs.sum(dim=1, keepdim=True)
        
        # KL散度损失：KL(q̃_i || p_i^(g))
        loss = F.kl_div(torch.log(q_probs + 1e-8), p_probs, reduction='batchmean')
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        return loss, missing_embeddings


class MissingAwareClassification(nn.Module):
    """
    Missing-Aware Classification Strategy
    引入缺失标志和补偿权重，按照论文公式实现
    """
    def __init__(self, temperature: float = 0.7, completion_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.completion_weight = completion_weight
    
    def compute_missing_flags(self, modal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算缺失标志 δ_i ∈ {0,1}
        Args:
            modal_features: {mod: [N, D]} 各模态特征
        Returns:
            flags: [N] 缺失标志 (1表示有缺失模态)
        """
        # 获取设备信息
        device = next(iter(modal_features.values())).device
        batch_size = next(iter(modal_features.values())).size(0)
        flags = torch.zeros(batch_size, device=device)
        
        # 检查每个样本是否有缺失模态
        for i in range(batch_size):
            missing_count = 0
            total_modalities = len(modal_features)
            
            for mod, features in modal_features.items():
                if features is None:
                    missing_count += 1
            
            # 如果有缺失模态，设置标志为1
            if missing_count > 0:
                flags[i] = 1
        
        return flags
    
    def forward(self, fused_features: torch.Tensor, 
                missing_embeddings: torch.Tensor,
                modal_features: Dict[str, torch.Tensor],
                global_anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算缺失感知分类，按照论文公式：
        ū_i = (1-δ_i)ũ_i + δ_i(ũ_i + γẑ_i^(miss))
        Args:
            fused_features: [N, D] 融合特征 ũ_i
            missing_embeddings: [N, D] 缺失模态嵌入 ẑ_i^(miss)
            modal_features: {mod: [N, D]} 各模态特征
            global_anchors: [C, D] global anchors
        Returns:
            final_probs: [N, C] 最终分类概率
            compensated_features: [N, D] 补偿后的特征
        """
        # 计算缺失标志 δ_i
        missing_flags = self.compute_missing_flags(modal_features)
        
        # 融合补偿表示：ū_i = (1-δ_i)ũ_i + δ_i(ũ_i + γẑ_i^(miss))
        # 等价于：ū_i = ũ_i + δ_i * γ * ẑ_i^(miss)
        compensated_features = fused_features + self.completion_weight * missing_embeddings * missing_flags.unsqueeze(1)
        
        # 计算最终分类概率
        sim = torch.matmul(compensated_features, global_anchors.t()) / self.temperature
        sim = torch.clamp(sim, min=-10, max=10)
        final_probs = F.softmax(sim, dim=1)
        final_probs = torch.clamp(final_probs, min=1e-8, max=1.0)
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True)
        
        return final_probs, compensated_features


class MultiChannelPromptDistillation(nn.Module):
    """
    完整的Multi-Channel Prompt Distillation模块
    """
    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.7,
                 completion_weight: float = 0.3, llm_model_name: str = "t5-base",
                 lambda_local: float = 1.0, lambda_fusion: float = 1.0, 
                 lambda_prompt: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.device = device
        
        # 三个蒸馏分支
        self.local_distillation = LocalChannelDistillation(temperature)
        self.fusion_distillation = FusionChannelDistillation(temperature)
        self.prompt_completion = PromptBasedCompletion(embed_dim, temperature, llm_model_name, device)
        self.missing_classification = MissingAwareClassification(temperature, completion_weight)
        
        # 损失权重
        self.lambda_local = lambda_local
        self.lambda_fusion = lambda_fusion
        self.lambda_prompt = lambda_prompt
        
        # 分类损失
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(self, modal_features: Dict[str, torch.Tensor],
                fused_features: torch.Tensor,
                local_anchors: Dict[str, torch.Tensor],
                global_anchors: torch.Tensor,
                category_prototypes: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算所有损失
        按照论文公式：L_total = L_cls^(fusion) + Σ_k λ_k L_distill^k
        Args:
            modal_features: {mod: [N, D]} 各模态特征
            fused_features: [N, D] 融合特征
            local_anchors: {mod: [C, D]} local anchors
            global_anchors: [C, D] global anchors
            category_prototypes: [C, D] 类别原型
            labels: [N] 真实标签
        Returns:
            losses: 包含所有损失的字典
        """
        # 1. Local-Channel Distillation: L_distill^local
        local_loss = self.local_distillation(modal_features, local_anchors)
        
        # 2. Fusion-Channel Distillation: L_distill^global
        fusion_loss = self.fusion_distillation(fused_features, global_anchors)
        
        # 3. Prompt-Based Completion: L_distill^prompt
        prompt_loss, missing_embeddings = self.prompt_completion(
            modal_features, global_anchors, category_prototypes
        )
        
        # 4. Missing-Aware Classification: L_cls^(fusion)
        final_probs, compensated_features = self.missing_classification(
            fused_features, missing_embeddings, modal_features, global_anchors
        )
        
        # 分类损失：L_cls^(fusion) = -Σ_i=1^N log p_i^(final)(y_i)
        cls_loss = self.classification_loss(final_probs, labels)
        
        # 总损失：L_total = L_cls^(fusion) + λ_local * L_distill^local + λ_fusion * L_distill^global + λ_prompt * L_distill^prompt
        total_loss = (cls_loss + 
                     self.lambda_local * local_loss +
                     self.lambda_fusion * fusion_loss +
                     self.lambda_prompt * prompt_loss)
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'local_loss': local_loss,
            'fusion_loss': fusion_loss,
            'prompt_loss': prompt_loss,
            'final_probs': final_probs,
            'compensated_features': compensated_features,
            'missing_embeddings': missing_embeddings
        } 