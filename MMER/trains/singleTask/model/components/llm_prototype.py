import torch
import torch.nn as nn
from typing import List, Optional
from .llm_encoder import create_llm_encoder

class LLMCategoryPrototype(nn.Module):
    """
    LLM类别原型生成器。
    通过可学习的prompt token + [MOD] + class描述，输入到冻结的LLM encoder，得到每个类别的prototype。
    """
    def __init__(self, num_classes: int, prompt_len: int, cls_len: int, embed_dim: int, 
                 llm_model_name: str = "t5-base", device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.prompt_len = prompt_len
        self.cls_len = cls_len
        self.embed_dim = embed_dim
        self.device = device
        
        # 创建LLM编码器
        self.llm_encoder = create_llm_encoder(llm_model_name, embed_dim, device)
        
        # 可学习prompt token
        self.learnable_prompt = nn.Parameter(torch.randn(num_classes, prompt_len, embed_dim))
        # 可学习class token
        self.learnable_cls = nn.Parameter(torch.randn(num_classes, cls_len, embed_dim))
        # [MOD] token（可选：可学习或固定）
        self.mod_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 类别描述文本（用于生成文本prompt）
        self.class_descriptions = {
            0: "negative emotion",
            1: "neutral emotion", 
            2: "positive emotion"
        }
        
        # 模态描述
        self.modality_descriptions = {
            'text': "This is a text.",
            'audio': "This is an audio.",
            'vision': "This is a video."
        }
    
    def create_text_prompts(self, modality: str = 'text') -> List[str]:
        """
        创建文本prompt
        Args:
            modality: 模态名称
        Returns:
            prompts: 每个类别的文本prompt列表
        """
        mod_desc = self.modality_descriptions.get(modality, "This is a multimodal input.")
        prompts = []
        
        for class_id in range(self.num_classes):
            class_desc = self.class_descriptions.get(class_id, f"class {class_id}")
            # 创建prompt: "This is a [modality]. The emotion is [class_description]."
            prompt = f"{mod_desc} The emotion is {class_desc}."
            prompts.append(prompt)
        
        return prompts
    
    def forward(self, modality: str = 'text', use_text_prompt: bool = True) -> torch.Tensor:
        """
        返回所有类别的prototype: [num_classes, embed_dim]
        Args:
            modality: 模态名称
            use_text_prompt: 是否使用文本prompt（True）或learnable token（False）
        """
        if use_text_prompt:
            # 使用文本prompt
            text_prompts = self.create_text_prompts(modality)
            # 通过LLM编码器编码文本
            prototypes = self.llm_encoder.llm_encoder.encode_text(text_prompts)
        else:
            # 使用learnable token
            # 拼接prompt: [P1...Pk] + [MOD] + [CLS1...CLSs]
            # shape: [num_classes, prompt_len+1+cls_len, embed_dim]
            prompt = torch.cat([
                self.learnable_prompt,  # [num_classes, prompt_len, embed_dim]
                self.mod_token.expand(self.num_classes, -1, -1),  # [num_classes, 1, embed_dim]
                self.learnable_cls  # [num_classes, cls_len, embed_dim]
            ], dim=1)
            
            # 通过LLM编码器
            prototypes = self.llm_encoder(prompt)
        
        return prototypes  # [num_classes, embed_dim]
    
    def get_prototype_for_modality(self, modality: str) -> torch.Tensor:
        """
        获取特定模态的类别原型
        Args:
            modality: 模态名称
        Returns:
            prototypes: [num_classes, embed_dim]
        """
        return self.forward(modality=modality, use_text_prompt=True)
    
    def get_all_modality_prototypes(self) -> dict:
        """
        获取所有模态的类别原型
        Returns:
            prototypes: {modality: [num_classes, embed_dim]}
        """
        prototypes = {}
        for modality in ['text', 'audio', 'vision']:
            prototypes[modality] = self.get_prototype_for_modality(modality)
        return prototypes 