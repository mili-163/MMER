import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import BertModel, ViTModel, Wav2Vec2Model, AutoTokenizer, AutoFeatureExtractor

class ModalityEncoder(nn.Module):
    """Modality-specific encoders with frozen pre-trained models and lightweight adapters"""
    
    def __init__(self, feature_dims: List[int], shared_dim: int = 512, 
                 use_pretrained: bool = True, device: str = 'cpu'):
        super(ModalityEncoder, self).__init__()
        self.shared_dim = shared_dim
        self.num_modalities = len(feature_dims)
        self.use_pretrained = use_pretrained
        self.device = device
        
        # 预训练编码器（冻结）
        if use_pretrained:
            self.pretrained_encoders = nn.ModuleDict({
                'text': BertModel.from_pretrained('bert-base-uncased'),
                'vision': ViTModel.from_pretrained('google/vit-base-patch16-224'),
                'audio': Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
            })
            
            # 冻结预训练参数并移动到指定设备
            for encoder in self.pretrained_encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False
                encoder.eval()
                encoder.to(device)
        
        # Lightweight adapters for each modality
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(shared_dim, shared_dim)
            ) for dim in feature_dims
        ])
        
        # Modality-specific classifiers for pseudo-label generation
        self.classifiers = nn.ModuleList([
            nn.Linear(shared_dim, 3)  # 3 classes for emotion recognition
            for _ in range(self.num_modalities)
        ])
        
        # 温度参数
        self.temperature = 0.7
    
    def encode_with_pretrained(self, modality: str, features: torch.Tensor) -> torch.Tensor:
        """
        使用预训练编码器编码特征
        Args:
            modality: 模态名称
            features: 输入特征
        Returns:
            encoded: 编码后的特征
        """
        if not self.use_pretrained or modality not in self.pretrained_encoders:
            return features
        
        encoder = self.pretrained_encoders[modality]
        
        with torch.no_grad():
            if modality == 'text':
                # 文本编码
                outputs = encoder(input_ids=features)
                encoded = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            elif modality == 'vision':
                # 视觉编码
                outputs = encoder(pixel_values=features)
                encoded = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            elif modality == 'audio':
                # 音频编码
                outputs = encoder(input_values=features)
                encoded = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            else:
                encoded = features
        
        return encoded
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode modality-specific features and generate pseudo-labels
        
        Args:
            features: Dict with keys 'text', 'audio', 'vision' containing modality features
            
        Returns:
            encoded_features: Dict of aligned embeddings
            pseudo_labels: Dict of pseudo-label probabilities
        """
        encoded_features = {}
        pseudo_labels = {}
        
        modality_names = ['text', 'audio', 'vision']
        
        for i, modality in enumerate(modality_names):
            if modality in features and features[modality] is not None:
                # 1. 预训练编码（如果启用）
                if self.use_pretrained:
                    pretrained_features = self.encode_with_pretrained(modality, features[modality])
                else:
                    pretrained_features = features[modality]
                
                # 2. 轻量级适配器投影
                encoded = self.adapters[i](pretrained_features)
                encoded_features[modality] = encoded
                
                # 3. 生成伪标签
                logits = self.classifiers[i](encoded)
                probs = F.softmax(logits / self.temperature, dim=-1)  # 使用温度参数
                pseudo_labels[modality] = probs
            else:
                encoded_features[modality] = None
                pseudo_labels[modality] = None
        
        return encoded_features, pseudo_labels 