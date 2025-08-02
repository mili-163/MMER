import torch
import torch.nn as nn
from transformers import T5EncoderModel, BertModel, AutoTokenizer, AutoModel
from typing import Optional, Union

class LLMEncoder(nn.Module):
    """
    LLM编码器的基础实现，支持T5、BERT等模型
    """
    def __init__(self, model_name: str = "t5-base", max_length: int = 512, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # 根据模型名称加载对应的encoder
        if "t5" in model_name.lower():
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif "bert" in model_name.lower():
            self.encoder = BertModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # 通用AutoModel
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 冻结参数并移动到设备
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()
        self.encoder.to(device)
        
        # 获取embedding维度
        if hasattr(self.encoder, 'config'):
            self.embed_dim = self.encoder.config.hidden_size
        else:
            self.embed_dim = 768  # 默认维度
    
    def encode_text(self, texts: list) -> torch.Tensor:
        """
        编码文本列表
        Args:
            texts: 文本列表
        Returns:
            embeddings: [batch_size, embed_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            if "t5" in self.model_name.lower():
                outputs = self.encoder(**inputs)
                # T5使用last_hidden_state
                embeddings = outputs.last_hidden_state[:, 0, :]  # 取第一个token
            else:
                outputs = self.encoder(**inputs)
                # BERT等使用last_hidden_state
                embeddings = outputs.last_hidden_state[:, 0, :]  # 取[CLS] token
        
        return embeddings
    
    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        直接forward，用于处理tokenized输入
        """
        with torch.no_grad():
            if "t5" in self.model_name.lower():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state[:, 0, :]  # 取第一个token
            else:
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                return outputs.last_hidden_state[:, 0, :]  # 取[CLS] token


class PromptLLMEncoder(nn.Module):
    """
    专门用于处理prompt的LLM编码器
    """
    def __init__(self, model_name: str = "t5-base", embed_dim: int = 768, device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        
        # 创建LLM编码器
        self.llm_encoder = LLMEncoder(model_name, device=device)
        
        # 投影层，将LLM输出投影到指定维度
        self.projection = nn.Linear(self.llm_encoder.embed_dim, embed_dim)
        
    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        处理prompt embeddings
        Args:
            prompt_embeddings: [batch_size, seq_len, embed_dim]
        Returns:
            encoded: [batch_size, embed_dim]
        """
        # 这里我们假设prompt_embeddings已经是tokenized的形式
        # 在实际使用中，可能需要先转换为token ids
        batch_size, seq_len, _ = prompt_embeddings.shape
        
        # 创建attention mask
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        # 通过LLM编码器
        with torch.no_grad():
            encoded = self.llm_encoder(
                input_ids=None,  # 这里我们直接使用embeddings
                attention_mask=attention_mask
            )
        
        # 投影到指定维度
        encoded = self.projection(encoded)
        
        return encoded


def create_llm_encoder(model_name: str = "t5-base", embed_dim: int = 768, device: str = "cpu") -> PromptLLMEncoder:
    """
    创建LLM编码器的工厂函数
    """
    return PromptLLMEncoder(model_name, embed_dim, device) 