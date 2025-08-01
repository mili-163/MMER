# Multimodal Emotion Recognition Components

这个目录包含了多模态情感识别框架的所有核心组件。

## 已实现的组件

### 1. Structure-aware Representation Learning ✅
**文件**: `structure_aware.py`, `modality_encoder.py`, `fusion_module.py`, `semantic_graph.py`, `gcn_enhancement.py`

第一个关键组件，包含：
- **模态编码器**: 将不同模态特征投影到共享语义空间
- **融合模块**: 自适应权重的多模态融合
- **语义图构建**: 跨模态语义图构建（包含正确的类别中心计算）
- **GCN增强**: 基于GCN的结构感知表示增强

### 2. Dual-Level Semantic Anchoring ✅
**文件**: `llm_encoder.py`, `llm_prototype.py`, `semantic_anchor.py`, `dual_level_anchoring.py`

第二个关键组件，包含：
- **LLM编码器**: 支持T5、BERT等基础模型
- **LLM原型生成**: 通过可学习prompt + [MOD] + class描述生成类别原型
- **语义锚点计算**: 计算local anchors和global anchors
- **双级锚点模块**: 整合LLM原型和语义锚点

## 组件详细说明

### LLM相关组件

#### LLMEncoder (`llm_encoder.py`)
- **功能**: LLM编码器的基础实现
- **支持模型**: T5、BERT等
- **特点**: 冻结参数，支持文本编码和tokenized输入

#### LLMCategoryPrototype (`llm_prototype.py`)
- **功能**: 生成每个类别的LLM原型
- **实现**: 
  - 可学习prompt token + [MOD] + class描述
  - 支持文本prompt和learnable token两种模式
  - 输入冻结LLM encoder，输出类别原型
- **输出**: [num_classes, embed_dim]

#### DualLevelSemanticAnchoring (`semantic_anchor.py`)
- **功能**: 计算双级语义锚点
- **实现**:
  - 计算样本与类别原型的相似度
  - 基于Top-K高置信样本生成local anchors
  - 基于结构增强特征生成global anchors
- **输出**: local_anchors, global_anchors, category_prototypes

#### DualLevelSemanticAnchoringModule (`dual_level_anchoring.py`)
- **功能**: 完整的双级语义锚点模块
- **整合**: LLM原型生成和语义锚点计算
- **接口**: 提供完整的锚点接口供第三部分使用

## 关键特性

### 1. LLM集成
- 支持多种LLM基础模型（T5、BERT等）
- 可学习的prompt设计
- 冻结LLM参数，只训练prompt部分

### 2. 双级锚点设计
- **Local Anchors**: 每个模态的Top-K高置信样本均值
- **Global Anchors**: 结构增强特征的Top-K均值
- **Category Prototypes**: LLM驱动的类别原型

### 3. 接口设计
所有锚点和原型都可通过接口输出，供第三部分使用：
```python
{
    'local_anchors': {mod: [C, D]},
    'global_anchors': [C, D],
    'category_prototypes': [C, D],
    'all_prototypes': {mod: [C, D]}
}
```

## 使用方法

### 基本使用
```python
from .components import DualLevelSemanticAnchoringModule

# 创建模块
module = DualLevelSemanticAnchoringModule(
    num_classes=3,
    embed_dim=256,
    top_k=8,
    lambda_entropy=0.1,
    llm_model_name="t5-base",
    device="cpu"
)

# 前向传播
result = module.forward(
    modal_features=modal_features,
    fused_features=fused_features,
    pseudo_labels=pseudo_labels,
    fused_pseudo_label=fused_pseudo_label,
    modality='text'
)
```

### 获取锚点接口
```python
# 获取锚点接口供第三部分使用
interface = module.get_anchor_interface()
```

## 测试

运行测试脚本验证组件功能：
```bash
# 测试第一个组件
python test_structure_aware.py

# 测试第二个组件
python test_dual_level_anchoring.py
```

## 数学公式

### 类别原型生成
- **文本Prompt**: T_p^(c) = [P_1^(c) ... P_k^(c)] || [MOD] || [CLS_1^(c) ... CLS_s^(c)]
- **原型计算**: p_c = T_LLM(T_p^(c)) ∈ R^D

### 相似度计算
- **相似度**: s_{i,c}^(m) = ⟨z_i^(m), p_c⟩ / (||z_i^(m)|| · ||p_c||) + λH(α_i)
- **熵项**: H(α_i) = -Σ α_i log(α_i)

### 锚点计算
- **Local Anchors**: c_c^(m) = (1/|I_c^(m)|) Σ_{i∈I_c^(m)} z_i^(m)
- **Global Anchors**: c_c^(g) = (1/|I_c^(g)|) Σ_{i∈I_c^(g)} ũ_i

## 下一步

1. ✅ Structure-aware Representation Learning组件
2. ✅ Dual-Level Semantic Anchoring组件
3. ⏳ Anchor-driven multi-channel prompt distillation组件
4. ⏳ 整合所有组件到完整模型中 