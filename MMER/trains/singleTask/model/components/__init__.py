# Core components for the multimodal emotion recognition framework
from .modality_encoder import ModalityEncoder
from .fusion_module import FusedMultimodalRepresentation
from .semantic_graph import CrossModalSemanticGraph, ClassCenterCalculator
from .gcn_enhancement import StructureAwareGCN
from .structure_aware import StructureAwareRepresentationLearning
from .llm_encoder import LLMEncoder, PromptLLMEncoder, create_llm_encoder
from .llm_prototype import LLMCategoryPrototype
from .semantic_anchor import DualLevelSemanticAnchoring
from .dual_level_anchoring import DualLevelSemanticAnchoringModule
from .prompt_distillation import (
    LocalChannelDistillation,
    FusionChannelDistillation,
    PromptBasedCompletion,
    MissingAwareClassification,
    MultiChannelPromptDistillation
)

__all__ = [
    'ModalityEncoder',
    'FusedMultimodalRepresentation', 
    'CrossModalSemanticGraph',
    'ClassCenterCalculator',
    'StructureAwareGCN',
    'StructureAwareRepresentationLearning',
    'LLMEncoder',
    'PromptLLMEncoder',
    'create_llm_encoder',
    'LLMCategoryPrototype',
    'DualLevelSemanticAnchoring',
    'DualLevelSemanticAnchoringModule',
    'LocalChannelDistillation',
    'FusionChannelDistillation',
    'PromptBasedCompletion',
    'MissingAwareClassification',
    'MultiChannelPromptDistillation'
] 