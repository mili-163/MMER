# MMER: Multimodal Emotion Recognition Framework
Incomplete multimodal learning struggles with anchor drift. We propose AD-MCPD, using a modality-aware graph, dual-level LLM anchoring, and multi-channel distillation to mitigate drift and enhance stability. AD-MCPD excels on CMU-MOSI and CMU-MOSEI benchmarks.

## Project Overview

This project implements a comprehensive multimodal emotion recognition framework, consisting of three core components：

1. **Structure-aware Representation Learning** - Structure-aware representation learning
2. **LLM-guided Dual-Level Semantic Anchoring** - LLM-guided dual-level semantic anchoring
3. **Anchor-driven Multi-Channel Prompt Distillation** - Anchor-driven multi-channel prompt distillation

## Dataset Support

- **MOSI** (CMU Multimodal Opinion Sentiment and Intensity)
- **MOSEI** (CMU Multimodal Opinion Sentiment, Emotions and Attributes)

## Project Structure

MMER/
├── trains/singleTask/model/
│   ├── components/           # Core component modules
│   │   ├── structure_aware.py      # Structure-aware representation learning
│   │   ├── llm_encoder.py          # LLM encoder
│   │   ├── llm_prototype.py        # LLM prototype generation
│   │   ├── semantic_anchor.py      # Semantic anchoring
│   │   ├── dual_level_anchoring.py # Dual-level anchoring module
│   │   └── prompt_distillation.py  # Multi-channel prompt distillation
│   ├── main_model.py        # Main model
│   └── trainer.py           # Trainer
├── config/                  # Configuration files
├── data_loader.py           # Data loader
├── run.py                   # Main running script
└── train.py                 # Training script

## Core Components
1. Structure-aware Representation Learning
Modality encoding and feature projection
Adaptive multimodal fusion
Cross-modal semantic graph construction
GCN-enhanced representation
2. LLM-guided Dual-Level Semantic Anchoring
LLM category prototype construction
Local and global semantic anchoring
Similarity-based anchor selection
3. Anchor-driven Multi-Channel Prompt Distillation
Local channel distillation
Fusion channel distillation
Prompt-based compensation distillation
Missing-aware classification strategy
Usage

## Environment Setup
```bash
pip install torch transformers numpy
```

### Running Training
```bash
python train.py
```

### Testing Components
```bash
python test_structure_aware.py      # Test structure-aware component
python test_dual_level_anchoring.py # Test dual-level anchoring component
python test_prompt_distillation.py  # Test multi-channel distillation component
```

## Loss Functions

The framework includes 4 loss functions:
Classification Loss (CrossEntropyLoss) - Primary task loss
Local Distillation Loss - Alignment between unimodal features and local anchors
Fusion Distillation Loss - Alignment between fused features and global anchors
Prompt Distillation Loss - Alignment between LLM-generated compensations and fused features
