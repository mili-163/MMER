#!/usr/bin/env python3
"""
Training script for MMER (Multimodal Emotion Recognition)
"""

from run import MMER_run

if __name__ == "__main__":
    # Run MMER training with default parameters
    MMER_run(
        model_name='mmer',
        dataset_name='mosi',
        seeds=[1111, 2222, 3333, 4444, 5555],
        mr=0.1
    )
