#!/usr/bin/env python3
"""
Main script for running MMER (Multimodal Emotion Recognition) experiments
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import MMERDataLoader
from trains.ATIO import ATIO
from trains.singleTask.model import create_model


def MMER_run(model_name='mmer', dataset_name='mosi', seeds=[1111, 2222, 3333, 4444, 5555], mr=0.1):
    """
    Run MMER experiments
    
    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset (mosi/mosei)
        seeds: List of random seeds for reproducibility
        mr: Missing rate for simulating missing modalities
    """
    print(f"Running MMER experiments with model: {model_name}, dataset: {dataset_name}")
    
    # Load configuration
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if model_name not in config:
        print(f"Error: Model '{model_name}' not found in config")
        return
    
    model_config = config[model_name]
    
    # Run experiments for each seed
    results = []
    for seed in seeds:
        print(f"\nRunning experiment with seed: {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create data loader
        data_loader = MMERDataLoader(dataset_name, model_config['datasetParams'][dataset_name])
        
        # Create model
        args = argparse.Namespace(**model_config['commonParams'])
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.feature_dims = data_loader.get_feature_dims()
        args.num_classes = 3  # For sentiment classification
        
        model = create_model(args)
        model.to(args.device)
        
        # Create trainer
        trainer = ATIO()
        
        # Train and test
        try:
            # Training
            print("Starting training...")
            train_results = trainer.do_train(model, data_loader.get_data_loaders(), return_epoch_results=True)
            
            # Testing
            print("Starting testing...")
            test_results = trainer.do_test(model, data_loader.get_test_loader())
            
            # Save results
            result = {
                'seed': seed,
                'train_results': train_results,
                'test_results': test_results
            }
            results.append(result)
            
            print(f"Seed {seed} completed successfully")
            
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            continue
    
    # Print summary
    print(f"\nExperiment Summary:")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Seeds: {seeds}")
    print(f"Completed runs: {len(results)}/{len(seeds)}")
    
    # Save results
    output_dir = f"results/{model_name}_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MMER experiments')
    parser.add_argument('--model', type=str, default='mmer', help='Model name')
    parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444, 5555], help='Random seeds')
    parser.add_argument('--mr', type=float, default=0.1, help='Missing rate')
    
    args = parser.parse_args()
    
    MMER_run(
        model_name=args.model,
        dataset_name=args.dataset,
        seeds=args.seeds,
        mr=args.mr
    )