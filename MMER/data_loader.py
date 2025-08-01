import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional


class MMERDataset(Dataset):
    """Dataset for MMER (Multimodal Emotion Recognition)"""
    
    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = data_path
        self.split = split
        
        # Load data
        self.data = self.load_data()
        
    def load_data(self):
        """Load dataset from pickle file"""
        file_path = os.path.join(self.data_path, f"{self.split}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract features
        text_feat = torch.tensor(item['text'], dtype=torch.float32) if 'text' in item else None
        audio_feat = torch.tensor(item['audio'], dtype=torch.float32) if 'audio' in item else None
        vision_feat = torch.tensor(item['vision'], dtype=torch.float32) if 'vision' in item else None
        
        # Extract label
        label = torch.tensor(item['label'], dtype=torch.float32)
        
        return {
            'text': text_feat,
            'audio': audio_feat,
            'vision': vision_feat,
            'labels': {'M': label}
        }


class MMERDataLoader:
    """Data loader for MMER experiments"""
    
    def __init__(self, dataset_name: str, config: Dict):
        self.dataset_name = dataset_name
        self.config = config
        self.data_path = config['data_path']
        self.batch_size = config.get('batch_size', 32)
        
        # Create datasets
        self.train_dataset = MMERDataset(self.data_path, 'train')
        self.val_dataset = MMERDataset(self.data_path, 'val')
        self.test_dataset = MMERDataset(self.data_path, 'test')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """Get train and validation data loaders"""
        return {
            'train': self.train_loader,
            'val': self.val_loader
        }
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        return self.test_loader
    
    def get_feature_dims(self) -> List[int]:
        """Get feature dimensions for each modality"""
        return self.config['feature_dims']
