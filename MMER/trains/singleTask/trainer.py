import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List
import numpy as np
import os
import json
from datetime import datetime


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class MMERTrainer:
    """
    Trainer for MMER (Multimodal Emotion Recognition) model
    """
    
    def __init__(self, args):
        self.args = args
        self.device = getattr(args, 'device', get_device())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = getattr(args, 'early_stop_patience', 10)
        
        # Print device information
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Create output directories
        self.output_dir = f"results/mmer_{getattr(args, 'dataset_name', 'mosi')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_model(self, model):
        """Setup model, optimizer, and scheduler"""
        self.model = model.to(self.device)
        
        # Setup optimizer
        if hasattr(self.args, 'optimizer') and self.args.optimizer.lower() == 'adamw':
            # Use different learning rates for BERT and other components
            bert_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if 'bert' in name.lower() or 'text' in name.lower():
                    bert_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': bert_params, 'lr': getattr(self.args, 'bert_learning_rate', 5e-5)},
                {'params': other_params, 'lr': self.args.learning_rate}
            ]
            
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=getattr(self.args, 'weight_decay', 1e-5)
            )
        elif hasattr(self.args, 'optimizer') and self.args.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=getattr(self.args, 'weight_decay', 1e-5)
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=getattr(self.args, 'weight_decay', 1e-5)
            )
        
        # Setup scheduler
        if hasattr(self.args, 'scheduler') and self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(self.args, 'step_size', 10),
                gamma=getattr(self.args, 'gamma', 0.5)
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
    
    def move_to_device(self, data):
        """Move data to the appropriate device"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.move_to_device(item) for item in data]
        else:
            return data
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = self.move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.args, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self.move_to_device(batch)
                
                # Forward pass
                output = self.model(batch)
                loss = output['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def do_train(self, model, dataloader: Dict[str, DataLoader], return_epoch_results: bool = False) -> Dict[str, Any]:
        """Main training loop"""
        self.setup_model(model)
        
        train_loader = dataloader['train']
        val_loader = dataloader['val']
        
        epoch_results = []
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        
        print(f"Starting training for {self.args.num_epochs} epochs...")
        
        for epoch in range(self.args.num_epochs):
            # Training
            train_results = self.train_epoch(train_loader)
            
            # Validation
            val_results = self.validate_epoch(val_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_results['val_loss'])
            else:
                self.scheduler.step()
            
            # Early stopping
            if val_results['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_results['val_loss']
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Epoch {epoch+1}: New best model saved (val_loss: {val_results['val_loss']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log progress
            print(f"Epoch {epoch+1}/{self.args.num_epochs}: "
                  f"Train Loss: {train_results['train_loss']:.4f}, "
                  f"Val Loss: {val_results['val_loss']:.4f}")
            
            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_results['train_loss'],
                'val_loss': val_results['val_loss']
            })
        
        # Load best model
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Loaded best model from {best_model_path}")
        
        if return_epoch_results:
            return {'epoch_results': epoch_results}
        else:
            return {'final_val_loss': self.best_val_loss}
    
    def test_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Test for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self.move_to_device(batch)
                
                # Forward pass
                output = self.model(batch)
                loss = output['loss']
                
                # Collect predictions and labels
                predictions = output['predictions']
                labels = batch['labels']['M'].squeeze(-1).long()
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate metrics
        predicted_labels = torch.argmax(all_predictions, dim=1)
        accuracy = (predicted_labels == all_labels).float().mean().item()
        
        avg_loss = total_loss / num_batches
        
        return {
            'test_loss': avg_loss,
            'accuracy': accuracy
        }
    
    def do_test(self, model, dataloader: DataLoader, mode: str = "TEST") -> Dict[str, float]:
        """Test the model"""
        self.model = model.to(self.device)
        results = self.test_epoch(dataloader)
        
        print(f"{mode} Results:")
        print(f"Loss: {results['test_loss']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        return results 