"""
Training module for SA-Test-Fix.

This module implements multi-stage training for repairing sentiment analysis models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm

from .model_retrainer import ModelRetrainer, MixedLoss

class MultiStageTrainer:
    """Multi-stage trainer for sentiment analysis models."""
    
    def __init__(self, model, device=None, total_epochs: int = 30,
                 rep_phase_ratio: float = 0.3, joint_phase_ratio: float = 0.4,
                 initial_lr: float = 2e-5, final_lr: float = 5e-6,
                 temperature: float = 0.1, initial_lambda: float = 0.1,
                 final_lambda: float = 0.9, rebuild_interval: int = 1):
        """
        Initialize the multi-stage trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
            total_epochs: Total number of training epochs
            rep_phase_ratio: Ratio of epochs for representation learning phase
            joint_phase_ratio: Ratio of epochs for joint optimization phase
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            temperature: Temperature parameter for contrastive loss
            initial_lambda: Initial weight for classification loss
            final_lambda: Final weight for classification loss
            rebuild_interval: Interval for rebuilding sample library
        """
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.total_epochs = total_epochs
        
        # Phase durations
        self.rep_phase_epochs = int(total_epochs * rep_phase_ratio)
        self.joint_phase_epochs = int(total_epochs * joint_phase_ratio)
        self.cls_phase_epochs = total_epochs - self.rep_phase_epochs - self.joint_phase_epochs
        
        # Learning rates
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        
        # Loss parameters
        self.temperature = temperature
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        
        # Training parameters
        self.rebuild_interval = rebuild_interval
        
        # Initialize loss function and optimizer
        self.loss_fn = MixedLoss(temperature=temperature, lambda_cls=initial_lambda)
        self.optimizer = optim.AdamW(model.model.parameters(), lr=initial_lr)
        
        # Initialize retrainer
        self.retrainer = ModelRetrainer(model, self.optimizer, self.loss_fn, device)
        
        # Best model tracking
        self.best_model = None
        self.best_val_loss = float('inf')
    
    def _freeze_layers(self, num_layers: int = 0):
        """
        Freeze the first num_layers layers of the model.
        
        Args:
            num_layers: Number of layers to freeze
        """
        if num_layers <= 0:
            return
        
        # Get all parameters
        params = list(self.model.model.parameters())
        
        # Freeze the first num_layers layers
        for param in params[:num_layers]:
            param.requires_grad = False
        
        # Unfreeze the rest
        for param in params[num_layers:]:
            param.requires_grad = True
    
    def _unfreeze_all_layers(self):
        """Unfreeze all layers of the model."""
        for param in self.model.model.parameters():
            param.requires_grad = True
    
    def _update_learning_rate(self, epoch: int):
        """
        Update learning rate based on current epoch.
        
        Args:
            epoch: Current epoch
        """
        # Linear decay from initial_lr to final_lr
        progress = min(1.0, epoch / self.total_epochs)
        lr = self.initial_lr - progress * (self.initial_lr - self.final_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _update_lambda(self, epoch: int):
        """
        Update lambda parameter based on current epoch.
        
        Args:
            epoch: Current epoch
        """
        # Linear increase from initial_lambda to final_lambda
        progress = min(1.0, epoch / self.total_epochs)
        lambda_cls = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        
        self.loss_fn.update_lambda(lambda_cls)
    
    def train(self, train_samples: List[Tuple], val_samples: List[Tuple], 
              sorted_samples: List[Tuple] = None, sorted_scores: List[float] = None,
              batch_size: int = 16, verbose: bool = True):
        """
        Train the model using multi-stage training.
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            sorted_samples: List of sorted samples for contrastive learning
            sorted_scores: List of sorted scores for contrastive learning
            batch_size: Batch size
            verbose: Whether to print progress information
            
        Returns:
            Trained model
        """
        # Use sorted samples if provided, otherwise use train samples
        contrastive_samples = sorted_samples if sorted_samples else train_samples
        
        # Phase 1: Representation Learning
        if verbose:
            print("Phase 1: Representation Learning")
        
        # Freeze classification head
        self._freeze_layers(len(list(self.model.model.parameters())) - 2)
        
        for epoch in range(self.rep_phase_epochs):
            # Update parameters
            self._update_learning_rate(epoch)
            self._update_lambda(epoch)
            
            # Rebuild sample library if needed
            if epoch % self.rebuild_interval == 0 or epoch == 0:
                # Select samples for this epoch
                if sorted_samples and sorted_scores:
                    # Select top samples based on scores
                    top_k = min(len(sorted_samples), batch_size * 10)
                    epoch_samples = sorted_samples[:top_k]
                else:
                    # Randomly select samples
                    epoch_samples = random.sample(train_samples, min(len(train_samples), batch_size * 10))
                
                # Build contrastive pairs
                anchors, positives, negatives, weights, labels, logits = self.retrainer.build_contrastive_pairs(
                    epoch_samples, batch_size=batch_size
                )
            
            # Train one epoch
            if anchors is not None:
                self.model.train_mode()
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=torch.cat([x["input_ids"].unsqueeze(0) for x, _, _, _, _ in epoch_samples]),
                    attention_mask=torch.cat([x["attention_mask"].unsqueeze(0) for x, _, _, _, _ in epoch_samples])
                )
                
                # Calculate loss
                loss, con_loss, cls_loss = self.loss_fn(
                    anchors, positives, negatives, weights, outputs.logits, labels
                )
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                if verbose:
                    print(f"Epoch {epoch+1}/{self.total_epochs}, Loss: {loss.item():.4f}, Con Loss: {con_loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}")
            
            # Validate
            val_loss = self._validate(val_samples, batch_size)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.model.state_dict())
        
        # Phase 2: Joint Optimization
        if verbose:
            print("Phase 2: Joint Optimization")
        
        # Gradually unfreeze layers
        self._unfreeze_all_layers()
        
        for epoch in range(self.rep_phase_epochs, self.rep_phase_epochs + self.joint_phase_epochs):
            # Update parameters
            self._update_learning_rate(epoch)
            self._update_lambda(epoch)
            
            # Rebuild sample library if needed
            if epoch % self.rebuild_interval == 0:
                # Select samples for this epoch
                if sorted_samples and sorted_scores:
                    # Select top samples based on scores
                    top_k = min(len(sorted_samples), batch_size * 10)
                    epoch_samples = sorted_samples[:top_k]
                else:
                    # Randomly select samples
                    epoch_samples = random.sample(train_samples, min(len(train_samples), batch_size * 10))
                
                # Build contrastive pairs
                anchors, positives, negatives, weights, labels, logits = self.retrainer.build_contrastive_pairs(
                    epoch_samples, batch_size=batch_size
                )
            
            # Train one epoch
            if anchors is not None:
                self.model.train_mode()
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=torch.cat([x["input_ids"].unsqueeze(0) for x, _, _, _, _ in epoch_samples]),
                    attention_mask=torch.cat([x["attention_mask"].unsqueeze(0) for x, _, _, _, _ in epoch_samples])
                )
                
                # Calculate loss
                loss, con_loss, cls_loss = self.loss_fn(
                    anchors, positives, negatives, weights, outputs.logits, labels
                )
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                if verbose:
                    print(f"Epoch {epoch+1}/{self.total_epochs}, Loss: {loss.item():.4f}, Con Loss: {con_loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}")
            
            # Validate
            val_loss = self._validate(val_samples, batch_size)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.model.state_dict())
        
        # Phase 3: Classification Optimization
        if verbose:
            print("Phase 3: Classification Optimization")
        
        for epoch in range(self.rep_phase_epochs + self.joint_phase_epochs, self.total_epochs):
            # Update parameters
            self._update_learning_rate(epoch)
            self._update_lambda(1.0)  # Focus on classification
            
            # Train one epoch with standard classification
            self.model.train_mode()
            
            # Process in batches
            for i in range(0, len(train_samples), batch_size):
                batch_samples = train_samples[i:i+batch_size]
                batch_x = []
                batch_labels = []
                
                for x, _, label, _, _ in batch_samples:
                    batch_x.append(x)
                    batch_labels.append(label if label is not None else 0)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=torch.cat([x["input_ids"].unsqueeze(0) for x in batch_x]),
                    attention_mask=torch.cat([x["attention_mask"].unsqueeze(0) for x in batch_x])
                )
                
                # Calculate loss
                labels = torch.tensor(batch_labels, device=self.device)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.total_epochs}, Loss: {loss.item():.4f}")
            
            # Validate
            val_loss = self._validate(val_samples, batch_size)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.model.state_dict())
        
        # Load best model
        if self.best_model is not None:
            self.model.model.load_state_dict(self.best_model)
        
        return self.model
    
    def _validate(self, val_samples: List[Tuple], batch_size: int) -> float:
        """
        Validate the model on validation samples.
        
        Args:
            val_samples: List of validation samples
            batch_size: Batch size
            
        Returns:
            Validation loss
        """
        self.model.eval_mode()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(val_samples), batch_size):
                batch_samples = val_samples[i:i+batch_size]
                batch_x = []
                batch_labels = []
                
                for x, _, label, _, _ in batch_samples:
                    batch_x.append(x)
                    batch_labels.append(label if label is not None else 0)
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=torch.cat([x["input_ids"].unsqueeze(0) for x in batch_x]),
                    attention_mask=torch.cat([x["attention_mask"].unsqueeze(0) for x in batch_x])
                )
                
                # Calculate loss
                labels = torch.tensor(batch_labels, device=self.device)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
