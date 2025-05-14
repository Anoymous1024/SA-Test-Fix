"""
Model retrainer module for SA-Test-Fix.

This module implements model retraining with contrastive learning for
repairing sentiment analysis models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    """Contrastive loss function for model retraining."""
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize the contrastive loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarity scores
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, anchors, positives, negatives, weights=None):
        """
        Calculate contrastive loss.
        
        Args:
            anchors: Anchor embeddings [batch_size, hidden_dim]
            positives: Positive embeddings [batch_size, hidden_dim]
            negatives: Negative embeddings [batch_size, num_negatives, hidden_dim]
            weights: Sample weights [batch_size]
            
        Returns:
            Contrastive loss
        """
        batch_size = anchors.shape[0]
        hidden_dim = anchors.shape[1]
        
        # Normalize embeddings
        anchors_norm = F.normalize(anchors, p=2, dim=1)
        positives_norm = F.normalize(positives, p=2, dim=1)
        
        # Calculate positive similarity
        pos_sim = torch.bmm(
            anchors_norm.view(batch_size, 1, hidden_dim),
            positives_norm.view(batch_size, hidden_dim, 1)
        ).squeeze(-1) / self.temperature
        
        # Calculate negative similarities
        neg_sims = []
        for i in range(batch_size):
            if i < len(negatives) and negatives[i] is not None:
                neg_emb = negatives[i]
                neg_emb_norm = F.normalize(neg_emb, p=2, dim=1)
                
                neg_sim = torch.mm(
                    anchors_norm[i:i+1],
                    neg_emb_norm.t()
                ) / self.temperature
                
                neg_sims.append(neg_sim.squeeze(0))
            else:
                # If no negatives, create a dummy negative with low similarity
                neg_sims.append(torch.tensor([-100.0], device=anchors.device))
        
        # Combine positive and negative similarities
        logits = []
        for i in range(batch_size):
            if i < len(neg_sims):
                logit = torch.cat([pos_sim[i:i+1], neg_sims[i]], dim=0)
                logits.append(logit)
        
        if not logits:
            return torch.tensor(0.0, device=anchors.device)
        
        logits = torch.stack(logits)
        
        # Apply weights if provided
        if weights is not None:
            logits = logits * weights.view(-1, 1)
        
        # Labels are always 0 (positive is the first element)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MixedLoss(nn.Module):
    """Mixed loss function combining contrastive and classification losses."""
    
    def __init__(self, temperature: float = 0.5, lambda_cls: float = 0.5):
        """
        Initialize the mixed loss function.
        
        Args:
            temperature: Temperature parameter for contrastive loss
            lambda_cls: Weight for classification loss
        """
        super(MixedLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.classification_loss = nn.CrossEntropyLoss()
        self.lambda_cls = lambda_cls
    
    def forward(self, anchors, positives, negatives, weights, logits, labels):
        """
        Calculate mixed loss.
        
        Args:
            anchors: Anchor embeddings
            positives: Positive embeddings
            negatives: Negative embeddings
            weights: Sample weights
            logits: Classification logits
            labels: Classification labels
            
        Returns:
            Mixed loss
        """
        con_loss = self.contrastive_loss(anchors, positives, negatives, weights)
        cls_loss = self.classification_loss(logits, labels)
        
        return (1 - self.lambda_cls) * con_loss + self.lambda_cls * cls_loss, con_loss, cls_loss
    
    def update_lambda(self, lambda_cls: float):
        """
        Update the classification loss weight.
        
        Args:
            lambda_cls: New weight for classification loss
        """
        self.lambda_cls = lambda_cls


class ModelRetrainer:
    """Model retrainer for sentiment analysis models."""
    
    def __init__(self, model, optimizer, loss_fn, device=None):
        """
        Initialize the model retrainer.
        
        Args:
            model: Model to retrain
            optimizer: Optimizer for training
            loss_fn: Loss function
            device: Device to use for training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.best_val_loss = float('inf')
    
    def build_contrastive_pairs(self, samples: List[Tuple], batch_size: int = 16):
        """
        Build contrastive pairs from samples.
        
        Args:
            samples: List of samples (x, x_prime, label, relation_type, test_name)
            batch_size: Batch size
            
        Returns:
            Tuple of (anchors, positives, negatives, weights, labels, logits)
        """
        self.model.eval_mode()
        device = self.device
        
        # Prepare data
        anchors = []
        positives = []
        negatives = []
        weights = []
        labels = []
        
        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_x = []
            batch_x_prime = []
            batch_labels = []
            batch_relation_types = []
            
            for x, x_prime, label, relation_type, _ in batch_samples:
                batch_x.append(x)
                batch_x_prime.append(x_prime)
                batch_labels.append(label if label is not None else 0)
                batch_relation_types.append(relation_type)
            
            # Get embeddings
            with torch.no_grad():
                batch_x_embeds = []
                batch_x_prime_embeds = []
                
                for j in range(len(batch_x)):
                    x_embed = self.model.get_embeddings(batch_x[j])
                    x_prime_embed = self.model.get_embeddings(batch_x_prime[j])
                    
                    batch_x_embeds.append(x_embed)
                    batch_x_prime_embeds.append(x_prime_embed)
                
                batch_x_embeds = torch.cat(batch_x_embeds, dim=0)
                batch_x_prime_embeds = torch.cat(batch_x_prime_embeds, dim=0)
            
            # Build contrastive pairs based on relation type
            batch_anchors = []
            batch_positives = []
            batch_negatives = []
            batch_weights = []
            
            for j, relation_type in enumerate(batch_relation_types):
                if relation_type == "identity":
                    # For identity relation, original and perturbed samples form a positive pair
                    anchor = batch_x_embeds[j].unsqueeze(0)
                    positive = batch_x_prime_embeds[j].unsqueeze(0)
                    
                    # Use other samples in the batch as negatives
                    neg_indices = [k for k in range(len(batch_samples)) if k != j]
                    if neg_indices:
                        neg_embeds = batch_x_embeds[neg_indices]
                    else:
                        # If no negatives, create a random one
                        neg_embeds = torch.randn_like(batch_x_embeds[j], device=device).unsqueeze(0)
                    
                    batch_anchors.append(anchor)
                    batch_positives.append(positive)
                    batch_negatives.append(neg_embeds)
                    batch_weights.append(1.0)  # Default weight
                else:
                    # For inequality relation, use SimCSE approach
                    anchor = batch_x_embeds[j].unsqueeze(0)
                    
                    # Generate positive using dropout
                    with torch.no_grad():
                        positive = self.model.get_embeddings(batch_x[j], apply_dropout=True)
                    
                    # Use perturbed sample as negative
                    neg_embed = batch_x_prime_embeds[j].unsqueeze(0)
                    
                    batch_anchors.append(anchor)
                    batch_positives.append(positive)
                    batch_negatives.append(neg_embed)
                    batch_weights.append(1.0)  # Default weight
            
            # Collect batch results
            anchors.extend(batch_anchors)
            positives.extend(batch_positives)
            negatives.extend(batch_negatives)
            weights.extend(batch_weights)
            labels.extend(batch_labels)
        
        # Convert to tensors
        if anchors and positives and negatives and weights and labels:
            anchors = torch.cat(anchors, dim=0)
            positives = torch.cat(positives, dim=0)
            weights = torch.tensor(weights, device=device)
            labels = torch.tensor(labels, device=device)
            
            # Get logits for classification loss
            logits = []
            for x, _, _, _, _ in samples:
                logit = self.model.predict(x)
                logits.append(logit)
            
            logits = torch.cat(logits, dim=0)
            
            return anchors, positives, negatives, weights, labels, logits
        else:
            return None, None, None, None, None, None
