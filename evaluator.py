"""
Evaluator module for SA-Test-Fix.

This module provides evaluation metrics for sentiment analysis models,
including accuracy, error rate, negative flip rate, and relative negative flip rate.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class Evaluator:
    """Evaluator class for sentiment analysis models."""
    
    def __init__(self, model, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def calculate_accuracy(self, data_loader) -> float:
        """
        Calculate accuracy on a dataset.
        
        Args:
            data_loader: DataLoader containing the dataset
            
        Returns:
            Accuracy score
        """
        self.model.eval_mode()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calculating accuracy"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                
                logits = self.model.predict(inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        return accuracy_score(all_labels, all_preds)
    
    def calculate_error_rate(self, test_cases) -> float:
        """
        Calculate error rate on test cases.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Error rate
        """
        self.model.eval_mode()
        total_cases = len(test_cases)
        error_count = 0
        
        for x, x_prime, label, relation_type, _ in tqdm(test_cases, desc="Calculating error rate"):
            # Get predictions
            x_logits = self.model.predict(x)
            x_prime_logits = self.model.predict(x_prime)
            
            x_pred = torch.argmax(x_logits, dim=1).item()
            x_prime_pred = torch.argmax(x_prime_logits, dim=1).item()
            
            # Check if the prediction violates the metamorphic relation
            if relation_type == "identity":
                # For identity relation, predictions should be the same
                if x_pred != x_prime_pred:
                    error_count += 1
            else:  # inequality relation
                # For inequality relation, check the direction of change
                if relation_type == "DIR_increasing" and x_prime_pred < x_pred:
                    error_count += 1
                elif relation_type == "DIR_decreasing" and x_prime_pred > x_pred:
                    error_count += 1
        
        return error_count / total_cases if total_cases > 0 else 0.0
    
    def calculate_negative_flip_rate(self, test_cases, original_model) -> float:
        """
        Calculate negative flip rate between the current model and the original model.
        
        Args:
            test_cases: List of test cases
            original_model: Original model for comparison
            
        Returns:
            Negative flip rate
        """
        self.model.eval_mode()
        original_model.eval_mode()
        
        total_cases = len(test_cases)
        flip_count = 0
        
        for x, _, label, _, _ in tqdm(test_cases, desc="Calculating negative flip rate"):
            # Get predictions from both models
            current_logits = self.model.predict(x)
            original_logits = original_model.predict(x)
            
            current_pred = torch.argmax(current_logits, dim=1).item()
            original_pred = torch.argmax(original_logits, dim=1).item()
            
            # Check if the prediction flipped from correct to incorrect
            if original_pred == label and current_pred != label:
                flip_count += 1
        
        return flip_count / total_cases if total_cases > 0 else 0.0
    
    def calculate_relative_negative_flip_rate(self, test_cases, original_model) -> float:
        """
        Calculate relative negative flip rate between the current model and the original model.
        
        Args:
            test_cases: List of test cases
            original_model: Original model for comparison
            
        Returns:
            Relative negative flip rate
        """
        self.model.eval_mode()
        original_model.eval_mode()
        
        correct_cases = 0
        flip_count = 0
        
        for x, _, label, _, _ in tqdm(test_cases, desc="Calculating relative negative flip rate"):
            # Get predictions from both models
            current_logits = self.model.predict(x)
            original_logits = original_model.predict(x)
            
            current_pred = torch.argmax(current_logits, dim=1).item()
            original_pred = torch.argmax(original_logits, dim=1).item()
            
            # Count cases where original model was correct
            if original_pred == label:
                correct_cases += 1
                
                # Check if the prediction flipped from correct to incorrect
                if current_pred != label:
                    flip_count += 1
        
        return flip_count / correct_cases if correct_cases > 0 else 0.0
    
    def evaluate_model(self, test_cases, original_model=None, data_loader=None) -> Dict:
        """
        Evaluate a model on multiple metrics.
        
        Args:
            test_cases: List of test cases
            original_model: Original model for comparison
            data_loader: DataLoader for accuracy calculation
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Calculate error rate
        results["error_rate"] = self.calculate_error_rate(test_cases)
        
        # Calculate accuracy if data_loader is provided
        if data_loader:
            results["accuracy"] = self.calculate_accuracy(data_loader)
        
        # Calculate negative flip rate and relative negative flip rate if original_model is provided
        if original_model:
            results["negative_flip_rate"] = self.calculate_negative_flip_rate(test_cases, original_model)
            results["relative_negative_flip_rate"] = self.calculate_relative_negative_flip_rate(test_cases, original_model)
        
        return results
