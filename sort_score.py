"""
Sample sorting module for SA-Test-Fix.

This module implements sample sorting methods for selecting the most valuable
test cases for model repair.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm

def calculate_deepgini(outputs) -> float:
    """
    Calculate DeepGini score for a model output.
    
    Args:
        outputs: Model output logits
        
    Returns:
        DeepGini score
    """
    probs = F.softmax(outputs, dim=1)
    return 1 - torch.sum(probs ** 2, dim=1)

def calculate_uncertainty(x_outputs, x_prime_outputs) -> float:
    """
    Calculate uncertainty score for a pair of model outputs.
    
    Args:
        x_outputs: Original sample model output
        x_prime_outputs: Perturbed sample model output
        
    Returns:
        Uncertainty score
    """
    x_uncertainty = calculate_deepgini(x_outputs)
    x_prime_uncertainty = calculate_deepgini(x_prime_outputs)
    
    # Average uncertainty
    return (x_uncertainty + x_prime_uncertainty) / 2

def calculate_violation_identity(x_outputs, x_prime_outputs) -> float:
    """
    Calculate violation score for identity relation.
    
    Args:
        x_outputs: Original sample model output
        x_prime_outputs: Perturbed sample model output
        
    Returns:
        Violation score
    """
    # For identity relation, predictions should be the same
    # Calculate cosine distance between probability distributions
    x_probs = F.softmax(x_outputs, dim=1)
    x_prime_probs = F.softmax(x_prime_outputs, dim=1)
    
    # 1 - cosine similarity = cosine distance
    return 1 - F.cosine_similarity(x_probs, x_prime_probs, dim=1)

def calculate_violation_inequality(x_outputs, x_prime_outputs, direction: str = "increasing") -> float:
    """
    Calculate violation score for inequality relation.
    
    Args:
        x_outputs: Original sample model output
        x_prime_outputs: Perturbed sample model output
        direction: Direction of expected change ("increasing" or "decreasing")
        
    Returns:
        Violation score
    """
    # For inequality relation, predictions should change in the expected direction
    x_probs = F.softmax(x_outputs, dim=1)
    x_prime_probs = F.softmax(x_prime_outputs, dim=1)
    
    # Get predicted class probabilities
    x_pred_class = torch.argmax(x_probs, dim=1)
    x_prime_pred_class = torch.argmax(x_prime_probs, dim=1)
    
    # Calculate violation based on direction
    if direction == "increasing":
        # For increasing direction, x_prime_pred_class should be >= x_pred_class
        violation = torch.where(
            x_prime_pred_class < x_pred_class,
            torch.ones_like(x_pred_class, dtype=torch.float32),
            torch.zeros_like(x_pred_class, dtype=torch.float32)
        )
    else:  # decreasing
        # For decreasing direction, x_prime_pred_class should be <= x_pred_class
        violation = torch.where(
            x_prime_pred_class > x_pred_class,
            torch.ones_like(x_pred_class, dtype=torch.float32),
            torch.zeros_like(x_pred_class, dtype=torch.float32)
        )
    
    return violation

def sort_samples(model, test_cases: List[Tuple], default_relation_type: str = "identity") -> Tuple[List[Tuple], List[float]]:
    """
    Sort test cases by their importance for model repair.
    
    Args:
        model: Model to repair
        test_cases: List of test cases (x, x_prime, label, relation_type, test_name)
        default_relation_type: Default relation type to use if not specified
        
    Returns:
        Tuple of (sorted_samples, scores)
    """
    model.eval_mode()
    device = next(model.model.parameters()).device
    
    scores = []
    identity_count = 0
    inequality_count = 0
    test_type_counts = {}
    
    for x, x_prime, label, relation_type, test_name in tqdm(test_cases, desc="Sorting samples"):
        # Count by relation type and test type
        if relation_type == "identity":
            identity_count += 1
        else:
            inequality_count += 1
        
        if test_name not in test_type_counts:
            test_type_counts[test_name] = 0
        test_type_counts[test_name] += 1
        
        # Get model outputs
        if isinstance(x, dict):
            # Handle tokenized inputs
            x_input_ids = x["input_ids"].to(device)
            x_attention_mask = x["attention_mask"].to(device)
            x_prime_input_ids = x_prime["input_ids"].to(device)
            x_prime_attention_mask = x_prime["attention_mask"].to(device)
            
            with torch.no_grad():
                x_outputs = model.model(
                    input_ids=x_input_ids.unsqueeze(0) if x_input_ids.dim() == 1 else x_input_ids,
                    attention_mask=x_attention_mask.unsqueeze(0) if x_attention_mask.dim() == 1 else x_attention_mask
                ).logits
                
                x_prime_outputs = model.model(
                    input_ids=x_prime_input_ids.unsqueeze(0) if x_prime_input_ids.dim() == 1 else x_prime_input_ids,
                    attention_mask=x_prime_attention_mask.unsqueeze(0) if x_prime_attention_mask.dim() == 1 else x_prime_attention_mask
                ).logits
        else:
            # Handle raw text inputs (should be tokenized first)
            raise ValueError("Test cases should be tokenized before sorting")
        
        # Calculate uncertainty score
        uncertainty = calculate_uncertainty(x_outputs, x_prime_outputs)
        
        # Calculate violation score based on relation type
        if relation_type == "identity":
            violation = calculate_violation_identity(x_outputs, x_prime_outputs)
        else:  # inequality relation
            # Determine direction based on test name
            if "Positive" in test_name or "Intensifier" in test_name:
                direction = "increasing"
            elif "Negative" in test_name or "Reducer" in test_name:
                direction = "decreasing"
            else:
                direction = "increasing"  # Default
            
            violation = calculate_violation_inequality(x_outputs, x_prime_outputs, direction)
        
        # Calculate final score
        score = violation * uncertainty
        
        # Convert to scalar if tensor
        if isinstance(score, torch.Tensor):
            score = score.item()
        
        scores.append(score)
    
    # Print statistics
    print(f"Identity relation samples: {identity_count}")
    print(f"Inequality relation samples: {inequality_count}")
    print("Test type distribution:")
    for test_type, count in test_type_counts.items():
        print(f"  {test_type}: {count}")
    
    # Sort samples by score
    sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
    sorted_samples = [test_cases[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    return sorted_samples, sorted_scores

def select_top_k_samples(sorted_samples: List[Tuple], sorted_scores: List[float], k: int) -> List[Tuple]:
    """
    Select the top k samples by score.
    
    Args:
        sorted_samples: List of sorted samples
        sorted_scores: List of sorted scores
        k: Number of samples to select
        
    Returns:
        List of selected samples
    """
    return sorted_samples[:k]
